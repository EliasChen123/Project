# 训练 HccePose (BF)。  
# 训练完成后，会在数据集文件夹下生成一个 `HccePose` 文件夹，  
# 用于保存每个物体的权重文件。  

# 示例：
# ```
# demo-tex-objs
# |--- HccePose
#     |--- obj_01
#     ...
#     |--- obj_10
# |--- models
# |--- train_pbr
# |--- train_pbr_xyz_GT_back
# |--- train_pbr_xyz_GT_front
# ```
# '''

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paths_to_add = [
    ROOT_DIR,                                  # 为了能找到 HccePose
    os.path.join(ROOT_DIR, "bop_toolkit"),     # 为了能让 bop_toolkit 内部逻辑跑通
]
for p in paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)
import torch, argparse
import itertools
import numpy as np
from tqdm import tqdm
from HccePose.bop_loader import bop_dataset, train_bop_dataset_back_front, test_bop_dataset_back_front
from HccePose.network_model import HccePose_BF_Net, HccePose_Loss, load_checkpoint, save_checkpoint, save_best_checkpoint
# 兼容新旧版本的写法
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from torch import optim
import torch.distributed as dist
from HccePose.visualization import vis_rgb_mask_Coord
# from HccePose.PnP_solver import solve_PnP, solve_PnP_comb
from HccePose.metric import add_s
from kasal.bop_toolkit_lib.inout import load_ply
from HccePose.EProPnP_solver import solve_EPro_PnP
from EPro_PnP.epropnp import EProPnP6DoF
from EPro_PnP.camera import PerspectiveCamera
from EPro_PnP.cost_fun import AdaptiveHuberPnPCost
from EPro_PnP.levenberg_marquardt import LMSolver,RSLMSolver
from EPro_PnP.rotation_conversions import matrix_to_quaternion
from EPro_PnP.monte_carlo_pose_loss import MonteCarloPoseLoss

def test(obj_ply, obj_info, net: HccePose_BF_Net, test_loader: torch.utils.data.DataLoader):
    net.eval()
    add_list_l = []
    # 添加 with torch.no_grad():
    with torch.no_grad():
        for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_R_m2c, cam_t_m2c) in tqdm(enumerate(test_loader)):
            if torch.cuda.is_available():
                rgb_c=rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                mask_vis_c=mask_vis_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                GT_Front_hcce = GT_Front_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                GT_Back_hcce = GT_Back_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                Bbox = Bbox.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                cam_K = cam_K.cpu().numpy()
            # 必须指定 device_type，否则 torch.amp.autocast 会报错
            with autocast(device_type='cuda', dtype=torch.float16):
                pred_results = net.inference_batch(rgb_c, Bbox)
                pred_mask = pred_results['pred_mask']
                coord_image = pred_results['coord_2d_image']
                pred_front_code_0 = pred_results['pred_front_code_obj']
                pred_back_code_0 = pred_results['pred_back_code_obj']
                # 获取前、后表面的预测权重
                pred_w2d = pred_results['pred_w2d']

                pred_mask_np = pred_mask.detach().cpu().numpy()
                pred_front_code_0_np = pred_front_code_0.detach().cpu().numpy()
                pred_back_code_0_np = pred_back_code_0.detach().cpu().numpy()
                coord_image_np = coord_image.detach().cpu().numpy()                
                pred_w2d_np = pred_w2d.detach().cpu().numpy()
                # 遍历 batch 中的每一张图片
                for i in range(pred_mask_np.shape[0]):
                    cam_R_m2c_i = cam_R_m2c[i].detach().cpu().numpy()
                    cam_t_m2c_i = cam_t_m2c[i].detach().cpu().numpy()
                    
                    # 调用修改好的 EPro-PnP
                    info = solve_EPro_PnP(
                        pred_front_code_0_np[i],
                        pred_w2d_np[i], 
                        pred_mask_np[i], 
                        coord_image_np[i], 
                        cam_K[i]
                    )
                    
                    # EPro-PnP 直接返回最优解，不需要像原版那样做特征点组合优化
                    if info['success']:
                        add_err = add_s(obj_ply, obj_info, [[cam_R_m2c_i, cam_t_m2c_i]], [[info['rot'], info['tvecs']]])[0]
                        add_list_l.append(add_err)
                    else:
                        add_list_l.append(0.0) # 求解失败则 ADD 误差记为 0
            torch.cuda.empty_cache()
    add_list_l = np.array(add_list_l)
    # add_list_l = np.mean(add_list_l, axis=0)
    print(add_list_l)
    max_acc_id = np.argmax(add_list_l)
    max_acc = np.max(add_list_l)
    print('max acc id: ', max_acc_id)
    print('max acc: ', max_acc)
    net.train()
    return max_acc_id, max_acc, add_list_l

# 在 s4_p2_train_bf_pbr.py 中添加函数

def hcce_encode_torch(code_tensor, iteration=8):
    """
    GPU 版本的 HCCE 编码
    Args:
        code_tensor: (B, 3, H, W) 范围在 [0, 255] 的 Tensor
    """
    # 确保输入是 int 类型进行位运算或者模拟位运算
    # 这里根据原逻辑复现，原逻辑使用了取模运算
    
    B, C, H, W = code_tensor.shape
    device = code_tensor.device
    hcce_images = torch.zeros((B, iteration * 3, H, W), device=device, dtype=torch.float32)
    
    # 拆分通道
    c0 = code_tensor[:, 0, :, :]
    c1 = code_tensor[:, 1, :, :]
    c2 = code_tensor[:, 2, :, :]
    channels = [c0, c1, c2]
    
    for i in range(iteration):
        divisor = 2**(iteration - i)
        norm_factor = 2**(iteration - i) - 1
        
        for k in range(3): # 对 x, y, z 三个通道
            # 模拟 numpy 的 % 运算
            temp = torch.fmod(channels[k], divisor)
            temp = temp.long().float() / norm_factor # 转回 float
            hcce_images[:, i + k*iteration, :, :] = temp

    check_hcce_images = hcce_images.clone()
    k_ = iteration
    
    # 应用后续的翻转逻辑 (原代码中的 check_hcce_images 处理)
    # 使用切片操作避免循环
    
    # 通道 x
    for i in range(k_ - 1):
        prev_layer = hcce_images[:, i, :, :]
        curr_layer = hcce_images[:, i + 1, :, :]
        mask = prev_layer >= 0.5
        check_hcce_images[:, i + 1, :, :][mask] = -curr_layer[mask] + 1
        
    # 通道 y
    for i in range(k_ - 1):
        prev_layer = hcce_images[:, i + k_, :, :]
        curr_layer = hcce_images[:, i + 1 + k_, :, :]
        mask = prev_layer >= 0.5
        check_hcce_images[:, i + 1 + k_, :, :][mask] = -curr_layer[mask] + 1

    # 通道 z
    for i in range(k_ - 1):
        prev_layer = hcce_images[:, i + k_*2, :, :]
        curr_layer = hcce_images[:, i + 1 + k_*2, :, :]
        mask = prev_layer >= 0.5
        check_hcce_images[:, i + 1 + k_*2, :, :][mask] = -curr_layer[mask] + 1
        
    return check_hcce_images

if __name__ == '__main__':
    # '''
    # When `ide_debug` is set to True, single-GPU mode is used, allowing IDE debugging.  
    # When `ide_debug` is set to False, DDP (Distributed Data Parallel) training is enabled.  

    # DDP Training:  
    # screen -S train_ddp
    # nohup python -u -m torch.distributed.launch --nproc_per_node 6 /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    
    # Single-GPU Training:  
    # nohup python -u /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    
    # ------------------------------------------------------    
    
    # 当 `ide_debug` 为 True 时，仅使用单卡，可在 IDE 中进行调试。  
    # 当 `ide_debug` 为 False 时，启用 DDP（分布式数据并行）训练。  

    # DDP 训练：  
    # screen -S train_ddp
    # nohup python -u -m torch.distributed.launch --nproc_per_node 6 /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    # torchrun --nproc_per_node=6 s4_p2_train_bf_pbr.py --local-rank 0


    # 单卡训练：
    # nohup python -u s4_p2_train_bf_pbr.py > s4_p2_train_bf_pbr.log 2>&1 &
    # nohup python -u s4_p2_train_bf_pbr.py >> s4_p2_train_bf_pbr.log 2>&1 &
    # '''
    
    ide_debug = True
    
    # Specify the path to the dataset folder.
    # 指定数据集文件夹的路径。
    TARGET_DIR = os.path.join(ROOT_DIR, "output")
    DATASET_PATH = os.path.join(TARGET_DIR, "trocar")
    
    # Specify the name of the subfolder in the dataset used for loading training data.
    # 指定数据集中用于加载训练数据的子文件夹名称。
    train_folder_name = 'train_pbr'
    
    # The range of object IDs for training.  
    # `start_obj_id` is the starting object ID, and `end_obj_id` is the ending object ID.
    # 训练的物体 ID 范围。  
    # `start_obj_id` 为起始物体 ID，`end_obj_id` 为终止物体 ID。
    start_obj_id = 1
    end_obj_id =1
    
    # Total number of training epochs.
    # 总训练轮数。
    total_iteration = 50001
    warm_up_step = 1000
    # Learning rate.
    # 学习率。
    lr = 0.0002
    
    # Number of samples per training epoch.
    # 每轮训练的样本数量。
    batch_size = 32

    accumulation_steps = 2
    # Number of worker processes used by the DataLoader.
    # DataLoader 的进程数量。
    num_workers = 8
    
    # The number of epochs between saving checkpoints.
    # 保存检查点的间隔轮数。
    log_freq = 500
    
    # Scaling ratio for 2D bounding boxes.
    # 2D 包围盒的缩放比例。
    padding_ratio = 1.5

    # 默认为 2，增加到 4 或 8 可以缓解 GPU 等待
    prefetch_factor = 2

    # Whether to enable EfficientNet.
    # 是否启用 EfficientNet。
    efficientnet_key = 'enable_b4'
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    parser = argparse.ArgumentParser()
    if ide_debug:
        parser.add_argument("--local-rank", default=0, type=int)
    else:
        parser.add_argument("--local-rank", default=-1, type=int)
    args = parser.parse_args()
    if not ide_debug:
        # 从环境变量获取 local_rank，这是 torchrun 的标准做法
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.barrier()
        world_size = torch.distributed.get_world_size()
    
    local_rank = args.local_rank
    # 确保设备设置正确
    torch.cuda.set_device(local_rank)
    if local_rank != 0:
        if ide_debug is True:
            pass
    CUDA_DEVICE = str(local_rank)
    np.random.seed(local_rank)
    bop_dataset_item = bop_dataset(DATASET_PATH, local_rank=local_rank)
    train_bop_dataset_back_front_item = train_bop_dataset_back_front(bop_dataset_item, train_folder_name, padding_ratio=padding_ratio, )
    
    # ratio = 0.01 means selecting 1% of samples from the dataset for testing.
    # ratio = 0.01 表示从数据集中选择 1% 的样本作为测试数据。
    test_bop_dataset_back_front_item = test_bop_dataset_back_front(bop_dataset_item, train_folder_name, padding_ratio=padding_ratio, ratio=0.01)
    pose_loss_fn = MonteCarloPoseLoss(momentum=0.01).to('cuda:'+CUDA_DEVICE)

    for obj_id in range(start_obj_id, end_obj_id + 1):
        
        
        obj_path = bop_dataset_item.obj_model_list[bop_dataset_item.obj_id_list.index(obj_id)]
        print(obj_path)
        obj_ply = load_ply(obj_path)
        obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]
        
        # Create the save path.
        # 创建保存路径。
        save_path = os.path.join(TARGET_DIR, 'HccePose', 'obj_%s'%str(obj_id).rjust(2, '0'))
        best_save_path = os.path.join(save_path, 'best_score')
        try: os.mkdir(os.path.join(TARGET_DIR, 'HccePose')) 
        except: 1
        try: os.mkdir(save_path) 
        except: 1
        try: os.mkdir(best_save_path) 
        except: 1

        # Get the 3D dimensions of the object.
        # 获取物体的 3D 尺寸。
        min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        
        # Define the loss function and neural network.
        # 定义损失函数和神经网络。
        loss_net = HccePose_Loss()
        scaler = GradScaler()
        net = HccePose_BF_Net(
                efficientnet_key = efficientnet_key,
                input_channels = 3, 
                min_xyz = min_xyz,
                size_xyz = size_xyz,
            )
        net_test = HccePose_BF_Net(
                efficientnet_key = efficientnet_key,
                input_channels = 3, 
                min_xyz = min_xyz,
                size_xyz = size_xyz,
            )
        if torch.cuda.is_available():
            net=net.to('cuda:'+CUDA_DEVICE)
            net_test=net_test.to('cuda:'+CUDA_DEVICE)
        optimizer=optim.Adam(net.parameters(), lr=lr)

        # Attempt to load weights from an interrupted training session.
        # 尝试加载中断训练时保存的权重。
        best_score = 0
        iteration_step = 0
        try:
            checkpoint_info = load_checkpoint(save_path, net, optimizer, local_rank=local_rank, CUDA_DEVICE=CUDA_DEVICE)
            best_score = checkpoint_info['best_score']
            iteration_step = checkpoint_info['iteration_step']
        except:
            print('no checkpoint')
        
        if not ide_debug:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], )
        
        # Update the training and testing data loaders respectively.
        # 分别更新训练和测试数据加载器。
        train_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        # train_loader = torch.utils.data.DataLoader(train_bop_dataset_back_front_item, batch_size=batch_size, 
        #                                         shuffle=True, num_workers=num_workers, drop_last=True) 
        train_loader = torch.utils.data.DataLoader(train_bop_dataset_back_front_item, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=num_workers, # 建议设置为 CPU 物理核心数
                                                pin_memory=True,    # 开启锁页内存，加速 CPU 到 GPU 传输
                                                persistent_workers=True, # 关键：避免每个 Epoch 结束后销毁进程重建
                                                prefetch_factor=prefetch_factor, # 新增：默认为 2，增加到 4 或 8 可以缓解 GPU 等待
                                                drop_last=True) 
        test_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        test_loader = torch.utils.data.DataLoader(test_bop_dataset_back_front_item,
                                                batch_size=batch_size, 
                                                shuffle=False, 
                                                num_workers=4, 
                                                drop_last=False) 
        
        # 实例化 EPro-PnP 训练组件
        pnp_solver = EProPnP6DoF(
            mc_samples=512, num_iter=4, normalize=True,
            solver=LMSolver(
                dof=6, num_iter=5, normalize=True,
                init_solver=RSLMSolver(dof=6, num_points=16, num_proposals=64, num_iter=3)
            )
        ).to('cuda:'+CUDA_DEVICE)
        camera = PerspectiveCamera()
        pnp_cost_fun = AdaptiveHuberPnPCost(relative_delta=0.5)
        
        # Train
        # 训练
        while True:
            end_training = False
            for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_R_m2c, cam_t_m2c) in enumerate(train_loader):
                # Test and save checkpoints only in the process where `local_rank = 0`.
                # 仅在 `local_rank = 0` 的进程中执行测试并保存检查点。
                if args.local_rank == 0:
                    if (iteration_step)%log_freq == 0 and iteration_step > 0:
                        if isinstance(net, torch.nn.parallel.DataParallel):
                            state_dict = net.module.state_dict()
                        elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
                            state_dict = net.module.state_dict()
                        else:
                            state_dict = net.state_dict()
                        # net_test.load_state_dict(state_dict)
                        # max_acc_id, max_acc, add_list_l = test(obj_ply, obj_info, net_test, test_loader, )
                        max_acc_id, max_acc, add_list_l = test(obj_ply, obj_info, net, test_loader)
                        if max_acc >= best_score:
                            best_score = max_acc
                            save_best_checkpoint(best_save_path, net, optimizer, best_score, iteration_step, keypoints_ = add_list_l)
                        loss_net.print_error_ratio()
                        save_checkpoint(save_path, net, iteration_step, best_score, optimizer, 3, keypoints_ = add_list_l)

                
                if torch.cuda.is_available():
                    rgb_c=rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    mask_vis_c=mask_vis_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    # GT_Front_hcce = GT_Front_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    # GT_Back_hcce = GT_Back_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    # 此时 GT_Front_hcce 实际上是原始坐标图 (B, 3, H, W)
                    GT_Front_raw = GT_Front_hcce.to('cuda:'+CUDA_DEVICE, non_blocking=True)
                    GT_Back_raw = GT_Back_hcce.to('cuda:'+CUDA_DEVICE, non_blocking=True)
                    # ================== 修改这几行：强制转换为 float32 ==================
                    Bbox = Bbox.to('cuda:'+CUDA_DEVICE, non_blocking=True).float()
                    cam_K = cam_K.to('cuda:'+CUDA_DEVICE, non_blocking=True).float()
                    cam_R_m2c = cam_R_m2c.to('cuda:'+CUDA_DEVICE, non_blocking=True).float() # (B, 3, 3)
                    cam_t_m2c = cam_t_m2c.to('cuda:'+CUDA_DEVICE, non_blocking=True).float() # (B, 3, 1)
                    # ====================================================================

                    # 将 GT 转换为 EPro-PnP 需要的 (B, 7) 格式
                    gt_q = matrix_to_quaternion(cam_R_m2c) # (B, 4)
                    pose_gt = torch.cat([cam_t_m2c.squeeze(-1), gt_q], dim=-1) # (B, 7)
                # 在 GPU 上进行编码
                GT_Front_hcce = hcce_encode_torch(GT_Front_raw)
                GT_Back_hcce = hcce_encode_torch(GT_Back_raw)
                # 必须指定 device_type，否则 torch.amp.autocast 会报错
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # 1. 接收网络输出 (pred_w2d 包含 4 个通道)
                    pred_mask, pred_front_back_code, pred_w2d = net(rgb_c)
                    pred_front_code = pred_front_back_code[:, :24, ...]
                    pred_back_code = pred_front_back_code[:, 24:, ...]
                    current_loss = loss_net(pred_front_code, pred_back_code, pred_mask, GT_Front_hcce, GT_Back_hcce, mask_vis_c)
                    
                    # 2. 可微地解码前、后表面的 3D 坐标
                    # 适配 DDP 结构下的函数调用
                    pred_front_code_prob = torch.sigmoid(pred_front_code)
                    pred_back_code_prob = torch.sigmoid(pred_back_code)

                    decode_func = net.module.hcce_decode if hasattr(net, 'module') else net.hcce_decode
                    
                    pred_front_xyz = decode_func(pred_front_code_prob.permute(0, 2, 3, 1)) / 255.0
                    pred_back_xyz = decode_func(pred_back_code_prob.permute(0, 2, 3, 1)) / 255.0
                    
                    pred_front_xyz = pred_front_xyz * size_xyz[None, None, None, :] + min_xyz[None, None, None, :]
                    pred_back_xyz = pred_back_xyz * size_xyz[None, None, None, :] + min_xyz[None, None, None, :]
                    
                    # 3. 拆分前向、后向 2D 权重
                    pred_w2d_permuted = pred_w2d.permute(0, 2, 3, 1) # [B, H, W, 4]
                    w2d = pred_w2d_permuted[..., :2]

                    # 4. 生成原图尺度下的 2D 绝对坐标网格
                    B, H_out, W_out, _ = pred_front_xyz.shape
                    y_out, x_out = torch.meshgrid(torch.arange(H_out, device=rgb_c.device), 
                                                  torch.arange(W_out, device=rgb_c.device), indexing='ij')
                    
                    # #  可微地解码 3D 坐标
                    # pred_front_code_permuted = pred_front_code.permute(0, 2, 3, 1)
                    # pred_front_xyz = net.hcce_decode(pred_front_code_permuted) / 255.0
                    # pred_front_xyz = pred_front_xyz * size_xyz[None, None, None, :] + min_xyz[None, None, None, :]
                    
                    # 归一化到 [0, 1] 相对位置 (加 0.5 是为了取像素中心)
                    x_norm = (x_out.float() + 0.5) / W_out 
                    y_norm = (y_out.float() + 0.5) / H_out
                    
                    # 取出 Bbox 的 x, y, w, h，并转移到 GPU 上
                    bx = Bbox[:, 0].view(B, 1, 1).to(rgb_c.device)
                    by = Bbox[:, 1].view(B, 1, 1).to(rgb_c.device)
                    bw = Bbox[:, 2].view(B, 1, 1).to(rgb_c.device)
                    bh = Bbox[:, 3].view(B, 1, 1).to(rgb_c.device)
                    
                    # 映射回原图绝对坐标
                    x2d_orig = bx + x_norm.unsqueeze(0) * bw
                    y2d_orig = by + y_norm.unsqueeze(0) * bh
                    coord_2d = torch.stack([x2d_orig, y2d_orig], dim=-1) # [B, 128, 128, 2]

                    # 前面的数据转 GPU 逻辑需要除以 1000 转为米
                    cam_t_m2c = cam_t_m2c.to('cuda:'+CUDA_DEVICE, non_blocking=True).float() / 1000.0 # [修改] 毫米转米
                    
                    # 5. 高效的 Batched EPro-PnP 端到端计算
                    loss_pose = 0.0
                    with autocast(device_type='cuda', enabled=False):
                        # --- Warm-up 策略：前 2000 步直接跳过，不仅防止 Loss 爆炸，还能极速提升初期训练速度 ---
                        if iteration_step < warm_up_step:
                            pose_loss_weight = 0.0
                            loss_pose = 0.0 * pred_w2d.sum()
                        else:
                            # 预热结束后，引入 PnP Loss，初始权重给小一点（根据前向传播的量级，1e-4 比较合适）
                            pose_loss_weight = 0.1
                            
                            valid_x3d = []
                            valid_x2d = []
                            valid_w2d = []
                            valid_pose_init = []
                            valid_cam_K = []
                            
                            max_pts = 512 # 单个表面最大采样点数
                            
                            # 这个 Python 循环只做数据的提取和 Padding，不涉及复杂运算，极快
                            for i in range(B):
                                mask_i = mask_vis_c[i]
                                if mask_i.dim() == 3:  
                                    mask_i = mask_i[0]

                                mask_idx = (mask_i > 0.5).nonzero(as_tuple=False)
                                if len(mask_idx) < 16:
                                    continue
                                
                                if len(mask_idx) > max_pts:
                                    sample_inds = torch.randperm(len(mask_idx), device=rgb_c.device)[:max_pts]
                                    mask_idx = mask_idx[sample_inds]
                                    
                                y_idx, x_idx = mask_idx[:, 0], mask_idx[:, 1]
                                # [修改] 1. 只取前表面 2. 除以 1000 转为米 3. 保留梯度(移除 detach)
                                # 解释：网络预测的 3D 点必须接收 EPro-PnP 传回的梯度，不能 detach！
                                x3d_i = pred_front_xyz[i, y_idx, x_idx, :].float() / 1000.0 
                                x2d_i = coord_2d[i, y_idx, x_idx, :].float()
                                w2d_i = w2d[i, y_idx, x_idx, :].float() # 只取前表面权重

                                # ★ 核心加速操作：Padding 对齐所有图片的大小 ★
                                # 当 w2d 填充为 0 时，PnP 算法会自动忽略这些背景填充点，完全等效于变长输入
                                N_cur = x3d_i.shape[0]
                                pad_len = max_pts - N_cur # 只用填补到 max_pts

                                if pad_len > 0:
                                    pad_x3d = torch.zeros((pad_len, 3), device=rgb_c.device)
                                    pad_x3d[:, 2] = 1.0  # 🌟 将深度设为1，避免透视投影除以0
                                    x3d_i = torch.cat([x3d_i, pad_x3d], dim=0)
                                    x2d_i = torch.cat([x2d_i, torch.zeros((pad_len, 2), device=rgb_c.device)], dim=0)
                                    w2d_i = torch.cat([w2d_i, torch.zeros((pad_len, 2), device=rgb_c.device)], dim=0)
                                    
                                valid_x3d.append(x3d_i)
                                valid_x2d.append(x2d_i)
                                valid_w2d.append(w2d_i)
                                valid_pose_init.append(pose_gt[i].float())
                                valid_cam_K.append(cam_K[i].float())
                                
                                # 控制最大参与 PnP 优化的图片数（如 8 张），释放多余显存
                                if len(valid_x3d) >= 8: 
                                    break
                            
                            # === 将所有合法图片拼接为一个真正的 Batch，一次性送入 PnP 计算 ===
                            if len(valid_x3d) > 0:
                                batched_x3d = torch.stack(valid_x3d)        # (B_v, 1024, 3)
                                batched_x2d = torch.stack(valid_x2d)        # (B_v, 1024, 2)
                                batched_w2d = torch.stack(valid_w2d)        # (B_v, 1024, 2)
                                batched_pose = torch.stack(valid_pose_init) # (B_v, 7)
                                batched_K = torch.stack(valid_cam_K)        # (B_v, 3, 3)
                                
                                camera.set_param(batched_K)
                                pnp_cost_fun.set_param(batched_x2d, batched_w2d)
                                
                                # 唯一一次调用！此时 GPU 矩阵乘法单元满载，且无需保留多个冗余计算图
                                _, _, batched_pose_opt_plus, _, batched_logweights, batched_cost = pnp_solver.monte_carlo_forward(
                                    batched_x3d, batched_x2d, batched_w2d, camera, pnp_cost_fun,
                                    pose_init=batched_pose, force_init_solve=True, with_pose_opt_plus=True
                                )
                                # 排除填充造成的 0 权重影响，计算真实的均值作为归一化因子
                                loss_mc = pose_loss_fn(batched_logweights, batched_cost)
                                # 可以按有效图片的数量求个均值
                                loss_pose = loss_mc.mean()
                                # 限制最大值防止早期偶尔爆点
                                loss_pose = torch.clamp(loss_pose, max=200.0)
                            else:
                                loss_pose = 0.0 * pred_w2d.sum()
                    # === 损失融合 ===
                    l_l = [
                        3*torch.sum(current_loss['Front_L1Losses']),
                        3*torch.sum(current_loss['Back_L1Losses']) ,
                        current_loss['mask_loss'],
                    ]
                    loss = l_l[0] + l_l[1] + l_l[2] + pose_loss_weight * loss_pose
                    loss = loss / accumulation_steps  # 1. 损失归一化

                
                if not ide_debug:
                    torch.distributed.barrier()  
                    nan_flag = torch.tensor([int(torch.isnan(loss).any())], device=loss.device)
                    dist.all_reduce(nan_flag, op=dist.ReduceOp.SUM)
                    if nan_flag.item() > 0:
                        for m in net.model.modules():
                            if isinstance(m, torch.nn.BatchNorm2d):
                                m.reset_running_stats()
                        continue
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0: # 2. 达到累积步数才更新
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                # torch.cuda.empty_cache()
                
                if args.local_rank == 0:
                    print('dataset:%s - obj%s'%(os.path.basename(DATASET_PATH), str(obj_id).rjust(2, '0')), 
                        "iteration_step:", iteration_step, 
                        "loss_front:", torch.sum(current_loss['Front_L1Losses']).item(),  
                        "loss_back:", torch.sum(current_loss['Back_L1Losses']).item(),  
                        "loss_mask:", current_loss['mask_loss'].item(),
                        "loss_pose:", loss_pose.item(),
                        "total_loss:", loss.item(),
                        flush=True
                    )
                    
                iteration_step = iteration_step + 1
                if iteration_step >=total_iteration:
                    end_training = True
                    break
            if end_training == True:
                if args.local_rank == 0:
                    print('end the training in iteration_step:', iteration_step)
                break
             