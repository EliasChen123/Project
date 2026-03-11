import torch, cv2, kornia
import numpy as np

def vis_rgb_mask_Coord(rgb_c, pred_mask, pred_front_code, pred_back_code, img_path = None):
    # 此函数用于 crop 后的局部可视化，保持不变或简化防止报错
    text_list = ['RGB', 'Mask']
    mean = torch.tensor([0.485, 0.456, 0.406]).to(rgb_c.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(rgb_c.device)
    def reverse_normalize(tensor):
        if tensor.dim() == 4:
            mean_ = mean.view(1, 3, 1, 1) 
            std_ = std.view(1, 3, 1, 1)
        else: 
            mean_ = mean.view(3, 1, 1)
            std_ = std.view(3, 1, 1)
        return tensor * std_ + mean_  
    reversed_rgb_c = reverse_normalize(rgb_c)
    reversed_rgb_c = reversed_rgb_c * 255
    reversed_rgb_c = kornia.geometry.transform.resize(
        reversed_rgb_c, 
        (128, 128),
        interpolation='bilinear',
    )
    reversed_rgb_c = reversed_rgb_c.permute(0,2,3,1)
    pred_mask_s = pred_mask[...,None].repeat(1,1,1,3) * 255
    pred_front_code = pred_front_code.clone() * 255
    pred_back_code = pred_back_code.clone() * 255
    s0,s1,s2 = reversed_rgb_c.shape[0]*reversed_rgb_c.shape[1], reversed_rgb_c.shape[2], reversed_rgb_c.shape[3]
    reversed_rgb_c = reversed_rgb_c.reshape((s0,s1,s2))
    pred_mask_s = pred_mask_s.reshape((s0,s1,s2))
    
    # 简化：只保留 RGB 和 Mask，移除后面的 code 可视化防止出错
    save_tensor = torch.cat([reversed_rgb_c, pred_mask_s], dim = 1)
    save_numpy = save_tensor.detach().cpu().numpy()
    
    if img_path is not None:
        cv2.imwrite(img_path, save_numpy)
    return save_numpy

def zero_other_masks_by_conf(pred_mask, conf_s):
    pred_mask = pred_mask.permute(1,0,2,3)
    B, N, H, W = pred_mask.shape 
    conf_s = conf_s[None,:,None,None].repeat(1, 1, H, W) * pred_mask
    max_indices = torch.argmax(conf_s, dim=1) 
    mask_indices = torch.arange(N, device=pred_mask.device).view(1, N, 1, 1) 
    is_max_mask = (max_indices.unsqueeze(1) == mask_indices) 
    max_mask = is_max_mask.float() 
    processed_masks = pred_mask * max_mask
    return processed_masks.permute(1,0,2,3)

def vis_rgb_mask_Coord_origin(cam_K, obj_ids_l, obj_ids_all, BBox_3d_l, Rts_l, conf_s, rgb_c, pred_mask, pred_front_code, pred_back_code, img_path = None):
    
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).to(rgb_c.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(rgb_c.device)
    def reverse_normalize(tensor):
        if tensor.dim() == 4:
            mean_ = mean.view(1, 3, 1, 1) 
            std_ = std.view(1, 3, 1, 1)
        else: 
            mean_ = mean.view(3, 1, 1)
            std_ = std.view(3, 1, 1)
        return tensor * std_ + mean_  
    reversed_rgb_c = reverse_normalize(rgb_c)
    reversed_rgb_c = reversed_rgb_c * 255
    reversed_rgb_c = reversed_rgb_c.permute(0,2,3,1) # (B, H, W, 3)
    
    pred_mask = zero_other_masks_by_conf(pred_mask, conf_s)
    
    pred_mask_s_copy = pred_mask.repeat(1,3,1,1).clone().permute(0,2,3,1)
    rand_RGB = torch.rand(size=(pred_mask.shape[0], 3),device=pred_mask.device)
    pred_mask_s = pred_mask.repeat(1,3,1,1) * rand_RGB[..., None,None] * 255
    pred_mask_s = pred_mask_s.permute(0,2,3,1)
    
    # 核心：生成覆盖 Mask 的原图
    reversed_rgb_c_mask = (reversed_rgb_c.clone()[0] + 0.5 * pred_mask_s.sum(dim=0)) / (pred_mask_s_copy.sum(dim=0)*0.5 + 1)
    
    s0,s1,s2 = reversed_rgb_c.shape[0]*reversed_rgb_c.shape[1], reversed_rgb_c.shape[2], reversed_rgb_c.shape[3]

    # 构造第一行 (Mask & 6D Poses)
    # 注意：这里我们不再追加 3D(Front) 和 3D(Back) 到 line_1，只保留这一张图
    line_1 = [reversed_rgb_c_mask]
    
    # 转换为 Numpy
    save_numpy_line_1 = torch.cat(line_1, dim = 1).detach().cpu().numpy()
    
    # 绘制文字
    cv2.putText(save_numpy_line_1, 'HccePose', (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0) , 3, cv2.LINE_AA)
    
    # 绘制 6D 姿态边框
    for i, (obj_id, Rt_i) in enumerate(zip(obj_ids_l, Rts_l)):
        BBox_3d = BBox_3d_l[obj_ids_all.index(obj_id)].copy().reshape((-1, 3))
        BBox_3d = (Rt_i[:3,:3] @ BBox_3d.T).T + Rt_i[:3,3:].reshape((-1, 3))
        BBox_3d[:, 0] = BBox_3d[:, 0] / BBox_3d[:, 2] * cam_K[0,0] + cam_K[0,2]
        BBox_3d[:, 1] = BBox_3d[:, 1] / BBox_3d[:, 2] * cam_K[1,1] + cam_K[1,2]
        BBox_2d = BBox_3d[:, :2].reshape((-1, 2, 2))
        
        rand_RGB_np = rand_RGB[i].clone().cpu().numpy() * 255
        rand_RGB_np = rand_RGB_np.astype(np.uint8)
        
        # 绘制线框 (只绘制在这一张图上)
        for BBox_2d_i in BBox_2d:
            cv2.line(save_numpy_line_1, BBox_2d_i[0].astype(np.int32), BBox_2d_i[1].astype(np.int32), 
                     (int(rand_RGB_np[0]), int(rand_RGB_np[1]), int(rand_RGB_np[2])), 2)
    
    # [关键修改]
    # save_numpy_2: 直接设为上面画好框的那张图 (Original + Mask + Pose)
    # 不再进行 resize * 2 操作
    # 不再拼接 Front/Back Code Map
    
    final_result_img = save_numpy_line_1 # 只有一张图，大小为原图大小 (1920x1080)
    
    # 为了兼容 tester.py 的接口，vis1 和 vis2 都返回这张图
    return final_result_img, final_result_img