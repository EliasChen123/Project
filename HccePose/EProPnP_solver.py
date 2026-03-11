import cv2
import torch
import numpy as np 
from scipy.spatial.transform import Rotation as R
from EPro_PnP.epropnp import EProPnP6DoF
from EPro_PnP.levenberg_marquardt import LMSolver,RSLMSolver # 引入 RSLMSolver(类RANSAC),底层优化器
from EPro_PnP.camera import PerspectiveCamera     # 相机模型
from EPro_PnP.cost_fun import AdaptiveHuberPnPCost      # 使用 Huber 鲁棒代价函数

def solve_EPro_PnP(pred_front_np, pred_w2d_np, pred_mask_np, coord_image_np, cam_K):
    """
    使用 EPro-PnP 替换原 OpenCV 求解逻辑，引入尺度转换 (mm -> m)
    """
    return_info = {
        'success': False,
        'rot': np.eye(3),
        'tvecs': np.zeros((3, 1)),
        'inliers': np.zeros((1)),
    }
    
    # 提取 Mask 覆盖区域的点
    mask = pred_mask_np > 0
    if not np.any(mask):
        return return_info
    # 1. 提取正面点，并将 3D 坐标从毫米转换为米 ( / 1000.0)
    x3d_np = (pred_front_np[mask].astype(np.float32)) / 1000.0
    x2d_np = coord_image_np[mask].astype(np.float32)
    w2d_np = pred_w2d_np[mask].astype(np.float32)

    # pts_f = pred_front_np[mask].astype(np.float32)
    # pts_2d = coord_image_np[mask].astype(np.float32)
    # w2d_f = pred_w2d_front_np[mask].astype(np.float32)

    # # 将前后向数据串联在一起
    # x3d_np = np.concatenate([pts_f], axis=0)
    # x2d_np = np.concatenate([pts_2d, pts_2d], axis=0)
    # w2d_np = np.concatenate([w2d_f, w2d_b], axis=0)

    if x3d_np.shape[0] <= 16:
        return return_info

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= 核心修复：提供初始位姿 (pose_init) =================
    # 使用 OpenCV EPnP 提供稳定的初始解，规避 RSLMSolver 在双表面歧义下的崩溃
    # 2. 仅使用正面数据进行初始姿态估计
    success_cv, rvec_cv, tvec_cv = cv2.solvePnP(
        x3d_np, x2d_np, cam_K, None, flags=cv2.SOLVEPNP_EPNP
    )
    # 增加严格的容错处理
    pose_init = None
    if success_cv and rvec_cv is not None and tvec_cv is not None:
        try:
            rot_cv, _ = cv2.Rodrigues(rvec_cv)
            quat_cv = R.from_matrix(rot_cv).as_quat() # Scipy: [x, y, z, w]
            quat_pt = np.array([quat_cv[3], quat_cv[0], quat_cv[1], quat_cv[2]]) 
            pose_init_np = np.concatenate([tvec_cv.flatten(), quat_pt])
            pose_init = torch.from_numpy(pose_init_np).unsqueeze(0).to(device, dtype=torch.float32)
        except:
            pose_init = None
    # ======================================================================
    x3d = torch.from_numpy(x3d_np).unsqueeze(0).to(device)  # (1, 2N, 3)
    x2d = torch.from_numpy(x2d_np).unsqueeze(0).to(device)  # (1, 2N, 2)
    w2d = torch.from_numpy(w2d_np).unsqueeze(0).to(device)  # (1, 2N, 2)
    cam_K_tensor = torch.from_numpy(cam_K).unsqueeze(0).to(device, dtype=torch.float32)

    # 实例化推理求解器
    camera = PerspectiveCamera(cam_mats=cam_K_tensor)
    cost_fun = AdaptiveHuberPnPCost(relative_delta=0.5)
    cost_fun.set_param(x2d, w2d)

    solver = LMSolver(
        dof=6, num_iter=10, normalize=True,
        init_solver=RSLMSolver(dof=6, num_points=16, num_proposals=64, num_iter=3)
    ).to(device)
    
    epropnp = EProPnP6DoF(
        mc_samples=512, num_iter=4, normalize=True, solver=solver
    ).to(device)

    with torch.no_grad():
        # fast_mode=True 禁用 LM trust region 进行快速收敛
        # 强制关闭 autocast，确保底层矩阵运算使用 Float32
        with torch.autocast(device_type='cuda', enabled=False):
            # 显式转换为 float() 确保万无一失
            # pose_opt, _, _, _ = epropnp(x3d.float(), x2d.float(), w2d.float(), camera, cost_fun, fast_mode=True)
            pose_opt, _, _, _ = epropnp(x3d.float(), x2d.float(), w2d.float(), camera, cost_fun, pose_init=pose_init, fast_mode=True
            )
    pose_opt_np = pose_opt.squeeze(0).cpu().numpy() # [tx, ty, tz, qw, qi, qj, qk]
    
    tvecs = (pose_opt_np[:3] * 1000.0).reshape(3, 1)
    quat = pose_opt_np[3:] # [w, x, y, z]
    # Scipy 接受的四元数格式为 [x, y, z, w]
    quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]]) 
    rot_mat = R.from_quat(quat_scipy).as_matrix()

    return_info['success'] = True
    return_info['rot'] = rot_mat
    return_info['tvecs'] = tvecs
    return return_info