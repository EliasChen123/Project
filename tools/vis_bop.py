import json
import os
import cv2
import numpy as np
import glob
import tkinter as tk  # 用于获取屏幕分辨率

def get_screen_size():
    """获取屏幕分辨率，如果获取失败则返回默认值"""
    try:
        root = tk.Tk()
        root.withdraw() # 隐藏主窗口
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        print("无法获取屏幕尺寸，使用默认 1920x1080")
        return 1920, 1080

def draw_axis(img, R, t, K, scale=0.01, thickness=2):
    """ 在图像上绘制 3D 坐标轴 """
    points_3d = np.float32([[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]])
    rvec, _ = cv2.Rodrigues(R)
    image_points, _ = cv2.projectPoints(points_3d, rvec, t, K, distCoeffs=None)
    image_points = image_points.reshape(-1, 2).astype(int)
    
    h, w = img.shape[:2]
    # 边界检查
    if 0 <= image_points[0][0] < w and 0 <= image_points[0][1] < h:
        origin = tuple(image_points[0])
        # 使用动态传入的 thickness
        img = cv2.line(img, origin, tuple(image_points[1]), (0, 0, 255), thickness) # X - Red
        img = cv2.line(img, origin, tuple(image_points[2]), (0, 255, 0), thickness) # Y - Green
        img = cv2.line(img, origin, tuple(image_points[3]), (255, 0, 0), thickness) # Z - Blue
    return img

def visualize_bop():
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(current_dir, "eye_trocar_v5", "bop_dataset_raw","chunk_000","train_pbr", "000000")
    # base_path = os.path.join(current_dir, "eye_trocar_v3", "bop_dataset_raw","train_pbr", "000000")
    
    rgb_path = os.path.join(base_path, "rgb")
    scene_camera_path = os.path.join(base_path, "scene_camera.json")
    scene_gt_path = os.path.join(base_path, "scene_gt.json")
    scene_gt_info_path = os.path.join(base_path, "scene_gt_info.json") 
    
    if not os.path.exists(scene_camera_path) or not os.path.exists(scene_gt_path):
        print(f"错误: 找不到必要的元数据文件。请检查路径: {base_path}")
        return

    with open(scene_camera_path, 'r') as f:
        scene_camera = json.load(f)
    with open(scene_gt_path, 'r') as f:
        scene_gt = json.load(f)
    
    scene_gt_info = {}
    if os.path.exists(scene_gt_info_path):
        with open(scene_gt_info_path, 'r') as f:
            scene_gt_info = json.load(f)

    img_files = sorted(glob.glob(os.path.join(rgb_path, "*.jpg")) + glob.glob(os.path.join(rgb_path, "*.png")))
    
    # --- 1. 获取屏幕大小 ---
    screen_w, screen_h = get_screen_size()
    print(f"检测到屏幕分辨率: {screen_w}x{screen_h}")
    print(f"找到 {len(img_files)} 张图片，开始可视化...")
    print(">>> 操作: [A]上一张 | [D/空格]下一张 | [Q]退出")

    idx = 0  
    while idx >= 0 and idx < len(img_files):
        img_file = img_files[idx]
        img_filename = os.path.basename(img_file)
        img_id_str = str(int(os.path.splitext(img_filename)[0]))
        
        img = cv2.imread(img_file)
        
        # --- 2. 动态计算线宽 (关键步骤) ---
        # 如果图片高度是 1080，线宽约 2；如果是 2160(4K)，线宽约 4
        # 这样保证缩放后线条依然清晰可见
        img_h, img_w = img.shape[:2]
        base_thickness = max(2, int(img_h * 0.002)) 
        font_scale = max(0.5, img_h * 0.001)

        if img_id_str in scene_gt:
            K = np.array(scene_camera[img_id_str]["cam_K"]).reshape(3, 3)
            objects = scene_gt[img_id_str]
            infos = scene_gt_info.get(img_id_str, [])

            for obj_idx, obj in enumerate(objects):
                R = np.array(obj["cam_R_m2c"]).reshape(3, 3)
                t = np.array(obj["cam_t_m2c"]) / 1000.0 
                # 传入动态线宽
                img = draw_axis(img, R, t, K, thickness=base_thickness)
                
                if obj_idx < len(infos):
                    bbox = infos[obj_idx].get("bbox_visib", infos[obj_idx].get("bbox_obj"))
                    if bbox and bbox[2] > 0 and bbox[3] > 0:
                        x, y, w, h = bbox
                        # BBox 线宽设为 base_thickness 的一半，但至少为 1
                        bbox_thick = max(1, int(base_thickness / 1.5))
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), bbox_thick)
                        cv2.putText(img, f"ID:{obj['obj_id']}", (x, y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), bbox_thick)
        else:
            cv2.putText(img, "No GT", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, (0, 255, 255), base_thickness)

        # --- 3. 自适应缩放逻辑 ---
        target_h = screen_h * 0.8  # 目标高度：屏幕的 80%
        target_w = screen_w * 0.8  # 目标宽度：屏幕的 80%
        
        scale_h = target_h / img_h
        scale_w = target_w / img_w
        
        # 选择较小的缩放比例以适应屏幕，且如果图片本来就小，不超过 1.0 (可选，去掉 min 则允许小图放大)
        display_scale = min(scale_h, scale_w, 1.0) 
        
        new_w = int(img_w * display_scale)
        new_h = int(img_h * display_scale)
        
        # 使用 INTER_AREA 插值，这是缩小图片时保持线条清晰的最佳算法
        vis_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow(f"BOP Vis", vis_img) 

        key = cv2.waitKey(0) 
        
        KEYS_PREV = [ord('a'), 81, 2424832, 65361] 
        KEYS_NEXT = [ord('d'), 32, 83, 2555904, 65363] 
        
        if key == ord('q'):
            break
        elif key in KEYS_PREV:
            idx = max(0, idx - 1)
        elif key in KEYS_NEXT:
            idx += 1
            
        print(f"Img: {img_filename} | Scale: {display_scale:.2f} | Thickness: {base_thickness}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_bop()