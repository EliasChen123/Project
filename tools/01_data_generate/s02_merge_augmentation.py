import os
import glob
import json
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# ==========================================
# 1. 配置与路径 (Configuration)
# ==========================================
# 动态获取路径，确保脚本在不同位置都能运行
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_DIR = os.path.join(ROOT_DIR, "output", "trocar")

RAW_DIR = os.path.join(TARGET_DIR, "bop_dataset_raw")

FINAL_DIR = os.path.join(TARGET_DIR, "train_pbr")

# 背景图片库路径
BG_DIR = os.path.join(os.path.dirname(ROOT_DIR), "public_data", "val2017")

# ==========================================
# 2. 辅助函数 (Helper Functions)
# ==========================================
def alpha_blend(fg_img, bg_img):
    """
    执行前景与背景的 Alpha Blending
    """
    h, w = fg_img.shape[:2]
    bg_h, bg_w = bg_img.shape[:2]
    
    if bg_h < h or bg_w < w:
        bg_resized = cv2.resize(bg_img, (w, h))
    else:
        # 随机裁剪一部分背景
        dy = np.random.randint(0, bg_h - h + 1)
        dx = np.random.randint(0, bg_w - w + 1)
        bg_resized = bg_img[dy:dy+h, dx:dx+w]

    alpha = fg_img[:, :, 3] / 255.0
    fg_rgb = fg_img[:, :, :3]
    alpha_exp = alpha[:, :, np.newaxis]
    
    # 混合公式: I = alpha * FG + (1 - alpha) * BG
    composite = (fg_rgb * alpha_exp + bg_resized * (1 - alpha_exp)).astype(np.uint8)
    return composite

# ==========================================
# 3. 主逻辑 (Main Execution)
# ==========================================
def main():
    # 0. 准备工作：清理旧数据
    if os.path.exists(FINAL_DIR):
        print(f"⚠️ 警告: 输出目录 {FINAL_DIR} 已存在，正在删除以重新生成...")
        shutil.rmtree(FINAL_DIR)
    os.makedirs(FINAL_DIR, exist_ok=True)

    # 1. 加载背景图片库
    print(f"📂 正在加载背景图片库: {BG_DIR}")
    bg_paths = glob.glob(os.path.join(BG_DIR, "*"))
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    bg_paths = [p for p in bg_paths if os.path.splitext(p)[1].lower() in valid_exts]
    
    if not bg_paths:
        raise FileNotFoundError(f"❌ 错误: 在 {BG_DIR} 未找到背景图片！请检查路径。")
    print(f"✅ 找到 {len(bg_paths)} 张背景图片。")

    # 2. 扫描原始数据块 (chunks)
    chunk_dirs = sorted([d for d in os.listdir(RAW_DIR) if d.startswith("chunk_")])
    if not chunk_dirs:
        raise FileNotFoundError(f"❌ 错误: 在 {RAW_DIR} 未找到以 'chunk_' 开头的文件夹。")
    
    print(f"🚀 发现 {len(chunk_dirs)} 个数据块，开始高效合并与处理...")

    # 全局场景计数器 (对应 000000, 000001, ...)
    global_scene_counter = 0

    # --- 遍历每一个 Chunk ---
    for chunk_name in tqdm(chunk_dirs, desc="Processing Chunks"):
        chunk_path = os.path.join(RAW_DIR, chunk_name)
        
        # 寻找该 chunk 下的 scene 文件夹 (通常是 train_pbr/000000)
        # BlenderProc 生成的结构通常固定，直接尝试定位
        potential_scene_path = os.path.join(chunk_path, "train_pbr", "000000")
        
        if not os.path.exists(potential_scene_path):
            # 备用搜索逻辑
            found = False
            for root, dirs, files in os.walk(chunk_path):
                if "000000" in dirs and "train_pbr" in root:
                    potential_scene_path = os.path.join(root, "000000")
                    found = True
                    break
            if not found:
                print(f"⚠️ 跳过 {chunk_name}: 未找到标准 Scene 数据结构")
                continue

        src_scene_path = potential_scene_path
        
        # 定义目标 Scene 路径 (例如: bop_dataset_final/train_pbr/000001)
        target_scene_path = os.path.join(FINAL_DIR, f"{global_scene_counter:06d}")
        os.makedirs(target_scene_path, exist_ok=True)

        # -------------------------------------------------
        # A. 直接复制元数据 (JSON)
        # -------------------------------------------------
        for json_name in ["scene_gt.json", "scene_camera.json", "scene_gt_info.json"]:
            src_json = os.path.join(src_scene_path, json_name)
            if os.path.exists(src_json):
                shutil.copy(src_json, os.path.join(target_scene_path, json_name))

        # -------------------------------------------------
        # B. 直接复制不需要修改的文件夹 (depth, mask, mask_visib)
        # -------------------------------------------------
        for folder_name in ["depth", "mask", "mask_visib"]:
            src_folder = os.path.join(src_scene_path, folder_name)
            dst_folder = os.path.join(target_scene_path, folder_name)
            if os.path.exists(src_folder):
                # copytree 效率极高
                shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)

        # -------------------------------------------------
        # C. 重写 RGB (加载 -> 合成 -> 保存)
        # -------------------------------------------------
        src_rgb_dir = os.path.join(src_scene_path, "rgb")
        dst_rgb_dir = os.path.join(target_scene_path, "rgb")
        os.makedirs(dst_rgb_dir, exist_ok=True)

        # 获取该文件夹下所有图片
        img_paths = sorted(glob.glob(os.path.join(src_rgb_dir, "*.png")))
        
        for img_path in img_paths:
            img_name = os.path.basename(img_path) # e.g. 000000.png
            
            # 读取前景 (RGBA)
            img_fg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img_fg is None: continue

            # 随机背景
            rand_bg_path = np.random.choice(bg_paths)
            rand_bg = cv2.imread(rand_bg_path)
            
            # 合成
            final_rgb = alpha_blend(img_fg, rand_bg)
            
            # 保存 (转为 jpg 以符合 PBR 常见格式，减小体积；若需 png 可修改后缀)
            save_name = os.path.splitext(img_name)[0] + ".jpg"
            cv2.imwrite(os.path.join(dst_rgb_dir, save_name), final_rgb)

        # 计数器递增
        global_scene_counter += 1

    # -------------------------------------------------
    # 4. 处理全局 Camera JSON
    # -------------------------------------------------
    # 假设每个 Chunk 的相机参数一致，取第一个 Chunk 的 camera.json 复制到数据集根目录
    if chunk_dirs:
        first_chunk_path = os.path.join(RAW_DIR, chunk_dirs[0])
        # 尝试在不同层级寻找 camera.json
        possible_cam_jsons = [
            os.path.join(first_chunk_path, 'camera.json'),
            os.path.join(first_chunk_path, 'bop_data', 'camera.json'), # BlenderProc 常见输出位置
            os.path.join(RAW_DIR, 'camera.json')
        ]
        
        found_cam = False
        target_cam_path = os.path.join(os.path.dirname(FINAL_DIR), 'camera.json')
        
        for cam_json in possible_cam_jsons:
            if os.path.exists(cam_json):
                print(f"[INFO] 找到相机参数文件: {cam_json} -> 复制到 dataset 根目录")
                shutil.copy(cam_json, target_cam_path)
                found_cam = True
                break
        if not found_cam:
            print(f"[WARNING] ⚠️ 未在源文件夹中找到全局 'camera.json'，请手动检查是否需要该文件。")

    print("\n" + "="*50)
    print(f"🎉 处理完成！")
    print(f"📂 输出目录: {FINAL_DIR}")
    print(f"🔢 总合并场景数 (Scenes): {global_scene_counter}")
    print("="*50)

if __name__ == "__main__":
    main()