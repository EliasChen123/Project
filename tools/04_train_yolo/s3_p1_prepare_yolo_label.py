# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import os, json
# 确保已经通过 export PYTHONPATH=$PYTHONPATH:$(pwd) 设置了路径，否则这里可能报错
from yolo_train.label import convert_train_pbr_2_yolo, generate_yaml

if __name__ == '__main__':
    
    # === 路径配置修复版 ===
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    TARGET_DIR = os.path.join(ROOT_DIR, 'output', 'trocar')

    DATASET_PATH = os.path.join(TARGET_DIR)
    
    TRAIN_PBR_PATH = os.path.join(DATASET_PATH, "train_pbr")
    
    OUTPUT_PATH  = os.path.join(TARGET_DIR, 'yolo11', 'train_obj_s')
    
    MODELS_INFO_PATH = os.path.join(DATASET_PATH, 'models', 'models_info.json')

    if not os.path.exists(TARGET_DIR):
        raise FileNotFoundError(f"Dataset path not found: {TARGET_DIR}")

    print(f"[INFO] Checking models info at: {MODELS_INFO_PATH}")
    print(f"[INFO] Input dataset path: {TRAIN_PBR_PATH}")
    print(f"[INFO] Output YOLO path: {OUTPUT_PATH}")
    
    # === 逻辑处理 ===
    
    # 检查文件是否存在
    if not os.path.exists(MODELS_INFO_PATH):
        raise FileNotFoundError(f"无法找到模型信息文件: {MODELS_INFO_PATH}\n请确认当前运行目录是在项目根目录下 (HCCEPose-main)")

    if not os.path.exists(TRAIN_PBR_PATH):
        raise FileNotFoundError(f"无法找到 PBR 数据集: {TRAIN_PBR_PATH}\n请确认之前的渲染步骤是否成功生成了 train_pbr 文件夹")

    # 提取所有物体 ID
    obj_id_list = []
    with open(MODELS_INFO_PATH, "r") as f:
        scene_gt_data = json.load(f)
    for key_ in scene_gt_data:
        obj_id_list.append(key_)
        
    # 将 BOP 格式转换为 YOLO 格式
    # 注意：这里传入的是修复后的绝对路径
    convert_train_pbr_2_yolo(TRAIN_PBR_PATH, OUTPUT_PATH, obj_id_list)
    generate_yaml(OUTPUT_PATH, obj_id_list)
    print("[INFO] Dataset preparation complete!")