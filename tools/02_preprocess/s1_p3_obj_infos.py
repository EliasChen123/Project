# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import os
import numpy as np
from kasal.utils import load_ply_model, load_json2dict, get_all_ply_obj, write_dict2json

if __name__ == '__main__':
    
    # Specify the dataset path. The `dataset_path` directory must contain a `models` folder.
    # 指定数据集的路径。`dataset_path`目录下必须包含一个`models`文件夹。
    # '''
    # trocar
    # |--- models
    #     |--- obj_000001.ply
    # '''
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    TARGET_DIR = os.path.join(ROOT_DIR, "output", "trocar")

    DATASET_PATH = os.path.join(TARGET_DIR, "models")

    # Retrieve all PLY files from the folder and its subfolders.
    # 获取该文件夹及其所有子文件夹中的所有PLY文件。
    MODELS_PATH = get_all_ply_obj(DATASET_PATH)

    # Iterate over all PLY files to compute the diameter, per-axis min/max of vertex coordinates, 
    # and the bounding-box side lengths. Load the rotational-symmetry prior file, 
    # then merge everything into `models_info.json`.
    # 遍历所有 PLY 文件，计算直径、顶点坐标各轴的最小/最大值，以及包围盒边长。
    # 加载旋转对称先验文件，最后合并所有信息生成 `models_info.json`。
    models_info = {}
    for model_path in MODELS_PATH:
        # --- 新增：跳过不符合命名规范的文件 ---
        filename = os.path.basename(model_path)
        if not filename.startswith('obj_'):
            print(f"[Warning] 跳过非标准命名文件: {filename}")
            continue
        # ----------------------------------

        ply_info = load_ply_model(model_path)
        model_info = {
            "diameter" : float(ply_info['diameter']),
            "max_x" : float(np.max(ply_info['vertices'], axis = 0)[0]),
            "max_y" : float(np.max(ply_info['vertices'], axis = 0)[1]),
            "max_z" : float(np.max(ply_info['vertices'], axis = 0)[2]),
            "min_x" : float(np.min(ply_info['vertices'], axis = 0)[0]),
            "min_y" : float(np.min(ply_info['vertices'], axis = 0)[1]),
            "min_z" : float(np.min(ply_info['vertices'], axis = 0)[2]),
            "size_x" : float(np.max(ply_info['vertices'], axis = 0)[0] - np.min(ply_info['vertices'], axis = 0)[0]),
            "size_y" : float(np.max(ply_info['vertices'], axis = 0)[1] - np.min(ply_info['vertices'], axis = 0)[1]),
            "size_z" : float(np.max(ply_info['vertices'], axis = 0)[2] - np.min(ply_info['vertices'], axis = 0)[2]),
        }
        symmetry_type_dict = None
        sym_type_file = os.path.join(os.path.dirname(model_path), os.path.basename(model_path).split('.')[0]+'_sym_type.json')
        if os.path.exists(sym_type_file):
            symmetry_type_dict = load_json2dict(sym_type_file)
        if symmetry_type_dict is not None:
            if 'symmetries_continuous' in symmetry_type_dict['current_obj_info']:
                model_info["symmetries_continuous"] = symmetry_type_dict['current_obj_info']["symmetries_continuous"]
            if 'symmetries_discrete' in symmetry_type_dict['current_obj_info']:
                model_info["symmetries_discrete"] = symmetry_type_dict['current_obj_info']["symmetries_discrete"]
        model_id = str(int(os.path.basename(model_path).split('.')[0].split('obj_')[1]))
        models_info[model_id] = model_info
    
    # Check whether the folder contains any objects; if so, generate `models_info.json`.
    # 判断该文件夹是否包含物体；若包含，则生成 `models_info.json`。
    if len(MODELS_PATH) > 0:
        models_info_path = os.path.join(os.path.dirname(model_path), 'models_info.json')
        write_dict2json(models_info_path, models_info)
        print(f"[Info] 已生成物体信息文件: {models_info_path}")
    else:
        print("[Warning] 未检测到任何符合命名规范的物体文件，未生成 models_info.json。")
    pass