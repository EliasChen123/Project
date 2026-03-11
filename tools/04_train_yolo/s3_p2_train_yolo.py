# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

# '''

# Train YOLOv11. After training, the folder structure is:
# ```
# demo-bin-picking
# |--- models
# |--- train_pbr
# |--- yolo11
#       |--- train_obj_s
#             |--- detection
#                 |--- obj_s
#                     |--- yolo11-detection-obj_s.pt
#             |--- images
#             |--- labels
#             |--- yolo_configs
#                 |--- data_objs.yaml
#             |--- autosplit_train.txt
#             |--- autosplit_val.txt
# ```

# ------------------------------------------------------    

# 训练 YOLOv11。训练完成后，文件夹结构如下：
# ```
# demo-bin-picking
# |--- models
# |--- train_pbr
# |--- yolo11
#       |--- train_obj_s
#             |--- detection
#                 |--- obj_s
#                     |--- yolo11-detection-obj_s.pt
#             |--- images
#             |--- labels
#             |--- yolo_configs
#                 |--- data_objs.yaml
#             |--- autosplit_train.txt
#             |--- autosplit_val.txt
# ```
# '''

import os
if __name__ == '__main__':

    # Specify the path to the dataset folder.
    # 指定数据集文件夹的路径。
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    TARGET_DIR = os.path.join(ROOT_DIR, "eye_trocar_v5")
    DATASET_PATH = os.path.join(TARGET_DIR, "bop_dataset_final")
    # dataset_path = 'trocar'
    
    # Specify the number of GPUs and the number of training epochs.  
    # For example, use 8 GPUs to train for 100 epochs.
    # 指定 GPU 的数量以及训练轮数。  
    # 例如使用 8 张 GPU 进行 100 轮训练。
    gpu_num = 1
    epochs = 100
    
    # Train
    # 开始训练
    dataset_name = os.path.basename(DATASET_PATH)
    task_suffix = 'detection'
    dataset_pbr_path = os.path.join(DATASET_PATH, 'train_pbr')
    train_multi_path = os.path.join(ROOT_DIR, 'yolo_train', 'train.py')
    data_objs_path = os.path.join(TARGET_DIR, 'yolo11', 'train_obj_s', 'yolo_configs', 'data_objs.yaml')
    save_dir = os.path.join(os.path.dirname(os.path.dirname(data_objs_path)), task_suffix, f"obj_s")
    
    model_name = f"yolo11-{task_suffix}-obj_s.pt"
    final_model_path = os.path.join(save_dir, model_name)
    obj_s_path = os.path.dirname(final_model_path)
    batch_size = 12*gpu_num
    # while 1:
    #     if not os.path.exists('%s'%obj_s_path):
    #         os.system("python %s --data_path '%s' --epochs %s --imgsz 640 --batch %s --gpu_num %s --task %s"%(train_multi_path, data_objs_path, str(epochs), str(batch_size), str(gpu_num), task_suffix))
    #     if os.path.exists('%s'%obj_s_path):
    #         break
    # pass
    # ✅ 安全的单次执行代码
    if not os.path.exists(obj_s_path):
        print(f"开始训练，使用命令: python {train_multi_path} ...")
        exit_code = os.system("python %s --data_path '%s' --epochs %s --imgsz 640 --batch %s --gpu_num %s --task %s"%(train_multi_path, data_objs_path, str(epochs), str(batch_size), str(gpu_num), task_suffix))
        
        if exit_code != 0:
            print("❌ 训练过程中发生错误，程序已停止。请检查日志。")
            exit(1)
    else:
        print("✅ 输出目录已存在，跳过训练。")
