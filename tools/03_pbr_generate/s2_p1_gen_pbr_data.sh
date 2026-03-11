#!/bin/bash
GPU_ID=$1
SCENE_NUM=$2

# === 关键修改：自动获取文件的绝对路径 ===
# 无论你传入的是相对路径还是绝对路径，这里都会强制转为绝对路径
# 这样即便后面 cd 到了别的目录，也能通过绝对路径找到原始文件
cc0textures=$(readlink -f "$3")
dataset_path=$(readlink -f "$4")
s2_p1_gen_pbr_data=$(readlink -f "$5")

for (( SCENE_ID=0; SCENE_ID<$SCENE_NUM; SCENE_ID++ ))
do
    SCENE_ID_PADDED=$(printf "%06d" $SCENE_ID)
    echo "Running scene $SCENE_ID_PADDED on GPU $GPU_ID"
    export EGL_DEVICE_ID=$GPU_ID
    
    # 1. 进入数据集目录（因为Python脚本需要根据当前目录获取数据集名称）
    cd "$dataset_path"
    
    # 2. 运行 BlenderProc
    # 此时 $s2_p1_gen_pbr_data 是绝对路径（例如 /home/user/.../script.py），所以一定能找到
    blenderproc run "$s2_p1_gen_pbr_data" $GPU_ID "$cc0textures"
done