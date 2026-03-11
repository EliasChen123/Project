import cv2
import os
import sys
import time
import numpy as np
from HccePose.bop_loader import bop_dataset
from HccePose.tester import Tester

if __name__ == '__main__':
    # ------------------- 1. 基础路径配置 -------------------
    sys.path.insert(0, os.getcwd())
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 【请确认】数据集文件夹路径
    dataset_path = os.path.join(ROOT_DIR, 'eye_trocar_v5')
    # ------------------- 2. 模型加载 -------------------
    # 【请确认】需要检测的物体ID列表
    obj_ids = [1]
    CUDA_DEVICE = '0'
    show_op = True
    print("⏳ 正在加载 YOLO 和 HccePose 模型，请稍候...")
    # 初始化数据加载器和测试器
    bop_dataset_item = bop_dataset(dataset_path)
    Tester_item = Tester(bop_dataset_item, show_op=show_op, CUDA_DEVICE=CUDA_DEVICE,efficientnet_key='b4')
    print("✅ 模型加载完成！")
    # ------------------- 3. 相机硬件配置 (基于您提供的脚本) -------------------
    # 使用 V4L2 后端打开设备号 0
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("❌ 无法打开 /dev/video0，请检查权限或设备号！")
    # 配置 MJPEG 编码以支持高帧率
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    # 设置 1080P 分辨率
    target_width = 1920
    target_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # 打印实际生效的参数以供核对
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📷 相机已打开: /dev/video6 | 分辨率: {int(actual_width)}x{int(actual_height)} | 格式: MJPEG")
    # ------------------- 4. 相机内参配置 (至关重要) -------------------
    # ⚠️【警告】⚠️
    # 分辨率改为 1920x1080 后，内参矩阵必须更新！
    # 下面提供的是一个【估算值】（假设光心在图像中心）。
    # 为了保证 6D 位姿的准确性，请务必使用标定板对该分辨率进行标定，并填入真实值。
    cam_K = np.array([
        [1500.0,    0.0, 960.0],  # fx,  0, cx (cx 约为 width/2)
        [   0.0, 1500.0, 540.0],  #  0, fy, cy (cy 约为 height/2)
        [   0.0,    0.0,   1.0],
    ])
    print("🚀 开始实时推理，按 'q' 键退出...")
    while True:
        t_start = time.time()
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 取帧失败，重试中...")
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # ------------------- 5. 核心推理流程 -------------------
        # HccePose 预测 (输入: 内参, 图像, 物体ID)
        # conf=0.5: YOLO检测置信度，根据实际环境光照调整
        results_dict = Tester_item.predict(cam_K, frame, obj_ids,
                                         conf=0.15, confidence_threshold=0.15)
        # ------------------- 6. 结果可视化 -------------------
        # 计算处理耗时与 FPS
        process_time = results_dict.get('time', 0.033)
        fps_hccepose = 1.0 / process_time
        # 获取包含坐标轴和边框的渲染图
        # tester.py 中 predict 返回的 'show_6D_vis2' 包含了最终的可视化结果
        vis_img = results_dict['show_6D_vis2']
        # 确保数据类型正确 (防报错)
        vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
        # 橙色变蓝色是因为模型输出是 RGB，而 OpenCV 显示需要 BGR
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        # 在画面左上角绘制 FPS
        # cv2.putText(vis_img, f"Inference FPS: {fps_hccepose:.2f}", (30, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        # 显示画面
        cv2.imshow("HccePose Real-Time (1080P) - Press 'q' to quit", vis_img)
        # 按键检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    # ------------------- 7. 资源释放 -------------------
    cap.release()
    cv2.destroyAllWindows()
    print("👋 程序已退出")

🚀 开始实时推理 (按 'q' 退出, 's' 保存截图)...

0: 384x640 (no detections), 146.2ms
Speed: 0.0ms preprocess, 146.2ms inference, 12.2ms postprocess per image at shape (1, 3, 384, 640)
👋 程序已退出
Traceback (most recent call last):
  File "/home/user/WorkSpace/HCCEPose/HCCEPose_Trocar/s4_p4_realtime_camera_demo.py", line 154, in <module>
    run_realtime_inference()
  File "/home/user/WorkSpace/HCCEPose/HCCEPose_Trocar/s4_p4_realtime_camera_demo.py", line 115, in run_realtime_inference
    vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
  File "/home/user/anaconda3/envs/hcce/lib/python3.10/site-packages/numpy/core/fromnumeric.py", line 2169, in clip
    return _wrapfunc(a, 'clip', a_min, a_max, out=out, **kwargs)
  File "/home/user/anaconda3/envs/hcce/lib/python3.10/site-packages/numpy/core/fromnumeric.py", line 56, in _wrapfunc
    return _wrapit(obj, method, *args, **kwds)
  File "/home/user/anaconda3/envs/hcce/lib/python3.10/site-packages/numpy/core/fromnumeric.py", line 45, in _wrapit
    result = getattr(asarray(obj), method)(*args, **kwds)
  File "/home/user/anaconda3/envs/hcce/lib/python3.10/site-packages/numpy/core/_methods.py", line 99, in _clip
    return um.clip(a, min, max, out=out, **kwargs)
TypeError: '>=' not supported between instances of 'NoneType' and 'int'