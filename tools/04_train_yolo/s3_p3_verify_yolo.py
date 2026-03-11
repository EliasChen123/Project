import os
import cv2
import sys
from ultralytics import YOLO

def batch_inference(model_path, input_dir, output_dir, conf_thres=0.25):
    """
    加载模型，对 input_dir 下的所有图片进行推理，并保存到 output_dir
    """
    # 1. 检查模型和输入目录
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件 -> {model_path}")
        return
    if not os.path.exists(input_dir):
        print(f"❌ 错误：找不到测试图片文件夹 -> {input_dir}")
        return

    # 2. 自动创建结果文件夹 (如果不存在)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 已创建结果文件夹: {output_dir}")

    print(f"🚀 正在加载模型: {model_path} ...")
    
    # 3. 加载模型 (只加载一次)
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 获取文件夹内所有图片文件
    # 支持常见的图片格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    # 按名称排序，保证处理顺序 (001.jpg, 002.jpg...)
    image_files.sort()

    if not image_files:
        print(f"⚠️ 警告: 在 {input_dir} 中没有找到图片文件。")
        return

    print(f"📥 准备处理 {len(image_files)} 张图片...")
    print("-" * 50)

    # 5. 循环处理每一张图片
    for i, file_name in enumerate(image_files):
        image_path = os.path.join(input_dir, file_name)
        save_path = os.path.join(output_dir, file_name) # 结果保持原文件名保存到新文件夹
        
        print(f"[{i+1}/{len(image_files)}] 正在处理: {file_name} ...", end="\r")

        # 执行推理
        results = model.predict(source=image_path, conf=conf_thres, save=False, show=False, verbose=False)

        # 处理结果并保存
        for result in results:
            # 绘制结果图
            plotted_image = result.plot()
            
            # 保存到 result 文件夹
            cv2.imwrite(save_path, plotted_image)
            
            # (可选) 打印每张图的详细检测信息，如果不想刷屏可以注释掉下面几行
            # boxes = result.boxes
            # if len(boxes) > 0:
            #     print(f"\n   检测到 {len(boxes)} 个目标:")
            #     for box in boxes:
            #         cls_name = model.names[int(box.cls[0])]
            #         conf = float(box.conf[0])
            #         print(f"   - {cls_name}: {conf:.2f}")

    print("\n" + "=" * 50)
    print(f"✅ 所有处理完成！")
    print(f"📂 结果已保存在: {output_dir}")

if __name__ == '__main__':
    # ================= 配置区域 =================
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 模型路径 (保持不变)
    my_model_path = os.path.join(current_dir, 'trocar/yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt')

    # 2. 设置输入文件夹 (存放 001.jpg 等原图)
    test_dir = os.path.join(current_dir, 'test/116') 

    # 3. 设置输出文件夹 (存放推理结果)
    result_dir = os.path.join(current_dir, 'result/116')
    
    # ===========================================

    batch_inference(my_model_path, test_dir, result_dir)