import matplotlib.pyplot as plt
import re
import argparse
import os

TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


TARGET_DIR = os.path.join(TOOLS_DIR, "05_bf_train")

OUTPUT_DIR = os.path.join(TOOLS_DIR, "loss_pic")

def parse_log(log_file):
    steps = []
    losses = {'total': [], 'front': [], 'back': [], 'mask': [], 'pose': []}
    acc_data = [] # (step, accuracy)
    current_acc = None
    
    # 正则表达式适配您的日志格式
    loss_pattern = re.compile(r"iteration_step:\s*(\d+).*?loss_front:\s*([-\d\.eE\+]+).*?loss_back:\s*([-\d\.eE\+]+).*?loss_mask:\s*([-\d\.eE\+]+).*?loss_pose:\s*([-\d\.eE\+]+).*?total_loss:\s*([-\d\.eE\+]+)")
    acc_pattern = re.compile(r"max acc:\s*([\d\.]+)")

    print(f"🔍 Loading log file: {log_file}")
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # 1. 提取精度 (Accuracy)
                acc_match = acc_pattern.search(line)
                if acc_match:
                    try:
                        current_acc = float(acc_match.group(1))
                    except ValueError:
                        pass
                    continue

                # 2. 提取损失 (Loss)
                loss_match = loss_pattern.search(line)
                if loss_match:
                    step = int(loss_match.group(1))
                    l_front = float(loss_match.group(2))
                    l_back = float(loss_match.group(3))
                    l_mask = float(loss_match.group(4))
                    l_pose = float(loss_match.group(5))
                    l_total = float(loss_match.group(6))

                    steps.append(step)
                    losses['front'].append(l_front)
                    losses['back'].append(l_back)
                    losses['mask'].append(l_mask)
                    losses['pose'].append(l_pose)
                    losses['total'].append(l_total)
                    
                    # 关联精度与当前步数
                    if current_acc is not None:
                        acc_data.append((step, current_acc))
                        current_acc = None
    except FileNotFoundError:
        print(f"❌ Error: File {log_file} not found.")
        return None, None, None

    return steps, losses, acc_data

def save_single_plot(x, y, title, ylabel, filename, color, output_dir):
    """辅助函数：绘制并保存单张图表"""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=title, color=color, linewidth=2)
    
    # 标注最小值（对于Loss曲线很有用）
    if 'Loss' in title and len(y) > 0:
        min_val = min(y)
        min_idx = y.index(min_val)
        plt.annotate(f'Min: {min_val:.4f}', 
                     xy=(x[min_idx], min_val), 
                     xytext=(x[min_idx], min_val + (max(y)-min_val)*0.05),
                     arrowprops=dict(facecolor=color, shrink=0.05),
                     fontsize=10)

    plt.title(f'{title} Convergence')
    plt.xlabel('Iteration Steps')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.ylim(-0.05, 10000)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close() # 关闭画布释放内存
    print(f"   💾 Saved: {save_path}")

def plot_all_curves(steps, losses, acc_data):
    if not steps:
        print("⚠️ No valid data found in log file.")
        return

    # 1. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created output directory: ./{OUTPUT_DIR}/")
    else:
        print(f"📁 Saving images to: ./{OUTPUT_DIR}/")

    print("-" * 30)

    # 2. 绘制各个 Loss 曲线
    # Total Loss (紫色)
    save_single_plot(steps, losses['total'], 'Total Loss', 'Loss Value', 
                     'loss_0_total.png', 'cyan', OUTPUT_DIR)
    
    # Front Loss (蓝色)
    save_single_plot(steps, losses['front'], 'Front Surface Loss', 'L1 Loss', 
                     'loss_1_front.png', 'royalblue', OUTPUT_DIR)
    
    # Back Loss (绿色)
    save_single_plot(steps, losses['back'], 'Back Surface Loss', 'L1 Loss', 
                     'loss_2_back.png', 'forestgreen', OUTPUT_DIR)
    
    # Mask Loss (橙色)
    save_single_plot(steps, losses['mask'], 'Mask Segmentation Loss', 'BCE Loss', 
                     'loss_3_mask.png', 'darkorange', OUTPUT_DIR)

    # Pose Loss (红色)
    save_single_plot(steps, losses['pose'], 'Pose Estimation Loss', 'L2 Loss', 
                     'loss_4_pose.png', 'crimson', OUTPUT_DIR)

    # 3. 绘制 Accuracy 曲线 (红色)
    if acc_data:
        acc_steps, acc_values = zip(*acc_data)
        plt.figure(figsize=(10, 6))
        plt.plot(acc_steps, acc_values, label='ADD-S Accuracy', color='crimson', linewidth=2, marker='o', markersize=4)
        
        # 标注最大精度
        max_acc = max(acc_values)
        max_idx = acc_values.index(max_acc)
        plt.annotate(f'Max: {max_acc:.4f}', 
                     xy=(acc_steps[max_idx], max_acc), 
                     xytext=(acc_steps[max_idx], max_acc - 0.1 if max_acc > 0.9 else max_acc + 0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05))

        plt.title('Validation Accuracy (ADD-S)')
        plt.xlabel('Iteration Steps')
        plt.ylabel('Accuracy (0-1)')
        plt.ylim(-0.05, 1.05) # 固定Y轴范围方便观察
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, 'metric_accuracy.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   💾 Saved: {save_path}")
    else:
        print("⚠️ No accuracy data found (Test not triggered yet).")

    print("-" * 30)
    print(f"✅ All plots generated in '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    # 自动寻找最新的log文件
    log_path = os.path.join(TARGET_DIR, "s4_p2_train_bf_pbr.log")
    if not os.path.exists(log_path):
        logs = [f for f in os.listdir('.') if f.endswith('.log')]
        if logs:
            logs.sort(key=os.path.getmtime, reverse=True)
            default_log = logs[0]
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--log', type=str, default=default_log, help='Path to log file')
    # parser.add_argument('--out', type=str, default='pic', help='Output directory for images')
    # args = parser.parse_args()

    if os.path.exists(log_path):
        steps, losses, acc_data = parse_log(log_path)
        plot_all_curves(steps, losses, acc_data)
    else:
        print(f"❌ Error: Log file '{log_path}' not found.")