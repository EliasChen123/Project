import subprocess
import os
import time
import sys

# 配置
TOTAL_IMAGES = 5000
BATCH_SIZE = 1000  # 每次重启进程前生成多少张
start_index = 5000

GENERATE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(GENERATE_DIR))
WORKER_SCRIPT = os.path.join(GENERATE_DIR, "s01_worker_trocar.py")
BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "trocar", "bop_dataset_raw")
BPROC_SOURCE_PATH = os.path.join(ROOT_DIR, "blenderproc")

# 确保输出目录存在
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)

# 主循环
for batch_idx in range(start_index, start_index + TOTAL_IMAGES, BATCH_SIZE):
    start_time = time.time()
    # 1. 定义当前批次的输出目录
    current_output_dir = os.path.join(BASE_OUTPUT_DIR, f"chunk_{batch_idx // BATCH_SIZE:03d}")
    print(f"========== 启动批次 {batch_idx} - {batch_idx + BATCH_SIZE} ==========")
    print(f"输出目录: {current_output_dir}")
    
    # 2. 调用子进程 (BlenderProc)
    cmd = [
        "blenderproc", "run", WORKER_SCRIPT, 
        str(batch_idx), 
        str(BATCH_SIZE), 
        current_output_dir
    ]
    # =======================================================
    # >>> 核心修改：构建专属环境变量 <<<
    # =======================================================
    # 1. 复制当前系统的所有环境变量
    worker_env = os.environ.copy()
    
    # 2. 仅在当前这个子进程的配置中，加入 PYTHONPATH
    # 获取原有的 PYTHONPATH (如果有的话)，然后把我们的路径拼接到最前面
    original_pythonpath = worker_env.get("PYTHONPATH", "")
    worker_env["PYTHONPATH"] = BPROC_SOURCE_PATH + os.pathsep + original_pythonpath
    
    print(f"   [调试] 子进程 PYTHONPATH 已注入: {BPROC_SOURCE_PATH}")
    try:
        # 使用 subprocess.run 等待进程完全结束
        result = subprocess.run(cmd, env=worker_env, check=False)
        if result.returncode != 0:
            print(f"❌ 批次 {batch_idx} 失败！返回码: {result.returncode}")
            # 可以选择 break 或者 continue，取决于是否允许部分失败
            break
            
        duration = time.time() - start_time
        print(f"✅ 批次完成，耗时: {duration:.1f} 秒")
        print(f"显存已完全释放，准备下一批次...\n")
        
    except FileNotFoundError:
        print("❌ 错误: 找不到 'blenderproc' 命令。请确保 conda 环境已激活。")
        break
    except KeyboardInterrupt:
        print("\n🛑 用户手动停止任务。")
        break

print("所有任务结束。")