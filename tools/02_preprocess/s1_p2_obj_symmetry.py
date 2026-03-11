# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

# KASAL is an interactive rotational symmetry analysis software that supports eight types of rotational symmetries.
# KASAL 是一个交互式旋转对称分析软件，支持 8 种旋转对称类型。

from kasal.app.polyscope_app import app # pip install kasal-6d
import os
import sys

if __name__ == '__main__':

    # Set the folder path; KASAL will automatically search for all PLY or OBJ files in the folder
    # 设置文件夹路径，KASAL 会自动查找文件夹下所有 PLY 或 OBJ 文件
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TARGET_DIR = os.path.join(ROOT_DIR, "output", "trocar")
    MESH_PATH = os.path.join(TARGET_DIR, "models")

    # Launch the graphical user interface (GUI) of KASAL
    # 启动 KASAL 的图形界面
    app(MESH_PATH)

    pass
    