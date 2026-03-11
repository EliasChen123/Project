import blenderproc as bproc
import bpy
import os

bproc.init()

# 1. 加载模型 (与 s01_worker_v2.py 路径保持一致)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "eye_trocar_v5", "models")
BLEND_PATH = os.path.join(MODELS_DIR, "trocar_v3.blend")
RAW_PLY_PATH = os.path.join(MODELS_DIR, "Trocar_merged_raw.ply")

print(f">>> Loading: {BLEND_PATH}")
loaded_objects = bproc.loader.load_blend(BLEND_PATH)

# 2. 获取对象
trocar_outer = bproc.filter.one_by_attr(loaded_objects, "name", "Trocar_outer")
trocar_inner = bproc.filter.one_by_attr(loaded_objects, "name", "Trocar_inner")

# 3. 执行合并 + 强制居中
print(">>> Merging objects...")
trocar_outer.join_with_other_objects([trocar_inner])
trocar_outer.set_name("obj_000001")

print(">>> Centering Origin (Bounds Center)...")
bpy.ops.object.select_all(action='DESELECT')
trocar_outer.blender_obj.select_set(True)
bpy.context.view_layer.objects.active = trocar_outer.blender_obj
# 确保原点位于几何中心
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

#  归零位置
# 为了确保导出的模型顶点坐标是相对于几何中心的，
# 我们将物体放置在世界坐标系的原点 (0,0,0)
trocar_outer.set_location([0, 0, 0])
trocar_outer.set_rotation_euler([0, 0, 0])

# 4. 导出为 PLY
print(f">>> Exporting to: {RAW_PLY_PATH}")
# Blender 4.0+ 使用 wm.ply_export
bpy.ops.wm.ply_export(
    filepath=RAW_PLY_PATH,
    export_selected_objects=True,
    apply_modifiers=True,
    export_normals=True,
    export_uv=True,
    # 注意：根据 BOP 标准，通常模型单位是 mm。
    # 如果你在 Blender 里是米(m)，导出时需要 scale=1000
    global_scale=1.0, 
    forward_axis='Y', 
    up_axis='Z'
)

print("✅ Raw merged PLY exported.")
print("👉 下一步：请运行官方 s1_p1 脚本处理此文件，生成最终的 obj_000001.ply")