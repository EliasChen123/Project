import blenderproc as bproc
import numpy as np
import sys
import os
import shutil
import glob
import gc
import mathutils
from bpy_extras.object_utils import world_to_camera_view
import bpy
# 生成放大版近距离trocar, 标注原点在trocar中心

# ========================================
# >>> 测试步骤 1: 检查对象层级与命名
# ========================================
# Imported 9 objects
# Selected 7 of the loaded objects by type
# 当前场景中包含 7 个对象。开始遍历检查：
#   - 对象名: Trocar_outer         | 父级: Trocar          | 类型: MESH
#   - 对象名: Trocar_inner         | 父级: Trocar          | 类型: MESH
#   - 对象名: Trocar               | 父级: None            | 类型: EMPTY
#   - 对象名: Eye_Shadow           | 父级: Eye             | 类型: MESH
#   - 对象名: Eye_Sclera           | 父级: Eye             | 类型: MESH
#   - 对象名: Eye_Iris             | 父级: Eye             | 类型: MESH
#   - 对象名: Eye                  | 父级: None            | 类型: EMPTY
# ==========================================
# 0. 接收参数 (由管理器传入)
# ==========================================
# 定义小批次大小：每 10 张写一次盘，内存占用仅约 700MB，非常安全
MINI_BATCH_SIZE = 100

MIN_FACTOR = 2.0

MAX_FACTOR = 4.0

bproc.init()

# 默认参数 (用于测试)
start_index = 0
num_images = 10

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ_DIR = os.path.join(ROOT_DIR, "output", "trocar")
MODELS_DIR = os.path.join(OBJ_DIR, "models")
BLEND_PATH = os.path.join(MODELS_DIR, "trocar.blend")

TEXTURE_DIR = os.path.join(MODELS_DIR, "Textures")
# output_dir = "temp"
output_dir = os.path.join(ROOT_DIR, "eye_trocar_v5", "bop_dataset_raw") #测试用

# 解析参数: s01_worker.py [start_index] [num_images] [output_dir]

if len(sys.argv) >= 4:
    start_index = int(sys.argv[-3])
    num_images = int(sys.argv[-2])
    output_dir = sys.argv[-1]

print(f">>> Worker 启动: ID {start_index} -> {start_index + num_images}, 输出: {output_dir}")

# 设置随机种子 (至关重要)
np.random.seed(start_index)

# ==========================================
# 1. 极速渲染配置
# ==========================================
# 5090 性能极强，单进程下可以适当提高一点质量，或者保持极速
bproc.renderer.set_max_amount_of_samples(32) 
bproc.renderer.set_noise_threshold(0.01)
bproc.renderer.set_denoiser("OPTIX") # 5090 必选

# 限制光线反弹，减少显存占用
bproc.renderer.set_light_bounces(
    diffuse_bounces=2,
    glossy_bounces=3,
    transmission_bounces=8, 
    transparent_max_bounces=8,
    volume_bounces=0
)

bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type="OPTIX")
# 单进程批处理可以开启 Persistent Data 加速，因为我们每1000帧会重启进程
bpy.context.scene.render.use_persistent_data = True
bproc.renderer.set_output_format(enable_transparency=True)

# 相机配置
WIDTH, HEIGHT = 1920, 1080
ASPECT_RATIO = WIDTH / HEIGHT
bproc.camera.set_resolution(WIDTH, HEIGHT)
bproc.camera.set_intrinsics_from_blender_params(lens=60, lens_unit='MILLIMETERS')
# 裁剪平面设置
for cam in bpy.data.cameras:
    cam.clip_start = 0.0001
    cam.clip_end = 100.0
# ==========================================
# 2. 资源加载 (保持原逻辑)
# ==========================================


loaded_objects = bproc.loader.load_blend(BLEND_PATH)
iris_obj = bproc.filter.one_by_attr(loaded_objects, "name", "Eye_Iris")
sclera_obj = bproc.filter.one_by_attr(loaded_objects, "name", "Eye_Sclera")
eye_names = ["Eye_Sclera", "Eye_Shadow", "Eye_Iris"]
eye_parts = [obj for obj in loaded_objects if any(n in obj.get_name() for n in eye_names)]
# eye_parts = [obj for obj in loaded_objects if obj.blender_obj.type == 'MESH' and "Eye" in obj.get_name()]

trocar_outer = bproc.filter.one_by_attr(loaded_objects, "name", "Trocar_outer")
trocar_inner = bproc.filter.one_by_attr(loaded_objects, "name", "Trocar_inner")

# 我们将 trocar_outer 放在列表第一位，以尽量保留其材质在 Slot 0 的位置。
trocar_outer.join_with_other_objects([trocar_inner])
trocar_outer.set_name("Trocar_Full")  # 重命名方便调试

bpy.ops.object.select_all(action='DESELECT')
trocar_outer.blender_obj.select_set(True)
bpy.context.view_layer.objects.active = trocar_outer.blender_obj
# type='ORIGIN_GEOMETRY', center='BOUNDS' 等同于 (min+max)/2
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

for part in eye_parts: part.set_cp("category_id", 0)
trocar_outer.set_cp("category_id", 1)


target_obj_for_gt = trocar_outer
material_target_obj = trocar_outer

# 材质加载
texture_files = glob.glob(os.path.join(TEXTURE_DIR, "*_D.tga"))
loaded_images = []
for tex_path in texture_files:
    try:
        img_name = os.path.basename(tex_path)
        if img_name not in bpy.data.images:
            img = bpy.data.images.load(tex_path)
        else:
            img = bpy.data.images[img_name]
        img.use_fake_user = True
        loaded_images.append(img)
    except: pass

def get_iris_image_node(obj):
    if not obj.get_materials(): return None
    mat = obj.get_materials()[0].blender_obj
    if not mat.use_nodes: return None
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE': return node
    return None
iris_image_node = get_iris_image_node(iris_obj)

def randomize_iris_texture():
    if iris_image_node and loaded_images:
        iris_image_node.image = np.random.choice(loaded_images)

# B. Trocar 材质 (增强版：支持医疗蓝、橙及材质区分)
def randomize_trocar_outer_material():
    # 获取合并后对象的所有材质
    materials = material_target_obj.get_materials()
    target_mat = materials[0]
    # (可选) 调试打印，确保改对人了
    # print(f"DEBUG: 正在随机化材质 -> {target_mat.get_name()}")
    # 随机决定颜色主题
    # 概率分配：30% 银色金属, 35% 医疗蓝塑料, 35% 医疗橙塑料
    rand_val = np.random.rand()
    
    if rand_val < 0.3:
        # --- 方案 A: 银色金属 (Silver Metal) ---
        # 颜色：从中灰到亮白
        val = np.random.uniform(0.5, 0.95)
        color = [val, val, val, 1.0]
        
        # 材质属性：高金属度，低粗糙度（反光强）
        metallic = np.random.uniform(0.8, 1.0)
        roughness = np.random.uniform(0.1, 0.4)
        
    elif rand_val < 0.65:
        # --- 方案 B: 医疗蓝 (Medical Blue) ---
        # 基础色 (RGBA): 稍微偏青的深蓝色
        # R: 0.0~0.1, G: 0.3~0.6, B: 0.7~1.0
        r = np.random.uniform(0.0, 0.1)
        g = np.random.uniform(0.3, 0.6)
        b = np.random.uniform(0.7, 1.0)
        color = [r, g, b, 1.0]
        
        # 材质属性：低金属度（塑料），中等粗糙度
        metallic = np.random.uniform(0.0, 0.2) 
        roughness = np.random.uniform(0.2, 0.6)
        
    else:
        # --- 方案 C: 医疗橙 (Medical Orange) ---
        # 基础色 (RGBA): 高饱和度的橙色
        # R: 0.9~1.0, G: 0.4~0.6, B: 0.0~0.1
        r = np.random.uniform(0.8, 1.0)
        g = np.random.uniform(0.3, 0.55) # 稍微波动，可能是深橙或浅橙
        b = np.random.uniform(0.0, 0.1)
        color = [r, g, b, 1.0]
        
        # 材质属性：低金属度（塑料）
        metallic = np.random.uniform(0.0, 0.2)
        roughness = np.random.uniform(0.2, 0.6)

    # 应用参数
    target_mat.set_principled_shader_value("Base Color", color)
    target_mat.set_principled_shader_value("Roughness", roughness)
    target_mat.set_principled_shader_value("Metallic", metallic)

# ==========================================
# 4. 相机采样函数 (核心改进)
# ==========================================
def check_point_is_visible(matrix_world, target_point_3d, margin=0.02):
    """
    检测 3D 点是否在相机视野范围内。
    :param matrix_world: 4x4 相机位姿矩阵 (numpy array or list)
    :param target_point_3d: 目标物体中心点坐标 (numpy array)
    :param margin: 边缘留白 (0.0~0.5), 0.05 表示需在画面 5%~95% 范围内
    :return: bool
    """
    scene = bpy.context.scene
    cam_obj = scene.camera
    
    # 1. 临时应用位姿到 Blender 相机对象 (用于计算投影)
    # 这是一个极快的属性赋值，不涉及场景重绘，非常高效
    cam_obj.matrix_world = mathutils.Matrix(matrix_world)
    
    # 2. 将 3D 世界坐标投影到 2D 归一化相机坐标 (0.0 ~ 1.0)
    # co_2d.x/y 是画面坐标, co_2d.z 是深度距离
    co_2d = world_to_camera_view(scene, cam_obj, mathutils.Vector(target_point_3d))
    
    # 3. 检查深度 (z > 0 表示在相机前方)
    if co_2d.z <= 0:
        return False
        
    # 4. 检查画面范围 (是否在 [0+margin, 1-margin] 之间)
    min_val = margin
    max_val = 1.0 - margin
    
    is_in_x = min_val <= co_2d.x <= max_val
    is_in_y = min_val <= co_2d.y <= max_val
    
    return is_in_x and is_in_y

def sample_valid_camera_pose(max_tries=100):
    """
    [改进点 1] 确保物体一定在画面内的几何计算
    [改进点 2] 增加几何投影验证模块 (Double Check)
    """
    global_up = np.array([0, 0, 1])
    
    # 100mm 镜头 @ 15cm 距离，半视野宽度约为 2.6cm
    # 为了保证 Trocar (半径~1.5mm) 完整在画面内，偏移量不应超过 1.5cm
    SAFE_LIMIT_OFFSET = 0.012  # 1.5cm (原代码曾设为 0.035 导致出画)

    for _ in range(max_tries):
        # 1. 采样位置
        location = bproc.sampler.shell(
            center=eye_center, 
            radius_min=cam_radius_min, 
            radius_max=cam_radius_max,
            elevation_min=30,
            elevation_max=90
        )
        
        # 2. 正面检查
        cam_vec = location - eye_center
        cam_vec /= np.linalg.norm(cam_vec)
        ctheta = np.cos(np.radians(80)) # 80度视锥角的一半
        if np.dot(target_vec, cam_vec) > ctheta:
            # 3. 计算 "Look At" 点偏移 (Jitter)
            view_dir = location - trocar_pos
            view_dir /= np.linalg.norm(view_dir)
            
            camera_right = np.cross(global_up, view_dir)
            if np.linalg.norm(camera_right) < 1e-6: camera_right = np.array([1,0,0])
            camera_right /= np.linalg.norm(camera_right)
            
            camera_up = np.cross(view_dir, camera_right)
            camera_up /= np.linalg.norm(camera_up)
            
            # [修正] 限制随机偏移范围
            rand_w = np.random.uniform(-1, 1) * SAFE_LIMIT_OFFSET * ASPECT_RATIO
            rand_h = np.random.uniform(-1, 1) * SAFE_LIMIT_OFFSET
            
            jitter = (camera_right * rand_w) + (camera_up * rand_h)
            noisy_poi = trocar_pos + jitter
            
            # 4. 生成矩阵
            rotation_matrix = bproc.camera.rotation_from_forward_vec(noisy_poi - location)
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            if check_point_is_visible(cam2world_matrix, trocar_pos, margin=0.05):
                # 验证通过，返回矩阵
                return cam2world_matrix
            else:
                # 验证失败 (可能是 jitter 太大或者角度太刁钻导致出画)，继续下一次循环
                continue
    return None

# 场景辅助对象
# min_vec = np.min([np.min(obj.get_bound_box(), axis=0) for obj in eye_parts], axis=0)
# max_vec = np.max([np.max(obj.get_bound_box(), axis=0) for obj in eye_parts], axis=0)
min_vec = np.min(sclera_obj.get_bound_box(), axis=0)
max_vec = np.max(sclera_obj.get_bound_box(), axis=0)

eye_center = (min_vec + max_vec) / 2
obj_size = np.linalg.norm(max_vec - min_vec)
cam_radius_min = obj_size * MIN_FACTOR
cam_radius_max = obj_size * MAX_FACTOR


trocar_bbox = target_obj_for_gt.get_bound_box()
trocar_pos = np.mean(trocar_bbox, axis=0)

target_vec = trocar_pos - eye_center
target_vec /= np.linalg.norm(target_vec)

light = bproc.types.Light()
light.set_type("POINT")
fill_light = bproc.types.Light()
fill_light.set_type("POINT")

bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id"])


# ==========================================
# 3. 循环生成
# ==========================================
# 在内存中缓存数据，最后一次性写入，避免反复 IO
chunk_colors, chunk_depths, chunk_poses = [], [], []

if os.path.exists(output_dir):
    try:
        shutil.rmtree(output_dir)
    except:
        pass

print(f">>> 开始生成任务，目标: {output_dir}")

for i in range(num_images):
    current_global_frame_id = start_index + i
    print(f"生成第{current_global_frame_id}张图片")

    bproc.utility.reset_keyframes()

    # 场景随机化
    light.set_location(bproc.sampler.shell(center=trocar_pos, radius_min=2, radius_max=5, elevation_min=10, elevation_max=80))
    light.set_energy(np.random.uniform(200, 800))
    fill_light.set_location(bproc.sampler.shell(center=trocar_pos, radius_min=2, radius_max=5, elevation_min=-20, elevation_max=80))
    fill_light.set_energy(np.random.uniform(50, 150))
    
    randomize_iris_texture()
    randomize_trocar_outer_material()
        
    
    # B. 获取相机位姿
    cam2world = sample_valid_camera_pose()
    if cam2world is None:
        print(f"⚠️ 第 {i} 帧采样失败，跳过。")
        continue

    # 添加当前这一帧的 Pose
    bproc.camera.add_camera_pose(cam2world, frame=0)
    bproc.camera.add_depth_of_field(
        focal_point_obj=target_obj_for_gt, 
        fstop_value=32.0,      # F32 是非常小的光圈，能保证整个物体都在景深范围内（全清晰）
    )
    
    # 渲染
    data = bproc.renderer.render()
    chunk_colors.append(data["colors"][0])
    chunk_depths.append(data["depth"][0])
    chunk_poses.append(cam2world)

    # --- D. 检查是否达到小批次阈值 ---
    # 每凑够 50 张，或者到了最后一张，就写入一次
    current_count = len(chunk_colors)
    if current_count >= MINI_BATCH_SIZE or (i == num_images - 1):
        if current_count > 0:
            print(f"    [IO 操作] 正在写入 {current_count} 张 (进度: {i+1}/{num_images})...")
            
            bproc.utility.reset_keyframes()
            batch_start_id = current_global_frame_id - current_count + 1
            # 恢复 Pose 给 Writer
            for idx, pose in enumerate(chunk_poses):
                bproc.camera.add_camera_pose(pose, frame=idx)

            bproc.writer.write_bop(
                output_dir=output_dir, 
                target_objects=[target_obj_for_gt], 
                depths=chunk_depths,
                colors=chunk_colors,
                append_to_existing_output=True,
                depth_scale=0.1, 
                save_world2cam=True,
                # calc_mask_info_coco=True,
                # ignore_dist_thres=100.0 
            )
            # 核心：写入后立即清空 Python 列表，释放 700MB 内存

            chunk_colors = []
            chunk_depths = []
            chunk_poses = []
            bproc.utility.reset_keyframes()
            # 强制垃圾回收
            gc.collect()
            
# ==========================================
# 4. 结束清理
# ==========================================
bproc.clean_up()
print(f"✅ Worker 任务完成: {output_dir}")