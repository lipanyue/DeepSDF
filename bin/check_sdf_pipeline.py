import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial import cKDTree 
import trimesh 
import sys 
import os 

# 添加Open3D支持
try:
    import open3d as o3d
except ImportError:
    print("Open3D not available, skipping Open3D-specific checks")

# 从原代码中添加KDTree支持（适配DeepSDF）
try:
    # 尝试导入DeepSDF的KDTree实现
    from deep_sdf.kdtree import KdTree_surf
except ImportError:
    # 如果失败，使用scipy的KDTree
    from scipy.spatial import KDTree
    print("Using scipy's KDTree instead of DeepSDF's implementation")


def check_sdf_generation_pipeline():
    """ 
    系统检查SDF生成管道的每一步 
    """ 
    print("="*70) 
    print("SDF生成管道检查") 
    print("="*70) 
    
    # 1. 检查原始数据 
    print("\n1. 检查原始数据源...") 
    check_data_source() 
    
    # 2. 检查SDF计算逻辑 
    print("\n2. 检查SDF计算逻辑...") 
    check_sdf_computation_logic() 
    
    # 3. 检查采样策略 
    print("\n3. 检查采样策略...") 
    check_sampling_strategy() 
    
    # 4. 检查数据格式 
    print("\n4. 检查数据格式...") 
    check_data_format() 
    
    # 5. 生成测试用例 
    print("\n5. 运行测试用例...") 
    run_test_cases() 


def check_data_source():
    """检查数据源（网格文件）"""
    print("  常见数据源格式：")
    print("  - .obj, .ply, .stl 网格文件")
    print("  - .off 文件")
    print("  - .glb, .glTF 文件")
    
    # 查找可能的网格文件
    mesh_extensions = ['.obj', '.ply', '.stl', '.off', '.glb', '.gltf']
    found_files = []
    
    for ext in mesh_extensions:
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith(ext):
                    found_files.append(os.path.join(root, file))
    
    if found_files:
        print(f"\n  找到 {len(found_files)} 个可能的网格文件：")
        for f in found_files[:5]:  # 只显示前5个
            print(f"    - {f}")
    else:
        print("  未找到网格文件")


def check_sdf_computation_logic():
    """检查SDF计算逻辑"""
    print("  SDF计算的常见方法：")
    print("  1. 从网格直接计算SDF")
    print("  2. 使用预计算的体素网格")
    print("  3. 使用深度学习方法")
    
    print("\n  关键检查点：")
    print("  - 符号是否正确（内部为负，外部为正）")
    print("  - 距离是否准确（欧几里得距离）")
    print("  - 是否进行了归一化")
    print("  - 边界处理是否正确")
    
    # 添加DeepSDF特定的检查
    print("\n  DeepSDF特定检查：")
    print("  - 是否使用了可见表面采样")
    print("  - 是否使用了法线投票确定符号")
    print("  - 采样方差是否合理")


def check_sampling_strategy():
    """检查采样策略"""
    print("  常见的SDF采样策略：")
    print("  1. 表面附近密集采样（用于捕捉细节）")
    print("  2. 均匀空间采样（用于覆盖整个空间）")
    print("  3. 重要性采样（基于曲率等）")
    
    print("\n  关键检查点：")
    print("  - 表面附近采样密度是否足够")
    print("  - 远处是否有足够样本")
    print("  - 正负样本比例是否平衡")
    print("  - 采样点是否在归一化边界内")
    
    # 添加DeepSDF特定采样策略
    print("\n  DeepSDF采样策略：")
    print("  - 通常94%的样本在表面附近")
    print("  - 6%的样本均匀分布在空间中")
    print("  - 使用正态分布进行表面附近扰动")


def check_data_format():
    """检查数据格式"""
    print("  DeepSDF标准格式：")
    print("  - 输入：归一化到[-1, 1]范围内的点")
    print("  - 输出：有符号距离值")
    print("  - 正样本（SDF > 0）：表面外的点")
    print("  - 负样本（SDF < 0）：表面内的点")
    
    print("\n  你的数据格式：")
    try:
        data = np.load("output.npz")
        print(f"  - 文件中的键：{list(data.keys())}")
        if 'pos' in data:
            print(f"  - pos形状：{data['pos'].shape}")
        if 'neg' in data:
            print(f"  - neg形状：{data['neg'].shape}")
    except Exception as e:
        print(f"  - 无法读取output.npz: {e}")


def run_test_cases():
    """运行测试用例验证SDF生成"""
    print("\n  运行测试用例...")
    
    # 测试1：简单球体的解析SDF
    test_sphere_sdf()
    
    # 测试2：立方体的近似SDF
    test_cube_sdf()
    
    # 测试3：加载并验证你的SDF数据
    test_your_sdf_data()


def test_sphere_sdf():
    """测试球体SDF生成"""
    print("\n  测试1：球体SDF（解析解）")
    
    # 生成球体SDF
    num_samples = 10000
    radius = 0.5
    
    # 在[-1, 1]范围内采样
    points = np.random.uniform(-1, 1, (num_samples, 3))
    
    # 计算球体SDF：距离球心的距离减去半径
    distances = np.linalg.norm(points, axis=1) - radius
    
    # 分离正负样本
    pos_mask = distances > 0
    neg_mask = distances <= 0
    
    pos_samples = np.column_stack([points[pos_mask], distances[pos_mask]])
    neg_samples = np.column_stack([points[neg_mask], distances[neg_mask]])
    
    print(f"    总样本数：{num_samples}")
    print(f"    正样本数：{len(pos_samples)} ({len(pos_samples)/num_samples:.1%})")
    print(f"    负样本数：{len(neg_samples)} ({len(neg_samples)/num_samples:.1%})")
    print(f"    SDF范围：[{np.min(distances):.3f}, {np.max(distances):.3f}]")
    
    # 检查梯度（球体表面附近）
    surface_points = points[np.abs(distances) < 0.05]
    if len(surface_points) > 0:
        # 计算表面点的梯度（对于球体应该是1）
        surface_distances = np.linalg.norm(surface_points, axis=1) - radius
        # 近似梯度计算
        gradients = []
        for i in range(len(surface_points)):
            # 找到最近的3个点
            dists = np.linalg.norm(surface_points - surface_points[i], axis=1)
            nearest = np.argsort(dists)[1:4]  # 跳过自己，取最近的3个
            if len(nearest) >= 3:
                # 近似梯度：SDF变化 / 距离变化
                sdf_diff = surface_distances[nearest] - surface_distances[i]
                dist_diff = dists[nearest]
                valid = dist_diff > 1e-6
                if np.sum(valid) > 0:
                    local_gradients = np.abs(sdf_diff[valid] / dist_diff[valid])
                    gradients.append(np.mean(local_gradients))
        
        if gradients:
            avg_gradient = np.mean(gradients)
            print(f"    表面附近平均梯度：{avg_gradient:.3f} (期望：≈1.0)")
    
    return pos_samples, neg_samples


def test_cube_sdf():
    """测试立方体SDF生成"""
    print("\n  测试2：立方体SDF（近似解）")
    
    num_samples = 10000
    cube_size = 0.8  # 边长
    
    # 在[-1, 1]范围内采样
    points = np.random.uniform(-1, 1, (num_samples, 3))
    
    # 计算立方体SDF（近似）
    # 到立方体表面的有符号距离
    q = np.abs(points) - cube_size/2
    exterior_dist = np.linalg.norm(np.maximum(q, 0), axis=1)
    interior_dist = np.minimum(np.max(q, axis=1), 0)
    distances = exterior_dist + interior_dist
    
    # 分离正负样本
    pos_mask = distances > 0
    neg_mask = distances <= 0
    
    pos_samples = np.column_stack([points[pos_mask], distances[pos_mask]])
    neg_samples = np.column_stack([points[neg_mask], distances[neg_mask]])
    
    print(f"    总样本数：{num_samples}")
    print(f"    正样本数：{len(pos_samples)} ({len(pos_samples)/num_samples:.1%})")
    print(f"    负样本数：{len(neg_samples)} ({len(neg_samples)/num_samples:.1%})")
    print(f"    SDF范围：[{np.min(distances):.3f}, {np.max(distances):.3f}]")
    
    return pos_samples, neg_samples


def test_your_sdf_data():
    """测试你的SDF数据"""
    print("\n  测试3：你的SDF数据验证")
    
    try:
        data = np.load("output.npz")
        
        if 'pos' in data and 'neg' in data:
            pos = data['pos']
            neg = data['neg']
            
            print(f"    正样本数：{len(pos)}")
            print(f"    负样本数：{len(neg)}")
            print(f"    总样本数：{len(pos) + len(neg)}")
            
            # 合并所有样本
            all_points = np.vstack([pos[:, :3], neg[:, :3]])
            all_sdf = np.hstack([pos[:, 3], neg[:, 3]])
            
            # 基本统计 
            print(f"    坐标范围：") 
            print(f"      X: [{np.min(all_points[:, 0]):.3f}, {np.max(all_points[:, 0]):.3f}]") 
            print(f"      Y: [{np.min(all_points[:, 1]):.3f}, {np.max(all_points[:, 1]):.3f}]") 
            print(f"      Z: [{np.min(all_points[:, 2]):.3f}, {np.max(all_points[:, 2]):.3f}]") 
            print(f"    SDF范围：[{np.min(all_sdf):.3f}, {np.max(all_sdf):.3f}]") 
            print(f"    SDF均值：{np.mean(all_sdf):.3f}") 
            
            # 检查符号 
            print(f"    符号统计：") 
            print(f"      正SDF比例：{np.sum(all_sdf > 0) / len(all_sdf):.1%}") 
            print(f"      负SDF比例：{np.sum(all_sdf < 0) / len(all_sdf):.1%}") 
            print(f"      零SDF比例：{np.sum(all_sdf == 0) / len(all_sdf):.1%}") 
            
            # 检查表面附近 
            surface_threshold = 0.01 
            surface_mask = np.abs(all_sdf) < surface_threshold 
            print(f"    表面附近样本（|SDF|<{surface_threshold}）：{np.sum(surface_mask)} ({np.sum(surface_mask)/len(all_sdf):.1%})") 
            
            # 检查归一化
            max_coord = np.max(np.abs(all_points))
            print(f"    归一化检查：")
            print(f"      最大坐标绝对值：{max_coord:.3f} (理想：≤1.0)")
            
            # 检查样本分布
            print(f"    样本分布：")
            print(f"      内部点(pos)比例：{len(pos) / (len(pos) + len(neg)):.1%} (理想：内部点应略多于外部点)")
            
        else:
            print("    错误：数据格式不正确")
            
    except Exception as e:
        print(f"    错误：无法读取文件 - {e}")

if __name__ == "__main__":
    check_sdf_generation_pipeline()
