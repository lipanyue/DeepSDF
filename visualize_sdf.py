import numpy as np 
import open3d as o3d 


def visualize_deepsdf_data(data): 
    """ 
    可视化DeepSDF数据（区分表面内外的点） 
    - 表面外的点：红色（近）→ 蓝色（远） 
    - 表面内的点：绿色（近）→ 蓝色（远） 
    """ 
    # 提取数据 
    pos_data = data['pos']  # 表面内的点 (距离 < 0) 
    neg_data = data['neg']  # 表面外的点 (距离 > 0) 
    # 合并所有点 
    all_xyz = np.vstack([pos_data[:, :3], neg_data[:, :3]]) 
    all_distances = np.hstack([pos_data[:, 3], neg_data[:, 3]]) 
    # 创建点云 
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(all_xyz) 
    # 初始化颜色数组 
    colors = np.zeros((len(all_distances), 3)) 
    # 区分表面内外的点 
    pos_mask = all_distances < 0  # 表面内的点 
    neg_mask = all_distances >= 0  # 表面外的点 

    # 对表面内的点：绿色（近）→ 蓝色（远） 
    if np.any(pos_mask): 
        pos_distances = np.abs(all_distances[pos_mask])  # 取绝对值 
        pos_normalized = (pos_distances - pos_distances.min()) / (pos_distances.max() - pos_distances.min() + 1e-6) 
        colors[pos_mask, 1] = 1.0 - pos_normalized  # 绿色通道 
        colors[pos_mask, 2] = pos_normalized         # 蓝色通道 
    # 对表面外的点：红色（近）→ 蓝色（远） 
    if np.any(neg_mask): 
        neg_distances = all_distances[neg_mask] 
        neg_normalized = (neg_distances - neg_distances.min()) / (neg_distances.max() - neg_distances.min() + 1e-6) 
        colors[neg_mask, 0] = 1.0 - neg_normalized  # 红色通道 
        colors[neg_mask, 2] = neg_normalized         # 蓝色通道 

    # 赋值颜色 
    pcd.colors = o3d.utility.Vector3dVector(colors) 
    # 可视化并保存图像
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="DeepSDF数据可视化", width=800, height=600, visible=False)  # visible=False for headless
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("sdf_visualization.png")
    print("SDF可视化图像已保存为 sdf_visualization.png")
    vis.destroy_window()


if __name__ == "__main__": 
    data = np.load("/root/DEEPSDF/DeepSDF/bin/output.npz") 
    print("文件构成：", data.files) 
    print("表面内点（距离 < 0）:", data['pos'].shape) 
    print("前5个表面内点：")
    print(data['pos'][:5]) 
    print("表面外点（距离 >= 0）:", data['neg'].shape) 
    print("前5个表面外点：")
    print(data['neg'][:5]) 
    visualize_deepsdf_data(data)
