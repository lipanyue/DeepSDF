import numpy as np
import matplotlib.pyplot as plt


def diagnose_sdf_issues(npz_path):
    """深度诊断SDF问题"""
    data = np.load(npz_path)
    pos = data['pos']  # 应该：SDF < 0
    neg = data['neg']  # 应该：SDF > 0
    
    print("="*60)
    print("SDF深度诊断报告")
    print("="*60)
    
    # 1. 检查符号定义是否颠倒
    print("\n1. 符号定义检查:")
    print(f"   pos样本名称: {len(pos)}个点，但SDF应为负")
    print(f"   neg样本名称: {len(neg)}个点，但SDF应为正")
    print(f"   pos的实际SDF范围: [{pos[:, 3].min():.6f}, {pos[:, 3].max():.6f}]")
    print(f"   neg的实际SDF范围: [{neg[:, 3].min():.6f}, {neg[:, 3].max():.6f}]")
    
    # 检查是否符号颠倒
    if pos[:, 3].min() > 0 and neg[:, 3].max() < 0:
        print("  ⚠️ 严重：符号完全颠倒了！pos应该是负值，neg应该是正值")
        print("  可能原因：生成SDF时符号计算错误")
    
    # 2. 检查采样分布
    print("\n2. 采样分布检查:")
    
    # 计算到中心的距离
    all_points = np.vstack([pos[:, :3], neg[:, :3]])
    all_sdf = np.hstack([pos[:, 3], neg[:, 3]])
    center = np.mean(all_points, axis=0)
    distances = np.linalg.norm(all_points - center, axis=1)
    
    # 按距离分箱统计
    bins = np.linspace(0, distances.max(), 11)
    bin_indices = np.digitize(distances, bins)
    
    print("  距离中心不同距离的SDF统计:")
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            sdf_in_bin = all_sdf[mask]
            print(f"    距离 [{bins[i-1]:.2f}, {bins[i]:.2f}]: "
                  f"{len(sdf_in_bin)}点, "
                  f"平均SDF={np.mean(sdf_in_bin):.3f}, "
                  f"正比例={np.sum(sdf_in_bin>0)/len(sdf_in_bin):.1%}")
    
    # 3. 检查梯度计算（简化版）
    print("\n3. 梯度合理性检查:")
    
    # 随机采样一些点，计算局部梯度
    np.random.seed(42)
    sample_indices = np.random.choice(len(all_points), min(100, len(all_points)), replace=False)
    
    gradients = []
    for idx in sample_indices:
        point = all_points[idx]
        sdf_value = all_sdf[idx]
        
        # 找到最近的5个点
        dists = np.linalg.norm(all_points - point, axis=1)
        nearest = np.argsort(dists)[1:6]  # 跳过自己
        
        # 近似梯度：SDF变化 / 距离变化
        if len(nearest) >= 3:
            sdf_diff = all_sdf[nearest] - sdf_value
            dist_diff = dists[nearest]
            
            # 避免除零
            valid = dist_diff > 1e-6
            if np.sum(valid) > 0:
                local_gradients = np.abs(sdf_diff[valid] / dist_diff[valid])
                gradients.append(np.mean(local_gradients))
    
    if gradients:
        print(f"  局部梯度估计: 平均={np.mean(gradients):.6f}, "
              f"范围=[{np.min(gradients):.6f}, {np.max(gradients):.6f}]")
        print(f"  期望梯度应在0.8-1.2范围内")
        
        if np.mean(gradients) < 0.5:
            print("  ⚠️ 严重：梯度太小，不是有效的SDF")
    
    # 4. 可视化诊断
    print("\n4. 生成诊断图...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 4.1 SDF值分布（分离pos和neg）
    ax1 = axes[0, 0]
    ax1.hist(pos[:, 3], bins=50, alpha=0.5, label='pos (应<0)', color='blue')
    ax1.hist(neg[:, 3], bins=50, alpha=0.5, label='neg (应>0)', color='red')
    ax1.axvline(x=0, color='black', linestyle='--')
    ax1.set_xlabel('SDF值')
    ax1.set_ylabel('频率')
    ax1.set_title('SDF值分布（按类别）')
    ax1.legend()
    
    # 4.2 空间分布
    ax2 = axes[0, 1]
    # 随机采样部分点显示
    sample_idx = np.random.choice(len(all_points), min(5000, len(all_points)), replace=False)
    scatter = ax2.scatter(all_points[sample_idx, 0], all_points[sample_idx, 1], 
                         c=all_sdf[sample_idx], cmap='RdBu_r', 
                         s=1, alpha=0.5, vmin=-0.1, vmax=0.1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY平面投影（SDF着色）')
    ax2.axis('equal')
    plt.colorbar(scatter, ax=ax2, shrink=0.8)
    
    # 4.3 SDF vs 到中心距离
    ax3 = axes[0, 2]
    ax3.scatter(distances[sample_idx], all_sdf[sample_idx], 
               s=1, alpha=0.3)
    ax3.set_xlabel('到中心的距离')
    ax3.set_ylabel('SDF值')
    ax3.set_title('SDF值 vs 距离')
    ax3.axhline(y=0, color='black', linestyle='--')
    ax3.grid(True, alpha=0.3)
    
    # 4.4 正负点空间分布
    ax4 = axes[1, 0]
    # 分别显示正负点
    pos_sample = np.random.choice(len(pos), min(2000, len(pos)), replace=False)
    neg_sample = np.random.choice(len(neg), min(2000, len(neg)), replace=False)
    
    ax4.scatter(pos[pos_sample, 0], pos[pos_sample, 1], 
               c='red', s=2, alpha=0.5, label='pos (SDF<0)')
    ax4.scatter(neg[neg_sample, 0], neg[neg_sample, 1], 
               c='blue', s=2, alpha=0.5, label='neg (SDF>0)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('正负点空间分布')
    ax4.legend()
    ax4.axis('equal')
    
    # 4.5 维度分析
    ax5 = axes[1, 1]
    dims = ['X', 'Y', 'Z']
    dim_means = []
    dim_stds = []
    
    for i, dim in enumerate(dims):
        dim_means.append(np.mean(all_points[:, i]))
        dim_stds.append(np.std(all_points[:, i]))
    
    x_pos = np.arange(len(dims))
    ax5.bar(x_pos - 0.2, dim_means, width=0.4, label='均值', alpha=0.7)
    ax5.bar(x_pos + 0.2, dim_stds, width=0.4, label='标准差', alpha=0.7)
    ax5.set_xlabel('维度')
    ax5.set_ylabel('值')
    ax5.set_title('各维度统计')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(dims)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 4.6 问题点识别
    ax6 = axes[1, 2]
    # 识别可能是问题的点（SDF符号与位置不符）
    # 简单假设：靠近中心的点应该是负的（内部），远离的是正的（外部）
    norm_distances = distances / distances.max()
    
    # 计算每个点的"问题分数"
    # 如果SDF为正但离中心近，或者SDF为负但离中心远，可能有问题
    problem_score = np.abs(norm_distances - (all_sdf > 0).astype(float))
    
    ax6.hist(problem_score, bins=50, alpha=0.7)
    ax6.set_xlabel('不一致分数')
    ax6.set_ylabel('频率')
    ax6.set_title('SDF符号与位置一致性')
    ax6.axvline(x=0.5, color='red', linestyle='--', label='阈值')
    ax6.legend()
    
    plt.suptitle(f'SDF数据深度诊断 - {npz_path}', fontsize=16)
    plt.tight_layout()
    # 保存图像而非显示，因为是在Docker环境中
    plt.savefig('sdf_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"  诊断图已保存为: sdf_diagnosis.png")
    plt.close()
    
    print("\n" + "="*60)
    print("诊断总结：")
    print("="*60)
    
    # 基于分析给出建议
    if np.mean(pos[:, 3]) < 0 and np.mean(neg[:, 3]) > 0:
        print("1. ✓ 符号定义正确：pos为负，neg为正")
    else:
        print("1. ⚠️ 符号可能有问题")
    
    if len(pos) / (len(pos) + len(neg)) < 0.1 or len(pos) / (len(pos) + len(neg)) > 0.9:
        print("2. ⚠️ 正负样本比例严重失衡")
    
    if np.abs(np.mean(gradients) if gradients else 0) < 0.5:
        print("3. ⚠️ 梯度太小，不符合SDF定义")
    
    print("\n建议：")
    print("1. 检查SDF生成代码，确保计算的是有符号距离")
    print("2. 检查采样策略，确保表面内外都有足够样本")
    print("3. 验证坐标归一化是否正确")
    print("4. 考虑重新生成SDF数据")


# 运行诊断
if __name__ == "__main__":
    diagnose_sdf_issues("output.npz")
