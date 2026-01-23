#!/usr/bin/env python3
"""
deep_sdf_quality_check.py
专门针对DeepSDF PreprocessMesh生成的.npz文件进行质量检查
重点检查梯度、采样分布和SDF定义的正确性
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
import os
import sys

class DeepSDFQualityChecker:
    """
    DeepSDF生成数据的质量检查器
    专门检查PreprocessMesh生成的数据
    """
    
    def __init__(self, npz_path, mesh_path=None):
        self.npz_path = npz_path
        self.mesh_path = mesh_path
        self.data = None
        self.pos = None  # SDF < 0 (内部点)
        self.neg = None  # SDF > 0 (外部点)
        
    def load_data(self):
        """加载DeepSDF数据"""
        print(f"📂 加载数据: {self.npz_path}")
        
        try:
            self.data = np.load(self.npz_path)
            
            if 'pos' in self.data and 'neg' in self.data:
                self.pos = self.data['pos']  # (x, y, z, sdf)
                self.neg = self.data['neg']  # (x, y, z, sdf)
                
                print(f"✅ 成功加载数据")
                print(f"   pos (内部点): {self.pos.shape}")
                print(f"   neg (外部点): {self.neg.shape}")
                return True
            else:
                print(f"❌ 数据格式错误: 缺少pos或neg键")
                return False
                
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    def check_deepsdf_conventions(self):
        """检查DeepSDF数据约定"""
        print("\n🔍 检查DeepSDF数据约定:")
        
        # DeepSDF约定: 
        # pos: SDF < 0 (内部点，负距离)
        # neg: SDF > 0 (外部点，正距离)
        
        pos_sdf = self.pos[:, 3]
        neg_sdf = self.neg[:, 3]
        
        # 1. 检查符号约定
        pos_negative = np.sum(pos_sdf < 0) / len(pos_sdf)
        neg_positive = np.sum(neg_sdf > 0) / len(neg_sdf)
        
        print(f"   pos中SDF<0的比例: {pos_negative:.2%} (期望100%)")
        print(f"   neg中SDF>0的比例: {neg_positive:.2%} (期望100%)")
        
        if pos_negative < 0.99:
            print(f"   ⚠️ 警告: pos中有{(1-pos_negative)*100:.1f}%的点的SDF≥0")
        
        if neg_positive < 0.99:
            print(f"   ⚠️ 警告: neg中有{(1-neg_positive)*100:.1f}%的点的SDF≤0")
        
        # 2. 检查采样比例 (DeepSDF通常94%在表面附近，6%均匀分布)
        print(f"\n  采样比例分析:")
        total_samples = len(self.pos) + len(self.neg)
        pos_ratio = len(self.pos) / total_samples
        neg_ratio = len(self.neg) / total_samples
        
        print(f"   内部点(pos)比例: {pos_ratio:.2%}")
        print(f"   外部点(neg)比例: {neg_ratio:.2%}")
        
        # 合并所有数据
        all_points = np.vstack([self.pos[:, :3], self.neg[:, :3]])
        all_sdf = np.hstack([self.pos[:, 3], self.neg[:, 3]])
        
        # 3. 检查表面附近采样比例
        surface_threshold = 0.01  # |SDF| < 0.01定义为表面附近
        surface_mask = np.abs(all_sdf) < surface_threshold
        surface_ratio = np.sum(surface_mask) / len(all_sdf)
        
        print(f"   表面附近(|SDF|<{surface_threshold})比例: {surface_ratio:.2%}")
        print(f"   DeepSDF通常: ~94%在表面附近，~6%均匀分布")
        
        return all_points, all_sdf
    
    def check_sdf_gradient(self, num_samples=1000):
        """
        检查SDF梯度 - 这是最关键的质量指标
        SDF在表面附近的梯度应该接近1.0
        """
        print("\n📈 检查SDF梯度 (最关键的质量指标):")
        
        # 合并所有数据
        all_points = np.vstack([self.pos[:, :3], self.neg[:, :3]])
        all_sdf = np.hstack([self.pos[:, 3], self.neg[:, 3]])
        
        # 1. 随机采样一些点检查梯度
        np.random.seed(42)
        if len(all_points) > num_samples:
            indices = np.random.choice(len(all_points), num_samples, replace=False)
            sample_points = all_points[indices]
            sample_sdf = all_sdf[indices]
        else:
            sample_points = all_points
            sample_sdf = all_sdf
        
        # 使用KDTree快速查找最近邻
        tree = cKDTree(all_points)
        
        gradients = []
        valid_samples = 0
        
        print(f"   检查{len(sample_points)}个点的局部梯度...")
        
        for i in range(len(sample_points)):
            point = sample_points[i]
            sdf_value = sample_sdf[i]
            
            # 查找最近的k个邻居 (不包括自身)
            distances, indices = tree.query(point, k=6)  # 包括自身，所以取6
            
            # 跳过自身和零距离
            valid_idx = indices[distances > 1e-6]
            if len(valid_idx) >= 2:
                # 取最近的2个邻居
                neighbor_indices = valid_idx[:2]
                neighbor_distances = distances[1:3]  # 跳过第一个(自身)
                neighbor_sdf = all_sdf[neighbor_indices]
                
                # 计算梯度近似值：ΔSDF / Δ距离
                for j in range(len(neighbor_indices)):
                    if neighbor_distances[j] > 1e-6:
                        gradient = np.abs((neighbor_sdf[j] - sdf_value) / neighbor_distances[j])
                        
                        # 只考虑表面附近的点（梯度有意义）
                        if np.abs(sdf_value) < 0.1:  # 表面附近0.1范围内
                            gradients.append(gradient)
                            valid_samples += 1
        
        if gradients:
            gradients = np.array(gradients)
            
            print(f"   有效梯度样本数: {valid_samples}")
            print(f"   梯度统计:")
            print(f"     平均值: {np.mean(gradients):.4f}")
            print(f"     中位数: {np.median(gradients):.4f}")
            print(f"     标准差: {np.std(gradients):.4f}")
            print(f"     范围: [{np.min(gradients):.4f}, {np.max(gradients):.4f}]")
            
            # 检查梯度是否接近1.0
            mean_grad = np.mean(gradients)
            if 0.9 <= mean_grad <= 1.1:
                print(f"   ✅ SDF梯度正常 (接近1.0)")
            elif mean_grad < 0.5:
                print(f"   ❌ 严重问题: SDF梯度过小 ({mean_grad:.4f})")
                print(f"      这可能意味着SDF值被错误缩放或不是真正的距离函数")
            elif mean_grad > 2.0:
                print(f"   ❌ 严重问题: SDF梯度过大 ({mean_grad:.4f})")
            else:
                print(f"   ⚠️ 警告: SDF梯度偏离1.0 ({mean_grad:.4f})")
            
            return gradients
        else:
            print(f"   ⚠️ 无法计算梯度: 没有有效的邻居点")
            return None
    
    def analyze_sampling_strategy(self):
        """分析采样策略是否符合DeepSDF论文"""
        print("\n🎯 分析采样策略:")
        
        # 合并所有数据
        all_points = np.vstack([self.pos[:, :3], self.neg[:, :3]])
        all_sdf = np.hstack([self.pos[:, 3], self.neg[:, 3]])
        
        # 1. 检查SDF值的分布
        print(f"   SDF值分布分析:")
        
        # 定义不同的区域
        regions = {
            '表面内(深)': all_sdf < -0.1,
            '表面附近内': (all_sdf >= -0.1) & (all_sdf < -0.01),
            '极表面附近': np.abs(all_sdf) < 0.01,
            '表面附近外': (all_sdf > 0.01) & (all_sdf <= 0.1),
            '表面外(远)': all_sdf > 0.1
        }
        
        total = len(all_sdf)
        for name, mask in regions.items():
            count = np.sum(mask)
            print(f"     {name}: {count}点 ({count/total:.2%})")
        
        # 2. 检查空间分布
        print(f"\n   空间分布检查:")
        
        # 计算到原点的距离
        distances = np.linalg.norm(all_points, axis=1)
        
        # 分箱统计
        bins = np.linspace(0, np.max(distances), 6)
        bin_indices = np.digitize(distances, bins)
        
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_sdf = all_sdf[mask]
                print(f"     距离 [{bins[i-1]:.2f}, {bins[i]:.2f}]: "
                      f"{np.sum(mask)}点, "
                      f"平均SDF={np.mean(bin_sdf):.3f}")
    
    def check_coordinate_normalization(self):
        """检查坐标归一化"""
        print("\n📏 检查坐标归一化:")
        
        all_points = np.vstack([self.pos[:, :3], self.neg[:, :3]])
        
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        max_abs = np.max(np.abs(all_points))
        
        print(f"   坐标范围:")
        print(f"     X: [{min_coords[0]:.6f}, {max_coords[0]:.6f}]")
        print(f"     Y: [{min_coords[1]:.6f}, {max_coords[1]:.6f}]")
        print(f"     Z: [{min_coords[2]:.6f}, {max_coords[2]:.6f}]")
        print(f"   最大绝对值: {max_abs:.6f}")
        
        if max_abs > 1.0:
            print(f"   ⚠️ 警告: 坐标超出[-1,1]范围")
            if max_abs > 1.2:
                print(f"   ❌ 严重: 坐标范围过大，可能导致训练问题")
        else:
            print(f"   ✅ 坐标在[-1,1]范围内")
    
    def generate_quality_report(self, gradients=None):
        """生成质量报告"""
        print("\n" + "="*80)
        print("DeepSDF数据质量报告")
        print("="*80)
        
        # 基本评分系统
        score = 0
        max_score = 10
        
        # 1. 符号约定检查
        pos_sdf = self.pos[:, 3]
        neg_sdf = self.neg[:, 3]
        
        pos_correct = np.sum(pos_sdf < 0) / len(pos_sdf)
        neg_correct = np.sum(neg_sdf > 0) / len(neg_sdf)
        
        if pos_correct > 0.99 and neg_correct > 0.99:
            print("✅ 符号约定: 优秀")
            score += 2
        elif pos_correct > 0.95 and neg_correct > 0.95:
            print("✅ 符号约定: 良好")
            score += 1.5
        else:
            print("❌ 符号约定: 有问题")
        
        # 2. 梯度检查
        if gradients is not None:
            mean_grad = np.mean(gradients)
            if 0.95 <= mean_grad <= 1.05:
                print("✅ SDF梯度: 优秀 (接近1.0)")
                score += 3
            elif 0.9 <= mean_grad <= 1.1:
                print("✅ SDF梯度: 良好")
                score += 2
            elif 0.7 <= mean_grad <= 1.3:
                print("⚠️ SDF梯度: 可接受")
                score += 1
            else:
                print(f"❌ SDF梯度: 严重问题 (均值={mean_grad:.4f})")
        
        # 3. 表面附近采样
        all_sdf = np.hstack([self.pos[:, 3], self.neg[:, 3]])
        surface_ratio = np.sum(np.abs(all_sdf) < 0.01) / len(all_sdf)
        
        if 0.1 <= surface_ratio <= 0.3:
            print(f"✅ 表面采样: 良好 ({surface_ratio:.1%})")
            score += 2
        elif surface_ratio > 0.3:
            print(f"⚠️ 表面采样: 可能过密 ({surface_ratio:.1%})")
            score += 1
        else:
            print(f"❌ 表面采样: 不足 ({surface_ratio:.1%})")
        
        # 4. 坐标归一化
        all_points = np.vstack([self.pos[:, :3], self.neg[:, :3]])
        max_abs = np.max(np.abs(all_points))
        
        if max_abs <= 1.0:
            print("✅ 坐标归一化: 优秀")
            score += 2
        elif max_abs <= 1.1:
            print("✅ 坐标归一化: 良好")
            score += 1.5
        elif max_abs <= 1.2:
            print("⚠️ 坐标归一化: 可接受")
            score += 1
        else:
            print(f"❌ 坐标归一化: 超出范围 (max_abs={max_abs:.3f})")
        
        # 5. 正负样本平衡
        total = len(self.pos) + len(self.neg)
        pos_ratio = len(self.pos) / total
        
        if 0.4 <= pos_ratio <= 0.6:
            print(f"✅ 样本平衡: 优秀 ({pos_ratio:.1%})")
            score += 1
        elif 0.3 <= pos_ratio <= 0.7:
            print(f"✅ 样本平衡: 良好 ({pos_ratio:.1%})")
            score += 0.5
        else:
            print(f"⚠️ 样本平衡: 需要关注 ({pos_ratio:.1%})")
        
        # 总体评价
        print(f"\n📊 总体质量分数: {score:.1f}/{max_score}")
        
        if score >= 8:
            print("🎉 优秀: 数据质量很高，适合用于DeepSDF训练")
        elif score >= 6:
            print("✅ 良好: 数据质量可以接受，可以考虑微调")
        elif score >= 4:
            print("⚠️ 一般: 数据质量一般，建议检查生成参数")
        else:
            print("❌ 较差: 数据质量有问题，需要重新生成")
    
    def visualize_analysis(self):
        """生成可视化分析图表"""
        print("\n📊 生成可视化分析...")
        
        # 合并所有数据
        all_points = np.vstack([self.pos[:, :3], self.neg[:, :3]])
        all_sdf = np.hstack([self.pos[:, 3], self.neg[:, 3]])
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. SDF分布直方图
        axes[0, 0].hist(all_sdf, bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='零值面')
        axes[0, 0].set_xlabel('SDF值')
        axes[0, 0].set_ylabel('频率')
        axes[0, 0].set_title('SDF值分布')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 表面附近SDF分布
        surface_mask = np.abs(all_sdf) < 0.1
        surface_sdf = all_sdf[surface_mask]
        axes[0, 1].hist(surface_sdf, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('SDF值 (|SDF|<0.1)')
        axes[0, 1].set_ylabel('频率')
        axes[0, 1].set_title('表面附近SDF分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. XY平面投影
        sample_idx = np.random.choice(len(all_points), min(5000, len(all_points)), replace=False)
        scatter = axes[0, 2].scatter(all_points[sample_idx, 0], all_points[sample_idx, 1],
                                     c=all_sdf[sample_idx], cmap='RdBu_r', 
                                     s=2, alpha=0.6, vmin=-0.1, vmax=0.1)
        axes[0, 2].set_xlabel('X')
        axes[0, 2].set_ylabel('Y')
        axes[0, 2].set_title('XY平面投影 (SDF着色)')
        axes[0, 2].axis('equal')
        plt.colorbar(scatter, ax=axes[0, 2], shrink=0.8)
        
        # 4. 坐标分布
        coord_names = ['X', 'Y', 'Z']
        coord_data = [all_points[:, i] for i in range(3)]
        
        bp = axes[1, 0].boxplot(coord_data, labels=coord_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        axes[1, 0].axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_ylabel('坐标值')
        axes[1, 0].set_title('坐标分布箱线图')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. SDF与距离关系
        distances = np.linalg.norm(all_points, axis=1)
        axes[1, 1].scatter(distances[sample_idx], all_sdf[sample_idx], 
                          s=1, alpha=0.3)
        axes[1, 1].set_xlabel('到原点的距离')
        axes[1, 1].set_ylabel('SDF值')
        axes[1, 1].set_title('SDF值 vs 距离')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 正负点分布
        pos_idx = np.random.choice(len(self.pos), min(2000, len(self.pos)), replace=False)
        neg_idx = np.random.choice(len(self.neg), min(2000, len(self.neg)), replace=False)
        
        axes[1, 2].scatter(self.pos[pos_idx, 0], self.pos[pos_idx, 1], 
                          c='red', s=2, alpha=0.5, label='pos (SDF<0)')
        axes[1, 2].scatter(self.neg[neg_idx, 0], self.neg[neg_idx, 1], 
                          c='blue', s=2, alpha=0.5, label='neg (SDF>0)')
        axes[1, 2].set_xlabel('X')
        axes[1, 2].set_ylabel('Y')
        axes[1, 2].set_title('正负点空间分布')
        axes[1, 2].legend()
        axes[1, 2].axis('equal')
        
        plt.suptitle(f'DeepSDF数据质量分析 - {os.path.basename(self.npz_path)}', fontsize=16)
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.splitext(self.npz_path)[0] + '_quality_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化图表已保存: {output_path}")
        
        plt.show()

def main():
    """主函数"""
    # 检查参数
    if len(sys.argv) < 2:
        print("使用方法: python deep_sdf_quality_check.py <npz文件路径> [网格文件路径]")
        print("示例: python deep_sdf_quality_check.py output.npz model.obj")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    mesh_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 创建检查器
    checker = DeepSDFQualityChecker(npz_path, mesh_path)
    
    # 加载数据
    if not checker.load_data():
        sys.exit(1)
    
    # 执行各项检查
    print("\n" + "="*80)
    print("开始质量检查")
    print("="*80)
    
    # 1. 检查DeepSDF约定
    all_points, all_sdf = checker.check_deepsdf_conventions()
    
    # 2. 检查SDF梯度 (最关键)
    gradients = checker.check_sdf_gradient(num_samples=2000)
    
    # 3. 分析采样策略
    checker.analyze_sampling_strategy()
    
    # 4. 检查坐标归一化
    checker.check_coordinate_normalization()
    
    # 5. 生成质量报告
    checker.generate_quality_report(gradients)
    
    # 6. 可视化分析
    checker.visualize_analysis()
    
    print("\n" + "="*80)
    print("检查完成!")
    print("="*80)
    
    # 如果梯度有问题，提供建议
    if gradients is not None:
        mean_grad = np.mean(gradients)
        if mean_grad < 0.5:
            print("\n🔧 修复建议 (梯度过小):")
            print("   1. 检查PreprocessMesh的--var参数，尝试增大方差")
            print("   2. 确保网格已正确归一化到[-1,1]范围内")
            print("   3. 验证网格是否为水密(watertight)网格")
            print("   4. 尝试使用测试模式: ./PreprocessMesh -m model.obj -o output.npz -t")
        elif mean_grad > 1.5:
            print("\n🔧 修复建议 (梯度过大):")
            print("   1. 检查PreprocessMesh的--var参数，尝试减小方差")
            print("   2. 检查网格尺寸是否正确")

if __name__ == "__main__":
    main()