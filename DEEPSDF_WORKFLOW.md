# DeepSDF完整工作流运行指南

## 1. 环境准备

### 1.1 激活虚拟环境
```bash
source /root/DEEPSDF/DeepSDF/DeepSDF_venv_new/bin/activate
```

### 1.2 验证环境
```bash
python -c 'import torch; import numpy; import trimesh; print("Environment ready!")'
```

## 2. 数据准备

### 2.1 下载ShapeNet数据集
```bash
# 请根据DeepSDF仓库说明下载ShapeNet数据集
# 数据集结构应如下：
# /path/to/ShapeNetCorev2/
# ├── 02691156/  # 飞机类别
# ├── 02828884/  # 基准类别
# └── ...
```

### 2.2 预处理ShapeNet数据
```bash
cd /root/DEEPSDF/DeepSDF
python preprocess_data.py --data_dir /path/to/ShapeNetCorev2 --split examples/splits/sv2_planes_test.json --skip
```

## 3. 生成SDF样本

### 3.1 编译SDF生成工具
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
```

### 3.2 生成训练和测试SDF样本
```bash
# 生成训练SDF样本
python preprocess_data.py --data_dir /path/to/ShapeNetCorev2 --split examples/splits/sv2_planes_train.json --skip

# 生成测试SDF样本
python preprocess_data.py --data_dir /path/to/ShapeNetCorev2 --split examples/splits/sv2_planes_test.json --skip
```

## 4. 模型训练

### 4.1 创建实验目录
```bash
mkdir -p experiments/planes
```

### 4.2 配置实验参数
```bash
# 可以根据需要修改examples/configs/plane.json文件
cp examples/configs/plane.json experiments/planes/config.json
```

### 4.3 开始训练
```bash
python train.py --experiment_name planes
```

### 4.4 监控训练进度
```bash
# 使用tensorboard监控（如果tensorboardX单独导入正常）
tensorboard --logdir experiments/planes/logs
```

## 5. 模型重建

### 5.1 从训练好的模型中重建3D模型
```bash
python reconstruct.py --experiment_name planes --checkpoint 2000 --split examples/splits/sv2_planes_test.json --data_dir /path/to/ShapeNetCorev2
```

### 5.2 重建结果位置
```
experiments/planes/reconstructions/2000/
├── 02691156_1234567890.ply  # 重建的PLY文件
└── ...
```

## 6. 模型评估

### 6.1 计算Chamfer距离
```bash
python evaluate.py --experiment_name planes --checkpoint 2000 --split examples/splits/sv2_planes_test.json --data_dir /path/to/ShapeNetCorev2
```

### 6.2 评估结果位置
```
experiments/planes/evaluation/2000/
├── chamfer_distance.json  # Chamfer距离结果
└── ...
```

## 7. 常见问题与解决方案

### 7.1 CUDA相关错误
- 确保所有代码中的`.cuda()`调用已替换为`.cpu()`
- 运行命令时添加`CUDA_VISIBLE_DEVICES=""`禁用GPU

### 7.2 Segmentation fault
- 尝试单独导入tensorboardX，或在不需要可视化时跳过
- 检查Python版本是否为3.6

### 7.3 依赖安装问题
- 使用提供的`requirements_clean.txt`重新安装依赖
- 确保使用`--no-cache-dir`选项避免缓存问题

## 8. 工作流流程图

```
数据准备 → SDF生成 → 模型训练 → 模型重建 → 模型评估
    ↓           ↓           ↓           ↓           ↓
ShapeNet → SDF样本 → 训练模型 → 3D模型 → 评估指标
```

## 9. 注意事项

1. 所有命令均在虚拟环境中执行
2. 根据实际硬件情况调整batch_size和其他参数
3. 训练过程可能需要数小时到数天，取决于数据集大小和硬件性能
4. 重建过程使用Marching Cubes算法，可能需要大量内存
5. 评估过程需要计算Chamfer距离，确保测试集包含完整的模型数据
