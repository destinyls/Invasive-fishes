# 鱼类分类系统 (基于对比学习)

本项目实现了一个基于 BYOL (Bootstrap Your Own Latent) 原理的高性能鱼类分类系统。系统通过比较输入图像与模板图像的特征相似度来确定类别，采用多种先进技术使准确率从典型的 60% 提升到 80–90%。

## 主要特性

- **改进的对比学习**: 基于同类不同帧的 BYOL 对比学习（相比单帧增强更有效）
- **多模板支持**: 每个类别可使用多个模板图像以提升准确率
- **强大的骨干网络**: 支持 `resnet50`、`resnet101` 和 `efficientnet_b2`
- **先进的训练方法**: 学习率调度 (CosineAnnealingLR)、早停、差异化学习率
- **🆕 增强的训练时间追踪**: 
  - 实时显示每个Step的剩余时间
  - 详细的训练进度预估（Epoch和总体进度）
  - 训练速度统计（每秒处理步数）
  - 可视化进度条和ETA显示
- **🆕 实时训练监控**: 独立的进度监控脚本，实时跟踪训练状态
- **全面的评估**: 详细指标、混淆矩阵、最易混淆类别对可视化
- **批量处理**: 高效的大规模数据批量推理

## 环境要求

- Python 3.6+
- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- Pillow >= 8.0.0
- numpy >= 1.19.0
- tqdm >= 4.50.0
- matplotlib >= 3.3.0
- scikit-learn >= 0.23.0
- seaborn >= 0.11.0

通过以下命令安装依赖:

```bash
pip install -r requirements.txt
```

## 项目结构

```
fish_classification/
├── data/
│   ├── template/             # 单模板 (每类一个)
│   └── ensemble_templates/   # 多模板 (每类多个)
├── models/                   # 保存的模型权重
├── results/                  # 评估结果与可视化 (evaluation, batch, prediction_result.png 等)
├── model.py                  # 基于 BYOL 架构的核心模型实现 (FishClassifier)
├── train.py                  # 🆕 增强的训练脚本 (包含详细时间跟踪)
├── evaluate.py               # 模型评估脚本
├── batch_inference.py        # 大规模数据批量推理脚本
├── dataset.py                # 数据集处理模块 (FishDataset)
├── predict.py                # 单张图像预测脚本
├── demo.py                   # 训练/预测/集成模板一体化演示脚本
├── show_training_time.py     # 训练时间信息显示工具
├── 🆕 show_training_progress.py  # 实时训练进度监控脚本
├── IMPROVEMENTS.md           # 改进细节文档
├── requirements.txt          # 环境依赖
└── README.md                 # 项目说明
```

## 使用方法

以下所有命令均在 `fish_classification/` 目录下运行。

### 1. 训练模型 (Training)

```bash
python train.py \
  --train_dir ../../data/fish-clip-dataset/train \
  --val_dir ../../data/fish-clip-dataset/val \
  --template_dir data/ensemble_templates \
  --ensemble_templates \
  --batch_size 32 \
  --image_size 448 \
  --num_workers 4 \
  --backbone resnet101 \
  --epochs 30 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --save_path models/fish_classifier.pth \
  --use_class_pairs True
```

- `--train_dir`: 训练集目录（按类别子文件夹组织）
- `--val_dir`: 验证集目录（可选）
- `--template_dir`: 模板图像目录（单模板或多模板）
- `--ensemble_templates`: 从训练集中生成多样化集成模板
- `--batch_size`: 批大小
- `--image_size`: 图像尺寸
- `--num_workers`: 数据加载进程数
- `--backbone`: 骨干网络，支持 `resnet50`, `resnet101`, `efficientnet_b2`
- `--epochs`, `--lr`, `--weight_decay`: 训练超参数
- `--save_path`: 模型保存路径
- `--use_class_pairs`: 使用同类不同帧的对比学习（默认为True）

### 2. 演示脚本 (Demo)

集成了训练、集成模板和预测三种模式：

```bash
# 训练演示
python demo.py \
  --mode train \
  --data_dir /path/to/data \
  --model_path models/fish_classifier.pth \
  --backbone resnet50 \
  --template_dir data/template

# 生成集成模板
python demo.py \
  --mode ensemble \
  --data_dir /path/to/data \
  --model_path models/fish_classifier.pth

# 单图预测
python demo.py \
  --mode predict \
  --image /path/to/image.jpg \
  --model_path models/fish_classifier.pth \
  --template_dir data/ensemble_templates
```

### 3. 单图预测 (Predict)

```bash
python predict.py \
  --input ../../clip_classification/fish.png \
  --model_path models/fish_classifier.pth \
  --template_dir data/ensemble_templates \
  --output predictions.json \
  --image_size 224 \
  --top_k 5 \
  --visualize
```

- `--input`: 输入图像路径
- `--template_dir`: 模板目录
- `--output`: 输出 JSON 文件路径
- `--top_k`: 显示 Top K 类别
- `--visualize`: 绘制并保存可视化结果

### 4. 模型评估 (Evaluate)

```bash
python evaluate.py \
  --val_dir /path/to/val \
  --model_path models/fish_classifier.pth \
  --template_dir data/ensemble_templates \
  --output_dir results/evaluation \
  --image_size 224 \
  --batch_size 32 \
  --backbone resnet50
```

- `--output_dir`: 评估结果保存目录

### 5. 批量推理 (Batch Inference)

```bash
python batch_inference.py \
  --val_dir ../../data/fish-clip-dataset/val \
  --model_path models/fish_classifier.pth \
  --template_dir data/ensemble_templates \
  --output_dir results/batch \
  --batch_size 64 \
  --image_size 224 \
  --num_workers 4
```

### 6. 🆕 增强的训练时间跟踪和监控

#### 6.1 训练期间的实时时间显示

新的训练脚本提供详细的时间跟踪功能：

**每个Step显示**：
- 当前Step执行时间
- 平均Step时间
- 当前Epoch剩余时间
- 总训练剩余时间
- 完成百分比

**示例输出**：
```
Epoch 5/30 | Step 127/500 | Loss: 0.2156 | LR: 1.23e-04
⏱️  Step: 2.3s | Avg: 2.1s | Epoch ETA: 13m 2s | Total ETA: 1h 25m | Progress: 16.8%
```

**Epoch完成时显示**：
```
================================================================================
📊 EPOCH 5/30 COMPLETED
📈 Train Loss: 0.2156 | LR: 1.23e-04
⏱️  This Epoch: 17m 30s | Avg/Epoch: 16m 45s
🕐 Elapsed: 1h 23m | Remaining: 7h 2m | ETA: 15:47:23
📊 Progress: 16.7% complete
================================================================================
```

#### 6.2 实时训练监控脚本

使用独立的监控脚本实时跟踪训练进度（无需中断训练）：

```bash
# 自动检测日志文件
python show_training_progress.py

# 指定日志文件
python show_training_progress.py --log-file training.log

# 设置刷新频率（秒）
python show_training_progress.py --refresh-rate 3
```

**监控界面示例**：
```
================================================================================
🐠 FISH CLASSIFICATION TRAINING MONITOR
================================================================================
🚀 Training Status: ✅ Running
📅 Started: 2024-01-15 10:30:25
📊 Epoch Progress: 5/30
🔄 Step Progress: 127/500
📈 Overall Progress: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 16.8%
📉 Latest Loss: 0.2156
⚙️  Learning Rate: 1.23e-04
🏆 Best Val Loss: 0.1892
⏱️  Time Remaining: 7h 2m
🔄 Last Update: 3s ago

================================================================================
🔄 Refreshing every 5s... (Press Ctrl+C to exit)
```

#### 6.3 查看历史训练时间信息

查看已保存模型的训练统计信息：

```bash
python show_training_time.py models/fish_classifier.pth
```

**显示内容**：
- 训练开始和结束时间
- 总训练时间
- 平均每个Epoch时间
- 最快/最慢Epoch时间
- 平均Step时间
- 训练速度（步/秒）
- 模型配置信息

#### 6.4 训练日志重定向

为了配合监控脚本，建议将训练日志重定向到文件：

```bash
# 方法1: 重定向到文件
python train.py [参数] > training.log 2>&1

# 方法2: 同时显示和保存
python train.py [参数] 2>&1 | tee training.log

# 后台运行训练，在另一个终端监控
nohup python train.py [参数] > training.log 2>&1 &
python show_training_progress.py --log-file training.log
```

#### 6.5 保存的时间信息

所有新训练的模型自动包含详细的时间信息：

- `training_start_time`: 训练开始时间戳
- `training_duration_seconds`: 总训练时间（秒）
- `epoch_times`: 每个Epoch的时间列表
- `step_times`: 每个Step的时间列表（用于统计分析）

## 模型架构

该模型实现了基于 BYOL 的对比学习双网络结构：

1. **双网络结构**  
   - 在线网络（Online）：骨干网络 + 投影头 + 预测头  
   - 目标网络（Target）：骨干网络 + 投影头 (EMA 更新)

2. **增强的投影头**  
   - 3 层 MLP，增加宽度和深度  
   - 输出维度为 512

3. **类别配对训练**
   - 使用来自相同类别但不同帧的图像对进行训练
   - 相比使用同一图像的不同增强，学习更多类内变化

4. **多模板分类**  
   - 单模板、子目录多模板和集成模板三种方式  
   - 跨模板最大相似度投票

## 预期性能

- **~75-80%** 单模板 + 增强架构  
- **~80-85%** 多模板 + 增强训练  
- **~85-90%** 集成模板 + 强大骨干
- **~90-95%** 类别配对训练 + 集成模板 + 强大骨干

更多详情请参见 [IMPROVEMENTS.md](IMPROVEMENTS.md)。 