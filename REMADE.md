# VIReid - Weakly Supervised Visible-Infrared Person Re-Identification

本项目基于论文《Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning》的实现。

## 📝 引用说明

本项目借鉴并实现了以下论文的方法：

**论文**: Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning

**作者**: Yafei Zhang, Lingqi Kong, Huafeng Li, Jie Wen

**官方仓库**: [https://github.com/KongLingqi2333/WSL-VIReID.git](https://github.com/KongLingqi2333/WSL-VIReID.git)

**论文引用**:
```bibtex
@misc{zhang2025weaklysupervisedvisibleinfraredperson,
      title={Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning}, 
      author={Yafei Zhang and Lingqi Kong and Huafeng Li and Jie Wen},
      year={2025},
      eprint={2507.12942},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.12942}, 
}
```

## 🎯 项目简介

本项目实现了一个弱监督可见光-红外跨模态行人重识别系统。主要解决在缺少完整标注数据情况下，如何有效地进行跨模态（可见光与红外）的行人匹配问题。

## ✨ 核心特性

- **跨模态匹配聚合 (CMA)**: 实现了 Cross Modal Match Aggregation 机制，用于可见光和红外模态之间的特征匹配
- **两阶段训练策略**: 
  - Phase 1: 初始训练阶段（Stage 1）
  - Phase 2: 跨模态协作一致性学习阶段（Stage 2）
- **记忆库机制**: 使用动量更新的记忆库存储和更新模态特征
- **多数据集支持**: 支持 SYSU-MM01、RegDB、LLCM 三个主流数据集

## 📁 项目结构

```
VIReid/
├── main.py                 # 主训练/测试入口
├── wsl.py                  # 弱监督学习核心模块（CMA实现）
├── utils.py                # 工具函数
├── demo.py                 # 演示脚本
├── pre_process_sysu.py     # SYSU数据集预处理
├── models/                 # 模型定义
│   ├── __init__.py         # 模型创建和管理
│   ├── agw.py              # AGW网络结构
│   ├── classifier.py       # 分类器模块
│   ├── clip_model.py       # CLIP模型实现
│   ├── loss.py             # 损失函数
│   ├── optim.py            # 优化器配置
│   └── build_clip/         # CLIP构建模块
├── task/                   # 训练和测试任务
│   ├── train.py            # 训练逻辑
│   └── test.py             # 测试和评估
├── *.sh                    # 各数据集训练脚本
└── requirements.txt        # 依赖库
```

## 🔧 环境配置

### 依赖项

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch 2.0.1+cu118
- torchvision 0.15.2+cu118
- setproctitle 1.3.3
- tqdm 4.65.0

### 推荐环境
- Python 3.8+
- CUDA 11.8
- GPU 内存 >= 12GB

## 📊 支持的数据集

1. **SYSU-MM01**: 可见光-红外跨模态数据集
2. **RegDB**: 可见光-红外配对数据集
3. **LLCM**: 低光照跨模态数据集

## 🚀 使用方法

### 数据准备

1. 下载对应数据集并放置在 `./datasets/` 目录下
2. 对于SYSU数据集，需要先运行预处理脚本：

```bash
python pre_process_sysu.py
```

### 训练

**RegDB 数据集**:
```bash
bash regdb.sh
# 或
python main.py --dataset regdb --arch resnet --mode train \
    --lr 0.00045 --stage1-epoch 50 --stage2-epoch 120 \
    --batch-pidnum 5 --trial 1
```

**SYSU 数据集**:
```bash
bash sysu.sh
# 或
python main.py --dataset sysu --arch clip-resnet --mode train \
    --lr 0.0003 --stage1-epoch 20 --stage2-epoch 120 \
    --batch-pidnum 8
```

**LLCM 数据集**:
```bash
bash llcm.sh
# 或
python main.py --dataset llcm --arch resnet --mode train \
    --lr 0.0003 --stage1-epoch 80 --stage2-epoch 120 \
    --batch-pidnum 8
```

### 测试

```bash
python main.py --dataset regdb --mode test \
    --model-path /path/to/checkpoint.pth
```

## 🎛️ 主要参数说明

| 参数             | 默认值 | 说明                          |
| ---------------- | ------ | ----------------------------- |
| `--dataset`      | regdb  | 数据集选择: sysu, llcm, regdb |
| `--arch`         | resnet | 网络架构: resnet, clip-resnet |
| `--lr`           | 0.0003 | 学习率 (RegDB: 0.00045)       |
| `--stage1-epoch` | 20     | 第一阶段训练轮数              |
| `--stage2-epoch` | 120    | 第二阶段训练轮数              |
| `--batch-pidnum` | 8      | 每批次ID数量 (RegDB: 5)       |
| `--weak-weight`  | 0.25   | 弱监督损失权重                |
| `--tri-weight`   | 0.25   | 三元组损失权重                |
| `--sigma`        | 0.8    | 动量更新因子                  |
| `--temperature`  | 3      | Softmax温度参数               |

## 📈 核心方法

### 跨模态匹配聚合 (CMA)

`wsl.py` 中实现的 CMA 模块包含：

1. **记忆库**: 维护可见光和红外模态的特征记忆
2. **特征提取**: 从训练数据中提取并保存模态特征
3. **标签生成**: 基于跨模态相似度生成伪标签
4. **动量更新**: 使用动量机制更新记忆库

### 两阶段训练

- **Stage 1**: 使用初始标注数据进行预训练，建立基础特征表示
- **Stage 2**: 利用 CMA 生成的伪标签进行跨模态协作学习

## 📊 评估指标

- **Rank-1, Rank-10, Rank-20**: 不同排名的准确率
- **mAP**: 平均精度均值
- **mINP**: 平均逆负惩罚

## 📄 许可证

请遵循原论文和官方仓库的许可协议。

## 🙏 致谢

感谢原论文作者提供的优秀工作和官方代码实现：
- 论文: [arXiv:2507.12942](https://arxiv.org/abs/2507.12942)
- 官方仓库: [WSL-VIReID](https://github.com/KongLingqi2333/WSL-VIReID.git)

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。