# PF-MGCD: Part-Based Fine-Grained Multi-Granularity Cross-Modal Distillation for Visible-Infrared Person Re-Identification

> 🌈 This repository implements **PF-MGCD**: a new state-of-the-art method for weakly-supervised visible-infrared person re-identification, based on multi-part memory and fine-grained graph distillation.

---

## 📖 目录 Contents

- [项目简介 Introduction](#项目简介-introduction)
- [方法创新 Highlights](#方法创新-highlights)
- [模型结构架构 Model Architecture](#模型结构架构-model-architecture)
- [项目文件结构 File Structure](#项目文件结构-file-structure)
- [环境与依赖 Requirements](#环境与依赖-requirements)
- [数据准备 Dataset Preparation](#数据准备-dataset-preparation)
- [训练与测试 Training & Testing](#训练与测试-training--testing)
- [参考/致谢 Reference & Acknowledgement](#参考致谢-reference--acknowledgement)
- [交流与贡献 Contact & Contribution](#交流与贡献-contact--contribution)

---

## 项目简介 Introduction

本仓库为视觉-红外行人重识别领域提出了一种**全新弱监督框架——PF-MGCD**，旨在解决现实中的标签不完备（仅有单模态ID/标签，无配对或无跨模态身份标注）情况下的跨模态检索难题。相较于旧版WSL-VIReID，本实现采用了多粒度记忆库和细粒度图传播等更为强大的跨模态协作机制。

This repo provides a modular PyTorch implementation of PF-MGCD for VI-ReID, supports three major datasets (SYSU-MM01, RegDB, LLCM), and is ready for academic or industrial cross-modality re-ID applications.

---

## 方法创新 Highlights

- **三分支非对称协同架构**：冻结Teacher分支初始化多粒度记忆库，Student分支联合图传播学习跨模态关联，极大提升弱监督场景下的泛化能力。
- **多粒度模态无关记忆库**：按部件/人体区域存储全身份“纯净”原型，提升跨模态特征的判别与鲁棒性。
- **ISG-DM无参解耦模块**：统计实例均值/方差提取风格特征，Instance Normalization + SE-Gate获取纯身份特征，特征正交，天然利于模态分解。
- **Fine-Grained Graph Distillation（图蒸馏）**：高置信度top-K记忆邻居指导软标签生成，提升无标签情况下的判别力。
- **自适应损失权重与两阶段训练**：Warmup+自适应熵加权损失，进一步优化跨模态对齐。

---

## 模型结构架构 Model Architecture

```
       ┌─────────────┐     ┌─────────┐      ┌─────────────┐
Input─▶│ PCB Backbone├─K─▶ │ ISG-DM  ├─K─▶  │ Multi-Part  │
Image  │(ResNet50)   │     │模块     │      │ Memory Bank │
       └─────────────┘     └─────────┘      └─────┬───────┘
                                                    │
        　　　　　　         　　　　　　          ▼
                  ┌─────────────────────────────┐
                  │Graph Propagation & Distill. │
                  └─────────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │ ID/Modality/Orth-Loss│
                    └─────────────────────┘
```

- **输入图像** → PCB切分（6个水平part）→ 每个part送ISG-DM提取纯身份/模态特征 → 建立K×N×256记忆库 → 通过图传播聚合记忆Top-K软标签，辅助损失监督。
- 核心损失：
  - 多粒度ID损失
  - 图蒸馏损失
  - 特征正交损失
  - 模态判别损失

---

## 项目文件结构 File Structure

```
vireid/
├── datasets/                  # 原始&适配后的数据加载模块
│   ├── sysu.py               # SYSU-MM01加载
│   ├── regdb.py              # RegDB加载
│   ├── llcm.py               # LLCM加载
│   ├── data_process.py       # 数据增强与变换
│   └── dataloader_adapter.py # 适配PF-MGCD统一数据流
├── models/                   # 所有模型核心模块
│   ├── pcb_backbone.py       # PCB骨干网络
│   ├── isg_dm.py             # ISG-DM解耦模块
│   ├── memory_bank.py        # 多粒度记忆库
│   ├── graph_propagation.py  # 细粒度图传播
│   ├── pfmgcd_model.py       # PF-MGCD主模型
│   ├── loss.py               # 各类损失
│   └── teacher_network.py    # Teacher分支
├── task/
│   ├── train.py              # 训练流程
│   └── test.py               # 测试流程（支持跨模态检索评估）
├── main.py                   # 主入口，参数与训练/测试控制
├── utils.py                  # 工具函数
├── requirements.txt          # 依赖包
├── sysu.sh/regdb.sh/llcm.sh  # 各数据集训练脚本
└── checkpoints/              # 权重保存
```

---

## 环境与依赖 Requirements

- Python 3.8+
- PyTorch 1.10+ (GPU建议)
- torchvision >=0.13
- numpy, pillow, tqdm, matplotlib 等
- 推荐环境：
  - CUDA 11.8+
  - GPU 显存⩾12G

安装依赖：
```bash
conda create -n pfmgcd python=3.8
conda activate pfmgcd
pip install torch torchvision tqdm numpy pillow matplotlib
# 或
pip install -r requirements.txt
```

---

## 数据准备 Dataset Preparation

请参考各数据集的官方说明，下载并放置到`datasets/`目录下：
- SYSU-MM01: `datasets/SYSU-MM01/`
- RegDB: `datasets/RegDB/`
- LLCM: `datasets/LLCM/`

目录下需包含官方数据划分txt或pkl文件。对于SYSU可选预处理`python pre_process_sysu.py`。

---
## 训练与测试 Training & Testing

### 训练（以 SYSU 为例）

```bash
bash sysu.sh
```
或直接命令行:
```bash
python main.py \
    --dataset sysu \
    --data-path ./datasets \
    --mode train \
    --num-parts 6 \
    --feature-dim 256 \
    --memory-momentum 0.9 \
    --batch-size 32 \
    --lr 0.0003 \
    --total-epoch 120 \
    --warmup-epochs 10 \
    ...（更多参数见sh脚本和main.py）
```

### 测试

```bash
python main.py \
    --mode test \
    --dataset sysu \
    --model-path ./checkpoints/sysu/pfmgcd_best.pth \
    --pool-parts
```

### 主要参数解释

- `--num-parts`: PCB切分部位数，建议6
- `--feature-dim`: 解耦后部件特征维度
- `--batch-size`, `--pid-numsample`, `--batch-pidnum`: 训练采样
- `--memory-momentum`: 记忆更新动量
- 损失相关: `--lambda-graph`, `--lambda-orth`, `--lambda-mod`
- `--relabel`: 是否打乱ID标签，提升弱监督泛化

---

## 评价指标 Evaluation

- Rank-1, Rank-5, Rank-10, Rank-20 准确率
- mAP (mean Average Precision)
- mINP (mean Inverse Negative Penalty)

---

## 参考/致谢 Reference & Acknowledgement

本仓库实现部分借鉴了如下项目与论文：

- KongLingqi2333/WSL-VIReID: [WSL-VIReID Code](https://github.com/KongLingqi2333/WSL-VIReID.git)
- SYSU-MM01, RegDB, LLCM datasets
- 论文: "Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning", arxiv:2507.12942

如使用本项目，请引用原论文和本仓库。

---

## 交流与贡献 Contact & Contribution

如有Bug反馈、算法交流、技术需求、或希望贡献代码，请[提交Issue](https://github.com/yyykkklll/VIReid/issues)，或邮箱联系 qlu.ykelong@gmail.com。

We welcome contributions! Pull requests or questions are warmly invited.

---

<div align="center"><b>🚀PF-MGCD: Towards Practical Cross-Modality Person ReID under Weak Supervision 🚀</b></div>
