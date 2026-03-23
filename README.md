# Llama3-8B LoRA 情感文本分类

## 项目简介

本项目基于 **Llama3-8B 大语言模型**，结合 **LoRA（Low-Rank Adaptation）** 与 **FlashAttention** 技术，实现情感文本分类任务。

模型用于识别六类情绪：

* joy（喜悦）
* sadness（悲伤）
* anger（愤怒）
* fear（恐惧）
* love（爱）
* surprise（惊讶）

在该任务上，模型取得 **0.9262 的准确率**，优于 Bert、Roberta 等主流模型。

---

## 技术方案

### 模型
<div align="center">
<img src="fig1.png" width="300">
</div>
* 基座模型：Llama3-8B
* 微调方式：LoRA（参数高效微调）
* 注意力优化：FlashAttention V2

### 训练方法

* 指令微调（Supervised Fine-Tuning）
* 冻结原始模型参数，仅训练 LoRA 低秩矩阵
* 使用 FP16 提升显存利用率

---

## 项目结构

```text
.
├── config/              # 训练配置
├── data/                # 数据集
├── scripts/             # 训练脚本
├── src/                 # 核心代码
├── evaluation/          # 评估模块
├── tests/               # 测试代码
├── cache/               # 缓存文件
├── fig1.png             # 模型结构图
├── fig2.png             # LoRA 示意图
├── fig3.png             # 数据分布图
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 方法说明

### Llama3-8B

Llama3-8B 是 Meta 发布的大语言模型，具备较强的语义理解能力，适用于文本分类、对话等任务。

其特点：

* 参数规模：8B
* 上下文长度：8K
* 使用 GQA（Grouped-Query Attention）优化计算效率

---

### LoRA 微调

LoRA 通过引入低秩矩阵对模型进行微调：

* 不更新原始模型参数
* 显著降低显存占用
* 提高训练效率
* 推理阶段无额外开销
<div align="center">
<img src="fig2.png" width="400">
</div>
---

### FlashAttention

FlashAttention 用于优化 Transformer 注意力计算：

* 降低显存访问开销
* 提高计算效率
* 支持更大 batch

---

## 数据集

使用六分类情感文本数据集，类别包括：

* joy / sadness / anger / fear / love / surprise
<div align="center">
<img src="fig3.png" width="500">
</div>
可以看到，数据分布相对均衡，其中 joy 和 sadness 占比较高，surprise 较少。

---

## 实验设置

| 参数         | 数值   |
| ---------- | ---- |
| 优化器        | Adam |
| 学习率        | 5e-5 |
| Batch Size | 5    |
| Epochs     | 3    |
| LoRA Rank  | 8    |
| 梯度累积       | 4    |
| 最大长度       | 512  |

---

## 实验结果

| 模型               | Accuracy   |
| ---------------- | ---------- |
| Bert-Base        | 0.9063     |
| Bert-Large       | 0.9086     |
| Roberta-Base     | 0.9125     |
| Roberta-Large    | 0.9189     |
| Llama3-8B (LoRA) | **0.9262** |

---

## 运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

---

### 2. 训练模型

```bash
bash scripts/train.sh
```

（或根据 config 配置运行训练脚本）

---

### 3. 评估模型

```bash
python evaluation/eval.py
```

---

## 项目说明

本项目主要用于：

* 探索大语言模型在情感分类任务中的表现
* 验证 LoRA 在小样本场景下的有效性
* 对比传统 NLP 模型与大模型的性能差异

---

## License

Apache License 2.0
