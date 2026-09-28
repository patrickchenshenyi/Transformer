# Transformer — 《Attention Is All You Need》复刻

从零复刻 Vaswani et al. (2017) 的 Transformer 架构，在 **WMT14 英语 → 德语** 机器翻译任务上训练。

## 结果

| 数据集 | BLEU |
| :--- | :--- |
| WMT14 EN-DE validation | **36.56** |
| WMT14 EN-DE test | **41.11** |

> test BLEU 高于 validation 是因为 test 评估子集规模较小。

## 实现特性

- 完整 Encoder–Decoder Transformer（6 层 / 8 头 / `d_model=512` / `FFN=2048` / `dropout=0.1`）
- 混合精度训练（AMP）+ 梯度裁剪（`clip=1.0`）
- 多机多卡分布式训练（PyTorch DDP，`torchrun` 启动）
- 断点重训（epoch 级 checkpoint resume）
- AdamW + Cosine Warmup 学习率调度（`lr=3e-4`，`warmup=4000`）
- TensorBoard 训练监控
- 从头训练的 BPE 分词器（`tokenizers` 库）
- 推理 + BLEU 评估（`sacrebleu`）

## 目录结构

```
.
├── train.py                 # 训练 / 断点重训 / 推理 单文件入口
├── notebooks/
│   └── transformer.ipynb    # 交互式 notebook 版本
├── tokenizer/
│   └── wmt14_bpe.json       # WMT14 EN-DE BPE 词表
├── weights/                 # 训练好的权重（Git LFS）
│   ├── transformer_v11_1_best.pt
│   └── transformer_v11_4_best.pt
├── logs/                    # 训练 / 评估日志
├── environment.yaml         # conda 环境（GPU, CUDA 12.4）
└── requirements.txt         # pip 依赖
```

## 快速开始

### 1. 环境

```bash
# 方式 A：conda（含 PyTorch + CUDA 12.4）
conda env create -f environment.yaml
conda activate Transformer_GPU

# 方式 B：pip（PyTorch 需自行按 CUDA 版本安装）
pip install -r requirements.txt
```

### 2. 数据

训练数据（HuggingFace `datasets` 格式）需放到数据根目录，代码通过环境变量 `DATA_ROOT` 定位（缺省 `/mnt/oss`，PAI 挂载点）：

```bash
export DATA_ROOT=/path/to/this/repo
```

`DATA_ROOT` 下需满足：

```
$DATA_ROOT/
├── datasets/WMT14_DE_EN_Saved/    # load_from_disk 读取
└── tokenizer/wmt14_bpe.json       # 本仓库已包含
```

### 3. 训练

```bash
torchrun --nproc_per_node=1 train.py --mode train
```

### 4. 断点重训 / 推理

```bash
python train.py --mode resume    --checkpoint weights/transformer_v11_4_best.pt
python train.py --mode inference --checkpoint weights/transformer_v11_4_best.pt
```

## 权重

| 文件 | 说明 |
| :--- | :--- |
| `weights/transformer_v11_4_best.pt` | 最终轮次（v11 round 4）验证集最优权重 |
| `weights/transformer_v11_1_best.pt` | 早期轮次（v11 round 1）最优权重 |

权重通过 [Git LFS](https://git-lfs.com/) 托管，克隆后自动下载；若需要完整训练集与中间 checkpoint，可参考 `.gitignore` 中记录的来源重新获取。

## 说明

- 模型与训练脚本最初在阿里云 PAI DSW 上开发，数据存于 OSS。本仓库只保留代码 + 最终权重（LFS）。
- 历史迭代版本（v8 ~ v12）见本地归档目录（已 gitignore，确认后可删除）。

## 参考

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
