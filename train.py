# %% [markdown]
# ### v5
# - 增加了混合精度训练
# 
# ### v6
# - 适配了pai dsw 挂载oss 的训练场景
# 
# ### v7
# - 支持 Data Parallel (单机多卡，分配不均)
# 
# ### v8
# - 取消 data parallel, 支持 distributed data parallel
# 
# ### v9
# - tensorboard
# 
# ### v10
# - 断点重训
# 
# ### v11
# - 用 AdamW + Cosine Warmup 代替 naom scheduler
# 
# ### v12
# - inference

# %% [markdown]
# 启动命令：
# 
# pip install -r requirements.txt
# torchrun --nproc_per_node=1 train.py --mode train

# %%

import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse

# data
from datasets import load_from_disk
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# model
import torch
import torch.nn as nn
import torch.nn.functional as F

# train
from torch import optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LinearLR
from transformers import get_cosine_schedule_with_warmup
from torch.amp import autocast, GradScaler

# 分布式训练
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 模型评估
import sacrebleu

# 训练监控
from torch.utils.tensorboard import SummaryWriter


# %%
VERSION = 11
ROUND = 4

# data
NUM_WORKERS = 2     # 0 for ipynb, >0 for accelaration
BATCH_SIZE = 160     # batch size for each GPU; A10 24GB;     

# 数据/模型根目录：默认 /mnt/oss 适配 PAI DSW 的 OSS 挂载点；
# 本地运行时用环境变量 DATA_ROOT 覆盖，例如 export DATA_ROOT=/path/to/this/repo
OSS_MOUNT_PATH = os.environ.get("DATA_ROOT", "/mnt/oss")
DATASET_DIR = os.path.join(OSS_MOUNT_PATH, "datasets", "WMT14_DE_EN_Saved")
BPE_PATH = os.path.join(OSS_MOUNT_PATH, "tokenizer", "wmt14_bpe.json")

# model
NUM_BLOCKS = 6
D_MODEL = 512           # hidden_dim
FF_DIM = 2048
NUM_HEADS = 8
DROP_OUT = 0.1

MAX_LEN_EN = 1000       # encoder 最长位置编码
MAX_LEN_DE = 1000       # decoder 最长解码长度
PAD_idx = 0
SOS_idx = 1
EOS_idx = 2

# train
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
NEW_CUR_LEARNING_RATE = None
CLIP_GRAD = 1.0          
LOG_INTERVAL = 100       # 每隔多少batch打印一次损失
WARMUP_STEPS = 4000

SAVE_DIR = os.path.join(OSS_MOUNT_PATH, "models", "Transformer", "models")
SAVE_PATH = os.path.join(SAVE_DIR, f"transformer_v{VERSION}_{ROUND}_best.pt")
CHECKPOINT_DIR = os.path.join(OSS_MOUNT_PATH, "models", "Transformer", "checkpoints", f"v{VERSION}_{ROUND}")
LOG_DIR = os.path.join(OSS_MOUNT_PATH, "models", "Transformer", "logs", f"v{VERSION}_{ROUND}")
NUM_EPOCHS_PER_LOG = 1  # 每隔多少epoch记录一次checkpoint


os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# %% [markdown]
# ### Part 1. 准备数据

# %% [markdown]
# #### Part 1.1. 下载/加载 数据集

# %%
def load_dataset():
    print(f"从本地加载数据集")
    dataset = load_from_disk(DATASET_DIR)
    dataset_train = dataset["train"]
    dataset_valid = dataset["validation"]

    print(f"训练集大小: {len(dataset_train)}")  # 应该是约 450 万
    print(f"验证集大小: {len(dataset_valid)}")  # 应该是约 3000

    return dataset_train, dataset_valid

# %% [markdown]
# #### Part 1.2. 训练/加载 tokenizer

# %%
def get_tokenizer(dataset_train = None):
    if not os.path.exists(BPE_PATH):  # ~5 min
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=37000,               # 常用大小
            special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],    # 加入特殊标记
            min_frequency=2,
        )

        def batch_iterator():
            for example in dataset_train:
                yield example['translation']['en']   # 源语言
                yield example['translation']['de']   # 目标语言
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

        # 保存 tokenizer 供后续使用
        tokenizer.save(BPE_PATH)  
        
    else:
        tokenizer = Tokenizer.from_file(BPE_PATH)
    
    return tokenizer

# %% [markdown]
# #### Part 1.3. 封装为 torch.utils.data.Dataset 类

# %%
class WMT14Dataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=None):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sos_id = tokenizer.token_to_id("<sos>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        self.pad_id = tokenizer.token_to_id("<pad>")

    # 加载时 tokenize & 添加首尾标记
    def __getitem__(self, idx):
        item = self.data[idx]['translation']
        src_text = item['en']
        tgt_text = item['de']

        src_ids = self.tokenizer.encode(src_text).ids
        tgt_ids = self.tokenizer.encode(tgt_text).ids
        
        if self.max_len is not None:    
            src_ids = src_ids[:self.max_len-2]
            tgt_ids = tgt_ids[:self.max_len-2]

        src = [self.sos_id] + src_ids + [self.eos_id]
        tgt = [self.sos_id] + tgt_ids + [self.eos_id]
        return {
            'src': torch.tensor(src, dtype=torch.long),
            'tgt': torch.tensor(tgt, dtype=torch.long),
            'src_len': len(src),
            'tgt_len': len(tgt)
        }

    def __len__(self):
        return len(self.data)

# %%
# class BucketBatchSampler(Sampler):
#     """
#     将长度相近的样本放入同一个batch
#     """
#     def __init__(self, lengths, batch_size, shuffle=True, bucket_boundaries=None):
#         """
#         Args:
#             lengths: 所有样本的长度列表 [len1, len2, ...]
#             batch_size: batch大小
#             shuffle: 是否打乱
#             bucket_boundaries: bucket的边界，如 [10, 20, 30, 40, 50]
#         """
#         self.lengths = np.array(lengths)
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.bucket_num = (len(self.lengths) + self.batch_size - 1) // self.batch_size  # 实际可能不会正好分成这么多
        
#         if bucket_boundaries is None:
#             max_len = int(np.percentile(lengths, 95))  # 95%分位数
#             self.bucket_boundaries = list(range(10, max_len + 10, 10))
#         else:
#             self.bucket_boundaries = bucket_boundaries
            
#         # 为每个样本分配bucket ID
#         self.bucket_ids = np.digitize(self.lengths, self.bucket_boundaries)
        
#     def __iter__(self):
#         # 按bucket组织样本索引
#         buckets = defaultdict(list)
#         for idx, bucket_id in enumerate(self.bucket_ids):
#             buckets[bucket_id].append(idx)
        
#         # 在每个bucket内打乱
#         if self.shuffle:
#             for bucket_id in buckets:
#                 np.random.shuffle(buckets[bucket_id])
        
#         batches = []
#         for bucket_id in sorted(buckets.keys()):
#             bucket = buckets[bucket_id]
#             # 将bucket分成多个batch
#             for i in range(0, len(bucket), self.batch_size):
#                 batch = bucket[i:i + self.batch_size]
#                 if len(batch) == self.batch_size or i + self.batch_size >= len(bucket):
#                     batches.append(batch)
#         self.bucket_num = len(batches)

#         # 全局打乱batch顺序
#         if self.shuffle:
#             np.random.shuffle(batches)
        
#         for batch in batches:
#             yield batch
    
#     def __len__(self):
#         return self.bucket_num

# %% [markdown]
# #### Part 1.4 封装为 torch.utils.data.DataLoader 批处理加载器

# %%
def collate_fn(batch):
    src_list = [item['src'] for item in batch]
    tgt_list = [item['tgt'] for item in batch]
    src_lens = torch.tensor([item['src_len'] for item in batch])
    tgt_lens = torch.tensor([item['tgt_len'] for item in batch])

    # 用 0 (PAD) 填充
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=0)

    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_len': src_lens,
        'tgt_len': tgt_lens
    }

# %%
def create_dataloaders(rank, world_size, dataset_train_processed, dataset_valid_processed):
    # 分布式 sampler
    train_sampler = DistributedSampler(
        dataset_train_processed,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    # 取消 ：BucketBatchSampler，因为不能和DistributedSampler同时使用
    # 使用 ：DistributedSampler + batch_size
    train_loader = DataLoader(
        dataset_train_processed,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True  # 确保所有 GPU 批次大小一致
    )
    
    # 验证集不需要分布式 sampler
    valid_loader = DataLoader(
        dataset_valid_processed,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, valid_loader, train_sampler

# %%
# 完整封装 数据加载、预处理、批处理 过程
def get_dataloaders(rank, world_size, dataset_train, dataset_valid, tokenizer):
    dataset_train_processed = WMT14Dataset(dataset_train, tokenizer, max_len=100)
    dataset_valid_processed = WMT14Dataset(dataset_valid, tokenizer, max_len=100)
    train_loader, valid_loader, train_sampler = create_dataloaders(rank, world_size, dataset_train_processed, dataset_valid_processed)
    return train_loader, valid_loader, train_sampler

# %% [markdown]
# ### Part 2 核心组件
# #### Part 2.1. Encoder部分
# - Embedding
# - Positional Encoding
# 
# - Pre Layer Norm
# - Multi-Head Attention + Padding Mask
# - Residual Connection
# - FeedForward
# 
# #### Part 2.2. Decoder部分
# - Embedding
# - Positional Encoding
# 
# - Pre Layer Norm
# - Multi-Head Self Attention + Causal Mask (1st) 
# - Multi-Head Cross Attention + Padding Mask(2nd)
# - Residual Connection
# - FeedForward
# 
# - Linear Output
# - Softmax
# 
# #### Part 2.3. 编码器-解码器架构
# - Encoder堆叠
# - Decoder 堆叠
# - Encoder-Decoder组合
# - Linear + Softmax
# - Next token generator
# 

# %% [markdown]
# #### Part 2.1. Encoder
# 

# %%
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        """
        Class Attribute:
            pe : [1, max_len, d_model]
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pos = torch.arange(max_len).unsqueeze(1)    # [max_len, 1]
        div_term = 1/ torch.pow(10000, torch.arange(0, d_model, 2) / d_model)    # [d_model / 2,]
        pe = torch.zeros([max_len, d_model])
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)    
        pe = pe.unsqueeze(0)            # [1, max_len. d_model]
        self.register_buffer("pe", pe)  # 1. 注册为模型状态但不参与训练 # 2. 随模型保存和加载
        
    def forward(self, x):
        """
        Input
            x : [batch, seq, d_model]
        """
        assert x.shape[1] <= self.max_len, "输入长度大于最大位置编码长度"
        return x + self.pe[:, 0:x.shape[1]]


# %%
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout = 0.1):
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        super().__init__()
        self.hidden_dim = hidden_dim    # hidden_dim = d_model
        self.num_heads = num_heads

        # 投影矩阵
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, Q, K, V, mask = None):
        """
        Input:
            Q, K, V : [batch_size, seq_len, hidden_dim]
            mask : broadcastable to [batch_size, num_heads, seq_len, seq_len]; to be masked 的位置为True

        Output:
            outputs : [batch, seq, hiddden_dim]
            weights : [batch, heads, seq, seq]
        """
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        batch_size, seq_len, _ = Q.shape
        Q = Q.reshape([batch_size, seq_len, self.num_heads, -1])  # [batch, seq, heads, d_k]
        K = K.reshape([batch_size, seq_len, self.num_heads, -1])
        V = V.reshape([batch_size, seq_len, self.num_heads, -1])

        Q = Q.permute([0, 2, 1, 3])   # [batch, heads, seq, d_k]
        K = K.permute([0, 2, 1, 3])
        V = V.permute([0, 2, 1, 3])

        # 计算注意力得分
        scale_factor = np.sqrt(self.hidden_dim // self.num_heads)
        attn_weights = torch.matmul(Q, torch.permute(K, [0, 1, 3, 2])) / scale_factor   # [batch, heads, seq_len, seq_len]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim = -1) # [batch, heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)

        # 计算输出张量
        attn_outputs = torch.matmul(attn_weights, V)    # [batch, heads, seq_len, d_v]
        attn_outputs = torch.permute(attn_outputs, [0, 2, 1, 3])    # [batch, seq_len, heads, d_v]
        attn_outputs = torch.reshape(attn_outputs, [batch_size, seq_len, -1])
        attn_outputs = self.W_o(attn_outputs)   

        return attn_outputs, attn_weights



# %%
class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout = 0.1):
        super().__init__()
        # self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm1 = nn.LayerNorm([hidden_dim,])
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.layer_norm2 = nn.LayerNorm([hidden_dim,])
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.act_func = F.relu
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask):
        """
        Input:
            X : [batch_size, seq_len, hidden_dim]

        Output:
            X : [batch_size, seq_len, hidden_dim]
        """
        X_norm = self.layer_norm1(X)    # pre-norm
        attn_outputs, attn_weights = self.attention(X_norm, X_norm, X_norm, mask)   
        attn_outputs = self.dropout(attn_outputs)
        X = X + attn_outputs            # residual connection

        X_norm = self.layer_norm2(X)    # pre-norm
        ff_outputs = self.linear2(self.act_func(self.linear1(X_norm)))
        ff_outputs = self.dropout(ff_outputs)
        X = X + ff_outputs              # residual connection

        return X
        

# %%
class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks, vocab_size, hidden_dim, num_heads, ff_dim, max_len, dropout = 0.1):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding_layer = SinusoidalPositionalEncoding(hidden_dim, max_len)
        self.transformer_blocks = nn.ModuleList(
            [EncoderBlock(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, X, padding_idx = 0):  
        """
        Input:
            X : [batch_size, seq_len]
        
        Output:
            X : [batch, seq, hidden_dim]
        """
        # Key侧padding_mask
        padding_mask = (X == padding_idx)           # True 是 padding
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
        
        X_embed = self.embedding_layer(X)           # [batch, seq, hidden_dim]
        X_embed = self.pos_encoding_layer(X_embed)  # [batch, seq, hidden_dim]
        X_embed = self.dropout(X_embed)
        
        for block in self.transformer_blocks:
            X_embed = block(X_embed, padding_mask)
        
        return X_embed



# %% [markdown]
# #### Part 2.2. Decoder

# %%
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, padding_mask = None):
        """
        Input: 
            Q : [batch, seq_len_de, hidden_dim]; Decoder Input
            K : [batch, seq_len_en, hidden_dim]; Encoder Output
            V : [batch, seq_len_en, hidden_dim]; Encoder Output
            padding_mask : broadcastable to [batch, num_heads, seq_len_de, seq_len_en]; to be masked 位置为True
        """
        # 投影
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # 分头: shape = [batch, seq, heads, d_k]
        batch_size = Q.shape[0]
        Q = Q.reshape([batch_size, Q.shape[1], self.num_heads, self.d_k])
        K = K.reshape([batch_size, K.shape[1], self.num_heads, self.d_k])
        V = V.reshape([batch_size, V.shape[1], self.num_heads, self.d_k])

        # 转置: shape = [batch, heads, seq, d_k]
        Q = Q.permute([0, 2, 1, 3])
        K = K.permute([0, 2, 1, 3])
        V = V.permute([0, 2, 1, 3])

        # 注意力得分
        attn_weights = torch.matmul(Q, K.permute([0, 1, 3, 2])) / np.sqrt(self.d_k) # [batch, heads, seq_q, seq_k]
        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(padding_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim = -1)
        attn_weights = self.dropout(attn_weights)

        # 注意力输出
        attn_output = torch.matmul(attn_weights, V)     # [batch, heads, seq_q, d_k]
        attn_output = attn_output.permute([0, 2, 1, 3]) # [batch, seq_q, heads, d_k]
        attn_output = attn_output.reshape([batch_size, attn_output.shape[1], -1])   # [batch, seq_q, hidden_dim]
        attn_output = self.W_o(attn_output)             # [batch, seq_q, hidden_dim]

        return attn_output, attn_weights
       


# %%
class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attention_1 = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.layer_norm_1 = nn.LayerNorm([hidden_dim,])
        self.attention_2 = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
        self.layer_norm_2 = nn.LayerNorm([hidden_dim,])
        self.feed_forward_1 = nn.Linear(hidden_dim, ff_dim)
        self.feed_forward_2 = nn.Linear(ff_dim, hidden_dim)
        self.layer_norm_3 = nn.LayerNorm([hidden_dim,])
        self.act_func = F.relu
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoder_out, decoder_in, padding_mask = None, causal_mask = None):
        '''
        intput:
            decoder_in : [batch, seq_len_de, hidden]
            encoder_out : [batch, seq_len_en, hidden]
            padding_mask : broadcastable to [batch, num_heads, seq_len_de, seq_len_en]
            causal_mask : broadcastable to [batch, num_heads, seq_len_de, seq_len_de]
        output:
            [batch, seq_len_de, hidden]
        '''
        X = decoder_in
        X_norm = self.layer_norm_1(decoder_in)  # pre-norm
        X_attn, _ = self.attention_1(X_norm, X_norm, X_norm, causal_mask)
        X_attn = self.dropout(X_attn)
        X = X_attn + X                          # residual connection

        X_norm = self.layer_norm_2(X)           # pre-norm
        X_attn, _ = self.attention_2(X_norm, encoder_out, encoder_out, padding_mask) # [batch, seq_len_de, hidden]
        X_attn = self.dropout(X_attn)
        X = X_attn + X                          # residual connection

        X_norm = self.layer_norm_3(X)
        X_ff = self.act_func(self.feed_forward_1(X_norm))
        X_ff = self.feed_forward_2(X_ff)
        X_ff = self.dropout(X_ff)
        X = X_ff + X
        return X
        

# %%
def create_padding_mask(encoder_out_len, seq_len_en, device = "cpu"):
    """
    Args:
        encoder_out_len: [batch_size,]，每个元素是序列长度 (可以是list或tensor)
        seq_len_en : encoder_output 中padding后的统一长度
    
    Returns:
        padding_mask: [batch, 1, 1, seq_len_en] 需要pad的位置为True
    """
    
    batch_size = encoder_out_len.shape[0]
    
    pos = torch.arange(seq_len_en, device=device)  # [seq_len_en]
    pos = pos.unsqueeze(0).expand(batch_size, -1)  # [batch, seq_len_en]
    valid_pos = encoder_out_len.reshape(-1, 1)     # [batch, 1]
    padding_mask = pos >= valid_pos                # [batch, seq_len_en], 需要pad的位置为True
    
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
    return padding_mask  # [batch, 1, 1, seq_len_en]

# %%

def create_causal_mask(seq_len_de, device='cpu'):
    '''
    Creates a causal mask for Transformer decoders relying on PyTorch broadcasting.
    
    Args:
        seq_len_de: The length of the decoder sequence.
        device: The device to create the tensor on (should match your attention scores).
        
    Returns:
        causal_mask: [1, 1, seq_len_de, seq_len_de] boolean tensor. 
                     True indicates future positions that must be masked.
    '''
    causal_mask = torch.triu(
        torch.ones((seq_len_de, seq_len_de), dtype=torch.bool, device=device), 
        diagonal=1
    )  # [seq_len, seq_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]
    
    return causal_mask

# %%
class TransformerDecoder(nn.Module):
    def __init__(self, num_blocks, vocab_size, hidden_dim, num_heads, ff_dim, max_len_de=MAX_LEN_DE, dropout = 0.1):
        super().__init__()
        self.num_heads = num_heads

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len = max_len_de)
        self.decoderBlocks = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)
        ])
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_out, decoder_in, encoder_out_len = None):
        """
        decoder_in : [batch, seq_len_de]
        encoder_outputs : [batch, seq_len_en, hidden_dim]
        encoder_out_len : [batch, ]; 
        """

        # padding mask
        seq_len_en = encoder_out.shape[1]
        padding_mask = create_padding_mask(encoder_out_len, seq_len_en, device=encoder_out.device)     # [batch, 1, seq_len_en, 1]

        # causal mask
        seq_len_de = decoder_in.shape[1]
        causal_mask = create_causal_mask(seq_len_de, device=decoder_in.device)

        # decode
        decoder_in_embed = self.embedding(decoder_in)             # [batch, seq_len_de, hidden]
        decoder_in_embed = self.pos_encoder(decoder_in_embed)
        decoder_in_embed = self.dropout(decoder_in_embed)
        for block in self.decoderBlocks:
            decoder_in_embed = block(encoder_out, decoder_in_embed, padding_mask, causal_mask)     
        logits = self.linear(decoder_in_embed)                   
        return logits                       #[batch, seq_len_de, vocab_size]              



# %% [markdown]
# #### Part 2.3. encoder-decoder

# %%
class Transformer(nn.Module):
    def __init__(self, num_blocks, vocab_size, hidden_dim, num_heads, ff_dim, max_len_en=MAX_LEN_EN, max_len_de=MAX_LEN_DE, dropout = 0.1, **kwargs):
        super().__init__()
        self.max_len_de = max_len_de

        self.encoder = TransformerEncoder(num_blocks, vocab_size, hidden_dim, num_heads, ff_dim, max_len_en, dropout = dropout)
        self.decoder = TransformerDecoder(num_blocks, vocab_size, hidden_dim, num_heads, ff_dim, max_len_de, dropout = dropout)

        self.apply(self._init_weights)      # 权重初始化

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, src, src_len, tgt=None, device = None):
        """
        Args:
            src : [batch, seq_len_src]
            src_len : [batch, ]
            tgt : [batch, seq_len_tgt] (Expected format: [SOS, t1, t2, ..., EOS, PAD, PAD])
        """
        batch_size = src.shape[0]
        if device is None:
            device = src.device
        
        # Step 1: Encoding
        encoder_out = self.encoder(src, PAD_idx)    # [batch, seq_src, hidden]

        # Step 2: Decoding
        if tgt is not None:     
            # --- TEACHER-FORCING MODE ---

            # 虽然decoder只作了一次解码，但因为causal mask，实际等价于:
            # 对decoder_in = tgt[:, :1], tgt[:, :2], ...,tgt[:, :-1],都作一次解码，并取最后一位logits[:, -1, :]用于eval
            decoder_in = tgt[:, :-1]    
            logits = self.decoder(encoder_out, decoder_in, src_len) 

            return logits   # [batch, seq_len_tgt - 1, vocab_size], compute loss against tgt[:, 1:] outside this class

        else:                   
            # --- INFERENCE MODE ---
            target_len = self.max_len_de

            decoder_in = torch.full([batch_size, 1], SOS_idx, dtype=torch.long, device=device)    
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device) # [batch]
            
            for i in range(target_len):
                logits = self.decoder(encoder_out, decoder_in, src_len)   # [batch, current_seq_len, vocab_size]
                next_token_logits = logits[:, -1, :]  # [batch, vocab_size], 取最后一个时间步
                _, next_token = next_token_logits.topk(1, dim=-1)   # [batch, 1]
                next_token = next_token.squeeze(-1)                 # [batch]

                next_token = torch.where(finished, torch.tensor(PAD_idx, device=device), next_token) # Force padding if already emitted EOS
                finished = finished | (next_token == EOS_idx)       # tensor bitwise OR (|)
                
                decoder_in = torch.cat([decoder_in, next_token.unsqueeze(1)], dim=1)

                if finished.all():  # Early stopping if all sequences hit EOS
                    break
            return decoder_in   # [batch, max_len_de]
        


# %% [markdown]
# ### Part 3 Training 

# %% [markdown]
# #### Part 3.1 Train one epoch

# %%
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler, scheduler, writer, rank, global_step):
    dist.barrier()
    world_size = dist.get_world_size()  # 获取 GPU 数量
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, batch in enumerate(dataloader):
        src = batch['src'].to(device, non_blocking=True)
        tgt = batch['tgt'].to(device, non_blocking=True)
        src_len = batch['src_len'].to(device, non_blocking=True)
        
        # 1. 前向传播
        with autocast(device_type='cuda'):      # 混合精度
            logits = model(src, src_len, tgt)
            targets = tgt[:, 1:].contiguous().view(-1)
            logits_flat = logits.contiguous().view(-1, logits.shape[-1])
            loss = criterion(logits_flat, targets)
            
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
        
        # 2. 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        
        # 3. 记录训练日志 
        global_step += world_size
        if rank == 0 and writer:
            if batch_idx % LOG_INTERVAL == 0:
                current_lr = optimizer.param_groups[0]['lr']
                # 计算当前的平均 loss (近似)
                avg_loss = total_loss / max(total_tokens, 1)
                
                # TensorBoard 记录日志
                writer.add_scalar('Loss/Training_Batch', loss.item(), global_step)
                writer.add_scalar('Loss/Training_Epoch_Avg', avg_loss, global_step)
                writer.add_scalar('Params/Learning_Rate', current_lr, global_step)
                
                # terminal 打印日志
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                      f"LR: {current_lr:.6f} | GPU: {gpu_memory:.2f}GB")
    
    # 返回当前进程的 loss（在主函数中汇总）
    epoch_loss = total_loss / max(total_tokens, 1)
    epoch_ppl = torch.exp(torch.tensor(epoch_loss)).item()
    return epoch_loss, epoch_ppl, global_step

# %% [markdown]
# #### Part 3.2 evaluation on valid set

# %%
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.inference_mode():
        for batch in dataloader:
            src = batch['src'].to(device, non_blocking=True)
            tgt = batch['tgt'].to(device, non_blocking=True)
            src_len = batch['src_len'].to(device, non_blocking=True)
            
            logits = model(src, src_len, tgt)
            targets = tgt[:, 1:].contiguous().view(-1)
            logits_flat = logits.contiguous().view(-1, logits.shape[-1])
            loss = criterion(logits_flat, targets)
            
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl

# %% [markdown]
# #### Part 3.3 主训练循环

# %%
def main_worker(rank, world_size, local_rank, device, resume_from = None):

    # 初始化 TensorBoard
    writer = None
    if rank == 0:
        # 加上时间戳或任务ID，避免覆盖旧日志
        log_dir = os.path.join(LOG_DIR, f"runs_{VERSION}_{ROUND}", datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
    

    # 1. 数据加载
    dataset_train, dataset_valid = load_dataset()
    if rank == 0:   # 加载 Tokenizer (仅 Rank 0 加载)
        tokenizer = get_tokenizer(dataset_train)
    dist.barrier() # 等待 rank 0 完成
    tokenizer = Tokenizer.from_file(BPE_PATH)    # 所有进程再次加载
    train_loader, valid_loader, train_sampler= get_dataloaders(rank, world_size, dataset_train, dataset_valid, tokenizer)
    if rank == 0:
        print("step 1 : 数据加载 finished")
    

    # 2. 创建模型
    # DDP 下必须在 ms.spawn()后进行
    vocab_size = tokenizer.get_vocab_size()
    model = Transformer(
        num_blocks=NUM_BLOCKS,
        vocab_size=vocab_size,
        hidden_dim=D_MODEL,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        max_len_en=MAX_LEN_EN,
        max_len_de=MAX_LEN_DE,
        dropout=DROP_OUT
    ).to(device)
    
    # 用于保存checkpoint & best model
    model_config = {        
        'num_blocks': NUM_BLOCKS,
        'vocab_size': vocab_size,
        'hidden_dim': D_MODEL,
        'num_heads': NUM_HEADS,
        'ff_dim': FF_DIM,
        'max_len_en': MAX_LEN_EN,
        'max_len_de': MAX_LEN_DE,
        'dropout': DROP_OUT,
        'pad_idx': PAD_idx,
        'sos_idx': SOS_idx,
        'eos_idx': EOS_idx,
    }
    if rank == 0:
        print("step 2 : 原始模型创建 finished")


    # 3. 从断点恢复
    start_epoch = 1
    global_step = 0
    best_val_loss = float('inf')
    if resume_from:
        if rank == 0:
            print(f"step 2 : 从 {resume_from} 加载模型...")

        # 3.1 加载模型权重
        # 此时 model 还是原始模型，没有用DDP包装，checkpoint 中的 key 不应带 'module.'
        # 如果之前的 checkpoint 是错误地带着 'module.' 保存的，这里需要清洗 key
        checkpoint = torch.load(resume_from, map_location=f'cuda:{local_rank}')
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 去除可能存在的 module. 前缀
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)   # 加载原始模型

        # 3.2 加载优化器和调度器状态
        optimizer_state_to_load = checkpoint.get('optimizer_state_dict')
        scheduler_state_to_load = checkpoint.get('scheduler_state_dict')
        
        # 3.4 恢复训练计数
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        steps_per_epoch = len(train_loader)
        global_step = (start_epoch - 1) * steps_per_epoch * world_size  # 假设每个 epoch 有 steps_per_epoch 个 batch
        
        if rank == 0:
            print(f"step 2 : 已加载 epoch {checkpoint['epoch']} ")
    
    # 4. 定义训练参数：loss, optim, scheduler, scaler
    # 包装 model，创建 optim, scheduler 必须在加载权重之后!
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_idx, label_smoothing=0.1)
    total_steps = len(train_loader) * NUM_EPOCHS 
    warmup_steps = int(0.1 * total_steps) 
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = GradScaler('cuda')     # 混合精度

    if rank == 0:
        print(f"step 3 : 初始化训练组件 finished")
    
    # 5. 从checkpoint恢复训练状态
    if resume_from:
        if optimizer_state_to_load is not None:
            optimizer.load_state_dict(optimizer_state_to_load)
        
        if scheduler_state_to_load:
            scheduler.load_state_dict(scheduler_state_to_load)
            
        if NEW_CUR_LEARNING_RATE is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = NEW_CUR_LEARNING_RATE   # 新的当前学习率（不是初始学习率）
        
        if rank == 0:
            print(f"step 3 : 训练组件状态恢复 finished")
    
    # 6. 主训练循环
    patience = 3
    patience_counter = 0
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)  #  每个epoch打乱顺序不同

        # 5.1 train one epoch
        train_loss, train_ppl, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler, scheduler, writer, rank, global_step
        )
        
        # 5.2 validation
        # 同步valid loss（所有进程取平均）
        val_loss, _val_ppl = evaluate(model, valid_loader, criterion, device)
        val_loss_tensor = torch.tensor(val_loss).to(device)     
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)  
        val_loss = val_loss_tensor.item() / world_size
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        
        # 5.3 日志 & 保存
        early_stop = False
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']    # 每个epoch结束时的LR
            total_norm_squared = 0                          # 记录梯度模长
            for p in model.module.parameters():             # 注意：此时 model 是 DDP，需访问 .module 获取原始参数       
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_squared += param_norm.item() ** 2
            total_norm = total_norm_squared ** 0.5
            

            # tensorboard 日志
            writer.add_scalar('Loss/Train_Loss', train_loss, epoch)
            writer.add_scalar('Metrics/Train_PPL', train_ppl, epoch)
            writer.add_scalar('Loss/Validation_Loss', val_loss, epoch)
            writer.add_scalar('Metrics/Validation_PPL', val_ppl, epoch)
            writer.add_scalar('Params/Learning_Rate', current_lr, epoch)
            writer.add_scalar('Gradients/Norm_Squared', total_norm_squared, epoch)
            writer.add_scalar('Gradients/Norm', total_norm, epoch)

            # terminal 日志
            print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
                    f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = {
                    'model_state_dict': model.module.state_dict(),
                    'config': model_config,
                    'val_loss': best_val_loss,
                    'epoch': epoch - 1 
                }
                torch.save(best_model, SAVE_PATH)
                print(f"✓ 最佳模型已保存")

            # early stop
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停：{patience} 个 epoch valid loss 未下降")
                    early_stop = True
            
            # 保存checkpoint
            if epoch % NUM_EPOCHS_PER_LOG == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch:03d}_time{timestamp}.pt")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'loss': val_loss,
                    'config': model_config,
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"✓ 检查点{epoch}已保存")

        # 广播早停信号
        early_stop_tensor = torch.tensor([1 if early_stop else 0]).to(device)
        dist.broadcast(early_stop_tensor, src=0)
        if early_stop_tensor.item() == 1:
            if rank != 0:
                print(f"Rank {rank}: 收到早停信号，退出训练。")
            break

    # 关闭tensorboard
    if rank == 0 and writer:
        writer.close()        

# %%
def main_worker(rank, world_size, local_rank, device, resume_from = None):

    # 初始化 TensorBoard
    writer = None
    if rank == 0:
        # 加上时间戳或任务ID，避免覆盖旧日志
        log_dir = os.path.join(LOG_DIR, f"runs_{VERSION}_{ROUND}", datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
    global_step = 0
    

    # 1. 数据加载
    dataset_train, dataset_valid = load_dataset()
    if rank == 0:   # 加载 Tokenizer (仅 Rank 0 加载)
        tokenizer = get_tokenizer(dataset_train)
    dist.barrier() # 等待 rank 0 完成
    tokenizer = Tokenizer.from_file(BPE_PATH)    # 所有进程再次加载
    train_loader, valid_loader, train_sampler= get_dataloaders(rank, world_size, dataset_train, dataset_valid, tokenizer)
    

    # 2. 创建模型
    # DDP 下必须在 ms.spawn()后进行
    vocab_size = tokenizer.get_vocab_size()
    model = Transformer(
        num_blocks=NUM_BLOCKS,
        vocab_size=vocab_size,
        hidden_dim=D_MODEL,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        max_len_en=MAX_LEN_EN,
        max_len_de=MAX_LEN_DE,
        dropout=DROP_OUT
    ).to(device)
    
    # 用于保存checkpoint & best model
    model_config = {        
        'num_blocks': NUM_BLOCKS,
        'vocab_size': vocab_size,
        'hidden_dim': D_MODEL,
        'num_heads': NUM_HEADS,
        'ff_dim': FF_DIM,
        'max_len_en': MAX_LEN_EN,
        'max_len_de': MAX_LEN_DE,
        'dropout': DROP_OUT,
        'pad_idx': PAD_idx,
        'sos_idx': SOS_idx,
        'eos_idx': EOS_idx,
    }

    # 3. 定义训练参数：loss, optim, scheduler, scaler
    # 务必严格遵守 Model -> DDP -> Optimizer -> Scheduler 的初始化顺序
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_idx, label_smoothing=0.1)
    total_steps = len(train_loader) * NUM_EPOCHS 
    warmup_steps = int(0.1 * total_steps) 
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = GradScaler('cuda')     # 混合精度
    
    best_val_loss = float('inf')
    start_epoch = 1
    patience = 3
    patience_counter = 0


    # 4 从断点恢复
    if resume_from:
        if rank == 0:
            print(f"从 {resume_from} 加载模型...")

        # 4.1 加载模型权重
        # 此时 model 还是原始的 Transformer，checkpoint 中的 key 不应带 'module.'
        # 如果之前的 checkpoint 是错误地带着 'module.' 保存的，这里需要清洗 key
        checkpoint = torch.load(resume_from, map_location=f'cuda:{local_rank}')
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 去除可能存在的 module. 前缀
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        # 4.2 加载优化器和调度器
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 4.3 覆盖学习率
        if NEW_CUR_LEARNING_RATE is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = NEW_CUR_LEARNING_RATE  # 新的当前学习率
        
        # 4.4 恢复训练状态
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        steps_per_epoch = len(train_loader)
        global_step = (start_epoch - 1) * steps_per_epoch * world_size  # 假设每个 epoch 有 steps_per_epoch 个 batch
        
        if rank == 0:
            print(f"从 epoch {checkpoint['epoch']} 恢复训练")
    
    # 包装 DDP (必须在加载权重之后!)
    model = DDP(model, device_ids=[local_rank])

    # 5. 主训练循环
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)  #  每个epoch打乱顺序不同

        # 5.1 train one epoch
        train_loss, train_ppl, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler, scheduler, writer, rank, global_step
        )
        
        # 5.2 validation
        # 同步valid loss（所有进程取平均）
        val_loss, _val_ppl = evaluate(model, valid_loader, criterion, device)
        val_loss_tensor = torch.tensor(val_loss).to(device)     
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)  
        val_loss = val_loss_tensor.item() / world_size
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        
        # 5.3 日志 & 保存
        early_stop = False
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']    # 每个epoch结束时的LR
            total_norm_squared = 0                          # 记录梯度模长
            for p in model.module.parameters():             # 注意：此时 model 是 DDP，需访问 .module 获取原始参数       
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_squared += param_norm.item() ** 2
            total_norm = total_norm_squared ** 0.5
            

            # tensorboard 日志
            writer.add_scalar('Loss/Train_Loss', train_loss, epoch)
            writer.add_scalar('Metrics/Train_PPL', train_ppl, epoch)
            writer.add_scalar('Loss/Validation_Loss', val_loss, epoch)
            writer.add_scalar('Metrics/Validation_PPL', val_ppl, epoch)
            writer.add_scalar('Params/Learning_Rate', current_lr, epoch)
            writer.add_scalar('Gradients/Norm_Squared', total_norm_squared, epoch)
            writer.add_scalar('Gradients/Norm', total_norm, epoch)

            # terminal 日志
            print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
                    f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = {
                    'model_state_dict': model.module.state_dict(),
                    'config': model_config,
                    'val_loss': best_val_loss,
                    'epoch': epoch - 1 
                }
                torch.save(best_model, SAVE_PATH)
                print(f"✓ 最佳模型已保存")

            # early stop
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停：{patience} 个 epoch valid loss 未下降")
                    early_stop = True
            
            # 保存checkpoint
            if epoch % NUM_EPOCHS_PER_LOG == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch:03d}_time{timestamp}.pt")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'loss': val_loss,
                    'config': model_config,
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"✓ 检查点{epoch}已保存")

        # 广播早停信号
        early_stop_tensor = torch.tensor([1 if early_stop else 0]).to(device)
        dist.broadcast(early_stop_tensor, src=0)
        if early_stop_tensor.item() == 1:
            if rank != 0:
                print(f"Rank {rank}: 收到早停信号，退出训练。")
            break

    # 关闭tensorboard
    if rank == 0 and writer:
        writer.close()        

# %% [markdown]
# ### Part 4 Inference

# %%
def diagnose_bleu(hypotheses, references, tokenizer):
    """诊断BLEU计算问题"""
    print("=" * 60)
    print("🔍 BLEU 诊断报告")
    print("=" * 60)
    
    # 1. 检查空字符串
    empty_hyp = [i for i, h in enumerate(hypotheses) if len(h.strip()) == 0]
    empty_ref = [i for i, r in enumerate(references) if len(r[0].strip()) == 0]
    
    print(f"\n📝 空字符串统计:")
    print(f"  空假设: {len(empty_hyp)}/{len(hypotheses)} ({100*len(empty_hyp)/len(hypotheses):.1f}%)")
    print(f"  空参考: {len(empty_ref)}/{len(references)} ({100*len(empty_ref)/len(references):.1f}%)")
    
    # 2. 检查长度分布
    hyp_lengths = [len(h.split()) for h in hypotheses]
    ref_lengths = [len(r[0].split()) for r in references]
    
    print(f"\n📏 长度统计:")
    print(f"  假设 - 平均: {np.mean(hyp_lengths):.1f}, 中位数: {np.median(hyp_lengths):.1f}, 最小: {min(hyp_lengths)}, 最大: {max(hyp_lengths)}")
    print(f"  参考 - 平均: {np.mean(ref_lengths):.1f}, 中位数: {np.median(ref_lengths):.1f}, 最小: {min(ref_lengths)}, 最大: {max(ref_lengths)}")
    
    # 3. 打印前5个样本
    print(f"\n📄 前5个样本:")
    for i in range(min(5, len(hypotheses))):
        print(f"\n样本 {i}:")
        print(f"  假设: {hypotheses[i][:100]}")
        print(f"  参考: {references[i][0][:100]}")
        print(f"  假设长度: {hyp_lengths[i]} 词")
        print(f"  参考长度: {ref_lengths[i]} 词")
        
        # 检查是否完全相同
        if hypotheses[i].strip() == references[i][0].strip():
            print(f"  ⚠️ 完全相同！")
    
    # 4. 检查 token IDs
    print(f"\n🔢 检查原始token IDs (第一个样本):")
    # 重新加载一个样本来检查原始IDs
    dataset = load_from_disk(DATASET_DIR)
    test_sample = dataset["test"][0]
    src_text = test_sample['translation']['en']
    tgt_text = test_sample['translation']['de']
    
    src_ids = tokenizer.encode(src_text).ids[:10]
    tgt_ids = tokenizer.encode(tgt_text).ids[:10]
    
    print(f"  源文本: {src_text[:100]}")
    print(f"  源token IDs (前10): {src_ids}")
    print(f"  参考文本: {tgt_text[:100]}")
    print(f"  参考token IDs (前10): {tgt_ids}")
    
    # 5. 手动计算一个样本的BLEU
    print(f"\n🎯 手动计算第一个样本:")
    from nltk.translate.bleu_score import sentence_bleu
    hyp_tokens = hypotheses[0].split()
    ref_tokens = references[0][0].split()
    bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0))
    bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"  假设: {hyp_tokens[:20]}")
    print(f"  参考: {ref_tokens[:20]}")
    print(f"  1-gram BLEU: {bleu_1:.4f}")
    print(f"  4-gram BLEU: {bleu_4:.4f}")

# 在 evaluate_model 中调用
def evaluate_model(checkpoint_path, tokenizer, device):
    # ... 您的代码 ...
    
    # 在计算BLEU之前添加诊断
    diagnose_bleu(hypotheses, references, tokenizer)
    
    # 计算BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print(f"\n🎯 BLEU: {bleu.score:.2f}")
    print(f"   Precisons: {bleu.precisions}")
    print(f"   BP: {bleu.bp}")

# %%
def evaluate_model(checkpoint_path, tokenizer, device):
    print(f"开始评估最佳模型")
    # 1. 加载测试集
    dataset = load_from_disk(DATASET_DIR)
    dataset_test = dataset["test"]
    dataset_test_processed = WMT14Dataset(dataset_test, tokenizer)
    test_loader = DataLoader(
        dataset_test_processed,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )


    # 2. 加载最佳模型
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model = Transformer(
            num_blocks=config['num_blocks'],
            vocab_size=config['vocab_size'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            ff_dim=config['ff_dim'],
            max_len_en=config.get('max_len_en', 50),  
            max_len_de=config.get('max_len_de', 50),
            dropout=config.get('dropout', 0.1),
        ).to(device)
    
    # vocab_size = tokenizer.get_vocab_size()
    # model = Transformer(
    #     num_blocks=NUM_BLOCKS,
    #     vocab_size=vocab_size,
    #     hidden_dim=D_MODEL,
    #     num_heads=NUM_HEADS,
    #     ff_dim=FF_DIM,
    #     max_len_en=MAX_LEN_EN,
    #     max_len_de=MAX_LEN_DE,
    #     dropout=DROP_OUT
    # ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    # 3. 生成译文并收集参考
    hypotheses, references = [], []
    sos_id = tokenizer.token_to_id("<sos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="生成译文"):
            src = batch['src'].to(device)
            src_len = batch['src_len'].to(device)
            generated = model(src, src_len, tgt=None, device=device)
            
            for i in range(generated.size(0)):
                tokens = generated[i].cpu().tolist()
                # 找到第一个EOS并截断
                if eos_id in tokens:
                    tokens = tokens[:tokens.index(eos_id)]
                # 移除SOS如果在开头
                if tokens and tokens[0] == sos_id:
                    tokens = tokens[1:]
                tokens = [t for t in tokens if t != pad_id]
                hyp_text = tokenizer.decode(tokens)
                hypotheses.append(hyp_text)
            
            for i in range(batch['tgt'].size(0)):
                ref_tokens = batch['tgt'][i].cpu().tolist()
                if eos_id in ref_tokens:
                    ref_tokens = ref_tokens[:ref_tokens.index(eos_id)]
                if ref_tokens and ref_tokens[0] == sos_id:
                    ref_tokens = ref_tokens[1:]
                ref_tokens = [t for t in ref_tokens if t != pad_id]
                ref_text = tokenizer.decode(ref_tokens)
                references.append([ref_text])


    
    # 5. 计算BLEU
    diagnose_bleu(hypotheses, references, tokenizer)

    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print(f"\nBLEU: {bleu.score:.2f}")
    print(f"Precisons: {bleu.precisions}")
    print(f"BP: {bleu.bp}")
    
    return bleu
    

# %% [markdown]
# # Part 5 main()

# %%
# 初始化分布式环境
def run_training(checkpoint = None):
    # 1. 获取分布式环境变量 (由 torchrun 自动注入)
    # 注意：在 torchrun 模式下，当前脚本就是 worker 进程，不需要 spawn
    rank = int(os.environ.get('RANK', 0))               # 进程在整个集群所有进程中的唯一编号，用于逻辑通信、日志保存、数据分片
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))   # 进程在当前机器内的编号，用于GPU索引。如 cuda:X, device_ids

    # 2. 设置进程当前 GPU
    # 在 dist.init_process_group()之前做
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # 3. 初始化分布式环境
    # 关键！所有进程必须执行此行
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        device_id=local_rank
    )
    print(f"DDP init on rank {rank}, world_size {world_size}, local_rank {local_rank}")

    # 4. main_worker
    main_worker(rank, world_size, local_rank, device, checkpoint)

    # 5. 清理
    dist.destroy_process_group()

# %%
def run_inference(checkpoint):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer()
    evaluate_model(checkpoint, tokenizer, device)

# %%
def main():
    # python Transformer_v10.py --mode train    训练模式
    # python Transformer_v10.py --mode resume --checkpoint ./checkpoints/best.pt    断点重训
    # python Transformer_v10.py --mode inference --checkpoint ./checkpoints/best.pt   推理模式（单卡）
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'resume'])
    parser.add_argument('--checkpoint', type=str)
    # parser.add_argument('--lr', type=float)
    # parser.add_argument('--new_cur_lr', type=float)
    args = parser.parse_args()
    
    # if args.lr is not None:
    #     LEARNING_RATE = args.lr
    # if args.new_cur_lr is not None:
    #     NEW_CUR_LEARNING_RATE = args.new_cur_lr
    # else:
    #     NEW_CUR_LEARNING_RATE = None

    if args.mode == 'train':
        run_training()
    elif args.mode == 'resume':
        run_training(args.checkpoint)
    else:
        run_inference(args.checkpoint)


if __name__ == "__main__":
    main()


