import torch
import torch.nn as nn
import math
from math import sqrt
import numpy as np
import torch.nn.functional as F
class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(AttentionLayer, self).__init__()

        self.norm = nn.LayerNorm(d_model)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # [B, T, D]
        B, T, D = x.shape
        queries = self.query_projection(x)
        keys = self.key_projection(x).transpose(1,2)
        values = self.value_projection(x)
        attn = torch.softmax(torch.matmul(queries, keys) / math.sqrt(D), -1)
        out = torch.matmul(attn, values) + x
        return self.out_projection(self.norm(out)) + out, attn


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MemoryAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head=8, max_len=1000, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # Q 仍然来自输入
        self.q_proj = nn.Linear(d_model, d_model)

        # ⭐ K 和 V 共享同一个记忆库
        self.memory = nn.Embedding(max_len, d_model)
        self.memory_adapter = nn.Linear(max_len, 1024)
        self.memory_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape

        # 1. Q 来自输入
        q = self.q_proj(x)  # [B, T, D]

        # 2. K 和 V 都来自记忆库（共享同一个）
        mem = self.memory.weight.unsqueeze(0).expand(B, -1, -1)  # [B, max_len, D]
        # 适配到 T
        if T <= mem.size(1):
            mem = mem[:, :T, :]
        else:
            mem = F.interpolate(mem.transpose(1, 2), size=T, mode='linear', align_corners=False).transpose(1, 2)
        mem = self.memory_proj(mem)  # [B, T, D]

        k = mem  # [B, T, D] ← K来自记忆
        v = mem  # [B, T, D] ← V来自记忆（与K共享）

        # 3. 多头拆分
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_head)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_head)

        # 4. 计算注意力：Q 与 K（记忆）的关系
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, T, T]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 5. 加权组合 V（也是记忆）
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')

        # 6. 融合注意力（兼容原有损失函数）
        attn_fused = torch.mean(attn, dim=1)  # [B, T, T]

        out = self.out_proj(out)
        return out, attn_fused