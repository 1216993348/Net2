import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class CEPNet(nn.Module):
    """
    基于时空掩码机制的反事实记忆网络 (Masked Counterfactual Network)
    训练时通过随机时空掩码阻断捷径，迫使参数网络硬背物理动力学流形
    测试时利用全局物理自洽性生成纯净反事实基准
    """
    def __init__(self, c_in=55, win_size=60, d_model=128, n_layers=3, mask_ratio=0.4, top_k=3):
        super(CEPNet, self).__init__()
        self.c_in = c_in
        self.win_size = win_size
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.top_k = top_k

        # 1. 物理传感器投影与时间位置编码
        self.enc_in = nn.Sequential(
            nn.Linear(c_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, win_size, d_model) * 0.02)
        # 掩码标记 (可学习的 MASK Token)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 2. 深度时空 Transformer 主干 (参数化物理知识库)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=4)
            for _ in range(n_layers * 2)  # 6 层深度 Transformer
        ])

        # 3. 反事实物理生成还原头
        self.dec_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, c_in)
        )

        # 4. 可学习通道物理刚度参数
        self.log_lambda = nn.Parameter(torch.zeros(c_in))

    def random_masking(self, x_emb):
        """随机掩码时间步：阻断恒等映射捷径"""
        B, L, D = x_emb.shape
        len_keep = int(L * (1 - self.mask_ratio))

        # 生成随机噪声并排序确定保留索引
        noise = torch.rand(B, L, device=x_emb.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        # 提取保留的上下文
        x_masked = torch.gather(x_emb, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 用 MASK Token 补全剩余位置
        mask_tokens = self.mask_token.repeat(B, L - len_keep, 1)
        x_full = torch.cat([x_masked, mask_tokens], dim=1)
        # 还原原始时间顺序
        x_restored = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        return x_restored

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape

        # 1. 线性投影到隐空间
        h = self.enc_in(x)

        # 2. 训练阶段执行随机掩码，测试阶段输入全量观测让模型生成自洽反事实
        if self.training:
            h = self.random_masking(h)

        h = h + self.pos_emb

        # 3. 深度 Transformer 推理
        for block in self.blocks:
            h = block(h)

        # 4. 生成反事实物理基准 X*
        x_star = self.dec_out(h)  # [B, L, C]

        # 5. Pearl 因果干预效应计算
        lambdas = F.softplus(self.log_lambda)
        intervention_effect = x - x_star
        channel_energies = lambdas * (intervention_effect ** 2)

        # 局部 Top-K 故障聚焦聚合
        topk_energies, _ = torch.topk(channel_energies, k=min(self.top_k, C), dim=-1)
        time_score = torch.mean(topk_energies, dim=-1)
        channel_attribution = torch.mean(channel_energies, dim=1)

        return x_star, time_score, channel_attribution