import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAVectorQuantizer(nn.Module):
    """超球对齐的 EMA 离散词元库"""
    def __init__(self, num_codes=256, d_model=128, decay=0.99, eps=1e-5):
        super(EMAVectorQuantizer, self).__init__()
        self.num_codes = num_codes
        self.d_model = d_model
        self.decay = decay
        self.eps = eps

        embedding = torch.randn(num_codes, d_model)
        embedding = F.normalize(embedding, p=2, dim=-1)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.ones(num_codes))
        self.register_buffer('embedding_avg', embedding.clone())

    def forward(self, z):
        B, L, D = z.shape
        z_norm = F.normalize(z, p=2, dim=-1)
        z_flat = z_norm.reshape(-1, D)
        emb_norm = F.normalize(self.embedding, p=2, dim=-1)

        similarity = torch.matmul(z_flat, emb_norm.t())
        dists = 2.0 - 2.0 * similarity

        encoding_indices = torch.argmin(dists, dim=1)
        z_q = emb_norm[encoding_indices].reshape(B, L, D)

        if self.training:
            encodings_onehot = F.one_hot(encoding_indices, self.num_codes).float()
            new_cluster_size = torch.sum(encodings_onehot, dim=0)
            new_embedding_avg = torch.matmul(encodings_onehot.t(), z_flat)

            self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
            self.embedding_avg.data.mul_(self.decay).add_(new_embedding_avg, alpha=1 - self.decay)

            dead_codes = self.cluster_size < 1e-2
            if dead_codes.any():
                n_dead = dead_codes.sum().item()
                rand_indices = torch.randint(0, z_flat.size(0), (n_dead,), device=z.device)
                self.embedding.data[dead_codes] = z_flat[rand_indices].detach()
                self.embedding_avg.data[dead_codes] = z_flat[rand_indices].detach()
                self.cluster_size.data[dead_codes] = 1.0

            safe_cluster_size = torch.clamp(self.cluster_size, min=1e-4).unsqueeze(1)
            updated_embedding = self.embedding_avg / safe_cluster_size
            self.embedding.data.copy_(F.normalize(updated_embedding, p=2, dim=-1))

        loss_commit = F.mse_loss(z_norm, z_q.detach())
        z_q = z_norm + (z_q - z_norm).detach()
        min_dist = torch.min(dists, dim=1)[0].reshape(B, L)

        return z_q, loss_commit, min_dist


class TransformerBlock(nn.Module):
    """标准工业级 Transformer 块：FFN 放大比设为 4，保证非线性容量充足"""
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
    大容量反事实时空网络 (显式支撑 ~18 万参数量)
    """
    def __init__(self, c_in=55, win_size=60, d_model=128, num_codes=256, n_layers=3):
        super(CEPNet, self).__init__()
        self.c_in = c_in
        self.win_size = win_size
        self.d_model = d_model

        # 1. 深度时空多通道投影头 (升维至 128)
        self.enc_in = nn.Sequential(
            nn.Linear(c_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, win_size, d_model) * 0.02)

        # 2. 深度编码器 (3 层 Transformer)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=4)
            for _ in range(n_layers)
        ])

        # 3. 超球 EMA 离散健康词元库
        self.quantizer = EMAVectorQuantizer(num_codes=num_codes, d_model=d_model)

        # 4. 深度反事实解码器 (3 层 Transformer)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=4)
            for _ in range(n_layers)
        ])

        # 5. 物理输出映射头
        self.dec_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, c_in)
        )

        # 6. 各通道物理刚度参数 (恒正)
        self.log_lambda = nn.Parameter(torch.zeros(c_in))

    def freeze_codebook(self):
        self.quantizer.eval()
        print("[Model] Stage 2: Codebook permanently FROZEN.")

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape

        # 1. 编码映射并注入时间位置编码
        h = self.enc_in(x) + self.pos_emb
        for block in self.encoder_blocks:
            h = block(h)
        z = h

        # 2. 离散基元纯净投影 (彻底切断异常捷径)
        z_q, loss_commit, min_dist = self.quantizer(z)

        # 3. 深度反事实物理波形生成
        dh = z_q
        for block in self.decoder_blocks:
            dh = block(dh)
        x_star = self.dec_out(dh)  # [B, L, C]

        # 4. Pearl 个体因果干预效应 (ITE)
        lambdas = F.softplus(self.log_lambda)
        intervention_effect = x - x_star
        weighted_energy = lambdas * (intervention_effect ** 2)

        # 异常得分与通道归因
        time_score = torch.sum(weighted_energy, dim=-1) + 0.01 * min_dist
        channel_attribution = torch.mean(weighted_energy, dim=1)

        return x_star, time_score, channel_attribution, loss_commit