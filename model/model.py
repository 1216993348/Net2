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

            # 死码重置
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


class CEPNet(nn.Module):
    """
    大容量反事实流形网络：
    引入多尺度膨胀因果卷积与 128 维特征流形，参数量提升至 ~20 万
    """
    def __init__(self, c_in=55, win_size=60, d_model=128, num_codes=256):
        super(CEPNet, self).__init__()
        self.c_in = c_in
        self.win_size = win_size
        self.d_model = d_model

        # 1. 深度时空多尺度编码器 (捕获局部高频与全局周期)
        self.enc_in = nn.Conv1d(c_in, d_model, kernel_size=1)
        self.enc_block1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1),
            nn.GroupNorm(8, d_model),
            nn.GELU()
        )
        self.enc_block2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, d_model),
            nn.GELU()
        )
        self.enc_block3 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(8, d_model),
            nn.GELU()
        )

        # 2. 扩容离散词元库
        self.quantizer = EMAVectorQuantizer(num_codes=num_codes, d_model=d_model)

        # 3. 增强解码器 (利用多尺度特征逆向合成波形)
        self.dec_block1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(8, d_model),
            nn.GELU()
        )
        self.dec_block2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, d_model),
            nn.GELU()
        )
        self.dec_block3 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1),
            nn.GroupNorm(8, d_model),
            nn.GELU()
        )
        self.dec_out = nn.Conv1d(d_model, c_in, kernel_size=1)

        # 4. 可学习通道刚度参数
        self.log_lambda = nn.Parameter(torch.zeros(c_in))

    def freeze_codebook(self):
        self.quantizer.eval()
        print("[Model] Stage 2: Codebook permanently FROZEN.")

    def forward(self, x):
        B, L, C = x.shape
        x_trans = x.transpose(1, 2).contiguous()

        # 1. 连续流形编码
        h = self.enc_in(x_trans)
        h = h + self.enc_block1(h)
        h = h + self.enc_block2(h)
        h = h + self.enc_block3(h)
        z = h.transpose(1, 2).contiguous()

        # 2. 离散基元纯净投影
        z_q, loss_commit, min_dist = self.quantizer(z)

        # 3. 反事实物理基准生成 X*
        qh = z_q.transpose(1, 2).contiguous()
        qh = qh + self.dec_block1(qh)
        qh = qh + self.dec_block2(qh)
        qh = qh + self.dec_block3(qh)
        x_star = self.dec_out(qh).transpose(1, 2).contiguous()

        # 4. Pearl 个体因果干预效应 (ITE)
        lambdas = F.softplus(self.log_lambda)
        intervention_effect = x - x_star
        weighted_energy = lambdas * (intervention_effect ** 2)

        time_score = torch.sum(weighted_energy, dim=-1) + 0.01 * min_dist
        channel_attribution = torch.mean(weighted_energy, dim=1)

        return x_star, time_score, channel_attribution, loss_commit