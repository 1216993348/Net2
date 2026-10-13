import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAVectorQuantizer(nn.Module):
    """
    全超球对齐的 EMA 离散词元库：
    确保索引检索、直通估计、解码器输入均在归一化特征空间完成，消除尺度漂移。
    """
    def __init__(self, num_codes=128, d_model=64, decay=0.99, eps=1e-5):
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
        # z: [B, L, D]
        B, L, D = z.shape
        # 全程在 L2 归一化空间中操作，彻底封死尺度漂移
        z_norm = F.normalize(z, p=2, dim=-1)
        z_flat = z_norm.reshape(-1, D)

        emb_norm = F.normalize(self.embedding, p=2, dim=-1)

        # 归一化余弦距离: ||a - b||^2 = 2 - 2 * <a, b>
        similarity = torch.matmul(z_flat, emb_norm.t())
        dists = 2.0 - 2.0 * similarity

        # 最近邻检索
        encoding_indices = torch.argmin(dists, dim=1)
        # 解码基元统一取归一化后的向量
        z_q = emb_norm[encoding_indices].reshape(B, L, D)

        # EMA 更新机制
        if self.training:
            encodings_onehot = F.one_hot(encoding_indices, self.num_codes).float()
            new_cluster_size = torch.sum(encodings_onehot, dim=0)
            new_embedding_avg = torch.matmul(encodings_onehot.t(), z_flat)

            self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
            self.embedding_avg.data.mul_(self.decay).add_(new_embedding_avg, alpha=1 - self.decay)

            # 死码重置：检测并激活未匹配基元
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

        # Commitment 损失 (严格在归一化空间计算，数值稳在 0~1 之间)
        loss_commit = F.mse_loss(z_norm, z_q.detach())

        # 直通估计器 (Straight-Through Estimator)
        z_q = z_norm + (z_q - z_norm).detach()

        min_dist = torch.min(dists, dim=1)[0].reshape(B, L)
        return z_q, loss_commit, min_dist


class CEPNet(nn.Module):
    """
    反事实流形网络：全超球对齐因果架构
    """
    def __init__(self, c_in=55, win_size=60, d_model=64, num_codes=128):
        super(CEPNet, self).__init__()
        self.c_in = c_in
        self.win_size = win_size
        self.d_model = d_model

        # 编码器 (带通道膨胀的特征提取)
        self.enc_conv1 = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1)
        self.enc_norm1 = nn.GroupNorm(8, d_model)
        self.enc_conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.enc_norm2 = nn.GroupNorm(8, d_model)

        # 超球离散词元库
        self.quantizer = EMAVectorQuantizer(num_codes=num_codes, d_model=d_model)

        # 解码器 (从纯净离散基元生成物理波形)
        self.dec_conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.dec_norm1 = nn.GroupNorm(8, d_model)
        self.dec_conv2 = nn.Conv1d(d_model, c_in, kernel_size=1)

        # 各传感器可学习刚度
        self.log_lambda = nn.Parameter(torch.zeros(c_in))

    def freeze_codebook(self):
        """冻结词表进入第二阶段"""
        self.quantizer.eval()
        print("[Model] Stage 2: Codebook permanently FROZEN.")

    def forward(self, x):
        B, L, C = x.shape
        x_trans = x.transpose(1, 2).contiguous()

        # 1. 连续流形映射
        h = F.gelu(self.enc_norm1(self.enc_conv1(x_trans)))
        h = F.gelu(self.enc_norm2(self.enc_conv2(h)))
        z = h.transpose(1, 2).contiguous()  # [B, L, d_model]

        # 2. 超球离散纯净投影
        z_q, loss_commit, min_dist = self.quantizer(z)

        # 3. 反事实物理基准生成 X*
        q_trans = z_q.transpose(1, 2).contiguous()
        dh = F.gelu(self.dec_norm1(self.dec_conv1(q_trans)))
        x_star = self.dec_conv2(dh).transpose(1, 2).contiguous()  # [B, L, C]

        # 4. Pearl 个体因果干预效应 (ITE)
        lambdas = F.softplus(self.log_lambda)
        intervention_effect = x - x_star
        weighted_energy = lambdas * (intervention_effect ** 2)

        # 异常得分：以物理能量为主，结合少量流形离散惩罚
        time_score = torch.sum(weighted_energy, dim=-1) + 0.01 * min_dist
        channel_attribution = torch.mean(weighted_energy, dim=1)

        return x_star, time_score, channel_attribution, loss_commit