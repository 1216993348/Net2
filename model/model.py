import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAVectorQuantizer(nn.Module):
    """
    工业级健壮的 EMA 离散词元库：
    1. 彻底解决除零 NaN (强下界截断 + 死码复活)
    2. 采用单位超球表征 (L2 归一化)，消灭数值范数发散
    """
    def __init__(self, num_codes=128, d_model=64, decay=0.99, eps=1e-5):
        super(EMAVectorQuantizer, self).__init__()
        self.num_codes = num_codes
        self.d_model = d_model
        self.decay = decay
        self.eps = eps

        # 初始化词表并在球面上归一化
        embedding = torch.randn(num_codes, d_model)
        embedding = F.normalize(embedding, p=2, dim=-1)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.ones(num_codes))  # 初始计数值设为 1，防除零
        self.register_buffer('embedding_avg', embedding.clone())

    def forward(self, z):
        # z: [B, L, D]
        B, L, D = z.shape
        z_flat = z.reshape(-1, D)

        # 1. 规范化特征与词表：数值范围死死锁在 [-1, 1]，杜绝数值漂移
        z_norm = F.normalize(z_flat, p=2, dim=-1)
        emb_norm = F.normalize(self.embedding, p=2, dim=-1)

        # 2. 余弦距离等价于归一化欧氏距离：||a - b||^2 = 2 - 2 * <a, b>
        # similarity: [B*L, num_codes]
        similarity = torch.matmul(z_norm, emb_norm.t())
        dists = 2.0 - 2.0 * similarity

        # 3. 最近邻检索匹配
        encoding_indices = torch.argmin(dists, dim=1)
        z_q = self.embedding[encoding_indices].reshape(B, L, D)

        # 4. 训练阶段的平滑 EMA 更新与死码保护
        if self.training:
            encodings_onehot = F.one_hot(encoding_indices, self.num_codes).float()
            new_cluster_size = torch.sum(encodings_onehot, dim=0)
            new_embedding_avg = torch.matmul(encodings_onehot.t(), z_flat)

            # EMA 更新计数值与特征和
            self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
            self.embedding_avg.data.mul_(self.decay).add_(new_embedding_avg, alpha=1 - self.decay)

            # 死码检测：如果某个词元连续衰减，计数值极低，立即用当前 batch 活跃样本重置
            dead_codes = self.cluster_size < 1e-2
            if dead_codes.any():
                n_dead = dead_codes.sum().item()
                rand_indices = torch.randint(0, z_flat.size(0), (n_dead,), device=z.device)
                self.embedding.data[dead_codes] = z_flat[rand_indices].detach()
                self.embedding_avg.data[dead_codes] = z_flat[rand_indices].detach()
                self.cluster_size.data[dead_codes] = 1.0

            # 核心防 NaN 保护：强制分母不小于 1e-4，杜绝除零
            safe_cluster_size = torch.clamp(self.cluster_size, min=1e-4).unsqueeze(1)
            updated_embedding = self.embedding_avg / safe_cluster_size
            self.embedding.data.copy_(F.normalize(updated_embedding, p=2, dim=-1))

        # 5. Commitment 损失 (拉近编码器特征到离散基元)
        loss_commit = F.mse_loss(z, z_q.detach())

        # 6. 直通估计器 (Straight-Through)
        z_q = z + (z_q - z).detach()

        # 提取距基元的绝对离散能量
        min_dist = torch.min(dists, dim=1)[0].reshape(B, L)

        return z_q, loss_commit, min_dist


class CEPNet(nn.Module):
    """
    反事实流形网络 (包含数值加固的因果编码、离散词元库与生成解码)
    """
    def __init__(self, c_in=55, win_size=60, d_model=64, num_codes=128):
        super(CEPNet, self).__init__()
        self.c_in = c_in
        self.win_size = win_size
        self.d_model = d_model

        # 1. 稳健编码器 (引入 LayerNorm 替代 BatchNorm，杜绝小批次震荡)
        self.enc_conv1 = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1)
        self.enc_norm1 = nn.GroupNorm(8, d_model)
        self.enc_conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.enc_norm2 = nn.GroupNorm(8, d_model)

        # 2. 健壮 EMA 离散词元库
        self.quantizer = EMAVectorQuantizer(num_codes=num_codes, d_model=d_model)

        # 3. 生成解码器
        self.dec_conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.dec_norm1 = nn.GroupNorm(8, d_model)
        self.dec_conv2 = nn.Conv1d(d_model, c_in, kernel_size=1)

        # 4. 可学习通道刚度参数
        self.log_lambda = nn.Parameter(torch.zeros(c_in))

    def freeze_codebook(self):
        """阶段二：冻结词表参数"""
        self.quantizer.eval()
        print("[Model] Stage 2: Codebook permanently FROZEN.")

    def forward(self, x):
        # x: [B, L, C] -> 转置为 Conv1d [B, C, L]
        B, L, C = x.shape
        x_trans = x.transpose(1, 2).contiguous()

        # 1. 连续流形编码
        h = F.gelu(self.enc_norm1(self.enc_conv1(x_trans)))
        h = F.gelu(self.enc_norm2(self.enc_conv2(h)))
        z = h.transpose(1, 2).contiguous()  # [B, L, d_model]

        # 2. 离散流形纯净投影
        z_q, loss_commit, min_dist = self.quantizer(z)

        # 3. 反事实物理基准生成 X*
        q_trans = z_q.transpose(1, 2).contiguous()
        dh = F.gelu(self.dec_norm1(self.dec_conv1(q_trans)))
        x_star = self.dec_conv2(dh).transpose(1, 2).contiguous()  # [B, L, C]

        # 4. Pearl 个体因果干预效应 (ITE)
        lambdas = F.softplus(self.log_lambda)
        intervention_effect = x - x_star
        weighted_energy = lambdas * (intervention_effect ** 2)

        # 时间步异常得分与通道归因贡献
        time_score = torch.sum(weighted_energy, dim=-1) + 0.1 * min_dist
        channel_attribution = torch.mean(weighted_energy, dim=1)

        return x_star, time_score, channel_attribution, loss_commit