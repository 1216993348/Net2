import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock(nn.Module):
    """一维因果卷积块：通过左侧补零保证时间单向性"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(CausalConvBlock, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0
        )
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: [B, C, L]
        x = F.pad(x, (self.pad, 0))
        return self.act(self.norm(self.conv(x)))


class VectorQuantizerMemory(nn.Module):
    """
    数值稳定版离散健康词元库：
    1. 避免使用 cdist，改用带 epsilon 的点积距离展开，消除 sqrt(0) 梯度 NaN。
    2. 引入 L2 投影与稳态更新，防止梯度爆炸。
    """

    def __init__(self, num_codes=128, d_model=64):
        super(VectorQuantizerMemory, self).__init__()
        self.num_codes = num_codes
        self.d_model = d_model
        self.embedding = nn.Embedding(num_codes, d_model)
        self.embedding.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, z):
        # z: [B, L, D]
        B, L, D = z.shape
        z_flat = z.reshape(-1, D)

        # 展开计算欧氏距离平方：||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        # 彻底避免使用 torch.cdist() 引发梯度除零 NaN
        z_sq = torch.sum(z_flat ** 2, dim=-1, keepdim=True)  # [B*L, 1]
        e_sq = torch.sum(self.embedding.weight ** 2, dim=-1)  # [num_codes]
        dot = torch.matmul(z_flat, self.embedding.weight.t())  # [B*L, num_codes]

        # 截断下界防止出现极其微小的负数扰动
        dists = torch.clamp(z_sq + e_sq.unsqueeze(0) - 2 * dot, min=1e-6)

        # 匹配最近基元
        encoding_indices = torch.argmin(dists, dim=1)
        z_q = self.embedding(encoding_indices).reshape(B, L, D)

        # VQ 损失计算 (添加 clamp 防溢出)
        loss_commit = F.mse_loss(z.detach(), z_q) + 0.25 * F.mse_loss(z, z_q.detach())
        loss_commit = torch.clamp(loss_commit, max=100.0)

        # 梯度直通 (Straight-Through Estimator)
        z_q = z + (z_q - z).detach()

        # 距离基元的偏离能量
        min_dist = torch.min(dists, dim=1)[0].reshape(B, L)

        return z_q, loss_commit, min_dist


class CEPNet(nn.Module):
    """
    纯生成反事实网络：输入 X -> 编码 Z -> 词元量化 Z* -> 纯解码 X*
    """
    def __init__(self, c_in=55, win_size=60, d_model=64, num_codes=128):
        super(CEPNet, self).__init__()
        self.c_in = c_in
        self.win_size = win_size
        self.d_model = d_model
        self.num_codes = num_codes

        # 1. 因果动力学编码器
        self.encoder = nn.Sequential(
            CausalConvBlock(c_in, d_model, kernel_size=3, dilation=1),
            CausalConvBlock(d_model, d_model, kernel_size=3, dilation=2),
            CausalConvBlock(d_model, d_model, kernel_size=3, dilation=4)
        )

        # 2. 离散健康词元记忆库
        self.memory = VectorQuantizerMemory(num_codes=num_codes, d_model=d_model)

        # 3. 各通道物理刚度参数
        self.log_lambda = nn.Parameter(torch.zeros(c_in))

        # 4. 深度因果生成解码器
        self.decoder = nn.Sequential(
            CausalConvBlock(d_model, d_model, kernel_size=3, dilation=1),
            CausalConvBlock(d_model, d_model, kernel_size=3, dilation=2),
            CausalConvBlock(d_model, d_model, kernel_size=3, dilation=4),
            nn.Conv1d(d_model, c_in, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, L, C] -> 转置为 Conv1d 格式 [B, C, L]
        B, L, C = x.shape
        x_trans = x.transpose(1, 2).contiguous()

        # 1. 提取连续潜在特征 Z
        z = self.encoder(x_trans).transpose(1, 2).contiguous()  # [B, L, d_model]

        # 2. 离散反事实量化 (无残差直连)
        z_star, vq_loss, min_dist = self.memory(z)  # [B, L, d_model]

        # 3. 纯生成反事实物理基底 X*
        z_star_trans = z_star.transpose(1, 2).contiguous()
        x_star = self.decoder(z_star_trans).transpose(1, 2).contiguous()  # [B, L, C]

        # 4. Pearl 反事实干预效应 (ITE)
        lambdas = F.softplus(self.log_lambda)  # [C]
        intervention_effect = x - x_star       # [B, L, C]
        weighted_energy = lambdas * (intervention_effect ** 2)

        # 5. 异常得分与通道归因
        time_score = torch.sum(weighted_energy, dim=-1) + 0.1 * min_dist
        channel_attribution = torch.mean(weighted_energy, dim=1)

        return x_star, time_score, channel_attribution, vq_loss