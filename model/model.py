import torch
import torch.nn as nn
import torch.nn.functional as F


class CEPNet(nn.Module):
    """
    基于二阶弹性势能与因果反事实流形网络 (CEP-Net)
    核心机制:
    1. 原型流形基元 (健康工况吸引子)
    2. 解析导数反事实投影 (闭式流形重建，杜绝直接恒等捷径)
    3. 自适应对角刚度矩阵 (Hessian 阻尼，解决 MSL 0/1 离散开关通道干扰)
    """
    def __init__(self, c_in=55, win_size=60, d_model=64, n_prototypes=16, tau=0.5):
        super(CEPNet, self).__init__()
        self.c_in = c_in
        self.win_size = win_size
        self.d_model = d_model
        self.K = n_prototypes
        self.tau = tau  # 势能温度系数

        # 1. 因果动力学编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(c_in, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        # 2. 健康工况流形原型基元 P: [K, d_model]
        self.prototypes = nn.Parameter(torch.randn(self.K, d_model) * 0.05)

        # 3. 反事实物理波形恢复解码器
        self.decoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, c_in, kernel_size=1)
        )

        # 4. 二阶刚度参数矩阵 (Hessian 对角元，通过 Softplus 保证正定物理阻尼)
        self.log_lambda = nn.Parameter(torch.zeros(self.K, c_in))

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        x_trans = x.transpose(1, 2).contiguous()

        # Step 1: 提取连续动力学特征 Z: [B, L, d_model]
        z = self.encoder(x_trans).transpose(1, 2).contiguous()

        # Step 2: 显式势能软寻址 (计算样本到各个健康吸引子的马氏/欧氏距离)
        # diff: [B, L, K, d_model]
        diff = z.unsqueeze(2) - self.prototypes.unsqueeze(0).unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # [B, L, K]

        # 玻尔兹曼分布权重: 决定系统瞬时工况对各健康原型的从属概率
        weights = F.softmax(-dist_sq / self.tau, dim=-1)  # [B, L, K]

        # Step 3: 显式反事实流形解析投影 (闭式凸组合重构，无离散截断误差)
        z_star = torch.einsum('blk,kd->bld', weights, self.prototypes)  # [B, L, d_model]

        # Step 4: 生成反事实健康物理基准 X* (Parallel World Signal)
        x_star = self.decoder(z_star.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        # Step 5: 二阶弹性势能与个体因果干预效应 (ITE)
        # 综合计算各工况下的有效刚度: [B, L, C]
        lambdas = F.softplus(self.log_lambda)
        effective_lambda = torch.einsum('blk,kc->blc', weights, lambdas)

        # 一阶干预残差 (恢复力)
        intervention_effect = x - x_star  # [B, L, C]

        # 二阶能量偏导: E = 1/2 * \Delta x^T * H * \Delta x
        channel_energies = effective_lambda * (intervention_effect ** 2)  # [B, L, C]

        # 吸引子势阱能量惩罚 (判断特征是否脱离正常健康流形凹槽)
        min_dist_sq, _ = torch.min(dist_sq, dim=-1)  # [B, L]

        # 综合异常得分: 局部通道能量 + 流形偏离势能
        time_score = torch.sum(channel_energies, dim=-1) + 0.05 * min_dist_sq
        channel_attribution = torch.mean(channel_energies, dim=1)

        return x_star, time_score, channel_attribution, dist_sq