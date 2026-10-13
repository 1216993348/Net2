import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .attn import AttentionLayer

from .embed import DataEmbedding, PositionalEmbedding



class TrendEnc(nn.Module):

    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, tr):
        super(TrendEnc, self).__init__()
        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.e_layers = e_layers
        self.attn_layers = nn.ModuleList([AttentionLayer(d_model) for _ in range(e_layers)])

        # 强力低通滤波器：大窗口平均池化，彻底过滤所有局部突变
        self.hard_low_pass = nn.AvgPool1d(kernel_size=11, stride=1, padding=5)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 1. 物理层面的强行低通化：输入特征直接被洗成平滑流
        x_trans = x.transpose(1, 2)
        x_smooth = self.hard_low_pass(x_trans).transpose(1, 2)

        h_first = self.emb(x_smooth)
        h = h_first + self.pos_emb(x.size(1))

        B, L, _ = x.shape
        att = []

        # 硬性对角线先验：锁死低通分布形态
        grid = torch.arange(L, device=x.device).float()
        dist_matrix = torch.abs(grid.unsqueeze(0) - grid.unsqueeze(1))
        low_pass_mask = torch.exp(- (dist_matrix ** 2) / (2 * (2.0 ** 2))).unsqueeze(0).unsqueeze(1)
        prior_loss = 0.0
        for i in range(self.e_layers):
            h, attn_w = self.attn_layers[i](h)

            # 融合硬性趋势分布
            if len(attn_w.shape) == 4:
                final_attn = attn_w * 0.3 + low_pass_mask * 0.7
            else:

                low_pass_mask_3d = low_pass_mask.squeeze(1)  # [B, L, L]
                final_attn = attn_w * 0.5 + low_pass_mask_3d * 0.5

                prior_loss += F.kl_div(
                    F.log_softmax(low_pass_mask_3d, dim=-1),
                    F.softmax(attn_w, dim=-1),
                    reduction='batchmean'
                )

            final_attn = final_attn + 1e-6
            final_attn = final_attn / final_attn.sum(dim=-1, keepdim=True)
            att.append(final_attn)

        return att, h, prior_loss


class SpikeEnc(nn.Module):

    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, fr):
        super(SpikeEnc, self).__init__()
        self.emb = DataEmbedding(c_in, d_model)
        # 🎯 注意：高通分支绝对不初始化也不使用任何 PositionalEmbedding！
        self.e_layers = e_layers
        #self.attn_layers = nn.ModuleList([AttentionLayer(d_model) for _ in range(e_layers)])
        self.attn_layers = nn.ModuleList([AttentionLayer(d_model) for _ in range(e_layers)])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, C = x.shape

        # 1. 物理层面的强行高通化：只保留微观一阶差分，彻底斩断宏观趋势
        x_left = F.pad(x, (0, 0, 1, 0), mode='replicate')[:, :-1, :]
        x_high = torch.abs(x - x_left)

        # 2. 映射特征（无位置编码，网络失去对绝对和相对时间的认知）
        h = self.emb(x_high)
        att = []

        # 硬性反转掩码：强迫注意力远离对角线
        grid = torch.arange(L, device=x.device).float()
        dist_matrix = torch.abs(grid.unsqueeze(0) - grid.unsqueeze(1))
        high_pass_mask = (1.0 - torch.exp(- (dist_matrix ** 2) / (2 * (6.0 ** 2)))).unsqueeze(0).unsqueeze(1)

        prior_loss = 0.0
        for i in range(self.e_layers):
            h, attn_w = self.attn_layers[i](h)

            # 强行扭转：让高通盯着非连续远端
            if len(attn_w.shape) == 4:
                flipped_attn = torch.flip(attn_w, dims=[-1]) * high_pass_mask
            else:
                high_pass_mask_3d = high_pass_mask.squeeze(1)  # [B, L, L]
                flipped_attn = torch.flip(attn_w, dims=[-1]) * high_pass_mask_3d

                prior_loss += F.kl_div(
                    F.log_softmax(high_pass_mask_3d, dim=-1),
                    F.softmax(attn_w, dim=-1),
                    reduction='batchmean'
                )

            final_attn = flipped_attn + 1e-6
            final_attn = final_attn / final_attn.sum(dim=-1, keepdim=True)
            att.append(final_attn)
        return att, h,prior_loss


class CycleEnc(nn.Module):

    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, sr=0.5):
        super(CycleEnc, self).__init__()
        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.e_layers = e_layers
        self.attn_layers = nn.ModuleList([AttentionLayer(d_model) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)


    def compute_period_batch(self, x):
        """完全批量化的周期计算"""
        B, L, C = x.shape

        # 一次性对所有通道做 FFT [B, C, L]
        x_fft = torch.fft.rfft(x, dim=1)  # [B, L/2+1, C]
        power_spec = torch.abs(x_fft) ** 2
        autocorr = torch.fft.irfft(power_spec, n=L, dim=1)  # [B, L, C]

        # 归一化 [B, L, C]
        autocorr = autocorr / (autocorr[:, 0:1, :] + 1e-8)

        # 跨通道平均 [B, L]
        autocorr_avg = autocorr.mean(dim=-1)

        # 找峰值
        search_range = L // 2  # 不是 min(L//2, 64)
        autocorr_trim = autocorr_avg[:, 1:search_range + 1]
        periods = torch.argmax(autocorr_trim, dim=1) + 1

        return periods.float()

    def forward(self, x):
        B, L, C = x.shape
        h = self.emb(x) + self.pos_emb(L)
        # 2. 动态计算当前样本的周期
        # 可以选择上面两种方法之一
        periods = self.compute_period_batch (x)  # [B]

        # 3. 周期先验掩码：基于动态计算的周期
        grid = torch.arange(L, device=x.device).float()
        dist_matrix = torch.abs(grid.unsqueeze(0) - grid.unsqueeze(1))  # [L, L]

        # 对batch中每个样本使用不同的周期
        # 扩展维度以便广播
        dist_expanded = dist_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        periods_expanded = periods.view(B, 1, 1, 1)  # [B, 1, 1, 1]

        # 计算相位
        period_phase = torch.cos(2 * math.pi * dist_expanded / (periods_expanded + 1e-5))

        # 归一化到 [0, 1]
        cycle_mask = (period_phase + 1.0) / 2.0  # [B, 1, L, L]

        # 4. 注意力计算（结合动态周期先验）
        att = []
        prior_loss = 0.0
        for i in range(self.e_layers):
            h, attn_w = self.attn_layers[i](h)
            a, b = 0.3, 0.7

            # 确保维度匹配
            if len(attn_w.shape) == 4:
                final_attn = attn_w * a + cycle_mask * b
            else:
                # 如果是3维，需要调整cycle_mask
                cycle_mask_3d = cycle_mask.squeeze(1)  # [B, L, L]
                final_attn = attn_w * a + cycle_mask_3d * b
                prior_loss += F.kl_div(
                    F.log_softmax(cycle_mask_3d, dim=-1),
                    F.softmax(attn_w, dim=-1),
                    reduction='batchmean'
                )

            # 归一化
            final_attn = final_attn + 1e-6
            final_attn = final_attn / final_attn.sum(dim=-1, keepdim=True)
            att.append(final_attn)

        return att, h, prior_loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class TriDecompNet(nn.Module):
    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5, dev=None):
        super(TriDecompNet, self).__init__()
        global device
        device = dev
        self.win_size = win_size

        # 1. 趋势低通分支
        # 1. 第一级：趋势提取器
        self.trend = TrendEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, tr)
        # 2. 突变高通分支
        self.spike = SpikeEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, fr)
        # 3. 新增：周期中频分支

        # 2. 第二级：周期提取器 & 突变提取器 (作用于残差空间)
        self.cycle = CycleEnc(c_in, c_out, d_model, e_layers, win_size, seq_size)
        self.spike = SpikeEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, fr)

        # 🎯 三分支联合解码器：输入维度是 3 * d_model，输出是 c_out
        # 3. 三分支联合重构解码器 (Concat: 3 * d_model -> c_out)
        self.joint_decoder = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, c_out)
        )

    def forward(self, x):
        trend_att, trend_h, tp_loss = self.trend(x)
        cycle_att, cycle_h, cp_loss = self.cycle(x)
        spike_att, spike_h, sp_loss = self.spike(x)

        # 联合重构：拼接三个层级的特征 [B, L, 3 * d_model]
        joint_h = torch.cat([trend_h, cycle_h, spike_h], dim=-1)
        rec_x = self.joint_decoder(joint_h)

        prior_loss = tp_loss+cp_loss+sp_loss / 3

        return trend_att, cycle_att, spike_att, rec_x, prior_loss*0.01