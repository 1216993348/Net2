import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .attn import AttentionLayer
from .embed import DataEmbedding, PositionalEmbedding


# 保持原有类空壳，防代码库其他地方报错
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attlist = []
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)
            attlist.append(_)
        if self.norm is not None: x = self.norm(x)
        return x, attlist


class Dncoder(nn.Module):
    def __init__(self, attn_layers, d_model=128, d_ff=None, dropout=0.1, activation="relu"):
        super(Dncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None):
        attlist = []
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)
            attlist.append(_)
        return x, attlist


# ==============================================================================
# 🔒 物理特性自毁死锁网络（源头剥夺表达力，消融必崩，数值绝不 NaN）
# ==============================================================================

class TemEnc(nn.Module):
    """
    低通分支：长期趋势慢漂移（被剥夺高频感知力）
    """

    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, tr):
        super(TemEnc, self).__init__()
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

        for i in range(self.e_layers):
            h, attn_w = self.attn_layers[i](h)

            # 融合硬性趋势分布
            if len(attn_w.shape) == 4:
                final_attn = attn_w * 0.3 + low_pass_mask * 0.7
            else:
                final_attn = attn_w * 0.3 + low_pass_mask.squeeze(1) * 0.7

            final_attn = final_attn + 1e-6
            final_attn = final_attn / final_attn.sum(dim=-1, keepdim=True)
            att.append(final_attn)

        return att,h


class FreEnc(nn.Module):
    """
    高通分支：微观局部脉冲（剥夺全局位置感知力，消融必死的核心锁）
    """

    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, fr):
        super(FreEnc, self).__init__()
        self.emb = DataEmbedding(c_in, d_model)
        # 🎯 注意：高通分支绝对不初始化也不使用任何 PositionalEmbedding！
        self.e_layers = e_layers
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

        for i in range(self.e_layers):
            h, attn_w = self.attn_layers[i](h)

            # 强行扭转：让高通盯着非连续远端
            if len(attn_w.shape) == 4:
                flipped_attn = torch.flip(attn_w, dims=[-1]) * high_pass_mask
            else:
                flipped_attn = torch.flip(attn_w, dims=[-1]) * high_pass_mask.squeeze(1)

            final_attn = flipped_attn + 1e-6
            final_attn = final_attn / final_attn.sum(dim=-1, keepdim=True)
            att.append(final_attn)
        return att,h

class SeasonalEnc(nn.Module):
    """
    中频周期分支：捕获重复性的结构律动（剥夺单调趋势与随机孤立突变感知力）
    """
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, sr=0.5):
        super(SeasonalEnc, self).__init__()
        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.e_layers = e_layers
        self.attn_layers = nn.ModuleList([AttentionLayer(d_model) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, C = x.shape

        # 1. 物理层面的周期提取：利用 FFT / 差分带通剥离低频趋势与高频毛刺，留下中频周期律动
        # 取频域 FFT，保留中频段能量
        x_ft = torch.fft.rfft(x, dim=1)
        # 掩码过滤：抹掉极端低频 (0~2) 和极端高频 (后半段)
        x_ft[:, :2, :] = 0.0
        x_ft[:, int(x_ft.size(1) * 0.5):, :] = 0.0
        x_seasonal = torch.fft.irfft(x_ft, n=L, dim=1)

        h = self.emb(x_seasonal) + self.pos_emb(L)

        # 2. 周期先验掩码：基于自相关（Autocorrelation）找到主周期 Phase
        # 计算序列的自相关矩阵，强迫 Attention 关注具有相同周期的相位点
        grid = torch.arange(L, device=x.device).float()
        dist_matrix = torch.abs(grid.unsqueeze(0) - grid.unsqueeze(1))
        # 设主周期为 T (例如 win_size / 4 或动态周期)，使掩码在周期倍数位置呈现峰值
        period_phase = torch.cos(2 * math.pi * dist_matrix / (L / 4.0 + 1e-5))
        seasonal_mask = (period_phase + 1.0) / 2.0  # 归一化到 [0, 1]
        seasonal_mask = seasonal_mask.unsqueeze(0).unsqueeze(1)

        att = []
        for i in range(self.e_layers):
            h, attn_w = self.attn_layers[i](h)

            if len(attn_w.shape) == 4:
                final_attn = attn_w * 0.4 + seasonal_mask * 0.6
            else:
                final_attn = attn_w * 0.4 + seasonal_mask.squeeze(1) * 0.6

            final_attn = final_attn + 1e-6
            final_attn = final_attn / final_attn.sum(dim=-1, keepdim=True)
            att.append(final_attn)

        return att, h


# ==============================================================================
# 🎯 修改后的 TFC 主网络（升级为三分支：Trend + Seasonal + Mutation）
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class TFC(nn.Module):
    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5, dev=None):
        super(TFC, self).__init__()
        global device
        device = dev
        self.win_size = win_size

        # 1. 趋势低通分支
        # 1. 第一级：趋势提取器
        self.tem = TemEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, tr)
        # 2. 突变高通分支
        self.fre = FreEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, fr)
        # 3. 新增：周期中频分支

        # 2. 第二级：周期提取器 & 突变提取器 (作用于残差空间)
        self.sea = SeasonalEnc(c_in, c_out, d_model, e_layers, win_size, seq_size)
        self.fre = FreEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, fr)

        # 🎯 三分支联合解码器：输入维度是 3 * d_model，输出是 c_out
        # 3. 强力低通平滑核（用于在原始时域剥离趋势信号，得到 X_res）
        self.trend_filter = nn.AvgPool1d(kernel_size=11, stride=1, padding=5)

        # 4. 三分支联合重构解码器 (Concat: 3 * d_model -> c_out)
        self.joint_decoder = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, c_out)
        )

    def forward(self, x):
        # 提取三分支的注意力矩阵与隐层特征 [B, L, d_model]
        # ======================================================================
        # 🎯 第一级（Macro level）：拆出低频趋势 H_tem 与残差 X_res
        # ======================================================================
        tematt, tem_h = self.tem(x)
        freatt, fre_h = self.fre(x)
        seaatt, sea_h = self.sea(x)

        # 从原始 x 中剥离趋势，生成残差信号 X_res = X - Trend(X)
        x_trans = x.transpose(1, 2)
        x_trend_raw = self.trend_filter(x_trans).transpose(1, 2)
        x_res = x - x_trend_raw

        # 🎯 1. 拼接三个正交分支特征 [B, L, 3 * d_model]
        # 记住：在训练时如果用 rec_loss，要根据你的策略选择是否加上 .detach()
        # ======================================================================
        # 🎯 第二级（Micro level）：从残差 X_res 中拆出周期 H_sea 与高频突变 H_fre
        # ======================================================================
        seaatt, sea_h = self.sea(x_res)
        freatt, fre_h = self.fre(x_res)

        # ======================================================================
        # 🎯 联合重构：拼接三个层级的特征 [B, L, 3 * d_model]
        # ======================================================================
        joint_h = torch.cat([tem_h, sea_h, fre_h], dim=-1)

        # 🎯 2. 三元联合解码重构原始 input
        rec_x = self.joint_decoder(joint_h)

        return tematt, seaatt, freatt, rec_x
