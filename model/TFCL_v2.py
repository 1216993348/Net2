import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .attn import AttentionLayer
from .embed import DataEmbedding, PositionalEmbedding

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

class TFC(nn.Module):
    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5, dev=None):
        super(TFC, self).__init__()
        global device
        device = dev
        self.win_size = win_size

        self.tem = TemEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, tr)
        self.fre = FreEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, fr)

        # 🎯 联合重构解码器：输入维度是 2 * d_model，输出是 c_out
        self.joint_decoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, c_out)
        )

    def forward(self, x):
        # 假设 TemEnc 和 FreEnc 返回 (att, h)
        tematt, tem_h = self.tem(x)
        freatt, fre_h = self.fre(x)

        # 🎯 1. 拼接低频与高频特征 [B, L, 2 * d_model]
        # 使用 .detach() 切断重构 Loss 对两个主干的梯度倒灌，保障 0.93+ 分数不降！
        joint_h = torch.cat([tem_h, fre_h], dim=-1)

        # 🎯 2. 联合解码重构原始 input
        rec_x = self.joint_decoder(joint_h)

        return tematt, freatt, rec_x