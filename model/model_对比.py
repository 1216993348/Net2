import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .attn import AttentionLayer
from .embed import DataEmbedding, PositionalEmbedding


class TrendEnc(nn.Module):

    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size):
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

            low_pass_mask_3d = low_pass_mask.squeeze(1)  # [B, L, L]

            prior_loss += F.kl_div(
                F.log_softmax(low_pass_mask_3d, dim=-1),
                F.softmax(attn_w, dim=-1),
                reduction='batchmean'
            )

            final_attn = attn_w + 1e-6
            final_attn = final_attn / final_attn.sum(dim=-1, keepdim=True)
            att.append(final_attn)

        return att, h, prior_loss


class SpikeEnc(nn.Module):

    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size):
        super(SpikeEnc, self).__init__()
        self.emb = DataEmbedding(c_in, d_model)
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
        prior_loss = 0.0
        for i in range(self.e_layers):
            h, attn_w = self.attn_layers[i](h)
            final_attn = attn_w + 1e-6
            final_attn = final_attn / final_attn.sum(dim=-1, keepdim=True)
            att.append(final_attn)
        return att, h, prior_loss


class StructEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size):
        super(StructEnc, self).__init__()
        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.attn_layers = nn.ModuleList([AttentionLayer(d_model) for _ in range(e_layers)])

    def forward(self, x):
        # x: [B, T, C]
        h = self.emb(x) # [B, T, D]
        h = h + self.pos_emb(x.size(1))
        attlist = []
        for attn_layer in self.attn_layers:
            h, _ = attn_layer(h)
            attlist.append(_)
        return attlist, h # att(list): [B, T, T]


def channel_mask(x, mask_ratio=0.1):
    # x: [B, L, C]
    B, L, C = x.shape

    mask = torch.rand(B, C, device=x.device) < mask_ratio

    x_masked = x.clone()
    mask = mask.unsqueeze(1).expand(-1, L, -1)
    x_masked[mask] = 0

    return x_masked, mask


class TriDecompNet(nn.Module):
    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, dev=None):
        super(TriDecompNet, self).__init__()
        global device
        device = dev
        self.win_size = win_size

        # 1. 第一级：趋势提取器
        self.trend = TrendEnc(c_in, c_out, d_model, e_layers, win_size, seq_size)
        # 2. 突变高通分支
        self.spike = SpikeEnc(c_in, c_out, d_model, e_layers, win_size, seq_size)
        # 3. 结构化全局分支
        self.gs = StructEnc(c_in, c_out, d_model, e_layers, win_size, seq_size)

        # 🎯 三分支联合解码器：输入维度是 2 * d_model，输出是 c_out
        self.joint_decoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, c_out)
        )

    def forward(self, x, training=False):
        mask = None
        # if training:
        #     x, mask = channel_mask(x, 0.20)

        trend_att, trend_h, tp_loss = self.trend(x)
        spike_att, spike_h, sp_loss = self.spike(x)
        gs_att, gs_h = self.gs(x)

        # 联合重构：拼接两个层级的特征 [B, L, 2 * d_model]
        joint_h = torch.cat([trend_h, spike_h], dim=-1)
        rec_x = self.joint_decoder(joint_h)

        prior_loss = tp_loss + sp_loss

        return trend_att, gs_att, spike_att, rec_x, prior_loss*0.01, mask, [trend_h, gs_att, spike_h]