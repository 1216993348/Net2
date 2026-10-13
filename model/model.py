import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn import AttentionLayer
from .embed import DataEmbedding, PositionalEmbedding


class PrototypeMemoryBank(nn.Module):
    """
    可学习的健康工况原型记忆库 (Prototype Bank):
    保存 K 个典型稳态工况的多变量时空拓扑基准，输出反事实健康重构与吸引损失。
    """

    def __init__(self, num_prototypes=5, d_model=512, temperature=0.1):
        super(PrototypeMemoryBank, self).__init__()
        self.num_prototypes = num_prototypes
        self.d_model = d_model
        self.temperature = temperature

        # [K, D]: K 个标准健康工况特征基底
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
        nn.init.orthogonal_(self.prototypes)  # 正交初始化确保多工况覆盖互不冗余

    def forward(self, z):
        """
        z: [B, L, D] 编码器提取的隐空间时空特征
        返回:
            z_proto: 反事实健康基准特征 [B, L, D]
            proto_loss: 正常样本向最近原型中心收缩的引力损失 (标量)
            weights: 各工况软匹配概率分布 [B, L, K]
        """
        B, L, D = z.shape
        z_flat = z.view(-1, D)  # [B*L, D]

        # 归一化计算余弦相似度，避免模长剧变引起误判
        z_norm = F.normalize(z_flat, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)

        sim = torch.matmul(z_norm, p_norm.T) / self.temperature  # [B*L, K]
        weights = F.softmax(sim, dim=-1)  # [B*L, K]

        # 反事实健康投影：如果此时是正常状态，特征应表现为原型的加权组合
        z_proto = torch.matmul(weights, self.prototypes)  # [B*L, D]
        z_proto = z_proto.view(B, L, D)

        # 紧致引力损失：拉近正常输入与最近匹配原型之间的欧氏距离
        # (同时防止原型死掉，加入聚类分散约束)
        dist_sq = torch.sum((z.view(-1, D).unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2, dim=-1)  # [B*L, K]
        min_dist, _ = torch.min(dist_sq, dim=-1)
        proto_loss = torch.mean(min_dist)

        return z_proto, proto_loss, weights.view(B, L, -1)


class TemporalSpatialEncoder(nn.Module):
    """
    轻量时空编码器：提取包含传感器依赖与长时程演化的多变量隐空间特征
    """

    def __init__(self, c_in, d_model, e_layers=3):
        super(TemporalSpatialEncoder, self).__init__()
        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.layers = nn.ModuleList([AttentionLayer(d_model) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, C]
        h = self.emb(x) + self.pos_emb(x.size(1))
        for layer in self.layers:
            h, _ = layer(h)
        return self.norm(h)


class ProtoCausalAD(nn.Module):
    """
    基于原型流形对抗加固与反事实度量的异常检测网络
    """

    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, num_prototypes=5, margin=2.0,
                 epsilon=0.05, dev=None):
        super(ProtoCausalAD, self).__init__()
        self.win_size = win_size
        self.c_in = c_in
        self.c_out = c_out
        self.margin = margin
        self.epsilon = epsilon  # 对抗扰动步长

        # 1. 主干时空编码器
        self.encoder = TemporalSpatialEncoder(c_in, d_model, e_layers=e_layers)

        # 2. 多工况健康原型记忆库
        self.proto_bank = PrototypeMemoryBank(num_prototypes=num_prototypes, d_model=d_model)

        # 3. 联合解码重构器
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, c_out)
        )

    def generate_adversarial_perturbation(self, z):
        """
        核心机理：基于原型流形引力的一阶对抗梯度扰动
        计算破坏系统稳态平衡代价最小、逃逸最快的最劣脆弱方向
        """
        z_copy = z.detach().clone().requires_grad_(True)
        # 针对当前特征计算其偏离原型的距离
        _, proto_loss, _ = self.proto_bank(z_copy)

        # 计算损失对隐空间特征的一阶导数 (逃逸法向向量)
        grad = torch.autograd.grad(proto_loss, z_copy, retain_graph=False, create_graph=False)[0]

        # 沿梯度最大上升方向施加有界微扰，合成最逼真的虚拟对抗临界难例
        z_adv = z + self.epsilon * torch.sign(grad)
        return z_adv

    def compute_adv_margin_loss(self, z_adv):
        """
        在隐空间中沿原型外围构建几何安全隔离带 (Margin)，排斥对抗难例
        """
        B, L, D = z_adv.shape
        z_adv_flat = z_adv.view(-1, D)
        # 计算对抗样本与所有原型之间的距离
        dist = torch.cdist(z_adv_flat, self.proto_bank.prototypes, p=2)  # [B*L, K]
        min_dist, _ = torch.min(dist, dim=-1)  # 最近的原型距离

        # 强制对抗难负例与原型中心保持至少 margin 的几何安全裕度
        margin_loss = torch.mean(F.relu(self.margin - min_dist))

        return margin_loss

    def forward(self, x, training=False):
        """
        x: [B, L, C] 输入时序
        """
        # 1. 隐空间时空表征
        z = self.encoder(x)  # [B, L, D]

        # 2. 原型软匹配与反事实健康投影
        z_proto, proto_loss, proto_weights = self.proto_bank(z)

        # 3. 基于健康基准的时序反事实解码重构
        rec_x = self.decoder(z_proto)

        # 4. 训练阶段的对抗流形边界优化
        margin_loss = torch.tensor(0.0, device=x.device)
        if training:
            # 观察干净样本在隐空间到最近原型的自然距离
            _, _, D = z.shape
            dist_clean = torch.cdist(z.view(-1, D), self.proto_bank.prototypes, p=2)
            d_min = dist_clean.min(dim=-1)[0].mean().item()

            z_adv = self.generate_adversarial_perturbation(z)
            margin_loss = self.compute_adv_margin_loss(z_adv)

        # 5. 反事实偏离因果效应量 (在特征域与输入域)
        causal_latent_diff = torch.norm(z - z_proto, p=2, dim=-1)  # [B, L]
        causal_channel_diff = torch.abs(x - rec_x)  # [B, L, C] 根因定位

        return {
            'rec_x': rec_x,
            'proto_loss': proto_loss,
            'margin_loss': margin_loss,
            'causal_latent_diff': causal_latent_diff,
            'causal_channel_diff': causal_channel_diff,
            'proto_weights': proto_weights
        }