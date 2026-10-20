import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn import AttentionLayer
from .embed import DataEmbedding, PositionalEmbedding


class BoundedPrototypeBank(nn.Module):
    """
    具备正样本安全包络与对抗排斥边界的原型记忆库:
    1. 维护 K 个离散稳态工况的核心坐标 P_k
    2. 训练阶段联合优化每个原型的紧致安全半径 R_k
    """
    def __init__(self, num_prototypes=5, d_model=512, temperature=0.1, delta=0.5, nu=0.1):
        super(BoundedPrototypeBank, self).__init__()
        self.num_prototypes = num_prototypes
        self.d_model = d_model
        self.temperature = temperature
        self.delta = delta      # 对抗推离安全裕度
        self.nu = nu            # 正样本边界容忍系数

        # [K, D]: K 个工况原型中心
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
        nn.init.orthogonal_(self.prototypes)

        # [K]: 每个原型独立的正样本安全包络半径 (可学习参数)
        self.radius = nn.Parameter(torch.ones(num_prototypes) * 1.0)

    def forward(self, z, z_adv=None, training=False):
        """
        z: [B, L, D]
        z_adv: [B, L, D] 虚拟对抗逃逸难例 (仅 training=True 时传入)
        """
        B, L, D = z.shape
        z_flat = z.view(-1, D)

        # 1. 余弦软投影流形 (凸组合)
        z_norm = F.normalize(z_flat, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)
        sim = torch.matmul(z_norm, p_norm.T) / self.temperature  # [B*L, K]
        weights = F.softmax(sim, dim=-1)

        z_proto = torch.matmul(weights, self.prototypes).view(B, L, D)

        # 2. 样本到所有原型中心的欧氏几何距离
        dists = torch.cdist(z_flat, self.prototypes, p=2)  # [B*L, K]
        min_dist, nearest_idx = torch.min(dists, dim=-1)

        if training:
            matched_radius = self.radius[nearest_idx]

            # 正样本包络损失: 压紧半径 R，并对跑出安全球 R 的正常样本施加惩罚
            pos_penalty = F.relu(min_dist ** 2 - matched_radius ** 2)
            loss_pos = torch.mean(matched_radius ** 2 + (1.0 / self.nu) * pos_penalty)

            # 负样本推离损失: 强制逃逸对抗样本被排斥在 (R + delta) 之外
            loss_neg = torch.tensor(0.0, device=z.device)
            if z_adv is not None:
                z_adv_flat = z_adv.view(-1, D)
                adv_dists = torch.cdist(z_adv_flat, self.prototypes, p=2)
                min_adv_dist, _ = torch.min(adv_dists, dim=-1)
                loss_neg = torch.mean(F.relu((matched_radius + self.delta) - min_adv_dist))

            total_proto_loss = loss_pos + loss_neg
            return z_proto, total_proto_loss, weights.view(B, L, -1)

        # 推理阶段返回投影表征、到最近原型的距离及归属索引
        return z_proto, min_dist.view(B, L), nearest_idx.view(B, L), weights.view(B, L, -1)


class TemporalSpatialEncoder(nn.Module):
    """
    轻量时空特征提取骨干
    """
    def __init__(self, c_in, d_model, e_layers=3):
        super(TemporalSpatialEncoder, self).__init__()
        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.layers = nn.ModuleList([AttentionLayer(d_model) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.emb(x) + self.pos_emb(x.size(1))
        for layer in self.layers:
            h, _ = layer(h)
        return self.norm(h)


class RobustProtoAD(nn.Module):
    """
    基于边界约束原型流形的鲁棒时序异常检测与自适应网络
    """
    def __init__(self, win_size, c_in, c_out, d_model=512, e_layers=3, num_prototypes=5,
                 delta=0.5, nu=0.1, epsilon=0.05):
        super(RobustProtoAD, self).__init__()
        self.win_size = win_size
        self.d_model = d_model
        self.epsilon = epsilon

        # 1. 时空编码骨干
        self.encoder = TemporalSpatialEncoder(c_in, d_model, e_layers=e_layers)

        # 2. 具备正负边界约束的原型记忆库
        self.proto_bank = BoundedPrototypeBank(
            num_prototypes=num_prototypes,
            d_model=d_model,
            delta=delta,
            nu=nu
        )

        # 3. 稳态流形解码重构器
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, c_out)
        )

    def generate_adversarial_perturbation(self, z):
        """
        沿原型吸引势能下降最快方向生成虚拟对抗逃逸难例
        """
        z_copy = z.detach().clone().requires_grad_(True)
        _, loss_pos, _ = self.proto_bank(z_copy, training=True)
        grad = torch.autograd.grad(loss_pos, z_copy, retain_graph=False, create_graph=False)[0]
        z_adv = z + self.epsilon * torch.sign(grad)
        return z_adv

    def forward(self, x, training=False):
        z = self.encoder(x)  # [B, L, D]

        if training:
            z_adv = self.generate_adversarial_perturbation(z)
            z_proto, proto_loss, proto_weights = self.proto_bank(z, z_adv=z_adv, training=True)
            rec_x = self.decoder(z_proto)
            return {
                'rec_x': rec_x,
                'proto_loss': proto_loss,
                'proto_weights': proto_weights
            }
        else:
            z_proto, min_dist, nearest_idx, proto_weights = self.proto_bank(z, training=False)
            rec_x = self.decoder(z_proto)
            latent_diff = torch.norm(z - z_proto, p=2, dim=-1)  # 隐空间流形偏离度 [B, L]
            return {
                'z': z,
                'rec_x': rec_x,
                'min_dist': min_dist,
                'nearest_idx': nearest_idx,
                'latent_diff': latent_diff,
                'proto_weights': proto_weights
            }