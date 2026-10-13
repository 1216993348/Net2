import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 基础组件 ====================

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x):
        return self.value_embedding(x) + self.position_embedding(x)


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super(AttentionLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.q_linear(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_linear(out)
        return out, attn


# ==================== Transformer重建器 ====================

class TransformerReconstructor(nn.Module):
    def __init__(self, d, T, d_model=256, n_heads=8, num_layers=3):
        super(TransformerReconstructor, self).__init__()
        self.d = d
        self.T = T
        self.d_model = d_model
        self.embedding = DataEmbedding(d, d_model)
        self.pos_embed = PositionalEmbedding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(d_model, d)

    def forward(self, x):
        # x: [B, d, T] -> 转置为 [B, T, d]
        x = x.permute(0, 2, 1)  # [B, T, d]
        embedded = self.embedding(x) + self.pos_embed(x)  # [B, T, d_model]
        enc_out = self.encoder(embedded)
        dec_out = self.decoder(enc_out, enc_out)
        out = self.proj_out(dec_out)  # [B, T, d]
        return out.permute(0, 2, 1)  # [B, d, T]


# ==================== GAT层 ====================

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.a = nn.Parameter(torch.randn(2 * out_dim, 1) * 0.01)

    def forward(self, h, adj):
        B, d, in_dim = h.shape
        Wh = torch.matmul(h, self.W)  # [B, d, out_dim]
        Wh1 = Wh.unsqueeze(2).expand(-1, -1, d, -1)
        Wh2 = Wh.unsqueeze(1).expand(-1, d, -1, -1)
        concat = torch.cat([Wh1, Wh2], dim=-1)
        e = torch.matmul(concat, self.a).squeeze(-1)
        e = F.leaky_relu(e, negative_slope=0.2)
        adj_mask = (adj > 0).float().unsqueeze(0)
        e = e.masked_fill(adj_mask == 0, -1e9)
        alpha = F.softmax(e, dim=-1)
        out = torch.matmul(alpha, Wh)
        return out, alpha


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super(GATEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphAttentionLayer(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim))
        self.layers.append(GraphAttentionLayer(hidden_dim, out_dim))

    def forward(self, x, adj):
        h = x
        for layer in self.layers:
            h, _ = layer(h, adj)
            h = F.relu(h)
        return h


# ==================== 分块时序卷积（修复版） ====================

class ChunkTemporalConv(nn.Module):
    def __init__(self, d, in_dim, num_chunks=5, kernel_size=3):
        super(ChunkTemporalConv, self).__init__()
        self.num_chunks = num_chunks
        # 输入通道数 = 变量数 d，输出通道数 = d
        self.conv = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        # x: [B, d, D]
        B, d, D = x.shape
        chunk_size = D // self.num_chunks
        chunks = torch.chunk(x, self.num_chunks, dim=-1)
        out_chunks = [self.conv(chunk) for chunk in chunks]
        return torch.cat(out_chunks, dim=-1)


# ==================== SPAGD主模型 ====================

class SPAGD(nn.Module):
    def __init__(self, d, T, hidden_dim=256, gat_layers=2, num_chunks=5,
                 top_k=10, percent_m=10, beta=0.01):
        super(SPAGD, self).__init__()
        self.d = d
        self.T = T
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.percent_m = percent_m
        self.beta = beta

        # 重建器
        self.reconstructor = TransformerReconstructor(d, T, d_model=hidden_dim)

        # GAT
        self.gat = GATEncoder(in_dim=T, hidden_dim=hidden_dim, out_dim=hidden_dim, num_layers=gat_layers)

        # 时序卷积（传入d）
        self.tcn = ChunkTemporalConv(d=d, in_dim=hidden_dim, num_chunks=num_chunks)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d * hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 静态图（训练时构建）
        self.register_buffer('A_static', None)

    def build_static_graph(self, X):
        """基于正常数据构建初始静态图"""
        # X: [B, d, T]，取batch平均
        X_mean = X.mean(dim=0)  # [d, T]
        X_norm = F.normalize(X_mean, dim=-1)  # [d, T]
        S = torch.mm(X_norm, X_norm.T)  # [d, d]
        S = torch.sigmoid(S)

        # Top-K稀疏化
        A = torch.zeros_like(S)
        topk_vals, topk_idx = torch.topk(S, self.top_k, dim=-1)
        for i in range(self.d):
            A[i, topk_idx[i]] = topk_vals[i]
        return A

    def compute_residuals(self, X, X_recon):
        """计算每个变量的重建残差"""
        return torch.abs(X - X_recon).mean(dim=-1).mean(dim=0)

    def adjust_graph(self, A, residuals):
        """基于残差动态调整图"""
        d = self.d
        m = max(1, int(d * self.percent_m / 100))
        _, top_idx = torch.topk(residuals, m)
        mask = torch.zeros(d, dtype=torch.bool, device=residuals.device)
        mask[top_idx] = True

        phi = torch.sigmoid(residuals)  # [d]

        # 调整相似度
        S_adj = A.clone()
        for i in range(d):
            for j in range(d):
                if mask[i] or mask[j]:
                    add = 0.0
                    if mask[i]:
                        add += phi[i]
                    if mask[j]:
                        add += phi[j]
                    if i == j:
                        add = phi[i]
                    S_adj[i, j] = A[i, j] + add

        # 对称化 + Top-K稀疏化
        S_adj = (S_adj + S_adj.T) / 2
        A_adj = torch.zeros_like(S_adj)
        topk_vals, topk_idx = torch.topk(S_adj, self.top_k, dim=-1)
        for i in range(d):
            A_adj[i, topk_idx[i]] = topk_vals[i]
        return A_adj

    def forward(self, X, mode='train'):
        """
        X: [B, d, T]
        mode: 'train' 或 'test'
        """
        B = X.shape[0]

        # ===== 训练模式 =====
        if mode == 'train':
            # 1. 重建
            X_recon = self.reconstructor(X)  # [B, d, T]
            L_sp = F.mse_loss(X_recon, X)

            # 2. 构建/更新静态图
            if self.A_static is None:
                self.A_static = self.build_static_graph(X)

            # 3. 动态图
            residuals = self.compute_residuals(X, X_recon)
            A_dynamic = self.adjust_graph(self.A_static, residuals)

            # 4. 双路GAT
            Z_norm = self.gat(X, self.A_static)  # [B, d, hidden_dim]
            Z_pert = self.gat(X_recon, A_dynamic)  # [B, d, hidden_dim]

            # 5. 时序卷积
            Z_norm = self.tcn(Z_norm)
            Z_pert = self.tcn(Z_pert)

            # 6. 分类
            Z_stack = torch.cat([Z_norm, Z_pert], dim=0)  # [2B, d, hidden_dim]
            Z_flat = Z_stack.view(Z_stack.size(0), -1)  # [2B, d*hidden_dim]
            pred = self.classifier(Z_flat).squeeze(-1)  # [2B]

            labels = torch.cat([torch.zeros(B), torch.ones(B)], dim=0).to(X.device)
            L_ad = F.binary_cross_entropy(pred, labels)

            # 总损失
            L_total = L_sp + self.beta * L_ad

            return {
                'loss': L_total,
                'L_sp': L_sp,
                'L_ad': L_ad,
                'X_recon': X_recon,
                'pred': pred,
                'residuals': residuals,
                'A_dynamic': A_dynamic
            }

        # ===== 测试模式 =====
        else:
            A = self.A_static if self.A_static is not None else torch.eye(self.d).to(X.device)
            Z = self.gat(X, A)
            Z = self.tcn(Z)
            Z_flat = Z.view(Z.size(0), -1)
            scores = self.classifier(Z_flat).squeeze(-1)
            return scores