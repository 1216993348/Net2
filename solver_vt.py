import torch
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from model.model import TriDecompNet
from data_factory.data_loader import get_loader_segment
import torch.nn.functional as F
import matplotlib.pyplot as plt

def my_kl_loss(p, q):
    """KL 散度计算，带数值稳定微量 eps"""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)


def my_cross_entropy(p, q):
    """
    交叉熵损失：-Σ p * log(q)
    适用于离散概率分布 p 和 q
    """
    return -torch.sum(p * (torch.log(q + 0.0001)), dim=-1)
def compute_discrepancy(input, output, temperature=1.0, keep_batch=False):
    trend_att, gs_att, spike_att, rec_x, prior_loss, mask, _ = output

    con_loss_total = 0.0
    layers_num = len(trend_att)

    for u in range(layers_num):
        trend_norm = trend_att[u]
        gs_norm = gs_att[u]
        spike_norm = spike_att[u]

        # 融合特征：趋势 + 突变
        res_norm = (trend_norm + spike_norm) / 2.0

        if keep_batch:
            # ===== 测试阶段：保留 [B, L] =====
            # G → F：用全局关系去"预测"融合特征
            con_forward = my_kl_loss(gs_norm, res_norm.detach())
            # F → G：用融合特征去"预测"全局关系
            con_backward = my_kl_loss(res_norm.detach(), gs_norm)
            con_macro = con_forward + con_backward
            con_loss_total += con_macro * (temperature / 2.0)
        else:
            # ===== 训练阶段：标量 =====
            con_forward = torch.mean(my_kl_loss(gs_norm, res_norm.detach()))
            con_backward = torch.mean(my_kl_loss(res_norm.detach(), gs_norm))
            con_macro = con_forward + con_backward
            con_loss_total += con_macro * (temperature / 2.0)

    con_loss = con_loss_total / layers_num

    # 重建损失
    if mask is not None:
        input = input[mask]
        rec_x = rec_x[mask]

    if keep_batch:
        rec_loss = torch.mean((input - rec_x) ** 2, dim=-1)
    else:
        rec_loss = torch.mean((input - rec_x) ** 2)

    return 0, con_loss, rec_loss, prior_loss

def compute_discrepancy2(input, output, temperature=1.0, keep_batch=False):

    trend_att, gs_att, spike_att, rec_x,prior_loss,mask,_ = output
    con_loss_total = 0.0
    layers_num = len(trend_att)

    for u in range(layers_num):
        # 归一化 Attention 概率分布
        trend_norm = trend_att[u]
        gs_att = gs_att[u]
        spike_norm = spike_att[u]

        # 🎯 残差部分的复合注意力：将微观的周期与高频融合，作为"残差总表达"
        res_norm = (trend_norm + spike_norm) / 2.0

        if keep_batch:
            # ===== 测试/评估阶段：保留 [B, L] 维数 =====
            # 1. 宏观对抗：趋势 trend_norm vs 残差总表达 res_norm
            con_macro = my_kl_loss(gs_att, res_norm.detach()) + my_kl_loss(res_norm.detach(), gs_att)

            # 组合两级对抗能量
            con_loss_total += (con_macro) * (temperature / 2.0)

        else:
            # ===== 训练阶段：归一化为标量 =====
            # 1. 宏观对抗
            con_macro = torch.mean(my_kl_loss(gs_att, res_norm.detach()) + my_kl_loss(res_norm.detach(), gs_att))

            # 组合两级 Loss
            con_loss_total += (con_macro ) * (temperature / 2.0)

    con_loss = con_loss_total / layers_num
    # ===== 重建损失计算 =====
    if mask != None:
        input = input[mask]
        rec_x = rec_x[mask]
    # ===== 重建损失计算 =====
    if keep_batch:
        rec_loss = torch.mean((input - rec_x) ** 2, dim=-1)
    else:
        rec_loss = torch.mean((input - rec_x) ** 2)

    return 0, con_loss, rec_loss, prior_loss


def adjust_predictions(gt, pred):
    """Point Adjustment: 异常段内任意点被识别则全段标记为异常"""
    pred = pred.copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre', dataset=self.dataset)

        # 🎯 三端通用（CPU / CUDA / Intel XPU）自动设备探测
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = torch.device('xpu')
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu}")
        else:
            self.device = torch.device('cpu')

        print(f"[Solver] Using execution device: {self.device}")
        self.build_model()

    def build_model(self):
        self.model = TriDecompNet(
            win_size=self.win_size,
            seq_size=self.seq_size,
            c_in=self.input_c,
            c_out=self.output_c,
            d_model=self.d_model,
            e_layers=self.e_layers,
            dev=self.device
        ).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] Total params: {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"[Model] Trainable params: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def vali(self, vali_loader):
        self.model.eval()
        loss_list = []
        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                output = self.model(input)

                adv_loss, con_loss, rec_loss,prior_loss = compute_discrepancy(input, output, temperature=1.0)
                if getattr(self, 'use_recon', False):
                    loss_list.append((rec_loss + con_loss + prior_loss).item())
                else:
                    loss_list.append((con_loss + prior_loss ).item())

        return np.average(loss_list)

    def train(self):
        print("======================TRAIN MODE======================")

        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(self.train_loader)

        for epoch in tqdm(range(self.num_epochs), disable=not self.print_tqdm):
            start_time = time.time()
            loss_list = []

            self.model.train()
            with tqdm(total=train_steps, disable=not self.print_tqdm) as pbar:
                for i, (input_data, labels) in enumerate(self.train_loader):
                    self.optimizer.zero_grad()

                    input = input_data.float().to(self.device)
                    output = self.model(input,True)

                    adv_loss, con_loss, rec_loss,prior_loss = compute_discrepancy(input, output, temperature=1.0)
                    if getattr(self, 'use_recon', False):
                        loss = rec_loss + con_loss + prior_loss
                    else:
                        loss = con_loss + prior_loss

                    loss_list.append(loss.item())
                    pbar.update(1)
                    loss.backward()
                    self.optimizer.step()

            train_loss = np.average(loss_list)
            vali_loss = self.vali(self.vali_loader)

            torch.save(self.model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))

            elapsed_time = time.time() - start_time

            # 🎯 内存/显存打印防崩溃保护
            if self.device.type == 'cuda':
                memory_usage = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            else:
                memory_usage = 0.0

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | Time: {4:.2f}s | Memory: {5:.2f}MB".format(
                    epoch + 1, train_steps, train_loss, vali_loss, elapsed_time, memory_usage))

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import os

    def visualize_attn_diff(self, num_abnormal=5, num_normal=5,
                            save_dir='./attn_vis',
                            sym_kl=True,
                            use_symmetric_matrix=True):
        """
        每层画出 4 个矩阵热力图（趋势 att / 结构 att / 突变 att / |GS-R| 差异矩阵），
        并在每个矩阵的 x 轴（key 时间）与 y 轴（query 时间）上，
        逐点标出 GT 异常位置（蓝色短竖线/横线），用来肉眼判断：
            —— 差异矩阵里亮的地方，是否正好落在异常时刻。
        GT 严格逐点，绝不连成段。
        """
        os.makedirs(save_dir, exist_ok=True)

        ckpt = os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)
        self.model.eval()

        # ---------------- 工具 ----------------
        def _sym(M):
            return (M + M.T) / 2.0 if use_symmetric_matrix else M

        def _sym_kl(p, q, eps=1e-4):
            p = p / (p.sum(-1, keepdims=True) + 1e-8)
            q = q / (q.sum(-1, keepdims=True) + 1e-8)
            kl_f = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)
            kl_b = np.sum(q * (np.log(q + eps) - np.log(p + eps)), axis=-1)
            return (kl_f + kl_b) / 2.0  # [L]

        def _diff_score(g, r):
            """逐时刻差异分数 [L]：与 solver 的 con_loss 口径一致"""
            if sym_kl:
                return _sym_kl(g, r)
            return np.abs(g - r).mean(axis=1)

        # ---------------- 收集样本 ----------------
        abnormal_pool, normal_pool = [], []

        with torch.no_grad():
            for input_data, labels in self.test_loader:
                input = input_data.float().to(self.device)
                output = self.model(input)
                trend_att, gs_att, spike_att, _, _, _, _ = output
                B, L = input.shape[0], input.shape[1]

                # GT 逐点向量 [B, L]
                if labels.ndim >= 2 and labels.shape[-1] == L:
                    lbl_all = labels.numpy()[:B, :L]
                else:
                    lbl_all = np.repeat(labels.numpy().reshape(B, -1)[:, :1], L, axis=1)
                lbl_all = (lbl_all > 0).astype(float)

                for b in range(B):
                    def _layer_att(att_list):
                        out = []
                        for l in range(len(att_list)):
                            a = att_list[l][b].detach().cpu().numpy()
                            if a.ndim == 3:
                                a = a.mean(axis=0)
                            out.append(_sym(a))
                        return out

                    sample = {
                        'trend': _layer_att(trend_att),
                        'gs': _layer_att(gs_att),
                        'spike': _layer_att(spike_att),
                        'label': lbl_all[b],  # [L] 0/1 逐点
                    }
                    is_abn = sample['label'].max() > 0
                    if is_abn and len(abnormal_pool) < num_abnormal:
                        abnormal_pool.append(sample)
                    elif (not is_abn) and len(normal_pool) < num_normal:
                        normal_pool.append(sample)

                if len(abnormal_pool) >= num_abnormal and len(normal_pool) >= num_normal:
                    break

        print(f"[Vis] collected: abnormal={len(abnormal_pool)}, normal={len(normal_pool)}")

        # ---------------- 单样本绘图 ----------------
        def plot_one(sample, title_prefix, fname):
            n_layers = len(sample['trend'])
            L = sample['trend'][0].shape[0]
            extent = [0, 100, 100, 0]  # x/y 均按 0~100 时间百分比
            label = sample['label']  # [L] 0/1 逐点
            anom_idx = np.where(label > 0)[0]
            anom_pct = 100.0 * anom_idx / max(L - 1, 1)  # 逐点 -> 百分比位置

            fig, axes = plt.subplots(
                n_layers, 4,
                figsize=(17, 3.9 * n_layers)
            )
            if n_layers == 1:
                axes = axes[None, :]

            for l in range(n_layers):
                t = sample['trend'][l]
                g = sample['gs'][l]
                s = sample['spike'][l]
                r = (t + s) / 2.0
                D = np.abs(g - r)  # 差异矩阵（保留，可视化核心）

                # 逐时刻差异分数（仅用于标题量化，不取代矩阵）
                score = _diff_score(g, r)
                if score.max() - score.min() > 1e-8:
                    score_n = (score - score.min()) / (score.max() - score.min())
                else:
                    score_n = np.zeros_like(score)
                peak_pct = 100.0 * int(np.argmax(score_n)) / max(L - 1, 1)

                panels = [
                    (t, 'viridis', f'TrendAtt L{l}'),
                    (g, 'viridis', f'GS(Struct)Att L{l}'),
                    (s, 'viridis', f'SpikeAtt L{l}'),
                    (D, 'hot', f'|GS - R| L{l}\nmeanDiff={D.mean():.4f}  peak@{peak_pct:.0f}%'),
                ]

                for j, (mat, cmap, title) in enumerate(panels):
                    ax = axes[l, j]
                    # 保留矩阵可视化
                    im = ax.imshow(mat, cmap=cmap, aspect='auto', extent=extent)
                    ax.set_title(title, fontsize=9)
                    ax.set_xlabel('key time (%)', fontsize=7)
                    if j == 0:
                        ax.set_ylabel('query time (%)', fontsize=7)
                    # 对角线参考
                    ax.plot([0, 100], [0, 100], color='w', lw=0.4, alpha=0.5)

                    # ⭐ 关键：逐点标 GT 异常位置
                    #   x 轴：异常时刻作为 key
                    #   y 轴：异常时刻作为 query
                    #   短竖线/短横线，各自独立，绝不连成段
                    for p in anom_pct:
                        ax.plot([p, p], [0, 2.5], color='cyan', lw=0.7, alpha=0.9)  # 顶部刻度
                        ax.plot([p, p], [97.5, 100], color='cyan', lw=0.7, alpha=0.9)  # 底部刻度
                        ax.plot([0, 2.5], [p, p], color='deepskyblue', lw=0.7, alpha=0.9)  # 左侧
                        ax.plot([97.5, 100], [p, p], color='deepskyblue', lw=0.7, alpha=0.9)  # 右侧

                    plt.colorbar(im, ax=ax, fraction=0.046)

            fig.suptitle(
                title_prefix + f'\nGT 逐点标注: 青=key时刻异常, 深蓝=query时刻异常 (#anom={len(anom_idx)}/{L})',
                fontsize=11
            )
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname), dpi=120)
            plt.close()

        # ---------------- 逐样本跑 ----------------
        abn_scores, nrm_scores = [], []

        for i, smp in enumerate(abnormal_pool):
            per_layer_kl = []
            for l in range(len(smp['gs'])):
                g = smp['gs'][l]
                r = (smp['trend'][l] + smp['spike'][l]) / 2.0
                per_layer_kl.append(_diff_score(g, r).mean())
            d = float(np.mean(per_layer_kl))
            abn_scores.append(d)
            plot_one(
                smp,
                f'[ABNORMAL #{i}] meanSymKL={d:.4f}',
                f'abnormal_{i:02d}_KL{d:.4f}.png'
            )

        for i, smp in enumerate(normal_pool):
            per_layer_kl = []
            for l in range(len(smp['gs'])):
                g = smp['gs'][l]
                r = (smp['trend'][l] + smp['spike'][l]) / 2.0
                per_layer_kl.append(_diff_score(g, r).mean())
            d = float(np.mean(per_layer_kl))
            nrm_scores.append(d)
            plot_one(
                smp,
                f'[NORMAL #{i}] meanSymKL={d:.4f}',
                f'normal_{i:02d}_KL{d:.4f}.png'
            )

        # ---------------- 总览 ----------------
        plt.figure(figsize=(10, 4))
        xa = np.arange(len(abn_scores))
        xn = np.arange(len(nrm_scores))
        plt.bar(xa - 0.2, abn_scores, width=0.4,
                label=f'Abnormal (mean={np.mean(abn_scores):.4f})',
                color='crimson')
        plt.bar(xn + 0.2, nrm_scores, width=0.4,
                label=f'Normal (mean={np.mean(nrm_scores):.4f})',
                color='steelblue')
        plt.ylabel('mean symmetric KL  GS vs (Trend+Spike)/2')
        plt.title('Attention discrepancy: Abnormal vs Normal')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'summary_bar.png'), dpi=120)
        plt.close()

        print(f'[Vis] Abnormal SymKL mean = {np.mean(abn_scores):.4f} ± {np.std(abn_scores):.4f}')
        print(f'[Vis] Normal   SymKL mean = {np.mean(nrm_scores):.4f} ± {np.std(nrm_scores):.4f}')
        print(f'[Vis] 图片已保存到 {save_dir}/')
    def test(self):
        checkpoint_path = os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        # (1) Find the threshold
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                output = self.model(input)
                adv_loss, con_loss, rec_loss,prior_loss = compute_discrepancy(input, output, temperature=temperature,
                                                                   keep_batch=True)

                # 🎯 测试阶段只用散度撕裂能量（避免 rec_loss 淹没正常/异常对比度）
                metric = torch.softmax((con_loss + rec_loss), dim=-1)

                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        thresh = np.percentile(test_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (2) Evaluation on the test set
        test_labels = []
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.test_loader):
                input = input_data.float().to(self.device)
                output = self.model(input)

                adv_loss, con_loss, rec_loss, prior_loss = compute_discrepancy(input, output, temperature=temperature,
                                                                   keep_batch=True)
                metric = torch.softmax((con_loss + rec_loss), dim=-1)

                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        # Point Adjustment 调整
        pred = adjust_predictions(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        roc_auc = roc_auc_score(gt, test_energy)

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
            accuracy, precision, recall, f_score))
        print(f"ROC AUC   : {roc_auc:.4f}")

        self.visualize_attn_diff(num_abnormal=5, num_normal=5)

        return accuracy, precision, recall, f_score