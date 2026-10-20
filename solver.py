import torch
import numpy as np
import os
import time
from tqdm import tqdm
from collections import deque
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from model.model import RobustProtoAD
from data_factory.data_loader import get_loader_segment


class Solver(object):
    DEFAULTS = {
        'num_prototypes': 5,
        'delta': 0.5,
        'nu': 0.1,
        'epsilon': 0.05,
        'momentum': 0.995,         # 原型无梯度平滑更新系数
        'drift_tolerance': 1.25,    # 允许自适应的最大半径倍率 (1 + λ)
        'smooth_len': 8,            # 动力学平滑时间校验队列长度
        'var_threshold': 0.15       # 队列内的最大方差阈值 (阻断突发故障)
    }

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

        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu}")
        else:
            self.device = torch.device('cpu')
        print(f"[Solver] Using execution device: {self.device}")
        self.build_model()

    def build_model(self):
        self.model = RobustProtoAD(
            win_size=self.win_size,
            c_in=self.input_c,
            c_out=self.output_c,
            d_model=self.d_model,
            e_layers=self.e_layers,
            num_prototypes=self.num_prototypes,
            delta=self.delta,
            nu=self.nu,
            epsilon=self.epsilon
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        print("====================== TRAIN ROBUST PROTO AD ======================")
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)

        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.model.train()
            train_loss = []

            for input_data, _ in self.train_loader:
                self.optimizer.zero_grad()
                input_tensor = input_data.float().to(self.device)

                output = self.model(input_tensor, training=True)
                rec_loss = torch.mean((input_tensor - output['rec_x']) ** 2)
                proto_loss = output['proto_loss']

                total_loss = rec_loss + 0.1 * proto_loss
                total_loss.backward()
                self.optimizer.step()
                train_loss.append(total_loss.item())

            # 验证集评估
            self.model.eval()
            val_loss = []
            with torch.no_grad():
                for input_data, _ in self.vali_loader:
                    input_tensor = input_data.float().to(self.device)
                    output = self.model(input_tensor, training=False)
                    rec_l = torch.mean((input_tensor - output['rec_x']) ** 2)
                    val_loss.append(rec_l.item())

            print(f"Epoch: {epoch + 1} | Train Loss: {np.mean(train_loss):.6f} | "
                  f"Vali Loss: {np.mean(val_loss):.6f} | Time: {time.time() - start_time:.2f}s")

            # 打印当前学出的原型安全半径状态
            r_str = ", ".join([f"{r.item():.3f}" for r in self.model.proto_bank.radius.data])
            print(f"       -> Learned Prototype Radii R_k: [{r_str}]")

        torch.save(self.model.state_dict(), os.path.join(path, f"{self.dataset}_checkpoint.pth"))

    def test(self):
        """
        流式因果推理与在线原型动量自适应测试 (TTA)
        """
        checkpoint_path = os.path.join(str(self.model_save_path), f"{self.dataset}_checkpoint.pth")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        # 绝对冻结骨干网络与解码器，完全阻断参数更新
        for p in self.model.parameters():
            p.requires_grad = False

        print("====================== DETERMINING BASELINE THRESHOLD ======================")
        # 1. 在干净参考流 (thre_loader) 上计算基础报警阈值与残差分位数
        thre_energies = []
        thre_rec_errs = []
        with torch.no_grad():
            for input_data, _ in self.thre_loader:
                input_tensor = input_data.float().to(self.device)
                out = self.model(input_tensor, training=False)
                rec_e = torch.mean((input_tensor - out['rec_x']) ** 2, dim=-1)  # [B, L]
                energy = out['latent_diff'] + rec_e
                thre_energies.append(energy.detach().cpu().numpy())
                thre_rec_errs.append(rec_e.detach().cpu().numpy())

        all_energy = np.concatenate(thre_energies, axis=0).reshape(-1)
        all_rec = np.concatenate(thre_rec_errs, axis=0).reshape(-1)

        # 决策阈值 (按预设异常率截断)
        thresh = np.percentile(all_energy, 100 - self.anormly_ratio)
        # 稳态残差上界 (超过此界限视为有残差偏移，可能触发漂移甄别)
        rec_base_upper = np.percentile(all_rec, 95)
        print(f"Anomaly Energy Decision Threshold: {thresh:.6f}")
        print(f"Normal Reconstruction Base Upper Bound: {rec_base_upper:.6f}")

        print("====================== STREAMING TEST-TIME ADAPTATION ======================")
        test_labels = []
        test_energies = []
        adapted_count = 0

        # 时间动力学滑动缓冲队列 (排除突发故障，仅接纳平滑漂移)
        temporal_buffer = deque(maxlen=self.smooth_len)

        for input_data, labels in tqdm(self.test_loader, desc="Streaming TTA Inference"):
            input_tensor = input_data.float().to(self.device)
            B, L, C = input_tensor.shape

            with torch.no_grad():
                out = self.model(input_tensor, training=False)
                z = out['z']                   # [B, L, D]
                min_dist = out['min_dist']     # [B, L]
                nearest_idx = out['nearest_idx']  # [B, L]
                rec_e = torch.mean((input_tensor - out['rec_x']) ** 2, dim=-1)  # [B, L]

                # 最终输出异常打分: 隐空间流形偏离 + 时域投影残差
                step_energy = out['latent_diff'] + rec_e
                test_energies.append(step_energy.cpu().numpy())
                test_labels.append(labels.numpy())

                # ========== 在线样本准入与原型演进机制 ==========
                # 对批次内的样本切片进行逐段因果校验
                for b in range(B):
                    b_rec = rec_e[b].mean().item()
                    b_dist = min_dist[b].mean().item()
                    b_k = nearest_idx[b, 0].item()  # 该切片匹配的主导原型
                    b_R = self.model.proto_bank.radius[b_k].item()
                    z_mean = z[b].mean(dim=0)  # [D]

                    # 准入门禁条件 1: 时域残差稍高于稳态基准，但未造成恶性破坏
                    cond_rec = b_rec > rec_base_upper

                    # 准入门禁条件 2: 严格位于学出的自适应几何包络内 [R_k, (1+λ)*R_k]
                    # 超出则判定为故障越界，坚决不可演进
                    cond_geom = (b_R < b_dist) and (b_dist <= self.drift_tolerance * b_R)

                    if cond_rec and cond_geom:
                        temporal_buffer.append(z_mean)
                        # 准入门禁条件 3: 时间动力学平滑度校验 (滑动窗口内方差稳定)
                        if len(temporal_buffer) == self.smooth_len:
                            buf_tensor = torch.stack(list(temporal_buffer), dim=0)  # [W, D]
                            local_var = torch.var(buf_tensor, dim=0).mean().item()

                            if local_var < self.var_threshold:
                                # 校验通过：确认为物理合法稳态漂移，执行受控 EMA 更新
                                p_curr = self.model.proto_bank.prototypes.data[b_k]
                                self.model.proto_bank.prototypes.data[b_k] = (
                                    self.momentum * p_curr + (1.0 - self.momentum) * z_mean
                                )
                                adapted_count += 1
                                temporal_buffer.clear()
                    else:
                        # 瞬态抖动或稳态正常，重置动力学缓冲
                        temporal_buffer.clear()

        print(f"[TTA Audit] Streaming Adaptation Completed. Total Momentum Updates: {adapted_count}")

        # 3. 评估指标计算
        test_energy = np.concatenate(test_energies, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        # Point Adjustment 标准对齐
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0: break
                    pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0: break
                    pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        acc = accuracy_score(gt, pred)
        p, r, f1, _ = precision_recall_fscore_support(gt, pred, average='binary')
        auc = roc_auc_score(gt, test_energy)

        print(f"Accuracy: {acc:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
        return acc, p, r, f1