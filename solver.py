import torch
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from model.model import ProtoCausalAD
from data_factory.data_loader import get_loader_segment
import torch.nn.functional as F


def compute_proto_loss(input_data, output, temperature=50.0, keep_batch=False):
    """
    计算基于原型因果的多任务联合损失与测试异常能量
    """
    rec_x = output['rec_x']
    proto_loss = output['proto_loss']
    margin_loss = output['margin_loss']
    causal_latent_diff = output['causal_latent_diff']  # [B, L]

    if keep_batch:
        # ===== 测试/评估阶段：保留 [B, L] 维度 =====
        rec_err = torch.mean((input_data - rec_x) ** 2, dim=-1)  # [B, L]

        # 🎯 保留之前的 Softmax 温度放大小 Trick:
        # 在窗口时间维度上对能量进行放大与聚焦，凸显瞬态尖峰与偏离对比度
        raw_energy = causal_latent_diff + rec_err
        metric = torch.softmax(raw_energy * (temperature / 2.0), dim=-1)
        return metric
    else:
        # ===== 训练阶段：联合标量损失 =====
        rec_loss = torch.mean((input_data - rec_x) ** 2)
        total_loss = rec_loss + 0.1 * proto_loss + 1 * margin_loss
        return total_loss, rec_loss, proto_loss, margin_loss

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

        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = torch.device('xpu')
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu}")
        else:
            self.device = torch.device('cpu')

        print(f"[Solver] Using execution device: {self.device}")
        self.build_model()

    def build_model(self):
        self.model = ProtoCausalAD(
            win_size=self.win_size,
            seq_size=self.seq_size,
            c_in=self.input_c,
            c_out=self.output_c,
            d_model=self.d_model,
            e_layers=self.e_layers,
            num_prototypes=getattr(self, 'num_prototypes', 5),
            margin=getattr(self, 'margin', 5.0),
            epsilon=getattr(self, 'epsilon', 0.05),
            dev=self.device
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Model] Total params: {total_params:,} ({total_params / 1e6:.2f}M)")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def vali(self, vali_loader):
        self.model.eval()
        loss_list = []
        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input_tensor = input_data.float().to(self.device)
                output = self.model(input_tensor, training=False)
                total_loss, _, _, _ = compute_proto_loss(input_tensor, output)
                loss_list.append(total_loss.item())
        return np.average(loss_list)

    def train(self):
        print("======================TRAIN PROTO-CAUSAL AD======================")
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(self.train_loader)

        for epoch in tqdm(range(self.num_epochs), disable=not self.print_tqdm):
            start_time = time.time()
            loss_list = []

            self.model.train()
            with tqdm(total=train_steps, disable=not self.print_tqdm) as pbar:
                for i, (input_data, _) in enumerate(self.train_loader):
                    self.optimizer.zero_grad()
                    input_tensor = input_data.float().to(self.device)

                    # 开启 training=True 触发隐空间对抗梯度生成
                    output = self.model(input_tensor, training=True)
                    total_loss, rec_l, proto_l, margin_l = compute_proto_loss(input_tensor, output)

                    loss_list.append(total_loss.item())
                    pbar.update(1)
                    total_loss.backward()
                    self.optimizer.step()

            train_loss = np.average(loss_list)
            vali_loss = self.vali(self.vali_loader)

            torch.save(self.model.state_dict(), os.path.join(path, f"{self.dataset}_checkpoint.pth"))
            elapsed_time = time.time() - start_time
            print(f"Epoch: {epoch + 1} | Train: {train_loss:.6f} | Vali: {vali_loss:.6f} | Time: {elapsed_time:.2f}s")

    def test(self):
        checkpoint_path = os.path.join(str(self.model_save_path), f"{self.dataset}_checkpoint.pth")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)
        self.model.eval()
        temperature = 50.0  # 沿用之前的超参数温度设置

        print("======================TEST & ROOT CAUSE INFERENCE======================")

        # 1. 动态阈值阶段：使用 Softmax 后的相对能量确定切分阈值
        energy_list = []
        with torch.no_grad():
            for i, (input_data, _) in enumerate(self.thre_loader):
                input_tensor = input_data.float().to(self.device)
                output = self.model(input_tensor, training=False)
                metric = compute_proto_loss(input_tensor, output, temperature=temperature, keep_batch=True)
                energy_list.append(metric.detach().cpu().numpy())

        all_energy = np.concatenate(energy_list, axis=0).reshape(-1)
        thresh = np.percentile(all_energy, 100 - self.anormly_ratio)
        print("Anomaly Decision Threshold:", thresh)

        # 2. 真实测试集推断
        test_labels = []
        test_energy_list = []
        root_causes = []

        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.test_loader):
                input_tensor = input_data.float().to(self.device)
                output = self.model(input_tensor, training=False)

                metric = compute_proto_loss(input_tensor, output, temperature=temperature, keep_batch=True)
                test_energy_list.append(metric.detach().cpu().numpy())
                test_labels.append(labels)

                # 通道因果偏离 (定位 Top-3 受损传感器)
                channel_err = output['causal_channel_diff'].detach().cpu().numpy()  # [B, L, C]
                top_channels = np.argsort(-np.mean(channel_err, axis=1), axis=-1)[:, :3]
                root_causes.append(top_channels)

        test_energy = np.concatenate(test_energy_list, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        # Point Adjustment
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