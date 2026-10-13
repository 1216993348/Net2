import torch
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from model.TFCL import TFC
from data_factory.data_loader import get_loader_segment
import torch.nn.functional as F


def my_kl_loss(p, q):
    """KL 散度计算，带数值稳定微量 eps"""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)

def my_kl_loss(p, q):
    """KL散度计算，带数值稳定微量 eps"""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)


def compute_discrepancy(input, output, temperature=1.0, keep_batch=False):
    """
    层级解耦（Hierarchical Decoupling）损失函数：
    1. 宏观对抗：Tem vs (Sea + Fre)
    2. 微观对抗：Sea vs Fre
    """
    tematt, seaatt, freatt, rec_x = output

    # 1. NaN 数值安全防御
    # for u in range(len(tematt)):
    #     if torch.isnan(tematt[u]).any(): tematt[u] = torch.nan_to_num(tematt[u], nan=0.0)
    #     if torch.isnan(seaatt[u]).any(): seaatt[u] = torch.nan_to_num(seaatt[u], nan=0.0)
    #     if torch.isnan(freatt[u]).any(): freatt[u] = torch.nan_to_num(freatt[u], nan=0.0)

    adv_loss_total = 0.0
    con_loss_total = 0.0
    layers_num = len(tematt)

    for u in range(layers_num):
        # 归一化 Attention 概率分布
        t_norm = tematt[u] / torch.unsqueeze(torch.sum(tematt[u], dim=-1) + 1e-6, dim=-1)
        s_norm = seaatt[u] / torch.unsqueeze(torch.sum(seaatt[u], dim=-1) + 1e-6, dim=-1)
        f_norm = freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1) + 1e-6, dim=-1)

        # 🎯 残差部分的复合注意力：将微观的周期与高频融合，作为“残差总表达”
        res_norm = (s_norm + f_norm) / 2.0

        if keep_batch:
            # ===== 测试/评估阶段：保留 [B, L] 维数 =====
            # 1. 宏观对抗：趋势 t_norm vs 残差总表达 res_norm
            adv_macro = my_kl_loss(t_norm, res_norm.detach()) + my_kl_loss(res_norm.detach(), t_norm)
            con_macro = my_kl_loss(res_norm, t_norm.detach()) + my_kl_loss(t_norm.detach(), res_norm)

            # 2. 微观对抗：周期 s_norm vs 高频突变 f_norm
            adv_micro = my_kl_loss(s_norm, f_norm.detach()) + my_kl_loss(f_norm.detach(), s_norm)
            con_micro = my_kl_loss(f_norm, s_norm.detach()) + my_kl_loss(s_norm.detach(), f_norm)

            # 组合两级对抗能量
            adv_loss_total += (adv_macro + adv_micro) * (temperature / 2.0)
            con_loss_total += (con_macro + con_micro) * (temperature / 2.0)

        else:
            # ===== 训练阶段：归一化为标量 =====
            # 1. 宏观对抗
            adv_macro = torch.mean(my_kl_loss(t_norm, res_norm.detach()) + my_kl_loss(res_norm.detach(), t_norm))
            con_macro = torch.mean(my_kl_loss(res_norm, t_norm.detach()) + my_kl_loss(t_norm.detach(), res_norm))

            # 2. 微观对抗
            adv_micro = torch.mean(my_kl_loss(s_norm, f_norm.detach()) + my_kl_loss(f_norm.detach(), s_norm))
            con_micro = torch.mean(my_kl_loss(f_norm, s_norm.detach()) + my_kl_loss(s_norm.detach(), f_norm))

            # 组合两级 Loss
            adv_loss_total += (adv_macro + adv_micro) * (temperature / 2.0)
            con_loss_total += (con_macro + con_micro) * (temperature / 2.0)

    adv_loss = adv_loss_total / layers_num
    con_loss = con_loss_total / layers_num

    # ===== 重建损失计算 =====
    if keep_batch:
        rec_loss = torch.mean((input - rec_x) ** 2, dim=-1)
    else:
        rec_loss = torch.mean((input - rec_x) ** 2)

    return adv_loss, con_loss, rec_loss


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
        self.model = TFC(
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

                adv_loss, con_loss, rec_loss = compute_discrepancy(input, output, temperature=1.0)
                if getattr(self, 'use_recon', False):
                    # 训练/验证使用 0.01 的弱重构系数
                    loss_list.append((0.01 * rec_loss + con_loss - adv_loss).item())
                else:
                    loss_list.append((con_loss - adv_loss).item())

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
                    output = self.model(input)

                    adv_loss, con_loss, rec_loss = compute_discrepancy(input, output, temperature=1.0)
                    if getattr(self, 'use_recon', False):
                        # 0.01 弱重构系数防梯度污染
                        loss = 0.01 * rec_loss + con_loss - adv_loss
                    else:
                        loss = con_loss - adv_loss

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
                adv_loss, con_loss, rec_loss = compute_discrepancy(input, output, temperature=temperature,
                                                                   keep_batch=True)

                # 🎯 测试阶段只用散度撕裂能量（避免 rec_loss 淹没正常/异常对比度）
                metric = torch.softmax((adv_loss + con_loss + rec_loss), dim=-1)

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

                adv_loss, con_loss, rec_loss = compute_discrepancy(input, output, temperature=temperature,
                                                                   keep_batch=True)
                metric = torch.softmax((adv_loss + con_loss + rec_loss), dim=-1)

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

        return accuracy, precision, recall, f_score