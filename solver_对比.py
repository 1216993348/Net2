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

# 引入您编写的课程异常注入器
from inject.curriculum_injector import CurriculumInjector
from inject.masked_universal_injector import MaskedUniversalInjector

def my_kl_loss(p, q):
    """KL 散度计算，带数值稳定微量 eps"""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)


def my_cross_entropy(p, q):
    """交叉熵损失：-Σ p * log(q)"""
    return -torch.sum(p * (torch.log(q + 0.0001)), dim=-1)


def compute_discrepancy(input, output, anomaly_mask=None, margin=5.0, keep_batch=False):
    """
    统一的跨视图关系差异计算 (带对比学习机制)
    """
    trend_att, gs_att, spike_att, rec_x, prior_loss, mask, _ = output

    temperature = 50.0
    con_loss_total = 0.0
    layers_num = len(trend_att)

    for u in range(layers_num):
        trend_norm = trend_att[u]
        gs_norm = gs_att[u]
        spike_norm = spike_att[u]

        # 融合特征：趋势 + 突变
        res_norm = (trend_norm + spike_norm) / 2.0

        if keep_batch:
            # ===== 测试阶段：保留 [B, L] 维度返回异常分数 =====
            con_forward = my_kl_loss(gs_norm, res_norm.detach())
            con_backward = my_kl_loss(res_norm.detach(), gs_norm)
            con_macro = con_forward + con_backward
            con_loss_total += con_macro * (temperature / 2.0)
        else:
            # ===== 训练阶段：基于是否注入了异常进行对比学习 =====
            con_forward = my_kl_loss(gs_norm, res_norm.detach())
            con_backward = my_kl_loss(res_norm.detach(), gs_norm)
            con_macro = con_forward + con_backward  # [B, L]

            if anomaly_mask is not None:
                # ------ 对比学习模式 ------
                loss_n = torch.tensor(0.0).to(input.device)
                loss_a = torch.tensor(0.0).to(input.device)

                # 正常样本 (mask==0): 拉近视图 (Minimize)
                if (anomaly_mask == 0).sum() > 0:
                    loss_n = con_macro[anomaly_mask == 0].mean()

                # 异常样本 (mask==1): 推远视图 (Maximize, 使用 Margin 防止梯度爆炸)
                if (anomaly_mask == 1).sum() > 0:
                    loss_a = torch.relu(margin - con_macro)[anomaly_mask == 1].mean()

                # 将正常和异常对比损失相加（异常权重设为1.5强化惩罚）
                con_loss_total += (loss_n + 1.5 * loss_a) * (temperature / 2.0)
            else:
                # ------ 无监督默认模式 (Vali) ------
                con_loss_total += torch.mean(con_macro) * (temperature / 2.0)

    con_loss = con_loss_total / layers_num

    # ================= 重建损失计算 =================
    if mask is not None:
        # 处理模型内部的通道随机掩码
        input = input[mask]
        rec_x = rec_x[mask]

    if keep_batch:
        rec_loss = torch.mean((input - rec_x) ** 2, dim=-1)
    else:
        # 如果存在异常注入掩码，则仅对正常序列点计算重建损失
        if anomaly_mask is not None and mask is None:
            rec_loss_all = torch.mean((input - rec_x) ** 2, dim=-1)
            if (anomaly_mask == 0).sum() > 0:
                rec_loss = rec_loss_all[anomaly_mask == 0].mean()
            else:
                rec_loss = torch.tensor(0.0).to(input.device)
        else:
            rec_loss = torch.mean((input - rec_x) ** 2)

    return 0, con_loss, rec_loss, prior_loss


def compute_discrepancy2(input, output, temperature=1.0, keep_batch=False):
    # (保留原有备用方法不变)
    trend_att, gs_att, spike_att, rec_x, prior_loss, mask, _ = output
    con_loss_total = 0.0
    layers_num = len(trend_att)
    for u in range(layers_num):
        trend_norm = trend_att[u]
        gs_att_u = gs_att[u]
        spike_norm = spike_att[u]
        res_norm = (trend_norm + spike_norm) / 2.0
        if keep_batch:
            con_macro = my_kl_loss(gs_att_u, res_norm.detach()) + my_kl_loss(res_norm.detach(), gs_att_u)
            con_loss_total += (con_macro) * (temperature / 2.0)
        else:
            con_macro = torch.mean(my_kl_loss(gs_att_u, res_norm.detach()) + my_kl_loss(res_norm.detach(), gs_att_u))
            con_loss_total += (con_macro) * (temperature / 2.0)
    con_loss = con_loss_total / layers_num
    if mask != None:
        input = input[mask]
        rec_x = rec_x[mask]
    if keep_batch:
        rec_loss = torch.mean((input - rec_x) ** 2, dim=-1)
    else:
        rec_loss = torch.mean((input - rec_x) ** 2)
    return 0, con_loss, rec_loss, prior_loss


def adjust_predictions(gt, pred):
    """Point Adjustment: 异常段内任意点被识别则全段标记为异常"""
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

        # 🎯 引入课程注入器参数
        self.injector = MaskedUniversalInjector()
        self.anomaly_ratio = 0.3  # batch 中增强样本比例
        self.margin = 5.0  # 对比学习界限

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

                # Vali 阶段不注入异常，直接运行无监督 Loss
                adv_loss, con_loss, rec_loss, prior_loss = compute_discrepancy(input, output)

                if getattr(self, 'use_recon', False):
                    loss_list.append((rec_loss + con_loss + prior_loss).item())
                else:
                    loss_list.append((con_loss + prior_loss).item())

        return np.average(loss_list)

    def train(self):
        print("======================TRAIN MODE (Contrastive)======================")

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

                    # 🎯 ============ 动态注入异常 (Curriculum Learning) ============
                    input_np = input_data.numpy()  # [B, L, C]
                    B, L, C = input_np.shape

                    injected_batch = np.zeros_like(input_np)
                    mask_batch = np.zeros((B, L))

                    for b in range(B):
                        if np.random.rand() < self.anomaly_ratio:
                            injected, mask = self.injector.inject(input_np[b])
                            injected_batch[b] = injected
                            mask_batch[b] = mask
                        else:
                            injected_batch[b] = input_np[b]
                            # mask_batch 默认全 0

                    input_tensor = torch.tensor(injected_batch).float().to(self.device)
                    mask_tensor = torch.tensor(mask_batch).float().to(self.device)
                    # =============================================================

                    output = self.model(input_tensor, True)

                    # 计算包含 Contrastive Loss 的散度
                    adv_loss, con_loss, rec_loss, prior_loss = compute_discrepancy(
                        input_tensor, output, anomaly_mask=mask_tensor, margin=self.margin
                    )

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

    def test(self):
        checkpoint_path = os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                output = self.model(input)

                # Test 阶段保持维度，不提供异常掩码
                adv_loss, con_loss, rec_loss, prior_loss = compute_discrepancy(input, output, keep_batch=True)

                # 🎯 测试阶段只用散度撕裂能量
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

                adv_loss, con_loss, rec_loss, prior_loss = compute_discrepancy(input, output, keep_batch=True)
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
        pred = np.array(pred)
        gt = np.array(gt)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        roc_auc = roc_auc_score(gt, test_energy)

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
            accuracy, precision, recall, f_score))
        print(f"ROC AUC   : {roc_auc:.4f}")

        return accuracy, precision, recall, f_score