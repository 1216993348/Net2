import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

# 复用外面的 data_loader
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_factory.data_loader import get_loader_segment

from model import SPAGD


def adjust_predictions(gt, pred):
    """Point Adjustment: 异常段内任意点被识别则全段标记为异常"""
    pred = pred.copy()
    anomaly_state = False
    n = len(gt)
    for i in range(n):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, n):
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


class Solver:
    def __init__(self, config):
        self.config = config

        # 设备
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{config.gpu}")
        else:
            self.device = torch.device('cpu')
        print(f"[Solver] Device: {self.device}")

        # 数据加载
        self.train_loader = get_loader_segment(
            config.data_path, config.batch_size, config.win_size,
            mode='train', dataset=config.dataset
        )
        self.test_loader = get_loader_segment(
            config.data_path, config.batch_size, config.win_size,
            mode='test', dataset=config.dataset
        )

        # 模型
        self.model = SPAGD(
            d=config.input_c,
            T=config.win_size,
            hidden_dim=config.d_model,
            gat_layers=config.e_layers,
            num_chunks=5,
            top_k=10,
            percent_m=10,
            beta=0.01
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        # 保存路径
        os.makedirs(config.model_save_path, exist_ok=True)
        self.checkpoint_path = os.path.join(config.model_save_path, f"{config.dataset}_checkpoint.pth")

    def train(self):
        print("====================== TRAIN ======================")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            loss_list = []

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            for input_data, _ in pbar:
                # input_data: [B, T, d] -> [B, d, T]
                X = input_data.float().to(self.device)
                X = X.permute(0, 2, 1)

                self.optimizer.zero_grad()
                outputs = self.model(X, mode='train')
                loss = outputs['loss']
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = np.mean(loss_list)
            print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.6f}")

            # 每轮保存
            torch.save(self.model.state_dict(), self.checkpoint_path)

    def test(self):
        print("====================== TEST ======================")

        # 加载模型
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            print(f"Loaded checkpoint: {self.checkpoint_path}")
        else:
            print("No checkpoint found, using current model weights.")

        self.model.eval()

        # ===== 1. 收集所有测试样本的分数和标签 =====
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for input_data, labels in self.test_loader:
                # input_data: [B, T, d]
                X = input_data.float().to(self.device).permute(0, 2, 1)
                scores = self.model(X, mode='test')  # [B]

                all_scores.append(scores.cpu().numpy())
                # 标签取每个窗口的最后一个时间点作为该窗口的标签
                # 或者取窗口内任意异常 -> 取max
                labels_np = labels.numpy()
                # 如果标签是 [B, win_size]，取max（窗口内任一异常即为异常）
                if len(labels_np.shape) > 1:
                    labels_np = labels_np.max(axis=1)
                all_labels.append(labels_np)

        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0).reshape(-1)

        # 确保长度一致
        min_len = min(len(all_scores), len(all_labels))
        all_scores = all_scores[:min_len]
        all_labels = all_labels[:min_len]

        print(f"Total test samples: {len(all_scores)}")

        # ===== 2. 确定阈值（用测试集自身） =====
        threshold = np.percentile(all_scores, 100 - self.config.anormly_ratio * 100)
        print(f"Threshold (top {100 - self.config.anormly_ratio * 100:.1f}%): {threshold:.6f}")

        # ===== 3. 预测 =====
        pred = (all_scores > threshold).astype(int)
        gt = all_labels.astype(int)

        # Point Adjustment
        pred_adj = adjust_predictions(gt, pred)

        # 指标
        roc_auc = roc_auc_score(gt, all_scores)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred_adj, average='binary')
        accuracy = accuracy_score(gt, pred_adj)

        print(f"\n{'=' * 40}")
        print(f"Results for {self.config.dataset}")
        print(f"{'=' * 40}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f_score:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print(f"{'=' * 40}")

        return accuracy, precision, recall, f_score, roc_auc