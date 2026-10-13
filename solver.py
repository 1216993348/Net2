import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)
from model.model import CEPNet
from data_factory.data_loader import get_loader_segment


def adjust_predictions(gt, pred):
    adjusted_pred = pred.copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and adjusted_pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                adjusted_pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                adjusted_pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            adjusted_pred[i] = 1
    return adjusted_pred


class Solver(object):
    def __init__(self, config):
        self.__dict__.update(config)

        # 保持 10 轮先快速验证
        self.num_epochs = getattr(self, 'num_epochs', 10)

        # 1. 数据加载
        self.train_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='train',
            dataset=self.dataset,
        )
        self.test_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='test',
            dataset=self.dataset,
        )
        self.thre_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='thre',
            dataset=self.dataset,
        )

        # 2. 设备探测
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = torch.device('xpu')
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu}")
        else:
            self.device = torch.device('cpu')
        print(f"[Solver] Active Device: {self.device}")

        # 3. 初始化 Transformer 级模型
        d_model = getattr(self, 'd_model', 128)
        num_codes = getattr(self, 'num_codes', 256)
        self.model = CEPNet(
            c_in=self.input_c,
            win_size=self.win_size,
            d_model=d_model,
            num_codes=num_codes,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Model] Transformer CEPNet Loaded | Epochs: {self.num_epochs} | Params: {total_params:,}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=getattr(self, 'lr', 1e-3), weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=1e-5
        )

    def train(self):
        print("======================TRAIN CEP-Net (TRANSFORMER EMA-VQ)======================")
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        train_steps = len(self.train_loader)
        freeze_epoch = int(self.num_epochs * 0.7)

        for epoch in range(self.num_epochs):
            if epoch == freeze_epoch:
                self.model.freeze_codebook()

            start_time = time.time()
            rec_list = []
            commit_list = []
            self.model.train()

            with tqdm(
                total=train_steps,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                ncols=100,
            ) as pbar:
                for i, (input_data, _) in enumerate(self.train_loader):
                    x = input_data.float().contiguous().to(self.device)

                    self.optimizer.zero_grad()
                    x_star, _, _, loss_commit = self.model(x)

                    loss_rec = torch.mean((x - x_star) ** 2)
                    total_loss = loss_rec + 0.25 * loss_commit

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    rec_list.append(loss_rec.item())
                    commit_list.append(loss_commit.item())

                    pbar.set_postfix({
                        'rec': f"{loss_rec.item():.4f}",
                        'commit': f"{loss_commit.item():.4f}",
                    })
                    pbar.update(1)

            self.scheduler.step()
            elapsed_time = time.time() - start_time
            print(
                f"Epoch: {epoch + 1} | Rec Loss: {np.mean(rec_list):.5f} |"
                f" Commit Loss: {np.mean(commit_list):.5f} | Time: {elapsed_time:.2f}s"
            )

        save_file = os.path.join(self.model_save_path, f"{self.dataset}_checkpoint.pth")
        torch.save(self.model.state_dict(), save_file)
        print(f"[Model] Checkpoint saved successfully: {save_file}")

    def test(self):
        ckpt = os.path.join(self.model_save_path, f"{self.dataset}_checkpoint.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

        print("======================CALCULATE THRESHOLD======================")
        thre_scores = []
        with torch.no_grad():
            for input_data, _ in self.thre_loader:
                x = input_data.float().contiguous().to(self.device)
                _, time_score, _, _ = self.model(x)
                step_score = time_score[:, -1].detach().cpu().numpy()
                thre_scores.append(step_score)

        thre_scores = np.concatenate(thre_scores, axis=0)
        anomaly_threshold = np.percentile(thre_scores, 100 - self.anormly_ratio)
        print(f"[Threshold] Selected Cutoff: {anomaly_threshold:.4f}")

        print("======================TEST & EVALUATION======================")
        test_scores = []
        test_labels = []
        all_attributions = []

        with torch.no_grad():
            for input_data, labels in self.test_loader:
                x = input_data.float().contiguous().to(self.device)
                _, time_score, channel_attr, _ = self.model(x)

                step_score = time_score[:, -1].detach().cpu().numpy()
                test_scores.append(step_score)
                all_attributions.append(channel_attr.detach().cpu().numpy())

                lbl = labels[:, -1].numpy() if labels.dim() > 1 else labels.numpy()
                test_labels.append(lbl)

        test_scores = np.concatenate(test_scores, axis=0)
        gt = np.concatenate(test_labels, axis=0).astype(int)
        all_attributions = np.concatenate(all_attributions, axis=0)

        pred = (test_scores > anomaly_threshold).astype(int)

        acc = accuracy_score(gt, pred)
        p, r, f1, _ = precision_recall_fscore_support(
            gt, pred, average='binary', zero_division=0
        )
        roc = roc_auc_score(gt, test_scores)
        prc = average_precision_score(gt, test_scores)

        print("-" * 65)
        print(f"[Strict Point-wise] Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
        print(f"ROC-AUC: {roc:.4f} | AUPRC: {prc:.4f} | Accuracy: {acc:.4f}")

        pred_pa = adjust_predictions(gt, pred)
        p_pa, r_pa, f1_pa, _ = precision_recall_fscore_support(
            gt, pred_pa, average='binary', zero_division=0
        )
        print(f"[Point-Adjusted]    Precision: {p_pa:.4f} | Recall: {r_pa:.4f} | F1: {f1_pa:.4f}")
        print("-" * 65)

        abnormal_indices = np.where(pred == 1)[0]
        if len(abnormal_indices) > 0:
            print("[Root Cause Attribution Demonstration]")
            for sample_idx in abnormal_indices[:3]:
                attr_sample = all_attributions[sample_idx]
                attr_percent = (attr_sample / (np.sum(attr_sample) + 1e-8)) * 100
                top_causes = np.argsort(-attr_percent)[:3]
                print(f"  Step {sample_idx:05d} Fault Attribution:")
                for rank, ch in enumerate(top_causes):
                    print(
                        f"    Top-{rank + 1} Cause: Channel {ch:02d}"
                        f" (Contribution: {attr_percent[ch]:.2f}%)"
                    )
        print("=" * 65)

        return acc, p, r, f1, prc