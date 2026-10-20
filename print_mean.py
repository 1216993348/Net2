import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

base = r"E:\project_workplace\Net2\dataset"

# ============ npy 格式 ============
npy_datasets = {
    "MSL":  ("MSL/MSL_train.npy",  "MSL/MSL_test.npy"),
    "SMAP": ("SMAP/SMAP_train.npy", "SMAP/SMAP_test.npy"),
    "SMD":  ("SMD/SMD_train.npy",  "SMD/SMD_test.npy"),
    "SWaT": ("SWaT/SWaT_train.npy", "SWaT/SWaT_test.npy"),
}
def report(name, train_X, test_X):
    # 统一去 nan
    train_X = pd.DataFrame(train_X).ffill().bfill().fillna(0).values
    test_X  = pd.DataFrame(test_X).ffill().bfill().fillna(0).values

    # 统一标准化（fit on train）
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_n = scaler.transform(train_X)
    test_n  = scaler.transform(test_X)

    # 统计
    diff = test_n.mean(axis=0) - train_n.mean(axis=0)
    print(f"{name}: 全局差 {diff.mean():.4f}, 最大特征差 {np.abs(diff).max():.4f}, "
          f">0.5 特征数 {(np.abs(diff)>0.5).sum()}/{len(diff)}")

def stat_npy(name, train_path, test_path):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[{name}] 文件缺失，跳过\n")
        return

    train = np.load(train_path)
    test  = np.load(test_path)

    if train.ndim == 1: train = train[:, None]
    if test.ndim  == 1: test  = test[:, None]

    # 去 nan
    train = np.nan_to_num(train, nan=0.0)
    test  = np.nan_to_num(test,  nan=0.0)

    report(name, train, test)

# ============ PSM ============
def stat_psm(psm_dir):
    train_csv = os.path.join(psm_dir, "train.csv")
    test_csv  = os.path.join(psm_dir, "test.csv")

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("[PSM] 文件缺失，跳过\n")
        return

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    # 去掉 timestamp 列
    drop_cols = [c for c in train_df.columns if "timestamp" in c.lower()]
    train_df = train_df.drop(columns=drop_cols)
    test_df  = test_df.drop(columns=drop_cols)

    # 只保留数值列
    train_num = train_df.select_dtypes(include=[np.number])
    test_num  = test_df.select_dtypes(include=[np.number])

    # 去掉 label 列（如果有）
    for c in list(train_num.columns):
        if c.lower() in ["label", "is_anomaly", "anomaly"]:
            train_num = train_num.drop(columns=[c])
    for c in list(test_num.columns):
        if c.lower() in ["label", "is_anomaly", "anomaly"]:
            test_num = test_num.drop(columns=[c])

    # 对齐列
    common_cols = [c for c in train_num.columns if c in test_num.columns]
    train_num = train_num[common_cols]
    test_num  = test_num[common_cols]

    train_X = train_num.values.astype(np.float32)
    test_X  = test_num.values.astype(np.float32)

    # 去 nan（前向填充）
    train_df2 = pd.DataFrame(train_X).fillna(method='ffill').fillna(method='bfill').fillna(0)
    test_df2  = pd.DataFrame(test_X).fillna(method='ffill').fillna(method='bfill').fillna(0)
    train_X = train_df2.values
    test_X  = test_df2.values

    report("PSM", train_X, test_X)

# ============ 主流程 ============
if __name__ == "__main__":
    for name, (tr, te) in npy_datasets.items():
        stat_npy(name, os.path.join(base, tr), os.path.join(base, te))

    stat_psm(os.path.join(base, "PSM"))