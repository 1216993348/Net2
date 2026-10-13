import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from huggingface_hub import hf_hub_download
import urllib.request
import json


def ensure_dataset_file(data_path, dataset_name, file_type):
    """
    确保数据集文件存在，如果不存在则从 Hugging Face 下载

    Args:
        data_path: 本地数据路径
        dataset_name: 数据集名称 (SMD, MSL, SMAP, SWaT, PSM)
        file_type: 文件类型 (train, test, test_label)

    Returns:
        file_path: 本地文件路径
    """
    # 文件映射
    file_map = {
        'SMD': {
            'train': 'SMD_train.npy',
            'test': 'SMD_test.npy',
            'test_label': 'SMD_test_label.npy'
        },
        'MSL': {
            'train': 'MSL_train.npy',
            'test': 'MSL_test.npy',
            'test_label': 'MSL_test_label.npy'
        },
        'SMAP': {
            'train': 'SMAP_train.npy',
            'test': 'SMAP_test.npy',
            'test_label': 'SMAP_test_label.npy'
        },
        'SWaT': {
            'train': 'SWaT_train.npy',
            'test': 'SWaT_test.npy',
            'test_label': 'SWaT_test_label.npy'
        },
        'PSM': {
            'train': 'train.csv',
            'test': 'test.csv',
            'test_label': 'test_label.csv'
        }
    }

    if dataset_name not in file_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if file_type not in file_map[dataset_name]:
        raise ValueError(f"Unknown file type: {file_type}")

    # PSM 是 CSV，其他是 npy
    relative_path = file_map[dataset_name][file_type]
    local_path = os.path.join(data_path, relative_path)

    # 创建目录
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # 如果文件不存在，从 Hugging Face 下载
    if not os.path.exists(local_path):
        print(f"Downloading {dataset_name}/{file_type} from Hugging Face...")
        try:
            # 使用 huggingface_hub 下载
            downloaded_path = hf_hub_download(
                repo_id="x1216/tsad-benchmark",
                filename=dataset_name + '/' + relative_path,
                local_dir=os.path.dirname(data_path),
                repo_type="dataset",
                local_dir_use_symlinks=False
            )
            print(f"Downloaded to: {downloaded_path}")
            return downloaded_path
        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise

    return local_path


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # 确保文件存在
        train_path = ensure_dataset_file(data_path, 'PSM', 'train')
        test_path = ensure_dataset_file(data_path, 'PSM', 'test')
        test_label_path = ensure_dataset_file(data_path, 'PSM', 'test_label')

        data = pd.read_csv(train_path)
        data = data.values[:, 1:]
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = pd.read_csv(test_path)
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = pd.read_csv(test_label_path).values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = ensure_dataset_file(data_path, 'MSL', 'train')
        test_path = ensure_dataset_file(data_path, 'MSL', 'test')
        test_label_path = ensure_dataset_file(data_path, 'MSL', 'test_label')

        data = np.load(train_path)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(test_path)
        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(test_label_path)
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = ensure_dataset_file(data_path, 'SMAP', 'train')
        test_path = ensure_dataset_file(data_path, 'SMAP', 'test')
        test_label_path = ensure_dataset_file(data_path, 'SMAP', 'test_label')

        data = np.load(train_path)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(test_path)
        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(test_label_path)
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = ensure_dataset_file(data_path, 'SMD', 'train')
        test_path = ensure_dataset_file(data_path, 'SMD', 'test')
        test_label_path = ensure_dataset_file(data_path, 'SMD', 'test_label')

        data = np.load(train_path)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(test_path)
        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(test_label_path)
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = ensure_dataset_file(data_path, 'SWaT', 'train')
        test_path = ensure_dataset_file(data_path, 'SWaT', 'test')
        test_label_path = ensure_dataset_file(data_path, 'SWaT', 'test_label')

        data = np.load(train_path, allow_pickle=True)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(test_path, allow_pickle=True)
        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(test_label_path, allow_pickle=True)
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=1)

    return data_loader