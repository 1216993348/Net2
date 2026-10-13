import os
import sys
import argparse
import configparser
import random
import numpy as np
import torch

from solver import Solver


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a+', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file path")
    args = parser.parse_args()

    # 读取配置
    config_parser = configparser.ConfigParser()
    config_parser.read(args.config)

    config = argparse.Namespace()

    # train
    config.lr = float(config_parser['train']['lr'])
    config.gpu = int(config_parser['train']['gpu'])
    config.num_epochs = int(config_parser['train']['epoch'])
    config.anormly_ratio = float(config_parser['train']['ar'])
    config.batch_size = int(config_parser['train']['bs'])
    config.seed = int(config_parser['train']['seed'])

    # data
    config.win_size = int(config_parser['data']['ws'])
    config.input_c = int(config_parser['data']['ic'])
    config.output_c = int(config_parser['data']['oc'])
    config.dataset = config_parser['data']['ds']
    config.data_path = config_parser['data']['dp']

    # param
    config.d_model = int(config_parser['param']['d'])
    config.e_layers = int(config_parser['param']['l'])

    # model
    config.mode = config_parser['model']['mode']
    config.model_save_path = config_parser['model']['msp']

    # 种子
    setup_seed(config.seed)

    # 日志
    os.makedirs('result', exist_ok=True)
    sys.stdout = Logger(f"result/{config.dataset}_spagd.log")

    print('=' * 60)
    print('SPAGD Configuration:')
    for k, v in vars(config).items():
        print(f'  {k}: {v}')
    print('=' * 60)

    # 训练/测试
    solver = Solver(config)

    if config.mode == 'train':
        solver.train()
        solver.test()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'te':
        solver.train()
        solver.test()
    else:
        solver.train()
        solver.test()


if __name__ == '__main__':
    main()