import os
import random
import argparse
import configparser

import numpy as np
from torch.backends import cudnn
import torch

from solver import Solver
import sys

# python main.py --config config/MSL.conf
# python main.py --config config/PSM.conf
# python main.py --config config/SMD.conf
# python main.py --config config/SMAP.conf
# python main.py --config config/SWaT.conf
class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


import random
import os
import numpy as np
import torch


def setup_universal_seed(seed=42):
    """
    全平台（CPU / CUDA GPU / Intel XPU）随机种子与确定性彻底锁定
    """
    # 1. Python & NumPy 基础种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # 2. PyTorch CPU 种子
    torch.manual_seed(seed)

    # 3. Nvidia CUDA 平台设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 4. Intel XPU 平台设置
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)
        # 强制 OneDNN / IPEX 底层启用确定性矩阵运算
        os.environ['DNNL_DETERMINISTIC'] = '1'
        os.environ['ONEDNN_DETERMINISTIC'] = '1'

    # 5. 强制全局确定性算法（兼容所有硬件）
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

# 🎯 代码入口处调用
setup_universal_seed(42)

def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        os.mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    else:
        solver.train()
        # solver.vali(solver.vali_loader)
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    parser.add_argument('--no_print_tqdm', action='store_false', dest='print_tqdm', default=True)

    args = parser.parse_args()
    fileconfig = configparser.ConfigParser()
    fileconfig.read(args.config)

    config = argparse.Namespace(**vars(args))
    config.lr = float(fileconfig['train']['lr'])
    config.gpu = fileconfig['train']['gpu']
    config.num_epochs = int(fileconfig['train']['epoch'])
    config.anormly_ratio = float(fileconfig['train']['ar'])
    config.batch_size = int(fileconfig['train']['bs'])
    config.seed = int(fileconfig['train']['seed'])

    # data
    config.win_size = int(fileconfig['data']['ws'])
    config.input_c = int(fileconfig['data']['ic'])
    config.output_c = int(fileconfig['data']['oc'])
    config.dataset = fileconfig['data']['ds']
    config.data_path = fileconfig['data']['dp']

    # param
    config.d_model = int(fileconfig['param']['d'])
    config.e_layers = int(fileconfig['param']['l'])
    config.seq_size = int(fileconfig['param']['ss'])

    # model
    config.mode = fileconfig['model']['mode']
    config.model_save_path = fileconfig['model']['msp']

    args = vars(config)

    # if config.seed is not None:
    #     setup_universal_seed(config.seed)

    sys.stdout = Logger("result/"+ config.dataset +".log", sys.stdout)
    config.use_recon = True
    print('\n\n')
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    # seed_list = [98, 93, 91]
    # # 后面的数字
    # extra_numbers = [38,46, 12]
    # extra_numbers2 = [58, 71, 35]
    # # 合并
    # all_numbers = seed_list + extra_numbers + extra_numbers2
    # for i in range(100):
    #     setup_universal_seed(i)  # 或者 setup_universal_seed(42 + i)
    #     print(f'seed:{i}')
    #     main(config)
    setup_universal_seed(config.seed)
    main(config)

