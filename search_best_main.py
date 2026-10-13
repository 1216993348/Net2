import os
import random
import argparse
import configparser
import json
import numpy as np
from torch.backends import cudnn
import torch
from solver import Solver
import sys
from datetime import datetime


# ... 保留原有的 Logger 类和 setup_universal_seed 函数 ...
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


def save_results(results, dataset_name, save_dir="grid_search_results"):
        """保存网格搜索结果 - 包含统计信息"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for key, value in results.items():
            bs, d = key
            if not value:
                continue

            filename = f"{dataset_name}_bs{bs}_d{d}_results.json"
            filepath = os.path.join(save_dir, filename)

            # 提取所有有效的f_score
            valid_results = {k: v for k, v in value.items() if 'error' not in v}
            if valid_results:
                f_scores = [v['f_score'] for v in valid_results.values()]
                accuracies = [v['accuracy'] for v in valid_results.values()]
                precisions = [v['precision'] for v in valid_results.values()]
                recalls = [v['recall'] for v in valid_results.values()]

                best_seed = max(valid_results, key=lambda x: valid_results[x]['f_score'])
                best_result = valid_results[best_seed]
                worst_seed = min(valid_results, key=lambda x: valid_results[x]['f_score'])
                worst_result = valid_results[worst_seed]

                # 计算统计数据
                stats = {
                    'f_score': {
                        'mean': np.mean(f_scores),
                        'std': np.std(f_scores),
                        'min': np.min(f_scores),
                        'max': np.max(f_scores),
                        'median': np.median(f_scores)
                    },
                    'accuracy': {
                        'mean': np.mean(accuracies),
                        'std': np.std(accuracies),
                        'min': np.min(accuracies),
                        'max': np.max(accuracies)
                    },
                    'precision': {
                        'mean': np.mean(precisions),
                        'std': np.std(precisions),
                        'min': np.min(precisions),
                        'max': np.max(precisions)
                    },
                    'recall': {
                        'mean': np.mean(recalls),
                        'std': np.std(recalls),
                        'min': np.min(recalls),
                        'max': np.max(recalls)
                    }
                }
            else:
                best_seed = None
                best_result = None
                worst_seed = None
                worst_result = None
                stats = None

            result_data = {
                'batch_size': bs,
                'd_model': d,
                'total_seeds_tested': len(value),
                'best_seed': best_seed,
                'best_f_score': best_result['f_score'] if best_result else None,
                'best_accuracy': best_result['accuracy'] if best_result else None,
                'best_precision': best_result['precision'] if best_result else None,
                'best_recall': best_result['recall'] if best_result else None,
                'worst_seed': worst_seed,
                'worst_f_score': worst_result['f_score'] if worst_result else None,
                'statistics': stats,
                'all_seeds': value  # 所有种子的详细结果
            }

            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)

            print(
                f"Saved: {filename} (seeds: {len(value)}, mean F1: {stats['f_score']['mean']:.4f} ± {stats['f_score']['std']:.4f})")

def main(config, bs=None, d=None, seed=None):
    """修改主函数以支持参数覆盖"""
    # 如果传入了参数，覆盖config
    if bs is not None:
        config.batch_size = bs
    if d is not None:
        config.d_model = d
    if seed is not None:
        setup_universal_seed(seed)

    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        os.makedirs(config.model_save_path)

    # 保存当前的超参数到config中供solver使用
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
        # 训练后立即测试
        config.mode = 'test'
        solver = Solver(vars(config))
        return solver.test()
    elif config.mode == 'test':
        return solver.test()
    else:
        solver.train()
        solver.vali(solver.vali_loader)
        return solver.test()

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

    print('print_tqdm:',config.print_tqdm)
    args = vars(config)

    # 设置日志
    if not os.path.exists("result"):
        os.makedirs("result")
    sys.stdout = Logger("result/" + config.dataset + "_grid_search.log", sys.stdout)

    config.use_recon = True
    print('\n\n')
    print('============ Single Combination Run ============')
    print(f'Dataset: {config.dataset}')
    print(f'Batch Size: {config.batch_size}')
    print(f'D Model: {config.d_model}')
    print(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=============================================')

    # 只运行当前配置指定的一个组合
    bs = config.batch_size
    d = config.d_model
    seeds = list(range(100))  # 100个种子

    # 存储结果
    all_results = {}
    key = (bs, d)
    all_results[key] = {}

    # 创建最终结果目录
    final_dir = "grid_search_results_final"
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)

    # 检查是否已经完成该组合（断点续跑）
    combo_file = f"{final_dir}/{config.dataset}_bs{bs}_d{d}_results.json"
    if os.path.exists(combo_file):
        print(f"组合 BS={bs}, D={d} 已完成，跳过")
        with open(combo_file, 'r') as f:
            existing_data = json.load(f)
            all_results[key] = existing_data['all_seeds']
    else:
        for seed in seeds:
            print(f'\nSeed: {seed} ({seed+1}/100)')

            try:
                setup_universal_seed(seed)

                # 设置模型保存路径
                config.model_save_path = f"checkpoints/{config.dataset}/bs{bs}_d{d}_seed{seed}"
                if not os.path.exists(config.model_save_path):
                    os.makedirs(config.model_save_path)

                # 执行训练和测试
                accuracy, precision, recall, f_score = main(config, bs=bs, d=d, seed=seed)

                all_results[key][seed] = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f_score': float(f_score)
                }

                print(f'F1={f_score:.4f}')

            except Exception as e:
                print(f'Error for Seed={seed}: {str(e)}')
                all_results[key][seed] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f_score': 0.0,
                    'error': str(e)
                }

            # 每10个种子保存一次
            if (seed + 1) % 10 == 0:
                save_results(all_results, config.dataset, "grid_search_results_intermediate")

        # 组合完成后保存最终结果
        print(f'\n组合 BS={bs}, D={d} 完成！保存最终结果...')
        save_results(all_results, config.dataset, final_dir)

        # 显示最佳结果
        if all_results[key]:
            best_seed = max(all_results[key], key=lambda x: all_results[key][x]['f_score'])
            best_result = all_results[key][best_seed]
            print(f'Best for BS={bs}, D={d}: Seed={best_seed}, F1={best_result["f_score"]:.4f}')
    #
    # # 分析结果
    # from analyze import analyze_results, plot_results
    # df = analyze_results(config.dataset, "grid_search_results_final")
    # if df is not None:
    #     plot_results(df, config.dataset)