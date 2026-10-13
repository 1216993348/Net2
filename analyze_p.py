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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    args = parser.parse_args()

    fileconfig = configparser.ConfigParser()
    fileconfig.read(args.config)

    # 读取基础配置
    parser.add_argument('--lr', type=float, default=fileconfig['train']['lr'])
    parser.add_argument('--gpu', type=str, default=fileconfig['train']['gpu'])
    parser.add_argument('--num_epochs', type=int, default=fileconfig['train']['epoch'])
    parser.add_argument('--anormly_ratio', type=float, default=fileconfig['train']['ar'])
    parser.add_argument('--batch_size', type=int, default=fileconfig['train']['bs'])
    parser.add_argument('--seed', type=int, default=fileconfig['train']['seed'])

    parser.add_argument('--win_size', type=int, default=fileconfig['data']['ws'])
    parser.add_argument('--input_c', type=int, default=fileconfig['data']['ic'])
    parser.add_argument('--output_c', type=int, default=fileconfig['data']['oc'])
    parser.add_argument('--dataset', type=str, default=fileconfig['data']['ds'])
    parser.add_argument('--data_path', type=str, default=fileconfig['data']['dp'])

    parser.add_argument('--d_model', type=int, default=fileconfig['param']['d'])
    parser.add_argument('--e_layers', type=int, default=fileconfig['param']['l'])
    parser.add_argument('--fr', type=float, default=fileconfig['param']['fr'])
    parser.add_argument('--tr', type=float, default=fileconfig['param']['tr'])
    parser.add_argument('--seq_size', type=int, default=fileconfig['param']['ss'])

    parser.add_argument('--mode', type=str, default=fileconfig['model']['mode'])
    parser.add_argument('--model_save_path', type=str, default=fileconfig['model']['msp'])

    config = parser.parse_args()
    args = vars(config)

    # 设置日志
    if not os.path.exists("result"):
        os.makedirs("result")

    config.use_recon = True
    # 分析结果
    from analyze import analyze_results, plot_results
    df = analyze_results(config.dataset, "grid_search_results_final_dchannel")
    if df is not None:
        plot_results(df, config.dataset)