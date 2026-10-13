# generate_configs_by_combination.py
import configparser
import os
import sys
import glob


def generate_configs(original_config_file):
    """
    根据原始配置文件生成对应的调参版本

    Args:
        original_config_file: 原始配置文件路径，如 "config/MSL.conf"
    """
    # 读取原始配置
    base_config = configparser.ConfigParser()
    base_config.read(original_config_file)

    # 获取数据集名称（用于文件命名）
    dataset = base_config.get('data', 'ds')

    # 从原始配置中提取基础参数
    base_params = {
        'train': {
            'lr': base_config.get('train', 'lr'),
            'gpu': base_config.get('train', 'gpu'),
            'epoch': base_config.get('train', 'epoch'),
            'ar': base_config.get('train', 'ar'),
            'seed': base_config.get('train', 'seed')
        },
        'data': {
            'ws': base_config.get('data', 'ws'),
            'ic': base_config.get('data', 'ic'),
            'oc': base_config.get('data', 'oc'),
            'ds': dataset,
            'dp': base_config.get('data', 'dp')
        },
        'param': {
            'd': base_config.get('param', 'd'),
            'l': base_config.get('param', 'l'),
            'fr': base_config.get('param', 'fr'),
            'tr': base_config.get('param', 'tr'),
            'ss': base_config.get('param', 'ss')
        },
        'model': {
            'mode': base_config.get('model', 'mode'),
            'msp': base_config.get('model', 'msp')
        }
    }

    # 调参范围
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    d_models = [16, 32, 64, 128, 256]

    # 创建输出目录
    output_dir = f"config/{dataset}_grid"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combo_id = 0
    generated_files = []

    for bs in batch_sizes:
        for d in d_models:
            config = configparser.ConfigParser()

            # 复制基础配置
            config['train'] = base_params['train'].copy()
            config['data'] = base_params['data'].copy()
            config['param'] = base_params['param'].copy()
            config['model'] = base_params['model'].copy()

            # 修改调参参数
            config['train']['bs'] = str(bs)
            config['param']['d'] = str(d)
            config['model']['msp'] = f'checkpoints/{dataset}/bs{bs}_d{d}'

            # 添加注释（可选）
            config['__comment__'] = {
                'original_config': os.path.basename(original_config_file),
                'note': f'Grid search config for BS={bs}, D={d}',
                'generated_by': 'generate_configs_by_combination.py'
            }

            filename = f"{output_dir}/{dataset}_bs{bs}_d{d}.conf"
            with open(filename, 'w') as f:
                config.write(f)

            combo_id += 1
            generated_files.append(filename)
            print(f"[{combo_id:2d}/30] Generated: {filename}")

    return generated_files


def generate_all_datasets():
    """为所有数据集生成调参配置"""
    # 查找所有原始配置文件
    original_configs = glob.glob("config/*.conf")
    # 排除已经生成的调参配置（包含 _grid 目录）
    original_configs = [f for f in original_configs if '_grid' not in f]

    print("Found original config files:")
    for f in original_configs:
        print(f"  - {f}")

    print("\n" + "=" * 60)

    all_files = []
    for config_file in original_configs:
        print(f"\nGenerating for: {config_file}")
        print("-" * 40)
        files = generate_configs(config_file)
        all_files.extend(files)

    return all_files


if __name__ == '__main__':
    print("Generating grid search configurations...")
    print("=" * 60)

    # 方式1：为所有数据集生成
    # all_files = generate_all_datasets()

    # 方式2：为指定数据集生成（取消注释并修改）
    # generate_configs("config/MSL.conf")
    # generate_configs("config/PSM.conf")
    # generate_configs("config/SMAP.conf")
    # generate_configs("config/SMD.conf")
    # generate_configs("config/SWaT.conf")

    # 方式3：命令行指定
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if os.path.exists(config_file):
            generate_configs(config_file)
        else:
            print(f"Error: {config_file} not found!")
    else:
        # 默认生成所有
        print("No config specified, generating for all datasets...")
        generate_all_datasets()

    print("\n" + "=" * 60)
    print("Done!")
    print("\nUsage:")
    print("  python generate_conf.py                    # 生成所有数据集")
    print("  python generate_conf.py config/MSL.conf    # 只生成MSL")
    print("  python generate_conf.py config/PSM.conf    # 只生成PSM")