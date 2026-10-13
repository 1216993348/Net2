import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def analyze_results(dataset_name, result_dir="grid_search_results_final"):
    """分析所有结果文件"""
    all_data = []

    if not os.path.exists(result_dir):
        print(f"Error: Directory {result_dir} not found!")
        return None

    files_found = 0
    for filename in os.listdir(result_dir):
        if filename.startswith(dataset_name) and filename.endswith('.json'):
            files_found += 1
            filepath = os.path.join(result_dir, filename)
            print(f"Reading: {filename}")

            with open(filepath, 'r') as f:
                data = json.load(f)

                f_scores = []
                accuracies = []
                precisions = []
                recalls = []

                for k, v in data['all_seeds'].items():
                    if 'error' not in v:
                        f_scores.append(v['f_score'])
                        accuracies.append(v['accuracy'])
                        precisions.append(v['precision'])
                        recalls.append(v['recall'])

                if f_scores:
                    all_data.append({
                        'batch_size': data['batch_size'],
                        'd_model': data['d_model'],
                        # 集中趋势
                        'mean_f1': np.mean(f_scores),
                        'median_f1': np.median(f_scores),
                        # 离散程度
                        'std_f1': np.std(f_scores),
                        'variance_f1': np.var(f_scores),
                        'min_f1': np.min(f_scores),
                        'max_f1': np.max(f_scores),
                        'range_f1': np.max(f_scores) - np.min(f_scores),
                        # 百分位数
                        'q25_f1': np.percentile(f_scores, 25),
                        'q75_f1': np.percentile(f_scores, 75),
                        'iqr_f1': np.percentile(f_scores, 75) - np.percentile(f_scores, 25),
                        # 最佳值
                        'best_f1': data['best_f_score'],
                        'best_seed': data['best_seed'],
                        'best_accuracy': data['best_accuracy'],
                        'best_precision': data['best_precision'],
                        'best_recall': data['best_recall'],
                        # 其他指标均值
                        'mean_accuracy': np.mean(accuracies),
                        'mean_precision': np.mean(precisions),
                        'mean_recall': np.mean(recalls),
                        'seeds_count': len(f_scores),
                        'all_f1_scores': f_scores
                    })

    if files_found == 0:
        print(f"No result files found for dataset: {dataset_name}")
        return None

    print(f"\nTotal files analyzed: {files_found}")
    df = pd.DataFrame(all_data)

    # 打印统计摘要
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    print(f"Total Configurations: {len(df)}")
    print(f"Seeds per Configuration: {df['seeds_count'].iloc[0]}")
    print(f"\nF1 Score Statistics across all configurations:")
    print(f"  Mean F1: {df['mean_f1'].mean():.4f} +/- {df['mean_f1'].std():.4f}")
    print(f"  Best F1: {df['best_f1'].max():.4f}")
    print(f"  Worst F1: {df['best_f1'].min():.4f}")

    # TOP 10 按均值排序
    df_sorted = df.sort_values('mean_f1', ascending=False)
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS BY MEAN F1:")
    print("=" * 80)
    print(df_sorted[['batch_size', 'd_model', 'mean_f1', 'std_f1', 'median_f1', 'best_f1', 'best_seed']].head(10).to_string(index=False))

    # 找出最稳定的配置（标准差最小）
    df_stable = df.sort_values('std_f1')
    print("\n" + "=" * 80)
    print("MOST STABLE CONFIGURATIONS (LOWEST STD):")
    print("=" * 80)
    print(df_stable[['batch_size', 'd_model', 'mean_f1', 'std_f1', 'variance_f1']].head(5).to_string(index=False))

    # 找出波动最大的配置（标准差最大）
    df_unstable = df.sort_values('std_f1', ascending=False)
    print("\n" + "=" * 80)
    print("MOST UNSTABLE CONFIGURATIONS (HIGHEST STD):")
    print("=" * 80)
    print(df_unstable[['batch_size', 'd_model', 'mean_f1', 'std_f1', 'variance_f1']].head(5).to_string(index=False))

    # 找出最佳配置
    best_overall = df.loc[df['best_f1'].idxmax()]
    print("\n" + "=" * 80)
    print("GLOBAL BEST CONFIGURATION:")
    print("=" * 80)
    print(f"Batch Size: {best_overall['batch_size']}")
    print(f"D Model: {best_overall['d_model']}")
    print(f"Best Seed: {best_overall['best_seed']}")
    print(f"Best F1: {best_overall['best_f1']:.4f}")
    print(f"Mean F1: {best_overall['mean_f1']:.4f} +/- {best_overall['std_f1']:.4f}")
    print(f"Median F1: {best_overall['median_f1']:.4f}")
    print(f"Variance: {best_overall['variance_f1']:.6f}")

    return df


def plot_results(df, dataset_name):
    """绘制结果图表 - 包含统计指标可视化"""
    if df is None or df.empty:
        print("No data to plot")
        return

    # 创建 2x4 的图表布局 (8个子图)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # 1. F1 Score 热力图 (Best F1)
    ax1 = axes[0]
    pivot_f1 = df.pivot(index='d_model', columns='batch_size', values='best_f1')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'Best F1'}, ax=ax1)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('D Model')
    ax1.set_title('Best F1 Score Heatmap (Higher=Better)')

    # 2. F1 Score 热力图 (Mean F1)
    ax2 = axes[1]
    pivot_mean = df.pivot(index='d_model', columns='batch_size', values='mean_f1')
    sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'Mean F1'}, ax=ax2)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('D Model')
    ax2.set_title('Mean F1 Score Heatmap (Higher=Better)')

    # 3. 标准差热力图 - 强制从0开始
    ax3 = axes[2]
    pivot_std = df.pivot(index='d_model', columns='batch_size', values='std_f1')
    std_max = pivot_std.max().max()
    # 创建colorbar，强制刻度从0开始
    cbar_kws_std = {
        'label': 'Std F1 (0=Perfect)',
        'ticks': np.linspace(0, std_max, 5)  # 从0到最大值均匀分布
    }
    sns.heatmap(pivot_std, annot=True, fmt='.4f', cmap='RdYlGn_r',
                vmin=0, vmax=std_max,
                cbar_kws=cbar_kws_std, ax=ax3)
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('D Model')
    ax3.set_title('Standard Deviation Heatmap (Green=More Stable)')

    # 4. 方差热力图 - 强制从0开始
    ax4 = axes[3]
    pivot_var = df.pivot(index='d_model', columns='batch_size', values='variance_f1')
    var_max = pivot_var.max().max()
    cbar_kws_var = {
        'label': 'Variance (0=Perfect)',
        'ticks': np.linspace(0, var_max, 5)  # 从0到最大值均匀分布
    }
    sns.heatmap(pivot_var, annot=True, fmt='.6f', cmap='RdYlGn_r',
                vmin=0, vmax=var_max,
                cbar_kws=cbar_kws_var, ax=ax4)
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('D Model')
    ax4.set_title('Variance Heatmap (Green=More Stable)')

    # 5. 箱线图
    ax5 = axes[4]
    box_data = []
    labels = []
    all_f1 = []
    for _, row in df.iterrows():
        f_scores = row.get('all_f1_scores', [])
        if f_scores:
            box_data.append(f_scores)
            labels.append(f"BS{row['batch_size']}\nD{row['d_model']}")
            all_f1.extend(f_scores)

    if box_data:
        f1_min = np.min(all_f1) if all_f1 else 0
        f1_max = np.max(all_f1) if all_f1 else 1

        bp = ax5.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, mean_val in zip(bp['boxes'], df['mean_f1']):
            if f1_max > f1_min:
                normalized = (mean_val - f1_min) / (f1_max - f1_min)
            else:
                normalized = 0.5
            color = plt.cm.RdYlGn(normalized)
            patch.set_facecolor(color)

        ax5.set_xlabel('(Batch Size, D Model)')
        ax5.set_ylabel('F1 Score')
        ax5.set_title('F1 Score Distribution by Combination')
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    else:
        ax5.text(0.5, 0.5, 'No data', ha='center', va='center')

    # 6. 统计指标对比
    ax6 = axes[5]
    x = np.arange(len(df))
    width = 0.25
    ax6.bar(x - width, df['mean_f1'], width, label='Mean', color='skyblue')
    ax6.bar(x, df['median_f1'], width, label='Median', color='lightgreen')
    ax6.bar(x + width, df['best_f1'], width, label='Best', color='coral')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f"BS{b}\nD{d}" for b, d in zip(df['batch_size'], df['d_model'])])
    ax6.set_xlabel('Combination')
    ax6.set_ylabel('F1 Score')
    ax6.set_title('Mean vs Median vs Best F1')
    ax6.legend()
    plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)

    # 7. 稳定性分析
    ax7 = axes[6]
    x = np.arange(len(df))
    width = 0.25
    ax7.bar(x - width, df['std_f1'], width, label='Std', color='lightcoral')
    ax7.bar(x, df['range_f1'], width, label='Range', color='lightblue')
    ax7.bar(x + width, df['iqr_f1'], width, label='IQR', color='lightyellow')
    ax7.set_xticks(x)
    ax7.set_xticklabels([f"BS{b}\nD{d}" for b, d in zip(df['batch_size'], df['d_model'])])
    ax7.set_xlabel('Combination')
    ax7.set_ylabel('Value')
    ax7.set_title('Stability Analysis (Lower=Better)')
    ax7.legend()
    plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')
    ax7.grid(True, alpha=0.3)

    # 8. 总结表格
    ax8 = axes[7]
    ax8.axis('off')

    best_idx = df['best_f1'].idxmax()
    best_row = df.loc[best_idx]
    stable_idx = df['std_f1'].idxmin()
    stable_row = df.loc[stable_idx]
    high_mean_idx = df['mean_f1'].idxmax()
    high_mean_row = df.loc[high_mean_idx]

    f1_min_val = df['mean_f1'].min()
    f1_max_val = df['mean_f1'].max()
    std_min_val = df['std_f1'].min()
    std_max_val = df['std_f1'].max()

    summary_text = f"""
=== STATISTICAL SUMMARY ===

[Global Best F1]
BS: {best_row['batch_size']}, D: {best_row['d_model']}
Best F1: {best_row['best_f1']:.4f}
Mean +/- Std: {best_row['mean_f1']:.4f} +/- {best_row['std_f1']:.4f}
Median: {best_row['median_f1']:.4f}
Variance: {best_row['variance_f1']:.6f}

[Most Stable (Lowest Std)]
BS: {stable_row['batch_size']}, D: {stable_row['d_model']}
Std: {stable_row['std_f1']:.4f}
Mean F1: {stable_row['mean_f1']:.4f}

[Highest Mean F1]
BS: {high_mean_row['batch_size']}, D: {high_mean_row['d_model']}
Mean F1: {high_mean_row['mean_f1']:.4f}
Std: {high_mean_row['std_f1']:.4f}

[Data Range]
Mean F1: {f1_min_val:.4f} ~ {f1_max_val:.4f}
Std F1: {std_min_val:.4f} ~ {std_max_val:.4f}

Total Configs: {len(df)}
Seeds/Config: {df['seeds_count'].iloc[0] if 'seeds_count' in df.columns else 100}
    """
    ax8.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_grid_search_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {dataset_name}_grid_search_analysis.png")
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "MSL"

    print(f"Analyzing results for dataset: {dataset_name}")
    print("=" * 80)

    df = analyze_results(dataset_name, "grid_search_results_final_dchannel")

    if df is not None:
        report_file = f"grid_search_results_final_dchannel/{dataset_name}_analysis_report.csv"
        df.to_csv(report_file, index=False)
        print(f"\nDetailed report saved to: {report_file}")

        # 另外保存统计摘要
        stats_file = f"grid_search_results_final_dchannel/{dataset_name}_statistics_summary.csv"
        stats_df = df[['batch_size', 'd_model', 'mean_f1', 'median_f1', 'std_f1',
                       'variance_f1', 'min_f1', 'max_f1', 'range_f1', 'q25_f1', 'q75_f1', 'iqr_f1', 'best_f1']]
        stats_df.to_csv(stats_file, index=False)
        print(f"Statistics summary saved to: {stats_file}")

        try:
            plot_results(df, dataset_name)
        except Exception as e:
            print(f"Plot generation failed: {e}")

        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)