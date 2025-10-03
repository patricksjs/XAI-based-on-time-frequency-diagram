import os
import random

import torch
import argparse
from data_loader import load_data  # 导入你原有的load_data函数（包含TimeSeriesDataset）
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

# 固定随机种子
random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_class_samples_spectrogram(args):
    ds, num_freq, num_slices = load_data(args.dataset, args)

    # 收集指定类别的所有样本索引
    class_indices = []
    for idx in range(len(ds)):
        _, label, _, _, _ = ds[idx]
        if int(label) == args.class_label:
            class_indices.append(idx)

    if not class_indices:
        raise ValueError(f"类别 {args.class_label} 在数据集中不存在！")

    # 随机选择5个样本（如果不足5个则全部显示）
    sample_indices = random.sample(class_indices, min(5, len(class_indices)))

    print(f"找到类别 {args.class_label} 的样本共 {len(class_indices)} 个，随机显示其中 5 个：{sample_indices}")

    # 创建保存目录
    save_dir = f'spectrogram_plots/{args.dataset}_class_{args.class_label}'
    os.makedirs(save_dir, exist_ok=True)

    # 循环绘制时频图
    for idx in sample_indices:
        data, label, _, _, _ = ds[idx]

        # 计算STFT
        f, t, spectrogram = stft(
            data,
            fs=args.fs,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            boundary='zeros',
            window='hann'
        )
        spec_magnitude = np.abs(spectrogram)

        plt.figure(figsize=(12, 6))
        im = plt.pcolormesh(t, f, spec_magnitude, cmap='viridis')

        plt.title(
            f'数据集：{args.dataset} | 类别：{int(label)} | 样本索引：{idx}\n'
            f'采样频率：{args.fs} Hz',
            fontsize=12
        )
        plt.xlabel('时间 (秒)', fontsize=10)
        plt.ylabel('频率 (Hz)', fontsize=10)
        cbar = plt.colorbar(im)
        cbar.set_label('幅度 (线性)', fontsize=10)

        plt.tight_layout()

        # 保存图片
        filename = f'{args.dataset}_class_{int(label)}_sample_{idx}.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"时频图已保存到：{save_path}")

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="显示指定类别的5个随机样本的时频图")

    # 1. 数据集相关参数
    parser.add_argument('--dataset', type=str, default='cincecgtorso',
                        choices=['toydata_final', 'mixedshapes', 'yoga', 'forda', 'fordb',
                                 'strawberry', 'cincecgtorso', 'gunpointmalefemale', 'arrowhead', 'twopatterns'],
                        help='要加载的数据集名称')
    parser.add_argument('--class_label', type=int, default=0,
                        help='要显示的样本类别（整数，例如 0、1、2）')

    # 2. STFT参数
    parser.add_argument('--fs', type=int, default=1, help='采样频率（Hz）')
    parser.add_argument('--nperseg', type=int, default=16, help='STFT窗口长度')
    parser.add_argument('--noverlap', type=int, default=8, help='STFT窗口重叠长度')

    args = parser.parse_args()

    show_class_samples_spectrogram(args)