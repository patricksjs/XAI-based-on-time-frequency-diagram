import os
import random
import torch
import argparse
from data_loader import load_data  # 假设data_loader.py中包含load_data函数
import matplotlib.pyplot as plt
import numpy as np

# 固定随机种子
random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_class_samples_lineplot(args):
    # 加载数据集
    ds, _, _ = load_data(args.dataset, args)

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
    save_dir = f'line_plots/{args.dataset}_class_{args.class_label}'
    os.makedirs(save_dir, exist_ok=True)

    # 循环绘制线图
    for idx in sample_indices:
        data, label, _, _, _ = ds[idx]
        # 将数据转换为numpy数组（如果是torch张量）
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        # 生成时间轴（假设采样频率为args.fs，时间长度为数据点数量 / 采样频率）
        time = np.arange(len(data)) / args.fs

        plt.figure(figsize=(12, 6))
        plt.plot(time, data, color='blue', linewidth=1.0)

        plt.title(
            f'数据集：{args.dataset} | 类别：{int(label)} | 样本索引：{idx}\n'
            f'采样频率：{args.fs} Hz',
            fontsize=12
        )
        plt.xlabel('时间 (秒)', fontsize=10)
        plt.ylabel('信号幅值', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # 保存图片
        filename = f'{args.dataset}_class_{int(label)}_sample_{idx}.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"线图已保存到：{save_path}")

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="显示指定类别的5个随机样本的线图")

    # 1. 数据集相关参数
    parser.add_argument('--dataset', type=str, default='computer',
                        choices=['toydata_final', 'mixedshapes', 'yoga', 'forda', 'fordb',
                                 'strawberry', 'cincecgtorso', 'gunpointmalefemale', 'arrowhead', 'twopatterns'],
                        help='要加载的数据集名称')
    parser.add_argument('--class_label', type=int, default=0,
                        help='要显示的样本类别（整数，例如 0、1、2）')

    # 2. 采样频率参数
    parser.add_argument('--fs', type=int, default=1, help='采样频率（Hz）')

    # 3. STFT参数（虽然绘制线图用不到，但load_data可能需要）
    parser.add_argument('--nperseg', type=int, default=16, help='STFT窗口长度')
    parser.add_argument('--noverlap', type=int, default=8, help='STFT窗口重叠长度')

    args = parser.parse_args()

    show_class_samples_lineplot(args)