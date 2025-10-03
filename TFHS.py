import os
import random
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.cluster import DBSCAN
from data_loader import load_data

# 固定随机种子
random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HomogeneousBlockDetector:
    def __init__(self, args):
        self.args = args

    def detect_peaks(self, spectrogram):
        """步骤1: 频率峰值检测"""
        M = spectrogram
        peaks = []

        # 计算全局统计量
        mu_M = np.mean(M)
        sigma_M = np.std(M)
        global_threshold = mu_M + self.args.beta * sigma_M

        # 遍历每个时间帧
        for l in range(M.shape[1]):
            frame = M[:, l]

            # 检测局部最大值
            for k in range(1, len(frame) - 1):
                if frame[k] >= frame[k - 1] and frame[k] >= frame[k + 1]:
                    # 相对能量阈值
                    relative_threshold = self.args.eta * np.max(frame)

                    # 检查所有阈值条件
                    if (frame[k] >= relative_threshold and
                            frame[k] >= global_threshold):
                        peaks.append({
                            'time_idx': l,
                            'freq_idx': k,
                            'magnitude': frame[k]
                        })
        return peaks

    def cluster_peaks(self, peaks, f, t):
        """步骤2: 峰值聚类 (DBSCAN)"""
        if len(peaks) == 0:
            return []

        # 准备特征向量 [频率索引, 归一化幅值]
        features = []
        magnitudes = [p['magnitude'] for p in peaks]
        if len(magnitudes) > 0:
            max_mag = max(magnitudes)
            min_mag = min(magnitudes)
            # 避免除零
            mag_range = max_mag - min_mag if max_mag != min_mag else 1.0
        else:
            mag_range = 1.0

        for p in peaks:
            # 归一化频率索引和幅值
            norm_freq = p['freq_idx'] / len(f)  # 频率索引归一化
            norm_mag = (p['magnitude'] - min_mag) / mag_range  # 幅值归一化
            features.append([norm_freq, norm_mag])

        features = np.array(features)

        # 使用DBSCAN聚类
        clustering = DBSCAN(
            eps=self.args.epsilon,
            min_samples=self.args.min_samples
        ).fit(features)

        labels = clustering.labels_

        # 组织聚类结果
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(peaks[i])

        return list(clusters.values())

    def enforce_temporal_continuity(self, clusters):
        """步骤3: 时域连续性处理"""
        continuous_clusters = []

        for cluster in clusters:
            if len(cluster) == 0:
                continue

            # 按时间排序
            cluster.sort(key=lambda x: x['time_idx'])

            current_group = [cluster[0]]
            for i in range(1, len(cluster)):
                # 检查时间连续性
                if cluster[i]['time_idx'] - current_group[-1]['time_idx'] <= self.args.gamma:
                    current_group.append(cluster[i])
                else:
                    # 时间不连续，开始新组
                    if len(current_group) >= self.args.min_samples:
                        continuous_clusters.append(current_group)
                    current_group = [cluster[i]]

            # 添加最后一组
            if len(current_group) >= self.args.min_samples:
                continuous_clusters.append(current_group)

        return continuous_clusters

    def expand_frequency_boundaries(self, cluster, spectrogram, f):
        """步骤4: 动态边界扩展"""
        if len(cluster) == 0:
            return None, None, None, None

        # 计算聚类的时间范围
        time_indices = [p['time_idx'] for p in cluster]
        l_l = min(time_indices)
        l_r = max(time_indices)

        # 计算聚类的中心频率
        freq_indices = [p['freq_idx'] for p in cluster]
        center_freq = int(np.mean(freq_indices))

        # 计算聚类内的最大能量
        cluster_energies = []
        for p in cluster:
            cluster_energies.append(spectrogram[p['freq_idx'], p['time_idx']])
        max_cluster_energy = max(cluster_energies) if cluster_energies else 0

        # 全局统计
        mu_M = np.mean(spectrogram)

        # 向左扩展寻找低频边界
        k_low = center_freq
        for k in range(center_freq, -1, -1):
            # 计算当前频率在聚类时间范围内的平均能量
            avg_energy = np.mean(spectrogram[k, l_l:l_r + 1])

            # 检查边界条件
            if (avg_energy <= self.args.tau_drop * max_cluster_energy or
                    avg_energy <= self.args.tau_global * mu_M):
                k_low = k
                break
            k_low = k

        # 向右扩展寻找高频边界
        k_high = center_freq
        for k in range(center_freq, spectrogram.shape[0]):
            # 计算当前频率在聚类时间范围内的平均能量
            avg_energy = np.mean(spectrogram[k, l_l:l_r + 1])

            # 检查边界条件
            if (avg_energy <= self.args.tau_drop * max_cluster_energy or
                    avg_energy <= self.args.tau_global * mu_M):
                k_high = k
                break
            k_high = k

        # 确保边界在有效范围内
        k_low = max(0, k_low)
        k_high = min(spectrogram.shape[0] - 1, k_high)

        return l_l, l_r, k_low, k_high

    def optimize_blocks(self, blocks, t, f,spectrogram):
        """步骤5: 时频块优化"""
        if not blocks:
            return blocks

        optimized_blocks = []

        # 简单的重叠合并策略
        merged_blocks = []
        for block in blocks:
            l_l, l_r, k_low, k_high = block

            # 计算块的能量
            block_energy = np.mean(spectrogram[k_low:k_high + 1, l_l:l_r + 1])
            global_energy = np.mean(spectrogram)

            # 过滤低能量块
            if block_energy >= self.args.min_energy_ratio * global_energy:
                merged = False
                for i, existing_block in enumerate(merged_blocks):
                    el_l, el_r, ek_low, ek_high = existing_block

                    # 计算重叠
                    time_overlap = max(0, min(l_r, el_r) - max(l_l, el_l))
                    freq_overlap = max(0, min(k_high, ek_high) - max(k_low, ek_low))

                    time_union = max(l_r, el_r) - min(l_l, el_l)
                    freq_union = max(k_high, ek_high) - min(k_low, ek_low)

                    if time_union > 0 and freq_union > 0:
                        overlap_ratio = (time_overlap * freq_overlap) / (time_union * freq_union)

                        if overlap_ratio >= self.args.delta:
                            # 合并块
                            new_l_l = min(l_l, el_l)
                            new_l_r = max(l_r, el_r)
                            new_k_low = min(k_low, ek_low)
                            new_k_high = max(k_high, ek_high)
                            merged_blocks[i] = (new_l_l, new_l_r, new_k_low, new_k_high)
                            merged = True
                            break

                if not merged:
                    merged_blocks.append(block)

        return merged_blocks

    def detect_blocks(self, spectrogram, t, f):
        """主检测函数"""
        print(f"检测时频图中的同质块...")

        # 步骤1: 峰值检测
        peaks = self.detect_peaks(spectrogram)
        print(f"检测到 {len(peaks)} 个峰值")

        if len(peaks) == 0:
            return []

        # 步骤2: 峰值聚类
        clusters = self.cluster_peaks(peaks, f, t)
        print(f"形成 {len(clusters)} 个聚类")

        # 步骤3: 时域连续性处理
        continuous_clusters = self.enforce_temporal_continuity(clusters)
        print(f"时域连续性处理后剩余 {len(continuous_clusters)} 个聚类")

        # 步骤4: 动态边界扩展
        blocks = []
        for i, cluster in enumerate(continuous_clusters):
            l_l, l_r, k_low, k_high = self.expand_frequency_boundaries(cluster, spectrogram, f)
            if l_l is not None:
                blocks.append((l_l, l_r, k_low, k_high))
                print(f"聚类 {i}: 时间 {l_l}-{l_r}, 频率 {k_low}-{k_high}")

        # 步骤5: 时频块优化
        optimized_blocks = self.optimize_blocks(blocks, t, f,spectrogram)
        print(f"优化后剩余 {len(optimized_blocks)} 个同质块")

        return optimized_blocks


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

    # 初始化检测器
    detector = HomogeneousBlockDetector(args)

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

        # 检测同质块
        blocks = detector.detect_blocks(spec_magnitude, t, f)

        # 绘制时频图
        plt.figure(figsize=(12, 6))
        im = plt.pcolormesh(t, f, spec_magnitude, cmap='viridis')

        # 用红色框标出检测到的同质块
        for i, (l_l, l_r, k_low, k_high) in enumerate(blocks):
            # 转换为实际的时间和频率值
            time_start = t[l_l]
            time_end = t[l_r]
            freq_start = f[k_low]
            freq_end = f[k_high]

            # 绘制红色矩形框
            rect = plt.Rectangle(
                (time_start, freq_start),
                time_end - time_start,
                freq_end - freq_start,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            plt.gca().add_patch(rect)

            # 添加标签
            plt.text(
                time_start,
                freq_end,
                f'Block {i + 1}',
                color='red',
                fontsize=8,
                verticalalignment='top'
            )

            # 输出块的范围信息
            print(f"同质块 {i + 1}: 时间范围 {time_start:.2f}-{time_end:.2f}s, "
                  f"频率范围 {freq_start:.2f}-{freq_end:.2f}Hz")

        plt.title(
            f'数据集：{args.dataset} | 类别：{int(label)} | 样本索引：{idx}\n'
            f'采样频率：{args.fs} Hz | 检测到 {len(blocks)} 个同质块',
            fontsize=12
        )
        plt.xlabel('时间 (秒)', fontsize=10)
        plt.ylabel('频率 (Hz)', fontsize=10)
        cbar = plt.colorbar(im)
        cbar.set_label('幅度 (线性)', fontsize=10)

        plt.tight_layout()

        # 保存图片
        filename = f'{args.dataset}_class_{int(label)}_sample_{idx}_with_blocks.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"带同质块标记的时频图已保存到：{save_path}")

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="显示指定类别的5个随机样本的时频图并检测同质块")

    # 1. 数据集相关参数
    parser.add_argument('--dataset', type=str, default='computer',
                        choices=['toydata_final', 'mixedshapes', 'yoga', 'forda', 'fordb',
                                 'strawberry', 'cincecgtorso', 'gunpointmalefemale', 'arrowhead', 'twopatterns'],
                        help='要加载的数据集名称')
    parser.add_argument('--class_label', type=int, default=0,
                        help='要显示的样本类别（整数，例如 0、1、2）')

    # 2. STFT参数
    parser.add_argument('--fs', type=int, default=1, help='采样频率（Hz）')
    parser.add_argument('--nperseg', type=int, default=16, help='STFT窗口长度')
    parser.add_argument('--noverlap', type=int, default=8, help='STFT窗口重叠长度')

    # 3. 同质块检测参数
    parser.add_argument('--eta', type=float, default=0.5,
                        help='相对能量阈值因子 (0-1)')
    parser.add_argument('--beta', type=float, default=1.5,
                        help='全局能量阈值系数')
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='DBSCAN聚类半径')
    parser.add_argument('--min_samples', type=int, default=2,
                        help='聚类最小样本数')
    parser.add_argument('--gamma', type=int, default=1,
                        help='时间连续性阈值')
    parser.add_argument('--tau_drop', type=float, default=0.6,
                        help='能量衰减阈值')
    parser.add_argument('--tau_global', type=float, default=1.2,
                        help='全局能量阈值系数')
    parser.add_argument('--delta', type=float, default=0.3,
                        help='重叠率阈值')
    parser.add_argument('--min_energy_ratio', type=float, default=0.1,
                        help='最小能量比阈值')

    args = parser.parse_args()

    show_class_samples_spectrogram(args)