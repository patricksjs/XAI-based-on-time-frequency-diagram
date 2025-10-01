import os
import random
import torch
import argparse
from data_loader import load_data  # 导入你原有的load_data函数
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
from scipy.signal.windows import hann
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 固定随机种子
random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 一、TFHS算法核心函数（从上一轮对话移植过来）
# ==============================================================================
def compute_stft(signal, fs=200, nperseg=128, noverlap=64):
    """计算信号的STFT幅度谱"""
    window = hann(nperseg, sym=False)
    f, t, Zxx = stft(
        signal,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap
    )
    M = np.abs(Zxx).T  # 转置为[时间帧, 频率点]
    return M, t, f


def detect_peaks(M, eta=0.5, beta=1.5):
    """检测时频矩阵中的显著峰值"""
    L, K = M.shape
    global_mean = np.mean(M)
    global_std = np.std(M)
    peaks = []

    for l in range(L):
        frame = M[l, :]
        frame_max = np.max(frame)
        if frame_max == 0:
            continue

        local_peaks = []
        for k in range(1, K - 1):
            if (frame[k] >= frame[k - 1]) and (frame[k] >= frame[k + 1]) and (frame[k] >= eta * frame_max):
                local_peaks.append((l, k, frame[k]))

        global_threshold = global_mean + beta * global_std
        for (l, k, mag) in local_peaks:
            if mag >= global_threshold:
                peaks.append((l, k, mag))

    return peaks


def cluster_peaks(peaks, eps=0.5, alpha=1.0, min_samples=1):
    """对峰值进行DBSCAN聚类"""
    if len(peaks) == 0:
        return []

    features = np.array([[k, mag] for (l, k, mag) in peaks])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled[:, 1] *= alpha

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(features_scaled)

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(peaks[i])

    return list(clusters.values())


def enforce_temporal_continuity(clusters, gamma=1):
    """确保聚类的时间连续性"""
    continuous_clusters = []

    for cluster in clusters:
        if len(cluster) <= 1:
            continuous_clusters.append(cluster)
            continue

        sorted_cluster = sorted(cluster, key=lambda x: x[0])
        current_subcluster = [sorted_cluster[0]]

        for i in range(1, len(sorted_cluster)):
            prev_l = sorted_cluster[i - 1][0]
            curr_l = sorted_cluster[i][0]
            if curr_l - prev_l <= gamma:
                current_subcluster.append(sorted_cluster[i])
            else:
                continuous_clusters.append(current_subcluster)
                current_subcluster = [sorted_cluster[i]]

        continuous_clusters.append(current_subcluster)

    return continuous_clusters


def dynamic_boundary_expansion(cluster, M, tau_drop=0.6, tau_global=1.2):
    """动态扩展聚类的频率边界"""
    ls = [p[0] for p in cluster]
    l_left = min(ls)
    l_right = max(ls)

    peak_energy = np.mean([M[l, k] for (l, k, mag) in cluster])
    global_mean = np.mean(M)
    global_threshold = tau_global * global_mean

    peak_ks = [p[1] for p in cluster]
    k_low = min(peak_ks)
    while True:
        candidate_k = k_low - 1
        if candidate_k < 0:
            break
        candidate_energy = np.mean(M[l_left:l_right + 1, candidate_k])
        if (candidate_energy >= tau_drop * peak_energy) and (candidate_energy >= global_threshold):
            k_low = candidate_k
        else:
            break

    k_high = max(peak_ks)
    K = M.shape[1]
    while True:
        candidate_k = k_high + 1
        if candidate_k >= K:
            break
        candidate_energy = np.mean(M[l_left:l_right + 1, candidate_k])
        if (candidate_energy >= tau_drop * peak_energy) and (candidate_energy >= global_threshold):
            k_high = candidate_k
        else:
            break

    return (l_left, l_right, k_low, k_high)


def optimize_time_freq_blocks(blocks, M, delta=0.3, segment_window=(2, 2)):
    """优化时频块：去除重叠、补全未分割区域"""
    L, K = M.shape
    mask = np.zeros((L, K), dtype=bool)
    for (l_left, l_right, k_low, k_high) in blocks:
        mask[l_left:l_right + 1, k_low:k_high + 1] = True

    def compute_overlap(block1, block2):
        l1_l, l1_r, k1_l, k1_r = block1
        l2_l, l2_r, k2_l, k2_r = block2
        l_overlap = max(0, min(l1_r, l2_r) - max(l1_l, l2_l) + 1)
        k_overlap = max(0, min(k1_r, k2_r) - max(k1_l, k2_l) + 1)
        overlap_area = l_overlap * k_overlap
        union_area = (l1_r - l1_l + 1) * (k1_r - k1_l + 1) + (l2_r - l2_l + 1) * (k2_r - k2_l + 1) - overlap_area
        return overlap_area / union_area if union_area > 0 else 0

    adjusted_blocks = blocks.copy()
    for i in range(len(adjusted_blocks)):
        for j in range(i + 1, len(adjusted_blocks)):
            block_i = adjusted_blocks[i]
            block_j = adjusted_blocks[j]
            overlap = compute_overlap(block_i, block_j)
            if overlap >= delta:
                l_i_l, l_i_r, k_i_l, k_i_r = block_i
                l_j_l, l_j_r, k_j_l, k_j_r = block_j

                if l_i_r >= l_j_l:
                    new_l_j_l = (l_i_r + l_j_l) // 2 + 1
                    adjusted_blocks[j] = (new_l_j_l, l_j_r, k_j_l, k_j_r)
                if k_i_r >= k_j_l:
                    new_k_j_l = (k_i_r + k_j_l) // 2 + 1
                    adjusted_blocks[j] = (l_j_l, l_j_r, new_k_j_l, k_j_r)

    w_l, w_k = segment_window
    unmasked = ~mask
    complement_blocks = []

    for l in range(0, L, w_l):
        for k in range(0, K, w_k):
            l_end = min(l + w_l - 1, L - 1)
            k_end = min(k + w_k - 1, K - 1)
            if np.any(unmasked[l:l_end + 1, k:k_end + 1]):
                complement_blocks.append((l, l_end, k, k_end))

    final_blocks = adjusted_blocks + complement_blocks
    return final_blocks


def TFHS_algorithm(signal, fs=200, nperseg=128, noverlap=64, **params):
    """TFHS算法主函数"""
    default_params = {
        'eta': 0.5, 'beta': 1.5, 'eps': 0.5, 'alpha': 1.0, 'gamma': 1,
        'tau_drop': 0.6, 'tau_global': 1.2, 'delta': 0.3, 'segment_window': (2, 2)
    }
    params = {**default_params, **params}

    M, t, f = compute_stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    if M.shape[0] == 0 or M.shape[1] == 0:
        raise ValueError("STFT计算失败，请检查信号长度（需大于nperseg）或参数")

    peaks = detect_peaks(M, eta=params['eta'], beta=params['beta'])
    if len(peaks) == 0:
        L, K = M.shape
        w_l, w_k = params['segment_window']
        final_blocks = [
            (l, min(l + w_l - 1, L - 1), k, min(k + w_k - 1, K - 1))
            for l in range(0, L, w_l)
            for k in range(0, K, w_k)
        ]
        return final_blocks, M, t, f

    clusters = cluster_peaks(peaks, eps=params['eps'], alpha=params['alpha'])
    continuous_clusters = enforce_temporal_continuity(clusters, gamma=params['gamma'])

    initial_blocks = []
    for cluster in continuous_clusters:
        block = dynamic_boundary_expansion(
            cluster, M, tau_drop=params['tau_drop'], tau_global=params['tau_global']
        )
        initial_blocks.append(block)

    final_blocks = optimize_time_freq_blocks(
        initial_blocks, M, delta=params['delta'], segment_window=params['segment_window']
    )

    return final_blocks, M, t, f


# ==============================================================================
# 二、集成到你的显示函数中
# ==============================================================================
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

        # 确保数据是1D numpy数组
        if isinstance(data, torch.Tensor):
            data = data.squeeze().cpu().numpy()
        if data.ndim > 1:
            data = data.flatten()

        # 使用TFHS算法分割时频图
        try:
            tfhs_blocks, M, t_stft, f_stft = TFHS_algorithm(
                data,
                fs=args.fs,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
                eta=0.5,  # 可根据需要调整TFHS参数
                beta=1.5,
                eps=0.5,
                gamma=1
            )
        except Exception as e:
            print(f"TFHS算法处理样本 {idx} 时出错: {e}")
            continue

        # 绘制时频图
        plt.figure(figsize=(12, 6))
        im = plt.pcolormesh(t_stft, f_stft, M.T, cmap='viridis')

        # 在时频图上叠加TFHS分割结果
        for (l_l, l_r, k_l, k_r) in tfhs_blocks:
            t_l = t_stft[l_l]
            t_r = t_stft[l_r]
            f_l = f_stft[k_l]
            f_r = f_stft[k_r]
            rect = plt.Rectangle(
                (t_l, f_l),
                t_r - t_l,
                f_r - f_l,
                fill=False,
                edgecolor='red',
                linewidth=1.5,
                linestyle='--',
                alpha=0.9
            )
            plt.gca().add_patch(rect)

        plt.title(
            f'数据集：{args.dataset} | 类别：{int(label)} | 样本索引：{idx}\n'
            f'采样频率：{args.fs} Hz | TFHS分割区域数：{len(tfhs_blocks)}',
            fontsize=12
        )
        plt.xlabel('时间 (秒)', fontsize=10)
        plt.ylabel('频率 (Hz)', fontsize=10)
        cbar = plt.colorbar(im)
        cbar.set_label('幅度 (线性)', fontsize=10)

        plt.tight_layout()

        # 保存图片
        filename = f'{args.dataset}_class_{int(label)}_sample_{idx}_tfhs.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"TFHS分割时频图已保存到：{save_path}")

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="显示指定类别的5个随机样本的时频图并应用TFHS分割")

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

    args = parser.parse_args()

    show_class_samples_spectrogram(args)