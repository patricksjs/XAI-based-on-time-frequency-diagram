import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import shap
from torch import istft, stft

from data_loader import load_data
from models.models import TransformerModel
import warnings
from scipy.signal import stft, istft

warnings.filterwarnings("ignore", category=UserWarning)


class STFTSHAPExplainer:
    def __init__(self, args, model, dataset):
        self.args = args
        self.model = model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset

        self.fs = args.fs
        self.nperseg = args.nperseg
        self.noverlap = args.noverlap
        print(f" 设备: {self.device} | STFT参数: fs={self.fs}, nperseg={self.nperseg}")

    def _get_stft(self, ts):
        """计算单个时域信号的STFT（返回频率轴、时间轴、振幅谱）"""
        # ts: 单个时域信号 (timesteps,) 或 (timesteps, features)（多变量转单通道）
        if ts.ndim > 1:
            ts = ts.mean(axis=1)  # 多变量→单通道（若单变量可删除此句）
        f, t, Zxx = stft(
            ts,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            boundary='zeros'
        )
        return f, t, np.abs(Zxx)  # 返回振幅谱（shape: (num_freq, num_time)）

    def _istft(self, spec_mag):
        """从振幅谱重构复数谱，再通过ISTFT转回时域信号"""
        # 随机相位重构复数谱（保持振幅不变，相位不影响模型核心决策）
        phase = np.random.uniform(0, 2 * np.pi, size=spec_mag.shape)
        Zxx = spec_mag * np.exp(1j * phase)
        # ISTFT转回时域
        _, xrec = istft(
            Zxx,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            boundary='zeros'
        )
        return xrec


    def _prepare_background_and_test(self, target_label=0, num_bg=5, num_test=1):
        bg_specs = []
        test_specs = []
        test_specs_2d = []
        test_labels = []

        # 如果没有指定目标类别，则随机选一个类
        if target_label is None:
            # 从数据集中随机选一个样本的类别作为目标类别
            target_label = int(self.dataset[0][1].item())

        # 收集目标类别的所有样本索引
        same_class_indices = [i for i, (_, label, _, _, _) in enumerate(self.dataset) if
                              int(label.item()) == target_label]
        print(f"为类别 {target_label} 找到 {len(same_class_indices)} 个样本")

        # 随机打乱
        np.random.shuffle(same_class_indices)

        # 取前 num_bg + num_test 个
        selected_indices = same_class_indices[:num_bg + num_test]

        for i, idx in enumerate(selected_indices):
            data, label, _, _, _ = self.dataset[idx]
            ts = data.numpy()
            f, t, spec_mag = self._get_stft(ts)
            spec_flat = spec_mag.flatten()

            if i < num_bg:
                bg_specs.append(spec_flat)
            else:
                test_specs.append(spec_flat)
                test_specs_2d.append(spec_mag)
                test_labels.append(int(label.item()))

        bg_specs = np.array(bg_specs)
        test_specs = np.array(test_specs)
        test_specs_2d = np.array(test_specs_2d)
        test_labels = np.array(test_labels)

        print(f"背景样本: {bg_specs.shape} (全部来自类别 {target_label})")
        print(f"测试样本: {test_specs_2d.shape} (全部来自类别 {target_label})")
        return bg_specs, test_specs, test_specs_2d, test_labels, f, t

    def _model_predict(self, spec_flat_batch):
        """KernelExplainer专用预测函数：输入展平的时频图→ISTFT→时域→模型预测概率"""
        # spec_flat_batch: (batch_size, num_freq*num_time) → 展平的批量时频图
        batch_size = spec_flat_batch.shape[0]
        probs_batch = []  # 批量预测概率（shape: (batch_size, num_classes)）

        # 先获取2D时频图的形状（从背景样本推导，假设所有样本形状一致）
        num_freq = self.args.nperseg // 2 + 1  # STFT默认频率点数：nperseg//2 +1
        num_time = spec_flat_batch.shape[1] // num_freq  # 时间切片数 = 展平维度 / 频率数

        for i in range(batch_size):
            # 1. 展平的时频图 → 恢复为2D时频图
            spec_flat = spec_flat_batch[i]
            spec_mag = spec_flat.reshape((num_freq, num_time))  # 恢复为2D: (num_freq, num_time)

            # 2. ISTFT转回时域信号
            xrec = self._istft(spec_mag)

            # 3. 调整时域长度（匹配Transformer输入的timesteps）
            if xrec.shape[0] > self.args.timesteps:
                xrec = xrec[:self.args.timesteps]
            elif xrec.shape[0] < self.args.timesteps:
                xrec = np.pad(xrec, (0, self.args.timesteps - xrec.shape[0]), mode='constant')

            # 4. 适配Transformer输入格式（(batch_size, timesteps, features)）
            x_tensor = torch.tensor(xrec, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            x_tensor = x_tensor.to(self.device)

            # 5. 模型预测（输出概率）
            with torch.no_grad():
                logits = self.model(x_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # 转为1D概率数组
            probs_batch.append(probs)

        return np.array(probs_batch)  # 返回(batch_size, num_classes)的概率数组

    def explain_and_visualize(self):
        """核心函数：KernelExplainer计算SHAP值 + 可视化"""
        # 1. 准备背景和测试数据（展平为1D向量，适配KernelExplainer）
        bg_specs, test_specs, test_specs_2d, test_labels, f, t = self._prepare_background_and_test()
        num_freq, num_time = test_specs_2d.shape[1], test_specs_2d.shape[2]  # 2D时频图形状

        # 2. 初始化KernelExplainer（关键替换！）
        print("\n初始化SHAP KernelExplainer...")
        # KernelExplainer要求：第一个参数是模型预测函数（输入1D向量，输出概率），第二个参数是背景样本
        explainer = shap.KernelExplainer(
            model=self._model_predict,  # 自定义的预测函数
            data=bg_specs,  # 背景样本（用于计算"基准预测"）
            link="logit"  # 分类任务推荐用logit链接函数，提升解释准确性
        )

        # 3. 计算测试样本的SHAP值（注意：KernelExplainer计算较慢，5个测试样本约1-2分钟）
        print("计算SHAP值（KernelExplainer可能需要1-2分钟，请耐心等待）...")
        # shap_values.shape: (num_test_samples, num_freq*num_time, num_classes)
        shap_values = explainer.shap_values(
            X=test_specs,  # 测试样本（展平后）
            nsamples="auto",  # 自动选择扰动次数（平衡速度和准确性）
            l1_reg="auto"  # 自动L1正则化，减少噪声
        )
        print(f"SHAP值计算完成 | 形状: {np.array(shap_values).shape}")

        # 4. 可视化（展平的SHAP值→恢复为2D热力图）
        print("\n生成SHAP热力图...")
        for idx in range(len(test_specs)):
            # 当前样本数据
            spec_2d = test_specs_2d[idx]  # 原始2D时频图
            true_label = test_labels[idx]  # 真实标签
            # 提取当前样本、当前标签的SHAP值→恢复为2D
            shap_flat = shap_values[idx][:, true_label]  # (num_freq*num_time,)
            shap_2d = shap_flat.reshape((num_freq, num_time))  # 恢复为2D热力图

            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"sample {idx + 1} | label: {true_label}", fontsize=14, fontweight="bold")

            # 子图1：原始STFT时频图
            im1 = ax1.imshow(
                spec_2d,
                origin="lower",
                aspect="auto",
                cmap="viridis"
            )
            ax1.set_xlabel("time [s]", fontsize=12)
            ax1.set_ylabel("frequency [Hz]", fontsize=12)
            ax1.set_title("original STFT time-frequency diagram（振幅谱）", fontsize=12)
            # 坐标轴刻度（适配你的数据维度）
            ax1.set_yticks(np.arange(0, num_freq, 2))
            ax1.set_yticklabels([f"{freq:.2f}" for freq in f[::2]])
            ax1.set_xticks(np.arange(0, num_time, 20))
            ax1.set_xticklabels([f"{time:.2f}" for time in t[::20]])
            plt.colorbar(im1, ax=ax1, label="（Amplitude）")

            # 子图2：SHAP热力图（KernelExplainer结果）
            im2 = ax2.imshow(
                shap_2d,
                origin="lower",
                aspect="auto",
                cmap="bwr"  # 红=正贡献，蓝=负贡献
            )
            ax2.set_xlabel("time [s]", fontsize=12)
            ax2.set_ylabel("frequency [Hz]", fontsize=12)
            ax2.set_title("SHAP heatmap（KernelExplainer）", fontsize=12)
            # 与子图1共享刻度
            ax2.set_yticks(np.arange(0, num_freq, 2))
            ax2.set_yticklabels([f"{freq:.2f}" for freq in f[::2]])
            ax2.set_xticks(np.arange(0, num_time, 20))
            ax2.set_xticklabels([f"{time:.2f}" for time in t[::20]])
            plt.colorbar(im2, ax=ax2, label="SHAP value（正=促进，负=抑制）")

            # 保存与显示
            plt.tight_layout()
            save_dir = "shap_heatmaps_kernel"
            os.makedirs(save_dir, exist_ok=True)
            # 正确代码
            save_path = os.path.join(save_dir, f"{args.dataset}_sample_{idx + 1}_label_{true_label}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"已保存热力图: {save_path}")
            plt.show()

        print("\nSHAP解释完成！所有热力图已保存到: ./shap_heatmaps_kernel")


# -------------------------- 2. 加载Transformer模型 --------------------------
def load_transformer_model(args):
    print(f"\n加载Transformer模型: {args.classification_model}")
    model = TransformerModel(args=args, num_classes=args.num_classes).to(args.device)
    # 加载权重（兼容PyTorch 2.6+的weights_only限制）
    try:
        checkpoint = torch.load(args.classification_model, map_location=args.device, weights_only=True)
    except:
        checkpoint = torch.load(args.classification_model, map_location=args.device, weights_only=False)
    # 加载state_dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("模型加载完成！")
    return model


# -------------------------- 3. 主函数（完整调用流程） --------------------------
if __name__ == "__main__":
    # 3.1 硬编码参数（与训练时一致）
    class Args:
        def __init__(self):
            # 模型与数据路径
            self.classification_model = r"C:\Users\34517\Desktop\zuhui\Time_is_not_Enough-main\classification_models\computer\transformer\transformer.pt"
            self.dataset = "computer"
            self.num_classes = 2  # 4类分类任务
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Transformer结构参数（必须与训练一致）
            self.num_layers = 2
            self.d_model = 64
            self.nhead = 8
            self.dim_feedforward = 256
            self.dropout = 0.2
            self.timesteps = 720  # cincecgtorso的时域长度

            # STFT参数（必须与data_loader一致）
            self.fs = 1
            self.nperseg = 16
            self.noverlap = 8

            # KernelExplainer参数（可调整）
            self.num_bg_samples = 50  # 背景样本数：KernelExplainer建议20-50（太多会变慢）
            self.num_test_samples = 3  # 测试样本数：3个足够（多了计算时间翻倍）


    # 初始化参数
    args = Args()
    print("=" * 60)
    print("参数配置 summary:")
    print(f"数据集: {args.dataset} | 模型: {os.path.basename(args.classification_model)}")
    print(f"Transformer: layers={args.num_layers}, d_model={args.d_model}")
    print(f"STFT: nperseg={args.nperseg} | KernelExplainer: 背景{args.num_bg_samples}个, 测试{args.num_test_samples}个")
    print("=" * 60)

    # 3.2 加载数据
    print("\n加载数据集...")
    dataset, _, _ = load_data(args.dataset, args)  # 直接传位置参数，适配你的load_data
    print(f"数据集加载完成: 总样本数={len(dataset)}")

    # 3.3 加载模型
    model = load_transformer_model(args)

    # 3.4 运行KernelExplainer解释
    explainer = STFTSHAPExplainer(args, model, dataset)
    explainer.explain_and_visualize()