import torch
import pandas as pd
import random
from collections import Counter
from data_loader import *
from scipy.fft import fft, ifft
from models import *
from utils.util import *


torch.set_num_threads(32)
random.seed(42)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XAITrainer():
    def __init__(self, args, net):
        self.args = args
        self.model = net.to(device)
        # 修改为
        import argparse
        with torch.serialization.safe_globals([argparse.Namespace]):
            a = torch.load(self.args.classification_model)
        self.model.load_state_dict(a['model_state_dict'])

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Initialize and seed the generator
        self.generator = torch.Generator()
        self.generator.manual_seed(911)

        ds, height, width = load_data(self.args.dataset, self.args)
        self.height = height
        self.width = width
        groups = generate_region_groups(self.height, self.width, 1, 1)
        self.args.groups = groups
        train_size = int(0.8 * len(ds))
        val_size = int(0.1 * len(ds))
        test_size = len(ds) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size], generator=self.generator)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)
        #xai
        self.label = self.args.label
        self.num_per = self.args.num_perturbations
        self.selected_regions = self.args.selected_regions
        self.groups = groups

    def insertion(self, selected_positions, positions_consider):
        inverse_data = []
        position_scores = []
        indices_probability_scores = []
        class_probability_scores = list(0. for i in range(self.num_per))
        total_count = 0
        for n, (data, labels, spectrogram, rbp_spec, rbp_signal) in enumerate(self.test_loader):
            #check if prediction and labels match
            data_check = torch.tensor(data)
            data_check = data_check.unsqueeze(1).float()
            data, data_check, labels, spectrogram, rbp_spec, rbp_signal= data.to(device), data_check.to(device), labels.to(device), spectrogram.to(device), rbp_spec.to(device), rbp_signal.to(device)
            output_check = self.model(data_check)
            _, predicted = torch.max(output_check, 1)

            data = data[(labels == self.label) & (predicted == self.label)]
            spectrogram = spectrogram[(labels == self.label) & (predicted == self.label)]
            rbp_spec = rbp_spec[(labels == self.label) & (predicted == self.label)]
            rbp_signal = rbp_signal[(labels == self.label) & (predicted == self.label)]
            labels = labels[(labels == self.label) & (predicted == self.label)]

            # Check if any samples remain after filtering
            if data.shape[0] == 0:
                continue  # Skip this batch if no samples remain

            spectrogram = spectrogram.unsqueeze(1).to(torch.complex64)

            masked_tensor = rbp_spec.to(torch.complex64).to(device)

            if not selected_positions:
                #generate random indices
                indices_list = generate_random_indices(self.num_per, len(self.groups), self.selected_regions)
                positions_consider = self.groups
                combinations = positions_consider

                for i in range(self.num_per):
                    new_tensor = masked_tensor.clone()

                    #flatten data, new_tensor and spectrogram
                    data = data.view(data.shape[0], 1, -1).to(torch.float64)
                    new_tensor = new_tensor.view(new_tensor.shape[0], 1, -1)
                    spectrogram = spectrogram.view(spectrogram.shape[0], 1, -1)

                    #all combinations of positions to consider
                    for j in [combinations[a] for a in indices_list[i]]:
                        new_tensor[:, :, j] = spectrogram[:, :, j]

                    new_tensor = new_tensor.reshape(new_tensor.shape[0],1,self.height,self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.height,self.width)

                    x = new_tensor.to(device)

                    #apply inverse stft
                    _,inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)

                    #apply forward pass
                    with torch.no_grad():
                        x = self.model(x.float())

                    x = torch.softmax(x, dim=1)
                    # x = x-initial_x
                    class_probability_scores[i] = torch.sum(x[:,self.label])

            else:
                #insert selected positions
                new_tensor = masked_tensor.clone()
                for position in selected_positions:
                    data = data.reshape(data.shape[0],1,-1).to(torch.float64)
                    new_tensor = new_tensor.reshape(new_tensor.shape[0],1,-1)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,-1)
                    new_tensor[:,:,position] = spectrogram[:,:,position]
                    new_tensor = new_tensor.reshape(new_tensor.shape[0],1,self.height,self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.height,self.width)

                if n == 0:
                    #remove selected positions from positions to consider
                    position_list = positions_consider
                    designated_positions = selected_positions

                    for sublist in designated_positions:
                        if sublist in position_list:
                            position_list.remove(sublist)

                    combinations = position_list #after removing selected regions
                    indices_list = generate_random_indices(self.num_per, len(positions_consider), self.selected_regions)

                for i in range(self.num_per):
                    altered_tensor = new_tensor.clone()
                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0],1,-1)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,-1)

                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0],1,self.height,self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.height,self.width)

                    x = altered_tensor.to(device)

                    _,inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)

                    #apply forward pass
                    with torch.no_grad():
                        x = self.model(x.float())

                    x = torch.softmax(x, dim=1)
                    # x = x - initial_x
                    class_probability_scores[i] = torch.sum(x[:,self.label])

            total_count += data.shape[0]

        if total_count == 0:
            print("No more samples for the classifier prediction label that matches the gt label. Recommend to train the classifier model more epochs or use different hyperparameters.")
            return selected_positions, positions_consider

        class_probability_scores = [x / total_count for x in class_probability_scores]
        count = Counter([index for sublist in indices_list for index in sublist])
        summed_scores_for_indices = sum_scores_for_each_index(indices_list, class_probability_scores)
        for i in range(len(summed_scores_for_indices)):
            indices_probability_scores.append(summed_scores_for_indices[i] / count[i])
        max_position = torch.argmax(torch.tensor(indices_probability_scores))
        max_position = combinations[max_position]

        selected_positions.append(max_position)
        return selected_positions, positions_consider

    def deletion(self, selected_positions, positions_consider):
        inverse_data = []
        position_scores = []
        indices_probability_scores = []
        class_probability_scores = list(0. for i in range(self.num_per))
        total_count = 0

        for n, (data, labels, spectrogram, rbp_spec, rbp_signal) in enumerate(self.test_loader):
            data_check = torch.tensor(data)
            data_check = data_check.unsqueeze(1).float()
            data, data_check, labels, spectrogram, rbp_spec, rbp_signal = data.to(device), data_check.to(device), labels.to(device), spectrogram.to(device), rbp_spec.to(device), rbp_signal.to(device)
            output_check = self.model(data_check)
            _, predicted = torch.max(output_check, 1)

            data = data[(labels == self.label) & (predicted == self.label)]
            spectrogram = spectrogram[(labels == self.label) & (predicted == self.label)]
            rbp_spec = rbp_spec[(labels == self.label) & (predicted == self.label)]
            rbp_signal = rbp_signal[(labels == self.label) & (predicted == self.label)]
            labels = labels[(labels == self.label) & (predicted == self.label)]

            # Check if any samples remain after filtering
            if data.shape[0] == 0:
                continue  # Skip this batch if no samples remain

            spectrogram = spectrogram.unsqueeze(1).to(torch.complex64)
            masked_tensor = rbp_spec.to(torch.complex64)

            if not selected_positions:
                indices_list = generate_random_indices(self.num_per, len(self.groups), self.selected_regions)
                positions_consider = self.groups
                combinations = positions_consider
                for i in range(self.num_per):
                    new_tensor = spectrogram.clone()
                    new_tensor = new_tensor.view(new_tensor.shape[0], 1, -1)
                    masked_tensor = masked_tensor.view(masked_tensor.shape[0], 1, -1)

                    for j in [combinations[a] for a in indices_list[i]]:
                        new_tensor[:, :, j] = masked_tensor[:, :, j]

                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, self.height, self.width)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, self.height, self.width)

                    x = new_tensor.to(device)

                    _, inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)

                    with torch.no_grad():
                        x = self.model(x.float())

                    x = torch.softmax(x, dim=1)
                    class_probability_scores[i] = torch.sum(x[:,self.label])

            else:
                new_tensor = spectrogram.clone()
                for position in selected_positions:
                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, -1)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, -1)
                    new_tensor[:,:,position] = masked_tensor[:,:,position]
                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, self.height, self.width)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, self.height, self.width)

                if n == 0:
                    position_list = positions_consider
                    designated_positions = selected_positions

                    for sublist in designated_positions:
                        if sublist in position_list:
                            position_list.remove(sublist)

                    combinations = position_list
                    indices_list = generate_random_indices(self.num_per, len(positions_consider), self.selected_regions)

                for i in range(self.num_per):
                    altered_tensor = new_tensor.clone()
                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0], 1, -1)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, -1)
                    for j in [combinations[a] for a in indices_list[i]]:
                        altered_tensor[:,:,j] = masked_tensor[:,:,j]
                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0], 1, self.height, self.width)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, self.height, self.width)

                    x = altered_tensor.to(device)

                    _, inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)

                    with torch.no_grad():
                        x = self.model(x.float())

                    x = torch.softmax(x, dim=1)
                    class_probability_scores[i] = torch.sum(x[:,self.label])

            total_count += data.shape[0]

        if total_count == 0:
            print("No more samples for the classifier prediction label that matches the gt label. Recommend to train the classifier model more epochs or use different hyperparameters.")
            return selected_positions, positions_consider

        class_probability_scores = [x / total_count for x in class_probability_scores]
        count = Counter([index for sublist in indices_list for index in sublist])
        summed_scores_for_indices = sum_scores_for_each_index(indices_list, class_probability_scores)
        for i in range(len(summed_scores_for_indices)):
            indices_probability_scores.append(summed_scores_for_indices[i] / count[i])
        min_position = torch.argmin(torch.tensor(indices_probability_scores))
        min_position = combinations[min_position]

        selected_positions.append(min_position)
        return selected_positions, positions_consider

    def combined(self, selected_positions, positions_consider):
        position_scores = []
        indices_probability_scores = []
        indices_probability_scores_del = []
        class_probability_scores = list(0. for i in range(self.num_per))
        class_probability_scores_del = list(0. for i in range(self.num_per))
        total_count = 0
        indices_list = []  # 避免完全无赋值的极端情况
        combinations = self.groups

        for n, (data, labels, spectrogram, rbp_spec, rbp_signal) in enumerate(self.test_loader):
            data_check = torch.tensor(data)
            data_check = data_check.unsqueeze(1).float()
            data, data_check, labels, spectrogram, rbp_spec, rbp_signal = data.to(device), data_check.to(device), labels.to(device), spectrogram.to(device), rbp_spec.to(device), rbp_signal.to(device)
            output_check = self.model(data_check)
            _, predicted = torch.max(output_check, 1)

            data = data[(labels == self.label) & (predicted == self.label)]
            spectrogram = spectrogram[(labels == self.label) & (predicted == self.label)]
            rbp_spec = rbp_spec[(labels == self.label) & (predicted == self.label)]
            rbp_signal = rbp_signal[(labels == self.label) & (predicted == self.label)]
            labels = labels[(labels == self.label) & (predicted == self.label)]

            # Check if any samples remain after filtering
            if data.shape[0] == 0:
                continue  # Skip this batch if no samples remain

            spectrogram = spectrogram.unsqueeze(1).to(torch.complex64)
            spectrogram_del = spectrogram.clone()

            masked_tensor = rbp_spec.to(torch.complex64)
            masked_tensor_del = masked_tensor.clone()

            if not selected_positions:
                indices_list = generate_random_indices(self.num_per, len(self.groups), self.selected_regions)
                positions_consider = self.groups
                combinations = positions_consider
                data = data.view(data.shape[0], 1, -1).to(torch.float64)
                for i in range(self.num_per):
                    new_tensor = masked_tensor.clone()
                    new_tensor_del = spectrogram_del.clone()
                    new_tensor = new_tensor.view(new_tensor.shape[0], 1, -1)
                    spectrogram = spectrogram.view(spectrogram.shape[0], 1, -1)
                    new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0], 1, -1)
                    masked_tensor_del = masked_tensor_del.reshape(masked_tensor_del.shape[0], 1, -1)

                    for j in [combinations[a] for a in indices_list[i]]:
                        new_tensor[:, :, j] = spectrogram[:, :, j]
                        new_tensor_del[:,:,j] = masked_tensor_del[:,:,j]

                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, self.height, self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, self.height, self.width)
                    new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0], 1, self.height, self.width)
                    masked_tensor_del = masked_tensor_del.reshape(masked_tensor_del.shape[0], 1, self.height, self.width)

                    x = new_tensor.to(device)
                    x_del = new_tensor_del.to(device)

                    _, inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)
                    _, inverse_stft_del = istft(x_del.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x_del = torch.tensor(inverse_stft_del).to(device)

                    with torch.no_grad():
                        x = self.model(x.float())
                        x_del = self.model(x_del.float())

                    x = torch.softmax(x, dim=1)
                    class_probability_scores[i] = torch.sum(x[:,self.label])
                    x_del = torch.softmax(x_del, dim=1)
                    class_probability_scores_del[i] = torch.sum(x_del[:,self.label])

            else:
                new_tensor = masked_tensor.clone()
                new_tensor_del = spectrogram.clone()

                for position in selected_positions:
                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, -1)
                    new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0], 1, -1)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, -1)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, -1)
                    new_tensor[:,:,position] = spectrogram[:,:,position]
                    new_tensor_del[:,:,position] = masked_tensor[:,:,position]
                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, self.height, self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, self.height, self.width)
                    new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0], 1, self.height, self.width)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, self.height, self.width)

                if not indices_list:
                    position_list = positions_consider
                    designated_positions = selected_positions

                    for sublist in designated_positions:
                        if sublist in position_list:
                            position_list.remove(sublist)

                    combinations = position_list
                    indices_list = generate_random_indices(self.num_per, len(positions_consider), self.selected_regions)

                for i in range(self.num_per):
                    altered_tensor = new_tensor.clone()
                    altered_tensor_del = new_tensor_del.clone()
                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0], 1, -1)
                    altered_tensor_del = altered_tensor_del.reshape(altered_tensor_del.shape[0], 1, -1)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, -1)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, -1)
                    for j in [combinations[a] for a in indices_list[i]]:
                        altered_tensor[:,:,j] = spectrogram[:,:,j]
                        altered_tensor_del[:,:,j] = masked_tensor[:,:,j]
                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0], 1, self.height, self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, self.height, self.width)
                    altered_tensor_del = altered_tensor_del.reshape(altered_tensor_del.shape[0], 1, self.height, self.width)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, self.height, self.width)

                    x = altered_tensor.to(device)
                    x_del = altered_tensor_del.to(device)

                    _, inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)
                    _, inverse_stft_del = istft(x_del.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x_del = torch.tensor(inverse_stft_del).to(device)

                    with torch.no_grad():
                        x = self.model(x.float())
                        x_del = self.model(x_del.float())

                    x = torch.softmax(x, dim=1)
                    class_probability_scores[i] = torch.sum(x[:,self.label])
                    x_del = torch.softmax(x_del, dim=1)
                    class_probability_scores_del[i] = torch.sum(x_del[:,self.label])

            total_count += data.shape[0]

        if total_count == 0:
            print("No more samples for the classifier prediction label that matches the gt label. Recommend to train the classifier model more epochs or use different hyperparameters.")
            return selected_positions, positions_consider

        class_probability_scores = [x / total_count for x in class_probability_scores]
        class_probability_scores_del = [x / total_count for x in class_probability_scores_del]

        count = Counter([index for sublist in indices_list for index in sublist])
        summed_scores_for_indices = sum_scores_for_each_index(indices_list, class_probability_scores)
        summed_scores_for_indices_del = sum_scores_for_each_index(indices_list, class_probability_scores_del)
        for i in range(len(summed_scores_for_indices)):
            indices_probability_scores.append(summed_scores_for_indices[i] / count[i])
        for i in range(len(summed_scores_for_indices_del)):
            indices_probability_scores_del.append(summed_scores_for_indices_del[i] / count[i])
        max_position = torch.argmax(self.args.insertion_weight * torch.tensor(indices_probability_scores) - self.args.deletion_weight * torch.tensor(indices_probability_scores_del))
        max_position = combinations[max_position]

        selected_positions.append(max_position)
        return selected_positions, positions_consider

    def get_region_bbox(self, region_idx):
        """
        根据区域索引获取边界框坐标 [x1, y1, x2, y2]
        （关键：需与 generate_region_groups 的区域划分逻辑一致，此处假设 groups 存储的是区域像素索引列表）
        示例逻辑：假设每个区域是 1x1 像素块（对应 __init__ 中 groups=generate_region_groups(...,1,1)）
        若区域是更大的块（如 3x3），需修改此函数计算块的左上角和右下角坐标
        """
        # 假设 self.groups[region_idx] 是单个像素的索引（如 flatten 后的索引）
        pixel_idx = self.groups[region_idx][0]  # 取区域内第一个像素（1x1 区域仅一个）
        # 将 flatten 索引转换为 (height, width) 坐标（y 对应 height，x 对应 width）
        y = pixel_idx // self.width
        x = pixel_idx % self.width
        # 1x1 区域的边界框（若区域更大，需调整 x2=x+block_width-1, y2=y+block_height-1）
        block_width = 1
        block_height = 1
        return [x, y, x + block_width - 1, y + block_height - 1]

    def visualize_important_regions(self, selected_positions, save_dir="./xai_visualizations"):
        """
        可视化并保存模型认为重要的区域（selected_positions）
        :param selected_positions: 由 insertion/deletion/combined 方法返回的重要区域索引列表
        :param save_dir: 图像保存目录
        """
        # 1. 创建保存目录（若不存在）
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 2. 从 test_loader 读取一个批次的有效数据（过滤后有样本的批次）
        data_batch = None
        spectrogram_batch = None
        for n, (data, labels, spectrogram, rbp_spec, rbp_signal) in enumerate(self.test_loader):
            # 复用原有过滤逻辑：只保留标签与预测一致的样本
            data_check = torch.tensor(data).unsqueeze(1).float().to(device)
            output_check = self.model(data_check)
            _, predicted = torch.max(output_check, 1)
            # 过滤样本
            mask = (labels == self.label) & (predicted == self.label)
            if mask.sum() > 0:
                # 取第一个有效样本（可修改为遍历所有有效样本）
                data_batch = data[mask][0].cpu().numpy()  # (height, width)
                spectrogram_batch = spectrogram[mask][0].cpu().numpy()  # (height, width)
                break

        if data_batch is None:
            print("无有效样本用于可视化，请先确保 test_loader 中有符合条件的样本")
            return

        # 3. 绘制并保存图像（以 spectrogram 为例，可替换为 data）
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # 显示 spectrogram（取实部，因为复数无法直接显示）
        im = ax.imshow(np.real(spectrogram_batch), cmap="viridis")
        plt.colorbar(im, ax=ax, label="Spectrogram Amplitude (Real Part)")

        # 4. 框选重要区域（遍历 selected_positions）
        for region_idx in selected_positions:
            x1, y1, x2, y2 = self.get_region_bbox(region_idx)
            # 绘制红色矩形框（线宽2，透明度0.8）
            rect = plt.Rectangle(
                (x1, y1),  # 左上角坐标
                x2 - x1 + 1,  # 宽度
                y2 - y1 + 1,  # 高度
                fill=False,
                edgecolor="red",
                linewidth=2,
                alpha=0.8,
                label="Important Region" if region_idx == selected_positions[0] else ""
            )
            ax.add_patch(rect)

        # 5. 设置图像标题和标签
        ax.set_title(f"Model Important Regions (Label: {self.label})", fontsize=14)
        ax.set_xlabel("Width (Frequency)", fontsize=12)
        ax.set_ylabel("Height (Time)", fontsize=12)
        # 只显示一次图例
        if selected_positions:
            ax.legend(loc="upper right")

        # 6. 保存图像（避免遮挡）
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"important_regions_label_{self.label}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"可视化结果已保存至：{save_path}")
















