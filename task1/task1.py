# train_bpp.py

import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from collections import namedtuple

# -----------------------------
# 数据生成
# -----------------------------

def rotate_item(item_size):
    """
    生成长方体的六种可能的旋转之一。

    参数:
        item_size (tuple): (x, y, z) 尺寸。

    返回:
        tuple: 旋转后的 (x, y, z) 尺寸。
    """
    rotations = list(itertools.permutations(item_size))
    return random.choice(rotations)

def generate_bpp_data_with_labels(container_size=(100, 100, 100), min_items=10, max_items=50):
    """
    生成装箱问题的数据，包括物品尺寸和放置位置。

    参数:
        container_size (tuple): 容器的尺寸 (x, y, z)。
        min_items (int): 最小物品数量。
        max_items (int): 最大物品数量。

    返回:
        tuple: (normalized_sizes, normalized_origins)
            normalized_sizes: 归一化后的物品尺寸列表 [(x, y, z), ...]。
            normalized_origins: 归一化后的物品放置位置列表 [(x_pos, y_pos, z_pos), ...]。
    """
    # 每个容器表示为 (origin_x, origin_y, origin_z, size_x, size_y, size_z)
    containers = [(0, 0, 0, container_size[0], container_size[1], container_size[2])]
    placements = []  # 最终放置的物品
    N = random.randint(min_items, max_items)

    while len(placements) < N and containers:
        index = random.randint(0, len(containers) - 1)
        container = containers.pop(index)
        origin_x, origin_y, origin_z, size_x, size_y, size_z = container

        # 随机选择一个轴进行分割：0=x, 1=y, 2=z
        axis = random.randint(0, 2)
        if axis == 0 and size_x > 1:
            p = random.uniform(0.1, size_x - 0.1)
            size1 = p
            size2 = size_x - p
            container1 = (origin_x, origin_y, origin_z, size1, size_y, size_z)
            container2 = (origin_x + size1, origin_y, origin_z, size2, size_y, size_z)
        elif axis == 1 and size_y > 1:
            p = random.uniform(0.1, size_y - 0.1)
            size1 = p
            size2 = size_y - p
            container1 = (origin_x, origin_y, origin_z, size_x, size1, size_z)
            container2 = (origin_x, origin_y + size1, origin_z, size_x, size2, size_z)
        elif axis == 2 and size_z > 1:
            p = random.uniform(0.1, size_z - 0.1)
            size1 = p
            size2 = size_z - p
            container1 = (origin_x, origin_y, origin_z, size_x, size_y, size1)
            container2 = (origin_x, origin_y, origin_z + size1, size_x, size_y, size2)
        else:
            # 无法分割，视为最终物品
            placements.append(container)
            continue

        # 随机旋转分割后的容器
        size1_rot = rotate_item((container1[3], container1[4], container1[5]))
        size2_rot = rotate_item((container2[3], container2[4], container2[5]))

        # 更新容器尺寸（旋转后）
        container1_rot = (container1[0], container1[1], container1[2], size1_rot[0], size1_rot[1], size1_rot[2])
        container2_rot = (container2[0], container2[1], container2[2], size2_rot[0], size2_rot[1], size2_rot[2])

        # 将旋转后的容器添加回容器列表
        containers.append(container1_rot)
        containers.append(container2_rot)

    # 剩余的容器视为最终放置的物品
    for container in containers:
        placements.append(container)

    # 归一化尺寸和位置
    normalized_sizes = []
    normalized_origins = []
    for placement in placements:
        origin_x, origin_y, origin_z, size_x, size_y, size_z = placement
        norm_origin_x = origin_x / container_size[0]
        norm_origin_y = origin_y / container_size[1]
        norm_origin_z = origin_z / container_size[2]
        norm_size_x = size_x / container_size[0]
        norm_size_y = size_y / container_size[1]
        norm_size_z = size_z / container_size[2]
        normalized_origins.append((norm_origin_x, norm_origin_y, norm_origin_z))
        normalized_sizes.append((norm_size_x, norm_size_y, norm_size_z))

    return normalized_sizes, normalized_origins

# -----------------------------
# 数据集与DataLoader
# -----------------------------

class BPPDataset(Dataset):
    """
    自定义装箱问题数据集。
    每个数据点包含物品尺寸和对应的放置位置。
    """
    def __init__(self, data):
        self.data = data  # 每个数据点为 (sizes, origins)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sizes, origins = self.data[idx]
        sizes = torch.tensor(sizes, dtype=torch.float32)       # [num_items, 3]
        origins = torch.tensor(origins, dtype=torch.float32)   # [num_items, 3]
        return sizes, origins

def collate_fn(batch):
    """
    自定义collate函数，处理可变长度的序列，通过填充使其长度一致。

    参数:
        batch (list): 包含多个 (sizes, origins) 元组的列表

    返回:
        tuple: 填充后的尺寸和位置张量
    """
    sizes_batch = [item[0] for item in batch]
    origins_batch = [item[1] for item in batch]

    # 找到批次中最大物品数量
    max_len = max(len(sizes) for sizes in sizes_batch)

    # 填充尺寸和位置
    padded_sizes = []
    padded_origins = []
    for sizes, origins in zip(sizes_batch, origins_batch):
        padding_size = max_len - len(sizes)
        if padding_size > 0:
            # 使用0进行填充
            padded_sizes.append(torch.cat([sizes, torch.zeros((padding_size, 3))], dim=0))
            padded_origins.append(torch.cat([origins, torch.zeros((padding_size, 3))], dim=0))
        else:
            padded_sizes.append(sizes)
            padded_origins.append(origins)

    # 堆叠成张量
    padded_sizes = torch.stack(padded_sizes)       # [batch_size, max_len, 3]
    padded_origins = torch.stack(padded_origins)   # [batch_size, max_len, 3]
    return padded_sizes, padded_origins

# -----------------------------
# 模型定义
# -----------------------------

class Encoder(nn.Module):
    """
    使用LSTM处理物品尺寸的编码器模块。
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        self.bn = nn.BatchNorm1d(hidden_dim)  # 批量归一化

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        _, (hidden, _) = self.lstm(x)
        hidden = self.bn(hidden[-1])  # 取最后一层的hidden状态
        return hidden  # [batch_size, hidden_dim]

class Decoder(nn.Module):
    """
    使用LSTM预测物品放置位置的解码器模块。
    """
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # 确保输出在[0,1]范围内

    def forward(self, x, hidden):
        # x: [batch_size, seq_len, hidden_dim]（解码器输入）
        # hidden: [batch_size, hidden_dim]
        batch_size = hidden.size(0)
        hidden = hidden.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)  # [num_layers, batch_size, hidden_dim]
        cell = torch.zeros_like(hidden)  # 初始化cell状态
        output, _ = self.lstm(x, (hidden, cell))
        output = self.fc(output)         # [batch_size, seq_len, output_dim]
        return self.sigmoid(output)      # [batch_size, seq_len, output_dim]

class Seq2Seq(nn.Module):
    """
    结合编码器和解码器的Seq2Seq模型。
    """
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # src: [batch_size, seq_len, input_dim]
        # trg: [batch_size, seq_len, hidden_dim]（此处传入全零作为起始输入）
        hidden = self.encoder(src)       # [batch_size, hidden_dim]
        trg_input = torch.zeros((trg.size(0), trg.size(1), hidden.size(1)), device=trg.device)  # [batch_size, seq_len, hidden_dim]
        output = self.decoder(trg_input, hidden)  # [batch_size, seq_len, output_dim]
        return output

# -----------------------------
# 自定义损失函数
# -----------------------------

def compute_overlap(predicted_origins, sizes):
    """
    计算已放置物品之间的总重叠体积。

    参数:
        predicted_origins (Tensor): [batch_size, seq_len, 3]
        sizes (Tensor): [batch_size, seq_len, 3]

    返回:
        Tensor: [batch_size] 每个批次的总重叠体积
    """
    batch_size, seq_len, _ = predicted_origins.size()

    # 扩展张量以进行成对比较
    origins1 = predicted_origins.unsqueeze(2)  # [batch_size, seq_len, 1, 3]
    origins2 = predicted_origins.unsqueeze(1)  # [batch_size, 1, seq_len, 3]

    sizes1 = sizes.unsqueeze(2)  # [batch_size, seq_len, 1, 3]
    sizes2 = sizes.unsqueeze(1)  # [batch_size, 1, seq_len, 3]

    # 计算重叠的最小坐标和最大坐标
    min_coords = torch.max(origins1, origins2)  # [batch_size, seq_len, seq_len, 3]
    max_coords = torch.min(origins1 + sizes1, origins2 + sizes2)  # [batch_size, seq_len, seq_len, 3]

    # 计算每个轴上的重叠量
    overlap = (max_coords - min_coords).clamp(min=0)  # [batch_size, seq_len, seq_len, 3]

    # 计算重叠体积
    overlap_volume = overlap.prod(dim=3)  # [batch_size, seq_len, seq_len]

    # 移除自身重叠（物品与自身的重叠不计）
    mask = torch.eye(seq_len, device=overlap_volume.device).bool()
    overlap_volume = overlap_volume.masked_fill(mask.unsqueeze(0), 0)

    # 计算每个批次的总重叠体积
    total_overlap = overlap_volume.sum(dim=(1, 2))  # [batch_size]

    return total_overlap

def compute_boundary(predicted_origins, sizes, container_size=(1.0, 1.0, 1.0)):
    """
    计算物品是否超出容器边界的总量。

    参数:
        predicted_origins (Tensor): [batch_size, seq_len, 3]
        sizes (Tensor): [batch_size, seq_len, 3]
        container_size (tuple): 容器尺寸 (x, y, z)，归一化到[0,1]

    返回:
        Tensor: [batch_size] 每个批次超出边界的总量
    """
    exceeds = (predicted_origins + sizes) > torch.tensor(container_size, device=predicted_origins.device).unsqueeze(0).unsqueeze(0)
    exceeds = exceeds.float().sum(dim=2)  # [batch_size, seq_len]

    # 计算每个批次超出边界的总量
    boundary_loss = exceeds.clamp(min=0).sum(dim=1)  # [batch_size]

    return boundary_loss

def custom_loss(predicted_origins, true_origins, sizes, container_size=(1.0, 1.0, 1.0), overlap_penalty=1.0, boundary_penalty=1.0):
    """
    自定义损失函数，结合位置误差、重叠惩罚和边界惩罚。

    参数:
        predicted_origins (Tensor): [batch_size, seq_len, 3]
        true_origins (Tensor): [batch_size, seq_len, 3]
        sizes (Tensor): [batch_size, seq_len, 3]
        container_size (tuple): 容器尺寸 (x, y, z)，归一化到[0,1]
        overlap_penalty (float): 重叠惩罚权重
        boundary_penalty (float): 边界惩罚权重

    返回:
        Tensor: 标量损失值
    """
    # 位置误差
    reg_loss = F.smooth_l1_loss(predicted_origins, true_origins)

    # 重叠惩罚
    overlap_loss = compute_overlap(predicted_origins, sizes)  # [batch_size]

    # 边界惩罚
    boundary_loss = compute_boundary(predicted_origins, sizes, container_size)  # [batch_size]

    # 总损失 = 位置误差 + 重叠惩罚 + 边界惩罚
    total_loss = reg_loss + overlap_penalty * overlap_loss.mean() + boundary_penalty * boundary_loss.mean()
    return total_loss

# -----------------------------
# 训练与验证函数
# -----------------------------

def validate_model(model, val_loader, criterion, container_size=(1.0,1.0,1.0)):
    """
    在验证集上评估模型性能。

    参数:
        model (nn.Module): 训练好的模型。
        val_loader (DataLoader): 验证集的DataLoader。
        criterion (function): 损失函数。
        container_size (tuple): 容器尺寸 (x, y, z)，归一化到[0,1]

    返回:
        float: 平均验证损失。
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sizes, origins in val_loader:
            sizes = sizes.to(device)
            origins = origins.to(device)
            output = model(sizes, torch.zeros_like(sizes).to(device))  # 解码器输入为全零张量
            loss = criterion(output, origins, sizes, container_size)
            val_loss += loss.item()
    model.train()
    return val_loss / len(val_loader)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, container_size=(1.0,1.0,1.0)):
    """
    训练模型。

    参数:
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练集的DataLoader。
        val_loader (DataLoader): 验证集的DataLoader。
        criterion (function): 损失函数。
        optimizer (Optimizer): 优化器。
        scheduler (Scheduler): 学习率调度器。
        num_epochs (int): 训练的轮数。
        container_size (tuple): 容器尺寸 (x, y, z)，归一化到[0,1]

    返回:
        None
    """
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for sizes, origins in train_loader:
            sizes = sizes.to(device)
            origins = origins.to(device)

            optimizer.zero_grad()
            output = model(sizes, torch.zeros_like(sizes).to(device))  # 解码器输入为全零张量
            loss = criterion(output, origins, sizes, container_size)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 验证
        val_loss = validate_model(model, val_loader, criterion, container_size)
        scheduler.step(val_loss)

        print(f'第{epoch+1}轮/共{num_epochs}轮, 训练损失: {epoch_loss/len(train_loader):.4f}, 验证损失: {val_loss:.4f}')

# -----------------------------
# 主执行部分
# -----------------------------

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 生成数据
    print('生成数据...')
    data = [generate_bpp_data_with_labels() for _ in range(1200)]
    train_data = data[:1000]
    val_data = data[1000:1100]
    test_data = data[1100:]

    # 创建数据集
    train_dataset = BPPDataset(train_data)
    val_dataset = BPPDataset(val_data)
    test_dataset = BPPDataset(test_data)

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    input_dim = 3
    hidden_dim = 512
    output_dim = 3
    num_layers = 2

    encoder = Encoder(input_dim, hidden_dim, num_layers).to(device)
    decoder = Decoder(hidden_dim, output_dim, num_layers).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    # 定义损失函数与优化器
    criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练模型
    num_epochs = 50
    print('开始训练...')
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)

    # 保存训练好的模型
    torch.save(model.state_dict(), 'bpp_seq2seq.pth')
    print('模型已保存到 bpp_seq2seq.pth')
