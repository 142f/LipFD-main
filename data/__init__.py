# 导入所需的库
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
# 从同级目录的 datasets 模块导入 AVLip 类
from .datasets import AVLip


def get_bal_sampler(dataset):
    """
    为数据集创建一个平衡采样器。

    参数:
        dataset: 数据集对象，期望具有 'datasets' 属性，该属性是一个包含子数据集的列表，
                 每个子数据集具有 'targets' 属性。

    返回:
        WeightedRandomSampler: 配置好的权重随机采样器。
    """
    targets = []
    # 遍历数据集中的所有子数据集
    for d in dataset.datasets:
        # 将每个子数据集的目标（标签）添加到 targets 列表中
        targets.extend(d.targets)

    # 计算每个类别的样本数量
    ratio = np.bincount(targets)
    # 计算每个类别的权重，权重与类别样本数量成反比
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    # 根据每个样本的类别分配权重
    sample_weights = w[targets]
    # 创建一个 WeightedRandomSampler 实例
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(opt):
    """
    根据提供的选项创建一个数据加载器。

    参数:
        opt: 包含数据加载器配置选项的对象。
             期望属性包括: serial_batches, isTrain, class_bal, batch_size, num_threads。

    返回:
        torch.utils.data.DataLoader: 配置好的数据加载器。
    """
    # 根据选项确定是否打乱数据
    # 如果是训练模式且未使用类别平衡，则打乱数据；否则不打乱
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    # 创建 AVLip 数据集实例
    dataset = AVLip(opt)

    # 如果启用了类别平衡，则获取平衡采样器
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    # 创建数据加载器实例
    data_loader = torch.utils.data.DataLoader(
        dataset,  # 数据集
        batch_size=opt.batch_size,  # 批量大小
        shuffle=True,  # 是否打乱数据（如果 sampler 为 None，则此参数生效）
        sampler=sampler,  # 采样器
        num_workers=int(opt.num_threads),  # 用于数据加载的工作线程数
    )
    return data_loader
