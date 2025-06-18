import torch # 导入 PyTorch 库，用于构建和训练神经网络
import numpy as np # 导入 NumPy 库，用于数值计算
import torch.nn as nn # 导入 torch.nn 模块，包含了构建神经网络所需的所有层和函数
from .clip import clip # 从本地 clip 模块导入 clip，用于加载 CLIP 模型
from .region_awareness import get_backbone # 从本地 region_awareness 模块导入 get_backbone 函数，用于获取区域感知骨干网络


class LipFD(nn.Module):
    """
    LipFD 模型类，继承自 torch.nn.Module。
    该模型结合了 CLIP 编码器和区域感知骨干网络，用于唇部伪造检测。
    """
    def __init__(self, name, num_classes=1):
        """
        初始化 LipFD 模型。

        Args:
            name (str): CLIP 模型的名称（例如 "RN50", "ViT-B/32"）。
            num_classes (int): 分类任务的类别数量，默认为 1。
        """
        super(LipFD, self).__init__()

        # 定义一个卷积层，用于对输入图像进行下采样
        # 输入通道 3 (RGB), 输出通道 3, 卷积核大小 5x5, 步长 5
        # 将 (1120, 1120) 尺寸的图像下采样到 (224, 224)，以适应 CLIP 模型的输入尺寸
        self.conv1 = nn.Conv2d(
            3, 3, kernel_size=5, stride=5
        )  # (1120, 1120) -> (224, 224)
        # 加载 CLIP 模型，包括编码器和预处理函数
        # device="cpu" 表示模型加载到 CPU，后续可以根据需要移动到 GPU
        self.encoder, self.preprocess = clip.load(name, device="cpu")
        # 获取区域感知骨干网络
        self.backbone = get_backbone()

    def forward(self, x, feature):
        """
        LipFD 模型的前向传播。

        Args:
            x (torch.Tensor): 输入图像张量。
            feature (torch.Tensor): 额外的特征张量，通常来自 CLIP 编码器。

        Returns:
            torch.Tensor: 骨干网络的输出。
        """
        # 将输入 x 和 feature 传递给区域感知骨干网络进行处理
        return self.backbone(x, feature)

    def get_features(self, x):
        """
        从输入图像中提取特征。

        Args:
            x (torch.Tensor): 输入图像张量。

        Returns:
            torch.Tensor: 经过 CLIP 编码器提取的特征。
        """
        # 首先通过 conv1 对输入图像进行下采样
        x = self.conv1(x)
        # 然后使用 CLIP 编码器对下采样后的图像进行编码，提取特征
        features = self.encoder.encode_image(x)
        return features


class RALoss(nn.Module):
    """
    区域感知损失 (Region Awareness Loss) 类，继承自 torch.nn.Module。
    该损失函数旨在惩罚 alphas_max 和 alphas_org 之间的差异。
    """
    def __init__(self):
        """
        初始化 RALoss。
        """
        super(RALoss, self).__init__()

    def forward(self, alphas_max, alphas_org):
        """
        计算区域感知损失。

        Args:
            alphas_max (list of torch.Tensor): 包含最大 alpha 值的张量列表。
            alphas_org (list of torch.Tensor): 包含原始 alpha 值的张量列表。

        Returns:
            torch.Tensor: 计算得到的损失值。
        """
        loss = 0.0 # 初始化总损失
        # 获取批次大小，假设 alphas_org 列表中的第一个张量不为空
        batch_size = alphas_org[0].shape[0]
        # 遍历 alpha 值的列表
        for i in range(len(alphas_org)):
            loss_wt = 0.0 # 初始化当前批次的加权损失
            # 遍历批次中的每个样本
            for j in range(batch_size):
                # 计算损失权重：10 除以 exp(alphas_max - alphas_org)
                # .to(alphas_max[i][j].device) 确保张量在正确的设备上
                loss_wt += torch.Tensor([10]).to(alphas_max[i][j].device) / np.exp(
                    alphas_max[i][j] - alphas_org[i][j]
                )
            # 将当前批次的平均加权损失累加到总损失中
            loss += loss_wt / batch_size
        return loss # 返回总损失
 