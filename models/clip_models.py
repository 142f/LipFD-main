from .clip import clip # 从本地 clip 模块导入 clip，用于加载 CLIP 模型
from PIL import Image # 导入 PIL 库，用于图像处理
import torch.nn as nn # 导入 torch.nn 模块，包含了构建神经网络所需的所有层和函数


CHANNELS = { # 定义一个字典，存储不同 CLIP 模型对应的特征维度
    "RN50" : 1024, # ResNet-50 模型的特征维度
    "ViT-L/14" : 768 # Vision Transformer Large/14 模型的特征维度
}

class CLIPModel(nn.Module):
    """
    CLIP 模型类，继承自 torch.nn.Module。
    该模型封装了 CLIP 模型的图像编码器，并添加了一个全连接层用于分类。
    """
    def __init__(self, name, num_classes=1):
        """
        初始化 CLIPModel。

        Args:
            name (str): CLIP 模型的名称（例如 "RN50", "ViT-L/14"）。
            num_classes (int): 分类任务的类别数量，默认为 1。
        """
        super(CLIPModel, self).__init__()

        # 加载 CLIP 模型，包括模型本身和预处理函数
        # self.preprocess 在训练过程中不会被直接使用，因为数据预处理通常在 Dataset 类中完成
        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        # 定义一个全连接层，将 CLIP 模型的特征映射到指定类别的输出
        self.fc = nn.Linear( CHANNELS[name], num_classes )
 

    def forward(self, x, return_feature=False):
        """
        CLIPModel 的前向传播。

        Args:
            x (torch.Tensor): 输入图像张量。
            return_feature (bool): 如果为 True，则返回原始特征而不是分类结果，默认为 False。

        Returns:
            torch.Tensor: 如果 return_feature 为 True，则返回图像特征；否则返回分类结果。
        """
        # 使用 CLIP 模型的图像编码器对输入图像进行编码，提取特征
        features = self.model.encode_image(x) 
        # 根据 return_feature 的值决定返回特征还是分类结果
        if return_feature:
            return features
        # 如果不返回特征，则通过全连接层进行分类
        return self.fc(features)

