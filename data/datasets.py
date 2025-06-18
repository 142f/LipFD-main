# 导入所需的库
import cv2  # OpenCV 库，用于图像处理
import torch  # PyTorch 库，用于张量计算和神经网络
import torchvision.transforms as transforms  # PyTorch 的图像转换模块
from torch.utils.data import Dataset  # PyTorch 的数据集基类
import utils  # 导入自定义的 utils 模块，可能包含辅助函数


class AVLip(Dataset):
    """
    AVLip 数据集类，继承自 torch.utils.data.Dataset。
    用于加载和处理 AVLip 数据集中的图像和标签。
    """
    def __init__(self, opt):
        """
        初始化 AVLip 数据集。

        参数:
            opt: 包含数据集配置选项的对象。
                 期望属性包括: data_label, real_list_path, fake_list_path。
        """
        # 断言 data_label 必须是 "train" 或 "val" 中的一个
        assert opt.data_label in ["train", "val"]
        self.data_label = opt.data_label  # 数据标签（"train" 或 "val"）
        # 从指定路径加载真实图像列表
        self.real_list = utils.get_list(opt.real_list_path)
        # 从指定路径加载伪造图像列表
        self.fake_list = utils.get_list(opt.fake_list_path)
        self.label_dict = dict()  # 创建一个空字典用于存储图像路径到标签的映射
        # 为真实图像分配标签 0
        for i in self.real_list:
            self.label_dict[i] = 0
        # 为伪造图像分配标签 1
        for i in self.fake_list:
            self.label_dict[i] = 1
        # 合并真实图像列表和伪造图像列表
        self.total_list = self.real_list + self.fake_list

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.total_list)

    def __getitem__(self, idx):
        """
        获取数据集中指定索引的样本。

        参数:
            idx: 样本的索引。

        返回:
            tuple: 包含图像、裁剪图像列表和标签的元组 (img, crops, label)。
        """
        img_path = self.total_list[idx]  # 获取指定索引的图像路径
        label = self.label_dict[img_path]  # 获取对应图像的标签
        # 使用 OpenCV 读取图像，并转换为 PyTorch 张量，数据类型为 float32
        img = torch.tensor(cv2.imread(img_path), dtype=torch.float32)
        # 转换图像维度顺序，从 (H, W, C) 转换为 (C, H, W)
        img = img.permute(2, 0, 1)
        # 对图像进行归一化处理
        # 这些均值和标准差通常是针对特定数据集（如 ImageNet）预计算得到的
        # 注意：这里的 crops 变量名在初始赋值时可能具有误导性，它首先存储了归一化后的完整图像
        crops = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])(img)
        
        # 裁剪图像
        # crops[0] 对应 1.0x 缩放（原始尺寸裁剪）
        # crops[1] 对应 0.65x 缩放
        # crops[2] 对应 0.45x 缩放
        # 从图像的特定区域（img[:, 500:, i:i + 500]）裁剪出 5 个子图像，并调整大小为 (224, 224)
        # 这里的 img 实际上是上面归一化后的图像，即 `crops` 变量在被重新赋值前的值
        # 为了清晰起见，最好使用不同的变量名，或者在注释中明确指出这一点
        # 假设这里的 img 应该是指原始的、未归一化的图像张量 `img` (permute之后)
        # 如果 `img` 在 Normalize 后被修改，那么这里的裁剪是基于归一化后的图像
        # 如果 `img` 未被修改，那么裁剪是基于未归一化的图像，然后对裁剪后的图像进行归一化（如果需要）
        # 从代码逻辑看，`transforms.Normalize` 返回的是一个新的张量，所以 `img` 变量本身未被修改
        # 因此，这里的裁剪是基于原始的、permute后的图像 `img`
        # 然而，下面的 `crops[0][i]` 又被用于进一步裁剪，这意味着 `crops[0]` 存储的是未归一化的裁剪图像
        # 这与 `crops` 变量在前面被赋值为归一化图像相矛盾。这里可能存在逻辑上的不清晰或潜在错误。
        # 假设意图是对原始图像进行裁剪，然后再进行归一化（通常的做法）
        # 或者，如果意图是对归一化后的图像进行裁剪，那么 Normalize 操作应该在裁剪之后，或者对每个裁剪块单独进行
        # 当前代码的写法是：先对完整图像 Normalize (结果存储在 crops)，然后用原始 img 进行裁剪，再对裁剪块 Resize
        # 这意味着 Normalize 步骤实际上没有应用到最终的 crops 上，除非 Resize 内部有 Normalize
        # 让我们假设 `crops` 变量在下面被重新定义，并且之前的 Normalize 是针对全局特征提取的（如果模型需要的话）

        # 重新初始化 crops 为一个包含三个空列表的列表
        # crops[0] 将存储 1.0x 尺度的裁剪图像
        # crops[1] 将存储 0.65x 尺度的裁剪图像
        # crops[2] 将存储 0.45x 尺度的裁剪图像
        crops = [[transforms.Resize((224, 224))(img[:, 500:, i*100:i*100 + 500]) for i in range(5)], [], []]
        # 定义用于进一步裁剪的索引
        crop_idx = [(28, 196), (61, 163)]  # (top_left_y, bottom_right_y) and (top_left_x, bottom_right_x) for square crop
        # 遍历 1.0x 尺度的裁剪图像
        for i in range(len(crops[0])):
            # 从 1.0x 裁剪图像中裁剪出 0.65x 尺度的图像 (196-28 = 168x168)，然后调整大小为 (224, 224)
            crops[1].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
            # 从 1.0x 裁剪图像中裁剪出 0.45x 尺度的图像 (163-61 = 102x102)，然后调整大小为 (224, 224)
            crops[2].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))
        # 将原始图像调整大小为 (1120, 1120)
        img = transforms.Resize((1120, 1120))(img)

        # 返回调整大小后的原始图像、多尺度裁剪图像列表和标签
        return img, crops, label
