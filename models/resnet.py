import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
"""
尝试从torch.hub导入load_state_dict_from_url函数，若失败则从torch.utils.model_zoo导入load_url并重命名为load_state_dict_from_url
用于后续加载预训练模型权重
"""

# 预训练模型的URL字典，键为模型名称，值为对应的权重文件下载地址
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


# 定义3x3卷积函数，默认带padding以保持特征图尺寸
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding，返回带填充的3x3卷积层"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# 定义1x1卷积函数，用于降维或升维
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution，返回1x1卷积层"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 基本残差块，用于ResNet-18和ResNet-34
class BasicBlock(nn.Module):
    # 输出通道扩张系数，BasicBlock不扩张，设为1
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,         # 输入通道数
        planes: int,           # 块内通道数
        stride: int = 1,       # 卷积步长，用于下采样
        downsample: Optional[nn.Module] = None,  # 下采样模块，用于残差连接的维度匹配
        groups: int = 1,       # 分组卷积组数
        base_width: int = 64,  # 基础宽度，用于ResNeXt变体
        dilation: int = 1,     # 空洞卷积扩张率
        norm_layer: Optional[Callable[..., nn.Module]] = None  # 归一化层类型
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 默认使用BatchNorm2d作为归一化层
        
        # BasicBlock不支持分组卷积和非标准宽度
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # BasicBlock不支持扩张率>1的空洞卷积
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # 当stride≠1时，conv1和downsample都会对输入进行下采样
        self.conv1 = conv3x3(inplanes, planes, stride)  # 第一个3x3卷积
        self.bn1 = norm_layer(planes)  # 第一个批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，inplace=True表示原地修改
        self.conv2 = conv3x3(planes, planes)  # 第二个3x3卷积
        self.bn2 = norm_layer(planes)  # 第二个批归一化层
        self.downsample = downsample  # 下采样模块
        self.stride = stride  # 保存步长

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # 保存输入作为残差连接的恒等映射

        out = self.conv1(x)  # 第一个卷积
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # 激活函数

        out = self.conv2(out)  # 第二个卷积
        out = self.bn2(out)  # 批归一化

        # 如果存在下采样模块，对输入进行下采样以匹配输出维度
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接：输出+恒等映射
        out = self.relu(out)  # 最终激活

        return out


# 瓶颈残差块，用于更深层的ResNet（如ResNet-50/101/152）
class Bottleneck(nn.Module):
    # 输出通道扩张系数，Bottleneck通常扩张4倍
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,         # 输入通道数
        planes: int,           # 块内基础通道数
        stride: int = 1,       # 卷积步长
        downsample: Optional[nn.Module] = None,  # 下采样模块
        groups: int = 1,       # 分组卷积组数
        base_width: int = 64,  # 基础宽度
        dilation: int = 1,     # 空洞卷积扩张率
        norm_layer: Optional[Callable[..., nn.Module]] = None  # 归一化层类型
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 默认归一化层
        
        # 计算实际卷积宽度，考虑分组和基础宽度
        width = int(planes * (base_width / 64.)) * groups
        
        # 当stride≠1时，conv2和downsample都会对输入进行下采样
        self.conv1 = conv1x1(inplanes, width)  # 第一个1x1卷积，用于降维
        self.bn1 = norm_layer(width)  # 批归一化
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # 第二个3x3卷积
        self.bn2 = norm_layer(width)  # 批归一化
        self.conv3 = conv1x1(width, planes * self.expansion)  # 第三个1x1卷积，用于升维
        self.bn3 = norm_layer(planes * self.expansion)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活
        self.downsample = downsample  # 下采样模块
        self.stride = stride  # 保存步长

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # 保存输入作为残差连接

        out = self.conv1(x)  # 1x1卷积降维
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # 3x3卷积
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # 1x1卷积升维
        out = self.bn3(out)

        # 下采样处理
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)  # 最终激活

        return out


# ResNet主体类，整合残差块构建完整网络
class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],  # 残差块类型（BasicBlock或Bottleneck）
        layers: List[int],  # 各阶段残差块数量，如[2,2,2,2]表示4个阶段，每个阶段2个块
        num_classes: int = 1000,  # 分类类别数，默认1000（ImageNet）
        zero_init_residual: bool = False,  # 是否将残差块最后一个BN的权重初始化为0
        groups: int = 1,  # 分组卷积组数
        width_per_group: int = 64,  # 每组宽度
        replace_stride_with_dilation: Optional[List[bool]] = None,  # 是否用空洞卷积代替步长
        norm_layer: Optional[Callable[..., nn.Module]] = None  # 归一化层类型
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 默认归一化层
        self._norm_layer = norm_layer  # 保存归一化层类型

        self.inplanes = 64  # 初始输入通道数
        self.dilation = 1  # 空洞卷积扩张率
        if replace_stride_with_dilation is None:
            # 每个元素表示是否用空洞卷积代替2x2步长，默认都不用
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation应是None或3元素元组")
        self.groups = groups  # 分组数
        self.base_width = width_per_group  # 基础宽度
        
        # 初始卷积层：7x7卷积，步长2，带padding
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # 激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层，下采样
        
        # 构建4个阶段的残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化，输出1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 分类全连接层

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming正态初始化卷积权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 归一化层权重初始化为1，偏置为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 按论文建议，将残差块最后一个BN的权重初始化为0，提升性能
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # 构建单个残差阶段的函数
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer  # 获取归一化层类型
        downsample = None  # 初始化下采样模块
        previous_dilation = self.dilation  # 保存之前的扩张率
        
        # 如果使用空洞卷积代替步长
        if dilate:
            self.dilation *= stride
            stride = 1  # 步长设为1
        
        # 当需要下采样或输入输出通道不匹配时，创建下采样模块
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []  # 初始化层列表
        # 添加第一个残差块，可能包含下采样
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion  # 更新输入通道数
        
        # 添加后续残差块，不包含下采样
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)  # 返回顺序容器

    # 前向传播实现，返回中间特征和最终输出
    def _forward_impl(self, x):
        # 注释基于输入尺寸为224*224的ImageNet图像
        out = {}  # 用于存储中间特征
        
        x = self.conv1(x)  # 7x7卷积
        x = self.bn1(x)    # 批归一化
        x = self.relu(x)   # 激活
        x = self.maxpool(x)  # 最大池化
        out['f0'] = x  # 输出特征：N*64*56*56 

        x = self.layer1(x)  # 第一阶段残差层
        out['f1'] = x  # 输出特征：N*64*56*56

        x = self.layer2(x)  # 第二阶段残差层
        out['f2'] = x  # 输出特征：N*128*28*28

        x = self.layer3(x)  # 第三阶段残差层
        out['f3'] = x  # 输出特征：N*256*14*14
        
        x = self.layer4(x)  # 第四阶段残差层
        out['f4'] = x  # 输出特征：N*512*7*7

        x = self.avgpool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展平特征
        out['penultimate'] = x  # 倒数第二层特征：N*512*block.expansion

        x = self.fc(x)  # 全连接分类层
        out['logits'] = x  # 最终logits：N*num_classes

        # 返回所有特征
        return out

        # 如需仅返回分类结果，取消下面注释
        # return x

    def forward(self, x):
        return self._forward_impl(x)  # 调用前向传播实现


# 通用ResNet模型创建函数
def _resnet(
    arch: str,         # 模型架构名称
    block: Type[Union[BasicBlock, Bottleneck]],  # 残差块类型
    layers: List[int],  # 各阶段块数
    pretrained: bool,  # 是否加载预训练权重
    progress: bool,    # 是否显示下载进度
    **kwargs: Any      # 其他参数
) -> ResNet:
    model = ResNet(block, layers, **kwargs)  # 创建ResNet模型
    if pretrained:
        # 从URL加载预训练权重
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)  # 加载权重
    return model


# 以下是不同深度ResNet的创建函数
def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18模型，来自论文"Deep Residual Learning for Image Recognition"

    Args:
        pretrained (bool): 若为True，返回ImageNet预训练模型
        progress (bool): 若为True，显示下载进度条
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34模型，来自同上论文

    Args:
        pretrained (bool): 若为True，返回ImageNet预训练模型
        progress (bool): 若为True，显示下载进度条
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50模型，来自同上论文

    Args:
        pretrained (bool): 若为True，返回ImageNet预训练模型
        progress (bool): 若为True，显示下载进度条
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101模型，来自同上论文

    Args:
        pretrained (bool): 若为True，返回ImageNet预训练模型
        progress (bool): 若为True，显示下载进度条
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152模型，来自同上论文

    Args:
        pretrained (bool): 若为True，返回ImageNet预训练模型
        progress (bool): 若为True，显示下载进度条
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)