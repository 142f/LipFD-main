import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch.nn.functional import softmax

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# 预训练模型URL字典（包含不同深度ResNet的权重地址）
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wideget_backbone50_2': 'https://download.pytorch.org/models/wideget_backbone50_2-95faca4d.pth',
    'wideget_backbone101_2': 'https://download.pytorch.org/models/wideget_backbone101_2-32ee1156.pth',
}


# 定义3x3卷积层（带填充，保持特征图尺寸）
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# 定义1x1卷积层（用于降维/升维）
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 基本残差块（适用于ResNet-18/34）
class BasicBlock(nn.Module):
    expansion: int = 1  # 输出通道扩张系数（BasicBlock不扩张）

    def __init__(
            self,
            inplanes: int,         # 输入通道数
            planes: int,           # 块内通道数
            stride: int = 1,       # 卷积步长（用于下采样）
            downsample: Optional[nn.Module] = None,  # 下采样模块（残差连接维度匹配）
            groups: int = 1,       # 分组卷积组数
            base_width: int = 64,  # 基础宽度（用于ResNeXt）
            dilation: int = 1,     # 空洞卷积扩张率
            norm_layer: Optional[Callable[..., nn.Module]] = None  # 归一化层类型
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 默认使用BatchNorm2d
        
        # 校验参数合法性（BasicBlock不支持分组/非标准宽度/扩张卷积）
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock仅支持groups=1和base_width=64')
        if dilation > 1:
            raise NotImplementedError("BasicBlock不支持dilation>1")
        
        # 构建残差路径（两个3x3卷积）
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample  # 残差连接下采样
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # 保存输入作为残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接维度匹配
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差相加
        out = self.relu(out)

        return out


# 瓶颈残差块（适用于ResNet-50/101/152）
class Bottleneck(nn.Module):
    expansion: int = 4  # 输出通道扩张系数（通常扩张4倍）

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
            norm_layer = nn.BatchNorm2d
        
        # 计算实际卷积宽度（考虑分组和基础宽度）
        width = int(planes * (base_width / 64.)) * groups
        
        # 构建瓶颈路径（1x1-3x3-1x1卷积）
        self.conv1 = conv1x1(inplanes, width)       # 降维
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # 特征提取
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)  # 升维
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 改进的ResNet模型（支持多输入融合和注意力机制）
class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],  # 残差块类型
            layers: List[int],  # 各阶段残差块数量
            num_classes: int = 1000,  # 分类类别数（默认为1，用于回归任务）
            zero_init_residual: bool = False,  # 残差块最后BN权重是否初始化为0
            groups: int = 1,  # 分组数
            width_per_group: int = 64,  # 每组宽度
            replace_stride_with_dilation: Optional[List[bool]] = None,  # 是否用空洞卷积替代步长
            norm_layer: Optional[Callable[..., nn.Module]] = None  # 归一化层类型
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation需为None或3元素元组")
        self.groups = groups
        self.base_width = width_per_group
        
        # 初始特征提取层（7x7卷积+最大池化）
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建4个阶段的残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 权重计算模块（融合区域特征和全局特征）
        self.get_weight = nn.Sequential(
            nn.Linear(512 * block.expansion + 768, 1),  # 输入为区域特征+全局特征
            nn.Sigmoid()  # 输出权重（0-1之间）
        )
        
        # 最终回归头（输出单个预测值）
        self.fc = nn.Linear(512 * block.expansion + 768, 1)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 残差块最后BN权重初始化（提升性能）
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """构建单个残差阶段（包含多个残差块）"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 构建下采样模块（维度不匹配时）
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, feature):
        """前向传播核心逻辑（支持多输入融合和注意力加权）"""
        # x: 多组输入图像列表（如不同视角/区域的图像）
        # feature: 全局特征向量（维度768，如ViT输出的全局特征）
        features, weights, parts, weights_max, weights_org = [list() for i in range(5)]
        
        # 遍历每组输入的不同样本
        for i in range(len(x[0])):
            features.clear()
            weights.clear()
            # 处理每组输入的所有图像
            for j in range(len(x)):
                f = x[j][i]  # 取出第j组第i个样本
                # 经过ResNet特征提取
                f = self.conv1(f)
                f = self.bn1(f)
                f = self.relu(f)
                f = self.maxpool(f)
                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                f = self.avgpool(f)
                f = torch.flatten(f, 1)  # 展平为特征向量
                
                # 融合区域特征和全局特征
                features.append(torch.cat([f, feature[i:i+1]], dim=1))
                # 计算该特征的权重
                weights.append(self.get_weight(features[-1]))

            # 堆叠特征和权重（维度：batch, feature_dim, group_num）
            features_stack = torch.stack(features, dim=2)
            weights_stack = torch.stack(weights, dim=2)
            # 对权重进行softmax归一化（沿group维度）
            weights_stack = softmax(weights_stack, dim=2)

            # 保存最大权重、原始权重和加权特征
            weights_max.append(weights_stack[:, :, :len(x)].max(dim=2)[0])
            weights_org.append(weights_stack[:, :, 0])
            parts.append(features_stack.mul(weights_stack).sum(2).div(weights_stack.sum(2)))
        
        # 堆叠所有组的加权特征并求平均
        parts_stack = torch.stack(parts, dim=0)
        out = parts_stack.sum(0).div(parts_stack.shape[0])

        # 最终回归预测
        pred_score = self.fc(out)

        return pred_score, weights_max, weights_org

    def forward(self, x, feature):
        """前向传播接口（调用_impl函数）"""
        return self._forward_impl(x, feature)


# 通用模型创建函数（加载预训练权重）
def _get_backbone(
        arch: str,         # 模型架构名称
        block: Type[Union[BasicBlock, Bottleneck]],  # 残差块类型
        layers: List[int],  # 各阶段块数
        pretrained: bool,  # 是否加载预训练权重
        progress: bool,    # 是否显示下载进度
        **kwargs: Any      # 其他参数
) -> ResNet:
    model = ResNet(block, layers, num_classes=1, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


# 创建ResNet-50模型（默认返回该模型）
def get_backbone(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """
    创建ResNet-50模型（改进版，支持多输入融合）
    
    Args:
        pretrained (bool): 若为True，加载ImageNet预训练权重
        progress (bool): 若为True，显示下载进度条
    """
    return _get_backbone('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


if __name__ == '__main__':
    # 示例用法
    model = get_backbone()
    # 构造输入数据：3组数据，每组5个样本，每个样本为(10, 3, 224, 224)的图像
    data = [[] for i in range(3)]
    for i in range(3):
        for j in range(5):
            data[i].append(torch.rand((10, 3, 224, 224)))
    # 构造全局特征：10个样本，每个样本768维
    feature = torch.rand((10, 768))
    # 前向传播
    pred_score, weights_max, weights_org = model(data, feature)