from typing import Callable, List, Optional  # 导入类型提示相关的模块
import torch  # 导入 PyTorch 库
from torch import Tensor  # 从 PyTorch 中导入 Tensor 类型
from .vision_transformer_utils import _log_api_usage_once # 从同级目录的 vision_transformer_utils 模块导入 _log_api_usage_once 函数
interpolate = torch.nn.functional.interpolate  # 获取 PyTorch 的插值函数

# 定义一个冻结的 BatchNorm2d 模块
# 在这个模块中，批归一化层的统计数据（均值和方差）以及可学习的仿射参数（权重和偏置）都是固定的，不会在训练过程中更新。
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d 层，其批处理统计数据和仿射参数是固定的。

    参数:
        num_features (int): 输入特征图的通道数 C。输入张量的形状应为 (N, C, H, W)。
        eps (float): 一个为保证数值稳定性而加到分母上的小值。默认为 1e-5。
    """
    def __init__(
        self,
        num_features: int,  # 输入特征的通道数
        eps: float = 1e-5,  # 为数值稳定性加到分母的小值
    ):
        super().__init__()  # 调用父类的构造函数
        _log_api_usage_once(self)  # 记录 API 使用情况
        self.eps = eps  # 保存 eps 值
        # 注册缓冲区 (buffer)，这些参数不会被视为模型的可训练参数。
        # 权重 (gamma) 初始化为全1张量。
        self.register_buffer("weight", torch.ones(num_features))
        # 偏置 (beta) 初始化为全0张量。
        self.register_buffer("bias", torch.zeros(num_features))
        # 运行均值 (running_mean) 初始化为全0张量。
        self.register_buffer("running_mean", torch.zeros(num_features))
        # 运行方差 (running_var) 初始化为全1张量。
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],  # 错误信息列表
    ):
        # 从 state_dict 中移除 'num_batches_tracked' 参数，因为它在 FrozenBatchNorm2d 中是不需要的。
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]  # 删除该键值对
        # 调用父类的 _load_from_state_dict 方法完成剩余的加载过程。
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: Tensor) -> Tensor:  # 前向传播函数
        # 将权重、偏置、运行均值和运行方差调整形状，以便进行广播操作。
        # 形状从 (num_features,) 变为 (1, num_features, 1, 1)，以匹配输入 x 的 (N, C, H, W) 形状中的 C 维度。
        w = self.weight.reshape(1, -1, 1, 1)  # 权重
        b = self.bias.reshape(1, -1, 1, 1)  # 偏置
        rv = self.running_var.reshape(1, -1, 1, 1)  # 运行方差
        rm = self.running_mean.reshape(1, -1, 1, 1)  # 运行均值
        # 计算归一化操作中的缩放因子 (scale) 和偏置项 (bias)。
        # 归一化公式: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
        # 这里等价于: y = x * [gamma / sqrt(Var[x] + eps)] + [beta - E[x] * gamma / sqrt(Var[x] + eps)]
        # scale = gamma / sqrt(Var[x] + eps)
        # bias = beta - E[x] * scale
        scale = w * (rv + self.eps).rsqrt()  # rsqrt() 计算平方根的倒数
        bias = b - rm * scale  # 计算新的偏置
        # 对输入 x 应用缩放和平移变换。
        return x * scale + bias

    def __repr__(self) -> str:  # 返回模块的字符串表示形式
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"  # 例如 FrozenBatchNorm2d(64, eps=1e-05)

# 定义一个可配置的卷积-归一化-激活 (Conv-Norm-Activation) 模块
# 这个模块将卷积层、归一化层和激活函数层按顺序组合在一起。
class ConvNormActivation(torch.nn.Sequential):
    """
    一个可配置的模块，按顺序执行卷积、归一化和激活操作。

    参数:
        in_channels (int): 输入特征图的通道数。
        out_channels (int): 输出特征图的通道数。
        kernel_size (int, 可选): 卷积核的大小。默认为 3。
        stride (int, 可选): 卷积的步幅。默认为 1。
        padding (int, tuple 或 str, 可选): 卷积的填充大小。如果为 None，则根据 kernel_size 和 dilation 自动计算以保持空间维度。默认为 None。
        groups (int, 可选): 卷积的分组数，用于分组卷积。默认为 1 (标准卷积)。
        norm_layer (Callable[..., torch.nn.Module], 可选): 归一化层。如果为 None，则不使用归一化层。默认为 torch.nn.BatchNorm2d。
        activation_layer (Callable[..., torch.nn.Module], 可选): 激活函数层。如果为 None，则不使用激活函数。默认为 torch.nn.ReLU。
        dilation (int): 卷积的膨胀系数。默认为 1。
        inplace (bool, 可选): 如果为 True，激活函数将执行原地操作。仅当 activation_layer 不为 None 时有效。默认为 True。
        bias (bool, 可选): 卷积层是否使用偏置项。如果为 None，则当 norm_layer 为 None 时，bias 为 True，否则为 False。默认为 None。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,  # 激活函数是否执行原地操作
        bias: Optional[bool] = None,  # 卷积层是否使用偏置
    ) -> None:
        # 如果未指定 padding，则自动计算 padding 大小，以保持输入和输出的空间维度相同（对于 stride=1 的情况）。
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        # 如果未指定 bias，则根据 norm_layer 是否存在来决定。
        # 如果没有归一化层 (norm_layer is None)，则卷积层通常需要偏置 (bias=True)。
        # 如果有归一化层，则偏置通常可以省略 (bias=False)，因为归一化层本身有偏置项。
        if bias is None:
            bias = norm_layer is None
        
        # 构建层列表
        layers = []
        # 1. 添加卷积层
        layers.append(
            torch.nn.Conv2d(
                in_channels,  # 输入通道数
                out_channels,  # 输出通道数
                kernel_size,  # 卷积核大小
                stride,  # 步幅
                padding,  # 填充
                dilation=dilation,  # 膨胀系数
                groups=groups,  # 分组数
                bias=bias,  # 是否使用偏置
            )
        )
        # 2. 如果指定了归一化层，则添加归一化层
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))  # 归一化层的输入通道数为 out_channels
        # 3. 如果指定了激活函数层，则添加激活函数层
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}  # 如果 inplace 不为 None，则传递 inplace 参数
            layers.append(activation_layer(**params))  # 创建激活函数实例
        
        super().__init__(*layers)  # 调用父类 torch.nn.Sequential 的构造函数，将层列表传入
        _log_api_usage_once(self)  # 记录 API 使用情况
        self.out_channels = out_channels  # 保存输出通道数，方便外部访问

# 定义 Squeeze-and-Excitation (SE) 模块
# SE 模块是一种通道注意力机制，通过显式地建模通道间的相互依赖关系来重新校准通道级别的特征响应。
# 参考论文: "Squeeze-and-Excitation Networks" (https://arxiv.org/abs/1709.01507)
class SqueezeExcitation(torch.nn.Module):
    """
    实现 Squeeze-and-Excitation 模块。
    论文中的 δ (delta) 对应这里的 `activation` 参数，σ (sigma) 对应这里的 `scale_activation` 参数。

    参数:
        input_channels (int): 输入特征图的通道数。
        squeeze_channels (int): "Squeeze" 操作后，第一个全连接层输出的通道数（即瓶颈层的通道数）。
        activation (Callable[..., torch.nn.Module], 可选): 第一个全连接层后的激活函数 (对应论文中的 δ)。默认为 torch.nn.ReLU。
        scale_activation (Callable[..., torch.nn.Module]): 第二个全连接层后的激活函数，用于生成通道权重 (对应论文中的 σ)。默认为 torch.nn.Sigmoid。
    """
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,  # 第二个全连接层后的激活函数 (sigma)
    ) -> None:
        super().__init__()  # 调用父类的构造函数
        _log_api_usage_once(self)  # 记录 API 使用情况
        
        # Squeeze 操作: 使用自适应全局平均池化将每个通道的空间维度压缩为 1x1。
        # 输入形状: (N, C, H, W) -> 输出形状: (N, C, 1, 1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        
        # Excitation 操作: 包含两个全连接层 (这里用 1x1 卷积实现) 和激活函数。
        # 第一个全连接层 (fc1): 将通道数从 input_channels 降维到 squeeze_channels (瓶颈层)。
        # 输入形状: (N, C, 1, 1) -> 输出形状: (N, squeeze_channels, 1, 1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        # 第二个全连接层 (fc2): 将通道数从 squeeze_channels 恢复到 input_channels。
        # 输入形状: (N, squeeze_channels, 1, 1) -> 输出形状: (N, input_channels, 1, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        
        # 第一个全连接层后的激活函数 (delta)。
        self.activation = activation()
        # 第二个全连接层后的激活函数 (sigma)，用于生成通道权重。
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:  # 计算通道注意力权重的辅助函数
        # 1. Squeeze: 全局平均池化，得到每个通道的全局信息。
        # input 形状: (N, C, H, W) -> scale 形状: (N, C, 1, 1)
        scale = self.avgpool(input)
        # 2. Excitation: 第一个全连接层 (降维) + 激活函数。
        # scale 形状: (N, C, 1, 1) -> (N, squeeze_channels, 1, 1)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        # 3. Excitation: 第二个全连接层 (升维) + 激活函数 (生成权重)。
        # scale 形状: (N, squeeze_channels, 1, 1) -> (N, input_channels, 1, 1)
        scale = self.fc2(scale)
        # scale 形状: (N, input_channels, 1, 1)，每个值在 (0, 1) 之间，表示对应通道的权重。
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:  # 前向传播函数
        # 1. 计算通道注意力权重 (scale)。
        # scale 形状: (N, C, 1, 1)
        scale = self._scale(input)
        # 2. 将计算得到的通道权重与原始输入特征图相乘 (通道加权)。
        # input 形状: (N, C, H, W)
        # scale 会自动广播到 (N, C, H, W) 的形状。
        return scale * input  # 输出形状: (N, C, H, W)
