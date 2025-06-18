import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional

import torch
import torch.nn as nn

# 导入工具函数（处理API使用日志、卷积标准化激活模块等）
from .vision_transformer_misc import ConvNormActivation
from .vision_transformer_utils import _log_api_usage_once

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# 预训练模型URL字典（不同配置的ViT模型权重地址）
model_urls = {
    "vit_b_16": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
    "vit_b_32": "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
    "vit_l_16": "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
    "vit_l_32": "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
}


# 卷积茎配置（用于替代传统补丁划分的多级卷积结构）
class ConvStemConfig(NamedTuple):
    out_channels: int          # 输出通道数
    kernel_size: int           # 卷积核尺寸
    stride: int                # 步长
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d  # 归一化层
    activation_layer: Callable[..., nn.Module] = nn.ReLU    # 激活函数层


# 多层感知机块（Transformer中的前馈网络）
class MLPBlock(nn.Sequential):
    """Transformer模型中的MLP块，包含线性变换、激活函数和Dropout"""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        # 第一层线性变换（升维）
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        # GELU激活函数（比ReLU更平滑的非线性函数）
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)  # 随机失活防止过拟合
        # 第二层线性变换（降维回原始维度）
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)  # 第二层Dropout

        # 权重初始化（Xavier均匀分布，适用于激活函数为非线性的场景）
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        # 偏置初始化（小正态分布，减少初始阶段的输出偏差）
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


# Transformer编码器块（包含注意力机制和MLP）
class EncoderBlock(nn.Module):
    """Transformer编码器的基础单元，包含自注意力和MLP子层"""

    def __init__(
        self,
        num_heads: int,            # 注意力头数
        hidden_dim: int,           # 隐藏层维度
        mlp_dim: int,              # MLP中间层维度
        dropout: float,            # Dropout概率
        attention_dropout: float,  # 注意力权重Dropout概率
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)  # 归一化层
    ):
        super().__init__()
        self.num_heads = num_heads

        # 自注意力子层
        self.ln_1 = norm_layer(hidden_dim)  # 层归一化
        # 多头自注意力（batch_first=True使输入格式为[batch, seq, dim]）
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # 输出Dropout

        # MLP子层
        self.ln_2 = norm_layer(hidden_dim)  # 层归一化
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)  # MLP块

    def forward(self, input: torch.Tensor):
        # 输入维度校验（应为[batch, seq_length, hidden_dim]）
        torch._assert(input.dim() == 3, f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}")
        x = self.ln_1(input)  # 层归一化
        # 自注意力计算（query/key/value相同，need_weights=False不返回注意力权重）
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)  # Dropout
        x = x + input  # 残差连接

        y = self.ln_2(x)  # 层归一化
        y = self.mlp(y)  # MLP计算
        return x + y  # 残差连接


# Transformer编码器（多层EncoderBlock的组合）
class Encoder(nn.Module):
    """Transformer模型的编码器，包含位置嵌入和多层编码器块"""

    def __init__(
        self,
        seq_length: int,           # 序列长度（补丁数+类令牌）
        num_layers: int,           # 编码器块层数
        num_heads: int,            # 注意力头数
        hidden_dim: int,           # 隐藏层维度
        mlp_dim: int,              # MLP中间层维度
        dropout: float,            # Dropout概率
        attention_dropout: float,  # 注意力权重Dropout概率
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)  # 归一化层
    ):
        super().__init__()
        # 位置嵌入（随机初始化的可学习参数，遵循BERT的初始化方式）
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)  # 输入Dropout

        # 构建多层编码器块
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer
            )
        self.layers = nn.Sequential(layers)  # 按顺序组合编码器块
        self.ln = norm_layer(hidden_dim)  # 最终层归一化

    def forward(self, input: torch.Tensor):
        # 输入维度校验
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        # 添加位置嵌入（广播到整个批次）
        input = input + self.pos_embedding
        # 经过所有编码器块和最终归一化
        return self.ln(self.layers(self.dropout(input)))


# 视觉Transformer主模型（Vision Transformer）
class VisionTransformer(nn.Module):
    """基于Transformer的视觉模型，实现图像分类任务"""

    def __init__(
        self,
        image_size: int,           # 输入图像尺寸（如224）
        patch_size: int,           # 图像补丁尺寸（如16）
        num_layers: int,           # 编码器层数
        num_heads: int,            # 注意力头数
        hidden_dim: int,           # 隐藏层维度
        mlp_dim: int,              # MLP中间层维度
        dropout: float = 0.0,      # Dropout概率
        attention_dropout: float = 0.0,  # 注意力Dropout概率
        num_classes: int = 1000,   # 分类类别数
        representation_size: Optional[int] = None,  # 特征表示维度（None表示直接分类）
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # 归一化层
        conv_stem_configs: Optional[List[ConvStemConfig]] = None  # 卷积茎配置（替代补丁划分）
    ):
        super().__init__()
        _log_api_usage_once(self)  # 记录API使用日志
        # 校验图像尺寸是否可被补丁尺寸整除
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        # 补丁划分模块（两种实现：传统卷积或多级卷积茎）
        if conv_stem_configs is not None:
            # 多级卷积茎（如ResNet风格的特征提取）
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    ConvNormActivation(
                        in_channels=prev_channels,
                        out_channels=config.out_channels,
                        kernel_size=config.kernel_size,
                        stride=config.stride,
                        norm_layer=config.norm_layer,
                        activation_layer=config.activation_layer,
                    ),
                )
                prev_channels = config.out_channels
            # 最后1x1卷积映射到隐藏维度
            seq_proj.add_module("conv_last", nn.Conv2d(prev_channels, hidden_dim, kernel_size=1))
            self.conv_proj: nn.Module = seq_proj
        else:
            # 传统补丁划分（直接通过卷积将图像划分为补丁）
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        # 计算补丁数量（图像尺寸/补丁尺寸的平方）
        num_patches = (image_size // patch_size) ** 2
        seq_length = num_patches  # 初始序列长度（仅补丁）

        # 类令牌（可学习参数，用于分类任务）
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1  # 序列长度+1（添加类令牌）

        # Transformer编码器
        self.encoder = Encoder(
            seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer
        )
        self.seq_length = seq_length  # 保存总序列长度

        # 分类头（两种模式：直接分类或带特征表示层）
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)  # 直接分类
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)  # 特征表示层
            heads_layers["act"] = nn.Tanh()  # Tanh激活
            heads_layers["head"] = nn.Linear(representation_size, num_classes)  # 分类层
        self.heads = nn.Sequential(heads_layers)

        # 权重初始化
        if isinstance(self.conv_proj, nn.Conv2d):
            # 传统补丁划分卷积初始化
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif hasattr(self.conv_proj, "conv_last") and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # 卷积茎最后一层初始化
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            # 特征表示层线性层初始化
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            # 分类层线性层初始化（权重和偏置全零，训练中学习）
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """处理输入图像，完成补丁划分和维度变换"""
        n, c, h, w = x.shape
        p = self.patch_size
        # 校验图像尺寸
        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p  # 垂直方向补丁数
        n_w = w // p  # 水平方向补丁数

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)：通过卷积划分补丁
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, n_h*n_w)：展平补丁
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, n_h*n_w) -> (n, n_h*n_w, hidden_dim)：调整维度顺序以适应Transformer
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        """前向传播过程，返回中间特征和分类结果"""
        out = {}  # 存储中间特征

        # 处理输入图像，得到补丁序列
        x = self._process_input(x)
        n = x.shape[0]  # 批次大小

        # 扩展类令牌到整个批次（广播机制）
        batch_class_token = self.class_token.expand(n, -1, -1)
        # 拼接类令牌和补丁序列（[batch, seq_length, hidden_dim]）
        x = torch.cat([batch_class_token, x], dim=1)

        # 经过Transformer编码器
        x = self.encoder(x)
        # 提取图像补丁特征（忽略类令牌）
        img_feature = x[:, 1:]
        H = W = int(self.image_size / self.patch_size)  # 补丁网格尺寸
        # 重塑为2D特征图（用于下游任务如检测分割）
        out['f4'] = img_feature.view(n, H, W, self.hidden_dim).permute(0, 3, 1, 2)

        # 提取类令牌特征（用于分类）
        x = x[:, 0]
        out['penultimate'] = x  # 分类前的特征表示

        # 经过分类头
        x = self.heads(x)
        out['logits'] = x  # 分类对数概率

        return out


# 通用Vision Transformer模型创建函数
def _vision_transformer(
    arch: str,                 # 模型架构名称
    patch_size: int,           # 补丁尺寸
    num_layers: int,           # 编码器层数
    num_heads: int,            # 注意力头数
    hidden_dim: int,           # 隐藏层维度
    mlp_dim: int,              # MLP中间层维度
    pretrained: bool,          # 是否加载预训练权重
    progress: bool,            # 是否显示下载进度
    **kwargs: Any              # 其他参数
) -> VisionTransformer:
    image_size = kwargs.pop("image_size", 224)  # 获取图像尺寸（默认224）

    # 创建VisionTransformer模型
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    # 加载预训练权重
    if pretrained:
        if arch not in model_urls:
            raise ValueError(f"No checkpoint is available for model type '{arch}'!")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)

    return model


# 以下是不同配置的ViT模型创建函数
def vit_b_16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    创建ViT-Base/16模型（隐藏维度768，补丁尺寸16）
    源自论文"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    """
    return _vision_transformer(
        arch="vit_b_16",
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )


def vit_b_32(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """创建ViT-Base/32模型（补丁尺寸32，其他同ViT-Base/16）"""
    return _vision_transformer(
        arch="vit_b_32",
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )


def vit_l_16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """创建ViT-Large/16模型（隐藏维度1024，补丁尺寸16，层数24）"""
    return _vision_transformer(
        arch="vit_l_16",
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )


def vit_l_32(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """创建ViT-Large/32模型（补丁尺寸32，其他同ViT-Large/16）"""
    return _vision_transformer(
        arch="vit_l_32",
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )


def interpolate_embeddings(
    image_size: int,                 # 新模型图像尺寸
    patch_size: int,                 # 新模型补丁尺寸
    model_state: "OrderedDict[str, torch.Tensor]",  # 预训练模型状态字典
    interpolation_mode: str = "bicubic",  # 插值模式
    reset_heads: bool = False        # 是否重置分类头
) -> "OrderedDict[str, torch.Tensor]":
    """
    插值位置嵌入以适应不同尺寸的图像（解决预训练模型与新模型尺寸不匹配问题）
    
    原理：将1D位置嵌入重塑为2D网格，通过插值调整尺寸，再重塑回1D
    """
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    # 计算新序列长度（补丁数+类令牌）
    new_seq_length = (image_size // patch_size) ** 2 + 1

    # 仅当序列长度不同时需要插值
    if new_seq_length != seq_length:
        # 分离类令牌和图像补丁嵌入
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)：转置维度
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        # 校验原位置嵌入是否为正方形网格
        torch._assert(seq_length_1d * seq_length_1d == seq_length, "seq_length is not a perfect square!")

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)：重塑为2D网格
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size  # 新网格尺寸

        # 2D插值（调整网格尺寸）
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)：重塑为1D
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)：转置维度
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        # 拼接类令牌和新图像补丁嵌入
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["encoder.pos_embedding"] = new_pos_embedding  # 更新位置嵌入

        if reset_heads:
            # 重置分类头（不加载预训练的分类层权重）
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state