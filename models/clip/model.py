from collections import OrderedDict # 导入 OrderedDict，用于保持字典中元素的插入顺序
from typing import Tuple, Union # 导入 Tuple 和 Union，用于类型提示

import numpy as np # 导入 NumPy 库，用于数值计算
import torch # 导入 PyTorch 库，用于构建和训练神经网络
import torch.nn.functional as F # 导入 PyTorch 的函数式 API，包含各种操作
from torch import nn # 导入 torch.nn 模块，包含了构建神经网络所需的所有层和函数


class Bottleneck(nn.Module):
    """
    ResNet 中的 Bottleneck 模块。
    """
    expansion = 4 # 瓶颈块的输出通道是输入通道的 4 倍

    def __init__(self, inplanes, planes, stride=1):
        """
        初始化 Bottleneck 模块。

        Args:
            inplanes (int): 输入特征图的通道数。
            planes (int): 中间层的通道数（输出通道的 1/4）。
            stride (int): 步长，用于控制下采样，默认为 1。
        """
        super().__init__()

        # 所有的卷积层步长都为 1。当 stride > 1 时，在第二个卷积后执行平均池化。
        # 第一个 1x1 卷积，用于降维
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二个 3x3 卷积，用于特征提取
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # 如果步长大于 1，则进行平均池化，否则使用恒等映射
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 第三个 1x1 卷积，用于升维
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None # 下采样层，用于处理维度不匹配的情况
        self.stride = stride

        # 如果需要下采样（步长大于 1 或输入输出通道不匹配），则创建下采样层
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # 下采样层在卷积前添加一个平均池化，后续卷积步长为 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)), # 平均池化
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), # 1x1 卷积
                ("1", nn.BatchNorm2d(planes * self.expansion)) # 批归一化
            ]))

    def forward(self, x: torch.Tensor):
        """
        Bottleneck 模块的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        identity = x # 保存原始输入，用于残差连接

        # 1x1 卷积 -> BN -> ReLU
        out = self.relu1(self.bn1(self.conv1(x)))
        # 3x3 卷积 -> BN -> ReLU
        out = self.relu2(self.bn2(self.conv2(out)))
        # 平均池化（如果 stride > 1）
        out = self.avgpool(out)
        # 1x1 卷积 -> BN
        out = self.bn3(self.conv3(out))

        # 如果存在下采样层，则对原始输入进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        # 最终 ReLU 激活
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    """
    用于 CLIP ResNet 视觉编码器的注意力池化层。
    它将空间特征图转换为单个嵌入向量。
    """
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        """
        初始化 AttentionPool2d。

        Args:
            spacial_dim (int): 输入特征图的空间维度（例如，如果输入是 7x7，则 spacial_dim=7）。
            embed_dim (int): 输入和输出嵌入的维度。
            num_heads (int): 多头注意力的头数。
            output_dim (int, optional): 最终输出的维度。如果为 None，则默认为 embed_dim。
        """
        super().__init__()
        # 可学习的位置嵌入，包含一个用于类别 token 的额外位置
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # 查询、键、值和输出投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        """
        AttentionPool2d 的前向传播。

        Args:
            x (torch.Tensor): 输入特征图，形状为 (N, C, H, W)。

        Returns:
            torch.Tensor: 池化后的输出嵌入，形状为 (N, output_dim)。
        """
        # 将输入从 NCHW 展平并转置为 (HW)NC 格式
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        # 在序列的开头添加一个平均池化后的特征作为类别 token
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # 添加位置嵌入
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # 执行多头注意力
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x, # 查询只使用类别 token
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        # 返回类别 token 的输出，并移除维度为 1 的维度
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    一个类似于 torchvision 中 ResNet 的 ResNet 类，但包含以下修改：
    - 现在有 3 个 "stem" 卷积层，而不是 1 个，并且使用平均池化代替最大池化。
    - 执行抗锯齿步幅卷积，其中在步长大于 1 的卷积前添加平均池化。
    - 最终的池化层是 QKV 注意力池化而不是平均池化。
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        """
        初始化 ModifiedResNet。

        Args:
            layers (list): 包含每个阶段 Bottleneck 块数量的列表。
            output_dim (int): 模型的输出维度。
            heads (int): 注意力池化中的头数。
            input_resolution (int): 输入图像的分辨率，默认为 224。
            width (int): 模型的宽度（即第一个卷积层的输出通道数），默认为 64。
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # 3 层 stem 结构
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # 残差层
        self._inplanes = width  # 这是一个在构建过程中使用的可变变量
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # ResNet 特征维度
        # 最终的注意力池化层
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        """
        构建 ResNet 的一个阶段（层）。

        Args:
            planes (int): 当前阶段的输出通道数。
            blocks (int): 当前阶段的 Bottleneck 块数量。
            stride (int): 第一个 Bottleneck 块的步长，默认为 1。

        Returns:
            nn.Sequential: 包含当前阶段所有 Bottleneck 块的序列模块。
        """
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        ModifiedResNet 的前向传播。

        Args:
            x (torch.Tensor): 输入图像张量。

        Returns:
            torch.Tensor: 模型的输出特征。
        """
        def stem(x):
            # 3 层 stem 的前向传播
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        # 确保输入张量的数据类型与第一个卷积层的权重数据类型一致
        x = x.type(self.conv1.weight.dtype)
        # 通过 stem 结构
        x = stem(x)
        # 通过四个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 通过注意力池化层
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """
    LayerNorm 的子类，用于处理 fp16 浮点数。
    继承自 nn.LayerNorm，并重写了 forward 方法以确保在计算时使用 float32 精度，
    然后将结果转换回原始数据类型，以避免在低精度计算中可能出现的数值问题。
    """

    def forward(self, x: torch.Tensor):
        """
        LayerNorm 的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 经过 Layer Normalization 后的张量。
        """
        orig_type = x.dtype
        # 将输入张量转换为 float32 进行计算，以确保精度
        ret = super().forward(x.type(torch.float32))
        # 将结果转换回原始数据类型
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    QuickGELU 激活函数。
    这是一个近似 GELU 激活函数的实现，计算速度更快。
    """
    def forward(self, x: torch.Tensor):
        """
        QuickGELU 的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 经过 QuickGELU 激活函数处理后的张量。
        """
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    残差注意力块。
    该模块结合了多头自注意力和前馈神经网络，并使用了残差连接和层归一化。
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        """
        初始化残差注意力块。

        Args:
            d_model (int): 模型的维度（特征维度）。
            n_head (int): 多头注意力的头数。
            attn_mask (torch.Tensor, optional): 注意力掩码，默认为 None。
        """
        super().__init__()

        # 多头自注意力层
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # 第一个层归一化
        self.ln_1 = LayerNorm(d_model)
        # 前馈神经网络（MLP），包含两个线性层和一个 QuickGELU 激活函数
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        # 第二个层归一化
        self.ln_2 = LayerNorm(d_model)
        # 注意力掩码
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        """
        执行注意力计算。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 经过注意力计算后的张量。
        """
        # 如果存在注意力掩码，将其移动到与输入张量相同的数据类型和设备
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 执行多头自注意力计算
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """
        残差注意力块的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 经过残差注意力块处理后的张量。
        """
        # 第一个残差连接和层归一化，然后是注意力计算
        x = x + self.attention(self.ln_1(x))
        # 第二个残差连接和层归一化，然后是 MLP 计算
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Transformer 模型。
    由多个 ResidualAttentionBlock 堆叠而成。
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        """
        初始化 Transformer 模型。

        Args:
            width (int): 模型的维度（特征维度）。
            layers (int): ResidualAttentionBlock 的层数。
            heads (int): 多头注意力的头数。
            attn_mask (torch.Tensor, optional): 注意力掩码，默认为 None。
        """
        super().__init__()
        self.width = width
        self.layers = layers
        # 堆叠 ResidualAttentionBlock
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        """
        Transformer 的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            tuple: 包含中间层输出和最终输出的元组。
                   out (dict): 包含每个 ResidualAttentionBlock 输出的字典，键为 'layerX'。
                   x (torch.Tensor): 最终输出张量。
        """
        out = {}
        # 遍历每个 ResidualAttentionBlock
        for idx, layer in enumerate(self.resblocks.children()):
            x = layer(x)
            # 存储每个层的输出，选择分类 token 的特征
            out['layer'+str(idx)] = x[0] # shape:LND. choose cls token feature   
        return out, x 

        # return self.resblocks(x)  # This is the original code 


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 模型。
    将图像分割成小块（patch），然后将这些 patch 线性嵌入并添加位置编码，
    最后通过 Transformer 编码器进行处理。
    """
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        """
        初始化 Vision Transformer。

        Args:
            input_resolution (int): 输入图像的分辨率（边长）。
            patch_size (int): 图像块（patch）的大小。
            width (int): 模型的维度（特征维度）。
            layers (int): Transformer 编码器的层数。
            heads (int): Transformer 中多头注意力的头数。
            output_dim (int): 输出特征的维度。
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # 卷积层用于将图像块嵌入到指定维度
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # 初始化分类 token 和位置编码
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # 位置编码，包括分类 token 的位置和所有图像块的位置
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        # Transformer 编码器前的层归一化
        self.ln_pre = LayerNorm(width)

        # Transformer 编码器
        self.transformer = Transformer(width, layers, heads)

        # Transformer 编码器后的层归一化
        self.ln_post = LayerNorm(width)
        # 投影层，用于将 Transformer 的输出投影到最终的输出维度
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))



    def forward(self, x: torch.Tensor):
        """
        Vision Transformer 的前向传播。

        Args:
            x (torch.Tensor): 输入图像张量。

        Returns:
            torch.Tensor: 最终的 CLIP 特征。
        """
        # 1. 卷积层处理输入图像，将其分割成 patch 并嵌入
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # 2. 将 patch 展平
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # 3. 调整维度顺序，使其符合 Transformer 输入要求
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # 4. 添加分类 token，并与 patch 嵌入拼接
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # 5. 添加位置编码
        x = x + self.positional_embedding.to(x.dtype)
        # 6. 预层归一化
        x = self.ln_pre(x)

        # 7. 调整维度顺序，使其符合 Transformer 输入要求 (NLD -> LND)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # 8. 通过 Transformer 编码器
        out, x = self.transformer(x)
        # 9. 调整维度顺序 (LND -> NLD)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 10. 后层归一化，并选择分类 token 的特征
        x = self.ln_post(x[:, 0, :])


        out['before_projection'] = x  

        # 11. 如果存在投影层，则进行投影
        if self.proj is not None:
            x = x @ self.proj
        out['after_projection'] = x 

        # Return both intermediate features and final clip feature 
        # return out
        
        # This only returns CLIP features 
        return x 


class CLIP(nn.Module):
    """
    CLIP (Contrastive Language-Image Pre-training) 模型。
    该模型结合了视觉编码器和文本编码器，通过对比学习将图像和文本映射到同一个嵌入空间。
    """
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        """
        初始化 CLIP 模型。

        Args:
            embed_dim (int): 嵌入维度。
            image_resolution (int): 图像分辨率。
            vision_layers (Union[Tuple[int, int, int, int], int]): 视觉编码器的层数。
                                                                    如果是元组，表示使用 ModifiedResNet；
                                                                    如果是整数，表示使用 VisionTransformer。
            vision_width (int): 视觉编码器的宽度（特征维度）。
            vision_patch_size (int): 视觉编码器中图像块的大小（仅 VisionTransformer 使用）。
            context_length (int): 文本编码器中上下文的最大长度。
            vocab_size (int): 文本编码器中词汇表的大小。
            transformer_width (int): 文本 Transformer 的宽度（特征维度）。
            transformer_heads (int): 文本 Transformer 中多头注意力的头数。
            transformer_layers (int): 文本 Transformer 的层数。
        """
        super().__init__()

        self.context_length = context_length

        # 根据 vision_layers 的类型选择视觉编码器
        if isinstance(vision_layers, (tuple, list)):
            # 如果是元组或列表，使用 ModifiedResNet
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            # 否则，使用 VisionTransformer
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # 文本 Transformer 编码器
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        out = {}
        for idx, layer in enumerate(self.resblocks.children()):
            x = layer(x)
            out['layer'+str(idx)] = x[0] # shape:LND. choose cls token feature   
        return out, x 

        # return self.resblocks(x)  # This is the original code 


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))



    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        out, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])


        out['before_projection'] = x  

        if self.proj is not None:
            x = x @ self.proj
        out['after_projection'] = x 

        # Return both intermediate features and final clip feature 
        # return out
        
        # This only returns CLIP features 
        return x 


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        # 文本 token 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # 文本位置编码
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        # 最终的层归一化
        self.ln_final = LayerNorm(transformer_width)

        # 文本特征投影层
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # 对数尺度参数，用于控制图像和文本特征之间的相似度计算
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 初始化模型参数
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        初始化模型参数。
        对 token 嵌入、位置编码、视觉编码器和文本 Transformer 的权重进行初始化。
        """
        # 初始化 token 嵌入权重
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        # 初始化位置编码
        nn.init.normal_(self.positional_embedding, std=0.01)

        # 如果视觉编码器是 ModifiedResNet，则初始化其注意力池化层和残差块的权重
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # 初始化 Transformer 块的权重
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # 初始化文本投影层的权重
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """
        构建注意力掩码。
        用于文本 Transformer，确保在计算注意力时不会看到未来的 token。
        """
        # 惰性创建因果注意力掩码，视觉 token 之间是完全注意力
        # PyTorch 使用加性注意力掩码；用 -inf 填充
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 将下三角部分置零
        return mask

    @property
    def dtype(self):
        """
        返回模型的数据类型。
        """
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        """
        编码图像。

        Args:
            image (torch.Tensor): 输入图像张量。

        Returns:
            torch.Tensor: 编码后的图像特征。
        """
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        """
        编码文本。

        Args:
            text (torch.Tensor): 输入文本 token 张量。

        Returns:
            torch.Tensor: 编码后的文本特征。
        """
        # 文本 token 嵌入
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        # 添加位置编码
        x = x + self.positional_embedding.type(self.dtype)
        # 调整维度顺序 (NLD -> LND)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # 通过 Transformer 编码器
        x = self.transformer(x)[1] # Only get the last layer output
        # 调整维度顺序 (LND -> NLD)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # 最终层归一化
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # 从 eot (end of text) 嵌入中获取特征 (eot_token 是每个序列中最高的数字)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        """
        CLIP 模型的前向传播。

        Args:
            image (torch.Tensor): 输入图像张量。
            text (torch.Tensor): 输入文本 token 张量。

        Returns:
            tuple: 包含图像-文本对数和文本-图像对数的元组。
                   logits_per_image (torch.Tensor): 图像到文本的相似度对数。
                   logits_per_text (torch.Tensor): 文本到图像的相似度对数。
        """
        # 编码图像和文本
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # 归一化特征
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算余弦相似度作为对数
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """
    将适用模型参数转换为 fp16 浮点数。
    """

    def _convert_weights_to_fp16(l):
        """
        递归函数，用于将模块中的权重转换为 fp16。
        """
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
