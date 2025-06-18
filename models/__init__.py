from .clip_models import CLIPModel # 从当前包导入 CLIPModel 类
from .LipFD import LipFD, RALoss # 从当前包导入 LipFD 模型和 RALoss 损失函数

VALID_NAMES = [ # 定义一个列表，包含所有有效的模型名称
    "CLIP:ViT-B/32", # CLIP ViT-B/32 模型
    "CLIP:ViT-B/16", # CLIP ViT-B/16 模型
    "CLIP:ViT-L/14", # CLIP ViT-L/14 模型
]


def get_model(name):
    """
    根据给定的模型名称获取对应的模型实例。

    Args:
        name (str): 模型的名称，必须是 VALID_NAMES 列表中的一个。

    Returns:
        CLIPModel: 对应的 CLIPModel 实例。

    Raises:
        AssertionError: 如果模型名称不在 VALID_NAMES 中或不以 "CLIP:" 开头。
    """
    assert name in VALID_NAMES, f"Model name {name} not in valid names: {VALID_NAMES}" # 确保模型名称有效
    if name.startswith("CLIP:"):
        return CLIPModel(name[5:]) # 如果是 CLIP 模型，则创建 CLIPModel 实例
    else:
        assert False, "Unsupported model type" # 如果不是 CLIP 模型，则抛出断言错误


def build_model(transformer_name):
    """
    根据给定的 Transformer 模型名称构建 LipFD 模型实例。

    Args:
        transformer_name (str): Transformer 模型的名称，必须是 VALID_NAMES 列表中的一个。

    Returns:
        LipFD: 对应的 LipFD 模型实例。

    Raises:
        AssertionError: 如果 Transformer 模型名称不在 VALID_NAMES 中或不以 "CLIP:" 开头。
    """
    assert transformer_name in VALID_NAMES, f"Transformer name {transformer_name} not in valid names: {VALID_NAMES}" # 确保 Transformer 模型名称有效
    if transformer_name.startswith("CLIP:"):
        return LipFD(transformer_name[5:]) # 如果是 CLIP Transformer 模型，则创建 LipFD 实例
    else:
        assert False, "Unsupported transformer type" # 如果不是 CLIP Transformer 模型，则抛出断言错误


def get_loss():
    """
    获取区域感知损失 (RALoss) 实例。

    Returns:
        RALoss: RALoss 实例。
    """
    return RALoss() # 返回 RALoss 实例
