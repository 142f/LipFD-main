# 导入 hashlib 库，用于计算文件的 SHA256 哈希值，以验证下载文件的完整性
import hashlib
# 导入 os 库，用于处理文件路径和目录操作
import os
# 导入 urllib 库，用于从 URL 下载文件
import urllib
# 导入 warnings 库，用于发出警告信息，例如当 PyTorch 版本过低时
import warnings
# 从 typing 库导入类型提示，增强代码可读性和健壮性
from typing import Any, Union, List

# 官方代码使用的是 pkg_resources，但 packaging 是一个更现代、更推荐的库来处理版本比较
# from pkg_resources import packaging
# 修改后，使用 packaging 库来解析和比较版本号
import packaging.version
# 导入 PyTorch 核心库
import torch
# 导入 PIL (Pillow) 库中的 Image 模块，用于图像处理
from PIL import Image
# 从 torchvision.transforms 导入一系列图像预处理操作
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# 导入 tqdm 库，用于在下载文件时显示一个美观的进度条
from tqdm import tqdm

# 从当前包（clip）中导入模型构建函数和分词器
from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    # 尝试从新版本的 torchvision.transforms 中导入 InterpolationMode
    # 这是处理图像缩放时插值方法的推荐方式
    from torchvision.transforms import InterpolationMode
    # 使用三次线性插值，这在 CLIP 中是默认的图像缩放方式
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    # 如果导入失败（通常是旧版本的 torchvision），则回退到使用 PIL.Image 中的常量
    BICUBIC = Image.BICUBIC


# 检查当前安装的 PyTorch 版本
if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    # 如果版本低于 1.7.1，发出警告，因为 CLIP 的某些功能（特别是 JIT）可能需要较新版本
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


# 定义此模块的公共 API，当使用 `from clip import *` 时，只有这些名称会被导入
__all__ = ["available_models", "load", "tokenize"]
# 初始化一个全局的分词器实例，用于后续的文本处理
_tokenizer = _Tokenizer()

# --------------------------------------------------------------------------------
# 预训练模型注册表
# --------------------------------------------------------------------------------
# 这是一个字典，存储了所有官方发布的 CLIP 模型名称及其对应的下载链接
# 键是模型名称（如 "RN50", "ViT-B/32"），值是模型的 Azure Blob Storage URL
# URL 的一部分（倒数第二个路径段）是模型的 SHA256 哈希值，用于后续校验
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    """
    一个带缓存和哈希校验的下载函数。
    :param url: 要下载的文件的 URL。
    :param root: 保存文件的根目录。
    :return: 下载文件的本地路径。
    """
    # 确保指定的根目录存在，如果不存在则创建
    os.makedirs(root, exist_ok=True)
    # 从 URL 中提取文件名
    filename = os.path.basename(url)

    # 从 URL 中提取预期的 SHA256 哈希值
    expected_sha256 = url.split("/")[-2]
    # 拼接出完整的文件下载目标路径
    download_target = os.path.join(root, filename)

    # 如果目标路径已存在，但它是一个目录而不是文件，则抛出错误
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    # 如果文件已经存在
    if os.path.isfile(download_target):
        # 计算已存在文件的 SHA256 哈希值
        # 如果哈希值与预期的匹配，说明文件有效，直接返回路径，避免重复下载
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            # 如果哈希值不匹配，发出警告并重新下载
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    # 使用 urllib 打开 URL，并以二进制写入模式打开本地文件
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        # 使用 tqdm 创建一个进度条，以可视化下载过程
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                # 以 8192 字节为块进行读取
                buffer = source.read(8192)
                # 如果读取到空内容，说明下载完成，跳出循环
                if not buffer:
                    break
                
                # 将读取到的数据块写入本地文件
                output.write(buffer)
                # 更新进度条
                loop.update(len(buffer))

    # 下载完成后，再次校验文件的哈希值，确保下载过程中没有发生错误
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    # 返回下载好的文件路径
    return download_target


def _convert_image_to_rgb(image):
    """
    一个辅助函数，确保输入的 PIL.Image 对象是 RGB 格式。
    这对于处理灰度图（'L'）或带 alpha 通道的图（'RGBA'）是必要的。
    """
    return image.convert("RGB")


def _transform(n_px: int):
    """
    创建一个 torchvision 的 transform 流水线。
    这是 CLIP 模型标准的图像预处理流程。
    :param n_px: 图像的目标分辨率（正方形）。
    :return: 一个 Compose 对象，包含了所有预处理步骤。
    """
    return Compose([
        # 1. 将图像最短边缩放到 n_px，保持宽高比，使用三次线性插值
        Resize(n_px, interpolation=BICUBIC),
        # 2. 从图像中心裁剪出 n_px x n_px 大小的区域
        CenterCrop(n_px),
        # 3. 确保图像是 RGB 格式
        _convert_image_to_rgb,
        # 4. 将 PIL.Image 对象转换为 PyTorch Tensor，并将像素值从 [0, 255] 缩放到 [0.0, 1.0]
        ToTensor(),
        # 5. 使用 CLIP 预训练时使用的特定均值和标准差对图像进行标准化
        #    这些值对于模型获得良好性能至关重要
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """返回所有可用的 CLIP 模型名称列表"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """
    加载一个 CLIP 模型。

    Parameters
    ----------
    name : str
        一个由 `clip.available_models()` 列出的模型名称，或者是一个包含 state_dict 的模型检查点文件的本地路径。

    device : Union[str, torch.device]
        加载模型的目标设备（例如 "cpu", "cuda", "cuda:1"）。

    jit : bool
        是否加载 JIT (Just-In-Time) 编译的优化模型。
        - True: 加载 JIT 模型，运行速度快，但不易于修改和调试。
        - False: (默认)加载非 JIT 模型，是一个标准的 nn.Module，更易于 "hack"（例如查看中间层、修改结构）。

    download_root: str
        下载模型文件的路径；默认使用 "~/.cache/clip"。

    Returns
    -------
    model : torch.nn.Module
        加载好的 CLIP 模型。

    preprocess : Callable[[PIL.Image], torch.Tensor]
        一个 torchvision transform 函数，用于将 PIL 图像转换为模型可以接受的 tensor 输入。
    """
    # 步骤 1: 确定模型文件的路径
    if name in _MODELS:
        # 如果 name 是一个官方模型名称，则从 URL 下载模型
        # 如果 download_root 未指定，则使用默认的缓存路径 `~/.cache/clip`
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        # 如果 name 是一个存在的本地文件路径，则直接使用它
        model_path = name
    else:
        # 如果 name 既不是官方模型也不是本地文件，则抛出错误
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # 步骤 2: 加载模型文件
    with open(model_path, 'rb') as opened_file:
        try:
            # 优先尝试以 JIT 模式加载
            # `torch.jit.load` 可以加载一个已保存的 JIT ScriptModule
            # 如果 jit=True，直接加载到目标 device；否则先加载到 CPU
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # 如果加载 JIT 失败（说明文件不是 JIT 归档，而是普通的 state_dict）
            if jit:
                # 如果用户明确要求 JIT，但文件不是，发出警告并回退到非 JIT 模式
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            # 使用 torch.load 加载 state_dict（权重）到 CPU
            # 先加载到 CPU 是一个好习惯，可以避免 GPU 内存问题
            state_dict = torch.load(opened_file, map_location="cpu")

    # 步骤 3: 构建和配置模型
    if not jit:
        # 如果是非 JIT 模式
        # 使用 build_model 函数和 state_dict 来构建模型结构并加载权重
        # state_dict() or model.state_dict()：如果之前加载了state_dict则用它，否则从JIT模型中提取
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            # 如果在 CPU 上运行，确保模型是 float32 类型
            model.float()
        # 返回构建好的模型和对应的图像预处理器
        return model, _transform(model.visual.input_resolution)

    # --- 以下是针对 JIT 模型的特殊处理 ---
    # JIT 模型会将设备和数据类型硬编码在计算图中，如果加载到不同设备或需要不同类型，需要进行 "patch"
    
    # 步骤 4 (JIT-only): 修正设备信息
    # 创建一个虚拟的 trace，目的是获取一个指向目标 device 的 JIT graph node
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        # 一个递归函数，用于遍历 JIT 模块并替换设备节点
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []
        if hasattr(module, "forward1"): # 有些模块可能有多个 graph
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                # 找到所有硬编码为 'cuda' 的设备常量节点
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    # 将其替换为我们之前创建的目标设备节点
                    node.copyAttributes(device_node)

    # 对模型及其子模块应用设备修正
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # 步骤 5 (JIT-only): 在 CPU 上修正数据类型为 float32
    if str(device) == "cpu":
        # 类似地，创建一个虚拟 trace 来获取 float32 数据类型的 JIT 节点
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            # 递归函数，遍历并替换数据类型节点
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"): # 'aten::to' 是 PyTorch 中类型转换的操作
                    inputs = list(node.inputs())
                    # 找到将数据类型转换为 half (float16) 的节点（其值为 5）
                    for i in [1, 2]:  # dtype 参数可能是第2或第3个输入
                        if inputs[i].node()["value"] == 5:
                            # 将其替换为 float32 节点
                            inputs[i].node().copyAttributes(float_node)

        # 应用数据类型修正
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        # 确保模型本身也设置为 float32
        model.float()

    # 返回 JIT 模型和对应的图像预处理器
    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    将输入的字符串（或字符串列表）转换为 token ID 序列。

    Parameters
    ----------
    texts : Union[str, List[str]]
        一个或多个待分词的输入字符串。

    context_length : int
        上下文长度。所有 CLIP 模型都使用固定的 77 作为上下文长度。

    truncate: bool
        如果文本编码后超过 `context_length`，是否进行截断。
        - True: 截断文本，并确保最后一个 token 是 EOT (end-of-text) token。
        - False: (默认) 抛出 RuntimeError。

    Returns
    -------
    一个二维 tensor，包含了 tokenization 的结果。
    形状为 [输入的字符串数量, context_length]。
    在 PyTorch < 1.8.0 版本中返回 LongTensor，因为旧版本的 `index_select` 操作要求索引为 long 类型。
    """
    # 如果输入是单个字符串，先将其放入列表中，以统一处理
    if isinstance(texts, str):
        texts = [texts]

    # 获取特殊的 "start of text" 和 "end of text" token 的 ID
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    
    # 对每个文本进行编码，并在两端加上 SOT 和 EOT token
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    # 根据 PyTorch 版本选择 tensor 的数据类型，以保证兼容性
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    # 遍历每个编码后的 token 序列，并填充到结果 tensor 中
    for i, tokens in enumerate(all_tokens):
        # 检查 token 序列长度是否超过上下文长度
        if len(tokens) > context_length:
            if truncate:
                # 如果允许截断，则只取前 context_length 个 token
                tokens = tokens[:context_length]
                # 强制将最后一个 token 设置为 EOT token
                tokens[-1] = eot_token
            else:
                # 如果不允许截断，则抛出错误
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        
        # 将处理好的 token 序列填充到结果 tensor 的对应行
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result