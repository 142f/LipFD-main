import os
import argparse
import torch


class BaseOptions:
    """
    基础选项类，用于定义和解析通用的命令行参数。
    """
    def __init__(self):
        """
        初始化 BaseOptions 类。
        设置 initialized 标志为 False。
        """
        self.initialized = False

    def initialize(self, parser):
        """
        初始化解析器，添加通用的命令行参数。

        参数:
            parser: argparse.ArgumentParser 对象。

        返回:
            argparse.ArgumentParser: 配置好参数的解析器对象。
        """
        parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14", help="see models/__init__.py")
        parser.add_argument("--fix_backbone", default=False)
        parser.add_argument("--fix_encoder", default=True)

        parser.add_argument("--real_list_path", default="./datasets/val/0_real")
        parser.add_argument("--fake_list_path", default="./datasets/val/1_fake")
        parser.add_argument("--data_label", default="train", help="label to decide whether train or validation dataset",)

        parser.add_argument( "--batch_size", type=int, default=10, help="input batch size")
        parser.add_argument("--gpu_ids", type=str, default="1", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",)
        parser.add_argument("--name", type=str, default="experiment_name", help="name of the experiment. It decides where to store samples and models",)
        parser.add_argument("--num_threads", default=0, type=int, help="# threads for loading data")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here",)
        parser.add_argument("--serial_batches",action="store_true",help="if true, takes images in order to make batches, otherwise takes them randomly",)
        self.initialized = True
        return parser

    def gather_options(self):
        """
        收集命令行选项。
        如果解析器未初始化，则先进行初始化。
        解析已知参数并返回解析后的参数。

        返回:
            argparse.Namespace: 包含所有解析后参数的命名空间对象。
        """
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        """
        打印所有选项及其默认值（如果已更改）。
        并将选项保存到磁盘上的 opt.txt 文件中。

        参数:
            opt: argparse.Namespace 对象，包含所有解析后的参数。
        """
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        # util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self, print_options=True):
        """
        解析命令行参数，并进行一些后处理。

        参数:
            print_options (bool): 是否打印解析后的选项。

        返回:
            argparse.Namespace: 包含所有解析后参数的命名空间对象。
        """
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        # 检查是否存在 suffix 参数，如果存在且不为空，则将其格式化并附加到 opt.name
        if hasattr(opt, 'suffix') and opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        # 将 gpu_ids 字符串按逗号分割，并转换为整数列表
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # 如果 gpu_ids 列表不为空，则设置 CUDA 设备
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # additional
        # opt.classes = opt.classes.split(',')
        # 将 rz_interp 字符串按逗号分割
        if hasattr(opt, 'rz_interp') and isinstance(opt.rz_interp, str):
            opt.rz_interp = opt.rz_interp.split(",")
        # 将 blur_sig 字符串按逗号分割，并转换为浮点数列表
        if hasattr(opt, 'blur_sig') and isinstance(opt.blur_sig, str):
            opt.blur_sig = [float(s) for s in opt.blur_sig.split(",")]
        # 将 jpg_method 字符串按逗号分割
        if hasattr(opt, 'jpg_method') and isinstance(opt.jpg_method, str):
            opt.jpg_method = opt.jpg_method.split(",")
        # 将 jpg_qual 字符串按逗号分割，并转换为整数列表
        # 如果 jpg_qual 列表长度为 2，则生成一个从第一个元素到第二个元素的整数范围列表
        # 如果 jpg_qual 列表长度大于 2，则抛出 ValueError
        if hasattr(opt, 'jpg_qual') and isinstance(opt.jpg_qual, str):
            opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(",")]
            if len(opt.jpg_qual) == 2:
                opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
            elif len(opt.jpg_qual) > 2:
                raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt
