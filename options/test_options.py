from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """
    测试选项类，继承自 BaseOptions。
    用于定义和解析测试阶段特有的命令行参数。
    """
    def initialize(self, parser):
        """
        初始化解析器，添加测试阶段特有的命令行参数。

        参数:
            parser: argparse.ArgumentParser 对象。

        返回:
            argparse.ArgumentParser: 配置好参数的解析器对象。
        """
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        self.isTrain = False
        return parser
