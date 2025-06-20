from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """
    训练选项类，继承自 BaseOptions。
    用于定义和解析训练阶段特有的命令行参数。
    """
    def initialize(self, parser):
        """
        初始化解析器，添加训练阶段特有的命令行参数。

        参数:
            parser: argparse.ArgumentParser 对象。

        返回:
            argparse.ArgumentParser: 配置好参数的解析器对象。
        """
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--loss_freq', type=int, default=100, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--epoch', type=int, default=100, help='total epoches')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=2e-9, help='initial learning rate for adam')
        parser.add_argument('--pretrained_model', type=str, default='./checkpoints/experiment_name/model_epoch_29.pth', help='model will fine tune on it if fine-tune is True')
        parser.add_argument('--fine-tune', type=bool, default=True)
        self.isTrain = True

        return parser
