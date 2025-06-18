import os
import torch
import torch.nn as nn
from models import build_model, get_loss


class Trainer(nn.Module):
    """
    Trainer 类，继承自 nn.Module。
    用于处理模型的训练过程，包括初始化、前向传播、损失计算和参数优化。
    """
    def __init__(self, opt):
        """
        初始化 Trainer。

        参数:
            opt: 包含训练配置选项的对象。
        """
        super().__init__() # nn.Module的初始化
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )
        # self.opt = opt # 重复赋值，可以移除
        self.model = build_model(opt.arch)

        self.step_bias = (
            0
            if not opt.fine_tune
            else int(opt.pretrained_model.split("_")[-1].split(".")[0]) + 1
        )
        if opt.fine_tune:
            state_dict = torch.load(opt.pretrained_model, map_location="cpu")
            self.model.load_state_dict(state_dict["model"])
            self.total_steps = state_dict["total_steps"]
            print(f"Model loaded @ {opt.pretrained_model.split('/')[-1]}")

        # 根据 opt.fix_encoder 决定是否固定编码器的参数
        # 如果 opt.fix_encoder 为 True，则将编码器相关的参数的 requires_grad 设置为 False
        # 注意：这里的 'params' 变量在 if opt.fix_encoder 块内部被赋值，
        # 如果 opt.fix_encoder 为 False，则 'params' 将不会被定义，
        # 这可能导致后续优化器初始化时使用未定义的 'params'。
        # 应该确保 'params' 在所有情况下都被正确初始化，例如默认为 self.model.parameters()
        params = self.model.parameters() # 默认情况下优化所有参数
        if opt.fix_encoder:
            fixed_params = []
            for name, p in self.model.named_parameters():
                if name.split(".")[0] in ["encoder"]:
                    p.requires_grad = False
                else:
                    # 这里原代码将非 encoder 的参数也设置了 p.requires_grad = False，
                    # 这可能不是预期的行为，通常 fine-tune 时会冻结一部分，训练另一部分。
                    # 如果意图是只训练模型的特定部分（非encoder），则这里逻辑正确。
                    # 如果意图是训练除encoder外的所有其他部分，则这里也应为 p.requires_grad = True
                    # 或者，如果意图是固定encoder，训练其他所有，那么这里应该是 p.requires_grad = True
                    # 假设意图是固定 encoder，训练其他所有参数
                    p.requires_grad = True 
                if p.requires_grad:
                    fixed_params.append(p)
            params = iter(fixed_params) # 确保 params 是一个可迭代对象

        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params, # 使用处理后的 params
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay # 使用处理后的 params
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss()

        self.model.to(self.device) # 将模型移动到指定设备

    def adjust_learning_rate(self, min_lr=1e-8):
        """
        调整学习率。
        将优化器中每个参数组的学习率除以 10，但不低于 min_lr。

        参数:
            min_lr (float): 最小学习率。

        返回:
            bool: 如果学习率成功调整则返回 True，否则返回 False。
        """
        for param_group in self.optimizer.param_groups:
            if param_group["lr"] < min_lr:
                return False
            param_group["lr"] /= 10.0
        return True

    def set_input(self, input_data):
        """
        设置模型的输入数据。

        参数:
            input_data (tuple): 包含输入图像、裁剪图像和标签的元组。
        """
        self.input = input_data[0].to(self.device)
        self.crops = [[t.to(self.device) for t in sublist] for sublist in input_data[1]]
        self.label = input_data[2].to(self.device).float()

    def forward(self):
        """
        执行模型的前向传播。
        计算模型的输出和损失。
        """
        self.get_features()
        self.output, self.weights_max, self.weights_org = self.model.forward(
            self.crops, self.features
        )
        self.output = self.output.view(-1)
        self.loss = self.criterion(
            self.weights_max, self.weights_org
        ) + self.criterion1(self.output, self.label)

    def get_loss(self):
        """
        获取当前计算的损失值。

        返回:
            float: 损失值。
        """
        loss_val = self.loss.item() # 使用 .item() 获取标量值
        return loss_val

    def optimize_parameters(self):
        """
        执行参数优化步骤。
        包括梯度清零、反向传播和优化器更新。
        """
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def get_features(self):
        """
        从输入图像中提取特征。
        """
        self.features = self.model.get_features(self.input).to(
            self.device
        )  # shape: (batch_size

    def eval(self):
        """
        将模型设置为评估模式。
        """
        self.model.eval()

    def train(self): # 添加 train 方法以将模型设置回训练模式
        """
        将模型设置为训练模式。
        """
        self.model.train()

    def test(self):
        """
        在不计算梯度的情况下执行模型的前向传播（用于测试）。
        """
        with torch.no_grad():
            self.forward()

    def save_networks(self, save_filename):
        """
        保存模型和优化器的状态字典。

        参数:
            save_filename (str): 保存文件的名称。
        """
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        torch.save(state_dict, save_path)
