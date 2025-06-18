from validate import validate  # 导入验证函数
from data import create_dataloader  # 导入数据加载器创建函数
from trainer.trainer import Trainer  # 导入训练器类
from options.train_options import TrainOptions  # 导入训练选项类


def get_val_opt():
    """
    创建验证数据集的选项配置
    
    返回:
        val_opt: 用于验证的TrainOptions实例，已设置为验证模式
    """
    val_opt = TrainOptions().parse(print_options=False)  # 创建训练选项实例并解析命令行参数，不打印选项
    val_opt.isTrain = False  # 设置为非训练模式
    val_opt.data_label = "val"  # 设置数据标签为验证集
    val_opt.real_list_path = "./datasets/val/0_real"  # 设置真实样本路径
    val_opt.fake_list_path = "./datasets/val/1_fake"  # 设置伪造样本路径
    return val_opt


if __name__ == "__main__":
    opt = TrainOptions().parse()  # 解析命令行参数，创建训练选项
    val_opt = get_val_opt()  # 获取验证选项
    model = Trainer(opt)  # 使用训练选项初始化训练器

    # 创建训练数据加载器
    data_loader = create_dataloader(opt)
    # 创建验证数据加载器
    val_loader = create_dataloader(val_opt)

    print("Length of data loader: %d" % (len(data_loader)))  # 打印训练数据加载器的长度
    print("Length of val  loader: %d" % (len(val_loader)))  # 打印验证数据加载器的长度

    for epoch in range(opt.epoch):  # 遍历训练轮数
        model.train()  # 设置模型为训练模式
        print("epoch: ", epoch + model.step_bias)  # 打印当前轮数（考虑步骤偏移）
        for i, (img, crops, label) in enumerate(data_loader):  # 遍历训练数据批次
            model.total_steps += 1  # 总步数加1

            model.set_input((img, crops, label))  # 设置输入数据
            model.forward()  # 前向传播
            loss = model.get_loss()  # 获取损失值

            model.optimize_parameters()  # 优化模型参数

            if model.total_steps % opt.loss_freq == 0:  # 如果达到损失打印频率
                print(
                    "Train loss: {}\tstep: {}".format(
                        model.get_loss(), model.total_steps
                    )
                )  # 打印训练损失和当前步数

        if epoch % opt.save_epoch_freq == 0:  # 如果达到模型保存频率
            print("saving the model at the end of epoch %d" % (epoch + model.step_bias))  # 打印保存信息
            model.save_trainer("model_epoch_%s.pth" % (epoch + model.step_bias))  # 保存模型

        model.eval()  # 设置模型为评估模式
        ap, fpr, fnr, acc = validate(model.model, val_loader, opt.gpu_ids)  # 在验证集上验证模型性能
        print(
            "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                epoch + model.step_bias, acc, ap, fpr, fnr
            )
        )  # 打印验证结果：准确率、平均精度、假阳性率、假阴性率
