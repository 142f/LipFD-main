import argparse  # 导入命令行参数解析模块
import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库
from data import AVLip  # 导入AVLip数据集类
import torch.utils.data  # 导入PyTorch数据工具
from models import build_model  # 导入模型构建函数
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score  # 导入评估指标计算函数


def validate(model, loader, gpu_id):
    """
    在验证集上评估模型性能
    
    参数:
        model: 待评估的模型
        loader: 数据加载器，包含验证数据
        gpu_id: GPU设备ID列表
    
    返回:
        ap: 平均精度(Average Precision)
        fpr: 假阳性率(False Positive Rate)
        fnr: 假阴性率(False Negative Rate)
        acc: 准确率(Accuracy)
    """
    print("validating...")  # 打印验证开始信息
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")  # 设置计算设备
    with torch.no_grad():  # 禁用梯度计算，减少内存使用并加速推理
        y_true, y_pred = [], []  # 初始化真实标签和预测标签列表
        for img, crops, label in loader:  # 遍历验证数据批次
            img_tens = img.to(device)  # 将图像数据移至指定设备
            crops_tens = [[t.to(device) for t in sublist] for sublist in crops]  # 将裁剪图像数据移至指定设备
            features = model.get_features(img_tens).to(device)  # 提取图像特征

            y_pred.extend(model(crops_tens, features)[0].sigmoid().flatten().tolist())  # 获取模型预测结果并应用sigmoid函数
            y_true.extend(label.flatten().tolist())  # 收集真实标签
    y_true = np.array(y_true)  # 转换为NumPy数组
    y_pred = np.where(np.array(y_pred) >= 0.5, 1, 0)  # 将预测概率转换为二分类标签（阈值0.5）

    # 计算评估指标
    ap = average_precision_score(y_true, y_pred)  # 计算平均精度
    cm = confusion_matrix(y_true, y_pred)  # 计算混淆矩阵
    tp, fn, fp, tn = cm.ravel()  # 提取真阳性、假阴性、假阳性、真阴性数量
    fnr = fn / (fn + tp)  # 计算假阴性率
    fpr = fp / (fp + tn)  # 计算假阳性率
    acc = accuracy_score(y_true, y_pred)  # 计算准确率
    return ap, fpr, fnr, acc  # 返回评估指标


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real_list_path", type=str, default="./datasets/val/0_real")  # 真实样本路径
    parser.add_argument("--fake_list_path", type=str, default="./datasets/val/1_fake")  # 伪造样本路径
    parser.add_argument("--max_sample", type=int, default=1000, help="max number of validate samples")  # 最大验证样本数
    parser.add_argument("--batch_size", type=int, default=10)  # 批次大小
    parser.add_argument("--data_label", type=str, default="val")  # 数据标签
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")  # 模型架构
    parser.add_argument("--ckpt", type=str, default="./checkpoints/ckpt.pth")  # 检查点路径
    parser.add_argument("--gpu", type=int, default=0)  # GPU设备ID

    opt = parser.parse_args()  # 解析命令行参数

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")  # 设置计算设备
    print(f"Using cuda {opt.gpu} for inference.")  # 打印使用的设备信息

    model = build_model(opt.arch)  # 构建模型
    state_dict = torch.load(opt.ckpt, map_location="cpu")  # 加载模型权重
    model.load_state_dict(state_dict["model"])  # 将权重加载到模型中
    print("Model loaded.")  # 打印模型加载完成信息
    model.eval()  # 设置模型为评估模式
    model.to(device)  # 将模型移至指定设备

    dataset = AVLip(opt)  # 创建数据集
    loader = data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True
    )  # 创建数据加载器
    ap, fpr, fnr, acc = validate(model, loader, gpu_id=[opt.gpu])  # 验证模型性能
    print(f"acc: {acc} ap: {ap} fpr: {fpr} fnr: {fnr}")  # 打印评估结果
