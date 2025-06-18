import os  # 导入操作系统模块，用于文件和目录操作

def get_list(path) -> list:
    """
    递归读取指定路径下的所有图像文件
    
    参数:
        path (str): 要搜索的根目录路径
        
    返回:
        list: 包含所有找到的图像文件的完整路径列表
    """
    image_list = list()  # 初始化空列表，用于存储图像文件路径
    for root, dirs, files in os.walk(path):  # 递归遍历目录树
        for f in files:  # 遍历当前目录中的所有文件
            if f.split('.')[1] in ['png', 'jpg', 'jpeg']:  # 检查文件扩展名是否为图像格式
                image_list.append(os.path.join(root, f))  # 将完整的图像文件路径添加到列表中
    return image_list  # 返回包含所有图像文件路径的列表