import os
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import feature as audio


"""
AVLips 数据集的结构:
AVLips
├── 0_real  # 真实视频
├── 1_fake  # 伪造视频
└── wav     # 音频文件
    ├── 0_real
    └── 1_fake
"""

############ 自定义参数 ##############
N_EXTRACT = 10   # 从视频中提取的图像数量
WINDOW_LEN = 5   # 每个窗口的帧数
MAX_SAMPLE = 100 # 处理的最大样本数（针对每个类别，real/fake）

audio_root = "./AVLips/wav"  # 音频文件根目录
video_root = "./AVLips"      # 视频文件根目录
output_root = "./datasets/AVLips" # 处理后数据集的输出根目录
############################################

labels = [(0, "0_real"), (1, "1_fake")] # 标签和对应的目录名

def get_spectrogram(audio_file):
    """
    从音频文件生成梅尔频谱图并保存为临时图像。

    参数:
        audio_file (str): 音频文件的路径。
    """
    data, sr = librosa.load(audio_file) # 加载音频数据和采样率
    # 计算梅尔频谱图并转换为分贝单位
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    plt.imsave("./temp/mel.png", mel) # 将频谱图保存为图像


def run():
    """
    主处理函数，遍历数据集，提取视频帧和音频频谱图，
    并将它们组合成新的图像数据保存。
    """
    sample_count = 0 # 当前处理的样本计数器，用于 MAX_SAMPLE
    for label_id, dataset_name in labels: # 遍历标签（真实/伪造）
        # 为输出数据集创建目录（如果不存在）
        if not os.path.exists(f"{output_root}/{dataset_name}"):
            os.makedirs(f"{output_root}/{dataset_name}", exist_ok=True)

        if sample_count >= MAX_SAMPLE and dataset_name == "1_fake": # 限制伪造样本数量，假设真实样本也想限制，可以调整逻辑
            # 注意：原逻辑是 i == MAX_SAMPLE，且 i 在外层循环递增，这意味着它会限制第一个类别处理完后的第二个类别。
            # 修改为 sample_count，并在内层视频循环后递增，以更准确地限制每个类别的样本数（如果这是意图）。
            # 或者，如果 MAX_SAMPLE 是总样本数上限，则此条件应在最外层。
            # 当前逻辑：当处理到 '1_fake' 且 sample_count 达到 MAX_SAMPLE 时中断。
            # 如果希望对每个类别都限制 MAX_SAMPLE，则 sample_count 应在类别循环开始时重置。
            # 假设 MAX_SAMPLE 是针对每个类别的。因此，在类别循环开始时重置 sample_count。
            break # 如果已达到最大样本数，则停止处理当前类别
        
        current_category_sample_count = 0 # 当前类别的样本计数器

        video_dir_path = f"{video_root}/{dataset_name}" # 当前类别视频的目录路径
        video_list = os.listdir(video_dir_path) # 获取视频文件列表
        print(f"Handling {dataset_name}...")
        for video_filename in tqdm(video_list): # 遍历视频文件
            if current_category_sample_count >= MAX_SAMPLE:
                break # 如果当前类别已达到最大样本数，则跳到下一个类别

            # 加载视频
            video_path = f"{video_dir_path}/{video_filename}"
            video_capture = cv2.VideoCapture(video_path)
            # fps = video_capture.get(cv2.CAP_PROP_FPS) # 获取视频帧率 (未使用)
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) # 获取视频总帧数

            if frame_count <= WINDOW_LEN: # 如果视频帧数不足一个窗口，则跳过
                print(f"Skipping video {video_filename} due to insufficient frames: {frame_count}")
                video_capture.release()
                continue

            # 从帧中选择 N_EXTRACT 个起始点
            # 确保起始点加上窗口长度后不会超出总帧数
            frame_indices = np.linspace(
                0,
                frame_count - WINDOW_LEN - 1, # 确保窗口不会越界
                N_EXTRACT,
                endpoint=True,
                dtype=np.int_)
            # frame_indices.sort() # linspace 默认生成有序序列
            
            # 获取选定帧序列（每个起始点开始的 WINDOW_LEN 帧）
            # 这里的 frame_sequence 包含了所有需要提取的帧的索引
            frame_sequence_flat = []
            for start_idx in frame_indices:
                for i in range(WINDOW_LEN):
                    frame_sequence_flat.append(start_idx + i)
            
            frame_list = [] # 用于存储提取的视频帧
            current_frame_idx = 0
            # 读取视频帧
            while video_capture.isOpened() and current_frame_idx <= frame_sequence_flat[-1]:
                ret, frame = video_capture.read()
                if not ret:
                    print(f"Error in reading frame {video_filename}: {current_frame_idx}")
                    break
                if current_frame_idx in frame_sequence_flat:
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # 转换为 RGBA (原代码)
                    # 考虑到后续拼接和 imsave，通常 BGR 或 RGB 即可，除非透明度有特殊用途
                    # 如果 mel 频谱图是灰度或 RGB，拼接时通道数需要匹配
                    # plt.imread 读取的 png 可能是 RGBA 或 RGB，需要统一
                    frame = cv2.resize(frame, (500, 500))  # 调整帧大小
                    frame_list.append(frame)
                current_frame_idx += 1
            video_capture.release()

            if not frame_list or len(frame_list) < N_EXTRACT * WINDOW_LEN:
                print(f"Could not extract enough frames from {video_filename}. Expected {N_EXTRACT*WINDOW_LEN}, got {len(frame_list)}")
                continue

            # 加载音频
            audio_filename_base = video_filename.split(".")[0]
            audio_file_path = f"{audio_root}/{dataset_name}/{audio_filename_base}.wav"
            if not os.path.exists(audio_file_path):
                print(f"Audio file not found for {video_filename}: {audio_file_path}")
                continue

            group_idx = 0 # 用于命名输出图像的分组索引
            get_spectrogram(audio_file_path) # 生成并保存梅尔频谱图
            # 加载频谱图 (plt.imread 可能会返回 float32 范围 [0,1] 的 RGBA 图像)
            mel_image = plt.imread("./temp/mel.png") 
            # 如果是 RGBA 且只需要 RGB，则转换: mel_image = mel_image[:, :, :3]
            # 将频谱图像素值从 [0,1] 转换为 [0,255] 并设为 uint8
            mel_image_uint8 = (mel_image * 255).astype(np.uint8)
            
            # 计算音频频谱图与视频帧的映射关系
            # mel_image_uint8.shape[1] 是频谱图的宽度（时间轴）
            # frame_count 是视频的总帧数
            # mapping 表示每视频帧对应多少频谱图宽度（像素）
            # 如果 mel_image_uint8 是 RGBA，shape[1] 是宽度，shape[2] 是通道数
            # 假设 mel_image_uint8 已经是适合处理的形状 (H, W, C) 或 (H, W)
            # 如果是灰度图，需要扩展维度以匹配彩色视频帧进行拼接
            if len(mel_image_uint8.shape) == 2: # 如果是灰度图 (H, W)
                mel_image_uint8 = cv2.cvtColor(mel_image_uint8, cv2.COLOR_GRAY2BGR) # 转为 BGR
            elif mel_image_uint8.shape[2] == 4: # 如果是 RGBA
                mel_image_uint8 = cv2.cvtColor(mel_image_uint8, cv2.COLOR_RGBA2BGR) # 转为 BGR

            # 确保 mel_image_uint8 是 3 通道 BGR 图像
            if mel_image_uint8.shape[2] != 3:
                print(f"Spectrogram for {audio_filename_base} is not 3-channel BGR after conversion. Skipping.")
                continue

            # mapping = mel_image_uint8.shape[1] / frame_count
            # 修正 mapping 的计算，应该基于原始频谱图的宽度（如果 get_spectrogram 保存的是原始尺寸）
            # 或者，如果 get_spectrogram 保存的图像尺寸固定，则 mapping 基于该固定尺寸
            # 假设 get_spectrogram 保存的 mel.png 宽度与原始频谱图时间轴相关
            # 这里的 mel_image_uint8.shape[1] 是读取的 png 图像的宽度
            # 如果 librosa.display.specshow 用于保存，其宽度可能与帧数不直接对应
            # 假设 mel.png 的宽度代表了整个音频的频谱图时间轴
            # 原始代码的 mapping 计算方式是正确的，如果 mel.png 宽度对应总帧数时间
            mapping = mel_image_uint8.shape[1] / frame_count

            # 遍历提取的帧窗口
            for k in range(N_EXTRACT): # k 是提取窗口的索引
                start_frame_in_video = frame_indices[k] # 当前窗口在原视频中的起始帧索引
                # 提取当前窗口的视频帧 (WINDOW_LEN 帧)
                current_window_frames = frame_list[k*WINDOW_LEN : (k+1)*WINDOW_LEN]
                if len(current_window_frames) != WINDOW_LEN:
                    print(f"Insufficient frames for window {k} in {video_filename}. Skipping window.")
                    continue

                try:
                    # 计算频谱图中对应当前视频窗口的部分
                    # begin_mel_col = np.round(start_frame_in_video * mapping)
                    # end_mel_col = np.round((start_frame_in_video + WINDOW_LEN) * mapping)
                    # 确保索引不越界
                    # begin_mel_col = int(max(0, begin_mel_col))
                    # end_mel_col = int(min(mel_image_uint8.shape[1], end_mel_col))
                    # if begin_mel_col >= end_mel_col:
                    #     print(f"Invalid mel segment for window {k} in {video_filename}. Skipping window.")
                    #     continue
                    # sub_mel = mel_image_uint8[:, begin_mel_col:end_mel_col, :3] # 取 BGR 通道
                    # # 将提取的频谱图部分调整大小以匹配拼接需求
                    # # 目标是 (500, 500 * WINDOW_LEN)
                    # sub_mel_resized = cv2.resize(sub_mel, (500 * WINDOW_LEN, 500))

                    # 修正频谱图提取和拼接逻辑
                    # 目标：频谱图部分高度500，宽度与拼接的视频帧总宽度一致 (500 * WINDOW_LEN)
                    begin_mel_col = int(round(start_frame_in_video * mapping))
                    end_mel_col = int(round((start_frame_in_video + WINDOW_LEN) * mapping))
                    # 确保索引有效且不越界
                    begin_mel_col = max(0, min(begin_mel_col, mel_image_uint8.shape[1] - 1))
                    end_mel_col = max(begin_mel_col + 1, min(end_mel_col, mel_image_uint8.shape[1]))

                    if begin_mel_col >= end_mel_col:
                        print(f"Warning: Mel segment has zero or negative width for {video_filename}, window {k}. Skipping.")
                        continue
                    
                    sub_mel_segment = mel_image_uint8[:, begin_mel_col:end_mel_col]
                    # 调整频谱图片段大小为 (高度=500, 宽度=500*WINDOW_LEN)
                    sub_mel_resized = cv2.resize(sub_mel_segment, (500 * WINDOW_LEN, 500))

                    # 水平拼接当前窗口的视频帧
                    concatenated_frames = np.concatenate(current_window_frames, axis=1)
                    
                    # 确保拼接的视频帧和调整大小后的频谱图具有相同的通道数 (BGR)
                    if concatenated_frames.shape[2] != 3 or sub_mel_resized.shape[2] != 3:
                        print(f"Channel mismatch for {video_filename}, window {k}. Frames: {concatenated_frames.shape[2]}, Mel: {sub_mel_resized.shape[2]}. Skipping.")
                        continue

                    # 垂直拼接频谱图和视频帧
                    # 频谱图在上，视频帧在下。总高度 1000，总宽度 500 * WINDOW_LEN
                    combined_image = np.concatenate((sub_mel_resized, concatenated_frames), axis=0)
                    
                    # 保存组合图像
                    output_image_path = f"{output_root}/{dataset_name}/{audio_filename_base}_{group_idx}.png"
                    # cv2.imwrite 会处理 BGR -> 文件 的转换
                    cv2.imwrite(output_image_path, combined_image)
                    group_idx += 1
                except Exception as e: # 更具体的异常捕获可能更好，例如 ValueError
                    print(f"Error processing/saving window {k} for {video_filename}: {e}")
                    continue
            current_category_sample_count +=1
        # sample_count += 1 # 原代码的 i+=1 位置，如果 MAX_SAMPLE 是总数限制


if __name__ == "__main__":
    # 创建输出根目录（如果不存在）
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    # 创建临时文件目录（如果不存在）
    if not os.path.exists("./temp"):
        os.makedirs("./temp", exist_ok=True)
    run()
