import os
import pandas as pd
import numpy as np
import json

def crop_center(data, target_shape, center=None):
    """
    从数据中裁剪指定大小的区域
    
    Args:
        data: 输入数据数组
        target_shape: 目标形状 (height, width)
        center: 中心点位置 (center_h, center_w)，如果为None则使用数据中心
    
    Returns:
        裁剪后的数据
    """
    h, w = data.shape
    th, tw = target_shape
    
    if center is None:
        # 默认使用数据中心
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
    else:
        # 使用指定的中心点
        center_h, center_w = center
        start_h = center_h - (th // 2)
        start_w = center_w - (tw // 2)
    
    # 确保索引在有效范围内
    start_h = max(0, min(start_h, h - th))
    start_w = max(0, min(start_w, w - tw))
    
    return data[start_h:start_h+th, start_w:start_w+tw]

def downsample(data, factor):
    return data[::factor, ::factor]

def process_row(row, target_shape, downsample_factor, keys, center=None):
    for key in keys:
        arr = np.array(json.loads(row[key]))
        arr = crop_center(arr, target_shape, center=center)
        if downsample_factor > 1:
            arr = downsample(arr, downsample_factor)
        row[key] = json.dumps(arr.tolist(), ensure_ascii=False)
    return row

def process_csv(input_path, output_path, target_shape, downsample_factor, keys, center=None):
    df = pd.read_csv(input_path)
    df = df.apply(lambda row: process_row(row, target_shape, downsample_factor, keys, center=center), axis=1)
    df.to_csv(output_path, index=False)

def process_folder(input_folder, output_folder, target_shape, downsample_factor, keys, center=None):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_csv(input_path, output_path, target_shape, downsample_factor, keys, center=center)
            print(f"Processed {filename}")

def process_wind_data(input_folder=None, output_folder=None, target_shape=(11,11), downsample_factor=1, center=None, input_filename=None, output_filename=None):
    """
    处理风速数据的缩小化
    
    Args:
        input_folder: 输入文件夹路径（当input_filename为None时使用）
        output_folder: 输出文件夹路径（当output_filename为None时使用）
        target_shape: 目标图像大小 (height, width)，默认(11, 11)
        downsample_factor: 分辨率缩小倍数，默认1
        center: 中心点位置 (center_h, center_w)，如果为None则使用数据中心
        input_filename: 可选，输入文件名（如果指定，则只处理该文件）
        output_filename: 可选，输出文件名（如果指定，则使用该文件名输出）
    
    Returns:
        None
    """
    if input_folder is None and output_folder is None and input_filename is None and output_filename is None:
        raise ValueError("input_folder, output_folder, input_filename, output_filename 不能同时为空")
    if input_filename is not None and output_filename is not None:
        # 处理单个文件
        input_path = input_filename
        output_path = output_filename
        process_csv(input_path, output_path, target_shape, downsample_factor, keys=['u100', 'v100'], center=center)
        print(f"Processed {input_path} -> {output_path}")
    else:
        # 处理整个文件夹
        process_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            target_shape=target_shape,
            downsample_factor=downsample_factor,
            keys=['u100', 'v100'],
            center=center
        )

def process_temp_geopotential_data(input_folder=None, output_folder=None, target_shape=(11,11), downsample_factor=1, center=None, input_filename=None, output_filename=None):
    """
    处理温度位势能数据的缩小化
    
    Args:
        input_folder: 输入文件夹路径（当input_filename为None时使用）
        output_folder: 输出文件夹路径（当output_filename为None时使用）
        target_shape: 目标图像大小 (height, width)，默认(11, 11)
        downsample_factor: 分辨率缩小倍数，默认1
        center: 中心点位置 (center_h, center_w)，如果为None则使用数据中心
        input_filename: 可选，输入文件名（如果指定，则只处理该文件）
        output_filename: 可选，输出文件名（如果指定，则使用该文件名输出）
    
    Returns:
        None
    """
    if input_folder is None and output_folder is None and input_filename is None and output_filename is None:
        raise ValueError("input_folder, output_folder, input_filename, output_filename 不能同时为空")
    
    if input_filename is not None and output_filename is not None:
        # 处理单个文件
        input_path = input_filename
        output_path = output_filename
        process_csv(input_path, output_path, target_shape, downsample_factor, keys=['t', 'z'], center=center)
        print(f"Processed {input_path} -> {output_path}")
    else:
        # 处理整个文件夹
        process_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            target_shape=target_shape,
            downsample_factor=downsample_factor,
            keys=['t', 'z'],
            center=center
        )

# if __name__ == "__main__":
#     # 1. 缩小风速数据
#     process_wind_data(
#         input_folder="data/processed",
#         output_folder="data/processed_small",
#         target_shape=(11, 11),
#         downsample_factor=1,
#         center=None  # 使用数据中心，可以传入 (center_h, center_w) 来指定中心点
#     )
    
#     # 2. 缩小温度、位势数据
#     process_temp_geopotential_data(
#         input_folder="data/processed_temp",
#         output_folder="data/processed_temp_small",
#         target_shape=(11, 11),
#         downsample_factor=1,
#         center=None  # 使用数据中心，可以传入 (center_h, center_w) 来指定中心点
#     )
  