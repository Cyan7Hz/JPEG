"""Module for JPEG compression: image_io.py

This module handles image input/output functionality
in the JPEG compression and decompression process.
"""

import numpy as np
from PIL import Image
import os
import json














# PART 1: 图像输入输出 -----------------------------------------------------------



def read_image(image_path):
    """
    读取图像文件并转换为numpy数组
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        numpy.ndarray: 图像数据，格式为(height, width, channels)，值范围[0, 255]
    
    Raises:
        FileNotFoundError: 当图像文件不存在时
        ValueError: 当图像格式不支持时
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    try:
        with Image.open(image_path) as img:
            # 转换为RGB格式（处理灰度图和彩色图）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 转换为numpy数组
            image_array = np.array(img, dtype=np.uint8)
            
            print(f"成功读取图像: {image_path}")
            print(f"图像尺寸: {image_array.shape[1]}x{image_array.shape[0]}")
            print(f"通道数: {image_array.shape[2]}")
            
            return image_array
    except Exception as e:
        raise ValueError(f"读取图像失败: {str(e)}")


def save_image(image_array, output_path):
    """
    保存numpy数组为图像文件
    
    Args:
        image_array: numpy数组，格式为(height, width, channels)，值范围[0, 255]
        output_path: 输出图像文件路径
    
    Raises:
        ValueError: 当图像数组格式不正确时
        IOError: 当保存失败时
    """
    # 确保图像数组格式正确
    if not isinstance(image_array, np.ndarray):
        raise ValueError("输入必须是numpy数组")
    
    # 确保值在有效范围内并转换为uint8
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    
    # 确保维度正确
    if len(image_array.shape) == 2:
        # 灰度图，添加通道维度
        image_array = np.expand_dims(image_array, axis=2)
    
    if image_array.shape[2] == 1:
        # 单通道灰度图
        img = Image.fromarray(image_array.squeeze(), mode='L')
    elif image_array.shape[2] == 3:
        # 三通道RGB图
        img = Image.fromarray(image_array, mode='RGB')
    else:
        raise ValueError(f"不支持的通道数: {image_array.shape[2]}")
    
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        img.save(output_path)
        print(f"成功保存图像: {output_path}")
    except Exception as e:
        raise IOError(f"保存图像失败: {str(e)}")




















# PART 2: 色彩空间转换 -----------------------------------------------------------



def rgb_to_ycrcb(image_array):
    """
    将RGB图像转换为YCrCb色彩空间
    
    Args:
        image_array: RGB图像的numpy数组，形状为(height, width, channels)
                    值范围应为[0, 255]
    
    Returns:
        numpy.ndarray: YCrCb图像的numpy数组，形状为(height, width, channels)
                      值范围为[0, 255]
    
    Raises:
        ValueError: 当输入图像格式不正确时
    """
    # 检查输入格式
    if not isinstance(image_array, np.ndarray):
        raise ValueError("输入必须是numpy数组")
    
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("输入必须是RGB图像，形状为(H, W, 3)")
    
    # 确保值在有效范围内
    image_array = np.clip(image_array, 0, 255)
    
    # 将图像值归一化到[0, 1]范围
    rgb_normalized = image_array / 255.0
    
    # 获取图像尺寸
    height, width = rgb_normalized.shape[:2]
    
    # 创建输出数组
    ycrcb_image = np.zeros_like(rgb_normalized)
    
    # 使用标准的RGB到YCrCb转换公式
    # Y  = 0.299*R + 0.587*G + 0.114*B
    # Cr = 0.500*R - 0.419*G - 0.081*B + 0.5
    # Cb = -0.169*R - 0.331*G + 0.500*B + 0.5
    
    # 提取R, G, B通道
    R = rgb_normalized[:, :, 0]
    G = rgb_normalized[:, :, 1]
    B = rgb_normalized[:, :, 2]
    
    # 计算Y, Cr, Cb通道
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = 0.500 * R - 0.419 * G - 0.081 * B + 0.5
    Cb = -0.169 * R - 0.331 * G + 0.500 * B + 0.5
    
    # 将Y, Cr, Cb赋值给输出数组
    ycrcb_image[:, :, 0] = Y
    ycrcb_image[:, :, 1] = Cr
    ycrcb_image[:, :, 2] = Cb
    
    # 将值缩放到[0, 255]范围并转换为uint8
    ycrcb_image = np.clip(ycrcb_image * 255.0, 0, 255).astype(np.uint8)
    
    return ycrcb_image


def ycrcb_to_rgb(image_array):
    """
    将YCrCb图像转换为RGB色彩空间
    
    Args:
        image_array: YCrCb图像的numpy数组，形状为(height, width, channels)
                    值范围应为[0, 255]
    
    Returns:
        numpy.ndarray: RGB图像的numpy数组，形状为(height, width, channels)
                      值范围为[0, 255]
    
    Raises:
        ValueError: 当输入图像格式不正确时
    """
    # 检查输入格式
    if not isinstance(image_array, np.ndarray):
        raise ValueError("输入必须是numpy数组")
    
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("输入必须是YCrCb图像，形状为(H, W, 3)")
    
    # 确保值在有效范围内
    image_array = np.clip(image_array, 0, 255)
    
    # 将图像值归一化到[0, 1]范围
    ycrcb_normalized = image_array / 255.0
    
    # 获取图像尺寸
    height, width = ycrcb_normalized.shape[:2]
    
    # 创建输出数组
    rgb_image = np.zeros_like(ycrcb_normalized)
    
    # 使用标准的YCrCb到RGB转换公式
    # R = Y + 1.402*(Cr-0.5)
    # G = Y - 0.344*(Cb-0.5) - 0.714*(Cr-0.5)
    # B = Y + 1.772*(Cb-0.5)
    
    # 提取Y, Cr, Cb通道
    Y = ycrcb_normalized[:, :, 0]
    Cr = ycrcb_normalized[:, :, 1]
    Cb = ycrcb_normalized[:, :, 2]
    
    # 计算R, G, B通道
    R = Y + 1.402 * (Cr - 0.5)
    G = Y - 0.344 * (Cb - 0.5) - 0.714 * (Cr - 0.5)
    B = Y + 1.772 * (Cb - 0.5)
    
    # 将R, G, B赋值给输出数组
    rgb_image[:, :, 0] = R
    rgb_image[:, :, 1] = G
    rgb_image[:, :, 2] = B
    
    # 将值缩放到[0, 255]范围并转换为uint8
    rgb_image = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    
    return rgb_image


























# PART 3: 灰度化处理 -----------------------------------------------------------



def convert_to_grayscale(image_array):
    """
    将RGB图像转换为灰度图
    
    Args:
        image_array: RGB图像的numpy数组
    
    Returns:
        numpy.ndarray: 灰度图的numpy数组
    """
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("输入必须是RGB图像")
    
    # 使用加权平均法转换为灰度图
    # Y = 0.299*R + 0.587*G + 0.114*B
    grayscale = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
    
    return grayscale.astype(np.uint8)
































# PART 4: JSON-numpy数组转换 --------------------------------------------------



def save_to_json(data, file_path):
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: JSON文件路径
    """
    # 确保目录存在
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理numpy数组，转换为可序列化格式
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return {
                'data': obj.tolist(),
                'shape': obj.shape,
                'dtype': str(obj.dtype)
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # 转换数据
    serializable_data = convert_numpy(data)
    
    # 保存到JSON文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功保存数据到: {file_path}")


def load_from_json(file_path):
    """
    从JSON文件加载数据
    
    Args:
        file_path: JSON文件路径
    
    Returns:
        加载的数据（如果是numpy数组则恢复为numpy.ndarray类型）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 恢复numpy数组
    def restore_numpy(obj):
        if isinstance(obj, dict) and 'data' in obj and 'shape' in obj and 'dtype' in obj:
            return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
        elif isinstance(obj, dict):
            return {key: restore_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [restore_numpy(item) for item in obj]
        else:
            return obj
    
    return restore_numpy(data)

