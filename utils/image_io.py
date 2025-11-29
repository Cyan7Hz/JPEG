"""Module for JPEG compression: image_io.py

This module handles image input/output functionality
in the JPEG compression and decompression process.
"""

import numpy as np
from PIL import Image
import os
import json

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


def load_image_for_jpeg_processing(image_path=None, to_grayscale=None, save_to_file=True):
    """
    加载图像并进行JPEG处理前的准备
    
    Args:
        image_path: 图像文件路径，如果为None则使用默认路径
        to_grayscale: 是否转换为灰度图
        save_to_file: 是否保存预处理结果到JSON文件
    
    Returns:
        numpy.ndarray: 处理后的图像数据
    """
    # 如果未提供路径，使用默认路径
    if image_path is None:
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 构建默认图像路径
        image_path = os.path.join(project_root, 'data', 'input.jpg')
    
    # 读取图像
    image_array = read_image(image_path)
    
    # 转换为灰度图（如果需要）
    if to_grayscale:
        image_array = convert_to_grayscale(image_array)
        # 添加批次维度，方便后续处理
        image_array = np.expand_dims(image_array, axis=2)
    
    # 保存预处理结果到JSON文件
    if save_to_file:
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 构建输出文件路径
        output_path = os.path.join(project_root, 'temp', 'preprocessed.json')
        
        # 准备要保存的数据
        preprocessed_data = {
            'image_array': image_array,
            'shape': image_array.shape,
            'dtype': str(image_array.dtype),
            'to_grayscale': to_grayscale
        }
        
        # 保存数据
        save_to_json(preprocessed_data, output_path)
    
    return image_array
