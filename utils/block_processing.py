"""Module for JPEG compression: block_processing.py

This module handles image blocking and block reconstruction functionality
for the JPEG compression and decompression process.
"""

import numpy as np
import os


def split_image_into_blocks(image_array, block_size=8):
    """
    将图像分割成大小为block_size x block_size的块
    
    Args:
        image_array: numpy数组，格式为(height, width, channels)或(height, width)
        block_size: 块的大小，默认为8
    
    Returns:
        tuple: (blocks, pad_info)
            - blocks: numpy数组，格式为(channels, num_blocks_h, num_blocks_w, block_size, block_size)
            - pad_info: 包含填充信息的字典
    """
    # 保存原始形状
    original_shape = image_array.shape
    
    # 确保图像是3D的 (如果是灰度图，添加通道维度)
    if len(original_shape) == 2:
        image_array = np.expand_dims(image_array, axis=2)
    
    height, width, channels = image_array.shape
    
    # 计算需要填充的大小，确保图像尺寸是block_size的倍数
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    
    # 进行填充（使用边缘像素值进行填充，避免边界效应）
    if pad_height > 0 or pad_width > 0:
        padded_image = np.pad(image_array, 
                             ((0, pad_height), (0, pad_width), (0, 0)), 
                             mode='edge')
    else:
        padded_image = image_array.copy()
    
    padded_shape = padded_image.shape
    
    # 计算块的数量
    num_blocks_h = padded_shape[0] // block_size
    num_blocks_w = padded_shape[1] // block_size
    
    # 创建块数组，按照新格式 (channels, num_blocks_h, num_blocks_w, block_size, block_size)
    blocks = np.zeros((channels, num_blocks_h, num_blocks_w, block_size, block_size), 
                     dtype=image_array.dtype)
    
    # 分割图像到块中
    for c in range(channels):
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                # 计算块的起始和结束索引
                h_start = i * block_size
                h_end = h_start + block_size
                w_start = j * block_size
                w_end = w_start + block_size
                
                # 提取块并放置在正确的通道位置
                blocks[c, i, j] = padded_image[h_start:h_end, w_start:w_end, c]
    
    # 如果原始图像是2D的，移除通道维度
    if len(original_shape) == 2:
        blocks = blocks.squeeze(axis=0)
    
    print(f"图像分割完成: {num_blocks_h}x{num_blocks_w}个{block_size}x{block_size}块")
    print(f"原始尺寸: {original_shape}，填充后尺寸: {padded_shape}")
    
    pad_info = {
        'original_shape': original_shape,
        'padded_shape': padded_shape,
        'num_blocks_h': num_blocks_h,
        'num_blocks_w': num_blocks_w
    }
    return blocks, pad_info


def reconstruct_image_from_blocks(blocks, pad_info):
    """
    将块重新组合成完整图像
    
    Args:
        blocks: numpy数组，格式为(channels, num_blocks_h, num_blocks_w, block_size, block_size)
        pad_info: 包含填充信息的字典
    
    Returns:
        numpy.ndarray: 重建的图像，格式与原始图像相同
    """
    # 从填充信息中获取必要参数
    original_shape = pad_info['original_shape']
    padded_shape = pad_info['padded_shape']
    num_blocks_h = pad_info['num_blocks_h']
    num_blocks_w = pad_info['num_blocks_w']
    
    # 获取块的维度信息
    if len(blocks.shape) == 4:
        # 灰度图（没有通道维度）
        _, block_size, _, _ = blocks.shape
        channels = 1
    else:
        # 彩色图
        channels, _, _, block_size, _ = blocks.shape
    
    # 计算重建图像的尺寸
    reconstructed_height = num_blocks_h * block_size
    reconstructed_width = num_blocks_w * block_size
    
    # 创建重建图像数组
    if channels == 1:
        reconstructed_image = np.zeros((reconstructed_height, reconstructed_width), 
                                      dtype=blocks.dtype)
        # 将块放置到重建图像中
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                # 计算块的起始和结束索引
                h_start = i * block_size
                h_end = h_start + block_size
                w_start = j * block_size
                w_end = w_start + block_size
                
                # 放置块
                reconstructed_image[h_start:h_end, w_start:w_end] = blocks[i, j]
    else:
        reconstructed_image = np.zeros((reconstructed_height, reconstructed_width, channels), 
                                      dtype=blocks.dtype)
        # 将块放置到重建图像中
        for c in range(channels):
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    # 计算块的起始和结束索引
                    h_start = i * block_size
                    h_end = h_start + block_size
                    w_start = j * block_size
                    w_end = w_start + block_size
                    
                    # 放置块
                    reconstructed_image[h_start:h_end, w_start:w_end, c] = blocks[c, i, j]
    
    # 裁剪到原始图像尺寸
    if len(original_shape) == 2:
        # 灰度图
        reconstructed_image = reconstructed_image[:original_shape[0], :original_shape[1]]
    else:
        # 彩色图
        reconstructed_image = reconstructed_image[:original_shape[0], :original_shape[1], :]
    
    print(f"图像重建完成，恢复到原始尺寸: {original_shape}")
    
    return reconstructed_image