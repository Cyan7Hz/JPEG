import numpy as np
import json
import os
from typing import List, Tuple, Union


def compute_dct_basis(size: int = 8) -> np.ndarray:
    """
    计算DCT变换的基矩阵
    
    Args:
        size: 变换块的大小，默认为8x8
        
    Returns:
        dct_basis: DCT基矩阵，形状为(size, size)
    """
    basis = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == 0:
                basis[i, j] = 1.0 / np.sqrt(size)
            else:
                basis[i, j] = np.sqrt(2.0 / size) * np.cos((2 * j + 1) * i * np.pi / (2 * size))
    return basis


def apply_dct_to_block(block: np.ndarray, basis: np.ndarray = None) -> np.ndarray:
    """
    对单个图像块应用离散余弦变换(FDCT)
    
    Args:
        block: 输入图像块，形状为(8, 8)
        basis: 预计算的DCT基矩阵，默认None（将自动计算）
        
    Returns:
        dct_block: DCT变换后的系数块，形状为(8, 8)
    """
    if block.shape != (8, 8):
        raise ValueError(f"输入块必须是8x8大小，当前形状: {block.shape}")
    
    # 计算DCT基矩阵
    if basis is None:
        basis = compute_dct_basis(8)
    
    # 应用DCT变换: 基矩阵 × 输入块 × 基矩阵转置
    dct_block = np.dot(np.dot(basis, block.astype(np.float64)), basis.T)
    
    return dct_block


def apply_idct_to_block(dct_block: np.ndarray, basis: np.ndarray = None) -> np.ndarray:
    """
    对单个DCT系数块应用逆离散余弦变换(IDCT)
    
    Args:
        dct_block: 输入DCT系数块，形状为(8, 8)
        basis: 预计算的DCT基矩阵，默认None（将自动计算）
        
    Returns:
        block: IDCT变换后的图像块，形状为(8, 8)
    """
    if dct_block.shape != (8, 8):
        raise ValueError(f"输入DCT块必须是8x8大小，当前形状: {dct_block.shape}")
    
    # 计算DCT基矩阵
    if basis is None:
        basis = compute_dct_basis(8)
    
    # 应用IDCT变换: 基矩阵转置 × DCT系数块 × 基矩阵
    block = np.dot(np.dot(basis.T, dct_block.astype(np.float64)), basis)
    
    return block


def perform_dct_on_blocks(blocks, normalize: bool = True):
    """
    对一组图像块执行DCT变换
    
    Args:
        blocks: 图像块，可以是以下两种格式之一：
                1. 列表格式：图像块列表，每个块形状为(8, 8)
                2. 数组格式：numpy数组，形状为(channels, num_blocks_h, num_blocks_w, block_size, block_size)
        normalize: 是否对输入数据进行归一化（减去128）
        
    Returns:
        dct_blocks: DCT变换后的系数块，格式与输入相同
    """
    basis = compute_dct_basis(8)
    
    # 检查输入类型
    if isinstance(blocks, list):
        # 原始列表格式处理
        dct_blocks = []
        for block in blocks:
            # 归一化：减去128以将像素值范围从[0, 255]转为[-128, 127]
            if normalize:
                block = block - 128.0
            
            # 应用DCT
            dct_block = apply_dct_to_block(block, basis)
            dct_blocks.append(dct_block)
        
        return dct_blocks
    else:
        # 新的数组格式处理 (channels, num_blocks_h, num_blocks_w, block_size, block_size)
        # 检查输入形状
        if len(blocks.shape) != 5:
            raise ValueError(f"输入块组必须是5维数组，当前形状: {blocks.shape}")
        
        channels, num_blocks_h, num_blocks_w, block_size_h, block_size_w = blocks.shape
        
        # 检查块大小是否为8x8
        if block_size_h != 8 or block_size_w != 8:
            raise ValueError(f"输入块必须是8x8大小，当前形状: {blocks.shape}")
        
        # 初始化输出数组
        dct_blocks = np.zeros_like(blocks, dtype=np.float64)
        
        # 对每个通道的每个块应用DCT
        for c in range(channels):
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    # 获取当前块
                    block = blocks[c, i, j]
                    
                    # 归一化：减去128以将像素值范围从[0, 255]转为[-128, 127]
                    if normalize:
                        block = block - 128.0
                    
                    # 应用DCT
                    dct_blocks[c, i, j] = apply_dct_to_block(block, basis)
        
        return dct_blocks


def perform_idct_on_blocks(dct_blocks, normalize: bool = True):
    """
    对一组DCT系数块执行IDCT变换
    
    Args:
        dct_blocks: DCT系数块，可以是以下两种格式之一：
                   1. 列表格式：DCT系数块列表，每个块形状为(8, 8)
                   2. 数组格式：numpy数组，形状为(channels, num_blocks_h, num_blocks_w, block_size, block_size)
        normalize: 是否对输出数据进行反归一化（加上128）
        
    Returns:
        blocks: IDCT变换后的图像块，格式与输入相同
    """
    basis = compute_dct_basis(8)
    
    # 检查输入类型
    if isinstance(dct_blocks, list):
        # 原始列表格式处理
        blocks = []
        for dct_block in dct_blocks:
            # 应用IDCT
            block = apply_idct_to_block(dct_block, basis)
            
            # 反归一化：加上128
            if normalize:
                block = block + 128.0
                # 确保像素值在有效范围内[0, 255]
                block = np.clip(block, 0, 255)
            
            blocks.append(block)
        
        return blocks
    else:
        # 新的数组格式处理 (channels, num_blocks_h, num_blocks_w, block_size, block_size)
        # 检查输入形状
        if len(dct_blocks.shape) != 5:
            raise ValueError(f"输入DCT块组必须是5维数组，当前形状: {dct_blocks.shape}")
        
        channels, num_blocks_h, num_blocks_w, block_size_h, block_size_w = dct_blocks.shape
        
        # 检查块大小是否为8x8
        if block_size_h != 8 or block_size_w != 8:
            raise ValueError(f"输入DCT块必须是8x8大小，当前形状: {dct_blocks.shape}")
        
        # 初始化输出数组
        blocks = np.zeros_like(dct_blocks, dtype=np.float64)
        
        # 对每个通道的每个块应用IDCT
        for c in range(channels):
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    # 获取当前DCT块
                    dct_block = dct_blocks[c, i, j]
                    
                    # 应用IDCT
                    block = apply_idct_to_block(dct_block, basis)
                    
                    # 反归一化：加上128
                    if normalize:
                        block = block + 128.0
                        # 确保像素值在有效范围内[0, 255]
                        block = np.clip(block, 0, 255)
                    
                    blocks[c, i, j] = block
        
        return blocks


def load_blocks_from_json(file_path: str) -> Tuple[List[np.ndarray], dict]:
    """
    从JSON文件加载图像块数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        blocks: 图像块列表
        metadata: 元数据字典
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        blocks_data = data['blocks']
        metadata = data['metadata']
        
        # 将列表转换回numpy数组
        blocks = []
        for block_data in blocks_data:
            block = np.array(block_data, dtype=np.float64)
            blocks.append(block)
        
        return blocks, metadata
    except Exception as e:
        print(f"加载块数据失败: {e}")
        raise


def save_dct_coefficients_to_json(dct_blocks: List[np.ndarray], metadata: dict, file_path: str) -> None:
    """
    将DCT系数保存到JSON文件
    
    Args:
        dct_blocks: DCT系数块列表
        metadata: 元数据字典
        file_path: 输出JSON文件路径
    """
    try:
        # 将numpy数组转换为列表以便JSON序列化
        blocks_data = [block.tolist() for block in dct_blocks]
        
        # 创建保存数据
        save_data = {
            'blocks': blocks_data,
            'metadata': metadata
        }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存到JSON文件
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        print(f"DCT系数已保存到: {file_path}")
    except Exception as e:
        print(f"保存DCT系数失败: {e}")
        raise


def main() -> None:
    """
    DCT变换模块的主函数，用于测试DCT和IDCT功能
    """
    # 测试1: 简单8x8块的DCT变换
    test_block = np.array([
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 55, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94]
    ], dtype=np.float64)
    
    # 应用DCT
    print("原始块:")
    print(test_block)
    
    dct_result = apply_dct_to_block(test_block - 128)
    print("\nDCT变换结果:")
    print(np.round(dct_result, 2))
    
    # 应用IDCT
    idct_result = apply_idct_to_block(dct_result) + 128
    print("\nIDCT逆变换结果:")
    print(np.round(idct_result, 2))
    
    # 计算重构误差
    error = np.mean(np.abs(test_block - idct_result))
    print(f"\n重构误差: {error:.6f}")


if __name__ == "__main__":
    main()