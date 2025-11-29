import numpy as np
import json
import os
from typing import List, Dict, Union, Tuple


# JPEG标准量化表 - 亮度分量（高质量）
JPEG_LUMINANCE_QUANT_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)

# JPEG标准量化表 - 色度分量（高质量）
JPEG_CHROMINANCE_QUANT_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float64)


def generate_quantization_table(quality: int, is_luminance: bool = True) -> np.ndarray:
    """
    根据质量因子生成量化表
    
    Args:
        quality: 质量因子，范围1-100，1表示最低质量，100表示最高质量
        is_luminance: 是否为亮度分量的量化表
        
    Returns:
        quantization_table: 8x8的量化表
    """
    if quality < 1 or quality > 100:
        raise ValueError("质量因子必须在1到100之间")
    
    # 选择基础量化表
    if is_luminance:
        base_table = JPEG_LUMINANCE_QUANT_TABLE.copy()
    else:
        base_table = JPEG_CHROMINANCE_QUANT_TABLE.copy()
    
    # 根据质量因子计算缩放因子
    if quality < 50:
        scale_factor = 50.0 / quality
    else:
        scale_factor = 2.0 - quality / 50.0
    
    # 生成量化表
    quant_table = np.round(base_table * scale_factor)
    
    # 确保量化值至少为1
    quant_table = np.maximum(1, quant_table)
    # 确保量化值不超过255
    quant_table = np.minimum(255, quant_table)
    
    return quant_table.astype(np.float64)


def quantize_block(dct_block: np.ndarray, quant_table: np.ndarray) -> np.ndarray:
    """
    对单个DCT系数块进行量化
    
    Args:
        dct_block: DCT变换后的系数块，形状为(8, 8)
        quant_table: 量化表，形状为(8, 8)
        
    Returns:
        quantized_block: 量化后的系数块，形状为(8, 8)
    """
    if dct_block.shape != (8, 8):
        raise ValueError(f"输入DCT块必须是8x8大小，当前形状: {dct_block.shape}")
    if quant_table.shape != (8, 8):
        raise ValueError(f"量化表必须是8x8大小，当前形状: {quant_table.shape}")
    
    # 量化操作：DCT系数 / 量化表值，四舍五入取整
    quantized_block = np.round(dct_block / quant_table)
    
    return quantized_block.astype(np.int32)


def dequantize_block(quantized_block: np.ndarray, quant_table: np.ndarray) -> np.ndarray:
    """
    对单个量化系数块进行逆量化
    
    Args:
        quantized_block: 量化后的系数块，形状为(8, 8)
        quant_table: 量化表，形状为(8, 8)
        
    Returns:
        dequantized_block: 逆量化后的DCT系数块，形状为(8, 8)
    """
    if quantized_block.shape != (8, 8):
        raise ValueError(f"输入量化块必须是8x8大小，当前形状: {quantized_block.shape}")
    if quant_table.shape != (8, 8):
        raise ValueError(f"量化表必须是8x8大小，当前形状: {quant_table.shape}")
    
    # 逆量化操作：量化系数 * 量化表值
    dequantized_block = quantized_block.astype(np.float64) * quant_table
    
    return dequantized_block


def quantize_blocks(dct_blocks: List[np.ndarray], quality: int = 80, component_type: str = 'luminance') -> Tuple[List[np.ndarray], np.ndarray]:
    """
    对多个DCT系数块进行量化
    
    Args:
        dct_blocks: DCT系数块列表，每个块形状为(8, 8)
        quality: 质量因子，范围1-100
        component_type: 分量类型，'luminance'（亮度）或 'chrominance'（色度）
        
    Returns:
        quantized_blocks: 量化后的系数块列表
        quant_table: 使用的量化表
    """
    # 生成量化表
    is_luminance = (component_type.lower() == 'luminance')
    quant_table = generate_quantization_table(quality, is_luminance)
    
    # 对每个块进行量化
    quantized_blocks = []
    for block in dct_blocks:
        quantized_block = quantize_block(block, quant_table)
        quantized_blocks.append(quantized_block)
    
    return quantized_blocks, quant_table


def dequantize_blocks(quantized_blocks: List[np.ndarray], quant_table: np.ndarray) -> List[np.ndarray]:
    """
    对多个量化系数块进行逆量化
    
    Args:
        quantized_blocks: 量化后的系数块列表，每个块形状为(8, 8)
        quant_table: 使用的量化表
        
    Returns:
        dequantized_blocks: 逆量化后的DCT系数块列表
    """
    # 对每个块进行逆量化
    dequantized_blocks = []
    for block in quantized_blocks:
        dequantized_block = dequantize_block(block, quant_table)
        dequantized_blocks.append(dequantized_block)
    
    return dequantized_blocks


def load_quantized_coefficients(file_path: str) -> Tuple[List[np.ndarray], np.ndarray, dict]:
    """
    从JSON文件加载量化系数和量化表
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        quantized_blocks: 量化系数块列表
        quant_table: 使用的量化表
        metadata: 元数据字典
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 加载量化系数块
        quantized_blocks_data = data['quantized_blocks']
        quantized_blocks = [np.array(block, dtype=np.int32) for block in quantized_blocks_data]
        
        # 加载量化表
        quant_table = np.array(data['quantization_table'], dtype=np.float64)
        
        # 加载元数据
        metadata = data['metadata'] if 'metadata' in data else {}
        
        return quantized_blocks, quant_table, metadata
    except Exception as e:
        print(f"加载量化系数失败: {e}")
        raise


def save_quantized_coefficients(quantized_blocks: List[np.ndarray], quant_table: np.ndarray, 
                               metadata: dict, file_path: str) -> None:
    """
    保存量化系数和量化表到JSON文件
    
    Args:
        quantized_blocks: 量化系数块列表
        quant_table: 使用的量化表
        metadata: 元数据字典
        file_path: 输出JSON文件路径
    """
    try:
        # 将numpy数组转换为列表以便JSON序列化
        quantized_blocks_data = [block.tolist() for block in quantized_blocks]
        quant_table_data = quant_table.tolist()
        
        # 创建保存数据
        save_data = {
            'quantized_blocks': quantized_blocks_data,
            'quantization_table': quant_table_data,
            'metadata': metadata
        }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存到JSON文件
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        print(f"量化系数已保存到: {file_path}")
    except Exception as e:
        print(f"保存量化系数失败: {e}")
        raise


def save_dequantized_coefficients(dequantized_blocks: List[np.ndarray], metadata: dict, file_path: str) -> None:
    """
    保存逆量化后的DCT系数到JSON文件
    
    Args:
        dequantized_blocks: 逆量化后的DCT系数块列表
        metadata: 元数据字典
        file_path: 输出JSON文件路径
    """
    try:
        # 将numpy数组转换为列表以便JSON序列化
        dequantized_blocks_data = [block.tolist() for block in dequantized_blocks]
        
        # 创建保存数据
        save_data = {
            'dequantized_blocks': dequantized_blocks_data,
            'metadata': metadata
        }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存到JSON文件
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        print(f"逆量化系数已保存到: {file_path}")
    except Exception as e:
        print(f"保存逆量化系数失败: {e}")
        raise


def load_dequantized_coefficients(file_path: str) -> Tuple[List[np.ndarray], dict]:
    """
    从JSON文件加载逆量化后的DCT系数
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        dequantized_blocks: 逆量化后的DCT系数块列表
        metadata: 元数据字典
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 加载逆量化系数块
        dequantized_blocks_data = data['dequantized_blocks']
        dequantized_blocks = [np.array(block, dtype=np.float64) for block in dequantized_blocks_data]
        
        # 加载元数据
        metadata = data['metadata'] if 'metadata' in data else {}
        
        return dequantized_blocks, metadata
    except Exception as e:
        print(f"加载逆量化系数失败: {e}")
        raise


def main() -> None:
    """
    量化和逆量化模块的主函数，用于测试功能
    """
    # 测试数据：一个简单的DCT系数块示例
    test_dct_block = np.array([
        [225.0, -12.0, 15.0, -8.0, 4.0, 2.0, -1.0, 0.0],
        [-30.0, 45.0, -20.0, 10.0, -5.0, 3.0, -2.0, 1.0],
        [25.0, -18.0, 12.0, -7.0, 4.0, -2.0, 1.0, -0.5],
        [-15.0, 10.0, -8.0, 5.0, -3.0, 2.0, -1.0, 0.5],
        [8.0, -5.0, 4.0, -2.0, 1.0, -0.5, 0.25, -0.1],
        [-4.0, 3.0, -2.0, 1.0, -0.5, 0.3, -0.2, 0.1],
        [2.0, -1.5, 1.0, -0.8, 0.4, -0.2, 0.1, -0.05],
        [-1.0, 0.8, -0.6, 0.4, -0.2, 0.1, -0.08, 0.04]
    ], dtype=np.float64)
    
    print("原始DCT系数块:")
    print(np.round(test_dct_block, 2))
    
    # 测试不同质量因子的量化
    for quality in [10, 50, 90]:
        print(f"\n质量因子: {quality}")
        
        # 生成量化表
        quant_table = generate_quantization_table(quality, is_luminance=True)
        print("量化表:")
        print(np.round(quant_table, 1))
        
        # 量化
        quantized_block = quantize_block(test_dct_block, quant_table)
        print("量化后的系数块:")
        print(quantized_block)
        
        # 逆量化
        dequantized_block = dequantize_block(quantized_block, quant_table)
        print("逆量化后的系数块:")
        print(np.round(dequantized_block, 2))
        
        # 计算量化误差
        error = np.mean(np.abs(test_dct_block - dequantized_block))
        print(f"量化误差: {error:.6f}")


if __name__ == "__main__":
    main()