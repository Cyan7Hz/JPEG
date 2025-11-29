import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any







# PART 1：DC 系数提取组件 -------------------------------------------------------------
def extract_dc_coefficients(quantized_blocks: List[np.ndarray]) -> List[int]:
    """
    从量化后的DCT系数块中提取DC系数
    
    Args:
        quantized_blocks: 量化后的DCT系数块列表
        
    Returns:
        dc_coefficients: DC系数列表
    """
    dc_coefficients = []
    for block in quantized_blocks:
        # DC系数位于块的左上角(0,0)位置
        dc_coefficient = int(block[0, 0])
        dc_coefficients.append(dc_coefficient)
    
    return dc_coefficients






















# PART 2：DC 码字计算组件 -------------------------------------------------------------
def calculate_dc_differences(dc_coefficients: List[int]) -> List[int]:
    """
    使用DPCM(差分脉冲编码调制)计算DC系数的差值
    
    Args:
        dc_coefficients: DC系数列表
        
    Returns:
        dc_differences: DC系数差值列表
    """
    if not dc_coefficients:
        return []
    
    dc_differences = [dc_coefficients[0]]  # 第一个块使用原始DC系数
    
    # 从第二个块开始，计算与前一个块DC系数的差值
    for i in range(1, len(dc_coefficients)):
        diff = dc_coefficients[i] - dc_coefficients[i-1]
        dc_differences.append(diff)
    
    return dc_differences


def calculate_bits_required(value: int) -> int:
    """
    计算表示一个整数所需的位数
    
    Args:
        value: 输入整数值
        
    Returns:
        bits: 所需位数
    """
    if value == 0:
        return 0
    
    # 计算绝对值的二进制位数
    abs_value = abs(value)
    bits = 0
    while abs_value > 0:
        bits += 1
        abs_value >>= 1
    
    return bits























# PART 3：DC 任务分发组件 -------------------------------------------------------------
def encode_dc_coefficients(quantized_blocks: List[np.ndarray]) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    对DC系数进行编码
    
    Args:
        quantized_blocks: 量化后的DCT系数块列表
        
    Returns:
        dc_encoded: DC编码结果列表，每个元素是(bits, value)元组
        metadata: 元数据字典
    """
    # 提取DC系数
    dc_coefficients = extract_dc_coefficients(quantized_blocks)
    
    # 计算DPCM差值
    dc_differences = calculate_dc_differences(dc_coefficients)
    
    # 编码差值
    dc_encoded = []
    for diff in dc_differences:
        # 计算差值所需的位数
        bits = calculate_bits_required(diff)
        dc_encoded.append((bits, diff))
    
    # # 构建元数据
    # metadata = {
    #     'block_count': len(quantized_blocks),
    #     'dc_coefficients': dc_coefficients,
    #     'dc_differences': dc_differences
    # }
    
    return dc_encoded


def decode_dc_coefficients(dc_encoded: List[Tuple[int, int]]) -> List[int]:
    """
    解码DC系数
    
    Args:
        dc_encoded: DC编码结果列表，每个元素是(bits, value)元组
        
    Returns:
        dc_coefficients: 重建的DC系数列表
    """
    if not dc_encoded:
        return []
    
    dc_coefficients = []
    previous_dc = 0
    
    for i, (bits, value) in enumerate(dc_encoded):
        if i == 0:
            # 第一个块直接使用编码值
            current_dc = value
        else:
            # 后续块使用差值加上前一个DC值
            current_dc = previous_dc + value
        
        dc_coefficients.append(current_dc)
        previous_dc = current_dc
    
    return dc_coefficients






















# PART 4：Block 重建组件 -------------------------------------------------------------
def decode_dc2blocks(ac_blocks: List[np.ndarray], dc_encoded: List[Tuple[int, int]]) -> List[np.ndarray]:
    """
    将解码后的DC系数放回到块中
    
    Args:
        ac_blocks: 只包含AC系数的块列表（DC位置为0或其他填充值）
        dc_coefficients: 解码后的DC系数列表
        
    Returns:
        reconstructed_blocks: 包含正确DC和AC系数的完整块列表
    """
    if len(ac_blocks) != len(dc_encoded):
        raise ValueError(f"块数量({len(ac_blocks)})与DC系数数量({len(dc_coefficients)})不匹配")
    
    dc_coefficients = decode_dc_coefficients(dc_encoded)
    reconstructed_blocks = []
    for i, block in enumerate(ac_blocks):
        # 创建块的副本以避免修改原始数据
        new_block = block.copy()
        # 将DC系数放置在块的左上角(0,0)位置
        new_block[0, 0] = dc_coefficients[i]
        reconstructed_blocks.append(new_block)
    
    return reconstructed_blocks


def separate_dc_and_ac(quantized_blocks: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
    """
    分离DC和AC系数
    
    Args:
        quantized_blocks: 量化后的DCT系数块列表
        
    Returns:
        dc_coefficients: DC系数列表
        ac_blocks: 只包含AC系数的块列表（DC位置为0）
    """
    dc_coefficients = []
    ac_blocks = []
    
    for block in quantized_blocks:
        # 提取DC系数
        dc = int(block[0, 0])
        dc_coefficients.append(dc)
        
        # 创建AC块（将DC位置设为0）
        ac_block = block.copy()
        ac_block[0, 0] = 0
        ac_blocks.append(ac_block)
    
    return dc_coefficients, ac_blocks



























# PART 5：文件操作组件 -------------------------------------------------------------
def save_dc_encoded(dc_encoded: List[Tuple[int, int]], file_path: str) -> None:
    """
    保存DC编码结果到JSON文件
    
    Args:
        dc_encoded: DC编码结果列表
        metadata: 元数据字典
        file_path: 输出文件路径
    """
    try:
        # 转换元组列表为可JSON序列化的格式
        encoded_data = [{
            'bits': bits,
            'value': value
        } for bits, value in dc_encoded]
        
        save_data = {
            'dc_encoded': encoded_data,
        }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存到JSON文件
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        print(f"DC编码结果已保存到: {file_path}")
    except Exception as e:
        print(f"保存DC编码结果失败: {e}")
        raise


def load_dc_encoded(file_path: str) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    从JSON文件加载DC编码结果
    
    Args:
        file_path: 输入文件路径
        
    Returns:
        dc_encoded: DC编码结果列表
        metadata: 元数据字典
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 将JSON数据转换回元组列表
        encoded_data = data['dc_encoded']
        dc_encoded = [(item['bits'], item['value']) for item in encoded_data]
        
        metadata = data['metadata'] if 'metadata' in data else {}
        
        return dc_encoded, metadata
    except Exception as e:
        print(f"加载DC编码结果失败: {e}")
        raise






def main() -> None:
    """
    DC编解码模块的主函数，用于测试功能
    """
    # 创建测试数据 - 模拟量化后的DCT系数块
    test_blocks = []
    # 生成几个测试块，每个块的DC系数有所变化
    base_dc = 100
    for i in range(5):
        block = np.zeros((8, 8), dtype=np.int32)
        # 设置DC系数，模拟相邻块DC系数的相关性
        block[0, 0] = base_dc + i * 5 + (i % 3) * 2
        # 添加一些AC系数
        block[0, 1] = 10 - i
        block[1, 0] = 5 - i
        test_blocks.append(block)
    
    print("原始量化块的DC系数:")
    for i, block in enumerate(test_blocks):
        print(f"块 {i}: DC = {block[0, 0]}")
    
    # 编码DC系数
    print("\n编码DC系数:")
    dc_encoded= encode_dc_coefficients(test_blocks)
    for i, (bits, value) in enumerate(dc_encoded):
        print(f"块 {i}: 位数 = {bits}, 值 = {value}")
    
    print(f"\nDC差值:")
    for i, diff in enumerate(metadata['dc_differences']):
        print(f"块 {i}: 差值 = {diff}")
    
    # 解码DC系数
    print("\n解码DC系数:")
    decoded_dc = decode_dc_coefficients(dc_encoded)
    for i, dc in enumerate(decoded_dc):
        print(f"块 {i}: DC = {dc}")
    
    # 验证解码结果
    original_dc = [block[0, 0] for block in test_blocks]
    is_correct = all(original == decoded for original, decoded in zip(original_dc, decoded_dc))
    print(f"\n解码结果正确性: {'正确' if is_correct else '错误'}")
    
    # 测试分离和重建
    print("\n测试分离DC和AC系数:")
    dc_coeffs, ac_blocks = separate_dc_and_ac(test_blocks)
    print(f"分离出的DC系数数量: {len(dc_coeffs)}")
    print(f"AC块数量: {len(ac_blocks)}")
    print(f"第一个AC块的DC位置值: {ac_blocks[0][0, 0]}")
    
    print("\n测试重建完整块:")
    reconstructed_blocks = reconstruct_blocks_with_dc(ac_blocks, dc_coeffs)
    # 验证重建结果
    is_reconstruction_correct = all(
        np.array_equal(original, reconstructed) 
        for original, reconstructed in zip(test_blocks, reconstructed_blocks)
    )
    print(f"重建结果正确性: {'正确' if is_reconstruction_correct else '错误'}")


if __name__ == "__main__":
    main()