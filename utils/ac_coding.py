import numpy as np
import json
import os
import struct
from typing import List, Tuple, Dict, Any, Optional

# PART 1：基础配置 -------------------------------------------------------------

ZIGZAG_ORDER = [
    (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
    (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
    (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
    (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
    (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
    (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
    (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
]


# PART 2：数值编解码组件 -------------------------------------------------------------

def get_size_in_bits(value: int) -> int:
    if value == 0: return 0
    return len(bin(abs(value))) - 2

def int_to_bitstring(value: int, bits: int) -> str:
    if bits == 0: return ''
    if value > 0: return bin(value)[2:].zfill(bits)
    else: return bin(value + ((1 << bits) - 1))[2:].zfill(bits)

def bitstring_to_int(bitstring: str) -> int:
    if not bitstring: return 0
    val = int(bitstring, 2)
    if bitstring[0] == '1': return val
    else: return val - ((1 << len(bitstring)) - 1)

# PART 3：扫描与RLE组件 -------------------------------------------------------------

def zigzag_scan(block: np.ndarray) -> List[int]:
    return [int(block[i, j]) for i, j in ZIGZAG_ORDER]

def inverse_zigzag_scan(scan_result: List[int]) -> np.ndarray:
    block = np.zeros((8, 8), dtype=np.int32)
    for idx, (i, j) in enumerate(ZIGZAG_ORDER):
        block[i, j] = scan_result[idx]
    return block

def run_length_encode(zigzag_scan: List[int]) -> List[Tuple[int, int, int]]:
    rle_result = []
    if not zigzag_scan: return [(0, 0, 0)]

    i = 0
    while i < len(zigzag_scan):
        run_len = 0
        curr = i
        while curr < len(zigzag_scan) and zigzag_scan[curr] == 0:
            run_len += 1
            curr += 1
            
        if curr >= len(zigzag_scan): break
        
        while run_len >= 16:
            rle_result.append((15, 0, 0))
            run_len -= 16
            
        val = zigzag_scan[curr]
        rle_result.append((run_len, val, get_size_in_bits(val)))
        i = curr + 1
    if i < len(zigzag_scan):
        rle_result.append((0, 0, 0))
    return rle_result

def run_length_decode(rle_result: List[Tuple[int, int, int]]) -> List[int]:
    """
    修正后的解码函数：正确处理 ZRL
    """
    zigzag = []
    if not rle_result: return [0]*63
    
    # AC
    for run_len, val, _ in rle_result[0:]:
        if run_len == 0 and val == 0: break # EOB
        
        # 添加连续的零
        zigzag.extend([0] * run_len)
        
        # 关键修正：必须添加 val，即使 val 是 0 (针对 ZRL)
        # ZRL (15, 0) -> 添加 15个0，然后添加 1个0 = 16个0
        zigzag.append(val)
            
    if len(zigzag) < 63:
        zigzag.extend([0] * (63 - len(zigzag)))
    elif len(zigzag) > 63: # 安全性截断
        zigzag = zigzag[:63]
        
    return zigzag





















# PART 4：任务分发组件 -------------------------------------------------------------
def encode_ac_coefficients(quantized_blocks: List[np.ndarray]) -> List[str]:
    """
    对AC系数进行编码
    
    Args:
        ac_blocks: AC系数块列表
        
    Returns:
        ac_encoded: AC编码结果列表，每个元素是一个位串
    """
    ac_encoded = []
    for block_idx, block in enumerate(quantized_blocks):
        # Z字形扫描
        zigzag = zigzag_scan(block)
        # 游程编码
        rle = run_length_encode(zigzag)
        # 注意：熵编码已移至coder.py模块
        ac_encoded.append(rle)
    return ac_encoded

import numpy as np
from typing import List, Tuple, Any, Callable

# --- 假设外部函数签名 (您需要在本地定义) ---
# run_length_decode 必须能正确处理 EOB 并返回 63个元素的 AC 序列
# run_length_decode(ac_rle: List[AC_RLE_Info]) -> np.ndarray (长度为 63)
# inverse_zigzag_scan(zigzag_sequence: np.ndarray) -> np.ndarray (8x8)
# DC_Decoded_Info = Tuple[int, int] 
# AC_RLE_Info = Tuple[int, int, int] 

# --- 辅助函数定义占位 (请替换为您自己的实现) ---
# def run_length_decode(ac_rle_list: List[AC_RLE_Info]) -> np.ndarray:
#     # 必须实现将 RLE 转换为 63 个 AC 系数的逻辑 (包括 ZRL 和 EOB 填充)
#     pass

# def inverse_zigzag_scan(zigzag_sequence: np.ndarray) -> np.ndarray:
#     # 必须实现将 64 个元素序列转换为 8x8 块的逻辑
#     pass

# --- 修正后的 decode_ac2blocks 函数 ---

def decode_ac2blocks(
    ac_rle_list: List[List[Tuple[int, int, int]]]
) -> List[np.ndarray]:
    """
    从 AC 的 RLE 编码信息重构 8x8 块列表，直流分量设置为 0。
    
    Args:
        ac_rle_list: AC RLE 编码信息列表，每个元素为 RLE 元组列表 [(run_len, value, size), ...]
        run_length_decode: 将 RLE 列表转换为 63 个 AC 系数序列的函数。
        inverse_zigzag_scan: 将 64 个元素序列转换回 8x8 块的函数。
        
    Returns:
        blocks: 重构的 8x8 块列表，每个块为 np.ndarray，直流分量为 0。
    """
    
    blocks = []
    
    # 遍历每个块的 AC 信息
    for ac_rle in ac_rle_list:
        
        # 1. RLE 解码 AC 系数
        # 假设 run_length_decode 返回一个长度为 63 的 numpy 数组，包含 AC 系数
        ac_zigzag_coeffs = run_length_decode(ac_rle)
        
        if len(ac_zigzag_coeffs) != 63:
             raise ValueError(f"run_length_decode 必须返回 63 个 AC 系数，实际返回 {len(ac_zigzag_coeffs)}")
        
        # 2. 构造完整的 64 元素 Zigzag 序列
        # 将 DC 分量 (强制为 0) 拼接到 AC 序列之前
        # np.insert 效率更高，但为了简单，这里使用 np.concatenate
        # 3. 逆 Zigzag 扫描
        block = inverse_zigzag_scan(ac_zigzag_coeffs)
        
        blocks.append(block)
    
    return blocks
























def save_ac_encoded(ac_encoded: List[Tuple[int, int]], file_path: str) -> None:
    """
    保存AC编码结果到JSON文件
    
    Args:
        ac_encoded: AC编码结果列表
        file_path: 输出文件路径
    """
    try:
        # 转换元组列表为可JSON序列化的格式
        encoded_data = []
        for rle_list in ac_encoded:
            block_data = [{
                'rle_tuple': (run_len, value, bits),
            } for run_len, value, bits in rle_list]
            encoded_data.append(block_data)
        
        save_data = {
            'ac_encoded': encoded_data,
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





def main():
    test_block = np.zeros((8, 8), dtype=np.int32)
    test_block[0, 0] = 100
    test_block[0, 1] = 15
    test_block[1, 0] = -10
    test_block[7, 7] = 5

    print("--- 原始数据 (部分) ---")
    print(test_block[:2])
    
    zigzag = zigzag_scan(test_block)
    rle = run_length_encode(zigzag)
    print(f"\n--- RLE编码结果 ---")
    print(rle[:5])  # 只打印前5个元素
    
    decoded_zigzag = run_length_decode(rle)
    recon_block = inverse_zigzag_scan(decoded_zigzag)
    
    print("\n--- 结果验证 ---")
    print(f"原始 DC: {test_block[0,0]}, 解码 DC: {recon_block[0,0]}")
    print(f"原始 最后AC: {test_block[7,7]}, 解码 最后AC: {recon_block[7,7]}")
    
    is_correct = np.array_equal(test_block, recon_block)
    print(f"完全一致: {'✅ 是' if is_correct else '❌ 否'}")

if __name__ == "__main__":
    main()