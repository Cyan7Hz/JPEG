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

SIMPLE_HUFFMAN_TABLE = {
    (0, 0):  '00',          # EOB
    (0, 1):  '010',
    (0, 2):  '011',
    (0, 3):  '100',
    (0, 4):  '101',         # 用于 AC 15, -10
    (0, 5):  '1100',
    (0, 6):  '11010',
    (0, 7):  '11011',       # 用于 DC 100
    (1, 1):  '11100',
    (1, 2):  '11101',
    (12, 3): '11110',       # 用于 AC 5
    (15, 0): '111110',      # ZRL
    'default': '1111111'
}

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

    dc_val = zigzag_scan[0]
    rle_result.append((0, dc_val, get_size_in_bits(dc_val)))
    
    i = 1
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
        
    rle_result.append((0, 0, 0))
    return rle_result

def run_length_decode(rle_result: List[Tuple[int, int, int]]) -> List[int]:
    """
    修正后的解码函数：正确处理 ZRL
    """
    zigzag = []
    if not rle_result: return [0]*64
    
    # DC
    zigzag.append(rle_result[0][1])
    
    # AC
    for run_len, val, _ in rle_result[1:]:
        if run_len == 0 and val == 0: break # EOB
        
        # 添加连续的零
        zigzag.extend([0] * run_len)
        
        # 关键修正：必须添加 val，即使 val 是 0 (针对 ZRL)
        # ZRL (15, 0) -> 添加 15个0，然后添加 1个0 = 16个0
        zigzag.append(val)
            
    if len(zigzag) < 64:
        zigzag.extend([0] * (64 - len(zigzag)))
    elif len(zigzag) > 64: # 安全性截断
        zigzag = zigzag[:64]
        
    return zigzag

# PART 4：熵编解码组件 -------------------------------------------------------------
# 注意：熵编码函数已移至coder.py模块

# PART 5：任务分发组件 -------------------------------------------------------------
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