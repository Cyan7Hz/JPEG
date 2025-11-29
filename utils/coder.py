import sys
from typing import List, Tuple, Dict, Any



# --- 类型定义 ---
# (bits, diff_value)
DC_Encoded_Info = Tuple[int, int] 
# (run_len, ac_value, bits)
AC_RLE_Info = List[Tuple[int, int, int]] 












# PART 0：AC DC 编码表 ----------------------------------------------------------------------
DC_HUFFMAN_TABLE: Dict[Tuple[int], str] = {
    # Size 0
    (0): '00', 
    # Size 1
    (1): '010',
    # Size 2
    (2): '011',
    # Size 3
    (3): '100',
    # Size 4
    (4): '101',
    # Size 5
    (5): '110',
    # Size 6
    (6): '1110',
    # Size 7
    (7): '11110',
    # Size 8
    (8): '111110',
    # Size 9
    (9): '1111110',
    # Size 10
    (10): '11111110',
    # Size 11
    (11): '111111110',
}

REV_DC_HUFFMAN_TABLE = {v: k for k, v in DC_HUFFMAN_TABLE.items()}


AC_HUFFMAN_TABLE: Dict[Tuple[int, int], str] = {
    # RUN=0
    (0, 0): '1010',      # EOB
    (0, 1): '00',
    (0, 2): '01',
    (0, 3): '100',
    (0, 4): '1011',
    (0, 5): '11010',
    (0, 6): '1111000',
    (0, 7): '11111000',
    (0, 8): '1111110110',
    (0, 9): '1111111110000010',
    (0, 10): '1111111110000011', # A = 10
    
    # RUN=1
    (1, 1): '1100',
    (1, 2): '11011',
    (1, 3): '1111001',
    (1, 4): '111110110',
    (1, 5): '11111110110',
    (1, 6): '1111111110000100',
    (1, 7): '1111111110000101',
    (1, 8): '1111111110000110',
    (1, 9): '1111111110000111',
    (1, 10): '1111111110001000', # A = 10
    
    # RUN=2
    (2, 1): '11100',
    (2, 2): '11111001',
    (2, 3): '1111110111',
    (2, 4): '111111110100',
    (2, 5): '111111111001001',
    (2, 6): '111111111001010',
    (2, 7): '111111111001011',
    (2, 8): '111111111001100',
    (2, 9): '111111111001101',
    (2, 10): '111111111001110', # A = 10
    
    # RUN=3
    (3, 1): '111010',
    (3, 2): '111110111',
    (3, 3): '111111110101',
    (3, 4): '1111111110001111',
    (3, 5): '1111111110010000',
    (3, 6): '1111111110010001',
    (3, 7): '1111111110010010',
    (3, 8): '1111111110010011',
    (3, 9): '1111111110010100',
    (3, 10): '1111111110010101', # A = 10
    
    # RUN=4
    (4, 1): '111011',
    (4, 2): '1111111000',
    (4, 3): '1111111110010110',
    (4, 4): '1111111110010111',
    (4, 5): '1111111110011000',
    (4, 6): '1111111110011001',
    (4, 7): '1111111110011010',
    (4, 8): '1111111110011011',
    (4, 9): '1111111110011100',
    (4, 10): '1111111110011101', # A = 10
    
    # RUN=5
    (5, 1): '1111010',
    (5, 2): '11111110111',
    (5, 3): '1111111110011110',
    (5, 4): '1111111110011111',
    (5, 5): '1111111110100000',
    (5, 6): '1111111110100001',
    (5, 7): '1111111110100010',
    (5, 8): '1111111110100011',
    (5, 9): '1111111110100100',
    (5, 10): '1111111110100101', # A = 10
    
    # RUN=6
    (6, 1): '1111011',
    (6, 2): '111111110110',
    (6, 3): '1111111110100110',
    (6, 4): '1111111110100111',
    (6, 5): '1111111110101000',
    (6, 6): '1111111110101001',
    (6, 7): '1111111110101010',
    (6, 8): '1111111110101011',
    (6, 9): '1111111110100100',
    (6, 10): '1111111110100101', # A = 10
    
    # RUN=7
    (7, 1): '11111010',
    (7, 2): '111111110111',
    (7, 3): '1111111110101110',
    (7, 4): '1111111110101111',
    (7, 5): '1111111110110000',
    (7, 6): '1111111110110001',
    (7, 7): '1111111110110010',
    (7, 8): '1111111110110011',
    (7, 9): '1111111110110100',
    (7, 10): '1111111110110101', # A = 10
    
    # RUN=8
    (8, 1): '111111000',
    (8, 2): '111111111000000',
    (8, 3): '1111111110110110',
    (8, 4): '1111111110110111',
    (8, 5): '1111111110111000',
    (8, 6): '1111111110111001',
    (8, 7): '1111111110111010',
    (8, 8): '1111111110111011',
    (8, 9): '1111111110111100',
    (8, 10): '1111111110111101', # A = 10
    
    # RUN=9
    (9, 1): '111111001',
    (9, 2): '1111111110111110',
    (9, 3): '1111111110111111',
    (9, 4): '1111111111000000',
    (9, 5): '1111111111000001',
    (9, 6): '1111111111000010',
    (9, 7): '1111111111000011',
    (9, 8): '1111111111000100',
    (9, 9): '1111111111000101',
    (9, 10): '1111111111000110', # A = 10
    
    # RUN=10 (A)
    (10, 1): '111111010',
    (10, 2): '1111111111000111',
    (10, 3): '1111111111001000',
    (10, 4): '1111111111001001',
    (10, 5): '1111111111001010',
    (10, 6): '1111111111001011',
    (10, 7): '1111111111001100',
    (10, 8): '1111111111001101',
    (10, 9): '1111111111001110',
    (10, 10): '1111111111001111', # A = 10
    
    # RUN=11 (B)
    (11, 1): '1111111001',
    (11, 2): '1111111111010000',
    (11, 3): '1111111111010001',
    (11, 4): '1111111111010010',
    (11, 5): '1111111111010011',
    (11, 6): '1111111111010100',
    (11, 7): '1111111111010101',
    (11, 8): '1111111111010110',
    (11, 9): '1111111111010111',
    (11, 10): '1111111111011000', # A = 10
    
    # RUN=12 (C)
    (12, 1): '1111111010',
    (12, 2): '1111111111011001',
    (12, 3): '1111111111011010',
    (12, 4): '1111111111011011',
    (12, 5): '1111111111011100',
    (12, 6): '1111111111011101',
    (12, 7): '1111111111011110',
    (12, 8): '1111111111011111',
    (12, 9): '1111111111100000',
    (12, 10): '1111111111100001', # A = 10
    
    # RUN=13 (D)
    (13, 1): '11111111000',
    (13, 2): '1111111111100010',
    (13, 3): '1111111111100011',
    (13, 4): '1111111111100100',
    (13, 5): '1111111111100101',
    (13, 6): '1111111111100110',
    (13, 7): '1111111111100111',
    (13, 8): '1111111111101000',
    (13, 9): '1111111111101001',
    (13, 10): '1111111111101010', # A = 10
    
    # RUN=14 (E)
    (14, 1): '1111111111101011',
    (14, 2): '1111111111101100',
    (14, 3): '1111111111101101',
    (14, 4): '1111111111101110',
    (14, 5): '1111111111101111',
    (14, 6): '1111111111110000',
    (14, 7): '1111111111110001',
    (14, 8): '1111111111110010',
    (14, 9): '1111111111110011',
    (14, 10): '1111111111110100', # A = 10
    
    # RUN=15 (F)
    (15, 0): '11111111001',      # F/0 (ZRL - Zero Run Length)
    (15, 1): '1111111111110101',
    (15, 2): '1111111111110110',
    (15, 3): '1111111111110111',
    (15, 4): '1111111111111000',
    (15, 5): '1111111111111001',
    (15, 6): '1111111111111010',
    (15, 7): '1111111111111011',
    (15, 8): '1111111111111100',
    (15, 9): '1111111111111101',
    (15, 10): '1111111111111110', # A = 10
}

REV_AC_HUFFMAN_TABLE = {v: k for k, v in AC_HUFFMAN_TABLE.items()}





















# PART 1：值 符号 编解码基础函数 ------------------------------------------------------

# def get_size_in_bits(value: int) -> int:
#     """计算表示一个整数所需的位数 (Size/Category)。"""
#     if value == 0: return 0
#     return abs(value).bit_length()

def int_to_bitstring(value: int, bits: int) -> str:
    """
    将整数值转换为幅度编码的比特流 (表 4 规则)。
    正值: 标准二进制。负值: V + (2^bits - 1)。
    """
    if bits == 0: return ''
    
    # 保证输出长度为 bits
    format_str = f'0{bits}b'
    
    if value > 0:
        # 正值：标准二进制
        return format(value, format_str)
    else:
        # 负值： raw_val = value + (2^bits - 1)
        # (1 << bits) 是 2^bits
        raw_val = value + ((1 << bits) - 1) 
        return format(raw_val, format_str)

def bitstring_to_int(bitstring: str) -> int:
    """将幅度编码的比特流解码回整数值。"""
    if not bitstring: return 0
    val = int(bitstring, 2)
    bits = len(bitstring)
    
    # 如果 MSB 为 1，它是正值的标准二进制
    if bitstring[0] == '1': 
        return val
    else: 
        # 负值解码规则: V = val - (2^bits - 1)
        return val - ((1 << bits) - 1)
        









# PART 2：行程 位长 Huffman 编解码函数 --------------------------------------------------
def dc_coder(dc_encoded: DC_Encoded_Info) -> str:
    """
    对 DC 系数进行 Huffman 编码。
    返回: 编码后的 DC 比特流
    """
    dc_stream = ""
    for dc_bits, dc_val in dc_encoded:
        dc_huff_key = (dc_bits)
        
        if dc_huff_key not in DC_HUFFMAN_TABLE:
            raise ValueError(f"DC 哈夫曼键 {dc_huff_key} 未定义。")
            
        dc_stream += DC_HUFFMAN_TABLE[dc_huff_key]
        
        if dc_bits > 0:
            dc_stream += int_to_bitstring(dc_val, dc_bits)
    return dc_stream

def ac_coder(ac_rle: AC_RLE_Info) -> str:
    """
    对 AC 系数进行行程编码和 Huffman 编码。
    返回: 编码后的 AC 比特流
    """
    ac_stream = ""
    for run_len, val, bits in ac_rle:
        ac_huff_key = (run_len, bits)
        
        if ac_huff_key not in AC_HUFFMAN_TABLE:
            raise ValueError(f"AC 哈夫曼键 {ac_huff_key} 未定义。")
            
        ac_stream += AC_HUFFMAN_TABLE[ac_huff_key]
        
        if bits > 0:
            ac_stream += int_to_bitstring(val, bits)
            
        if ac_huff_key == (0, 0): # EOB
            break
            
    return ac_stream





















# PART 3：任务分发模块 -------------------------------------------------------------

def encode_and_merge_blocks(
    dc_info_list: List[DC_Encoded_Info], 
    ac_rle_list: List[List[AC_RLE_Info]]
) -> str:
    """
    将 DC 和 AC 编码信息按块交错编码并合并成一个完整的比特流。
    """
    if len(dc_info_list) != len(ac_rle_list):
        raise ValueError("DC 和 AC 块的数量必须匹配。")
        
    final_bitstream = ""
    
    for i in range(len(dc_info_list)):
        # 1. 编码 DC
        dc_stream = dc_coder([dc_info_list[i]])
            
        # 2. 编码 AC
        ac_stream = ac_coder(ac_rle_list[i])
                
        # 3. 合并: DC 紧接 AC
        final_bitstream += dc_stream
        final_bitstream += ac_stream
        
    return final_bitstream

def huffman_decode_next(bitstream: str, pos: int, rev_table: Dict[str, Tuple[int, int]]) -> Tuple[Tuple[int, int], int, int]:
    """
    从比特流中查找下一个哈夫曼码字。
    返回: (key, code_len, pos_next)
    """
    # 查找最长码字长度
    max_len = max(len(code) for code in rev_table) if rev_table else 16
    
    for l in range(1, max_len + 1):
        if pos + l > len(bitstream):
            break
        code = bitstream[pos : pos + l]
        if code in rev_table:
            key = rev_table[code]
            return key, l, pos + l
            
    raise ValueError(f"在位置 {pos} 无法找到有效的哈夫曼码字。")

def decode_and_separate_blocks(bitstream: str) -> Tuple[List[DC_Encoded_Info], List[List[AC_RLE_Info]]]:
    """
    解码并分离出 DC 和 AC 编码信息。
    """
    pos = 0
    dc_decoded_list: List[DC_Decoded_Info] = []
    ac_decoded_list: List[List[AC_RLE_Info]] = []
    
    while pos < len(bitstream):
        
        # --- 1. 解码 DC ---
        key, code_len, pos = huffman_decode_next(bitstream, pos, REV_DC_HUFFMAN_TABLE)
        
        if key[0] != 0:
            raise ValueError(f"解码错误：块必须以 DC 编码开始，在位置 {pos - code_len} 找到了 AC 键 {key}。")
            
        dc_size = key[1]
        dc_value = 0
        
        # 解码 DC Amplitude
        if dc_size > 0:
            dc_value = bitstring_to_int(bitstream[pos : pos + dc_size])
            pos += dc_size
            
        dc_decoded_list.append((dc_size, dc_value))
        
        # --- 2. 解码 AC ---
        ac_block_rle: List[AC_RLE_Info] = []
        
        while True:
            key, code_len, pos = huffman_decode_next(bitstream, pos, REV_HUFFMAN_TABLE)
            
            run_len, ac_size = key
            
            # EOB 检查
            if key == (0, 0):
                ac_block_rle.append((0, 0, 0)) # EOB 结构
                break
            
            ac_value = 0
            # 解码 AC Amplitude
            if ac_size > 0:
                ac_value = bitstring_to_int(bitstream[pos : pos + ac_size])
                pos += ac_size
                
            ac_block_rle.append((run_len, ac_value, ac_size))
            
        ac_decoded_list.append(ac_block_rle)
        
    return dc_decoded_list, ac_decoded_list


































if __name__ == '__main__':
    # --- 编码数据示例 ---
    
    # 假设 DC DPCM 差值信息：[(Size, Diff_Value), ...]
    DC_INFO: List[DC_Encoded_Info] = [
        (7, 100),   # 块 0: 100
        (3, -5),    # 块 1: -5
        (0, 0)      # 块 2: 0 (无变化)
    ]
    
    # 假设 AC RLE 编码信息：[ [(Run, Value, Size), ...], ...]
    AC_RLE: List[List[AC_RLE_Info]] = [
        # 块 0
        [(0, 5, 3), (1, -10, 4), (0, 0, 0)],
        # 块 1
        [(12, 1, 3), (0, 0, 0)], 
        # 块 2 (两个 ZRL 示例 + EOB)
        [(15, 0, 0), (15, 0, 0), (0, 0, 0)]
    ]

    print("--- JPEG 熵编解码模块验证 ---")
    
    # 1. 编码和合并
    merged_bitstream = encode_and_merge_blocks(DC_INFO, AC_RLE)
    
    print("\n✅ 编码和合并完成。")
    print(f"   合并比特流长度: {len(merged_bitstream)}")
    print(f"   比特流片段: {merged_bitstream[:40]}...")

    # 2. 解码和分离
    try:
        decoded_dc, decoded_ac = decode_and_separate_blocks(merged_bitstream)

        # 3. 验证数据是否一致
        print("\n--- 解码结果验证 ---")
        
        # DC 验证
        dc_match = DC_INFO == decoded_dc
        print(f"DC 列表匹配: {dc_match}")
        if not dc_match:
            print("  原始 DC:", DC_INFO)
            print("  解码 DC:", decoded_dc)
            
        # AC 验证
        ac_match = AC_RLE == decoded_ac
        print(f"AC 列表匹配: {ac_match}")
        if not ac_match:
            print("  原始 AC (块 0):", AC_RLE[0])
            print("  解码 AC (块 0):", decoded_ac[0])
            
        # 最终断言
        assert DC_INFO == decoded_dc
        assert AC_RLE == decoded_ac
        print("\n🎉 编码、合并和解码循环验证成功！")

    except ValueError as e:
        print(f"\n❌ 致命错误：解码失败。{e}")