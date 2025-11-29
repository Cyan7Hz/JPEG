# JPEG编解码配置参数

# 图像处理参数
BLOCK_SIZE = 8  # 分块大小
QUALITY_FACTOR = 80  # 质量因子(1-100)
COMPONENT_TYPE = 'luminance'  # 分量类型: 'luminance'或'chrominance'

# 文件路径配置
INPUT_IMAGE_PATH = 'data/input.jpg'  # 输入图像路径
OUTPUT_IMAGE_PATH = 'data/output.jpg'  # 输出图像路径

# 中间结果保存路径
SAVE_INTERMEDIATE_RESULTS = {
    'blocks': False,
    'dct_coeffs': False,  # 不保存DCT系数
    'quantized_coeffs': False,  # 不保存量化系数
    'dc_encoded': False,
    'ac_encoded': True,
    'dc_decoded': False,
    'ac_decoded': False,
    'dequantized_coeffs': False,
    'reconstructed_blocks': True
}
OUTPUT_DIR = 'output'  # 输出目录
BLOCKS_FILE = 'output/blocks.json'  # 分块结果
DCT_COEFFS_FILE = 'output/dct_coeffs.json'  # DCT系数
QUANTIZED_COEFFS_FILE = 'output/quantized_coeffs.json'  # 量化系数
DC_ENCODED_FILE = 'output/dc_encoded.json'  # DC编码结果
AC_ENCODED_FILE = 'output/ac_encoded.bin'  # AC熵编码结果（二进制）
DC_DECODED_FILE = 'output/dc_decoded.json'  # DC解码结果
AC_DECODED_FILE = 'output/ac_decoded.json'  # AC解码结果
DEQUANTIZED_COEFFS_FILE = 'output/dequantized_coeffs.json'  # 逆量化系数
RECONSTRUCTED_BLOCKS_FILE = 'output/reconstructed_blocks.json'  # 重建块

# 处理配置
NORMALIZE_DCT = True  # DCT变换时是否归一化（减去128）