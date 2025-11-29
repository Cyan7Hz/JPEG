import os
import numpy as np
from utils import image_io, block_processing, dct_transform, quantization, dc_coding, entropy_coding
import config


def encode_image(image_path: str) -> dict:
    """图像编码流程"""

    print("=== 开始图像编码 ===")
    
    # 确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 1. 读取图像，转换为YCrCb色彩空间
    print(f"读取图像: {image_path}")
    image = image_io.read_image(image_path)
    image = image_io.rgb_to_ycrcb(image)
    image_height, image_width, image_channels = image.shape
    
    # 图像元数据
    metadata = {
        'height': image_height,
        'width': image_width,
        'channels': image_channels,
        'block_size': config.BLOCK_SIZE,
        'quality_factor': config.QUALITY_FACTOR
    }
    
    # 2. 图像分块
    print("图像分块...")
    blocks, pad_info = block_processing.split_image_into_blocks(image, config.BLOCK_SIZE)
    metadata['pad_info'] = pad_info
    
    # 保存分块结果（根据配置）
    if config.SAVE_INTERMEDIATE_RESULTS['blocks']:
        image_io.save_to_json({
            'blocks': [block.tolist() for block in blocks],
            'metadata': metadata
        }, config.BLOCKS_FILE)
        print(f"分块结果已保存到: {config.BLOCKS_FILE}")
    
    # 3. DCT变换
    print("执行DCT变换...")
    dct_blocks = dct_transform.perform_dct_on_blocks(blocks, normalize=config.NORMALIZE_DCT)
    
    # 保存DCT系数（根据配置）
    if config.SAVE_INTERMEDIATE_RESULTS['dct_coeffs']:
        dct_transform.save_dct_coefficients_to_json(dct_blocks, metadata, config.DCT_COEFFS_FILE)
        print(f"DCT系数已保存到: {config.DCT_COEFFS_FILE}")
    
        # 4. 量化
    print("执行量化...")
    quantized_blocks, quant_tables = quantization.quantize_blocks(
        dct_blocks, config.QUALITY_FACTOR, config.QUANTIZATION_MODE
    )
    metadata['quantization_tables'] = [table.tolist() for table in quant_tables]
    
    # 保存量化系数（根据配置）
    if config.SAVE_INTERMEDIATE_RESULTS['quantized_coeffs']:
        # 根据量化模式保存相应的量化表
        quantization.save_quantized_coefficients(
            quantized_blocks, quant_tables, metadata, config.QUANTIZED_COEFFS_FILE
        )
    print(f"量化表已保存到: {config.QUANTIZED_COEFFS_FILE}")
    
    # 5. 分离DC和AC系数
    print("分离DC和AC系数...")
    dc_coefficients, ac_blocks = dc_coding.separate_dc_and_ac(quantized_blocks)
    
    # 6. DC系数编码
    print("编码DC系数...")
    dc_encoded, dc_metadata = dc_coding.encode_dc_coefficients(quantized_blocks)
    
    # 保存DC编码结果（根据配置）
    if config.SAVE_INTERMEDIATE_RESULTS['dc_encoded']:
        dc_coding.save_dc_encoded(dc_encoded, dc_metadata, config.DC_ENCODED_FILE)
        print(f"DC编码结果已保存到: {config.DC_ENCODED_FILE}")
    
    # 7. AC系数熵编码
    print("对AC系数进行熵编码...")
    encoded_ac_data = []
    for block_idx, ac_block in enumerate(ac_blocks):
        # Z字形扫描
        zigzag = entropy_coding.zigzag_scan(ac_block)
        # 游程编码
        rle = entropy_coding.run_length_encode(zigzag)
        # 熵编码
        encoded_bits = entropy_coding.entropy_encode(rle)
        encoded_ac_data.append(encoded_bits)
    
    # 保存AC熵编码结果（根据配置）
    if config.SAVE_INTERMEDIATE_RESULTS['ac_encoded']:
        with open(config.AC_ENCODED_FILE, 'w') as f:
            # 为了简化，这里以文本形式保存位串
            for bitstring in encoded_ac_data:
                f.write(bitstring + '\n')
        print(f"AC熵编码结果已保存到: {config.AC_ENCODED_FILE}")
    
    print("=== 图像编码完成 ===")
    return metadata


def decode_image(metadata: dict) -> np.ndarray:
    """图像解码流程"""
    print("\n=== 开始图像解码 ===")
    
    # 1. 读取DC编码结果并解码
    print("解码DC系数...")
    dc_encoded, _ = dc_coding.load_dc_encoded(config.DC_ENCODED_FILE)
    dc_coefficients = dc_coding.decode_dc_coefficients(dc_encoded)
    
    # 保存DC解码结果（根据配置）
    if config.SAVE_INTERMEDIATE_RESULTS['dc_decoded']:
        dc_coding.save_dc_decoded(dc_coefficients, {}, config.DC_DECODED_FILE)
        print(f"DC解码结果已保存到: {config.DC_DECODED_FILE}")
    
    # 2. 读取AC熵编码结果并解码
    print("解码AC系数...")
    decoded_ac_blocks = []
    
    # 读取编码的AC数据
    with open(config.AC_ENCODED_FILE, 'r') as f:
        encoded_ac_data = [line.strip() for line in f]
    
    for bitstring in encoded_ac_data:
        # 熵解码
        decoded_rle = entropy_coding.entropy_decode(bitstring)
        # 游程解码
        decoded_zigzag = entropy_coding.run_length_decode(decoded_rle)
        # 逆Z字形扫描
        ac_block = entropy_coding.inverse_zigzag_scan(decoded_zigzag)
        decoded_ac_blocks.append(ac_block)
    
    # 保存AC解码结果（根据配置）
    if config.SAVE_INTERMEDIATE_RESULTS['ac_decoded']:
        image_io.save_to_json({
            'ac_blocks': [block.tolist() for block in decoded_ac_blocks],
            'metadata': metadata
        }, config.AC_DECODED_FILE)
        print(f"AC解码结果已保存到: {config.AC_DECODED_FILE}")
    
    # 3. 重建完整的DCT块（合并DC和AC）
    print("重建DCT块...")
    reconstructed_blocks = dc_coding.reconstruct_blocks_with_dc(
        decoded_ac_blocks, dc_coefficients
    )
    
    # 保存重建块（根据配置）
    if config.SAVE_INTERMEDIATE_RESULTS['reconstructed_blocks']:
        image_io.save_to_json({
            'blocks': [block.tolist() for block in reconstructed_blocks],
            'metadata': metadata
        }, config.RECONSTRUCTED_BLOCKS_FILE)
        print(f"重建块已保存到: {config.RECONSTRUCTED_BLOCKS_FILE}")
    
    # 4. 逆量化
    print("执行逆量化...")
    quant_tables = [np.array(table, dtype=np.float64) for table in metadata['quantization_tables']]
    # 获取通道数，从分块信息中推断
    channels = 3  # 默认为3通道彩色图像
    if 'channels' in metadata:
        channels = metadata['channels']
    dequantized_blocks = quantization.dequantize_blocks(
        reconstructed_blocks, quant_tables, channels
    )
    
    # 5. IDCT变换
    print("执行IDCT变换...")
    idct_blocks = dct_transform.perform_idct_on_blocks(
        dequantized_blocks, normalize=config.NORMALIZE_DCT
    )
    
    # 6. 重建图像
    print("重建图像...")
    image_height = metadata['height']
    image_width = metadata['width']
    pad_info = metadata['pad_info']
    
    reconstructed_image = block_processing.reconstruct_image_from_blocks(
        idct_blocks, image_height, image_width, config.BLOCK_SIZE, pad_info
    )
    
    print("=== 图像解码完成 ===")
    return reconstructed_image


def main():
    """主函数"""
    # 编码图像
    metadata = encode_image(config.INPUT_IMAGE_PATH)
    
    # 解码图像
    reconstructed_image = decode_image(metadata)
    
    # 保存重建的图像
    image_io.save_image(reconstructed_image, config.OUTPUT_IMAGE_PATH)
    print(f"\n重建图像已保存到: {config.OUTPUT_IMAGE_PATH}")
    
    # 计算PSNR（峰值信噪比）评估重建质量
    original_image = image_io.read_image(config.INPUT_IMAGE_PATH)
    if original_image.ndim == 3:
        original_image = image_io.convert_to_grayscale(original_image)
    
    # 确保尺寸一致
    if original_image.shape != reconstructed_image.shape:
        # 如果不一致，截取匹配的部分
        min_height = min(original_image.shape[0], reconstructed_image.shape[0])
        min_width = min(original_image.shape[1], reconstructed_image.shape[1])
        original_image = original_image[:min_height, :min_width]
        reconstructed_image = reconstructed_image[:min_height, :min_width]
    
    # 计算MSE
    mse = np.mean((original_image.astype(np.float64) - reconstructed_image.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255.0 ** 2 / mse)
    
    print(f"PSNR: {psnr:.2f} dB")


if __name__ == "__main__":
    main()