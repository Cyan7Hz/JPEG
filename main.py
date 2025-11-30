import os
import numpy as np
from utils import image_io, block_processing, dct_transform, quantization, dc_coding, ac_coding, coder
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
        'quality_factor': config.QUALITY_FACTOR,
        'quan_mode': config.QUANTIZATION_MODE,
        'quan_tables': [],
        'bit_stream': ''
    }
    
    # 2. 图像分块
    print("图像分块...")
    blocks = block_processing.split_image_into_blocks(image, config.BLOCK_SIZE)
    
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
    metadata['quan_tables'] = [table.tolist() for table in quant_tables]
    
    # 保存量化系数（根据配置）
    if config.SAVE_INTERMEDIATE_RESULTS['quantized_coeffs']:
        # 根据量化模式保存相应的量化表
        quantization.save_quantized_coefficients(
            quantized_blocks, quant_tables, metadata, config.QUANTIZED_COEFFS_FILE
        )
    print(f"量化表已保存到: {config.QUANTIZED_COEFFS_FILE}")
    
    # # 5. 分离DC和AC系数
    # print("分离DC和AC系数...")
    # dc_coefficients, ac_blocks = dc_coding.separate_dc_and_ac(quantized_blocks)
    
    # 6. DC系数编码
    print("编码DC系数...")
    dc_encoded = dc_coding.encode_dc_coefficients(quantized_blocks)
    
    # 保存DC编码结果（根据配置）
    if config.SAVE_INTERMEDIATE_RESULTS['dc_encoded']:
        dc_coding.save_dc_encoded(dc_encoded, config.DC_ENCODED_FILE)
        print(f"DC编码结果已保存到: {config.DC_ENCODED_FILE}")
    
    # 7. AC系数熵编码
    print("对AC系数进行熵编码...")
    ac_encoded = ac_coding.encode_ac_coefficients(quantized_blocks)
    
    # 保存AC熵编码结果（根据配置）
    if config.SAVE_INTERMEDIATE_RESULTS['ac_encoded']:
        ac_coding.save_ac_encoded(ac_encoded, config.AC_ENCODED_FILE)
        print(f"AC熵编码结果已保存到: {config.AC_ENCODED_FILE}")
    
    # 8. ACDC 二进制编码
    print("对DC,AC系数进行二进制编码...")
    final_bitstream = coder.encode_acdc2bits(dc_encoded, ac_encoded)
    metadata['bit_stream'] = final_bitstream
    print(f"最终比特流长度: {len(final_bitstream)}")

# =======================================================================
    # 9. 新增：计算压缩比
    # =======================================================================
    
    NUM_COEFFS_PER_BLOCK = 64
    DC_COEFF_BIT_DEPTH = 26  # 假设熵编码前每个tuple用16位存储
    AC_COEFF_BIT_DEPTH = 16  # 假设熵编码前每个tuple用16位存储
    AMP_BIT_DEPTH = 8  # 假设熵编码前每个像素点用24位存储

    num_blocks = len(quantized_blocks)
    
    print("\n--- 压缩比分析 ---")
    
    if len(final_bitstream) > 0:
        
        # --- 1. 系数级压缩比 (熵编码效率) ---
        size_init = num_blocks * NUM_COEFFS_PER_BLOCK * (AMP_BIT_DEPTH)
        cr_all = size_init / len(final_bitstream)

        size_pre_entropy = num_blocks * (DC_COEFF_BIT_DEPTH + (NUM_COEFFS_PER_BLOCK-1) * AC_COEFF_BIT_DEPTH)
        cr_entropy = size_pre_entropy / len(final_bitstream)

        print(f"1. 熵编码前存储 block list 的理论大小(16bit/dc_coeff, 16bit/ac_coeff): {size_pre_entropy} bits")
        print(f"   => 无损编码单步压缩比 (CR_Entropy): {cr_entropy:.4f}")
        # --- 2. 图像级近似压缩比 (总体效果) ---
        size_original_image = image_height * image_width * image_channels * 8 # 8bit/pixel
        cr_image_approx = size_original_image / len(final_bitstream)
        
        print(f"2. 原始图像像素数据大小 (8bit/pixel): {size_original_image} bits")
        print(f"   => 图像级近似压缩比 (CR_Image): {cr_image_approx:.4f}")
        
    else:
        print("最终比特流长度为零，无法计算压缩比。")
        
    # =======================================================================

    print("=== 图像编码完成 ===")
    return metadata


def decode_image(metadata: dict) -> np.ndarray:
    """图像解码流程"""
    print("\n=== 开始图像解码 ===")
    # 1. 提取比特流
    bitstream = metadata.get('bit_stream')
    print(f"比特流长度: {len(bitstream)}")
    if not bitstream:
        raise ValueError("Metadata 中缺少 'bit_stream' 字段或比特流为空。")
    
    # 2. 二进制解码
    print("对DC,AC系数进行二进制解码...")
    dc_decoded, ac_decoded = coder.decode_bits2acdc(bitstream)


    if config.SAVE_INTERMEDIATE_RESULTS['ac_decoded']:
        ac_coding.save_ac_encoded(ac_decoded, config.AC_DECODED_FILE)
        print(f"AC解码结果已保存到: {config.AC_DECODED_FILE}")

    if config.SAVE_INTERMEDIATE_RESULTS['dc_decoded']:
        dc_coding.save_dc_encoded(dc_decoded, config.DC_DECODED_FILE)
        print(f"DC解码结果已保存到: {config.DC_DECODED_FILE}")
    
    # 3. 重建完整的DCT块（合并DC和AC）
    print("重建DCT块...")
    ac_blocks = ac_coding.decode_ac2blocks(ac_decoded)
    reconstructed_blocks = dc_coding.decode_dc2blocks(
        ac_blocks, dc_decoded  
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
    quant_tables = [np.array(table, dtype=np.float64) for table in metadata['quan_tables']]
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
    reconstructed_image = block_processing.reconstruct_image_from_blocks(
        idct_blocks, metadata
    )
    
    # 7. 将 YCbCr 转换为 RGB
    print("将 YCbCr 转换为 RGB...")
    reconstructed_image = image_io.ycrcb_to_rgb(reconstructed_image)
    
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
    
    # 确保两个图像具有相同的色彩空间
    # 如果重建图像是灰度图像，将原始图像也转换为灰度图像
    if reconstructed_image.ndim == 2:
        if original_image.ndim == 3:
            original_image = image_io.convert_to_grayscale(original_image)
    # 如果重建图像是彩色图像，保持原始图像为彩色图像
    # （因为read_image已经将其转换为RGB格式）
    
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