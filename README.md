# JPEG静图像压缩实验项目

## 项目概述
本项目实现了基于Python的JPEG静图像压缩编解码系统，通过模块化设计展示了JPEG压缩的完整流程，包括图像分块、2D DCT变换、量化、熵编码等关键步骤，以及对应的解码过程。

## 项目结构
```
├── README.md # 项目说明文档 
├── requirements.txt # 依赖库列表 
├── script/ # 执行脚本目录 
├── utils/ # 核心功能模块 
├── data/ # 图像数据目录 
├── temp/ # 中间结果目录 
└── tests/ # 测试文件目录
```

### utils目录（核心模块）
- `config.py`：配置参数（量化矩阵等）
- `image_io.py`：图像读写与预处理
- `block_processing.py`：图像分块与块重组
- `dct_transform.py`：2D DCT正变换
- `idct_transform.py`：2D IDCT反变换
- `quantization.py`：量化处理
- `dequantization.py`：反量化处理
- `dc_encoding.py`：DC系数编码
- `dc_decoding.py`：DC系数解码
- `entropy_encoding.py`：熵编码
- `entropy_decoding.py`：熵解码
- `helpers.py`：辅助函数

### script目录（执行脚本）
- `run_encoding.sh`：完整编码流程
- `run_decoding.sh`：完整解码流程
- `step0_read.sh`：读取图像
- `step1_dct.sh`：单独执行DCT变换
- `step2_quantize.sh`：单独执行量化
- `step3_dc_encode.sh`：单独执行DC编码
- `step4_entropy_encode.sh`：单独执行熵编码
- `step5_entropy_decode.sh`：单独执行熵解码
- `step6_dc_decode.sh`：单独执行DC解码
- `step7_dequantize.sh`：单独执行反量化
- `step8_idct.sh`：单独执行IDCT变换

### temp目录（中间结果）
- `image_blocks.json`：分块后的图像数据
- `dct_coefficients.json`：DCT变换系数
- `quantized_coeffs.json`：量化后的系数
- `dc_encoded.json`：DC编码结果
- `entropy_encoded.bin`：熵编码二进制文件
- `entropy_decoded.json`：熵解码结果
- `dc_decoded.json`：DC解码结果
- `dequantized_coeffs.json`：反量化后的系数
- `reconstructed_blocks.json`：重建的图像块

## 安装与配置

1. 确保已安装Python 3.6+
2. 安装项目依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 设置脚本执行权限（Linux/Mac）：
   ```bash
   chmod +x script/*.sh
   ```

## 使用方法

### 完整编码流程
```bash
./script/run_encoding.sh
```

### 完整解码流程
```bash
./script/run_decoding.sh
```

### 单独执行各个步骤
```bash
bash ./script/step0_read.sh
bash ./script/step1_dct.sh
bash ./script/step2_quantize.sh
```

### 环境设置（首次使用）
```bash
./script/setup.sh
```

## 各模块详细说明

### 1. 图像输入/输出模块 (image_io.py)
- `read_image()`：读取图像并转换为RGB或灰度格式
- `save_image()`：保存处理后的图像
- `preprocess_image()`：图像预处理（尺寸调整、归一化等）

### 2. 块处理模块 (block_processing.py)
- `image_to_blocks()`：将图像分割为8×8块
- `blocks_to_image()`：将8×8块重组为完整图像

### 3. DCT变换模块 (dct_transform.py)
- `compute_dct()`：对图像块执行2D DCT变换
- `compute_dct_fast()`：快速DCT算法实现

### 4. 量化模块 (quantization.py)
- `quantize()`：使用标准JPEG量化矩阵进行量化
- `get_quantization_matrix()`：获取质量因子对应的量化矩阵

### 5. DC编码模块 (dc_encoding.py)
- `dc_encode()`：对DC系数进行差分编码
- `predict_dc()`：DC系数预测

### 6. 熵编码模块 (entropy_encoding.py)
- `entropy_encode()`：对系数进行游程编码和Huffman编码
- `run_length_encode()`：游程编码实现

## 中间文件格式说明

- **JSON格式文件**：存储数组数据，便于查看和调试
  - 图像块、DCT系数、量化结果等
- **二进制文件**：存储最终压缩结果
  - entropy_encoded.bin：熵编码后的二进制压缩数据

## 测试方法

1. 准备测试图像：将测试图像放入`data/input.jpg`
2. 运行完整编解码流程：
   ```bash
   ./script/run_encoding.sh
   ./script/run_decoding.sh
   ```
3. 比较原始图像和重建图像：检查`data/output.jpg`

## 实验扩展建议

1. 实现不同质量因子下的压缩效果对比
2. 测试不同DCT算法（如FFT-based DCT）的性能差异
3. 探索不同熵编码方法（如算术编码）的压缩效率
4. 实现彩色图像的YCbCr颜色空间转换

## 注意事项

1. 本实验项目使用有损压缩方法，重建图像与原始图像会有一定差异
2. 图像尺寸最好是8的倍数，否则会进行填充处理
3. 处理大图像时可能需要较长计算时间
4. 中间结果文件用于调试，实际应用中可优化存储流程

## 参考资料

1. JPEG标准文档（ISO/IEC 10918）
2. 数字图像处理教材
3. 离散余弦变换理论与应用