# JPEG 图像压缩与解压缩系统

本项目实现了完整的JPEG图像压缩和解压缩流程，包括色彩空间转换、图像分块、DCT变换、量化、熵编码以及对应的解码过程。

## 项目结构
```
├── main.py # 主程序入口 
├── config.py # 配置文件 
├── compare.py # DC编码解码数据对比工具 
├── value.txt # 临时数据文件 
├── utils/ # 工具模块 
│ ├── image_io.py # 图像输入输出处理 
│ ├── block_processing.py # 图像分块与重组 
│ ├── dct_transform.py # DCT/IDCT变换 
│ ├── quantization.py # 量化与反量化 
│ ├── dc_coding.py # DC系数编码/解码 
│ ├── ac_coding.py # AC系数编码/解码 
│ └── coder.py # 二进制编解码 
├── data/ # 测试图像数据 
│ ├── input.jpg # 示例输入图像 
│ ├── input_1.jpg # 额外测试图像 
│ └── output.jpg # 输出图像 
├── output/ # 输出文件目录 
│ ├── dc_encoded.json # DC编码结果 
│ ├── ac_encoded.json # AC编码结果 
│ ├── dc_decoded.json # DC解码结果 
│ ├── ac_decoded.json # AC解码结果 
│ └── reconstructed_blocks.json # 重建块数据 
└── tests/ # 测试目录（当前为空）
```


## 功能特性

### 编码流程
1. **图像读取**: 读取RGB图像并转换为YCbCr色彩空间
2. **图像分块**: 将图像分割为8x8的块
3. **DCT变换**: 对每个块执行离散余弦变换
4. **量化**: 使用量化表对DCT系数进行量化
5. **系数分离**: 分离DC系数和AC系数
6. **DC编码**: 对DC系数进行DPCM编码
7. **AC编码**: 对AC系数进行游程编码和熵编码
8. **二进制编码**: 将编码结果转换为比特流

### 解码流程
1. **二进制解码**: 从比特流中解析编码数据
2. **AC解码**: 解析AC系数
3. **DC解码**: 解析并重建DC系数
4. **系数重组**: 合并DC和AC系数形成完整块
5. **反量化**: 对量化系数进行反量化处理
6. **IDCT变换**: 执行逆离散余弦变换
7. **图像重组**: 将块重新组合成完整图像
8. **色彩转换**: 将YCbCr转换回RGB色彩空间



## 使用方法
### 安装依赖

```bash
pip install numpy
pip install Pillow
```
### 运行完整流程

```bash
python main.py
```

### 运行数据对比工具

```bash
python compare.py
```

该工具用于分析DC编码和解码数据之间的差异，帮助调试编码解码过程中的问题。

## 配置选项

在 `config.py` 文件中可以调整以下参数:

- `BLOCK_SIZE`: 块大小 (默认: 8)
- `QUALITY_FACTOR`: 质量因子 (默认: 80)
- `QUANTIZATION_MODE`: 量化模式 (默认: 'single')
- `INPUT_IMAGE_PATH`: 输入图像路径
- `OUTPUT_IMAGE_PATH`: 输出图像路径
- `OUTPUT_DIR`: 输出目录

## 输出文件

编码解码过程中会生成以下文件:

- `output/dc_encoded.json`: DC编码结果
- `output/ac_encoded.json`: AC编码结果
- `output/dc_decoded.json`: DC解码结果
- `output/ac_decoded.json`: AC解码结果
- `output/reconstructed_blocks.json`: 重建块数据
- `output/output.jpg`: 输出图像

## 性能评估

程序会计算并输出以下指标:
- PSNR (峰值信噪比): 衡量重建图像质量
- 比特流长度: 压缩后的数据大小

