# RayDP源码分析电子书

本项目用于生成RayDP源码分析的电子书，支持PDF和EPUB格式。

## 目录结构

```
raydp-book/
├── generate_book.py    # 电子书生成脚本
├── README.md           # 项目说明文档
├── raydp-source-code-analysis.pdf  # 生成的PDF版本
└── raydp-source-code-analysis.epub  # 生成的EPUB版本
```

## 依赖安装

需要安装以下依赖包：

```bash
pip install markdown ebooklib weasyprint
```

## 使用方法

1. 确保已经安装了所有依赖包
2. 运行生成脚本：

```bash
python generate_book.py
```

3. 脚本将生成两个文件：
   - `raydp-source-code-analysis.pdf`：PDF格式电子书
   - `raydp-source-code-analysis.epub`：EPUB格式电子书

## 内容来源

电子书内容来源于`../raydp/source_code_analysis/`目录下的Markdown文档，包括：

- 核心架构分析
- Spark-on-Ray 运行原理深度分析
- Spark Run on Ray 详细分析

## 自定义

如果需要修改生成的电子书内容或格式，可以编辑`generate_book.py`脚本中的以下部分：

- `self.chapters`：调整章节顺序或添加/删除章节
- `generate_pdf`方法：修改PDF的CSS样式
- `generate_epub`方法：修改EPUB的元数据和格式

## 注意事项

1. 确保`../raydp/source_code_analysis/`目录下存在所需的Markdown文档
2. 生成PDF可能需要安装额外的系统依赖，如`libcairo2`、`libpango1.0-0`等
3. 如果遇到编码问题，请确保所有Markdown文件都是UTF-8编码