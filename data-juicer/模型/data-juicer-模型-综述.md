### 1. 文本处理模型
- **fasttext**: 用于语言识别，通过`prepare_fasttext_model`加载，主要使用`lid.176.bin`模型进行多语言识别
- **kenlm**: 用于语言建模和困惑度计算，通过`prepare_kenlm_model`加载，支持不同语言的n-gram语言模型
- **nltk**: 用于自然语言处理的基础任务，如句子分割(punkt)，通过`prepare_nltk_model`加载
- **nltk_pos_tagger**: 用于词性标注，通过`prepare_nltk_pos_tagger`加载
- **sentencepiece**: 用于文本分词，通过`prepare_sentencepiece_for_lang`加载，支持多种语言
- **spacy**: 用于高级自然语言处理，支持英文和中文，通过`prepare_spacy_model`加载
- **embedding**: 用于生成文本嵌入向量，支持不同的池化策略，通过`prepare_embedding_model`加载

### 2. 图像处理模型
- **fastsam**: 用于快速图像分割，通过`prepare_fastsam_model`加载，基于YOLO架构
- **yolo**: 用于目标检测，通过`prepare_yolo_model`加载，支持多种YOLO版本
- **recognizeAnything**: 用于图像标签识别，通过`prepare_recognizeAnything_model`加载
- **simple_aesthetics**: 用于图像美学评分，通过`prepare_simple_aesthetics_model`加载
- **opencv_classifier**: 用于基于OpenCV的图像分类，通过`prepare_opencv_classifier`加载

### 3. 多模态模型
- **huggingface**: 用于加载各种Hugging Face模型，通过`prepare_huggingface_model`加载，支持文本、图像、音频等多种模态
- **CLIP**: 在过滤器中广泛使用，如`ImageTextSimilarityFilter`、`ImagePairSimilarityFilter`等，用于计算图像-文本、图像-图像的相似度
- **BLIP**: 用于图文匹配分数计算，如在`ImageTextMatchingFilter`中使用
- **Owl-ViT**: 用于短语定位和召回率计算，如在`PhraseGroundingRecallFilter`中使用
- **EasyOCR**: 用于OCR文本检测，如在`video_ocr_area_ratio_filter.py`中使用

### 4. 视频处理模型
- **video_blip**: 用于视频描述生成，通过`prepare_video_blip_model`加载，是BLIP模型的视频处理增强版
- **dwpose**: 用于人体姿态估计，通过`prepare_dwpose_model`加载，可提取视频中的人体关键点

### 5. 生成模型
- **diffusion**: 用于图像生成任务，支持image2image、text2image、inpainting等，通过`prepare_diffusion_model`加载
- **sdxl-prompt-to-prompt**: 用于SDXL模型的提示词修改，通过`prepare_sdxl_prompt2prompt`加载
- **vggt**: 用于视频生成，通过`prepare_vggt_model`加载

### 6. 大型语言模型和API
- **api**: 用于与OpenAI兼容的API模型交互，通过`prepare_api_model`加载，支持聊天和嵌入API
- **vllm**: 用于高效大型语言模型推理，通过`prepare_vllm_model`加载，支持张量并行

### 模型管理机制
- data-juicer使用统一的模型管理机制，通过`MODEL_FUNCTION_MAPPING`字典将模型类型映射到对应的准备函数
- 提供了模型缓存、下载和加载功能，支持自动下载缺失的模型文件
- 支持GPU加速和分布式处理，通过`get_model`函数实现模型的延迟加载和设备分配

这些模型在data-juicer的各种操作中被广泛使用，如过滤器、映射器等，用于数据清洗、分析、增强和转换等任务。