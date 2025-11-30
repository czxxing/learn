

## 1. 基本属性指标

### image_width 和 image_height
- **实现位置**：<mcfile name="image_shape_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/image_shape_filter.py"></mcfile> 中的 `ImageShapeFilter` 类
- **计算方式**：
  - 检查指标是否已计算（避免重复计算）
  - 通过 `load_image` 加载图像数据
  - 直接从图像对象获取 `.width` 和 `.height` 属性
  - 结果存储在 `sample[Fields.stats][StatsKeys.image_width]` 和 `sample[Fields.stats][StatsKeys.image_height]`

### image_sizes
- **实现位置**：<mcfile name="image_size_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/image_size_filter.py"></mcfile> 中的 `ImageSizeFilter` 类
- **计算方式**：
  - 优先使用图像字节数据长度（更高效）：`[len(img_bytes) for img_bytes in sample[self.image_bytes_key]]`
  - 备选方案：通过 `get_file_size` 函数获取图像文件大小
  - 结果存储在 `sample[Fields.stats][StatsKeys.image_sizes]`

### aspect_ratios
- **实现位置**：<mcfile name="image_aspect_ratio_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/image_aspect_ratio_filter.py"></mcfile> 中的 `ImageAspectRatioFilter` 类
- **计算方式**：
  - 加载图像后计算宽高比：`images[key].width / images[key].height`
  - 结果存储在 `sample[Fields.stats][StatsKeys.aspect_ratios]`

## 2. 内容分析指标

### face_ratios
- **实现位置**：<mcfile name="image_face_ratio_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/image_face_ratio_filter.py"></mcfile> 中的 `ImageFaceRatioFilter` 类
- **计算方式**：
  - 使用 OpenCV 人脸检测模型（默认 `haarcascade_frontalface_alt.xml`）
  - 通过 `detect_faces` 函数检测人脸位置
  - 计算最大人脸面积与图像总面积的比例：`max([w * h for _, _, w, h in dets], default=0.0) / image_area`
  - 结果存储在 `sample[Fields.stats][StatsKeys.face_ratios]`

### face_detections
- **实现机制**：通过 `mm_utils.py` 中的 `detect_faces` 函数实现
- **处理方式**：在 `ImageFaceRatioFilter` 和 `ImageFaceCountFilter` 中调用，返回人脸位置坐标

### face_counts
- **实现位置**：<mcfile name="image_face_count_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/image_face_count_filter.py"></mcfile> 中的 `ImageFaceCountFilter` 类
- **计算方式**：
  - 使用 OpenCV 人脸检测模型检测人脸
  - 统计每张图像中检测到的人脸数量：`len(dets)`
  - 结果存储在 `sample[Fields.stats][StatsKeys.face_counts]`

## 3. 质量评估指标

### image_aesthetics_scores
- **实现位置**：<mcfile name="image_aesthetics_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/image_aesthetics_filter.py"></mcfile> 中的 `ImageAestheticsFilter` 类
- **计算方式**：
  - 使用 Hugging Face 美学评分模型（默认 `shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE`）
  - 通过模型预测图像的美学分数
  - 对于特定模型，分数需要除以 10 进行归一化
  - 结果存储在 `sample[Fields.stats][StatsKeys.image_aesthetics_scores]`

### image_nsfw_score
- **实现位置**：<mcfile name="image_nsfw_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/image_nsfw_filter.py"></mcfile> 中的 `ImageNSFWFilter` 类
- **计算方式**：
  - 使用 Hugging Face NSFW 检测模型（默认 `Falconsai/nsfw_image_detection`）
  - 通过模型预测并计算 NSFW 概率：`[float(scores[1]) for scores in torch.softmax(logits, dim=-1)]`
  - 结果存储在 `sample[Fields.stats][StatsKeys.image_nsfw_score]`

### image_watermark_prob
- **实现位置**：<mcfile name="image_watermark_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/image_watermark_filter.py"></mcfile> 中的 `ImageWatermarkFilter` 类
- **计算方式**：
  - 使用 Hugging Face 水印检测模型（默认 `amrul-hzz/watermark_detector`）
  - 通过模型预测图像中存在水印的概率：`[float(probs[1]) for probs in torch.softmax(logits, dim=-1)]`
  - 结果存储在 `sample[Fields.stats][StatsKeys.image_watermark_prob]`

## 共同特点和设计模式

1. **避免重复计算**：所有指标计算前都会检查是否已计算过
2. **上下文缓存**：使用 `context` 参数缓存已加载的图像，提高多指标计算效率
3. **批量处理支持**：大多数过滤器支持批量处理，通过 `compute_stats_single` 或 `compute_stats_batched` 方法实现
4. **统一存储机制**：所有指标都存储在 `sample[Fields.stats][StatsKeys.指标名]` 中
5. **灵活的过滤策略**：支持 "any"（任一图像满足条件）或 "all"（所有图像满足条件）的过滤策略
6. **设备优化**：对于需要模型的指标（美学、NSFW、水印检测），支持 CUDA 加速
        
          
# data-juicer 图像指标计算的逐行代码讲解

## 1. image_shape_filter.py - 图像尺寸指标(image_width/image_height)计算

```python
import sys
import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys  # 导入常量定义
from data_juicer.utils.mm_utils import load_data_with_context, load_image  # 导入图像加载工具
from ..base_op import OPERATORS, Filter  # 导入基类
from ..op_fusion import LOADED_IMAGES  # 导入图像加载缓存装饰器

# 注册为操作符，同时注册到已加载图像的装饰器中
@OPERATORS.register_module("image_shape_filter")
@LOADED_IMAGES.register_module("image_shape_filter")
class ImageShapeFilter(Filter):  # 继承自Filter基类
    """过滤图像尺寸在特定范围内的样本"""

    _batched_op = True  # 标记为支持批量操作

    def __init__(self,
        min_width: int = 1,
        max_width: int = sys.maxsize,
        min_height: int = 1,
        max_height: int = sys.maxsize,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        # 初始化父类
        super().__init__(*args, **kwargs)
        # 设置尺寸过滤阈值
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        # 验证过滤策略参数
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        # 设置过滤策略：any表示任一图像满足条件就保留，all表示所有图像都要满足条件
        self.any = any_or_all == "any"

    def compute_stats_single(self, sample, context=False):
        # 检查指标是否已经计算过（避免重复计算）
        if StatsKeys.image_width in sample[Fields.stats] and StatsKeys.image_height in sample[Fields.stats]:
            return sample

        # 检查样本中是否有图像
        if self.image_key not in sample or not sample[self.image_key]:
            # 如果没有图像，设置为空数组
            sample[Fields.stats][StatsKeys.image_width] = np.array([], dtype=np.int64)
            sample[Fields.stats][StatsKeys.image_height] = np.array([], dtype=np.int64)
            return sample

        # 获取图像键列表
        loaded_image_keys = sample[self.image_key]
        # 加载图像（支持上下文缓存，避免重复加载）
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        # 遍历所有图像，获取每个图像的宽度和高度
        whs = {key: (images[key].width, images[key].height) for key in images}
        # 将宽度和高度分别存储到stats中，保持与原图像顺序一致
        sample[Fields.stats][StatsKeys.image_width] = [whs[key][0] for key in loaded_image_keys]
        sample[Fields.stats][StatsKeys.image_height] = [whs[key][1] for key in loaded_image_keys]
        return sample

    def process_single(self, sample):
        # 从stats中获取计算好的宽度和高度
        ws = sample[Fields.stats][StatsKeys.image_width]
        hs = sample[Fields.stats][StatsKeys.image_height]
        # 如果没有图像，直接返回保留
        if len(ws) <= 0:
            return True
        # 计算每个图像是否满足宽高条件
        keep_bools = np.array(
            [
                self.get_keep_boolean(w, self.min_width, self.max_width)
                and self.get_keep_boolean(h, self.min_height, self.max_height)
                for w, h in zip(ws, hs)
            ]
        )

        # 根据策略决定是否保留样本
        if self.any:
            return keep_bools.any()  # 任一图像满足条件就保留
        else:
            return keep_bools.all()  # 所有图像都要满足条件才保留
```

## 2. image_size_filter.py - 图像文件大小(image_sizes)计算

```python
import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys  # 导入常量定义
from data_juicer.utils.mm_utils import get_file_size, size_to_bytes  # 导入文件大小计算工具
from ..base_op import OPERATORS, Filter  # 导入基类

@OPERATORS.register_module("image_size_filter")
class ImageSizeFilter(Filter):  # 继承自Filter基类
    """过滤图像文件大小在特定范围内的样本"""

    _batched_op = True  # 标记为支持批量操作

    def __init__(self, min_size: str = "0", max_size: str = "1TB", any_or_all: str = "any", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 将人类可读的大小格式（如KB、MB）转换为字节数
        self.min_size = size_to_bytes(min_size)
        self.max_size = size_to_bytes(max_size)
        # 验证过滤策略参数
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

    def compute_stats_single(self, sample, context=False):
        # 检查指标是否已经计算过
        if StatsKeys.image_sizes in sample[Fields.stats]:
            return sample

        # 检查样本中是否有图像
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_sizes] = np.array([], dtype=np.float64)
            return sample

        # 优化：对于文件大小计算，优先使用已有的字节数据，避免重新加载图像
        if self.image_bytes_key in sample and len(sample[self.image_bytes_key]) == len(sample[self.image_key]):
            # 直接计算字节数据的长度作为文件大小
            sample[Fields.stats][StatsKeys.image_sizes] = [len(img_bytes) for img_bytes in sample[self.image_bytes_key]]
        else:
            # 如果没有字节数据，则通过文件路径获取文件大小
            sample[Fields.stats][StatsKeys.image_sizes] = [
                get_file_size(img_path) for img_path in sample[self.image_key]
            ]

        return sample

    def process_single(self, sample):
        # 从stats中获取计算好的图像大小
        image_sizes = sample[Fields.stats][StatsKeys.image_sizes]
        # 计算每个图像是否满足大小条件
        keep_bools = np.array(
            [self.get_keep_boolean(image_size, self.min_size, self.max_size) for image_size in image_sizes]
        )
        # 如果没有图像，直接返回保留
        if len(keep_bools) <= 0:
            return True

        # 根据策略决定是否保留样本
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
```

## 3. image_face_ratio_filter.py - 人脸比例(face_ratios)计算

```python
import os
import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader  # 懒加载，优化导入时间
from data_juicer.utils.mm_utils import detect_faces, load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, UNFORKABLE, Filter
from ..op_fusion import LOADED_IMAGES

# 懒加载OpenCV
cv2 = LazyLoader("cv2", "opencv-python")

OP_NAME = "image_face_ratio_filter"

# 注册为不可分叉操作符（可能因为模型加载），同时注册为操作符和已加载图像装饰器
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageFaceRatioFilter(Filter):
    """过滤人脸比例在特定范围内的样本"""

    # 人脸检测的默认参数
    _default_kwargs = {
        "scaleFactor": 1.1,  # 图像金字塔缩放因子
        "minNeighbors": 3,   # 每个候选矩形应该保留的邻近数
        "minSize": None,     # 最小人脸尺寸
        "maxSize": None,     # 最大人脸尺寸
    }

    def __init__(
        self,
        cv_classifier: str = "",  # OpenCV分类器路径
        min_ratio: float = 0.0,    # 最小人脸比例
        max_ratio: float = 0.4,    # 最大人脸比例
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # 如果没有指定分类器路径，使用OpenCV默认的人脸检测分类器
        if cv_classifier == "":
            cv_classifier = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt.xml")
        # 设置比例阈值
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        # 合并默认参数和用户提供的参数
        self.extra_kwargs = self._default_kwargs
        for key in kwargs:
            if key in self.extra_kwargs:
                self.extra_kwargs[key] = kwargs[key]

        # 验证过滤策略参数
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

        # 准备OpenCV分类器模型
        self.model_key = prepare_model(model_type="opencv_classifier", model_path=cv_classifier)

    def compute_stats_single(self, sample, context=False):
        # 检查指标是否已经计算过
        if StatsKeys.face_ratios in sample[Fields.stats]:
            return sample

        # 检查样本中是否有图像
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.face_ratios] = np.array([], dtype=np.float64)
            return sample

        # 加载图像
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        # 获取模型
        model = get_model(self.model_key)

        # 对每张图像进行人脸检测
        face_detections = {}
        for key, image in images.items():
            # 使用detect_faces函数检测人脸
            face_detections[key] = detect_faces(image, model, **self.extra_kwargs)
        logger.debug(f"detections: {face_detections}")

        # 计算每张图像的最大人脸面积比例
        face_area_ratios = {}
        for key, dets in face_detections.items():
            # 计算图像总面积
            image_area = images[key].width * images[key].height
            # 计算最大人脸面积与图像面积的比例
            # 如果没有检测到人脸，default=0.0
            face_area_ratios[key] = max([w * h for _, _, w, h in dets], default=0.0) / image_area
        logger.debug(f"ratios: {face_area_ratios}")

        # 将计算结果存储到stats中，保持与原图像顺序一致
        sample[Fields.stats][StatsKeys.face_ratios] = [face_area_ratios[key] for key in loaded_image_keys]
        return sample

    def process_single(self, sample):
        # 从stats中获取计算好的人脸比例
        face_ratios = sample[Fields.stats][StatsKeys.face_ratios]
        # 如果没有图像，直接返回保留
        if len(face_ratios) <= 0:
            return True

        # 计算每个图像是否满足人脸比例条件
        keep_bools = np.array(
            [self.get_keep_boolean(face_ratio, self.min_ratio, self.max_ratio) for face_ratio in face_ratios]
        )

        # 根据策略决定是否保留样本
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
```

## 4. image_aesthetics_filter.py - 图像美学评分(image_aesthetics_scores)计算

```python
import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ...utils.model_utils import get_model, prepare_model
from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

# 懒加载PyTorch
torch = LazyLoader("torch")

OP_NAME = "image_aesthetics_filter"

@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageAestheticsFilter(Filter):
    """过滤图像美学评分在特定范围内的样本"""

    _accelerator = "cuda"  # 指定使用CUDA加速

    def __init__(
        self,
        hf_scorer_model: str = "",  # Hugging Face模型名称
        trust_remote_code: bool = False,  # 是否信任远程代码
        min_score: float = 0.5,  # 最小美学评分
        max_score: float = 1.0,  # 最大美学评分
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        # 设置内存需求
        kwargs["mem_required"] = "1500MB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        # 如果没有指定模型，使用默认的美学评分模型
        if hf_scorer_model == "":
            hf_scorer_model = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
        # 设置评分阈值
        self.min_score = min_score
        self.max_score = max_score

        # 验证过滤策略参数
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

        # 准备美学评分模型
        self.model_key = prepare_model(
            model_type="simple_aesthetics",
            pretrained_model_name_or_path=hf_scorer_model,
            trust_remote_code=trust_remote_code,
        )
        # 检查是否需要将评分除以10进行归一化
        self.need_normalized_by_ten = "shunk031/aesthetics-predictor" in hf_scorer_model

    def compute_stats_single(self, sample, rank=None, context=False):
        # 检查指标是否已经计算过
        if StatsKeys.image_aesthetics_scores in sample[Fields.stats]:
            return sample

        # 检查样本中是否有图像
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_aesthetics_scores] = np.array([], dtype=np.float64)
            return sample

        # 加载图像
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        # 获取模型和处理器
        model, processor = get_model(self.model_key, rank, self.use_cuda())
        # 预处理图像
        inputs = processor(images=list(images.values()), return_tensors="pt").to(model.device)
        # 禁用梯度计算，节省内存
        with torch.no_grad():
            # 模型推理
            outputs = model(**inputs)
        # 检查是否需要归一化
        if self.need_normalized_by_ten:
            aesthetics_scores = outputs.logits / 10.0
        else:
            aesthetics_scores = outputs.logits

        # 将张量转换为Python浮点数
        aesthetics_scores = [aesthetics_score.item() for aesthetics_score in aesthetics_scores]

        logger.debug(f"aesthetics_scores: {aesthetics_scores}")

        # 存储评分结果
        sample[Fields.stats][StatsKeys.image_aesthetics_scores] = aesthetics_scores
        return sample

    def process_single(self, sample):
        # 从stats中获取计算好的美学评分
        aesthetics_scores = sample[Fields.stats][StatsKeys.image_aesthetics_scores]
        # 如果没有图像，直接返回保留
        if len(aesthetics_scores) <= 0:
            return True

        # 计算每个图像是否满足美学评分条件
        keep_bools = np.array(
            [
                self.get_keep_boolean(aesthetics_score, self.min_score, self.max_score)
                for aesthetics_score in aesthetics_scores
            ]
        )

        # 根据策略决定是否保留样本
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
```

## 5. image_nsfw_filter.py - 图像NSFW评分(image_nsfw_score)计算

```python
import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

# 懒加载PyTorch
torch = LazyLoader("torch")

OP_NAME = "image_nsfw_filter"

@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageNSFWFilter(Filter):
    """过滤图像NSFW评分在特定范围内的样本"""

    _accelerator = "cuda"  # 指定使用CUDA加速

    def __init__(
        self,
        hf_nsfw_model: str = "Falconsai/nsfw_image_detection",  # 默认NSFW检测模型
        trust_remote_code: bool = False,
        min_score: float = 0.0,  # 最小NSFW评分
        max_score: float = 0.5,  # 最大NSFW评分（通常设置较低值来过滤不良内容）
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        # 设置内存需求
        kwargs["mem_required"] = "1GB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        # 设置评分阈值
        self.min_score = min_score
        self.max_score = max_score
        # 验证过滤策略参数
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"
        # 准备NSFW检测模型
        self.model_key = prepare_model(
            model_type="huggingface", pretrained_model_name_or_path=hf_nsfw_model, trust_remote_code=trust_remote_code
        )

    def compute_stats_single(self, sample, rank=None, context=False):
        # 检查指标是否已经计算过
        if StatsKeys.image_nsfw_score in sample[Fields.stats]:
            return sample

        # 检查样本中是否有图像
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_nsfw_score] = np.array([], dtype=np.float64)
            return sample

        # 加载图像
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        # 获取模型和处理器
        model, processor = get_model(self.model_key, rank, self.use_cuda())
        # 预处理图像
        images = [images[key] for key in images]
        inputs = processor(images=images, return_tensors="pt").to(model.device)
        # 禁用梯度计算
        with torch.no_grad():
            # 模型推理
            outputs = model(**inputs)
        # 获取模型输出的logits
        logits = outputs.logits
        # 计算softmax并提取NSFW类别的概率（假设索引1是NSFW类别）
        nsfw_scores = [float(scores[1]) for scores in torch.softmax(logits, dim=-1)]

        # 存储NSFW评分
        sample[Fields.stats][StatsKeys.image_nsfw_score] = nsfw_scores

        return sample

    def process_single(self, sample, rank=None):
        # 从stats中获取计算好的NSFW评分
        itm_scores = sample[Fields.stats][StatsKeys.image_nsfw_score]
        # 如果没有图像，直接返回保留
        if len(itm_scores) <= 0:
            return True

        # 计算每个图像是否满足NSFW评分条件
        keep_bools = np.array(
            [self.get_keep_boolean(itm_score, self.min_score, self.max_score) for itm_score in itm_scores]
        )

        # 根据策略决定是否保留样本
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
```

## 6. image_watermark_filter.py - 图像水印检测(image_watermark_prob)计算

```python
import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

# 懒加载PyTorch
torch = LazyLoader("torch")

OP_NAME = "image_watermark_filter"

@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageWatermarkFilter(Filter):
    """过滤图像水印概率在特定阈值以下的样本"""

    _accelerator = "cuda"  # 指定使用CUDA加速

    def __init__(
        self,
        hf_watermark_model: str = "amrul-hzz/watermark_detector",  # 默认水印检测模型
        trust_remote_code: bool = False,
        prob_threshold: float = 0.8,  # 水印概率阈值
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        # 设置内存需求
        kwargs["mem_required"] = "500MB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        # 设置水印概率阈值
        self.prob_threshold = prob_threshold
        # 验证过滤策略参数
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"
        # 准备水印检测模型
        self.model_key = prepare_model(
            model_type="huggingface",
            pretrained_model_name_or_path=hf_watermark_model,
            trust_remote_code=trust_remote_code,
        )

    def compute_stats_single(self, sample, rank=None, context=False):
        # 检查指标是否已经计算过
        if StatsKeys.image_watermark_prob in sample[Fields.stats]:
            return sample

        # 检查样本中是否有图像
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_watermark_prob] = np.array([], dtype=np.float64)
            return sample

        # 加载图像
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        # 获取模型和处理器
        model, processor = get_model(self.model_key, rank, self.use_cuda())
        # 预处理图像
        images = [images[key] for key in images]
        inputs = processor(images=images, return_tensors="pt").to(model.device)
        # 模型推理
        outputs = model(**inputs)
        # 获取模型输出的logits
        logits = outputs.logits
        # 计算softmax并提取水印类别的概率（假设索引1是水印类别）
        watermark_probs = [float(probs[1]) for probs in torch.softmax(logits, dim=-1)]

        # 存储水印概率
        sample[Fields.stats][StatsKeys.image_watermark_prob] = watermark_probs

        return sample

    def process_single(self, sample, rank=None):
        # 从stats中获取计算好的水印概率
        itm_probs = sample[Fields.stats][StatsKeys.image_watermark_prob]
        # 如果没有图像，直接返回保留
        if len(itm_probs) <= 0:
            return True

        # 计算每个图像是否满足水印概率条件（只检查上限）
        keep_bools = np.array([self.get_keep_boolean(itm_prob, None, self.prob_threshold) for itm_prob in itm_probs])

        # 根据策略决定是否保留样本
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
```

## 关键设计模式和工作流程

1. **统一的指标计算模式**：
   - 所有过滤器都使用`compute_stats_single`方法计算指标
   - 首先检查指标是否已计算（避免重复计算）
   - 处理无图像的边界情况
   - 加载图像（支持上下文缓存）
   - 计算指标并存储到`sample[Fields.stats]`中

2. **图像加载优化**：
   - 使用`load_data_with_context`函数支持图像缓存
   - 标记在`LOADED_IMAGES`中的操作符共享已加载的图像
   - 对于某些指标（如文件大小），优先使用字节数据避免完整加载

3. **模型管理**：
   - 使用`prepare_model`和`get_model`统一管理模型加载和复用
   - 支持CUDA加速的模型会标记`_accelerator = "cuda"`
   - 对于需要大量内存的模型，设置`mem_required`参数

4. **统一的过滤逻辑**：
   - 使用`get_keep_boolean`方法检查是否满足阈值条件
   - 支持"any"和"all"两种过滤策略
   - 处理无图像的边界情况

5. **懒加载机制**：
   - 使用`LazyLoader`懒加载大型库（如OpenCV、PyTorch）
   - 减少导入时间，优化启动性能

这些设计使得data-juicer能够高效地计算各种图像指标，同时避免重复计算和不必要的资源消耗，支持大规模数据集的处理。
        