
          
## 视频指标收集过程分析

### 1. 基本属性指标

#### 1.1 视频时长 (video_duration)

```python
# VideoDurationFilter.compute_stats_single 方法中
video_durations = {}
for video_key, video in videos.items():
    stream = video.streams.video[0]
    video_durations[video_key] = round(stream.duration * stream.time_base)
    if not context:
        close_video(video)

sample[Fields.stats][StatsKeys.video_duration] = [
    video_durations[video_key] for video_key in sample[self.video_key]
]
```

收集过程：
- 使用av库加载视频文件
- 从视频流对象中获取duration和time_base属性
- 计算实际时长：stream.duration * stream.time_base
- 四舍五入取整后存储到stats字段

#### 1.2 视频宽度 (video_width) 和高度 (video_height)

```python
# VideoResolutionFilter.compute_stats_single 方法中
video_width, video_height = dict(), dict()
for video_key, video in videos.items():
    # default to load the first stream
    video_stream = video.streams.video[0]
    video_width[video_key] = video_stream.codec_context.width
    video_height[video_key] = video_stream.codec_context.height

sample[Fields.stats][StatsKeys.video_width] = [video_width[video_key] for video_key in sample[self.video_key]]
sample[Fields.stats][StatsKeys.video_height] = [video_height[video_key] for video_key in sample[self.video_key]]
```

收集过程：
- 加载视频文件并获取视频流
- 直接从codec_context中读取width和height属性
- 分别存储到对应的stats字段中

#### 1.3 视频宽高比 (video_aspect_ratios)

```python
# VideoAspectRatioFilter.compute_stats_single 方法中
video_aspect_ratios = {}
for key, video in videos.items():
    stream = video.streams.video[0]
    video_aspect_ratios[key] = stream.codec_context.width / stream.codec_context.height
    if not context:
        close_video(video)

sample[Fields.stats][StatsKeys.video_aspect_ratios] = [video_aspect_ratios[key] for key in loaded_video_keys]
```

收集过程：
- 加载视频文件并获取视频流
- 计算宽高比：width / height
- 存储到stats字段中

### 2. 内容分析指标

#### 2.1 视频OCR区域比例 (video_ocr_area_ratio)

```python
# VideoOcrAreaRatioFilter.compute_stats_single 方法中
for video_key, container in videos.items():
    # 提取采样帧
    sampled_frames = extract_video_frames_uniformly(container, self.frame_sample_num)
    images = [f.to_image() for f in sampled_frames]
    frame_ocr_area_ratios = []
    for idx, image in enumerate(images):
        # 使用EasyOCR检测文本
        horizontal_list, free_list = reader.detect(np.asarray(image))
        total_area = image.width * image.height
        # 计算矩形文本区域面积
        rect_area = 0
        for xmin, xmax, ymin, ymax in horizontal_list[0]:
            if xmax < xmin or ymax < ymin:
                continue
            rect_area += (xmax - xmin) * (ymax - ymin)
        # 计算自由形态文本区域面积
        quad_area = 0
        for points in free_list[0]:
            triangle1 = points[:3]
            quad_area += triangle_area(*triangle1)
            triangle2 = points[2:] + [points[0]]
            quad_area += triangle_area(*triangle2)
        text_area = rect_area + quad_area
        frame_ocr_area_ratios.append(text_area / total_area)
    # 计算所有采样帧的平均OCR区域比例
    video_ocr_area_ratios[video_key] = np.mean(frame_ocr_area_ratios)

sample[Fields.stats][StatsKeys.video_ocr_area_ratio] = [
    video_ocr_area_ratios[video_key] for video_key in sample[self.video_key]
]
```

收集过程：
- 使用均匀采样从视频中提取指定数量的帧
- 对每一帧使用EasyOCR进行文本检测
- 计算两种类型的文本区域面积：矩形文本区域和自由形态文本区域
- 矩形区域使用矩形面积公式计算
- 自由形态区域通过将四边形分割为两个三角形计算面积
- 计算文本区域占整个帧面积的比例
- 对所有采样帧的比例取平均值作为视频的OCR区域比例

### 3. 质量评估指标

#### 3.1 视频美学分数 (video_aesthetic_score) 和视频帧美学分数 (video_frames_aesthetics_score)

```python
# VideoAestheticsFilter.compute_stats_single 方法中
for key, video in videos.items():
    # 根据配置提取关键帧或均匀采样帧
    if self.frame_sampling_method == "all_keyframes":
        frames = extract_key_frames(video)
    elif self.frame_sampling_method == "uniform":
        frames = extract_video_frames_uniformly(video, self.frame_num)
    else:
        frames = []
    
    frame_images = [frame.to_image() for frame in frames]
    
    if len(frame_images) > 0:
        # 使用预训练的美学评估模型
        model, processor = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())
        inputs = processor(images=frame_images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        if self.need_normalized_by_ten:
            aesthetics_score = outputs.logits / 10.0
        else:
            aesthetics_score = outputs.logits
        
        # 根据配置进行结果聚合
        if self.reduce_mode == "avg":
            aesthetics_score = float(aesthetics_score.mean())
        elif self.reduce_mode == "max":
            aesthetics_score = float(aesthetics_score.max())
        else:
            aesthetics_score = float(aesthetics_score.min())
    else:
        aesthetics_score = 0.0
    
    aesthetics_scores.append(aesthetics_score)

sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = aesthetics_scores
```

收集过程：
- 支持两种帧采样方法：提取所有关键帧或均匀采样指定数量的帧
- 将采样的帧转换为图像格式
- 使用预训练的美学评估模型（默认为'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE'）
- 处理图像并通过模型获取美学分数
- 根据模型类型可能需要将分数除以10进行归一化
- 支持三种聚合模式：平均值(avg)、最大值(max)或最小值(min)
- 存储到stats字段中

#### 3.2 视频运动分数 (video_motion_score)

```python
# VideoMotionScoreFilter.compute_stats_single 和 compute_flow 方法中
def compute_flow(self, prev_frame, curr_frame):
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is None:
        flow = None
    else:
        flow = self.model(prev_frame, curr_frame, None, **self.extra_kwargs)
    return flow, curr_frame

# 在主方法中
with VideoCapture(video_key) as cap:
    # 计算采样步长
    fps = cap.get(cv2.CAP_PROP_FPS)
    sampling_fps = min(self.sampling_fps, fps)
    sampling_step = round(fps / sampling_fps)
    
    # 处理每一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调整帧大小
        if new_size != (height, width):
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        
        # 计算光流
        flow, prev_frame = self.compute_flow(prev_frame, frame)
        if flow is None:
            continue
        
        # 计算运动幅度
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        frame_motion_score = np.mean(mag)
        if self.relative:
            frame_motion_score /= np.hypot(*frame.shape[:2])
        video_motion_scores.append(frame_motion_score)
        
        # 跳过采样步长的帧
        frame_count += sampling_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    
    # 计算平均运动分数
    unique_motion_scores[video_key] = np.mean(video_motion_scores or [-1])

sample[Fields.stats][StatsKeys.video_motion_score] = [unique_motion_scores[key] for key in loaded_video_keys]
```

收集过程：
- 使用OpenCV的VideoCapture打开视频
- 根据指定的采样帧率计算采样步长
- 对视频帧进行采样并转换为灰度图
- 使用OpenCV的calcOpticalFlowFarneback算法计算两帧之间的光流
- 计算光流的幅度(magnitude)作为运动强度的指标
- 可选地将运动分数相对于帧对角线长度进行归一化
- 计算所有采样帧的平均运动分数作为视频的运动分数

#### 3.3 视频NSFW分数 (video_nsfw_score)

```python
# VideoNSFWFilter.compute_stats_single 方法中
for video_key, video in videos.items():
    # 提取采样帧
    if self.frame_sampling_method == "all_keyframes":
        frames = extract_key_frames(video)
    elif self.frame_sampling_method == "uniform":
        frames = extract_video_frames_uniformly(video, self.frame_num)
    else:
        frames = []
    
    frame_images = [frame.to_image() for frame in frames]
    
    if len(frame_images) > 0:
        # 使用预训练的NSFW检测模型
        inputs = processor(images=frame_images, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        # 获取NSFW类别的分数（通常是索引1）
        cur_scores = [scores[1] for scores in torch.softmax(logits, dim=-1)]
        cur_scores = torch.Tensor(cur_scores)
        
        # 根据配置聚合结果
        if self.reduce_mode == "avg":
            cur_score = cur_scores.mean()
        elif self.reduce_mode == "max":
            cur_score = cur_scores.max()
        else:
            cur_score = cur_scores.min()
    else:
        cur_score = 0.0
    
    nsfw_scores.append(float(cur_score))

sample[Fields.stats][StatsKeys.video_nsfw_score] = nsfw_scores
```

收集过程：
- 支持两种帧采样方法：提取所有关键帧或均匀采样指定数量的帧
- 将采样的帧转换为图像格式
- 使用预训练的NSFW检测模型处理图像
- 从模型输出中获取NSFW类别的概率分数
- 支持三种聚合模式：平均值(avg)、最大值(max)或最小值(min)
- 存储到stats字段中

#### 3.4 视频水印概率 (video_watermark_prob)

```python
# VideoWatermarkFilter.compute_stats_single 方法中
for video_key, video in videos.items():
    # 提取采样帧
    if self.frame_sampling_method == "all_keyframes":
        frames = extract_key_frames(video)
    elif self.frame_sampling_method == "uniform":
        frames = extract_video_frames_uniformly(video, self.frame_num)
    else:
        frames = []
    
    frame_images = [frame.to_image() for frame in frames]
    
    if len(frame_images) > 0:
        # 使用预训练的水印检测模型
        inputs = processor(images=frame_images, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        # 获取水印类别的概率
        cur_probs = [probs[1] for probs in torch.softmax(logits, dim=-1)]
        cur_probs = torch.Tensor(cur_probs)
        
        # 根据配置聚合结果
        if self.reduce_mode == "avg":
            cur_prob = cur_probs.mean()
        elif self.reduce_mode == "max":
            cur_prob = cur_probs.max()
        else:
            cur_prob = cur_probs.min()
    else:
        cur_prob = 0.0
    
    watermark_probs.append(float(cur_prob))

sample[Fields.stats][StatsKeys.video_watermark_prob] = watermark_probs
```

收集过程：
- 支持两种帧采样方法：提取所有关键帧或均匀采样指定数量的帧
- 将采样的帧转换为图像格式
- 使用预训练的水印检测模型处理图像
- 从模型输出中获取水印类别的概率分数
- 支持三种聚合模式：平均值(avg)、最大值(max)或最小值(min)
- 存储到stats字段中

### 总结

所有视频指标的收集都遵循相似的流程：
1. 首先检查指标是否已经计算过
2. 检查样本是否包含视频数据
3. 加载视频文件
4. 根据不同指标的需求提取视频信息（直接读取属性、采样帧等）
5. 进行特定的计算或模型推理
6. 结果聚合（如果需要）
7. 存储到样本的stats字段中

其中，基本属性指标（时长、宽高、宽高比）的计算较为简单，直接从视频流中读取信息；而质量评估和内容分析指标则需要更复杂的处理，如帧采样、OCR检测、深度学习模型推理等。
        