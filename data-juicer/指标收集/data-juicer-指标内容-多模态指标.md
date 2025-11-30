          
## 多模态相似度指标收集过程分析

我将详细分析7个多模态和文本相似度指标的收集实现过程：

### 1. 图文相似度(image_text_similarity) - <mcfile name="image_text_similarity_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/image_text_similarity_filter.py"></mcfile>

**实现流程：**
```python
# 1. 检查指标是否已计算
if StatsKeys.image_text_similarity in sample[Fields.stats]:
    return sample

# 2. 检查是否包含图像
if self.image_key not in sample or not sample[self.image_key]:
    sample[Fields.stats][StatsKeys.image_text_similarity] = np.array([], dtype=np.float64)
    return sample

# 3. 加载图像
loaded_image_keys = sample[self.image_key]
sample, images = load_data_with_context(
    sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
)

# 4. 处理文本分块并计算相似度
text = sample[self.text_key]
offset = 0
similarity = []
model, processor = get_model(self.model_key, rank, self.use_cuda())

for chunk in text.split(SpecialTokens.eoc):
    count = chunk.count(SpecialTokens.image)
    if count == 0 or len(chunk) == 0:
        continue
    
    # 5. 提取文本和图像块
    text_chunk = remove_special_tokens(chunk)
    image_chunk = []
    for image_key in loaded_image_keys[offset : offset + count]:
        image = images[image_key]
        # 可选的图像翻转处理
        if self.horizontal_flip: image = ImageOps.mirror(image)
        if self.vertical_flip: image = ImageOps.flip(image)
        image_chunk.append(image)

    # 6. 使用CLIP模型计算相似度
    inputs = processor(
        text=text_chunk,
        images=image_chunk,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.text_config.max_position_embeddings,
        padding=True,
    ).to(model.device)

    outputs = model(**inputs)
    chunk_logits = outputs.logits_per_text / 100.0

    # 7. 聚合策略选择
    if self.reduce_mode == "avg":
        chunk_similarity = chunk_logits.mean()
    elif self.reduce_mode == "max":
        chunk_similarity = chunk_logits.max()
    else:
        chunk_similarity = chunk_logits.min()

    similarity.append(float(chunk_similarity))
    offset += count

# 8. 存储结果
sample[Fields.stats][StatsKeys.image_text_similarity] = similarity
```

### 2. 图文匹配分数(image_text_matching_score) - <mcfile name="image_text_matching_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/image_text_matching_filter.py"></mcfile>

**实现流程：**
```python
# 1-5. 与图文相似度相同的检查和加载步骤

# 6. 使用BLIP模型计算匹配分数
inputs = processor(
    text=text_chunk,
    images=image_chunk,
    return_tensors="pt",
    truncation=True,
    max_length=model.config.text_config.max_position_embeddings,
    padding=True,
).to(model.device)

outputs = model(**inputs)
# BLIP特有的ITM分数计算，取类别1的softmax值
itm_scores = outputs.itm_score.detach().cpu().softmax(dim=-1)[:, 1]

# 7. 聚合策略选择
if self.reduce_mode == "avg":
    chunk_itm_score = itm_scores.mean()
elif self.reduce_mode == "max":
    chunk_itm_score = itm_scores.max()
else:
    chunk_itm_score = itm_scores.min()

matching_scores.append(float(chunk_itm_score))

# 8. 存储结果
sample[Fields.stats][StatsKeys.image_text_matching_score] = matching_scores
```

### 3. 短语定位召回率(phrase_grounding_recall) - <mcfile name="phrase_grounding_recall_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/phrase_grounding_recall_filter.py"></mcfile>

**实现流程：**
```python
# 1-5. 基础检查和加载

# 6. 文本处理和实体提取
text_this_chunk = remove_special_tokens(chunk)
ners_this_chunk = run_ner(text_this_chunk, pos_tagger)  # 使用NLTK提取名词短语
num_ners = len(ners_this_chunk)
if num_ners <= 0:
    recalls.append(1.0)  # 无实体时召回率为1
    continue

# 7. 加载图像
images_this_chunk = []
for image_key in loaded_image_keys[offset : offset + count]:
    # 图像加载和翻转处理...

# 8. 使用Owl-ViT模型定位短语
ners_batch = [ners_this_chunk] * len(images_this_chunk)
inputs = processor(
    text=ners_batch, images=images_this_chunk, return_tensors="pt", padding=True, truncation=True
).to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1] for img in images_this_chunk]).to(model.device)
    results = processor.post_process_object_detection(
        outputs, threshold=self.conf_thr, target_sizes=target_sizes
    )

# 9. 计算召回率
image_recalls = []
for idx, result in enumerate(results):
    scores = result["scores"]
    labels = result["labels"]
    boxes = result["boxes"]

    # 排序并限制预测数量
    order_idx = scores.argsort(descending=True)
    scores = scores[order_idx].tolist()[:num_ners]
    labels = labels[order_idx].tolist()[:num_ners]
    boxes = boxes[order_idx].tolist()[:num_ners]

    # NMS后处理和大框过滤
    hit = {}
    for box, label, score in zip(boxes, labels, scores):
        # 已命中的实体跳过
        if ners_this_chunk[label] in hit: continue
        
        # 过滤几乎覆盖整个图像的框
        xmin, ymin, xmax, ymax = box
        box_area = (xmax - xmin) * (ymax - ymin)
        if 1.0 * box_area / image_area > self.large_area_ratio_thr: continue
        
        # NMS类似的重叠框过滤
        suppressed = False
        for ner in hit:
            if iou(box, hit[ner][0]) > self.iou_thr:
                suppressed = True
                break
        if suppressed: continue

        hit[ners_this_chunk[label]] = (box, score)

    # 计算召回率 = 命中实体数 / 总实体数
    recall = 1.0 * len(hit) / num_ners
    image_recalls.append(recall)

# 10. 聚合策略
if self.reduce_mode == "avg":
    image_recall = sum(image_recalls) / len(image_recalls)
elif self.reduce_mode == "max":
    image_recall = max(image_recalls)
else:
    image_recall = min(image_recalls)

recalls.append(image_recall)

# 11. 存储结果
sample[Fields.stats][StatsKeys.phrase_grounding_recall] = recalls
```

### 4. 视频帧文本相似度(video_frames_text_similarity) - <mcfile name="video_frames_text_similarity_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/video_frames_text_similarity_filter.py"></mcfile>

**实现流程：**
```python
# 1-3. 基础检查和加载视频

# 4. 视频帧采样
for video_key in loaded_video_keys[offset : offset + count]:
    video = videos[video_key]
    sampled_frames_key = video_key + self.sampled_frames_key_suffix

    # 从缓存或直接提取帧
    if context and sampled_frames_key in sample[Fields.context]:
        frames = sample[Fields.context][sampled_frames_key]
    else:
        # 根据采样方法提取帧
        if self.frame_sampling_method == "all_keyframes":
            frames = extract_key_frames(video)
        elif self.frame_sampling_method == "uniform":
            frames = extract_video_frames_uniformly(video, self.frame_num)
        
        # 缓存采样结果
        if context:
            sample[Fields.context][sampled_frames_key] = frames

    # 转换为PIL图像并处理
    frame_images = [frame.to_image() for frame in frames]
    for image in frame_images:
        # 图像翻转处理...
        video_frame_images_chunk.append(image)

# 5. CLIP模型计算相似度（与图文相似度相同）
if len(video_frame_images_chunk) > 0:
    inputs = processor(
        text=text_chunk,
        images=video_frame_images_chunk,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.text_config.max_position_embeddings,
        padding=True,
    ).to(model.device)

    outputs = model(**inputs)
    chunk_logits = outputs.logits_per_text / 100.0

    # 聚合策略
    if self.reduce_mode == "avg":
        chunk_similarity = chunk_logits.mean()
    elif self.reduce_mode == "max":
        chunk_similarity = chunk_logits.max()
    else:
        chunk_similarity = chunk_logits.min()
else:
    chunk_similarity = 0.0

# 6. 存储结果
sample[Fields.stats][StatsKeys.video_frames_text_similarity] = similarity

# 7. 清理资源
if not context:
    for vid_key in videos:
        close_video(videos[vid_key])
```

### 5. 文本嵌入相似度(text_embd_similarity) - <mcfile name="text_embd_similarity_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/text_embd_similarity_filter.py"></mcfile>

**实现流程：**
```python
# 1. 准备验证集特征（初始化时或手动调用）
def prepare_valid_feature(self, dataset, n_shot=None, *args, **kwargs):
    n_shot = n_shot or len(dataset)
    dataset = dataset.select(range(0, n_shot))
    embeddings = self._get_embd(dataset)  # 计算验证集嵌入
    self.valid_feature.update({"embeddings": embeddings})

# 2. 计算单个样本嵌入
def _get_embd_single(self, sample, rank=None):
    model = get_model(self.model_key, rank, self.use_cuda())
    text = self._sample_to_text(sample)
    
    if self.is_hf_model:
        # 本地HF模型
        with torch.no_grad():
            embedding = model.encode(text)
    else:
        # API调用
        try:
            embedding = model(text, dimensions=self.ebd_dim, encoding_format="float")
        except Exception as e:
            # 尝试分段处理长文本
            sub_seq_length = len(text) // 9
            text_list = [text[i * sub_seq_length : (i + 1) * sub_seq_length] for i in range(10)]
            embedding = model(text_list, dimensions=self.ebd_dim, encoding_format="float")
    
    return embedding

# 3. 计算相似度
if StatsKeys.text_embd_similarity in sample[Fields.stats]:
    return sample

assert self.valid_feature_ready, "Validation feature not ready yet. Call prepare_valid_feature first."

embedding = self._get_embd_single(sample, rank)
try:
    # 计算与验证集所有嵌入的平均余弦相似度
    similarity = (
        torch.nn.functional.cosine_similarity(
            torch.tensor(embedding).view(1, -1), torch.from_numpy(self.valid_feature["embeddings"])
        )
        .mean()
        .item()
    )
except Exception as e:
    similarity = None

# 4. 存储结果
sample[Fields.stats][StatsKeys.text_embd_similarity] = similarity
```

### 6. 文本对相似度(text_pair_similarity) - <mcfile name="text_pair_similarity_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/text_pair_similarity_filter.py"></mcfile>

**实现流程：**
```python
# 1. 检查指标是否已计算
if StatsKeys.text_pair_similarity in sample[Fields.stats]:
    return sample

# 2. 检查文本对是否存在
if (
    self.text_key not in sample
    or len(sample[self.text_key]) == 0
    or self.text_key_second not in sample
    or len(sample[self.text_key_second]) == 0
):
    sample[Fields.stats][StatsKeys.text_pair_similarity] = np.array([], dtype=np.float64)
    return sample

# 3. 获取模型
model, processor = get_model(self.model_key, rank, self.use_cuda())

# 4. 获取两个文本
text1 = sample[self.text_key]
text2 = sample[self.text_key_second]

# 5. 提取文本特征
text_tensors = processor([text1, text2], padding=True, return_tensors="pt").to(model.device)
text_features = model.get_text_features(**text_tensors)

# 6. 计算余弦相似度
similarity = torch.cosine_similarity(text_features[0], text_features[1], dim=0)

# 7. 存储结果
sample[Fields.stats][StatsKeys.text_pair_similarity] = [similarity]
```

### 7. 图像对相似度(image_pair_similarity) - <mcfile name="image_pair_similarity_filter.py" path="/home/czx/PycharmProjects/data-juicer/data_juicer/ops/filter/image_pair_similarity_filter.py"></mcfile>

**实现流程：**
```python
# 1. 检查指标是否已计算
if StatsKeys.image_pair_similarity in sample[Fields.stats]:
    return sample

# 2. 验证是否包含两个不同图像
if (
    self.image_key not in sample
    or not len(sample[self.image_key]) == 2
    or sample[self.image_key][0] == sample[self.image_key][1]
):
    raise ValueError("Each sample must include two images.")

# 3. 加载图像
loaded_image_keys = sample[self.image_key]
sample, images = load_data_with_context(
    sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
)

# 4. 获取模型
model, processor = get_model(self.model_key, rank, self.use_cuda())

# 5. 提取图像特征
image_list = []
for temp_key in images.keys():
    image_list.append(images[temp_key])
image_tensors = processor.image_processor(image_list, return_tensors="pt")["pixel_values"]
image1_batch_feature = model.get_image_features(image_tensors[0].unsqueeze(0).to(model.device))
image2_batch_feature = model.get_image_features(image_tensors[1].unsqueeze(0).to(model.device))

# 6. 计算余弦相似度
similarity = torch.cosine_similarity(image1_batch_feature, image2_batch_feature, dim=1)

# 7. 存储结果
sample[Fields.stats][StatsKeys.image_pair_similarity] = similarity.cpu()
```

## 共同模式与特点

1. **统一的缓存机制**：所有指标都先检查是否已在`sample[Fields.stats]`中计算过，避免重复计算

2. **模型复用**：
   - CLIP模型被广泛用于多种相似度计算
   - BLIP模型专用于图文匹配评分
   - Owl-ViT模型专用于短语定位

3. **灵活的聚合策略**：大多数多模态指标支持`avg`、`max`、`min`三种聚合模式

4. **内存优化**：通过上下文缓存和资源释放（如关闭视频）优化内存使用

5. **多GPU支持**：通过`rank`参数支持分布式处理

6. **统一的筛选策略**：大多数过滤器支持`any`或`all`筛选策略

这些相似度指标为多模态数据的质量评估和过滤提供了全面的量化依据，支持从不同维度评估数据质量。
        