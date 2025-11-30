


          
让我搜索data_juicer/ops目录下与duplicate相关的代码实现，以分析其执行逻辑。
        
            
toolName: search_codebase
            
status: success
          
query: data_juicer/ops目录下duplicate操作相关的代码实现，包括去重操作符的定义和执行逻辑
          
让我查看base_op.py中Deduplicator类的完整实现，这是所有去重操作符的基类。
        
            
toolName: view_files
            
status: success
          
            
filePath: /home/czx/PycharmProjects/data-juicer/data_juicer/ops/base_op.py
          
让我查看document_deduplicator.py文件的完整实现，以了解具体的去重执行逻辑。
        
            
toolName: view_files
            
status: success
          
            
filePath: /home/czx/PycharmProjects/data-juicer/data_juicer/ops/deduplicator/document_deduplicator.py
          
让我查看ray_basic_deduplicator.py文件，了解分布式去重的实现机制。
        
            
toolName: view_files
            
status: success
          
            
filePath: /home/czx/PycharmProjects/data-juicer/data_juicer/ops/deduplicator/ray_basic_deduplicator.py
          
让我查看document_simhash_deduplicator.py文件，了解基于SimHash的近似去重实现。
        
            
toolName: view_files
            
status: success
          
            
filePath: /home/czx/PycharmProjects/data-juicer/data_juicer/ops/deduplicator/document_simhash_deduplicator.py
          
让我查看document_simhash_deduplicator.py文件的剩余部分，以获取完整的process方法实现。
        
            
toolName: view_files
            
status: success
          
            
filePath: /home/czx/PycharmProjects/data-juicer/data_juicer/ops/deduplicator/document_simhash_deduplicator.py
          
# Operator中Deduplicator的执行逻辑分析

## 1. Deduplicator基类设计

`Deduplicator`是所有去重操作符的基类，定义在`base_op.py`中，继承自`OP`类：

```python
class Deduplicator(OP):
    def __init__(self, *args, **kwargs):
        super(Deduplicator, self).__init__(*args, **kwargs)
        # 根据批处理模式应用异常捕获装饰器
        if self.is_batched_op():
            self.compute_hash = catch_map_batches_exception(...)
        else:
            self.compute_hash = catch_map_single_exception(...)

    def compute_hash(self, sample):
        # 计算样本哈希值的抽象方法
        raise NotImplementedError

    def process(self, dataset, show_num=0):
        # 处理数据集并返回去重后的数据集和重复对
        raise NotImplementedError

    def run(self, dataset, *, exporter=None, tracer=None, reduce=True):
        # 执行流程：先运行父类初始化，再计算哈希，最后处理去重
        dataset = super(Deduplicator, self).run(dataset)
        new_dataset = dataset.map(self.compute_hash, ...)
        if reduce:
            show_num = tracer.show_num if tracer else 0
            new_dataset, dup_pairs = self.process(new_dataset, show_num)
            if tracer:
                tracer.trace_deduplicator(self._name, dup_pairs)
        free_models()
        return new_dataset
```

**核心执行流程**：
1. 初始化时应用异常处理装饰器
2. `run`方法实现两阶段执行模式：先计算哈希，再进行去重处理
3. 子类必须实现`compute_hash`和`process`两个抽象方法
4. 支持跟踪器记录去重过程中的重复样本对

## 2. DocumentDeduplicator：精确匹配去重

`DocumentDeduplicator`实现基于MD5哈希的精确文本匹配去重：

```python
@OPERATORS.register_module("document_deduplicator")
class DocumentDeduplicator(Deduplicator):
    def __init__(self, lowercase: bool = False, ignore_non_character: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lowercase = lowercase
        self.remove_non_character_regex = re.compile(...) if ignore_non_character else None

    def compute_hash(self, sample):
        # 跳过已计算的哈希
        if HashKeys.hash in sample:
            return sample
        
        text = sample[self.text_key]
        # 文本预处理
        if self.lowercase:
            text = text.lower()
        if self.remove_non_character_regex:
            text = self.remove_non_character_regex.sub("", text)
        
        # 计算MD5哈希
        sample[HashKeys.hash] = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
        return sample

    def process(self, dataset, show_num=0):
        # 样本过少时无需去重
        if len(dataset) <= 1:
            return dataset, {}
        
        # 样本重复对用于追踪
        dup_hashes = None
        if show_num > 0:
            # 构建哈希到ID的映射并找出重复样本
            hash2ids = defaultdict(set)
            for sid, hash_val in enumerate(dataset[HashKeys.hash]):
                hash2ids[hash_val].add(sid)
            dup_samples = sorted(..., key=lambda x: len(x[1]), reverse=True)
            dup_hashes = set([...])
        
        # 过滤函数：保留首次出现的哈希值
        def _filter_dup_helper(sample, hashes):
            hash_val = sample[HashKeys.hash]
            # 收集重复对用于追踪
            if show_num > 0 and hash_val in dup_hashes and len(dup_pairs[hash_val]) < 2:
                dup_pairs[hash_val].append(sample)
            # 保留唯一哈希
            if hash_val in hashes:
                return False
            else:
                hashes.add(hash_val)
                return True
        
        hashes = set()
        dup_pairs = {...}
        # 执行过滤
        dataset = dataset.filter(_filter_dup_helper, ...)
        return dataset, dup_pairs
```

**关键特性**：
- 支持文本预处理选项：转小写、忽略非字母字符
- 使用MD5哈希进行精确匹配
- 保留首次出现的样本，过滤后续重复项
- 支持收集重复样本对用于追踪分析

## 3. RayBasicDeduplicator：分布式去重

`RayBasicDeduplicator`是一个特殊实现，虽然功能是去重，但继承自`Filter`类而非`Deduplicator`类：

```python
class RayBasicDeduplicator(Filter):
    def __init__(self, backend: str = "ray_actor", redis_address: str = "redis://localhost:6379", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 根据配置选择后端
        if backend == "ray_actor":
            dedup_set_num = int(ray.cluster_resources().get("CPU") / 2)
            self.backend = ActorBackend(dedup_set_num)
        elif backend == "redis":
            self.backend = RedisBackend(redis_address)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def compute_stats_single(self, sample, context=False):
        # 计算哈希并检查是否唯一
        md5_value = self.calculate_hash(sample, context)
        sample[HashKeys.is_unique] = self.backend.is_unique(md5_value)
        return sample

    def process_single(self, sample):
        # 基于is_unique标志决定是否保留样本
        return sample[HashKeys.is_unique]
```

**分布式后端设计**：
1. **ActorBackend**：使用Ray Actor实现分布式去重
   - 创建多个DedupSet Actor实例
   - 使用一致性哈希将哈希值分配到不同Actor
   - 通过`ray.get`获取远程调用结果

2. **RedisBackend**：使用Redis实现分布式去重
   - 连接到Redis服务器
   - 使用`SETNX`命令实现原子性检查和插入

## 4. DocumentSimhashDeduplicator：近似去重

`DocumentSimhashDeduplicator`实现基于SimHash的近似文本去重：

```python
@OPERATORS.register_module(OP_NAME)
class DocumentSimhashDeduplicator(Deduplicator):
    def __init__(self, tokenization: str = "space", window_size: int = 6, lowercase: bool = True,
                 ignore_pattern: Optional[str] = None, num_blocks: int = 6, hamming_distance: int = 4,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化SimHash计算参数
        self.tokenization = tokenization
        self.window_size = window_size
        self.lowercase = lowercase
        # 其他参数初始化...

    def compute_hash(self, sample):
        # 跳过已计算的哈希
        if HashKeys.simhash in sample:
            return sample
        
        text = sample[self.text_key]
        # 文本预处理
        if self.lowercase:
            text = text.lower()
        if self.ignore_pattern:
            text = self.ignore_pattern.sub("", text)
        
        # 根据分词方式生成shingles
        tokens = []
        if self.tokenization == "character":
            # 字符级分词
            tokens = [str.encode(text[i:i+self.window_size]) for i in range(len(text)-self.window_size+1)]
        elif self.tokenization == "punctuation":
            # 标点分词
            tokens = self.punctuation_pattern.split(text)
            tokens = [str.encode(" ".join(tokens[i:i+self.window_size])) for i in range(len(tokens)-self.window_size+1)]
        elif self.tokenization == "space":
            # 空格分词
            tokens = split_on_whitespace(text)
            tokens = [str.encode(" ".join(tokens[i:i+self.window_size])) for i in range(len(tokens)-self.window_size+1)]
        
        # 计算SimHash
        sample[HashKeys.simhash] = str(np.uint64(simhash.compute(map(simhash.unsigned_hash, tokens))))
        return sample

    def process(self, dataset, show_num=0):
        # 样本过少时无需去重
        if len(dataset) <= 1:
            return dataset, {}
        
        # 使用SimHash库查找相似样本
        matches = simhash.find_all(np.uint64(dataset[HashKeys.simhash]), self.num_blocks, self.hamming_distance)
        
        # 构建相似图和聚类
        graph = defaultdict(dict)
        for x, y in matches:
            graph[str(x)][str(y)] = graph[str(y)][str(x)] = True
        
        # 构建哈希到ID的映射
        hash2ids = defaultdict(set)
        for sid, hash_val in enumerate(dataset[HashKeys.simhash]):
            hash2ids[hash_val].add(str(sid))
        
        # BFS聚类相似文档
        hash2cluster = {}
        visited = set()
        cluster_id = 0
        dup_pairs = {}
        
        # 对每个哈希值进行BFS聚类
        hashes = set(dataset[HashKeys.simhash])
        while hashes:
            hash_val = hashes.pop()
            if hash_val in visited:
                continue
            
            if hash_val not in graph:
                continue
            
            # BFS遍历相似图
            q = deque([hash_val])
            visited.add(hash_val)
            hash2cluster[hash_val] = cluster_id
            # 收集重复对
            if show_num > 0 and len(dup_pairs) < show_num:
                dup_pairs[cluster_id] = []
            
            while q:
                curr = q.popleft()
                for neighbor in graph[curr]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    q.append(neighbor)
                    hash2cluster[neighbor] = cluster_id
            
            cluster_id += 1
        
        # 过滤函数：每个聚类只保留第一个样本
        def _filter_simhash_dup_helper(sample, visited_clusters, visited_hashes):
            sample_hash_val = sample[HashKeys.simhash]
            if sample_hash_val not in hash2cluster:
                # 单样本聚类
                if sample_hash_val in visited_hashes:
                    return False
                else:
                    visited_hashes.add(sample_hash_val)
                    return True
            else:
                # 常规聚类
                cluster_num = hash2cluster[sample_hash_val]
                # 收集重复对
                if show_num > 0 and cluster_num in dup_pairs and len(dup_pairs[cluster_num]) < 2:
                    dup_pairs[cluster_num].append(sample)
                # 检查聚类是否已访问
                if cluster_num in visited_clusters:
                    return False
                else:
                    visited_clusters.add(cluster_num)
                    return True
        
        # 执行过滤
        cluster_record = set()
        hash_record = set()
        dataset = dataset.filter(_filter_simhash_dup_helper, ...)
        return dataset, dup_pairs
```

**核心技术特点**：
- 支持三种分词方式：字符级、标点级、空格级
- 使用滑动窗口生成shingles
- 通过Hamming距离阈值判断文本相似性
- 采用图结构和BFS算法进行相似文档聚类
- 每个聚类仅保留第一个样本

## 5. 执行流程总结

**Deduplicator的完整执行流程**：

1. **初始化阶段**：
   - 配置去重参数（哈希方法、相似度阈值等）
   - 设置异常处理装饰器
   - 初始化分布式后端（如需）

2. **哈希计算阶段**：
   - 通过`dataset.map`并行计算所有样本的哈希值
   - 根据配置进行文本预处理（转小写、去除非字符等）
   - 支持缓存已计算的哈希值避免重复计算

3. **去重处理阶段**：
   - **精确匹配**：直接比较哈希值，保留首次出现的样本
   - **近似匹配**：构建相似图、聚类，每个聚类保留一个样本
   - **分布式处理**：使用Ray Actor或Redis进行大规模数据去重

4. **结果追踪**：
   - 收集重复样本对用于分析
   - 通过tracer记录去重统计信息

## 6. 技术特点总结

1. **分层设计**：基类定义通用流程，子类实现特定哈希算法
2. **两阶段执行**：哈希计算与去重处理分离，支持并行优化
3. **多样化实现**：支持精确匹配和近似匹配多种去重策略
4. **分布式支持**：通过Ray和Redis实现大规模数据去重
5. **灵活配置**：支持文本预处理、相似度阈值等多维度参数调整
6. **追踪机制**：内置重复样本收集和统计功能，便于分析和调试

这种设计使得Deduplicator能够灵活适应不同的去重需求，从简单的精确匹配到复杂的近似去重，从小规模数据到大规模分布式处理，都能高效执行。
        