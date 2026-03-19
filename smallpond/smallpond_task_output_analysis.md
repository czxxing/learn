# SmallPond Task 输出数据到磁盘机制详细分析

## 1. 概述

在 SmallPond 中，Task 之间通过磁盘上的 Parquet 文件传递数据。每个 Task 执行完成后，需要将其输出数据集（DataSet）保存到磁盘，供下游 Task 使用。

本文档详细分析：
1. 哪些 Task 会将数据输出到磁盘
2. 各 Task 的数据写入方式
3. `dump()` 函数的执行内容

## 2. Task 执行位置与 dump 关系

### 2.1 关键属性：exec_on_scheduler

SmallPond 通过 `exec_on_scheduler` 属性区分任务的执行位置：

| 属性值 | 执行位置 | 是否执行 dump |
|--------|----------|--------------|
| `True` | 调度器节点（主节点） | ❌ 不执行 dump |
| `False`（默认） | Ray Worker（从节点） | ✅ 执行 dump |

定义在 [workqueue.py:100-101](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/workqueue.py#L100-L101)：

```python
# WorkItem 基类
@property
def exec_on_scheduler(self) -> bool:
    return False  # 默认在 Worker 上执行
```

### 2.2 dump 执行时机

`dump()` 只在 `run_on_ray()` 中被调用：

```python
# task.py:990-1031
@ray.remote
def exec_task(task: Task, *inputs: DataSet) -> DataSet:
    # ... 执行任务 ...
    
    status = task.exec()
    if status != WorkStatus.SUCCEED:
        raise task.exception
    
    # ★ 核心：保存输出到磁盘
    os.makedirs(os.path.dirname(task.ray_dataset_path), exist_ok=True)
    dump(task.output, task.ray_dataset_path, atomic_write=True)
    return task.output
```

## 3. 输出数据到磁盘的 Task 类型

### 3.1 执行 dump 的 Task（exec_on_scheduler=False）

| Task 类型 | 说明 | 数据写入方式 |
|----------|------|-------------|
| `SqlEngineTask` | 执行 SQL 查询 | DuckDB `COPY TO` 写入 Parquet |
| `ArrowComputeTask` | Arrow 计算任务 | Arrow Table 写入 Parquet |
| `HashPartitionTask` | 哈希分区任务 | 分区数据写入磁盘 |
| `DataSinkTask` | 数据 Sink 任务 | 收集/复制/链接文件 |
| 其他大部分 Task | 在 Worker 上执行 | 各自的数据写入逻辑 |

### 3.2 不执行 dump 的 Task（exec_on_scheduler=True）

| Task 类型 | 说明 | 执行位置 |
|----------|------|----------|
| `DataSourceTask` | 数据源任务 | 调度器（只读） |
| `MergeDataSetsTask` | 合并数据集任务 | 调度器（逻辑任务） |
| `EvenlyDistributedPartitionProducerTask` | 均匀分区生产者 | 调度器（分区逻辑） |
| `RootTask` | 根任务（虚拟） | 调度器 |

## 4. 详细分析：SqlEngineTask

### 4.1 关键参数：materialize_output

SqlEngineTask 通过 `materialize_output` 参数控制是否将数据写入磁盘：

```python
# task.py:2228-2232
self.materialize_output = materialize_output  # 默认为 True
self.materialize_in_memory = materialize_in_memory  # 默认为 False
```

### 4.2 materialize_output=True（默认）

将 SQL 结果写入 Parquet 文件：

```python
# task.py:2340-2357
if last_query and self.materialize_output:
    sql_query = f"""
COPY (
  {sql_query}
) TO '{output_path}' (
    FORMAT PARQUET,
    KV_METADATA {self.parquet_kv_metadata_str()},
    {self.compression_options},
    ROW_GROUP_SIZE {self.parquet_row_group_size},
    ROW_GROUP_SIZE_BYTES {self.parquet_row_group_bytes},
    {dictionary_encoding_cfg}
    PER_THREAD_OUTPUT {self.per_thread_output},
    FILENAME_PATTERN '{output_filename}.{{i}}',
    OVERWRITE_OR_IGNORE true)
"""
```

### 4.3 materialize_in_memory=True

在内存中执行查询，不写入磁盘：

```python
# task.py:2324-2332
if last_query and self.materialize_in_memory:
    # 在内存中创建临时表
    self.merge_metrics(
        self.exec_query(conn, f"EXPLAIN ANALYZE CREATE OR REPLACE TEMP TABLE temp_query_result AS {sql_query}")
    )
    # 用 select * 返回结果，不写入磁盘
    sql_query = f"select * from temp_query_result"
```

### 4.4 完整流程

```
SqlEngineTask.run()
    │
    ├──► 创建 DuckDB 内存连接
    │
    ├──► process_batch()
    │       │
    │       ├──► create_input_views()  注册输入数据为视图
    │       │
    │       ├──► 格式化 SQL 查询
    │       │
    │       └──► 执行查询
    │               │
    │               └──► materialize_output=True
    │                       │
    │                       └──► COPY TO 'output_path' (Parquet)
    │
    └──► 设置 self.dataset = ParquetDataSet([...], root_dir=output_path)
```

## 5. 详细分析：HashPartitionTask

### 5.1 分区写入方式

HashPartitionTask 将数据按哈希值分配到 N 个分区：

```python
# task.py:2600-2610
@property
def partition_query(self):
    if self.shuffle_only:
        partition_query = r"SELECT * FROM {0}"
    else:
        if self.random_shuffle:
            # 随机分区
            hash_values = f"random() * {2147483647 // self.npartitions * self.npartitions}"
        else:
            # 按列哈希分区
            hash_values = f"hash( concat_ws( '##', {', '.join(self.hash_columns)} ) )"
        
        partition_keys = f"CAST({hash_values} AS UINT64) % {self.npartitions}::UINT64 AS {self.data_partition_column}"
```

### 5.2 数据写入实现

两种写入方式：

**Hive 分区写入**：
```python
# task.py:2665-2683
def write_hive_partitions(self, conn, batch_index, input_dataset):
    copy_query_result = f"""
COPY (
  SELECT * FROM temp_query_result
) TO '{self.runtime_output_abspath}' (
    FORMAT PARQUET,
    OVERWRITE_OR_IGNORE,
    WRITE_PARTITION_COLUMNS,
    PARTITION_BY {self.data_partition_column},
    ...
)"""
```

**扁平分区写入**：
```python
# task.py:2700-2750
def write_flat_partitions(self, conn, batch_index, input_dataset):
    # 为每个分区生成过滤查询
    partition_filters = [
        (
            partition_idx,
            f"SELECT * FROM temp_query_result WHERE {self.data_partition_column} = {partition_idx}"
        )
        for partition_idx in range(self.npartitions)
    ]
    
    # 并行写入各分区
    with contextlib.ExitStack() as stack:
        db_conns = [stack.enter_context(conn.cursor()) for _ in range(self.num_workers)]
        self.perf_metrics["num output rows"] += sum(
            self.io_workers.map(write_partition_data, db_conns, partition_batches)
        )
```

### 5.3 分区文件结构

**扁平分区**：
```
output/
 ├── partition_0-0.parquet
 ├── partition_0-1.parquet
 ├── partition_1-0.parquet
 └── ...
```

**Hive 分区**：
```
output/
 ├── __partition_key__=0/
 │   ├── data-0.parquet
 │   └── data-1.parquet
 ├── __partition_key__=1/
 │   └── data-0.parquet
 └── ...
```

## 6. dump() 函数执行内容

### 6.1 函数定义

定义在 [filesystem.py:58-122](file:///home/czx/PycharmProjects/smallpond/smallpond/io/filesystem.py#L58-L122)：

```python
def dump(obj: Any, path: str, buffering=2 * MB, atomic_write=False) -> int:
    def write_to_file(fout):
        # 使用 zstd 压缩 + cloudpickle 序列化
        with zstd.ZstdCompressor().stream_writer(fout, closefd=False) as zstd_writer:
            cloudpickle.dump(obj, zstd_writer)
    
    if atomic_write:
        # 原子写入：先写临时文件，再重命名
        directory, filename = os.path.split(path)
        with tempfile.NamedTemporaryFile("wb", dir=directory, prefix=filename, delete=False) as fout:
            write_to_file(fout)
            size = fout.tell()
            os.rename(fout.name, path)  # 原子操作
    else:
        with open(path, "wb", buffering=buffering) as fout:
            write_to_file(fout)
            size = fout.tell()
    
    return size
```

### 6.2 序列化流程

```
task.output (DataSet 对象)
        │
        ▼ cloudpickle.serialize()
        │   (将 Python 对象序列化为字节流)
        ▼
        ▼ zstd.compress()
        │   (使用 zstd 压缩算法压缩)
        ▼
        ▼ 写入临时文件 (tempfile.NamedTemporaryFile)
        │   (先写到临时文件，保证完整性)
        ▼
        ▼ os.rename() 原子重命名
        │   (原子操作，确保文件要么完全存在，要么不存在)
        ▼
        ▼ 最终文件: {node_id}/{task_key}.pickle
```

### 6.3 atomic_write=True 的作用

| 步骤 | 说明 |
|------|------|
| 1. 创建临时文件 | 使用 `tempfile.NamedTemporaryFile` |
| 2. 序列化写入 | 将 DataSet 序列化并压缩写入临时文件 |
| 3. 获取文件大小 | `fout.tell()` |
| 4. 原子重命名 | `os.rename()` 是原子操作 |
| 5. 结果 | 文件要么完全存在，要么完全不存在 |

### 6.4 对应的读取操作

```python
# task.py:985-988
if os.path.exists(self.ray_dataset_path):
    logger.info(f"task {self.key} already finished, skipping")
    output = load(self.ray_dataset_path)  # 对应的读取
    self._dataset_ref = ray.put(output)
    return self._dataset_ref
```

读取函数：

```python
# filesystem.py:125-131
def load(path: str, buffering=2 * MB) -> Any:
    with open(path, "rb", buffering=buffering) as fin:
        with zstd.ZstdDecompressor().stream_reader(fin) as zstd_reader:
            obj = cloudpickle.load(zstd_reader)
            return obj
```

## 7. Task.output 属性

每个 Task 需要设置 `self.dataset` 属性，作为任务的输出：

```python
# task.py:700-702
@property
def output(self) -> DataSet:
    return self.dataset or ParquetDataSet(["*"], root_dir=self.final_output_abspath)
```

### 7.1 SqlEngineTask 设置 output

```python
# task.py:2365-2367
# 在 process_batch 中执行 COPY TO 后，
# self.dataset 会被设置为输出的 ParquetDataSet
```

### 7.2 HashPartitionTask 设置 output

```python
# task.py:2520-2528
def initialize(self):
    # 初始化分区数据集列表
    self.partitioned_datasets = [
        ParquetDataSet([], root_dir=self.runtime_output_abspath)
        for _ in range(self.npartitions)
    ]

# 写入数据时更新
self.partitioned_datasets[partition_idx].paths.append(partition_filename)
```

## 8. 执行流程图

### 8.1 完整数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Ray Worker 执行环境                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  exec_task(task, *inputs)                                                   │
│         │                                                                   │
│         ▼                                                                   │
│  task.input_datasets = inputs                                               │
│         │                                                                   │
│         ▼                                                                   │
│  status = task.exec()                                                       │
│         │                                                                   │
│         ├──► task.initialize()                                             │
│         │                                                                   │
│         ├──► task.run()  ★ 核心执行逻辑                                     │
│         │         │                                                         │
│         │         ├──► SqlEngineTask.run()                                  │
│         │         │         │                                               │
│         │         │         ├──► DuckDB 连接                                 │
│         │         │         ├──► COPY TO Parquet  ★ 写入磁盘                │
│         │         │         └──► self.dataset = ParquetDataSet([...])      │
│         │         │                                                         │
│         │         └──► HashPartitionTask.run()                              │
│         │                   │                                               │
│         │                   ├──► 哈希分区计算                               │
│         │                   ├──► 分区写入磁盘 ★                            │
│         │                   └──► self.dataset = partitioned_datasets       │
│         │                                                                   │
│         ├──► task.finalize()                                                │
│         │                                                                   │
│         ▼                                                                   │
│  dump(task.output, ray_dataset_path, atomic_write=True) ★ 序列化保存       │
│         │                                                                   │
│         ├──► cloudpickle.dump()  序列化                                     │
│         ├──► zstd.compress()     压缩                                        │
│         ├──► 写临时文件                                                     │
│         └──► os.rename()        原子重命名                                   │
│                                                                             │
│         ▼                                                                   │
│  return task.output                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 数据传递流程

```
                    ┌─────────────────────────────┐
                    │    上游 Task                │
                    │   (HashPartitionTask)      │
                    │                             │
                    │  partition_0.parquet       │
                    │  partition_1.parquet       │
                    │  partition_2.parquet       │
                    └──────────────┬──────────────┘
                                   │
                                   │ dump() 保存为 pickle
                                   │ task.output (ParquetDataSet)
                                   ▼
                    ┌─────────────────────────────┐
                    │   磁盘: node_id/task.pickle │
                    │   (压缩的 DataSet 对象)     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │    下游 Task                 │
                    │   (SqlEngineTask)           │
                    │                             │
                    │  ray.get() 读取 pickle      │
                    │  load() 解压反序列化         │
                    │                             │
                    │  self.input_datasets        │
                    │  = [ParquetDataSet(...)]    │
                    └─────────────────────────────┘
```

## 9. 总结

### 9.1 Task 输出磁盘机制

| 组件 | 说明 |
|------|------|
| **执行位置** | `exec_on_scheduler=False` 的 Task 在 Ray Worker 上执行 |
| **dump 触发** | `run_on_ray()` 中调用 `dump(task.output, ...)` |
| **写入内容** | 序列化的 DataSet 对象（ParquetDataSet） |
| **序列化** | cloudpickle + zstd 压缩 |
| **原子写入** | 临时文件 + os.rename() |

### 9.2 SqlEngineTask 输出控制

| 参数 | 行为 |
|------|------|
| `materialize_output=True`（默认） | SQL 结果写入 Parquet 文件 |
| `materialize_in_memory=True` | 结果保留在内存临时表中 |

### 9.3 数据流转

```
Task A 执行
    │
    ▼ run() 中写入 Parquet 文件
    │
    ▼ 设置 self.dataset
    │
    ▼ dump() 序列化保存为 pickle
    │
    ▼ 下游 Task 读取
    │
    ▼ load() 反序列化
    │
    ▼ 设置 self.input_datasets
```