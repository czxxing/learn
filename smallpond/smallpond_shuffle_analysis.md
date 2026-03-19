# SmallPond 数据 Shuffle 机制详细分析

## 1. 概述

在分布式数据处理中，**Shuffle** 是指将数据按照某种规则重新分配到不同分区的过程。SmallPond 提供了多种分区策略来实现数据 shuffle，本文将详细分析其实现机制。

## 2. Shuffle 核心组件

### 2.1 逻辑层 Node

| Node 类型 | 作用 |
|-----------|------|
| `EvenlyDistributedPartitionNode` | 均匀分区，按文件或行数均匀分配 |
| `HashPartitionNode` | 哈希分区，按指定列的哈希值分配 |
| `ShuffleNode` | 纯 shuffle，不进行哈希计算，直接按分区列分配 |
| `DataSetPartitionNode` | 统一入口，根据参数选择不同分区策略 |

### 2.2 执行层 Task

| Task 类型 | 作用 |
|-----------|------|
| `EvenlyDistributedPartitionProducerTask` | 均匀分区执行器 |
| `HashPartitionTask` | 哈希分区基类 |
| `HashPartitionDuckDbTask` | DuckDB 引擎的哈希分区 |
| `HashPartitionArrowTask` | Arrow 引擎的哈希分区 |

## 3. 分区策略详解

### 3.1 均匀分区 (Evenly Distributed Partitioning)

均匀分区将数据**尽可能均匀**地分配到各个分区，不改变数据的顺序。

```
原始数据: [file1, file2, file3, file4, file5, file6]
分区数: 3

均匀分区结果:
分区0: [file1, file2]
分区1: [file3, file4]
分区2: [file5, file6]
```

#### 核心代码

**EvenlyDistributedPartitionNode** ([node.py:L1379-L1510](file:///home/czx/PycharmProjects/smallpond/smallpond/logical/node.py#L1379-L1510)):

```python
class EvenlyDistributedPartitionNode(PartitionNode):
    @Node.task_factory
    def create_producer_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ):
        return EvenlyDistributedPartitionProducerTask(
            runtime_ctx,
            input_deps,
            partition_infos,
            self.npartitions,
            self.dimension,
            self.partition_by_rows,
            self.random_shuffle,
            self.cpu_limit,
            self.memory_limit,
        )
```

**EvenlyDistributedPartitionProducerTask.run()** ([task.py:L1518-L1526](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L1518-L1526)):

```python
def run(self) -> bool:
    input_dataset = self.input_datasets[0]
    if isinstance(input_dataset, ParquetDataSet) and self.partition_by_rows:
        # 按行数均匀分区
        self.partitioned_datasets = input_dataset.partition_by_rows(self.npartitions, self.random_shuffle)
    else:
        # 按文件均匀分区
        self.partitioned_datasets = input_dataset.partition_by_files(self.npartitions, self.random_shuffle)
    return True
```

### 3.2 哈希分区 (Hash Partitioning)

哈希分区根据指定列的**哈希值**将数据分配到不同分区相同哈希值的数据会被分配到同一分区。

```
原始数据:
| id | name  |
|----|-------|
| 1  | Alice |
| 2  | Bob   |
| 3  | Carol |
| 4  | Dave  |
| 5  | Eve   |

分区数: 2, 按 id 列哈希

哈希计算: hash(id) % 2
id=1 → hash(1)%2=1 → 分区1
id=2 → hash(2)%2=0 → 分区0
id=3 → hash(3)%2=1 → 分区1
id=4 → hash(4)%2=0 → 分区0
id=5 → hash(5)%2=1 → 分区1

分区0: [id=2, name=Bob], [id=4, name=Dave]
分区1: [id=1, name=Alice], [id=3, name=Carol], [id=5, name=Eve]
```

#### 核心代码

**HashPartitionNode** ([node.py:L1558-L1698](file:///home/czx/PycharmProjects/smallpond/smallpond/logical/node.py#L1558-L1698)):

```python
class HashPartitionNode(PartitionNode):
    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        npartitions: int,
        hash_columns: List[str] = None,
        data_partition_column: str = None,
        *,
        random_shuffle: bool = False,      # 随机分区
        shuffle_only: bool = False,          # 仅 shuffle，不哈希
        drop_partition_column: bool = False,
        use_parquet_writer: bool = False,
        hive_partitioning: bool = False,
        # ... 其他参数
    ):
        # random_shuffle: 使用 random() 作为哈希函数
        # shuffle_only: 直接使用 data_partition_column 作为分区键
        self.hash_columns = ["random()"] if random_shuffle else hash_columns
        self.random_shuffle = random_shuffle
        self.shuffle_only = shuffle_only
```

**HashPartitionDuckDbTask.partition_query** ([task.py:L2605-L2620](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2605-L2620)):

```python
@property
def partition_query(self):
    if self.shuffle_only:
        # shuffle_only: 直接使用分区列
        partition_query = r"SELECT * FROM {0}"
    else:
        if self.random_shuffle:
            # random_shuffle: 使用 random() 函数
            hash_values = f"random() * {2147483647 // self.npartitions * self.npartitions}"
        else:
            # 按指定列哈希
            hash_values = f"hash( concat_ws( '##', {', '.join(self.hash_columns)} ) )"
        
        # 计算分区键
        partition_keys = f"CAST({hash_values} AS UINT64) % {self.npartitions}::UINT64 AS {self.data_partition_column}"
        
        partition_query = f"""
  SELECT *,
    {partition_keys}
  FROM (
    SELECT COLUMNS(c -> c != '{self.data_partition_column}') FROM {{0}}
  )"""
    return partition_query
```

### 3.3 Shuffle Only

`shuffle_only=True` 模式不进行哈希计算，直接使用 `data_partition_column` 列的值作为分区键。

```
原始数据:
| id | category | value |
|----|----------|-------|
| 1  | A        | 100   |
| 2  | B        | 200   |
| 3  | A        | 150   |
| 4  | C        | 300   |

分区数: 3, shuffle_only=True, data_partition_column="category"

分区A (category='A'): [id=1, value=100], [id=3, value=150]
分区B (category='B'): [id=2, value=200]
分区C (category='C'): [id=4, value=300]
```

## 4. Shuffle 执行流程

### 4.1 Planner 生成 Task

在 [planner.py:L144-L210](file:///home/czx/PycharmProjects/smallpond/smallpond/logical/planner.py#L144-L210) 中，Planner 为分区节点生成 Producer Task 和 Consumer Task：

```python
def visit_partition_node(self, node: PartitionNode, depth: int) -> TaskGroup:
    all_input_deps = [task for dep in node.input_deps for task in self.visit(dep)]
    
    # 1. 计算需要的 producer 任务数量
    num_producer_tasks = max(1, min(max_num_producer_tasks, num_parallel_tasks))
    
    # 2. 创建 producer tasks（执行分区逻辑）
    producer_tasks = [
        node.create_producer_task(runtime_ctx, [split_dataset], partition_infos)
        for split_dataset in split_dataset_tasks
    ]
    
    # 3. 为每个分区创建 consumer tasks
    return [
        node.create_consumer_task(runtime_ctx, producer_tasks, [
            PartitionInfo(),
            PartitionInfo(partition_idx, node.npartitions, node.dimension),
        ])
        for partition_idx in range(node.npartitions)
    ]
```

### 4.2 HashPartitionTask 执行流程

**HashPartitionDuckDbTask.run()** ([task.py:L2600-L2610](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2600-L2610)):

```python
def run(self) -> bool:
    input_dataset = self.input_datasets[0]
    input_batches = input_dataset.partition_by_size(self.max_batch_size)

    for batch_index, input_batch in enumerate(input_batches):
        # 使用 DuckDB 执行分区
        self.partition(batch_index, input_batch)

    return True
```

**HashPartitionDuckDbTask.partition()** ([task.py:L2626-L2645](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2626-L2645)):

```python
def partition(self, batch_index: int, input_dataset: ParquetDataSet):
    with duckdb.connect(database=":memory:") as conn:
        self.prepare_connection(conn)
        
        if self.hive_partitioning:
            # Hive 风格分区输出
            self.load_input_batch(conn, batch_index, input_dataset, sort_by_partition_key=True)
            self.write_hive_partitions(conn, batch_index, input_dataset)
        else:
            # 扁平分区输出
            self.load_input_batch(conn, batch_index, input_dataset, sort_by_partition_key=True)
            self.write_flat_partitions(conn, batch_index, input_dataset)
```

### 4.3 数据写入分区

**write_flat_partitions()** ([task.py:L2700-L2760](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2700-L2760)):

```python
def write_flat_partitions(self, conn, batch_index, input_dataset):
    # 为每个分区生成过滤查询
    partition_filters = [
        (
            partition_idx,
            f"SELECT * FROM temp_query_result WHERE {self.data_partition_column} = {partition_idx}"
        )
        for partition_idx in range(self.npartitions)
    ]
    
    # 并行写入各个分区
    with contextlib.ExitStack() as stack:
        db_conns = [stack.enter_context(conn.cursor()) for _ in range(self.num_workers)]
        self.perf_metrics["num output rows"] += sum(
            self.io_workers.map(write_partition_data, db_conns, partition_batches)
        )
```

## 5. 数据流转示例

### 完整 Shuffle 流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           原始 Parquet 数据集                                │
│                     1000 万行，100 个文件                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DataSetPartitionNode (npartitions=10)                    │
│                    均匀分区 by rows                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  分区0: 100万行  分区1: 100万行  ...  分区9: 100万行                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HashPartitionNode (npartitions=20)                        │
│                    按 id 列哈希分区                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  输入: 10 个分区数据                                                         │
│  处理: 对每条记录计算 hash(id) % 20                                          │
│  输出: 20 个分区文件 (每个分区包含来自多个输入分区的数据)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SqlEngineNode                                             │
│                    对每个分区执行 SQL 处理                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  分区0-19 分别在独立进程中处理                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    最终输出                                                  │
│  按哈希分区存储的 Parquet 文件                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Shuffle 示例代码

来自 [examples/shuffle_data.py](file:///home/czx/PycharmProjects/smallpond/examples/shuffle_data.py):

```python
def shuffle_data(input_paths, num_data_partitions=10, num_hash_partitions=10):
    ctx = Context()
    dataset = ParquetDataSet(input_paths, union_by_name=True)
    data_files = DataSourceNode(ctx, dataset)
    
    # Step 1: 均匀分区
    data_partitions = DataSetPartitionNode(
        ctx,
        (data_files,),
        npartitions=num_data_partitions,
        partition_by_rows=True,
        random_shuffle=True,  # 随机打乱输入文件
    )
    
    # Step 2: 哈希分区
    urls_partitions = HashPartitionNode(
        ctx,
        (data_partitions,),
        npartitions=num_hash_partitions,
        hash_columns=None,          # 使用 random_shuffle
        random_shuffle=True,        # 随机分区
        engine_type="duckdb",
    )
    
    # Step 3: 排序（可选）
    shuffled_urls = SqlEngineNode(
        ctx,
        (urls_partitions,),
        r"select *, cast(random() * 2147483647 as integer) as sort_key from {0} order by sort_key",
        cpu_limit=16,
    )
    
    # Step 4: 重新分区（可选）
    repartitioned = DataSetPartitionNode(
        ctx,
        (shuffled_urls,),
        npartitions=num_out_data_partitions,
        partition_by_rows=True,
    )
    
    plan = LogicalPlan(ctx, repartitioned)
    return plan
```

## 6. 关键参数

### 6.1 HashPartitionNode 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `npartitions` | 分区数量 | 必填 |
| `hash_columns` | 哈希列列表 | None |
| `data_partition_column` | 分区键列名 | `__partition_key__` |
| `random_shuffle` | 是否随机分区 | False |
| `shuffle_only` | 是否仅 shuffle | False |
| `drop_partition_column` | 是否删除分区列 | False |
| `use_parquet_writer` | 是否使用 ParquetWriter | False |
| `hive_partitioning` | 是否使用 Hive 分区 | False |
| `engine_type` | 引擎类型 (`duckdb`/`arrow`) | `duckdb` |

### 6.2 分区模式说明

| 模式 | hash_columns | random_shuffle | shuffle_only | 说明 |
|------|--------------|-----------------|--------------|------|
| 指定列哈希 | 非空 | False | False | 按指定列哈希值分区 |
| 随机分区 | - | True | False | 使用 random() 随机分区 |
| 纯 Shuffle | - | False | True | 直接使用分区列值 |

## 7. 输出格式

### 7.1 扁平分区 (Flat Partitioning)

```
output/
 ├── partition_0-0.parquet
  ├── partition_0-1.parquet
  ├── partition_1-0.parquet
  ├── partition_1-1.parquet
  └── ...
```

### 7.2 Hive 分区 (Hive Partitioning)

```
output/
  ├── __partition_key__=0/
  │   ├── data-0.parquet
  │   └── data-1.parquet
  ├── __partition_key__=1/
  │   └── data-0.parquet
  └── ...
```

## 8. 总结

SmallPond 的数据 shuffle 机制通过以下核心组件实现：

1. **逻辑层 (Node)**: 定义分区策略和参数
2. **执行层 (Task)**: 实现具体的分区逻辑
3. **Planner**: 将逻辑计划转换为执行计划，生成多个 Producer/Consumer Tasks
4. **DuckDB/Arrow**: 实际执行数据分区和处理

主要分区策略：
- **均匀分区**: 按文件或行数均匀分配，适合初步分区
- **哈希分区**: 按列哈希值分配，保证相同值在同一分区
- **随机分区**: 完全随机分配，用于数据打散
- **Shuffle Only**: 直接按分区列值分配，适合已有分区键的场景