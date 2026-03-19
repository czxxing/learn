# SmallPond 全局聚合机制详细分析

## 1. 问题分析

用户问询：SmallPond 是否只能做到局部的数据聚合，做不到全局数据的聚合？

通过代码分析，答案是：**SmallPond 支持全局聚合，但需要用户手动实现两阶段聚合**。

## 2. 数据分区与并行处理

### 2.1 分区执行模型

SmallPond 采用**分区并行**模型：

```
原始数据 (1000万行)
        │
        ▼ DataSetPartitionNode (npartitions=10)
        │
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│分区0  │分区1  │分区2  │分区3  │分区4  │分区5  │分区6  │分区7  │分区8  │分区9  │
│100万行│100万行│100万行│100万行│100万行│100万行│100万行│100万行│100万行│100万行│
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
        │           │           │                   │           │
        ▼           ▼           ▼                   ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐         ┌─────────┐ ┌─────────┐
    │SqlEngine│ │SqlEngine│ │SqlEngine│   ...   │SqlEngine│ │SqlEngine│
    │  Task   │ │  Task   │ │  Task   │         │  Task   │ │  Task   │
    └─────────┘ └─────────┘ └─────────┘         └─────────┘ └─────────┘
        │           │           │                   │           │
        ▼           ▼           ▼                   ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐         ┌─────────┐ ┌─────────┐
    │聚合结果1 │ │聚合结果2 │ │聚合结果3 │   ...   │聚合结果9 │ │聚合结果10│
    │(局部聚合)│ │(局部聚合)│ │(局部聚合)│         │(局部聚合)│ │(局部聚合)│
    └─────────┘ └─────────┘ └─────────┘         └─────────┘ └─────────┘
```

### 2.2 局部聚合的问题

如果在每个分区上直接执行 `GROUP BY`：

```sql
SELECT host, count(*) as cnt FROM {0} GROUP BY host
```

每个分区只返回该分区内的聚合结果：

```
分区0: [host=A, cnt=100], [host=B, cnt=50]
分区1: [host=A, cnt=80], [host=C, cnt=120]
分区2: [host=B, cnt=60], [host=D, cnt=90]
...
```

这**不是全局聚合**，因为：
- `host=A` 的总数 = 100 + 80 + ... (分布在多个分区)
- 每个分区只计算了自己分区内的 `count`

## 3. 全局聚合实现机制

### 3.1 两阶段聚合 (Two-Phase Aggregation)

SmallPond 支持通过手动设计实现**两阶段聚合**：

```
阶段1: 局部聚合
┌─────────────────────────────────────────────────────────────┐
│  每个分区执行: SELECT host, count(*) as cnt FROM {0}       │
│  GROUP BY host                                              │
│                                                             │
│  分区0: [host=A, cnt=100], [host=B, cnt=50]              │
│  分区1: [host=A, cnt=80], [host=C, cnt=120]              │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
阶段2: 合并聚合
┌─────────────────────────────────────────────────────────────┐
│  将所有分区的局部结果合并，重新执行全局聚合:                  │
│  SELECT host, sum(cnt) as cnt FROM {0} GROUP BY host        │
│                                                             │
│  全局结果: [host=A, cnt=180+...], [host=B, cnt=50+60...]  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 实现方式

SmallPond 提供两种方式实现全局聚合：

#### 方式1: DataSetPartitionNode(npartitions=1)

将所有分区合并为一个分区，然后执行全局聚合：

```python
# 阶段1: 局部聚合
partial_agg = SqlEngineNode(
    ctx,
    (partitioned_data,),
    r"SELECT host, count(*) as cnt FROM {0} GROUP BY host"
)

# 阶段2: 合并为一个分区
merged = DataSetPartitionNode(ctx, (partial_agg,), npartitions=1)

# 阶段3: 全局聚合
global_agg = SqlEngineNode(
    ctx,
    (merged,),
    r"SELECT host, sum(cnt) as cnt FROM {0} GROUP BY host"
)
```

#### 方式2: ConsolidateNode

使用 `ConsolidateNode` 按指定维度合并分区：

```python
# 阶段1: 局部聚合
partial_agg = SqlEngineNode(
    ctx,
    (partitioned_data,),
    r"SELECT host, count(*) as cnt FROM {0} GROUP BY host"
)

# 阶段2: 按 host_partition 维度合并
consolidated = ConsolidateNode(ctx, partial_agg, dimensions=["host_partition"])

# 阶段3: 全局聚合
global_agg = SqlEngineNode(
    ctx,
    (consolidated,),
    r"SELECT host, sum(cnt) as cnt FROM {0} GROUP BY host"
)
```

### 3.3 核心组件

#### MergeDataSetsTask

在 [task.py:1219-1237](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L1219-L1237)：

```python
class MergeDataSetsTask(Task):
    @property
    def exec_on_scheduler(self) -> bool:
        return True  # 在调度器上执行

    @property
    def self_contained_output(self):
        return False

    def run(self) -> bool:
        datasets = self.input_datasets
        # 合并多个数据集
        self.dataset = datasets[0].merge(datasets)
        logger.info(f"created merged dataset: {self.dataset}")
        return True
```

关键特点：
- `exec_on_scheduler=True`：在调度器节点执行
- 不跨网络 shuffle，只是合并文件路径
- 调用 `datasets[0].merge(datasets)` 合并 DataSet

#### ConsolidateNode

在 [node.py:1150-1179](file:///home/czx/PycharmProjects/smallpond/smallpond/logical/node.py#L1150-L1179)：

```python
class ConsolidateNode(Node):
    """
    Consolidate partitions into larger ones.
    """
    
    def __init__(self, ctx: Context, input_dep: Node, dimensions: List[str]):
        """
        Effectively reduces the number of partitions without shuffling the data across the network.
        """
        self.dimensions = set(list(dimensions) + [PartitionInfo.toplevel_dimension])
```

关键特点：
- 按指定维度合并分区
- 不跨网络 shuffle（数据不需要重新分配）
- 只是将相同维度的分区文件组合在一起

### 3.4 Planner 中的处理

在 [planner.py:124-142](file:///home/czx/PycharmProjects/smallpond/smallpond/logical/planner.py#L124-L142)：

```python
def visit_consolidate_node(self, node: ConsolidateNode, depth: int) -> TaskGroup:
    # 获取输入的所有 tasks
    input_deps_taskgroups = [self.visit(dep, depth + 1) for dep in node.input_deps]
    
    # 按分区维度分组
    input_deps_groupby_partitions: Dict[Tuple, List[Task]] = defaultdict(list)
    for task in input_deps_taskgroups[0]:
        partition_infos = tuple(info for info in task.partition_infos if info.dimension in node.dimensions)
        input_deps_groupby_partitions[partition_infos].append(task)
    
    # 为每组创建一个 MergeDataSetsTask
    return [
        node.create_task(self.runtime_ctx, input_deps, partition_infos) 
        for partition_infos, input_deps in input_deps_groupby_partitions.items()
    ]
```

## 4. 全局 Limit 实现

SmallPond 在 `LimitNode` 中也实现了两阶段聚合：

在 [planner.py:265-271](file:///home/czx/PycharmProjects/smallpond/smallpond/logical/planner.py#L265-L271)：

```python
def visit_limit_node(self, node: LimitNode, depth: int) -> TaskGroup:
    # 阶段1: 每个分区执行局部 limit
    all_input_deps = self.visit(node.input_deps[0], depth + 1)
    partial_limit_tasks = [
        node.create_task(self.runtime_ctx, [task], task.partition_infos) 
        for task in all_input_deps
    ]
    
    # 阶段2: 合并所有分区的局部结果
    merge_task = node.create_merge_task(self.runtime_ctx, partial_limit_tasks, [PartitionInfo()])
    
    # 阶段3: 在合并后的数据上执行全局 limit
    global_limit_task = node.create_task(self.runtime_ctx, [merge_task], merge_task.partition_infos)
    return [global_limit_task]
```

## 5. 示例：两阶段聚合

来自 [tests/test_partition.py:355-410](file:///home/czx/PycharmProjects/smallpond/tests/test_partition.py#L355-L410)：

```python
ctx = Context()
parquet_files = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
data_source = DataSourceNode(ctx, parquet_files)

# 阶段1: 按 host 分区
partition_by_hosts = HashPartitionNode(
    ctx,
    (data_source,),
    npartitions=3,
    hash_columns=["host"],
    data_partition_column="host_partition",
)

# 阶段2: 按 url 进一步分区 (嵌套分区)
partition_by_hosts_x_urls = HashPartitionNode(
    ctx,
    (partition_by_hosts,),
    npartitions=5,
    hash_columns=["url"],
    data_partition_column="url_partition",
    nested=True,
)

# 阶段3: 局部聚合 (每个分区内的 count)
initial_reduce = r"select host, count(*) as cnt from {0} group by host"
url_count_by_hosts_x_urls1 = SqlEngineNode(
    ctx,
    (partition_by_hosts_x_urls,),
    initial_reduce,
)

# 阶段4: 按 host_partition 维度合并分区
url_count_by_hosts1 = SqlEngineNode(
    ctx,
    (ConsolidateNode(ctx, url_count_by_hosts_x_urls1, ["host_partition"]),),
    # 阶段5: 全局聚合 (sum 所有分区的 count)
    r"select host, cast(sum(cnt) as bigint) as cnt from {0} group by host",
)
```

## 6. 总结：全局聚合能力

### 6.1 SmallPond 支持全局聚合

| 机制 | 说明 |
|------|------|
| **两阶段聚合** | 用户手动设计：局部聚合 → 合并 → 全局聚合 |
| **ConsolidateNode** | 按维度合并分区，不跨网络 shuffle |
| **DataSetPartitionNode(npartitions=1)** | 将所有分区合并为一个分区 |
| **MergeDataSetsTask** | 在调度器上执行，合并 DataSet |

### 6.2 局限性

1. **需要手动实现**：不像 Spark/Dask，SmallPond 不自动进行两阶段聚合优化
2. **不跨网络 shuffle**：ConsolidateNode 只在同一节点内合并文件
3. **调度器瓶颈**：MergeDataSetsTask 在调度器上执行，大数据量时可能有瓶颈

### 6.3 使用建议

```python
# 全局聚合的正确方式
# 1. 局部聚合
partial = SqlEngineNode(ctx, (data,), "SELECT key, sum(val) as val FROM {0} GROUP BY key")

# 2. 合并分区
merged = DataSetPartitionNode(ctx, (partial,), npartitions=1)

# 3. 全局聚合
global_result = SqlEngineNode(ctx, (merged,), "SELECT key, sum(val) as val FROM {0} GROUP BY key")
```

### 6.4 与其他框架对比

| 框架 | 全局聚合 | 自动两阶段优化 |
|------|---------|--------------|
| Spark | ✅ 自动 | ✅ 自动 |
| Dask | ✅ 自动 | ✅ 自动 |
| **SmallPond** | ✅ 支持 | ❌ 手动 |
| DuckDB (单机) | ✅ 自动 | N/A |

## 7. 结论

**SmallPond 能够实现全局聚合**，但需要用户手动设计两阶段聚合：
1. 第一阶段：每个分区执行局部聚合
2. 第二阶段：合并分区后执行全局聚合

这是因为 SmallPond 采用**分区并行**模型，优先考虑**本地计算**和**避免跨网络 shuffle**，通过 `ConsolidateNode` 或 `DataSetPartitionNode(npartitions=1)` 实现分区合并。