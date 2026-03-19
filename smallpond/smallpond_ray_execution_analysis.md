# SmallPond Shuffle 任务在 Ray 上的执行流程详细分析

## 1. 概述

本文档详细分析 SmallPond 中 Shuffle 任务是如何在 Ray 分布式计算框架上执行的，从任务提交到最终交给 SqlEngineTask 执行的完整流程。

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Ray 集群                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │   Ray Head  │    │  Worker 1   │    │  Worker 2   │                     │
│  │  (调度器)   │    │  (执行器)   │    │  (执行器)   │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│         │                  │                  │                             │
│         │         ┌────────┴────────┐         │                             │
│         │         │  Ray Worker     │         │                             │
│         │         │  Process Pool  │         │                             │
│         │         │  ┌───────────┐ │         │                             │
│         │         │  │ Task-0   │ │         │                             │
│         │         │  │ exec_task │ │         │                             │
│         │         │  │ (DuckDB) │ │         │                             │
│         │         │  └───────────┘ │         │                             │
│         │         │  ┌───────────┐ │         │                             │
│         │         │  │ Task-1   │ │         │                             │
│         │         │  │ exec_task │ │         │                             │
│         │         │  └───────────┘ │         │                             │
│         │         └────────────────┘         │                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 执行模式选择

SmallPond 支持三种执行模式，通过 [driver.py](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/driver.py#L32) 配置：

```python
parser.add_argument("mode", choices=["executor", "scheduler", "ray"], default="executor")
```

| 模式 | 说明 |
|------|------|
| `executor` | 使用内置 Executor 本地执行 |
| `scheduler` | 分布式调度模式 |
| `ray` | 使用 Ray 进行分布式执行 |

### 3.1 Ray 模式入口

在 [driver.py:L322-L326](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/driver.py#L322-L326)：

```python
if args.mode == "ray":
    assert plan is not None
    sp = smallpond.init(_remove_output_root=args.remove_output_root)
    DataFrame(sp, plan.root_node).compute()  # 触发 Ray 执行
    retval = True
```

## 4. Ray 执行流程详解

### 4.1 入口：DataFrame.compute()

当调用 `DataFrame.compute()` 时，触发整个执行流程：

```python
# dataframe.py:L263-L275
def _compute(self) -> List[DataSet]:
    for retry_count in range(3):
        try:
            # 核心：使用 ray.get 获取所有任务的执行结果
            return ray.get([task.run_on_ray() for task in self._get_or_create_tasks()])
        except ray.exceptions.RuntimeEnvSetupError as e:
            logger.error(f"found ray RuntimeEnvSetupError, retrying...\n{e}")
            time.sleep(10 << retry_count)
    raise RuntimeError("Failed to compute data after 3 retries")
```

### 4.2 任务创建：Planner 生成 Tasks

在执行之前，需要先将 Logical Plan 转换为 Execution Plan：

```python
# dataframe.py:L210-L230
def _get_or_create_tasks(self) -> List[Task]:
    # 1. 优化逻辑计划
    if self.optimized_plan is None:
        self.optimized_plan = Optimizer(...).visit(self.plan)
    
    # 2. 创建 Planner
    planner = Planner(self.session._runtime_ctx)
    
    # 3. 访问计划并生成 Tasks
    return planner.visit(self.optimized_plan)
```

### 4.3 核心：Task.run_on_ray()

这是将任务提交到 Ray 执行的核心方法，定义在 [task.py:L977-L1055](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L977-L1055)：

```python
def run_on_ray(self) -> ray.ObjectRef:
    """
    Run the task on Ray.
    Return an `ObjectRef`, which can be used with `ray.get` to wait for the output dataset.
    """
    # 1. 如果任务已经启动，直接返回引用
    if self._dataset_ref is not None:
        return self._dataset_ref

    # 2. 检查任务是否已完成（从磁盘读取）
    if os.path.exists(self.ray_dataset_path):
        logger.info(f"task {self.key} already finished, skipping")
        output = load(self.ray_dataset_path)
        self._dataset_ref = ray.put(output)
        return self._dataset_ref

    # 3. 复制任务对象（避免修改原始任务）
    task = copy.copy(self)
    task.input_deps = {dep_key: None for dep_key in task.input_deps}

    # 4. 定义 Ray 远程执行函数
    @ray.remote
    def exec_task(task: Task, *inputs: DataSet) -> DataSet:
        # 设置进程名称
        mp.current_process().name = task.key

        # 检查重试次数
        task.retry_count = 0
        while os.path.exists(task.ray_marker_path):
            task.retry_count += 1
            if task.retry_count > DEFAULT_MAX_RETRY_COUNT:
                raise RuntimeError(f"task {task.key} failed after {task.retry_count} retries")

        # 创建标记文件
        Path(task.ray_marker_path).touch()

        # 注入输入数据
        task.input_datasets = list(inputs)
        
        # ★ 核心：执行任务
        status = task.exec()  # 调用 WorkItem.exec()
        
        if status != WorkStatus.SUCCEED:
            raise task.exception or RuntimeError(f"task {task.key} failed with status {status}")

        # 保存输出到磁盘
        os.makedirs(os.path.dirname(task.ray_dataset_path), exist_ok=True)
        dump(task.output, task.ray_dataset_path, atomic_write=True)
        return task.output

    # 5. 配置 Ray 任务选项
    remote_function = exec_task.options(
        name=f"{task.node_id}.{self.__class__.__name__}",  # 用于 Ray Dashboard 分组
        num_cpus=self.cpu_limit,
        num_gpus=self.gpu_limit,
        memory=int(self.memory_limit),
    )

    # 6. 提交任务到 Ray
    # 递归处理所有输入依赖
    self._dataset_ref = remote_function.remote(
        task, 
        *[dep.run_on_ray() for dep in self.input_deps.values()]
    )
    return self._dataset_ref
```

### 4.4 任务执行：WorkItem.exec()

当 Ray Worker 执行 `exec_task` 时，内部调用 `task.exec()`，这是定义在 [workqueue.py:L154-L177](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/workqueue.py#L154-L177) 的模板方法：

```python
def exec(self, cq: Optional["WorkQueue"] = None) -> WorkStatus:
    if self.status == WorkStatus.INCOMPLETE:
        try:
            self.start_time = time.time()
            self.exec_cq = cq
            
            # ★ 步骤1：初始化
            self.initialize()
            
            # ★ 步骤2：执行任务逻辑（多态方法）
            if self.run():
                self.status = WorkStatus.SUCCEED
                # ★ 步骤3：完成清理
                self.finalize()
            else:
                self.status = WorkStatus.FAILED
                
        except Exception as ex:
            logger.opt(exception=ex).error(f"{repr(self)} crashed with error.")
            self.status = WorkStatus.CRASHED
            self.exception = ex
        finally:
            # ★ 步骤4：资源清理
            self.cleanup()
            self.finish_time = time.time()
    return self.status
```

## 5. Shuffle 任务的完整执行链

### 5.1 Shuffle 任务类型

对于 Shuffle 操作，会涉及以下 Task 类型：

| Task 类型 | 作用 | run() 方法 |
|-----------|------|------------|
| `EvenlyDistributedPartitionProducerTask` | 均匀分区生产者 | 拆分输入数据 |
| `HashPartitionTask` | 哈希分区基类 | - |
| `HashPartitionDuckDbTask` | DuckDB 哈希分区 | 执行哈希分区逻辑 |
| `SqlEngineTask` | SQL 引擎任务 | 执行用户 SQL |

### 5.2 执行流程图

```
DataFrame.compute()
    │
    ▼
ray.get([task.run_on_ray() for task in tasks])
    │
    ├──► Task-0.run_on_ray()   (HashPartitionDuckDbTask)
    │       │
    │       ├──► 递归处理输入依赖
    │       │       input_deps[0].run_on_ray()
    │       │
    │       ├──► ray.remote(exec_task) 提交到 Ray
    │       │
    │       └──► Ray Worker 执行:
    │               exec_task(task, *inputs)
    │                   │
    │                   ├──► task.exec()
    │                   │       │
    │                   │       ├──► task.initialize()
    │                   │       │
    │                   │       ├──► task.run()  ★ 执行哈希分区
    │                   │       │       │
    │                   │       │       ├──► partition_query = ...
    │                   │       │       ├──► DuckDB 执行哈希计算
    │                   │       │       └──► 写入分区文件
    │                   │       │
    │                   │       └──► task.finalize()
    │                   │
    │                   └──► dump(output, ray_dataset_path)
    │
    ├──► Task-1.run_on_ray()   (SqlEngineTask)
    │       │
    │       └──► Ray Worker 执行:
    │               exec_task(task, *inputs)
    │                   │
    │                   ├──► task.exec()
    │                   │       │
    │                   │       ├──► task.initialize()
    │                   │       │
    │                   │       ├──► task.run()  ★ 执行 SQL
    │                   │       │       │
    │                   │       │       ├──► DuckDB 连接
    │                   │       │       ├──► 执行 SQL 查询
    │                   │       │       └──► 写入输出
    │                   │       │
    │                   │       └──► task.finalize()
    │                   │
    │                   └──► dump(output, ray_dataset_path)
    │
    └──► ... 更多任务
```

### 5.3 SqlEngineTask.run() 详解

以 SqlEngineTask 为例，看具体如何执行：

定义在 [task.py:L2284-L2304](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2284-L2304)：

```python
def run(self) -> bool:
    # 1. 检查是否跳过空输入
    if self.skip_when_any_input_empty:
        return True

    # 2. 处理批次（支持大数据集分批处理）
    if self.batched_processing and isinstance(self.input_datasets[0], ParquetDataSet):
        input_batches = [[batch] for batch in self.input_datasets[0].partition_by_size(self.max_batch_size)]
    else:
        input_batches = [self.input_datasets]

    # 3. 对每个批次执行 SQL
    for batch_index, input_batch in enumerate(input_batches):
        # 创建 DuckDB 内存数据库连接
        with duckdb.connect(database=":memory:", config={"allow_unsigned_extensions": "true"}) as conn:
            # 准备连接（注册函数等）
            self.prepare_connection(conn)
            
            # 处理批次
            self.process_batch(batch_index, input_batch, conn)

    return True
```

### 5.4 HashPartitionDuckDbTask.run() 详解

Shuffle 操作中的哈希分区任务：

定义在 [task.py:L2574-L2600](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2574-L2600)：

```python
def run(self) -> bool:
    self.add_elapsed_time()
    if self.skip_when_any_input_empty:
        return True

    # 获取输入数据集
    input_dataset = self.input_datasets[0]
    assert isinstance(input_dataset, ParquetDataSet)
    
    # 按大小分批
    input_batches = input_dataset.partition_by_size(self.max_batch_size)

    # 对每个批次执行分区
    for batch_index, input_batch in enumerate(input_batches):
        logger.info(f"partitioning batch #{batch_index+1}/{len(input_batches)}")
        
        # 执行分区逻辑
        self.partition(batch_index, input_batch)

    return True
```

分区逻辑在 [task.py:L2626-L2645](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2626-L2645)：

```python
def partition(self, batch_index: int, input_dataset: ParquetDataSet):
    with duckdb.connect(database=":memory:") as conn:
        self.prepare_connection(conn)
        
        if self.hive_partitioning:
            # Hive 风格分区
            self.load_input_batch(conn, batch_index, input_dataset, sort_by_partition_key=True)
            self.write_hive_partitions(conn, batch_index, input_dataset)
        else:
            # 扁平分区
            self.load_input_batch(conn, batch_index, input_dataset, sort_by_partition_key=True)
            self.write_flat_partitions(conn, batch_index, input_dataset)
```

## 6. 数据依赖与调度

### 6.1 任务依赖图

Ray 自动处理任务间的数据依赖：

```python
# task.py:L1043-L1047
self._dataset_ref = remote_function.remote(
    task, 
    *[dep.run_on_ray() for dep in self.input_deps.values()]  # 递归获取依赖的 ObjectRef
)
```

这创建了一个 DAG，Ray 会自动：
1. 先执行没有依赖的任务（如 DataSourceTask）
2. 当依赖任务完成后再执行下游任务
3. 自动处理失败重试

### 6.2 任务状态持久化

Ray 执行器使用文件系统持久化任务状态：

```python
# task.py:L745-L760
@property
def ray_marker_path(self) -> str:
    """任务已启动的标记文件路径"""
    return os.path.join(self.ctx.started_task_dir, f"{self.node_id}.{self.key}.{self.retry_count}")

@property
def ray_dataset_path(self) -> str:
    """任务输出数据集的路径"""
    return os.path.join(self.ctx.completed_task_dir, str(self.node_id), f"{self.key}.pickle")
```

## 7. 关键代码位置汇总

| 功能 | 文件位置 |
|------|----------|
| Ray 模式入口 | [driver.py:L322-L326](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/driver.py#L322-L326) |
| DataFrame.compute() | [dataframe.py:L263-L275](file:///home/czx/PycharmProjects/smallpond/smallpond/dataframe.py#L263-L275) |
| Task.run_on_ray() | [task.py:L977-L1055](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L977-L1055) |
| exec_task 远程函数 | [task.py:L990-L1029](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L990-L1029) |
| WorkItem.exec() | [workqueue.py:L154-L177](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/workqueue.py#L154-L177) |
| SqlEngineTask.run() | [task.py:L2284-L2304](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2284-L2304) |
| HashPartitionDuckDbTask.run() | [task.py:L2574-L2600](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2574-L2600) |

## 8. 执行示例

### 8.1 Shuffle 数据流

```
原始数据 (Parquet)
      │
      ▼ DataSourceNode.run_on_ray()
 ┌────────────┐
 │ DataSource │
 │   Task     │
 └────────────┘
      │ ObjectRef
      ▼
 ┌────────────────────────────────────────┐
 │ HashPartitionNode (10 个分区)         │
 │                                        │
 │ Task-0.run_on_ray() ──► 分区 0        │
 │ Task-1.run_on_ray() ──► 分区 1        │
 │ ...                                    │
 │ Task-9.run_on_ray() ──► 分区 9        │
 └────────────────────────────────────────┘
      │ 各分区数据
      ▼
 ┌────────────────────────────────────────┐
 │ SqlEngineNode (10 个分区并行)          │
 │                                        │
 │ Task-0.run_on_ray() ──► SQL 处理      │
 │ Task-1.run_on_ray() ──► SQL 处理      │
 │ ...                                    │
 │ Task-9.run_on_ray() ──► SQL 处理      │
 └────────────────────────────────────────┘
      │ 结果数据
      ▼
 最终输出 (Parquet)
```

### 8.2 代码示例

```python
from smallpond import Session
from smallpond.logical.dataset import ParquetDataSet
from smallpond.logical.node import (
    Context, DataSourceNode, DataSetPartitionNode, 
    HashPartitionNode, SqlEngineNode, LogicalPlan
)

# 1. 创建执行会话（Ray 模式）
ctx = Context()
session = Session(ctx)

# 2. 构建逻辑计划
dataset = ParquetDataSet(["input/*.parquet"])
data_source = DataSourceNode(ctx, dataset)

# 3. 均匀分区
partitions = DataSetPartitionNode(ctx, (data_source,), npartitions=10)

# 4. 哈希分区 (Shuffle)
shuffled = HashPartitionNode(ctx, (partitions,), npartitions=20, random_shuffle=True)

# 5. SQL 处理
result = SqlEngineNode(ctx, (shuffled,), "SELECT * FROM {0}")

# 6. 创建 DataFrame 并执行
df = DataFrame(session, result)
df.compute()  # ★ 触发 Ray 执行

# 7. 获取结果
output = df.get_output()
```

## 9. 总结

SmallPond 在 Ray 上的执行流程：

1. **入口**：调用 `DataFrame.compute()` 触发执行
2. **规划**：Planner 将 Logical Plan 转换为多个 Tasks
3. **提交**：每个 Task 调用 `run_on_ray()` 提交到 Ray
4. **执行**：
   - Ray 自动调度任务依赖
   - Worker 进程执行 `exec_task()` 函数
   - 内部调用 `task.exec()` → `task.run()`
5. **SQL 执行**：SqlEngineTask 使用 DuckDB 执行用户 SQL
6. **持久化**：结果保存到磁盘，供下游任务使用

关键设计：
- **模板方法模式**：`WorkItem.exec()` 定义执行骨架，`Task.run()` 实现具体逻辑
- **Ray 远程函数**：动态创建 `@ray.remote` 函数执行任务
- **DAG 依赖**：Ray 自动处理任务间依赖和调度
- **状态持久化**：使用文件系统实现任务状态共享