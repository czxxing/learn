

## `task.exec()` 和 `SqlEngineTask.run()` 的关系

根据代码分析，它们是**框架层与业务层**的关系：

### 1. `task.exec()` - 框架层（任务生命周期管理）

定义在 [workqueue.py:154-177](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/workqueue.py#L154-L177)：

```python
def exec(self, cq: Optional["WorkQueue"] = None) -> WorkStatus:
    if self.status == WorkStatus.INCOMPLETE:
        try:
            self.start_time = time.time()
            self.exec_cq = cq
            self.initialize()          # 1. 初始化
            if self.run():             # 2. 执行任务逻辑
                self.status = WorkStatus.SUCCEED
                self.finalize()        # 3. 完成后的清理
            else:
                self.status = WorkStatus.FAILED
        except Exception as ex:
            self.status = WorkStatus.CRASHED
            self.exception = ex
        finally:
            self.cleanup()             # 4. 总是执行清理
            self.finish_time = time.time()
    return self.status
```

### 2. `SqlEngineTask.run()` - 业务层（具体执行逻辑）

定义在 [task.py:2284-2304](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2284-L2304)：

```python
def run(self) -> bool:
    if self.skip_when_any_input_empty:
        return True

    if self.batched_processing and isinstance(self.input_datasets[0], ParquetDataSet):
        input_batches = [[batch] for batch in self.input_datasets[0].partition_by_size(self.max_batch_size)]
    else:
        input_batches = [self.input_datasets]

    for batch_index, input_batch in enumerate(input_batches):
        with duckdb.connect(database=":memory:", config={"allow_unsigned_extensions": "true"}) as conn:
            self.prepare_connection(conn)
            self.process_batch(batch_index, input_batch, conn)  # 执行 SQL 查询

    return True
```

### 3. 调用关系图

```
task.exec()  (WorkItem 基类)
    │
    ├──► initialize()   (Task 初始化)
    │
    ├──► run()          (多态方法 - 子类实现)
    │       │
    │       └──► SqlEngineTask.run() 
    │               │
    │               └──► DuckDB 执行 SQL 查询
    │
    ├──► finalize()     (Task 完成清理)
    │
    └──► cleanup()      (Task 资源释放)
```

### 4. 多态设计

- `Task` 类是一个基类，`exec()` 是模板方法，定义了任务的完整生命周期
- `run()` 是一个抽象方法（由子类重写），实现具体的业务逻辑
- `SqlEngineTask`、`ArrowComputeTask` 等子类都重写了 `run()` 方法来实现自己的业务逻辑

这种设计遵循了**模板方法模式**，框架负责通用流程（初始化→执行→清理），子类负责具体实现。




## Task 生成流程

### 1. 核心组件

| 组件 | 文件 | 作用 |
|------|------|------|
| **Node** | [node.py](file:///home/czx/PycharmProjects/smallpond/smallpond/logical/node.py) | 逻辑计划节点，表示数据转换操作 |
| **Task** | [task.py](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py) | 执行计划任务，对应具体的执行单元 |
| **Planner** | [planner.py](file:///home/czx/PycharmProjects/smallpond/smallpond/logical/planner.py) | 将 Logical Plan 转换为 Execution Plan |

### 2. 生成机制

每个 Node 类通过 `@Node.task_factory` 装饰器定义 `create_task` 方法，例如 [SqlEngineNode](file:///home/czx/PycharmProjects/smallpond/smallpond/logical/node.py#L1082-L1115)：

```python
@Node.task_factory
def create_task(
    self,
    runtime_ctx: RuntimeContext,
    input_deps: List[Task],        # 依赖的 Task
    partition_infos: List[PartitionInfo],  # 分区信息
) -> SqlEngineTask:
    return self.spawn(
        runtime_ctx,
        input_deps,
        partition_infos,
        self.sql_queries,
        cpu_limit=self.cpu_limit,
        memory_limit=self.memory_limit,
        # ... 其他参数
    )
```

### 3. Planner 的转换逻辑

[Planner](file:///home/czx/PycharmProjects/smallpond/smallpond/logical/planner.py#L1-L100) 使用访问者模式遍历 Node DAG：

```python
class Planner(LogicalPlanVisitor[TaskGroup]):
    def visit_data_source_node(self, node, depth):
        return [node.create_task(self.runtime_ctx, [], [PartitionInfo()])]

    def visit_partition_node(self, node, depth):
        # 递归获取输入的 Tasks
        all_input_deps = [task for dep in node.input_deps for task in self.visit(dep)]
        # 根据分区数量生成多个 Task
        return [node.create_task(self.runtime_ctx, input_deps, partition_infos) ...]

    def visit_sql_engine_node(self, node, depth):
        # 类似逻辑...
        return [node.create_task(...) for ...]
```

### 4. 分区与 Task 数量的关系

Task 的数量与**分区（PartitionInfo）**紧密相关：

- **DataSourceNode**：1 个 Task
- **PartitionNode**：根据 `npartitions` 生成多个 Task
- **ConsolidateNode**：按分区维度合并 Task

例如，如果一个 `PartitionNode` 有 `npartitions=10`，它会为每个分区生成一个 Task。

### 5. 总结

```
LogicalPlan (Node DAG)
        │
        ▼
    Planner
        │
        ├──► 遍历每个 Node
        │
        ├──► 调用 node.create_task(runtime_ctx, input_deps, partition_infos)
        │
        ▼
ExecutionPlan (Task DAG)
```

Task 本质上是 **Logical Plan 在执行时刻的实例化**，它包含了：
- 输入依赖（其他 Task 的输出）
- 分区信息
- 资源配置（CPU、GPU、内存限制）
- 具体的执行逻辑（`run()` 方法）







          
是的，你的理解是正确的。让我用代码来说明这个过程：

        
          
## 分区后生成多个 Task 的机制

以 `npartitions=10` 为例：

```
┌─────────────────────────────────────────────────────────────────┐
│                    PartitionNode (npartitions=10)              │
├─────────────────────────────────────────────────────────────────┤
│  输入: 1 个 Task (来自 DataSourceNode)                           │
│  输出: 10 个 Tasks                                              │
│                                                                 │
│  input_deps=[Task-0]                                           │
│         │                                                       │
│         ▼                                                       │
│  ┌──────┴──────┐                                                │
│  │  partition  │  Task-0: PartitionInfo[@toplevel][0/10]       │
│  │  splitter   │  Task-1: PartitionInfo[@toplevel][1/10]       │
│  └──────┬──────┘  ...                                          │
│         │        Task-9: PartitionInfo[@toplevel][9/10]         │
│         ▼                                                       │
│  [Task-0, Task-1, ..., Task-9]                                  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼ 下游 SqlEngineNode 为每个分区创建 Task
┌─────────────────────────────────────────────────────────────────┐
│                      SqlEngineNode                              │
├─────────────────────────────────────────────────────────────────┤
│  输入: [Task-0, Task-1, ..., Task-9]                             │
│  输出: 10 个 SqlEngineTasks                                     │
│                                                                 │
│  SqlEngineTask-0: input_deps=[Task-0],  partition=[0/10]        │
│  SqlEngineTask-1: input_deps=[Task-1],  partition=[1/10]        │
│  ...                                                            │
│  SqlEngineTask-9: input_deps=[Task-9],  partition=[9/10]       │
│                                                                 │
│  每个 Task 只处理对应分区的数据                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 关键代码

在 [planner.py:195-205](file:///home/czx/PycharmProjects/smallpond/smallpond/logical/planner.py#L195-L205)：

```python
# 为每个分区创建一个 consumer task
return [
    node.create_consumer_task(
        self.runtime_ctx,
        producer_tasks,
        [
            PartitionInfo(),
            PartitionInfo(partition_idx, node.npartitions, node.dimension),
        ],
    )
    for partition_idx in range(node.npartitions)  # range(10) = [0,1,2,3,4,5,6,7,8,9]
]
```

### 运行时执行

每个 SqlEngineTask 独立运行在自己的进程中：

```python
# SqlEngineTask.run() 中
def run(self) -> bool:
    # self.partition_infos 包含当前 Task 的分区信息
    # 例如: [PartitionInfo[@toplevel][3/10]]
    
    # 只读取当前分区对应的输入数据
    input_batches = [self.input_datasets[0].partition_by_size(self.max_batch_size)]
    
    # 在 DuckDB 中执行 SQL
    with duckdb.connect(database=":memory:") as conn:
        self.process_batch(batch_index, input_batch, conn)
    
    return True
```

### 总结

| 阶段 | 说明 |
|------|------|
| **规划阶段** | Planner 为每个分区创建一个 Task |
| **分区数量** | `npartitions=N` → 生成 `N` 个 Task |
| **数据分布** | 每个 Task 只处理对应分区的数据 |
| **并行执行** | 多个 Task 可在不同 Executor 上并行运行 |




## 处理流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    原始 Parquet 数据集                               │
│                     (1000 万行数据)                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ 按分区拆分
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│分区 0   │分区 1   │分区 2   │分区 3   │ ...     │分区 8   │分区 9   │
│100万行  │100万行  │100万行  │100万行  │         │100万行  │100万行  │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
      │           │           │                   │           │
      ▼           ▼           ▼                   ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐         ┌─────────┐ ┌─────────┐
│Task-0   │ │Task-1   │ │Task-2   │  ...    │Task-8   │ │Task-9   │
│DuckDB   │ │DuckDB   │ │DuckDB   │         │DuckDB   │ │DuckDB   │
│处理分区0│ │处理分区1│ │处理分区2│         │处理分区8│ │处理分区9│
│100万行  │ │100万行  │ │100万行  │         │100万行  │ │100万行  │
└─────────┘ └─────────┘ └─────────┘         └─────────┘ └─────────┘
      │           │           │                   │           │
      ▼           ▼           ▼                   ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐         ┌─────────┐ ┌─────────┐
│输出文件0│ │输出文件1│ │输出文件2│  ...    │输出文件8│ │输出文件9│
└─────────┘ └─────────┘ └─────────┘         └─────────┘ └─────────┘
```

## 关键代码说明

### 1. 输入数据的分区过滤

在 [task.py:2289-2293](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2289-L2293)：

```python
def run(self) -> bool:
    if self.batched_processing and isinstance(self.input_datasets[0], ParquetDataSet):
        # 按分区过滤，只读取当前 Task 对应分区的数据
        input_batches = [[batch] for batch in self.input_datasets[0].partition_by_size(self.max_batch_size)]
    else:
        input_batches = [self.input_datasets]
```

每个 Task 的 `input_datasets` 已经包含了**只属于该分区**的数据。

### 2. DuckDB 视图创建

在 [task.py:2310](file:///home/czx/PycharmProjects/smallpond/smallpond/execution/task.py#L2310)：

```python
def process_batch(self, batch_index, input_datasets, conn):
    # 只把当前分区数据注册为 DuckDB 视图
    input_views = self.create_input_views(conn, input_datasets)
```

### 3. SQL 查询执行

每个 Task 独立执行 SQL：

```python
for query_index, sql_query in enumerate(self.sql_queries):
    # 只查询当前分区数据
    sql_query = sql_query.format(*input_views, ...)
    
    # 在当前分区的数据上执行
    conn.execute(sql_query)
```

### 4. 输出写入

```python
# 输出到对应分区的文件
output_filename = f"{self.output_filename}-{batch_index}.{query_index}"
# 例如: SqlEngineTask-3-0.parquet → 分区 3 的输出
```

## 数据流转总结

| 步骤 | 说明 |
|------|------|
| **1. 分区** | 原始数据按分区键分成 N 份 |
| **2. Task 生成** | 每个分区生成一个 SqlEngineTask |
| **3. 数据过滤** | 每个 Task 只读取对应分区的输入数据 |
| **4. DuckDB 处理** | 在内存中处理该分区的数据 |
| **5. 输出** | 写入对应的输出文件 |

这样设计的好处是**并行处理**：10 个分区可以同时在 10 个进程中运行 DuckDB，显著提升处理速度。
        
        
        
        