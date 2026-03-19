核心的 Ray 任务执行机制：

```python
def run_on_ray(self) -> ray.ObjectRef:
    @ray.remote
    def exec_task(task: Task, *inputs: DataSet) -> DataSet:
        # 执行任务
        status = task.exec()
        # 原子化输出结果
        dump(task.output, task.ray_dataset_path, atomic_write=True)
       return task.output

    remote_function = exec_task.options(
        name=f"{task.node_id}.{self.__class__.__name__}",
        num_cpus=self.cpu_limit,
        num_gpus=self.gpu_limit,
        memory=int(self.memory_limit),
    )
    self._dataset_ref = remote_function.remote(task, *[dep.run_on_ray() for dep in self.input_deps.values()])
    return self._dataset_ref
```
### 4. 数据处理流程

```
用户代码 → DataFrame → LogicalPlan → Tasks → Ray Remote Functions
```
1. **DataFrame API** ([`smallpond/dataframe.py`](smallpond/dataframe.py))：提供 `read_parquet`, `repartition`, `filter`, `select` 等操作
2. **Logical Plan** ([`smallpond/logical/node.py`](smallpond/logical/node.py))：构建逻辑执行计划
3. **Planner** ([`smallpond/logical/planner.py`](smallpond/logical/planner.py))：将逻辑计划转换为物理任务
4. **Task 执行**：每个 Task 作为 Ray 远程函数执行

###  架构图

```
┌─────────────────────────────────────────────────────────┐
│                    Smallpond Session                    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  DataFrame  │ -> │ LogicalPlan │ -> │   Planner   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
├─────────────────────────────────────────────────────────┤
│                      Ray Cluster                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │ Worker1 │  │ Worker2 │  │ Worker3 │  │ WorkerN │     │
│  │  Task   │  │  Task   │  │  Task   │  │  Task   │     │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │
└─────────────────────────────────────────────────────────┘
```




