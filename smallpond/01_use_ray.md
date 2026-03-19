
用户代码 → DataFrame → LogicalPlan → Tasks → Ray Remote Functions

1.DataFrame API (smallpond/dataframe.py)：提供 read_parquet, repartition, filter, select 等操作
2.Logical Plan (smallpond/logical/node.py)：构建逻辑执行计划
3.Planner (smallpond/logical/planner.py)：将逻辑计划转换为物理任务
4.Task 执行：每个 Task 作为 Ray 远程函数执行

架构图
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
