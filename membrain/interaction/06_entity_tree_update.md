# Stage 6: 实体树更新

## 概述

这一阶段负责在数据持久化后更新实体树结构。实体树是 MemBrain 的核心数据结构，用于组织和管理实体之间的层级关系。

## 代码位置

- **更新器入口**: [entity_tree_updater.py](file:///home/project/MemBrain/membrain/memory/application/entity_tree_updater.py)
- **树计算管道**: [pipeline.py](file:///home/project/MemBrain/membrain/memory/core/entity_tree/pipeline.py)
- **树存储**: [entity_tree_store.py](file:///home/project/MemBrain/membrain/infra/persistence/entity_tree_store.py)

## 详细代码分析

### 6.1 更新器入口

```python
# entity_tree_updater.py

class EntityTreeUpdater:
    def __init__(self, store: EntityTreeStore) -> None:
        self._store = store

    async def update(
        self,
        task_id: int,
        batch_id: str,
        embed_client,
        registry,
        factory,
    ) -> list[str]:
        # 1. 查找受影响的实体
        touched_entity_ids = self._store.find_touched_entities(task_id, batch_id)
        if not touched_entity_ids:
            return []

        # 2. 锁定实体
        with self._store.lock_entities(task_id, touched_entity_ids) as db:
            # 3. 加载状态
            state = self._store.load_update_state(
                task_id, batch_id, touched_entity_ids=touched_entity_ids, db=db
            )
            
            if not state.targets:
                return []

            # 4. 计算更新
            result = await compute_entity_tree_updates(
                task_id=state.task_id,
                targets=state.targets,
                embed_client=embed_client,
                registry=registry,
                factory=factory,
            )
            
            # 5. 应用更新
            self._store.apply_updates(task_id, state, result, db=db)
            db.commit()

        return result.profiled_entities
```

### 6.2 查找受影响实体

```python
# entity_tree_store.py

def find_touched_entities(self, task_id: int, batch_id: str) -> list[str]:
    """查找本批次新增事实涉及的实体。"""
    
    query = (
        select(FactRefModel.entity_id)
        .join(FactModel, FactModel.id == FactRefModel.fact_id)
        .where(
            FactModel.task_id == task_id,
            FactModel.batch_id == batch_id,
            FactModel.status == "active",
        )
        .distinct()
    )
    
    results = db.execute(query).fetchall()
    return [row[0] for row in results]
```

### 6.3 树更新计算

```python
# pipeline.py

async def compute_entity_tree_updates(
    task_id: int,
    targets: list[dict],
    embed_client,
    registry,
    factory,
) -> TreeUpdateResult:
    """计算实体树的更新。"""
    
    profiled_entities = []
    all_updates = []
    
    for target in targets:
        entity_id = target["entity_id"]
        new_facts = target["new_facts"]
        existing_tree = target["existing_tree"]
        
        # 合并新事实到现有集合
        if existing_tree:
            existing_fact_ids = set(existing_tree.fact_ids)
            all_fact_ids = existing_fact_ids | {f.id for f in new_facts}
        else:
            all_fact_ids = {f.id for f in new_facts}
        
        # 提取 Aspect
        aspects = extract_aspects_from_facts(new_facts)
        
        # 构建/更新树
        if existing_tree:
            tree = merge_aspects(existing_tree, aspects)
        else:
            tree = create_entity_tree(entity_id, aspects)
        
        # 审计重组
        tree = audit_and_rebalance(tree)
        
        all_updates.append(tree)
        
        if len(aspects) > 1:
            profiled_entities.append(entity_id)
    
    return TreeUpdateResult(
        profiled_entities=profiled_entities,
        updates=all_updates,
    )
```

## 实体树结构示例

```
实体: Caroline (uuid-1)
树结构:
└── root
    ├── Career (工作)
    │   └── fact_ids: [1, 5, 10]
    ├── Family (家庭)
    │   └── fact_ids: [2, 7]
    │   ├── 父母
    │   │   └── fact_ids: [2]
    │   └── 兄弟姐妹
    │       └── fact_ids: [7]
    ├── Social (社交)
    │   └── fact_ids: [3, 8, 9]
    └── Interests (兴趣)
        └── fact_ids: [4, 6]
```

## 处理流程

```
输入:
  batch_id = "batch-123"
  touched_entity_ids = ["uuid-1", "uuid-2"]

Step 1: 查找受影响实体
  touched = [entity_id for fact in new_facts]

Step 2: 锁定实体 (防止并发)
  SELECT * FROM entity_trees WHERE entity_id IN (?) FOR UPDATE

Step 3: 加载状态
  - 获取新事实列表
  - 获取现有树结构

Step 4: 计算更新
  - 合并事实
  - 提取 Aspects
  - 构建树结构

Step 5: 应用更新
  - INSERT / UPDATE entity_trees

输出:
  profiled_entities: 已剖析的实体列表
```

## 总结

这一阶段完成写入过程的核心功能：

| 功能 | 说明 |
|------|------|
| 查找受影响实体 | 找出需要更新的实体 |
| 锁定机制 | 防止并发冲突 |
| Aspect 提取 | 按维度组织事实 |
| 树构建/合并 | 更新树结构 |

**关键输出**:
- `EntityTreeModel`: 更新的实体树
- `profiled_entities`: 包含多个 Aspect 的实体