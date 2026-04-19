# Stage 5: 实体树更新 - Entity Tree Update

## 概述

实体树更新是 MemBrain 写入过程的最后一个阶段，负责在批次数据持久化后更新实体树结构。实体树是 MemBrain 记忆系统的核心数据结构，用于组织和管理实体之间的层级关系。

这一阶段会：
1. 识别本批次影响的实体
2. 加载实体的当前状态
3. 计算新的树结构
4. 应用更新并持久化

## 代码位置

- **更新器入口**: [entity_tree_updater.py](file:///home/project/MemBrain/membrain/memory/application/entity_tree_updater.py)
- **树计算管道**: [pipeline.py](file:///home/project/MemBrain/membrain/memory/core/entity_tree/pipeline.py)
- **树存储**: [entity_tree_store.py](file:///home/project/MemBrain/membrain/infra/persistence/entity_tree_store.py)

## 详细代码分析

### 5.1 更新器入口

```python
# membrain/memory/application/entity_tree_updater.py

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
        """更新批次相关的实体树。"""
        
        # ═══════════════════════════════════════════════════════════════
        # Step 1: 找出本批次影响的实体
        # ═══════════════════════════════════════════════════════════════
        
        touched_entity_ids = self._store.find_touched_entities(task_id, batch_id)
        if not touched_entity_ids:
            return []  # 无影响的实体
        
        # ═══════════════════════════════════════════════════════════════
        # Step 2: 锁定实体（防止并发冲突）
        # ═══════════════════════════════════════════════════════════════
        
        with self._store.lock_entities(task_id, touched_entity_ids) as db:
            
            # ═══════════════════════════════════════════════════════════════
            # Step 3: 加载更新状态
            # ═══════════════════════════════════════════════════════════════
            
            state = self._store.load_update_state(
                task_id,
                batch_id,
                touched_entity_ids=touched_entity_ids,
                db=db,
            )
            
            if not state.targets:
                return []
            
            # ═══════════════════════════════════════════════════════════════
            # Step 4: 计算树更新
            # ═══════════════════════════════════════════════════════════════
            
            result = await compute_entity_tree_updates(
                task_id=state.task_id,
                targets=state.targets,
                embed_client=embed_client,
                registry=registry,
                factory=factory,
            )
            
            # ═══════════════════════════════════════════════════════════════
            # Step 5: 应用更新
            # ═══════════════════════════════════════════════════════════════
            
            self._store.apply_updates(task_id, state, result, db=db)
            db.commit()
        
        return result.profiled_entities
```

### 5.2 查找受影响的实体

```python
# membrain/infra/persistence/entity_tree_store.py

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

### 5.3 加载更新状态

```python
def load_update_state(
    self,
    task_id: int,
    batch_id: str,
    touched_entity_ids: list[str],
    db,
) -> UpdateState:
    """加载需要更新的实体状态。"""
    
    # 查询新添加的事实
    new_facts = db.query(FactModel).filter(
        FactModel.task_id == task_id,
        FactModel.batch_id == batch_id,
        FactModel.status == "active",
    ).all()
    
    # 查询已有的实体树结构
    existing_trees = db.query(EntityTreeModel).filter(
        EntityTreeModel.task_id == task_id,
        EntityTreeModel.entity_id.in_(touched_entity_ids),
    ).all()
    
    # 构建更新状态
    targets = []
    for entity_id in touched_entity_ids:
        # 获取该实体相关的新事实
        entity_facts = [
            f for f in new_facts
            if any(fr.entity_id == entity_id for fr in f.refs)
        ]
        
        # 获取该实体的现有树（如果有）
        existing_tree = next(
            (t for t in existing_trees if t.entity_id == entity_id),
            None
        )
        
        targets.append({
            "entity_id": entity_id,
            "new_facts": entity_facts,
            "existing_tree": existing_tree,
        })
    
    return UpdateState(
        task_id=task_id,
        targets=targets,
    )
```

### 5.4 树更新计算

```python
# membrain/memory/core/entity_tree/pipeline.py

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
        
        # ═══════════════════════════════════════════════════════════════
        # Step 1: 更新事实集合
        # ═══════════════════════════════════════════════════════════════
        
        if existing_tree:
            # 追加新事实到现有集合
            existing_fact_ids = set(existing_tree.fact_ids)
            all_fact_ids = existing_fact_ids | {f.id for f in new_facts}
        else:
            # 新实体，只需新事实
            all_fact_ids = {f.id for f in new_facts}
        
        # ═══════════════════════════════════════════════════════════════
        # Step 2: 提取 Aspect（方面/维度）
        # ═══════════════════════════════════════════════════════════════
        
        aspects = extract_aspects_from_facts(new_facts)
        
        # ═══════════════════════════════════════════════════════════════
        # Step 3: 构建/更新树结构
        # ═══════════════════════════════════════════════════════════════
        
        if existing_tree:
            # 合并 aspect 到现有树
            tree = merge_aspects(existing_tree, aspects)
        else:
            # 创建新树
            tree = create_entity_tree(entity_id, aspects)
        
        # ═══════════════════════════════════════════════════════════════
        # Step 4: 审计和重组（如需要）
        # ═══════════════════════════════════════════════════════════════
        
        tree = audit_and_rebalance(tree)
        
        all_updates.append(tree)
        
        # 如果有多个 aspect，标记为已剖析
        if len(aspects) > 1:
            profiled_entities.append(entity_id)
    
    return TreeUpdateResult(
        profiled_entities=profiled_entities,
        updates=all_updates,
    )
```

### 5.5 Aspect 提取

```python
def extract_aspects_from_facts(facts: list[FactModel]) -> dict[str, list[FactModel]]:
    """从事实中提取 aspect（方面/维度）。"""
    
    aspects: dict[str, list[FactModel]] = defaultdict(list)
    
    for fact in facts:
        # 尝试从事实文本中推断 aspect
        aspect = infer_aspect(fact.text)
        aspects[aspect].append(fact)
    
    return dict(aspects)


def infer_aspect(fact_text: str) -> str:
    """从事实文本推断所属的 aspect。"""
    
    # 简单的基于关键词的分类
    # 实际实现可能使用 LLM 进行分类
    
    text_lower = fact_text.lower()
    
    if any(kw in text_lower for kw in ["work", "job", "company", "career"]):
        return "Career"
    elif any(kw in text_lower for kw in ["family", "parent", "sibling", "child"]):
        return "Family"
    elif any(kw in text_lower for kw in ["friend", "hang out", "meet"]):
        return "Social"
    elif any(kw in text_lower for kw in ["live", "home", "house", "city"]):
        return "Residence"
    elif any(kw in text_lower for kw in ["hobby", "interest", "play", "sport"]):
        return "Interests"
    else:
        return "General"
```

### 5.6 树结构创建

```python
def create_entity_tree(entity_id: str, aspects: dict[str, list[FactModel]]) -> EntityTree:
    """为实体创建新的树结构。"""
    
    nodes = []
    
    # 创建根节点
    root = TreeNode(
        node_id=f"{entity_id}_root",
        entity_id=entity_id,
        label="root",
        fact_ids=[],
        children=[],
    )
    nodes.append(root)
    
    # 为每个 aspect 创建子节点
    for aspect_name, aspect_facts in aspects.items():
        aspect_node = TreeNode(
            node_id=f"{entity_id}_{aspect_name}",
            entity_id=entity_id,
            label=aspect_name,
            fact_ids=[f.id for f in aspect_facts],
            children=[],
        )
        nodes.append(aspect_node)
        root.children.append(aspect_node)
    
    return EntityTree(
        entity_id=entity_id,
        nodes=nodes,
        root=root,
    )
```

### 5.7 合并 Aspect

```python
def merge_aspects(
    existing_tree: EntityTree,
    new_aspects: dict[str, list[FactModel]],
) -> EntityTree:
    """将新的 aspects 合并到现有树中。"""
    
    # 找到根节点
    root = existing_tree.root
    
    # 现有的 aspect 节点映射
    existing_aspects = {node.label: node for node in root.children}
    
    for aspect_name, aspect_facts in new_aspects.items():
        if aspect_name in existing_aspects:
            # 追加事实到现有 aspect
            existing_node = existing_aspects[aspect_name]
            existing_node.fact_ids.extend([f.id for f in aspect_facts])
        else:
            # 创建新的 aspect 节点
            new_node = TreeNode(
                node_id=f"{existing_tree.entity_id}_{aspect_name}",
                entity_id=existing_tree.entity_id,
                label=aspect_name,
                fact_ids=[f.id for f in aspect_facts],
                children=[],
            )
            root.children.append(new_node)
    
    return existing_tree
```

### 5.8 审计和重组

```python
def audit_and_rebalance(tree: EntityTree) -> EntityTree:
    """审计树结构并进行必要的重组。"""
    
    # 检查是否有过于庞大的节点
    root = tree.root
    
    for node in tree.nodes:
        if len(node.fact_ids) > MAX_FACTS_PER_NODE:
            # 拆分大节点
            node = split_node(node)
    
    # 检查是否有空的 aspect 节点
    root.children = [c for c in root.children if c.fact_ids]
    
    return tree


def split_node(node: TreeNode) -> list[TreeNode]:
    """将大节点拆分为多个子节点。"""
    
    # 基于主题将事实分组
    groups = group_facts_by_topic(node.fact_ids)
    
    # 创建子节点
    children = []
    for topic, fact_ids in groups.items():
        child = TreeNode(
            node_id=f"{node.node_id}_{topic}",
            entity_id=node.entity_id,
            label=topic,
            fact_ids=fact_ids,
            children=[],
        )
        children.append(child)
    
    node.children.extend(children)
    node.fact_ids = []  # 清空父节点的事实
    
    return children
```

### 5.9 应用更新

```python
def apply_updates(
    self,
    task_id: int,
    state: UpdateState,
    result: TreeUpdateResult,
    db,
) -> None:
    """将计算的更新应用到数据库。"""
    
    for update in result.updates:
        # 检查树是否已存在
        existing = db.query(EntityTreeModel).filter(
            EntityTreeModel.task_id == task_id,
            EntityTreeModel.entity_id == update.entity_id,
        ).first()
        
        if existing:
            # 更新现有树
            existing.fact_ids = update.get_all_fact_ids()
            existing.structure_json = update.to_json()
            existing.updated_at = datetime.utcnow()
        else:
            # 创建新树
            new_tree = EntityTreeModel(
                task_id=task_id,
                entity_id=update.entity_id,
                fact_ids=update.get_all_fact_ids(),
                structure_json=update.to_json(),
            )
            db.add(new_tree)
        
        db.flush()
```

## 数据结构

### EntityTreeModel

```python
class EntityTreeModel(Base):
    __tablename__ = "entity_trees"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, nullable=False, index=True)
    entity_id = Column(String, nullable=False, index=True)
    fact_ids = Column(JSON)                    # 所有事实 ID 列表
    structure_json = Column(JSON)              # 树结构 JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("task_id", "entity_id", name="uix_task_entity"),
    )
```

### TreeNode

```python
@dataclass
class TreeNode:
    node_id: str
    entity_id: str
    label: str                    # 节点标签 (如 "Career", "Family")
    fact_ids: list[int]          # 该节点包含的事实 ID
    children: list[TreeNode]     # 子节点
```

### EntityTree

```python
@dataclass
class EntityTree:
    entity_id: str
    nodes: list[TreeNode]
    root: TreeNode
    
    def get_all_fact_ids(self) -> list[int]:
        """递归获取所有事实 ID。"""
        ids = []
        def _collect(node):
            ids.extend(node.fact_ids)
            for child in node.children:
                _collect(child)
        _collect(self.root)
        return ids
```

## 树结构示例

### 实体树结构

```
实体: Caroline (uuid-1)
树结构:
└── root
    ├── Career
    │   ├── 工作经历 (fact_ids: [1, 5, 10])
    │   └── 技能 (fact_ids: [3, 8])
    ├── Family
    │   ├── 父母 (fact_ids: [2, 7])
    │   └── 兄弟姐妹 (fact_ids: [4, 9])
    ├── Social
    │   └── 朋友 (fact_ids: [6])
    └── Interests
        ├── 爱好 (fact_ids: [11])
        └── 活动 (fact_ids: [12])
```

### JSON 表示

```json
{
  "entity_id": "uuid-1",
  "root": {
    "node_id": "uuid-1_root",
    "label": "root",
    "fact_ids": [],
    "children": [
      {
        "node_id": "uuid-1_Career",
        "label": "Career",
        "fact_ids": [1, 5, 10, 3, 8],
        "children": []
      },
      {
        "node_id": "uuid-1_Family",
        "label": "Family",
        "fact_ids": [2, 7, 4, 9],
        "children": []
      }
    ]
  }
}
```

## 完整处理流程

```
输入:
  task_id = 1
  batch_id = "batch-123"
  touched_entity_ids = ["uuid-1", "uuid-2"]

Step 1: 查找受影响的实体
  touched = ["uuid-1", "uuid-2"]

Step 2: 锁定实体
  lock("uuid-1", "uuid-2")

Step 3: 加载状态
  state = {
    "task_id": 1,
    "targets": [
      {
        "entity_id": "uuid-1",
        "new_facts": [fact_1, fact_2, fact_3],
        "existing_tree": Tree(...),
      },
      {
        "entity_id": "uuid-2",
        "new_facts": [fact_4],
        "existing_tree": None,
      }
    ]
  }

Step 4: 计算更新
  对于 uuid-1:
    - 合并新事实到现有树
    - 提取 aspects: {"Career": [f1], "Family": [f2, f3]}
    - 审计重组
  
  对于 uuid-2:
    - 创建新树
    - 提取 aspects: {"General": [f4]}

Step 5: 应用更新
  - 保存树结构到数据库
  - 提交事务

Step 6: 返回
  返回 profiled_entities = ["uuid-1", "uuid-2"]
```

## 配置参数

```python
# membrain/config.py

# 树配置
MAX_FACTS_PER_NODE = 50           # 每个节点最大事实数
DEFAULT_ASPECTS = [               # 默认 aspect 列表
    "Career",
    "Family", 
    "Social",
    "Residence",
    "Interests",
    "General",
]

# 锁定配置
ENTITY_LOCK_TIMEOUT = 30          # 实体锁定超时时间（秒）
```

## 并发控制

```python
# membrain/infra/persistence/entity_tree_store.py

@contextmanager
def lock_entities(self, task_id: int, entity_ids: list[str]):
    """锁定实体，防止并发更新冲突。"""
    
    # 使用数据库行锁
    query = (
        select(EntityTreeModel)
        .where(
            EntityTreeModel.task_id == task_id,
            EntityTreeModel.entity_id.in_(entity_ids),
        )
        .with_for_update()  # 行锁
    )
    
    # 执行查询，获取锁
    rows = db.execute(query).fetchall()
    
    try:
        yield db
    finally:
        # 事务提交后自动释放锁
        pass
```

## 总结

实体树更新阶段的核心逻辑：

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 查找受影响实体 | 识别需要更新的实体 |
| 2 | 锁定实体 | 防止并发冲突 |
| 3 | 加载状态 | 获取现有树和新事实 |
| 4 | 计算更新 | 提取 aspects，构建树结构 |
| 5 | 应用更新 | 持久化到数据库 |

**设计亮点**:

1. **层级组织**: 将事实按 aspect 分类到树结构中
2. **增量更新**: 只更新受影响的实体
3. **并发控制**: 锁定机制防止冲突
4. **自动拆分**: 大节点自动拆分保持平衡
5. **审计机制**: 确保树结构健康

这一阶段完成了写入过程的最后一步，将事实组织成有结构的实体树，为后续的检索提供了高效的组织基础。

---

## 写入过程完整流程总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MemBrain 写入过程 (5 个阶段)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: 实体提取                                                  │
│  ├── 第一轮: 基础 LLM 提取                                          │
│  ├── 加载已有实体上下文                                              │
│  └── 第二轮: 上下文增强提取                                          │
│                                                                      │
│  Stage 2: 事实生成                                                  │
│  ├── 调用 fact-generator agent                                      │
│  ├── 实体覆盖验证 (实时)                                            │
│  └── 回退过滤机制                                                   │
│                                                                      │
│  Stage 3: 实体消重                                                  │
│  ├── Layer 1: 精确匹配 (小写+空白折叠)                            │
│  ├── Layer 2: MinHash + Jaccard 模糊匹配                         │
│  └── Layer 3: LLM 语义匹配                                         │
│                                                                      │
│  Stage 4: 数据库持久化                                              │
│  ├── 构建引用映射                                                   │
│  ├── 创建/更新实体                                                  │
│  ├── 插入事实 + 生成向量                                           │
│  └── 建立事实-实体关联                                              │
│                                                                      │
│  Stage 5: 实体树更新                                                │
│  ├── 查找受影响实体                                                 │
│  ├── 加载现有状态                                                   │
│  ├── 提取 Aspects                                                  │
│  ├── 构建/合并树结构                                               │
│  └── 持久化更新                                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```