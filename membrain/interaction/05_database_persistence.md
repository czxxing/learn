# Stage 5: 数据库持久化

## 概述

这一阶段负责将提取的实体和事实写入数据库，包括：
1. 创建或更新实体记录
2. 插入事实记录
3. 建立事实-实体关联
4. 生成嵌入向量

## 代码位置

- **持久化入口**: [memory_ingest_store.py](file:///home/project/MemBrain/membrain/infra/persistence/memory_ingest_store.py#L188-L213)
- **批量写入**: [batch_writer.py](file:///home/project/MemBrain/membrain/infra/persistence/batch_writer.py)

## 详细代码分析

### 5.1 持久化入口

```python
# memory_ingest_store.py

def persist_batch(
    self,
    task_id: int,
    batch_id: str,
    facts: list[dict],
    decisions: list[dict],
    embed_client,
    batch_index: int | None,
    session_number: int | None,
) -> None:
    """将批次数据持久化到数据库。"""
    
    with self._transactions.write() as db:
        # 构建引用映射
        ref_to_entity_id = entity_queries.build_ref_map(db, task_id)
        
        # 执行批量写入
        write_batch_results(
            db=db,
            task_id=task_id,
            batch_id=batch_id,
            facts=facts,
            decisions=decisions,
            embed_client=embed_client,
            ref_to_entity_id=ref_to_entity_id,
            batch_index=batch_index,
            session_number=session_number,
        )
```

### 5.2 引用映射构建

```python
# entities.py

def build_ref_map(db, task_id: int) -> dict[str, str]:
    """构建实体引用到实体 ID 的映射。"""
    
    query = (
        select(EntityModel.canonical_ref, EntityModel.entity_id)
        .where(EntityModel.task_id == task_id)
    )
    results = db.execute(query).fetchall()
    
    # 构建映射字典（统一小写）
    ref_map = {}
    for canonical_ref, entity_id in results:
        ref_map[canonical_ref.lower()] = entity_id
    
    return ref_map
```

### 5.3 批量写入实现

```python
# batch_writer.py

def write_batch_results(
    db,
    task_id: int,
    batch_id: str,
    facts: list[dict],
    decisions: list[dict],
    embed_client,
    ref_to_entity_id: dict[str, str],
    batch_index: int | None,
    session_number: int | None,
) -> None:
    """执行批量写入操作。"""
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 1: 处理实体决策
    # ═══════════════════════════════════════════════════════════════
    
    for decision in decisions:
        if decision["action"] == "create":
            # 创建新实体
            entity_id = create_entity(
                db=db,
                task_id=task_id,
                canonical_ref=decision["canonical_ref"],
                desc=decision.get("updated_desc", ""),
            )
            ref_to_entity_id[decision["canonical_ref"].lower()] = entity_id
            
        elif decision["action"] == "merge":
            # 更新已有实体描述
            target_id = decision["target_entity_id"]
            if decision.get("updated_desc"):
                update_entity_description(
                    db=db,
                    entity_id=target_id,
                    new_desc=decision["updated_desc"],
                )
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 2: 准备事实数据
    # ═══════════════════════════════════════════════════════════════
    
    fact_records = []
    for fact in facts:
        # 提取事实中的实体引用
        entity_refs = _ENTITY_BRACKET_RE.findall(fact["text"])
        
        # 转换引用为实体 ID
        entity_ids = []
        for ref in entity_refs:
            ref_lower = ref.lower()
            if ref_lower in ref_to_entity_id:
                entity_ids.append(ref_to_entity_id[ref_lower])
        
        # 生成嵌入向量
        text_embedding = embed_client.embed([fact["text"]])[0]
        
        fact_records.append({
            "task_id": task_id,
            "batch_id": batch_id,
            "session_number": session_number,
            "text": fact["text"],
            "text_embedding": text_embedding,
            "time_info": fact.get("time"),
            "status": "active",
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 3: 批量插入事实
    # ═══════════════════════════════════════════════════════════════
    
    db.bulk_insert(FactModel, fact_records)
    db.flush()  # 获取插入的事实 ID
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 4: 建立事实-实体关联
    # ═══════════════════════════════════════════════════════════════
    
    # 获取刚插入的事实 ID
    fact_ids = get_fact_ids_by_batch(db, task_id, batch_id)
    
    ref_records = []
    for fact_id, fact in zip(fact_ids, facts):
        entity_refs = _ENTITY_BRACKET_RE.findall(fact["text"])
        
        for ref in entity_refs:
            ref_lower = ref.lower()
            if ref_lower in ref_to_entity_id:
                ref_records.append({
                    "fact_id": fact_id,
                    "entity_id": ref_to_entity_id[ref_lower],
                    "alias_text": ref,
                })
    
    # 批量插入引用关系
    db.bulk_insert(FactRefModel, ref_records)
```

### 5.4 实体创建

```python
def create_entity(
    db,
    task_id: int,
    canonical_ref: str,
    desc: str,
) -> str:
    """创建新实体并返回 entity_id。"""
    
    entity_id = str(uuid.uuid4())
    
    # 生成描述嵌入向量
    desc_embedding = embed_client.embed([desc])[0] if desc else None
    
    entity = EntityModel(
        entity_id=entity_id,
        task_id=task_id,
        canonical_ref=canonical_ref,
        desc=desc,
        desc_embedding=desc_embedding,
    )
    
    db.add(entity)
    db.flush()
    
    return entity_id
```

### 5.5 实体描述更新

```python
def update_entity_description(
    db,
    entity_id: str,
    new_desc: str,
) -> None:
    """更新实体描述（追加新描述）。"""
    
    entity = db.query(EntityModel).filter(
        EntityModel.entity_id == entity_id
    ).first()
    
    if entity:
        # 追加新描述
        if entity.desc:
            entity.desc = f"{entity.desc}; {new_desc}"
        else:
            entity.desc = new_desc
        
        # 重新生成嵌入向量
        entity.desc_embedding = embed_client.embed([entity.desc])[0]
        
        db.flush()
```

## 数据模型

### EntityModel

```python
class EntityModel(Base):
    __tablename__ = "entities"
    
    entity_id = Column(String, primary_key=True)
    task_id = Column(Integer, nullable=False, index=True)
    canonical_ref = Column(String, nullable=False)
    desc = Column(Text)
    desc_embedding = Column(Vector(768))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### FactModel

```python
class FactModel(Base):
    __tablename__ = "facts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, nullable=False, index=True)
    batch_id = Column(String, nullable=False, index=True)
    session_number = Column(Integer, nullable=True)
    text = Column(Text, nullable=False)
    text_embedding = Column(Vector(768))
    status = Column(String, default="active")
    fact_ts = Column(DateTime, default=datetime.utcnow)
```

### FactRefModel

```python
class FactRefModel(Base):
    __tablename__ = "fact_refs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fact_id = Column(Integer, ForeignKey("facts.id"), nullable=False)
    entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
    alias_text = Column(String)
    
    __table_args__ = (
        Index("idx_fact_refs_entity", "entity_id"),
    )
```

## 写入流程图

```
输入:
  decisions = [
      {"action": "create", "canonical_ref": "Lisa", "updated_desc": "sister"},
      {"action": "merge", "target_entity_id": "uuid-1", "updated_desc": "met at coffee shop"},
  ]
  facts = [
      {"text": "[User] met [Lisa] at [coffee shop]", "time": "yesterday"},
      {"text": "[Lisa] is [User]'s sister", "time": None},
  ]

    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 构建引用映射                                                     │
│ ref_to_entity_id = {"caroline": "uuid-1", "coffee shop": "uuid-2", ...}│
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 处理实体决策                                                      │
│                                                                     │
│ • action="create": 创建新实体                                      │
│   - INSERT entities (entity_id, canonical_ref, desc, desc_embedding)│
│   - 更新 ref_to_entity_id                                        │
│                                                                     │
│ • action="merge": 更新已有实体                                     │
│   - UPDATE entities SET desc = CONCAT(desc, ?)                  │
│   - UPDATE entities SET desc_embedding = embed(desc)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 准备事实数据                                                     │
│                                                                     │
│ • 提取 [entity] 引用                                            │
│ • 转换为实体 ID                                                  │
│ • 生成 text_embedding                                            │
│                                                                     │
│ fact_records = [                                                   │
│   {                                                                │
│     "task_id": 1,                                                │
│     "batch_id": "batch-123",                                     │
│     "session_number": 1,                                          │
│     "text": "[User] met [Lisa] at [coffee shop]",              │
│     "text_embedding": [...],                                      │
│     "time_info": "yesterday",                                    │
│     "status": "active"                                           │
│   },                                                              │
│ ]                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 批量插入事实                                                     │
│                                                                     │
│ INSERT INTO facts (task_id, batch_id, text, text_embedding, ...)│
│ VALUES (...), (...), ...                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 获取插入的事实 ID                                                 │
│                                                                     │
│ SELECT id FROM facts WHERE batch_id = 'batch-123'                │
│ fact_ids = [1, 2]                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 建立事实-实体关联                                                 │
│                                                                     │
│ ref_records = [                                                    │
│   {"fact_id": 1, "entity_id": "uuid-1", "alias_text": "User"}, │
│   {"fact_id": 1, "entity_id": "uuid-3", "alias_text": "Lisa"}, │
│   {"fact_id": 1, "entity_id": "uuid-2", "alias_text": "coffee shop"},│
│   {"fact_id": 2, "entity_id": "uuid-3", "alias_text": "Lisa"}, │
│   {"fact_id": 2, "entity_id": "uuid-1", "alias_text": "User"}, │
│ ]                                                                 │
│                                                                     │
│ INSERT INTO fact_refs (fact_id, entity_id, alias_text)          │
│ VALUES (...), (...), ...                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 提交事务                                                         │
│ COMMIT                                                            │
└─────────────────────────────────────────────────────────────────┘
```

## 输入示例

```python
# 输入
decisions = [
    {
        "action": "create",
        "canonical_ref": "Lisa",
        "updated_desc": "User's sister"
    },
    {
        "action": "merge",
        "target_entity_id": "uuid-1",  # Caroline
        "updated_desc": "met at coffee shop yesterday"
    }
]

facts = [
    {
        "text": "[User] met [Lisa] at [coffee shop] [yesterday]",
        "time": "yesterday"
    },
    {
        "text": "[Lisa] is [User]'s sister",
        "time": None
    }
]

ref_to_entity_id = {
    "user": "uuid-1",
    "caroline": "uuid-1",
    "coffee shop": "uuid-2"
}

# 执行后的数据库记录

# entities 表
# | entity_id | task_id | canonical_ref | desc | ...
# | uuid-1    | 1       | Caroline      | User's mother; met at coffee shop yesterday | ...
# | uuid-3    | 1       | Lisa          | User's sister | ...

# facts 表
# | id | task_id | batch_id | text | text_embedding | ...
# | 1  | 1       | batch-1 | [User] met [Lisa] at [coffee shop] [yesterday] | [...] | ...
# | 2  | 1       | batch-1 | [Lisa] is [User]'s sister | [...] | ...

# fact_refs 表
# | fact_id | entity_id | alias_text |
# | 1       | uuid-1    | User       |
# | 1       | uuid-3    | Lisa       |
# | 1       | uuid-2    | coffee shop|
# | 2       | uuid-3    | Lisa       |
# | 2       | uuid-1    | User       |
```

## 与后续阶段的关联

```
Stage 5: 数据库持久化
    │
    ├──→ EntityModel: 实体记录
    │
    ├──→ FactModel: 事实记录
    │
    ├──→ FactRefModel: 事实-实体关联
    │
    └──→ 触发 Stage 6: 实体树更新
```

## 关键设计决策

### 1. 事务保证

```python
with self._transactions.write() as db:
    # 所有操作在一个事务中
    # 失败时全部回滚
```

### 2. 批量写入

```python
# 使用 bulk_insert 提高性能
db.bulk_insert(FactModel, fact_records)
```

### 3. 向量同步

```python
# 事实和实体同时生成向量
text_embedding = embed_client.embed([fact["text"]])[0]
desc_embedding = embed_client.embed([entity.desc])[0]
```

### 4. 追加描述

```python
# 合并时追加而非覆盖
entity.desc = f"{entity.desc}; {new_desc}"
```

## 总结

这一阶段的核心功能：

| 功能 | 说明 |
|------|------|
| 引用映射构建 | 建立实体引用到 ID 的映射 |
| 实体创建/更新 | 写入实体记录 |
| 事实插入 | 批量写入事实及向量 |
| 关联建立 | 建立事实-实体关系 |
| 事务保证 | 原子性操作 |

**关键输出**:
- 持久化的实体记录
- 持久化的事实记录
- 事实-实体关联关系
- 触发实体树更新