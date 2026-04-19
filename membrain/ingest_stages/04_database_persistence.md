# Stage 4: 数据库持久化 - Database Persistence

## 概述

数据库持久化是 MemBrain 写入过程的第四阶段，负责将提取的事实和实体写入到数据库中。这一阶段将前面阶段生成的结构化数据转换为数据库记录，同时生成嵌入向量用于后续的语义检索。

持久化是写入流程的关键环节，需要确保数据的一致性和完整性。

## 代码位置

- **持久化入口**: [memory_ingest_store.py](file:///home/project/MemBrain/membrain/infra/persistence/memory_ingest_store.py#L188-L213)
- **批量写入**: [batch_writer.py](file:///home/project/MemBrain/membrain/infra/persistence/batch_writer.py)
- **事务管理**: [transaction_manager.py](file:///home/project/MemBrain/membrain/infra/transaction_manager.py)

## 详细代码分析

### 4.1 持久化入口

```python
# membrain/infra/persistence/memory_ingest_store.py

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
        # Step 1: 构建引用映射
        # 将批处理中的实体引用映射到数据库中的实体 ID
        ref_to_entity_id = entity_queries.build_ref_map(db, task_id)
        
        # Step 2: 执行批量写入
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

### 4.2 引用映射构建

```python
# membrain/infra/queries/entities.py

def build_ref_map(db, task_id: int) -> dict[str, str]:
    """构建实体引用到实体 ID 的映射。"""
    
    query = (
        select(EntityModel.canonical_ref, EntityModel.entity_id)
        .where(EntityModel.task_id == task_id)
    )
    results = db.execute(query).fetchall()
    
    # 构建映射字典
    ref_map = {}
    for canonical_ref, entity_id in results:
        ref_map[canonical_ref.lower()] = entity_id
    
    return ref_map
```

### 4.3 批量写入实现

```python
# membrain/infra/persistence/batch_writer.py

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
    # Step 1: 处理实体决策
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
    # Step 2: 处理事实
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
    
    # 批量插入事实
    db.bulk_insert(FactModel, fact_records)
    
    # ═══════════════════════════════════════════════════════════════
    # Step 3: 建立事实-实体关联
    # ═══════════════════════════════════════════════════════════════
    
    # 获取刚插入的事实的 ID
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

### 4.4 实体创建

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
    db.flush()  # 确保获取 ID
    
    return entity_id
```

### 4.5 实体描述更新

```python
def update_entity_description(
    db,
    entity_id: str,
    new_desc: str,
) -> None:
    """更新实体描述。"""
    
    # 查询现有实体
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
    canonical_ref = Column(String, nullable=False)  # 规范名称
    desc = Column(Text)                            # 描述
    desc_embedding = Column(Vector(768))           # 描述向量
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
    text = Column(Text, nullable=False)           # 事实文本
    text_embedding = Column(Vector(768))          # 文本向量
    status = Column(String, default="active")    # 状态: active/archived
    fact_ts = Column(DateTime, default=datetime.utcnow)
```

### FactRefModel

```python
class FactRefModel(Base):
    __tablename__ = "fact_refs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fact_id = Column(Integer, ForeignKey("facts.id"), nullable=False)
    entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
    alias_text = Column(String)                   # 引用文本
    
    __table_args__ = (
        Index("idx_fact_refs_entity", "entity_id"),
    )
```

## 完整写入流程

```
输入:
  decisions = [
      {"action": "create", "canonical_ref": "Lisa", "updated_desc": "sister"},
      {"action": "merge", "target_entity_id": "uuid-1", "updated_desc": "met at coffee shop"},
  ]
  facts = [
      {"text": "[Lisa] is [Caroline]'s sister", "time": None},
      {"text": "[Caroline] met [Lisa] at [coffee shop] [yesterday]", "time": "yesterday"},
  ]

Step 1: 构建引用映射
  ref_to_entity_id = {
      "caroline": "uuid-1",
      "coffee shop": "uuid-2",
      ...
  }

Step 2: 处理实体决策
  - Lisa: 创建新实体 uuid-3
  - Caroline (uuid-1): 追加描述 "met at coffee shop"

Step 3: 插入事实
  - 事实 1: "[Lisa] is [Caroline]'s sister"
  - 事实 2: "[Caroline] met [Lisa] at [coffee shop] [yesterday]"

Step 4: 生成嵌入向量
  - text_embedding_1 = embed("[Lisa] is [Caroline]'s sister")
  - text_embedding_2 = embed("[Caroline] met [Lisa] at [coffee shop] [yesterday]")

Step 5: 建立关联
  - fact_ref_1: (fact_id=1, entity_id=uuid-3, alias_text="Lisa")
  - fact_ref_1: (fact_id=1, entity_id=uuid-1, alias_text="Caroline")
  - fact_ref_2: (fact_id=2, entity_id=uuid-1, alias_text="Caroline")
  - fact_ref_2: (fact_id=2, entity_id=uuid-3, alias_text="Lisa")
  - fact_ref_2: (fact_id=2, entity_id=uuid-2, alias_text="coffee shop")
```

## 事务管理

```python
# membrain/infra/transaction_manager.py

class TransactionManager:
    """数据库事务管理器。"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    @contextmanager
    def read(self):
        """只读事务。"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    @contextmanager
    def write(self):
        """读写事务。"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()  # 提交事务
        except Exception:
            session.rollback()  # 回滚事务
            raise
        finally:
            session.close()
```

**事务特性**:
- **原子性**: 所有操作要么全部成功，要么全部回滚
- **一致性**: 写入失败时自动回滚
- **隔离性**: 每个批次独立处理

## 配置参数

```python
# membrain/config.py

# 数据库配置
DATABASE_URL = "postgresql://user:pass@localhost/membrain"
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20

# 嵌入向量配置
EMBEDDING_DIMENSION = 768  # 向量维度
EMBEDDING_BATCH_SIZE = 32  # 批处理大小
```

## 错误处理

### 1. 嵌入生成失败

```python
try:
    text_embedding = embed_client.embed([fact["text"]])[0]
except Exception as e:
    log.warning("Embedding failed for fact: %s", e)
    text_embedding = None  # 允许空向量
```

### 2. 实体 ID 映射缺失

```python
for ref in entity_refs:
    ref_lower = ref.lower()
    if ref_lower in ref_to_entity_id:
        entity_ids.append(ref_to_entity_id[ref_lower])
    else:
        log.warning("Entity ref not in map: %s", ref)
        # 跳过未映射的引用
```

### 3. 事务回滚

```python
try:
    with self._transactions.write() as db:
        write_batch_results(...)
except Exception as e:
    log.error("Batch write failed: %s", e)
    # 事务自动回滚
    raise
```

## 总结

数据库持久化阶段的核心逻辑：

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 构建引用映射 | 将实体引用映射到数据库 ID |
| 2 | 处理实体决策 | 创建或更新实体记录 |
| 3 | 插入事实 | 批量写入事实及向量 |
| 4 | 建立关联 | 写入事实-实体引用关系 |

**设计亮点**:

1. **事务保证**: 原子性操作确保数据一致性
2. **批量写入**: 高效处理大量数据
3. **向量同步**: 事实和实体同时生成向量
4. **引用追踪**: 完整记录实体引用关系
5. **错误容忍**: 部分失败不影响整体

这一阶段将结构化数据写入数据库，为后续的检索提供了数据基础。