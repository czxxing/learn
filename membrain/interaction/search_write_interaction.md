# MemBrain 数据生成与搜索交互过程

## 概述

MemBrain 是一个记忆系统，其核心功能是将用户对话转化为结构化记忆，并在需要时检索这些记忆来回答问题。理解数据生成（写入）和搜索的交互过程，是掌握 MemBrain 整体架构的关键。

本文档详细分析这两个过程的交互机制，展示数据如何在系统中流转。

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MemBrain 架构                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐                           ┌─────────────┐                │
│   │  用户输入    │                           │  用户查询   │                │
│   │ (消息/对话) │                           │ (问题)     │                │
│   └──────┬──────┘                           └──────┬──────┘                │
│          │                                          │                       │
│          ▼                                          ▼                       │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │                    数据生成 (写入) 过程                           │    │
│   │                                                                   │    │
│   │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │    │
│   │   │ 实体提取 │ → │ 事实生成 │ → │ 实体消重 │ → │ 持久化   │ → 实体树  │    │
│   │   └─────────┘   └─────────┘   └─────────┘   └─────────┘        │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                     │                                      │
│                                     ▼                                      │
│                          ┌──────────────────┐                            │
│                          │    数据库         │                            │
│                          │  • entities      │                            │
│                          │  • facts         │                            │
│                          │  • entity_trees  │                            │
│                          │  • sessions      │                            │
│                          └────────┬─────────┘                            │
│                                   │                                      │
│                                   ▼                                      │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │                    数据搜索 (检索) 过程                           │    │
│   │                                                                   │    │
│   │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │    │
│   │   │查询扩展 │ → │多路径   │ → │结果融合 │ → │实体解析 │ → ...  │    │
│   │   └─────────┘   └─────────┘   └─────────┘   └─────────┘        │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                     │                                      │
│                                     ▼                                      │
│                          ┌──────────────────┐                            │
│                          │   LLM 回答       │                            │
│                          └──────────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 数据流详解

### 阶段 1: 数据写入时的实体解析交互

当新数据写入时，实体解析模块会与数据库进行多次交互：

```python
# 阶段 1: 实体提取
# ─────────────────────────────────────────────────────────────────────────
# 输入: 用户消息
# "Caroline met her sister Lisa at the coffee shop yesterday."

# 调用链:
# 1. 加载已有实体上下文 (用于第二轮增强提取)
entity_context = self._ingest_store.load_extraction_context(
    entity_names=["Caroline", "Lisa", "coffee shop"],  # 第一次提取的实体
    task_id=task_pk,
    embed_client=self._embed_client,
)

# 内部实现: 查询数据库中已存在的实体
# SELECT * FROM entities WHERE task_id = ? AND canonical_ref IN (?)
# 返回: [
#   EntityContext(entity_id="uuid-1", canonical_ref="Caroline", aliases=[...], desc=...),
#   EntityContext(entity_id="uuid-2", canonical_ref="Lisa", ...),
# ]
```

### 阶段 2: 数据写入时的实体消重交互

实体消重需要与数据库进行深度交互：

```python
# 阶段 2: 实体消重 - 检索候选实体
# ─────────────────────────────────────────────────────────────────────────

# 1. 检索候选实体池
entries, by_entity_id, aliases_by_entity = retrieve_candidate_pool(
    entity_names=["Caroline", "Lisa", "coffee shop"],
    task_id=task_pk,
    db=db,
    embed_client=embed_client,
)

# 内部实现:
# - BM25 搜索: SELECT * FROM entities WHERE ...
# - 向量搜索: SELECT * FROM entities ORDER BY desc_embedding <-> ?
# - 合并结果，构建候选池

# 2. 三层消重
# - Layer 1: 精确匹配 (内存中)
# - Layer 2: MinHash + Jaccard (内存中)
# - Layer 3: LLM 语义匹配 (可能需要更多 DB 查询)

# 3. 决策结果
decisions = [
    {"action": "merge", "batch_ref": "Caroline", "target_entity_id": "uuid-1"},
    {"action": "create", "batch_ref": "Lisa", "target_entity_id": None},
    {"action": "create", "batch_ref": "coffee shop", "target_entity_id": None},
]
```

### 阶段 3: 数据持久化时的写入操作

```python
# 阶段 3: 持久化 - 写入数据库
# ─────────────────────────────────────────────────────────────────────────

# 1. 创建新实体 (对于 action="create" 的决策)
new_entities = [
    {"canonical_ref": "Lisa", "desc": "Caroline's sister"},
    {"canonical_ref": "coffee shop", "desc": "Local coffee shop"},
]

# 写入操作:
# INSERT INTO entities (entity_id, task_id, canonical_ref, desc, desc_embedding)
# VALUES (?, ?, ?, ?, ?)

# 2. 更新已有实体描述 (对于 action="merge" 的决策)
# UPDATE entities SET desc = CONCAT(desc, ?) WHERE entity_id = ?

# 3. 写入事实
new_facts = [
    {"text": "[Caroline] met [Lisa] at [coffee shop] [yesterday]", "time": "yesterday"},
]

# 写入操作:
# INSERT INTO facts (task_id, batch_id, session_number, text, text_embedding, status)
# VALUES (?, ?, ?, ?, ?, 'active')

# 4. 建立事实-实体关联
fact_refs = [
    {"fact_id": 1, "entity_id": "uuid-1", "alias_text": "Caroline"},
    {"fact_id": 1, "entity_id": "uuid-2", "alias_text": "Lisa"},
    {"fact_id": 1, "entity_id": "uuid-3", "alias_text": "coffee shop"},
]

# 写入操作:
# INSERT INTO fact_refs (fact_id, entity_id, alias_text)
# VALUES (?, ?, ?)
```

### 阶段 4: 实体树更新

```python
# 阶段 4: 实体树更新
# ─────────────────────────────────────────────────────────────────────────

# 1. 查找本批次影响的实体
touched_entities = find_touched_entities(task_id, batch_id)
# SELECT DISTINCT entity_id FROM fact_refs 
#   JOIN facts ON fact_refs.fact_id = facts.id 
#   WHERE facts.batch_id = ?

# 2. 加载现有树结构
existing_trees = load_entity_trees(task_id, touched_entities)
# SELECT * FROM entity_trees WHERE entity_id IN (?)

# 3. 创建/更新树结构
# INSERT / UPDATE entity_trees
```

### 阶段 5: 搜索时的数据交互

当用户提出问题时，搜索过程会与数据库进行多次交互：

```python
# 搜索过程 - 查询扩展阶段
# ─────────────────────────────────────────────────────────────────────────

# 1. 加载实体上下文 (用于实体感知的查询扩展)
entity_context = load_extraction_context(
    entity_names=[],  # 初始为空
    task_id=task_pk,
    embed_client=embed_client,
)

# 2. 多路径检索
# Path A: BM25 搜索
bm25_results = bm25_search(query, task_id, db, limit=20)
# SELECT * FROM facts 
#   WHERE task_id = ? AND status = 'active'
#   ORDER BY ts_rank(text, query) DESC LIMIT 20

# Path B: 向量搜索
embed_results = embedding_search(query_vector, task_id, db, limit=20)
# SELECT * FROM facts 
#   WHERE task_id = ? AND status = 'active'
#   ORDER BY text_embedding <-> ? LIMIT 20

# Path C: 实体树搜索
tree_results = tree_beam_search(query, task_id, db, limit=20)
# SELECT * FROM entity_trees WHERE ...
# 然后查询各树节点中的事实
```

## 完整交互流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     写入-搜索完整交互流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  时间线 →                                                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        写入过程                                       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│       │                                                                   │
│       ▼                                                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │
│  │  用户消息     │     │  实体提取    │     │  事实生成    │           │
│  │ "Caroline    │ ──▶ │  LLM 调用    │ ──▶ │  LLM 调用    │           │
│  │  met Lisa    │     │  (第一轮)    │     │              │           │
│  │  yesterday"  │     └──────┬───────┘     └──────┬───────┘           │
│  └──────────────┘            │                     │                   │
│                               │                     ▼                   │
│                               │            ┌──────────────────┐         │
│                               │            │ 实体覆盖验证     │         │
│                               │            │ (检查引用合法性) │         │
│                               │            └────────┬─────────┘         │
│                               │                     │                   │
│                               ▼                     ▼                   │
│                      ┌──────────────────────────────────────────┐       │
│                      │         实体消重 (三层策略)              │       │
│                      │                                          │       │
│                      │  ┌────────────────────────────────────┐ │       │
│                      │  │ Layer 1: 精确匹配                  │ │       │
│                      │  │   DB 查询: 标准化名称匹配         │ │       │
│                      │  └────────────────────────────────────┘ │       │
│                      │                  │                       │       │
│                      │                  ▼                       │       │
│                      │  ┌────────────────────────────────────┐ │       │
│                      │  │ Layer 2: MinHash + Jaccard       │ │       │
│                      │  │   DB 查询: 获取候选实体           │ │       │
│                      │  └────────────────────────────────────┘ │       │
│                      │                  │                       │       │
│                      │                  ▼                       │       │
│                      │  ┌────────────────────────────────────┐ │       │
│                      │  │ Layer 3: LLM 语义匹配             │ │       │
│                      │  │   DB 查询: 获取实体描述           │ │       │
│                      │  └────────────────────────────────────┘ │       │
│                      └──────────────────────────────────────────┘       │
│                               │                                           │
│                               ▼                                           │
│                      ┌──────────────────────────────────────────┐       │
│                      │            数据库持久化                   │       │
│                      │                                          │       │
│                      │  ┌────────────────────────────────────┐  │       │
│                      │  │ INSERT entities                   │  │       │
│                      │  │ INSERT facts                      │  │       │
│                      │  │ INSERT fact_refs                  │  │       │
│                      │  │ UPDATE entity_trees               │  │       │
│                      │  └────────────────────────────────────┘  │       │
│                      └──────────────────────────────────────────┘       │
│                               │                                           │
│  ════════════════════════════════════════════════════════════════════   │
│                               │                                           │
│                               ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        搜索过程 (后续查询)                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│       │                                                                   │
│       ▼                                                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │
│  │  用户问题    │     │  查询扩展    │     │  多路径检索  │           │
│  │ "When did   │ ──▶ │  LLM 生成    │ ──▶ │  BM25+向量   │           │
│  │  Caroline   │     │  多版本查询  │     │  +树搜索     │           │
│  │  meet Lisa?" │     └──────┬───────┘     └──────┬───────┘           │
│  └──────────────┘            │                     │                   │
│                               │                     ▼                   │
│                               │            ┌──────────────────┐         │
│                               │            │  DB 多次查询    │         │
│                               │            │ (6 条路径)     │         │
│                               │            └────────┬─────────┘         │
│                               │                     │                   │
│                               ▼                     ▼                   │
│                      ┌──────────────────────────────────────────┐       │
│                      │            结果融合                      │       │
│                      │         (RRF / Rerank)                  │       │
│                      └────────────────────┬───────────────────┘       │
│                                           │                            │
│                                           ▼                            │
│                      ┌──────────────────────────────────────────┐       │
│                      │            实体引用解析                   │       │
│                      │                                          │       │
│                      │  DB 查询:                                │       │
│                      │  - 获取所有唯一实体 ID                   │       │
│                      │  - 解析 [Caroline] → "Caroline (user)" │       │
│                      └────────────────────┬───────────────────┘       │
│                                           │                            │
│                                           ▼                            │
│                      ┌──────────────────────────────────────────┐       │
│                      │            会话检索                     │       │
│                      │  DB 查询: 按会话聚合事实                 │       │
│                      └────────────────────┬───────────────────┘       │
│                                           │                            │
│                                           ▼                            │
│                      ┌──────────────────────────────────────────┐       │
│                      │            上下文打包                    │       │
│                      │  - Token 预算控制                        │       │
│                      │  - 格式化输出                           │       │
│                      └────────────────────┬───────────────────┘       │
│                                           │                            │
│                                           ▼                            │
│                               ┌──────────────────┐                    │
│                               │   LLM 生成回答   │                    │
│                               └──────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 关键交互点

### 交互点 1: 实体上下文加载

**写入时**:
- 目的: 为第二轮实体提取提供已有实体信息
- DB 操作: SELECT entities, fact_refs
- 时机: 第一轮实体提取后

```python
# 伪代码
def load_extraction_context(entity_names, task_id, embed_client):
    # 1. BM25 搜索
    bm25_hits = bm25_search(entity_names, task_id, db)
    
    # 2. 向量搜索
    embed_hits = embedding_search(embed(entity_names), task_id, db)
    
    # 3. 合并结果
    return merge_and_format(bm25_hits, embed_hits)
```

### 交互点 2: 实体消重候选检索

**写入时**:
- 目的: 为三层消重提供候选实体
- DB 操作: SELECT entities, entity_trees
- 时机: 实体消重开始时

```python
# 伪代码
def retrieve_candidate_pool(entity_names, task_id, db, embed_client):
    candidates = []
    
    for name in entity_names:
        # 1. 精确查询
        exact = db.query(Entity).filter(
            Entity.task_id == task_id,
            Entity.canonical_ref == name
        ).all()
        candidates.extend(exact)
        
        # 2. 向量查询
        vector_hits = db.query(Entity).filter(
            Entity.task_id == task_id
        ).order_by(
            Entity.desc_embedding.cosine_distance(embed(name))
        ).limit(10).all()
        candidates.extend(vector_hits)
    
    return deduplicate(candidates)
```

### 交互点 3: 搜索时的多路径检索

**搜索时**:
- 目的: 从不同角度检索相关事实
- DB 操作: 6 条独立路径的查询
- 时机: 查询扩展后

| 路径 | DB 操作 | 索引 |
|------|---------|------|
| A: BM25 | `SELECT ... ORDER BY ts_rank` | BM25 索引 |
| B: 向量 | `SELECT ... ORDER BY <->` | 向量索引 |
| B2: HyDE | `SELECT ... ORDER BY <->` | 向量索引 |
| B3: 事件 | `SELECT ... WHERE text LIKE` | 全文索引 |
| C: 树 | `SELECT * FROM entity_trees` | JSON 索引 |
| D: Tantivy | 全文搜索引擎 | 倒排索引 |

### 交互点 4: 实体引用解析

**搜索时**:
- 目的: 将事实中的实体引用解析为完整的实体信息
- DB 操作: SELECT entities, fact_refs
- 时机: 结果融合后

```python
# 伪代码
def resolve_entity_references(facts, task_id, db):
    # 1. 收集所有唯一的实体 ID
    all_entity_ids = collect_unique_entity_ids(facts)
    
    # 2. 批量查询实体信息
    entities = db.query(Entity).filter(
        Entity.entity_id.in_(all_entity_ids)
    ).all()
    
    # 3. 构建映射
    entity_map = {e.entity_id: e for e in entities}
    
    # 4. 替换引用
    for fact in facts:
        fact.entity_ref = entity_map[fact.entity_id].canonical_ref
        # "[Caroline]" → "Caroline (user's mother)"
    
    return facts
```

## 数据一致性保证

### 1. 事务保证

```python
# 写入过程使用事务
with self._transactions.write() as db:
    # 所有 DB 操作都在一个事务中
    create_entities(...)
    insert_facts(...)
    create_fact_refs(...)
    update_entity_trees(...)
    # 如果任何操作失败，全部回滚
```

### 2. 锁定机制

```python
# 实体树更新时锁定
with self._store.lock_entities(task_id, touched_entity_ids) as db:
    # 加载当前状态
    state = load_state(...)
    # 计算更新
    result = compute_updates(...)
    # 应用更新
    apply_updates(...)
    db.commit()
    # 提交后自动释放锁
```

### 3. 异步索引更新

```python
# 写入后异步更新索引
def after_persist(batch_id):
    # 异步更新 BM25 索引
    update_bm25_index(batch_id)
    
    # 异步更新向量索引
    update_vector_index(batch_id)
```

## 性能优化策略

### 1. 批量操作

```python
# 批量插入而非逐条插入
db.bulk_insert(FactModel, fact_records)  # 一次插入多条
```

### 2. 索引优化

```python
# 关键索引
Index("idx_facts_task_id", "task_id")
Index("idx_facts_batch_id", "batch_id")
Index("idx_facts_embedding", "text_embedding")  # 向量索引
Index("idx_fact_refs_entity", "entity_id")
Index("idx_entities_canonical", "task_id", "canonical_ref")
```

### 3. 缓存策略

```python
# 实体解析索引在批处理期间缓存
indexes = build_resolver_indexes(entries, ...)  # 构建一次
for decision in decisions:
    result = layer1_exact(decision.ref, indexes)  # 多次使用
```

## 总结

MemBrain 的数据生成和搜索是一个紧密交互的过程：

| 阶段 | 写入操作 | 搜索操作 | 交互点 |
|------|----------|----------|--------|
| 入口 | 消息解析 | 问题解析 | - |
| 扩展 | - | 查询扩展 | - |
| 提取 | 实体提取 | - | 加载实体上下文 |
| 生成 | 事实生成 | - | 实体覆盖验证 |
| 消重 | 实体消重 | - | 检索候选实体 |
| 检索 | - | 多路径检索 | DB 多路径查询 |
| 融合 | - | 结果融合 | - |
| 解析 | - | 实体引用解析 | 批量实体查询 |
| 持久化 | 数据库写入 | - | INSERT/UPDATE |
| 树更新 | 实体树更新 | - | SELECT entity_trees |
| 返回 | - | 上下文打包 | - |

**核心交互原则**:
1. **写入为搜索准备**: 写入时生成的数据结构（实体树、向量）直接服务于搜索
2. **搜索触发写入**: 搜索时加载的上下文可以用于触发新的写入
3. **一致性保证**: 事务和锁定机制确保数据一致性
4. **性能优化**: 批量操作、索引优化、缓存策略提升效率