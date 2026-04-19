# Stage 1: 实体提取 - Entity Extraction

## 概述

实体提取是 MemBrain 写入过程的第一阶段，负责从用户对话消息中识别和提取出需要记忆的实体（如人物、地点、组织等）。这是整个记忆系统的起点，提取的实体将用于后续的事实生成和实体消重阶段。

MemBrain 采用**两轮提取策略**来提高实体提取的准确性和召回率。

## 代码位置

- **主入口**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py#L158-L211)
- **实体构建**: [message_text.py](file:///home/project/MemBrain/membrain/memory/application/message_text.py)
- **Agent 配置**: [factory.py](file:///home/project/MemBrain/membrain/agents/factory.py)

## 详细代码分析

### 1.1 入口函数

```python
async def _extract_entities(
    self,
    messages_text: str,
    context_text: str,
    task_pk: int,
) -> list[str]:
    """Stage 1: Entity extraction with two-pass refinement."""
    
    # ═══════════════════════════════════════════════════════════════
    # 第一轮：基础实体提取
    # ═══════════════════════════════════════════════════════════════
    
    entity_extractor, extractor_settings = self._factory.get_agent(
        "entity-extractor",
        profile=self._profile,
    )
    
    # 构建提示词
    prompts = self._registry.render_prompts(
        "entity-extractor",
        profile=self._profile,
        context_messages=context_text,      # 上下文消息
        messages_json=messages_text,        # 待提取的消息
        entity_context="",                   # 初始无上下文
    )
    
    # 调用 LLM 提取实体
    result = await run_agent_with_retry(
        entity_extractor,
        instructions=prompts,
        model_settings=extractor_settings,
    )
    
    # 获取第一轮提取的实体列表
    entity_names = result.output.entities
    log.debug("entity-extractor pass-1 -> %d entities", len(entity_names))

    # ═══════════════════════════════════════════════════════════════
    # 第二轮：基于上下文的增强提取
    # ═══════════════════════════════════════════════════════════════
    
    # 加载已有实体的上下文信息
    entity_context = self._ingest_store.load_extraction_context(
        entity_names=entity_names,
        task_id=task_pk,
        embed_client=self._embed_client,
    )
    
    if entity_context:
        # 构建已知实体文本
        known_entities = build_known_entities_text(entity_context)
        
        # 再次调用提取器，传入已有实体作为上下文
        prompts = self._registry.render_prompts(
            "entity-extractor",
            profile=self._profile,
            context_messages=context_text,
            messages_json=messages_text,
            entity_context=known_entities,  # 注入已知实体
        )
        
        result = await run_agent_with_retry(
            entity_extractor,
            instructions=prompts,
            model_settings=extractor_settings,
        )
        
        entity_names = result.output.entities
        log.debug(
            "entity-extractor pass-2 -> %d entities (ctx=%d)",
            len(entity_names),
            len(entity_context),
        )
    else:
        log.debug("entity-extractor pass-2 skipped (no candidates)")

    return entity_names
```

### 1.2 上下文加载

```python
# membrain/infra/persistence/memory_ingest_store.py

def load_extraction_context(
    self,
    entity_names: list[str],
    task_id: int,
    embed_client,
) -> list[EntityContext]:
    """加载已有实体的上下文信息，用于增强提取。"""
    with self._transactions.read() as db:
        return _retrieve_entity_context_for_extraction(
            entity_names=entity_names,
            task_id=task_id,
            db=db,
            embed_client=embed_client,
        )
```

### 1.3 上下文检索实现

```python
# membrain/infra/persistence/memory_ingest_store.py

def _retrieve_entity_context_for_extraction(
    entity_names: list[str],
    task_id: int,
    db,
    embed_client,
    top_k: int | None = None,
    per_query_limit: int | None = None,
) -> list[EntityContext]:
    if top_k is None:
        top_k = settings.EXTRACTION_CONTEXT_TOP_K  # 默认 20
    if per_query_limit is None:
        per_query_limit = settings.EXTRACTION_CONTEXT_PER_QUERY  # 默认 5

    # ═══════════════════════════════════════════════════════════════
    # Step 1: BM25 搜索候选实体
    # ═══════════════════════════════════════════════════════════════
    per_query_eids: list[list[str]] = []
    for name in entity_names:
        hits = _bm25_search(name, task_id, db, limit=per_query_limit * 3)
        seen: set[str] = set()
        eids: list[str] = []
        for _, eid in hits:
            if eid not in seen:
                seen.add(eid)
                eids.append(eid)
        per_query_eids.append(eids[:per_query_limit])

    # ═══════════════════════════════════════════════════════════════
    # Step 2: 向量搜索增强
    # ═══════════════════════════════════════════════════════════════
    try:
        vecs = embed_client.embed(entity_names)
    except Exception:
        log.warning("Embedding failed for extraction context, BM25-only", exc_info=True)
        vecs = []

    for i, vec in enumerate(vecs):
        rows = _embedding_search(vec, task_id, db, limit=per_query_limit)
        embed_eids = [row[0] for row in rows]
        if i < len(per_query_eids):
            existing = set(per_query_eids[i])
            for eid in embed_eids:
                if eid not in existing:
                    existing.add(eid)
                    per_query_eids[i].append(eid)
            per_query_eids[i] = per_query_eids[i][:per_query_limit]
        else:
            per_query_eids.append(embed_eids[:per_query_limit])

    # ═══════════════════════════════════════════════════════════════
    # Step 3: 交叉合并候选
    # ═══════════════════════════════════════════════════════════════
    selected_eids = _interleave_candidates(per_query_eids, top_k)
    if not selected_eids:
        return []

    # ═══════════════════════════════════════════════════════════════
    # Step 4: 构建实体上下文
    # ═══════════════════════════════════════════════════════════════
    by_eid = entity_queries.find_merge_targets(db, task_id, selected_eids)
    aliases_map = _fetch_aliases_by_entity(db, set(selected_eids))

    return [
        EntityContext(
            entity_id=eid,
            canonical_ref=by_eid[eid].canonical_ref,
            aliases=[
                alias
                for alias in aliases_map.get(eid, [])
                if alias != by_eid[eid].canonical_ref
            ],
            desc=by_eid[eid].desc or "",
        )
        for eid in selected_eids
        if eid in by_eid
    ]
```

### 1.4 实体上下文格式化

```python
# membrain/memory/application/message_text.py

def build_known_entities_text(entity_context: list[EntityContext]) -> str:
    """将实体上下文格式化为提示词文本。"""
    if not entity_context:
        return ""
    
    lines = ["## Known Entities from Memory"]
    for ctx in entity_context:
        parts = [f"- **{ctx.canonical_ref}**"]
        if ctx.aliases:
            parts.append(f"(also: {', '.join(ctx.aliases)})")
        if ctx.desc:
            parts.append(f": {ctx.desc}")
        lines.append(" ".join(parts))
    
    return "\n".join(lines)
```

## 两轮提取策略详解

### 第一轮：基础提取

```
输入:
  - messages_text: "Caroline met her sister Lisa at the coffee shop yesterday."
  - context_text: ""  (无上下文)
  - entity_context: "" (无已有实体)

处理:
  1. 调用 entity-extractor agent
  2. Agent 分析消息，识别实体
  3. 输出实体列表

输出:
  entity_names = ["Caroline", "Lisa"]
```

### 第二轮：上下文增强提取

```
输入:
  - messages_text: "Caroline met her sister Lisa at the coffee shop yesterday."
  - context_text: ""  (无上下文)
  - entity_context: [
      EntityContext(
          entity_id="uuid-1",
          canonical_ref="Caroline",
          aliases=["Carol", "C"],
          desc="User's mother, lives in Boston"
      ),
      EntityContext(
          entity_id="uuid-2", 
          canonical_ref="Lisa",
          aliases=["Lis"],
          desc="Caroline's sister"
      )
    ]

处理:
  1. 构建已知实体文本
     "## Known Entities from Memory
      - **Caroline** (also: Carol, C): User's mother, lives in Boston
      - **Lisa** (also: Lis): Caroline's sister"
  
  2. 再次调用 entity-extractor agent，传入已知实体
  3. Agent 利用上下文信息进行更准确的提取

输出:
  entity_names = ["Caroline", "Lisa", "coffee shop"]
  (可能发现第一轮遗漏的 "coffee shop")
```

## 数据结构

### EntityContext

```python
@dataclass
class EntityContext:
    entity_id: str           # 实体 UUID
    canonical_ref: str       # 规范名称
    aliases: list[str]      # 别名列表
    desc: str              # 实体描述
```

### 示例数据

```python
# 第一轮提取结果
pass1_entities = ["Caroline", "Lisa"]

# 加载的上下文
entity_context = [
    EntityContext(
        entity_id="550e8400-e29b-41d4-a716-446655440000",
        canonical_ref="Caroline",
        aliases=["Carol", "C"],
        desc="User's mother, met at family gathering last summer"
    ),
    EntityContext(
        entity_id="550e8400-e29b-41d4-a716-446655440001",
        canonical_ref="Lisa",
        aliases=["Lis", "L"],
        desc="Caroline's younger sister"
    ),
]

# 第二轮提取结果
pass2_entities = ["Caroline", "Lisa", "coffee shop", "Boston"]
```

## 配置参数

```python
# membrain/config.py

# 提取上下文配置
EXTRACTION_CONTEXT_TOP_K = 20          # 最多加载的实体上下文数量
EXTRACTION_CONTEXT_PER_QUERY = 5       # 每个查询返回的候选数量

# Agent 重试配置
AGENT_MAX_RETRIES = 3
AGENT_RETRY_DELAY = 1.0  # seconds
```

## 为什么需要两轮提取

### 1. 上下文增强

第一轮提取可能因为缺乏上下文而遗漏实体。通过加载已有实体的信息，第二轮可以：
- 识别更具体的实体类型
- 发现与已知实体相关的新实体
- 避免重复提取已有实体

### 2. 实体消重的准备

第一轮提取后，系统会查找数据库中已有的相似实体，为第二轮提供：
- 实体别名信息
- 实体描述信息
- 帮助区分同名实体

### 3. 渐进式精确

```
第一轮: "广撒网" → 尽可能多地提取实体
第二轮: "精筛选" → 利用上下文信息精确化
```

## 完整示例

### 输入消息

```python
messages = [
    {"role": "user", "content": "I had lunch with my friend John at the Italian restaurant downtown yesterday."},
    {"role": "assistant", "content": "That sounds nice! Which Italian restaurant did you go to?"},
    {"role": "user", "content": "It was Luigi's Pizza. John said it's his favorite place."}
]
```

### 处理流程

```
Step 1: 消息格式化
messages_text = """
User: I had lunch with my friend John at the Italian restaurant downtown yesterday.
Assistant: That sounds nice! Which Italian restaurant did you go to?
User: It was Luigi's Pizza. John said it's his favorite place.
"""

Step 2: 第一轮提取
  调用 entity-extractor
  
  输入提示词:
  """
  Extract all named entities from the following messages.
  
  Messages:
  [messages_text]
  """
  
  输出:
  entity_names = ["John", "Italian restaurant", "downtown", "Luigi's Pizza"]

Step 3: 加载实体上下文
  搜索数据库中这些实体的已有信息
  
  假设已有:
  - John: "User's colleague at tech company"
  - Luigi's Pizza: "Local pizza place, 123 Main St"

Step 4: 第二轮提取
  调用 entity-extractor (带上下文)
  
  输入提示词:
  """
  Extract all named entities from the following messages.
  
  Known Entities from Memory:
  - **John**: User's colleague at tech company
  - **Luigi's Pizza**: Local pizza place, 123 Main St
  
  Messages:
  [messages_text]
  """
  
  输出:
  entity_names = ["John", "Luigi's Pizza", "downtown", "yesterday"]
  
  (可能过滤掉过于笼统的 "Italian restaurant")

Step 5: 返回最终实体列表
final_entities = ["John", "Luigi's Pizza", "downtown", "yesterday"]
```

### 实体类型分类

提取的实体可能包括：

| 类型 | 示例 | 用途 |
|------|------|------|
| 人物 | John, Lisa, Caroline | 社交关系 |
| 地点 | Luigi's Pizza, downtown, Boston | 位置信息 |
| 组织 | Google, Tech Corp | 工作/兴趣 |
| 时间 | yesterday, last week | 时间线 |
| 事件 | birthday party, meeting | 活动记忆 |
| 物品 | coffee, book | 物品记忆 |

## 错误处理

### 1. LLM 调用失败

```python
try:
    result = await run_agent_with_retry(...)
    entity_names = result.output.entities
except Exception as e:
    log.warning("entity-extractor failed: %s", e)
    entity_names = []  # 返回空列表，后续阶段可处理
```

### 2. 上下文加载失败

```python
entity_context = self._ingest_store.load_extraction_context(...)
# 如果失败或无结果，第二轮会被跳过
if entity_context:
    # 执行第二轮
    ...
else:
    log.debug("entity-extractor pass-2 skipped (no candidates)")
```

### 3. 空实体列表

```python
if not entity_names:
    log.info("No entities extracted from messages")
    return []  # 继续后续流程，但无实体可处理
```

## 总结

实体提取阶段的核心逻辑：

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 第一轮 LLM 提取 | 基础实体识别 |
| 2 | 加载已有实体上下文 | 为增强提取准备 |
| 3 | 第二轮 LLM 提取 | 上下文增强的精确提取 |
| 4 | 返回实体列表 | 供后续阶段使用 |

**设计亮点**:

1. **两轮策略**: 平衡召回率和精确度
2. **上下文感知**: 利用已有记忆增强提取
3. **BM25 + 向量混合检索**: 确保上下文加载的准确性
4. **渐进式精确**: 先广后精的策略
5. **错误容忍**: 失败时优雅降级

这一阶段为后续的事实生成提供了实体基础，是整个记忆系统的关键起点。