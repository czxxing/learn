# Stage 3: 批量提取与实体事实生成

## 概述

这一阶段是记忆提取的核心，负责将对话消息转化为结构化的实体和事实数据。包括：
1. 消息批次准备
2. 实体提取（两轮策略）
3. 事实生成（带实体覆盖验证）
4. 回退处理

## 代码位置

- **批量提取器**: [batch_ingestor.py](file:///home/project/MemBrain/membrain/memory/application/batch_ingestor.py)
- **提取工作流**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py#L158-L275)
- **消息格式化**: [message_text.py](file:///home/project/MemBrain/membrain/memory/application/message_text.py)

## 详细代码分析

### 3.1 批量提取入口

```python
# batch_ingestor.py

class BatchIngester:
    """Use-case object for a single extraction batch."""

    async def ingest_batch(
        self,
        task_pk: int,
        messages: list[dict],
        context_size: int = 0,
        session_number: int | None = None,
        profile: str | None = None,
    ) -> BatchResult:
        """Ingest a single batch of messages into memory."""

        # 生成批次 ID
        batch_id = str(uuid.uuid7())
        batch_index = self._batch_counter
        self._batch_counter += 1

        # 分离上下文消息和待提取消息
        context_messages = messages[:context_size] if context_size else []
        extract_messages = messages[context_size:]

        # 格式化文本
        context_text = format_lines(context_messages) if context_messages else ""
        messages_text = format_lines(extract_messages)

        # 执行提取工作流
        return await self._run_batch(
            batch_index=batch_index,
            messages_text=messages_text,
            context_text=context_text,
            task_pk=task_pk,
            batch_id=batch_id,
            session_number=session_number,
            profile=profile,
        )
```

### 3.2 消息格式化

```python
# message_text.py

def format_lines(messages: list[dict]) -> str:
    """Format messages as text lines."""
    
    lines = []
    for msg in messages:
        speaker = msg.get("speaker", "unknown")
        content = msg.get("content", "")
        
        # 格式: [speaker]: content
        lines.append(f"[{speaker}]: {content}")
    
    return "\n".join(lines)
```

**格式化示例**:

```python
# 输入消息
messages = [
    {"speaker": "user", "content": "I had lunch with John."},
    {"speaker": "assistant", "content": "That sounds nice!"},
    {"speaker": "user", "content": "We went to Luigi's Pizza."}
]

# 格式化输出
"""
[user]: I had lunch with John.
[assistant]: That sounds nice!
[user]: We went to Luigi's Pizza.
"""
```

### 3.3 两轮实体提取

```python
# ingest_workflow.py - _extract_entities

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
    
    entity_extractor, extractor_settings = self._factory.get_agent("entity-extractor")
    prompts = self._registry.render_prompts(
        "entity-extractor",
        context_messages=context_text,
        messages_json=messages_text,
        entity_context="",  # 初始无上下文
    )
    
    result = await run_agent_with_retry(entity_extractor, instructions=prompts, ...)
    entity_names = result.output.entities
    log.debug("entity-extractor pass-1 -> %d entities", len(entity_names))

    # ═══════════════════════════════════════════════════════════════
    # 第二轮：基于上下文的增强提取
    # ═══════════════════════════════════════════════════════════════
    
    # 加载已有实体上下文
    entity_context = self._ingest_store.load_extraction_context(
        entity_names=entity_names,
        task_id=task_pk,
        embed_client=self._embed_client,
    )
    
    if entity_context:
        known_entities = build_known_entities_text(entity_context)
        prompts = self._registry.render_prompts(
            "entity-extractor",
            context_messages=context_text,
            messages_json=messages_text,
            entity_context=known_entities,
        )
        
        result = await run_agent_with_retry(entity_extractor, instructions=prompts, ...)
        entity_names = result.output.entities
        log.debug("entity-extractor pass-2 -> %d entities", len(entity_names))

    return entity_names
```

### 3.4 实体上下文加载

```python
# memory_ingest_store.py

def _retrieve_entity_context_for_extraction(
    entity_names: list[str],
    task_id: int,
    db,
    embed_client,
    top_k: int = 20,
    per_query_limit: int = 5,
) -> list[EntityContext]:
    """检索已有实体上下文用于增强提取。"""
    
    # Step 1: BM25 搜索候选
    per_query_eids = []
    for name in entity_names:
        hits = _bm25_search(name, task_id, db, limit=per_query_limit * 3)
        eids = [eid for _, eid in hits if eid not in seen]
        per_query_eids.append(eids[:per_query_limit])

    # Step 2: 向量搜索增强
    vecs = embed_client.embed(entity_names)
    for i, vec in enumerate(vecs):
        rows = _embedding_search(vec, task_id, db, limit=per_query_limit)
        embed_eids = [row[0] for row in rows]
        # 合并 BM25 和向量结果
        ...

    # Step 3: 交叉合并
    selected_eids = _interleave_candidates(per_query_eids, top_k)

    # Step 4: 构建实体上下文
    by_eid = entity_queries.find_merge_targets(db, task_id, selected_eids)
    aliases_map = _fetch_aliases_by_entity(db, set(selected_eids))

    return [
        EntityContext(
            entity_id=eid,
            canonical_ref=by_eid[eid].canonical_ref,
            aliases=aliases_map.get(eid, []),
            desc=by_eid[eid].desc or "",
        )
        for eid in selected_eids if eid in by_eid
    ]
```

### 3.5 事实生成

```python
# ingest_workflow.py - _generate_facts

async def _generate_facts(
    self,
    entity_names: list[str],
    messages_text: str,
    context_text: str,
) -> tuple[list[str], list[dict]]:
    """Stage 2: Fact generation with entity coverage validation."""
    
    # 获取 fact-generator agent
    fact_generator, generator_settings = self._factory.get_agent("fact-generator")
    
    # 注册实体覆盖验证器
    _register_entity_coverage_validator(fact_generator)
    
    # 构建提示词
    entity_list_json = json.dumps(entity_names, ensure_ascii=False)
    prompts = self._registry.render_prompts(
        "fact-generator",
        entity_list_json=entity_list_json,
        context_messages=context_text,
        messages_json=messages_text,
    )
    
    allowed_refs = set(entity_names)
    
    # 第一次调用（带约束）
    try:
        result = await run_agent_with_retry(
            fact_generator,
            instructions=prompts,
            model_settings=generator_settings,
            deps={"allowed_entity_refs": allowed_refs},
        )
        facts = [f.model_dump() for f in result.output.facts]
        
    except Exception as exc:
        # 回退：无约束调用 + 过滤
        log.warning("fact-generator failed, retrying without constraint")
        
        result = await run_agent_with_retry(
            fact_generator,
            instructions=prompts,
            model_settings=generator_settings,
            deps={"allowed_entity_refs": set()},
        )
        facts = [f.model_dump() for f in result.output.facts]
        
        # 过滤非法引用
        entity_names, facts = _apply_fact_generator_fallback(facts, allowed_refs)
    
    return entity_names, facts
```

### 3.6 实体覆盖验证器

```python
def _register_entity_coverage_validator(agent: Agent) -> None:
    """确保生成的事实只引用已提取的实体。"""
    
    if id(agent) in _registered:
        return
    _registered.add(id(agent))

    @agent.output_validator
    async def validate_entity_coverage(ctx: RunContext[dict], result) -> object:
        allowed_refs: set[str] = ctx.deps.get("allowed_entity_refs", set())
        if not allowed_refs:
            return result
        
        # 检查事实中的所有引用
        fact_refs: set[str] = set()
        for fact in result.facts:
            fact_refs.update(_ENTITY_BRACKET_RE.findall(fact.text))
        
        illegal = fact_refs - allowed_refs
        if illegal:
            raise ModelRetry(
                f"这些引用不在实体列表中: {sorted(illegal)}"
            )
        
        return result
```

## 处理流程图

```
Stage 2: 会话摘要完成
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 批量提取入口                                                    │
│ BatchIngester.ingest_batch(...)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 消息格式化                                                       │
│ format_lines(messages)                                          │
│ "[user]: message                                               │
│  [assistant]: message"                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: 实体提取 (两轮)                                       │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │ 第一轮: 基础提取                                        │     │
│   │ • 调用 entity-extractor agent                         │     │
│   │ • 输出: entity_names = ["John", "Luigi's Pizza"]    │     │
│   └─────────────────────────────────────────────────────────┘     │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │ 加载已有实体上下文                                      │     │
│   │ • BM25 搜索 + 向量搜索                                │     │
│   │ • 返回 EntityContext 列表                             │     │
│   └─────────────────────────────────────────────────────────┘     │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │ 第二轮: 上下文增强提取                                  │     │
│   │ • 传入已知实体作为上下文                              │     │
│   │ • 输出: entity_names = ["John", "Luigi's Pizza",      │     │
│   │           "favorite restaurant"]                      │     │
│   └─────────────────────────────────────────────────────────┘     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: 事实生成                                               │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │ 构建提示词                                             │     │
│   │ • entity_list_json = ["John", "Luigi's Pizza", ...] │     │
│   │ • messages_json = formatted messages                  │     │
│   └─────────────────────────────────────────────────────────┘     │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │ 调用 fact-generator (带实体约束)                        │     │
│   │ • 验证器检查 [entity] 引用                           │     │
│   │ • 如果验证失败，抛出 ModelRetry                      │     │
│   └─────────────────────────────────────────────────────────┘     │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │ 回退处理 (如果验证失败)                                │     │
│   │ • 无约束重新调用                                      │     │
│   │ • 过滤非法引用                                        │     │
│   └─────────────────────────────────────────────────────────┘     │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │ 输出事实列表                                           │     │
│   │ facts = [                                             │     │
│   │   {"text": "[User] had lunch with [John]...", ...},  │     │
│   │   {"text": "[John] likes [Luigi's Pizza]...", ...}, │     │
│   │ ]                                                     │     │
│   └─────────────────────────────────────────────────────────┘     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 返回 BatchResult                                                │
│ • entities: 实体列表                                           │
│ • facts: 事实列表                                             │
└─────────────────────────────────────────────────────────────────┘
```

## 输入示例

```python
# 输入消息
messages = [
    {"speaker": "user", "content": "I had lunch with my friend John at Luigi's Pizza yesterday."},
    {"speaker": "assistant", "content": "That sounds nice!"},
    {"speaker": "user", "content": "John said it's his favorite restaurant. We've been going there for years."}
]

# 格式化后
messages_text = """
[user]: I had lunch with my friend John at Luigi's Pizza yesterday.
[assistant]: That sounds nice!
[user]: John said it's his favorite restaurant. We've been going there for years.
"""

# 第一轮提取结果
pass1_entities = ["John", "Luigi's Pizza"]

# 加载的上下文（假设数据库中已有 Caroline 实体）
entity_context = [
    EntityContext(
        entity_id="uuid-1",
        canonical_ref="John",
        aliases=["Johnny"],
        desc="User's colleague at work"
    ),
    EntityContext(
        entity_id="uuid-2",
        canonical_ref="Luigi's Pizza",
        aliases=["Luigi"],
        desc="Local Italian restaurant"
    )
]

# 第二轮提取结果
pass2_entities = ["John", "Luigi's Pizza", "favorite restaurant"]

# 生成的事实
facts = [
    {
        "text": "[User] had lunch with [John] at [Luigi's Pizza] [yesterday]",
        "time": "yesterday"
    },
    {
        "text": "[John] said [Luigi's Pizza] is [favorite restaurant]",
        "time": None
    },
    {
        "text": "[User] and [John] have been going to [Luigi's Pizza] for years",
        "time": "years"
    }
]
```

## 与后续阶段的关联

```
Stage 3: 批量提取与实体事实生成
    │
    ├──→ entity_names: 实体名称列表
    │         │
    │         └──→ Stage 4: 实体消重
    │
    └──→ facts: 事实列表
              │
              └──→ Stage 4: 实体消重
                        │
                        └──→ Stage 5: 数据库持久化
```

## 关键设计决策

### 1. 两轮实体提取策略

```
第一轮: "广撒网"
  - 无先验知识，尽可能多提取
  - 可能遗漏或重复

第二轮: "精准定位"
  - 利用已有实体上下文
  - 识别更具体的实体
  - 避免重复提取
```

### 2. 实体覆盖验证

```python
# 确保事实只引用已提取的实体
allowed_refs = set(entity_names)
deps = {"allowed_entity_refs": allowed_refs}
```

**防止问题**:
- LLM 幻觉生成未存在的实体
- 事实引用不一致
- 数据质量问题

### 3. 回退机制

```python
# 第一次失败 → 无约束重新调用 → 过滤
try:
    result = await run_agent_with_retry(..., deps={"allowed_entity_refs": allowed_refs})
except:
    result = await run_agent_with_retry(..., deps={"allowed_entity_refs": set()})
    _, facts = _apply_fact_generator_fallback(facts, allowed_refs)
```

**保证鲁棒性**:
- 验证失败不导致完全失败
- 尽可能保留有效数据

## 总结

这一阶段的核心功能：

| 功能 | 说明 |
|------|------|
| 消息格式化 | 将消息列表转为文本格式 |
| 两轮实体提取 | 基础提取 + 上下文增强 |
| 实体上下文加载 | BM25 + 向量混合检索 |
| 事实生成 | LLM 生成 + 实体覆盖验证 |
| 回退处理 | 验证失败时的降级处理 |

**关键输出**:
- `entity_names`: 提取的实体列表
- `facts`: 生成的事实列表
- 触发下一阶段：实体消重