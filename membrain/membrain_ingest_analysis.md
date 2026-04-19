# MemBrain 写入过程详细分析

## 概述

MemBrain 的写入（摄取/ingestion）过程是将新的对话消息转化为结构化记忆的核心流程。与搜索过程类似，写入过程也是一个复杂的多阶段管道，涉及 LLM 调用、实体解析、事实生成和数据库持久化。

整个写入过程可以概括为以下主要阶段：

```
用户消息 → 消息预处理 → 实体提取 → 事实生成 → 实体消重 → 数据库持久化 → 实体树更新
```

## 核心架构

### 1. 入口：BatchIngester

**文件**: [batch_ingestor.py](file:///home/project/MemBrain/membrain/memory/application/batch_ingestor.py)

```python
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
        
        batch_id = str(uuid.uuid7())
        batch_index = self._batch_counter
        self._batch_counter += 1

        # 分离上下文消息和待提取消息
        context_messages = messages[:context_size] if context_size else []
        extract_messages = messages[context_size:]
        
        # 格式化文本
        context_text = format_lines(context_messages) if context_messages else ""
        messages_text = format_lines(extract_messages)

        # 执行工作流
        return await self._run_batch(...)
```

**关键参数**:
- `task_pk`: 任务标识符
- `messages`: 消息列表
- `context_size`: 上下文消息数量（用于提供背景信息）
- `session_number`: 会话编号
- `profile`: 配置档案（如 "personamemv2"）

### 2. 工作流选择

**文件**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py)

```python
_WORKFLOW_REGISTRY: dict[str | None, type[IngestWorkflow]] = {
    None: DefaultIngestWorkflow,
    "personamemv2": PersonaMemIngestWorkflow,
}

def get_workflow(profile: str | None, ...) -> IngestWorkflow:
    cls = _WORKFLOW_REGISTRY.get(profile, DefaultIngestWorkflow)
    return cls(ingest_store, tree_updater, embed_client, registry, factory, profile)
```

**工作流类型**:

| 工作流 | 说明 | 适用场景 |
|--------|------|----------|
| `DefaultIngestWorkflow` | 完整的 4 阶段管道 | 大多数场景 |
| `PersonaMemIngestWorkflow` | 简化版本，跳过实体提取 | PersonaMem v2 |

## 详细阶段分析

### Stage 1: 实体提取 (Entity Extraction)

**文件**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py#L158-L211)

```python
async def _extract_entities(
    self,
    messages_text: str,
    context_text: str,
    task_pk: int,
) -> list[str]:
    """Stage 1: Entity extraction with two-pass refinement."""
    
    # 第一轮：基础实体提取
    entity_extractor, extractor_settings = self._factory.get_agent("entity-extractor")
    prompts = self._registry.render_prompts(
        "entity-extractor",
        context_messages=context_text,
        messages_json=messages_text,
        entity_context="",  # 初始无上下文
    )
    result = await run_agent_with_retry(entity_extractor, instructions=prompts, ...)
    entity_names = result.output.entities
    
    # 第二轮：基于已有实体进行上下文增强提取
    entity_context = self._ingest_store.load_extraction_context(
        entity_names=entity_names,
        task_id=task_pk,
        embed_client=self._embed_client,
    )
    
    if entity_context:
        known_entities = build_known_entities_text(entity_context)
        # 再次调用提取器，传入已知实体作为上下文
        prompts = self._registry.render_prompts(
            "entity-extractor",
            entity_context=known_entities,
            ...
        )
        result = await run_agent_with_retry(...)
        entity_names = result.output.entities
    
    return entity_names
```

**两轮提取策略**:

1. **第一轮**: 从消息中提取初始实体列表
2. **第二轮**: 基于已有实体作为上下文，进行更精确的提取

**为什么需要两轮**:
- 第一轮可能遗漏一些实体
- 第二轮利用已提取的实体信息，提高召回率和准确性

### Stage 2: 事实生成 (Fact Generation)

**文件**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py#L213-L275)

```python
async def _generate_facts(
    self,
    entity_names: list[str],
    messages_text: str,
    context_text: str,
) -> tuple[list[str], list[dict]]:
    """Stage 2: Fact generation with entity coverage validation."""
    
    fact_generator, generator_settings = self._factory.get_agent("fact-generator")
    
    # 注册实体覆盖验证器
    _register_entity_coverage_validator(fact_generator)
    
    prompts = self._registry.render_prompts(
        "fact-generator",
        entity_list_json=entity_list_json,
        context_messages=context_text,
        messages_json=messages_text,
    )
    
    # 允许的实体引用集合
    allowed_refs = set(entity_names)
    
    try:
        # 带实体约束调用事实生成器
        result = await run_agent_with_retry(
            fact_generator,
            instructions=prompts,
            deps={"allowed_entity_refs": allowed_refs},
        )
        facts = [f.model_dump() for f in result.output.facts]
    except Exception:
        # 如果失败，回退到无约束模式
        result = await run_agent_with_retry(
            fact_generator,
            instructions=prompts,
            deps={"allowed_entity_refs": set()},
        )
        # 过滤掉包含非法引用的事实
        entity_names, facts = _apply_fact_generator_fallback(facts, allowed_refs)
    
    return entity_names, facts
```

**实体覆盖验证器**:

```python
def _register_entity_coverage_validator(agent: Agent) -> None:
    @agent.output_validator
    async def validate_entity_coverage(ctx: RunContext[dict], result) -> object:
        allowed_refs: set[str] = ctx.deps.get("allowed_entity_refs", set())
        
        # 检查事实中的所有引用是否在允许列表中
        fact_refs: set[str] = set()
        for fact in result.facts:
            fact_refs.update(_ENTITY_BRACKET_RE.findall(fact.text))
        
        illegal = fact_refs - allowed_refs
        if illegal:
            raise ModelRetry(
                f"这些括号引用不在实体列表中: {sorted(illegal)}"
            )
        return result
```

### Stage 3: 实体消重与规范化 (Entity Resolution)

**文件**: [entity_resolver.py](file:///home/project/MemBrain/membrain/memory/core/entity_resolver.py)

这是 MemBrain 最核心的实体解析模块，采用三层消重策略：

#### Layer 1: 精确匹配

```python
def layer1_exact(new_ref: str, indexes: ResolverIndexes) -> ResolverDecision | str | None:
    """Return ResolverDecision on match, LAYER1_AMBIGUOUS on multi-entity hit, None on miss."""
    norm = _normalize(new_ref)  # 小写 + 空白折叠
    matches = indexes.normalized_map.get(norm, [])
    
    if not matches:
        return None  # 数据库中无匹配
    
    unique_eids = {e.entity_id for e in matches}
    if len(unique_eids) > 1:
        return LAYER1_AMBIGUOUS  # 多个实体匹配同一名称
    
    return ResolverDecision(
        new_entity_ref=new_ref,
        action="merge",
        target_entity_id=next(iter(unique_eids)),
        resolved_via="exact",
    )
```

#### Layer 2: MinHash + Jaccard 模糊匹配

```python
def layer2_minhash(new_ref: str, indexes: ResolverIndexes) -> ResolverDecision | None:
    fuzzy = _normalize_fuzzy(new_ref)  # 只保留字母数字
    
    # 熵过滤：太简单的名称跳过
    if not _has_high_entropy(fuzzy):
        return None
    
    # 生成 3-gram shingles
    new_shingles = _shingles(fuzzy)
    
    # MinHash 签名
    sig = _minhash_signature(new_shingles)
    
    # LSH 桶查找候选
    candidate_eids: set[str] = set()
    for band_idx, band in enumerate(_lsh_bands(sig)):
        candidate_eids.update(indexes.lsh_buckets.get((band_idx, band), []))
    
    # Jaccard 相似度计算
    best_eid = None
    best_score = 0.0
    for entry in indexes.entries:
        if entry.entity_id not in candidate_eids:
            continue
        score = _jaccard(new_shingles, entry.shingles)
        if score > best_score:
            best_score = score
            best_eid = entry.entity_id
    
    if best_eid is not None and best_score >= settings.RESOLVER_JACCARD_THRESHOLD:
        return ResolverDecision(
            new_entity_ref=new_ref,
            action="merge",
            target_entity_id=best_eid,
            resolved_via="minhash",
        )
    return None
```

#### Layer 3: LLM 语义匹配

```python
async def layer3_llm(
    unresolved_refs: list[str],
    unresolved_descs: dict[str, str],
    indexes: ResolverIndexes,
    registry,
    factory,
    profile: str | None = None,
) -> list[ResolverDecision]:
    """Send unresolved new entities + deduplicated candidates to LLM."""
    
    # 准备新实体和已有实体的上下文
    new_entities_ctx = [
        {"id": i, "ref": ref, "desc": unresolved_descs.get(ref, "")}
        for i, ref in enumerate(unresolved_refs)
    ]
    
    # 调用 LLM 进行语义匹配
    agent, agent_settings = factory.get_agent("entity-resolver", profile=profile)
    prompts = registry.render_prompts(
        "entity-resolver",
        new_entities_json=new_json,
        existing_entities_json=existing_json,
    )
    result = await run_agent_with_retry(agent, instructions=prompts, ...)
    
    # 解析 LLM 返回的决策
    decisions = []
    for res in result.output.resolutions:
        if res.matched_entity_id == -1:
            decisions.append(ResolverDecision(new_entity_ref=ref, action="keep"))
        else:
            decisions.append(ResolverDecision(
                new_entity_ref=ref,
                action="merge",
                target_entity_id=target_eid,
                resolved_via="llm",
            ))
    
    return decisions
```

**三层消重流程图**:

```
新实体引用
    │
    ▼
┌─────────────────────────────────┐
│ Layer 1: 精确匹配               │
│ (小写 + 空白折叠后匹配)        │
└──────────────┬──────────────────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
   匹配                无匹配
     │                   │
     ▼                   ▼
┌─────────┐    ┌─────────────────────────────────┐
│ 合并到   │    │ Layer 2: MinHash + Jaccard    │
│ 已有实体 │    │ (3-gram + LSH + 相似度)        │
└─────────┘    └──────────────┬──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                  匹配                无匹配
                    │                   │
                    ▼                   ▼
            ┌───────────┐    ┌─────────────────────────────────┐
            │ 合并到    │    │ Layer 3: LLM 语义匹配           │
            │ 已有实体 │    │ (发送到 LLM 判断是否相同实体)   │
            └───────────┘    └──────────────┬──────────────────┘
                                           │
                                 ┌─────────┴─────────┐
                                 ▼                   ▼
                               匹配                  无匹配
                                 │                   │
                                 ▼                   ▼
                         ┌───────────┐      ┌─────────────┐
                         │ 合并到   │      │ 创建新实体  │
                         │ 已有实体 │      │            │
                         └───────────┘      └─────────────┘
```

### Stage 4: 持久化 (Persistence)

**文件**: [memory_ingest_store.py](file:///home/project/MemBrain/membrain/infra/persistence/memory_ingest_store.py#L188-L213)

```python
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
    with self._transactions.write() as db:
        # 构建引用映射
        ref_to_entity_id = entity_queries.build_ref_map(db, task_id)
        
        # 写入批次结果
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

**写入内容**:

1. **实体 (Entities)**: 新的或更新的实体记录
2. **事实 (Facts)**: 从消息中提取的具体事实
3. **事实引用 (Fact Refs)**: 事实中引用的实体映射
4. **嵌入向量 (Embeddings)**: 事实和实体的向量表示

### Stage 5: 实体树更新 (Entity Tree Update)

**文件**: [entity_tree_updater.py](file:///home/project/MemBrain/membrain/memory/application/entity_tree_updater.py)

```python
async def update(
    self,
    task_id: int,
    batch_id: str,
    embed_client,
    registry,
    factory,
) -> list[str]:
    """Update entity tree after batch persistence."""
    
    # 1. 为新实体创建树节点
    # 2. 将新事实挂载到叶子节点
    # 3. 审计和重组树结构
    # 4. 传播更新到父节点
```

## 完整流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户消息输入                                    │
│              [{role: "user", content: "..."},                       │
│               {role: "assistant", content: "..."}]                 │
└─────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BatchIngester.ingest_batch                        │
│  • 生成 batch_id                                                      │
│  • 分离上下文消息和提取消息                                           │
│  • 格式化消息文本                                                   │
└─────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    IngestWorkflow.run_batch                          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Stage 1: 实体提取 (Entity Extraction)                     │   │
│  │  • 第一轮：基础提取                                        │   │
│  │  • 加载已有实体上下文                                       │   │
│  │  • 第二轮：上下文增强提取                                  │   │
│  │ 输出: entity_names (实体名称列表)                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Stage 2: 事实生成 (Fact Generation)                       │   │
│  │  • 调用 fact-generator agent                               │   │
│  │  • 实体覆盖验证                                           │   │
│  │  • 回退处理                                               │   │
│  │ 输出: facts (事实列表), entity_names                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Stage 3: 实体消重 (Entity Resolution)                     │   │
│  │  • Layer 1: 精确匹配 (小写+空白折叠)                      │   │
│  │  • Layer 2: MinHash + Jaccard 模糊匹配                    │   │
│  │  • Layer 3: LLM 语义匹配                                  │   │
│  │  • 实体规范化 (可选)                                      │   │
│  │ 输出: decisions (包含 merge/create 动作)                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Stage 4: 持久化 (Persistence)                             │   │
│  │  • 写入实体表                                             │   │
│  │  • 写入事实表                                             │   │
│  │  • 写入事实引用表                                         │   │
│  │  • 生成嵌入向量                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Stage 5: 实体树更新 (Entity Tree Update)                 │   │
│  │  • 创建树节点                                             │   │
│  │  • 挂载事实到叶子                                         │   │
│  │  • 审计重组                                               │   │
│  │  • 向上传播                                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         返回 BatchResult                              │
│  • batch_index, batch_id                                            │
│  • entities (实体列表)                                              │
│  • facts (事实列表)                                                 │
│  • decisions (决策列表)                                             │
│  • profiled_entities (已更新树结构的实体)                           │
└─────────────────────────────────────────────────────────────────────┘
```

## 配置参数

**文件**: [config.py](file:///home/project/MemBrain/membrain/config.py)

```python
# 实体解析配置
RESOLVER_CANDIDATE_TOP_K = 10
RESOLVER_JACCARD_THRESHOLD = 0.9      # Jaccard 相似度阈值
RESOLVER_ENTROPY_THRESHOLD = 1.5      # 熵阈值
RESOLVER_MIN_NAME_LENGTH = 6           # 最小名称长度
RESOLVER_MIN_TOKEN_COUNT = 2          # 最少 token 数
RESOLVER_MINHASH_PERMUTATIONS = 32    # MinHash 排列数
RESOLVER_MINHASH_BAND_SIZE = 4        # LSH 桶大小
RESOLVER_LLM_ENABLED = True           # 是否启用 LLM 消重

# 提取上下文配置
EXTRACTION_CONTEXT_TOP_K = 20          # 上下文实体数量
EXTRACTION_CONTEXT_PER_QUERY = 5       # 每查询限制

# 实体规范化
CANONICALIZER_ENABLED = True
```

## PersonaMem v2 特殊处理

**文件**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py#L443-L493)

```python
class PersonaMemIngestWorkflow(IngestWorkflow):
    """PersonaMem workflow: skip entity extraction and LLM resolution."""
    
    async def run_batch(self, ...):
        # Stage 1: 固定实体（跳过提取）
        entity_names = ["User"]  # 只有一个实体：用户自己
        
        # Stage 2: 事实生成（相同）
        
        # Stage 3: 简化消重（只做精确查找，无 LLM）
        # 查找 User 实体是否存在
        user_ctx = self._ingest_store.load_extraction_context(
            entity_names=["User"], task_id=task_pk, embed_client=embed_client
        )
        
        if user_ctx:
            decisions[0]["action"] = "merge"
            decisions[0]["target_entity_id"] = user_ctx.entity_id
        
        # Stage 4: 持久化（相同）
```

**PersonaMem v2 特点**:
- 只追踪用户（"User"）相关的记忆
- 跳过实体提取步骤
- 简化实体消重（只需查找 User 实体）
- 适合单用户对话场景

## 数据模型

### Entity 表

```python
class EntityModel:
    entity_id: str       # UUID
    task_id: int
    canonical_ref: str   # 规范名称，如 "Caroline"
    desc: str           # 描述
    desc_embedding: Vector  # 描述嵌入向量
```

### Fact 表

```python
class FactModel:
    id: int
    task_id: int
    batch_id: str       # 批次 ID
    session_number: int # 会话编号
    text: str           # 事实文本
    text_embedding: Vector  # 文本嵌入向量
    status: str         # 'active' | 'archived'
    fact_ts: Timestamp  # 时间戳
```

### FactRef 表

```python
class FactRefModel:
    id: int
    fact_id: int        # 关联事实
    entity_id: str      # 关联实体
    alias_text: str     # 引用文本
```

## 总结

MemBrain 的写入过程是一个高度自动化的多阶段管道：

| 阶段 | 操作 | 关键技术 |
|------|------|----------|
| 1 | 实体提取 | LLM (entity-extractor) + 两轮迭代 |
| 2 | 事实生成 | LLM (fact-generator) + 实体覆盖验证 |
| 3 | 实体消重 | 精确匹配 + MinHash + LLM 三层策略 |
| 4 | 数据库持久化 | 事务 + 批量写入 + 向量生成 |
| 5 | 实体树更新 | 层级结构 + 审计 + 传播 |

**设计亮点**:

1. **两轮实体提取**: 利用已有上下文提高准确性
2. **三层实体消重**: 精确 → 模糊 → LLM，平衡效果和成本
3. **实体覆盖验证**: 确保生成的事实只引用已提取的实体
4. **回退机制**: 验证失败时自动回退到更宽松的模式
5. **Profile 支持**: 支持不同场景的不同工作流