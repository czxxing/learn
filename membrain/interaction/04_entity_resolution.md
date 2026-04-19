# Stage 4: 实体消重与规范化

## 概述

这一阶段负责将提取的实体与数据库中已有的实体进行匹配和消重，确保不创建重复的实体记录。MemBrain 采用三层消重策略：
1. 精确匹配
2. MinHash + Jaccard 模糊匹配
3. LLM 语义匹配

## 代码位置

- **实体解析**: [entity_resolver.py](file:///home/project/MemBrain/membrain/memory/core/entity_resolver.py)
- **工作流入口**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py#L277-L350)
- **数据库查询**: [entities.py](file:///home/project/MemBrain/membrain/infra/queries/entities.py)

## 详细代码分析

### 4.1 消重工作流入口

```python
# ingest_workflow.py

async def _resolve_entities(
    self,
    entities: list[dict],
    facts: list[dict],
    task_pk: int,
    batch_id: str,
) -> list[dict]:
    """Stage 3: Entity resolution and canonicalization."""
    
    # 步骤 1: 从提取的实体构建议息决策
    decisions = _build_decisions_from_extraction(entities)
    
    # 步骤 2: 执行实体解析
    resolved = await self._ingest_store.resolve_entity_decisions(
        task_id=task_pk,
        decisions=decisions,
        embed_client=self._embed_client,
        registry=self._registry,
        factory=self._factory,
        profile=self._profile,
    )

    # 步骤 3: 记录日志
    merge_count = sum(1 for d in resolved.decisions if d["action"] == "merge")
    create_count = len(resolved.decisions) - merge_count
    log.info(
        "    [resolve] %d decisions (merge=%d, create=%d)",
        len(resolved.decisions), merge_count, create_count
    )

    # 步骤 4: 实体规范化（可选）
    if settings.CANONICALIZER_ENABLED:
        canon_candidates = [
            d for d in resolved.decisions if d["action"] in ("merge", "create")
        ]
        if canon_candidates:
            canon_map = await _batch_canonicalize(
                canon_candidates, self._registry, self._factory, profile=self._profile
            )
            for index, decision in enumerate(canon_candidates):
                if index in canon_map:
                    decision["canonical_ref"] = canon_map[index]["canonical_ref"]
                    decision["updated_desc"] = canon_map[index]["merged_desc"]

    return resolved.decisions
```

### 4.2 构建决策

```python
# ingest_workflow.py

def _build_decisions_from_extraction(entities: list[dict]) -> list[dict]:
    """从提取的实体构建议息决策。"""
    return [
        {
            "batch_ref": entity["ref"],      # 批处理中的引用
            "action": "create",               # 默认创建
            "target_ref": None,
            "canonical_ref": entity["ref"],   # 规范引用
            "updated_desc": entity.get("desc", ""),
        }
        for entity in entities
    ]
```

### 4.3 三层消重策略

#### Layer 1: 精确匹配

```python
def _normalize(ref: str) -> str:
    """小写 + 空白折叠"""
    return re.sub(r"\s+", " ", ref.lower()).strip()


def layer1_exact(new_ref: str, indexes: ResolverIndexes) -> ResolverDecision | str | None:
    """精确匹配：标准化后查找"""
    
    norm = _normalize(new_ref)
    matches = indexes.normalized_map.get(norm, [])
    
    if not matches:
        return None  # 无匹配
    
    unique_eids = {e.entity_id for e in matches}
    if len(unique_eids) > 1:
        return LAYER1_AMBIGUOUS  # 多个匹配
    
    return ResolverDecision(
        new_entity_ref=new_ref,
        action="merge",
        target_entity_id=next(iter(unique_eids)),
        resolved_via="exact",
    )
```

#### Layer 2: MinHash + Jaccard 模糊匹配

```python
def _normalize_fuzzy(ref: str) -> str:
    """只保留字母数字"""
    cleaned = re.sub(r"[^a-z0-9' ]", " ", _normalize(ref))
    return re.sub(r"\s+", " ", cleaned).strip()


def _shingles(fuzzy: str) -> set[str]:
    """生成 3-gram shingles"""
    s = fuzzy.replace(" ", "")
    if len(s) < 3:
        return {s}
    return {s[i:i+3] for i in range(len(s) - 2)}


def layer2_minhash(new_ref: str, indexes: ResolverIndexes) -> ResolverDecision | None:
    """MinHash + Jaccard 模糊匹配"""
    
    fuzzy = _normalize_fuzzy(new_ref)
    if not _has_high_entropy(fuzzy):
        return None  # 熵太低，跳过
    
    new_shingles = _shingles(fuzzy)
    sig = _minhash_signature(new_shingles)
    
    # LSH 查找候选
    candidate_eids = set()
    for band_idx, band in enumerate(_lsh_bands(sig)):
        candidate_eids.update(indexes.lsh_buckets.get((band_idx, band), []))
    
    # Jaccard 相似度
    best_eid, best_score = None, 0.0
    for entry in indexes.entries:
        if entry.entity_id not in candidate_eids:
            continue
        score = _jaccard(new_shingles, entry.shingles)
        if score > best_score:
            best_score, best_eid = score, entry.entity_id
    
    if best_eid and best_score >= settings.RESOLVER_JACCARD_THRESHOLD:
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
    """LLM 语义匹配"""
    
    if not unresolved_refs or not settings.RESOLVER_LLM_ENABLED:
        return [ResolverDecision(new_entity_ref=r, action="keep") for r in unresolved_refs]
    
    # 构建上下文
    new_entities_ctx = [
        {"id": i, "ref": ref, "desc": unresolved_descs.get(ref, "")}
        for i, ref in enumerate(unresolved_refs)
    ]
    
    # 去重已有实体
    seen_eids = set()
    deduped_candidates = []
    eid_by_candidate_id = {}
    
    for entry in indexes.entries:
        if entry.entity_id in seen_eids:
            continue
        seen_eids.add(entry.entity_id)
        cid = len(deduped_candidates)
        ent_model = indexes.by_entity_id.get(entry.entity_id)
        deduped_candidates.append({
            "id": cid,
            "canonical_ref": ent_model.canonical_ref if ent_model else entry.name,
            "aliases": indexes.aliases_by_entity.get(entry.entity_id, []),
            "desc": ent_model.desc if ent_model else "",
        })
        eid_by_candidate_id[cid] = entry.entity_id
    
    # 调用 LLM
    agent, agent_settings = factory.get_agent("entity-resolver")
    prompts = registry.render_prompts(
        "entity-resolver",
        new_entities_json=json.dumps(new_entities_ctx),
        existing_entities_json=json.dumps(deduped_candidates),
    )
    
    result = await run_agent_with_retry(agent, instructions=prompts, ...)
    
    # 解析决策
    decisions = []
    for res in result.output.resolutions:
        if res.matched_entity_id == -1:
            decisions.append(ResolverDecision(new_entity_ref=ref, action="keep"))
        else:
            target_eid = eid_by_candidate_id[res.matched_entity_id]
            decisions.append(ResolverDecision(
                new_entity_ref=ref, action="merge", target_entity_id=target_eid, resolved_via="llm"
            ))
    
    return decisions
```

### 4.4 实体规范化（可选）

```python
async def _batch_canonicalize(
    merge_candidates: list[dict],
    registry: TaskRegistry,
    factory: AgentFactory,
    profile: str | None = None,
) -> dict[int, dict]:
    """使用 LLM 规范化实体名称和描述"""
    
    entities_input = [
        {
            "idx": index,
            "old_canonical_name": decision.get("old_canonical_ref", ""),
            "all_aliases": decision.get("all_aliases", [decision["batch_ref"]]),
            "old_description": decision.get("old_description", ""),
            "new_facts": decision.get("updated_desc", ""),
        }
        for index, decision in enumerate(merge_candidates)
    ]
    
    agent, agent_settings = factory.get_agent("entity-canonicalizer")
    prompts = registry.render_prompts(
        "entity-canonicalizer",
        profile=profile,
        entities_json=json.dumps(entities_input),
    )
    
    result = await run_agent_with_retry(agent, instructions=prompts, ...)
    
    return {item.idx: {"canonical_ref": item.canonical_ref, "merged_desc": item.merged_desc}
            for item in result.output.results}
```

## 消重流程图

```
输入: 新提取的实体
    entities = [{"ref": "John"}, {"ref": "Caroline"}, {"ref": "Carol"}]
    
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 构建决策                                                        │
│ decisions = [                                                    │
│   {"batch_ref": "John", "action": "create", ...},              │
│   {"batch_ref": "Caroline", "action": "create", ...},         │
│   {"batch_ref": "Carol", "action": "create", ...}             │
│ ]                                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 检索候选实体池                                                  │
│ • BM25 搜索                                                    │
│ • 向量搜索                                                     │
│ • 构建索引                                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 遍历每个决策                                                    │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │ 实体: "John"                                           │     │
│   │   │                                                   │     │
│   │   ▼                                                   │     │
│   │ Layer 1: 精确匹配                                      │     │
│   │   normalize("John") → "john"                          │     │
│   │   查找 normalized_map["john"]                         │     │
│   │   → 匹配到 entity_id="uuid-1"                       │     │
│   │   → action="merge", target="uuid-1", via="exact"    │     │
│   └─────────────────────────────────────────────────────────┘     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │ 实体: "Caroline"                                        │     │
│   │   │                                                   │     │
│   │   ▼                                                   │     │
│   │ Layer 1: 精确匹配                                      │     │
│   │   → 无匹配 → Layer 2                                   │     │
│   │ Layer 2: MinHash + Jaccard                            │     │
│   │   → 无匹配 (相似度<0.9) → Layer 3                    │     │
│   │ Layer 3: LLM                                          │     │
│   │   → 无匹配 → action="create"                         │     │
│   └─────────────────────────────────────────────────────────┘     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │ 实体: "Carol"                                           │     │
│   │   │                                                   │     │
│   │   ▼                                                   │     │
│   │ Layer 1: 精确匹配                                      │     │
│   │   → 无匹配 → Layer 2                                  │     │
│   │ Layer 2: MinHash + Jaccard                            │     │
│   │   → 匹配 Caroline (Jaccard=0.95)                     │     │
│   │   → action="merge", target="uuid-2", via="minhash"  │     │
│   └─────────────────────────────────────────────────────────┘     │
│                                                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 实体规范化 (可选)                                               │
│ • 调用 entity-canonicalizer agent                              │
│ • 规范化名称和描述                                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 输出决策                                                        │
│ decisions = [                                                    │
│   {"batch_ref": "John", "action": "merge", "target_entity_id": "uuid-1", "resolved_via": "exact"},     │
│   {"batch_ref": "Caroline", "action": "create"},              │
│   {"batch_ref": "Carol", "action": "merge", "target_entity_id": "uuid-2", "resolved_via": "minhash"} │
│ ]                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 输入示例

```python
# 输入：提取的实体
entities = [
    {"ref": "John", "desc": "User's colleague"},
    {"ref": "Caroline", "desc": "User's mother"},
    {"ref": "Carol", "desc": "Met at family gathering"}
]

# 数据库已有实体
existing_entities = {
    "uuid-1": {"canonical_ref": "John", "aliases": ["Johnny"]},
    "uuid-2": {"canonical_ref": "Caroline", "aliases": ["Carol", "C"]}
}

# 消重结果
decisions = [
    {
        "batch_ref": "John",
        "action": "merge",
        "target_entity_id": "uuid-1",
        "canonical_ref": "John",
        "resolved_via": "exact"
    },
    {
        "batch_ref": "Caroline",
        "action": "create",
        "target_entity_id": None,
        "canonical_ref": "Caroline",
        "resolved_via": None
    },
    {
        "batch_ref": "Carol",
        "action": "merge",
        "target_entity_id": "uuid-2",  # 合并到 Caroline
        "canonical_ref": "Caroline",    # 使用规范名称
        "resolved_via": "minhash"
    }
]
```

## 与后续阶段的关联

```
Stage 4: 实体消重与规范化
    │
    └──→ decisions: 决策列表
              │
              ├──→ action="merge": 更新已有实体
              │
              └──→ action="create": 创建新实体
                        │
                        └──→ Stage 5: 数据库持久化
```

## 关键设计决策

### 1. 三层递进策略

```
Layer 1: 精确匹配 (O(1))
  - 最快，最准确
  - 处理大多数情况

Layer 2: 模糊匹配 (O(k))
  - 处理拼写变体、别名
  - 使用 MinHash + LSH 优化

Layer 3: LLM 语义 (O(n) LLM)
  - 最后兜底
  - 处理复杂消重场景
```

### 2. 熵过滤

```python
def _has_high_entropy(fuzzy: str) -> bool:
    """跳过信息量低的字符串"""
    # 过滤: "aaa", "123", "ab" 等
```

### 3. 规范化时机

- 消重后进行规范化
- 合并新描述到已有实体

## 总结

这一阶段的核心功能：

| 功能 | 说明 |
|------|------|
| 构建决策 | 从提取的实体构建议息决策 |
| 精确匹配 | 小写+空白折叠后匹配 |
| 模糊匹配 | MinHash + Jaccard 相似度 |
| LLM 匹配 | 语义相似度判断 |
| 规范化 | 统一实体名称和描述 |

**关键输出**:
- `decisions`: 包含 merge/create 动作的决策列表
- `resolved_via`: 消重方式 (exact/minhash/llm)