# Stage 3: 实体消重 - Entity Resolution

## 概述

实体消重是 MemBrain 写入过程的核心阶段之一，负责将提取的实体与数据库中已有的实体进行匹配和消重。这个阶段采用**三层消重策略**，从精确匹配到模糊匹配再到 LLM 语义匹配，层层递进，确保实体被正确地合并到已有的记忆网络中。

这是 MemBrain 最复杂也最关键的模块，直接影响记忆的准确性和一致性。

## 代码位置

- **实体解析器**: [entity_resolver.py](file:///home/project/MemBrain/membrain/memory/core/entity_resolver.py)
- **解析决策**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py#L277-L350)
- **数据库查询**: [entities.py](file:///home/project/MemBrain/membrain/infra/queries/entities.py)

## 三层消重策略概览

```
新实体引用
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 1: 精确匹配 (Exact Match)                        │
│ • 小写 + 空白折叠后匹配                               │
│ • O(1) 时间复杂度                                     │
└──────────────────────┬──────────────────────────────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
           匹配                无匹配
             │                   │
             ▼                   ▼
        ┌─────────┐    ┌──────────────────────────────────┐
        │ 合并    │    │ Layer 2: 模糊匹配 (Fuzzy Match)  │
        │ 到已有 │    │ • MinHash + Jaccard 相似度       │
        └─────────┘    │ • LSH 桶快速查找                 │
                       └──────────────┬───────────────────┘
                                      │
                            ┌─────────┴─────────┐
                            ▼                   ▼
                          匹配                无匹配
                            │                   │
                            ▼                   ▼
                    ┌───────────┐    ┌──────────────────────────────────┐
                    │ 合并     │    │ Layer 3: LLM 语义匹配            │
                    │ 到已有   │    │ • 发送 LLM 判断是否相同实体       │
                    └───────────┘    │ • 适用于复杂消重场景             │
                                     └──────────────┬───────────────────┘
                                                     │
                                                   ┌─┴───────────────────┐
                                                   ▼                     ▼
                                                 匹配                   无匹配
                                                   │                     │
                                                   ▼                     ▼
                                           ┌───────────┐        ┌─────────────┐
                                           │ 合并     │        │ 创建新实体  │
                                           │ 到已有   │        │            │
                                           └───────────┘        └─────────────┘
```

## 详细代码分析

### 3.1 实体消重入口

```python
# membrain/memory/application/ingest_workflow.py

async def _resolve_entities(
    self,
    entities: list[dict],
    facts: list[dict],
    task_pk: int,
    batch_id: str,
) -> list[dict]:
    """Stage 3: Entity resolution and canonicalization."""
    
    # Step 1: 从提取的实体构建议息决策
    decisions = _build_decisions_from_extraction(entities)
    
    # Step 2: 执行实体解析
    resolved = await self._ingest_store.resolve_entity_decisions(
        task_id=task_pk,
        decisions=decisions,
        embed_client=self._embed_client,
        registry=self._registry,
        factory=self._factory,
        profile=self._profile,
    )

    # Step 3: 记录日志
    merge_count = sum(1 for d in resolved.decisions if d["action"] == "merge")
    create_count = len(resolved.decisions) - merge_count
    log.info(
        "    [resolve] %d decisions (merge=%d, create=%d)",
        len(resolved.decisions),
        merge_count,
        create_count,
    )

    # Step 4: 实体规范化（可选）
    if settings.CANONICALIZER_ENABLED:
        canon_candidates = [
            d for d in resolved.decisions if d["action"] in ("merge", "create")
        ]
        if canon_candidates:
            canon_map = await _batch_canonicalize(...)
            # 更新决策中的规范引用

    return resolved.decisions
```

### 3.2 实体解析入口

```python
# membrain/memory/core/entity_resolver.py

async def resolve_entities(
    decisions: list[dict],
    db,
    task_id: int,
    embed_client,
    registry,
    factory,
    profile: str | None = None,
) -> list[dict]:
    """Run three-layer resolution on create decisions."""
    
    # 筛选出需要创建的决策
    create_decisions = [d for d in decisions if d["action"] == "create"]
    if not create_decisions:
        return decisions
    
    # 获取候选实体池
    entity_names = [d["canonical_ref"] for d in create_decisions]
    entries, by_entity_id, aliases_by_entity = retrieve_candidate_pool(
        entity_names, task_id, db, embed_client
    )
    
    if not entries:
        return decisions
    
    # 构建索引
    indexes = build_resolver_indexes(entries, by_entity_id, aliases_by_entity)
    
    # 三层消重
    resolved = []
    unresolved = []
    ambiguous = []
    
    for decision in create_decisions:
        new_ref = decision["canonical_ref"]
        
        # Layer 1: 精确匹配
        result = layer1_exact(new_ref, indexes)
        
        if result is None:
            # Layer 2: MinHash 模糊匹配
            result = layer2_minhash(new_ref, indexes)
            
            if result is None:
                # 记录为待处理
                unresolved.append((new_ref, decision))
                continue
        
        if result == LAYER1_AMBIGUOUS:
            ambiguous.append((new_ref, decision))
            continue
        
        # 转换为决策格式
        resolved.append({
            "action": result.action,
            "batch_ref": new_ref,
            "target_entity_id": result.target_entity_id,
            ...
        })
    
    # Layer 3: LLM 消重（处理未匹配的）
    if unresolved:
        llm_decisions = await layer3_llm(
            [u[0] for u in unresolved],
            {u[0]: u[1].get("updated_desc", "") for u in unresolved},
            indexes,
            registry,
            factory,
            profile,
        )
        # 合并 LLM 决策
    
    return final_decisions
```

## Layer 1: 精确匹配 (Exact Match)

### 标准化函数

```python
# membrain/memory/core/entity_resolver.py

def _normalize(ref: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", ref.lower()).strip()


def layer1_exact(
    new_ref: str, indexes: ResolverIndexes
) -> ResolverDecision | str | None:
    """Return ResolverDecision on match, LAYER1_AMBIGUOUS on multi-entity hit, None on miss."""
    
    # 标准化新实体引用
    norm = _normalize(new_ref)
    
    # 在标准化映射中查找
    matches = indexes.normalized_map.get(norm, [])
    
    if not matches:
        return None  # 无匹配
    
    # 检查是否匹配多个不同实体
    unique_eids = {e.entity_id for e in matches}
    if len(unique_eids) > 1:
        return LAYER1_AMBIGUOUS  # 多个实体匹配
    
    # 返回合并决策
    return ResolverDecision(
        new_entity_ref=new_ref,
        action="merge",
        target_entity_id=next(iter(unique_eids)),
        resolved_via="exact",
    )
```

### 示例

```
输入: "Caroline"
数据库已有: ["Caroline", "caroline", "CAROLINE"]

处理:
  1. 标准化: "caroline"
  2. 查找 normalized_map["caroline"]
  3. 找到匹配: [Entry(entity_id="uuid-1", name="Caroline"), ...]
  4. 唯一实体: {"uuid-1"}
  5. 返回: merge 到 uuid-1

输入: "John"
数据库: 无匹配

处理:
  1. 标准化: "john"
  2. 查找 normalized_map["john"]
  3. 无匹配: []
  4. 返回: None (进入 Layer 2)
```

## Layer 2: MinHash + Jaccard 模糊匹配

### 2.1 模糊标准化

```python
def _normalize_fuzzy(ref: str) -> str:
    """Produce alphanumeric-only form for n-gram shingles."""
    cleaned = re.sub(r"[^a-z0-9' ]", " ", _normalize(ref))
    return re.sub(r"\s+", " ", cleaned).strip()
```

### 2.2 Shingles 生成

```python
def _shingles(fuzzy: str) -> set[str]:
    """3-gram shingles from space-stripped fuzzy form."""
    s = fuzzy.replace(" ", "")
    if not s:
        return set()
    if len(s) < 3:
        return {s}
    return {s[i : i + 3] for i in range(len(s) - 2)}
```

**示例**:
```
输入: "caroline"
处理:
  1. 去除空格: "caroline"
  2. 生成 3-gram:
     - car, arol, roli, olin, line
     - 集合: {"car", "arol", "roli", "olin", "line"}

输入: "Carol"
处理:
  1. 去除空格: "carol"
  2. 生成 3-gram:
     - car, aro, rol
     - 集合: {"car", "aro", "rol"}
     
相似度: Jaccard({"car","arol","roli","olin","line"}, {"car","aro","rol"})
     = 1 / 5 = 0.2 (重叠 only "car")
```

### 2.3 MinHash 签名

```python
def _hash_shingle(shingle: str, seed: int) -> int:
    digest = blake2b(f"{seed}:{shingle}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def _minhash_signature(shingles: set[str]) -> tuple[int, ...]:
    if not shingles:
        return tuple()
    n = settings.RESOLVER_MINHASH_PERMUTATIONS  # 默认 32
    return tuple(min(_hash_shingle(sh, seed) for sh in shingles) for seed in range(n))
```

### 2.4 LSH 桶构建

```python
def _lsh_bands(sig: tuple[int, ...]) -> list[tuple[int, ...]]:
    band_size = settings.RESOLVER_MINHASH_BAND_SIZE  # 默认 4
    sig_list = list(sig)
    bands = []
    for start in range(0, len(sig_list), band_size):
        band = tuple(sig_list[start : start + band_size])
        if len(band) == band_size:
            bands.append(band)
    return bands
```

**LSH 原理图解**:

```
MinHash 签名: [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12]
              (12 个哈希值)

Band Size = 4:

Band 0: [h1, h2, h3, h4]  → 桶 key: (0, (h1,h2,h3,h4))
Band 1: [h5, h6, h7, h8]  → 桶 key: (1, (h5,h6,h7,h8))
Band 2: [h9, h10, h11, h12] → 桶 key: (2, (h9,h10,h11,h12))

两个实体的签名如果在任意 band 上相同，则可能相似
(大大减少需要比较的对数)
```

### 2.5 模糊匹配函数

```python
def layer2_minhash(new_ref: str, indexes: ResolverIndexes) -> ResolverDecision | None:
    """MinHash + Jaccard fuzzy matching."""
    
    fuzzy = _normalize_fuzzy(new_ref)
    
    # 熵过滤：太简单的名称跳过
    if not _has_high_entropy(fuzzy):
        return None
    
    # 生成 shingles 和签名
    new_shingles = _shingles(fuzzy)
    sig = _minhash_signature(new_shingles)
    
    # LSH 查找候选
    candidate_eids: set[str] = set()
    for band_idx, band in enumerate(_lsh_bands(sig)):
        candidate_eids.update(indexes.lsh_buckets.get((band_idx, band), []))
    
    if not candidate_eids:
        return None
    
    # 计算 Jaccard 相似度
    best_eid = None
    best_score = 0.0
    for entry in indexes.entries:
        if entry.entity_id not in candidate_eids:
            continue
        score = _jaccard(new_shingles, entry.shingles)
        if score > best_score:
            best_score = score
            best_eid = entry.entity_id
    
    # 检查阈值
    if best_eid is not None and best_score >= settings.RESOLVER_JACCARD_THRESHOLD:
        return ResolverDecision(
            new_entity_ref=new_ref,
            action="merge",
            target_entity_id=best_eid,
            resolved_via="minhash",
        )
    return None
```

### 熵过滤

```python
def _has_high_entropy(fuzzy: str) -> bool:
    """Check if the string has sufficient information content."""
    
    # 长度检查
    token_count = len(fuzzy.split())
    if (
        len(fuzzy) < settings.RESOLVER_MIN_NAME_LENGTH  # 默认 6
        and token_count < settings.RESOLVER_MIN_TOKEN_COUNT  # 默认 2
    ):
        return False
    
    # 熵计算
    stripped = fuzzy.replace(" ", "")
    if not stripped:
        return False
    
    counts: dict[str, int] = {}
    for ch in stripped:
        counts[ch] = counts.get(ch, 0) + 1
    
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    
    return entropy >= settings.RESOLVER_ENTROPY_THRESHOLD  # 默认 1.5
```

**熵过滤示例**:
```
"abc": 字符 {'a','b','c'} 各出现 1 次
     熵 = -3 * (1/3) * log2(1/3) ≈ 1.58 → 通过

"aaa": 字符 {'a'} 出现 3 次
     熵 = -1 * 1 * log2(1) = 0 → 过滤

"北京": 中文字符
     需要特殊处理
```

## Layer 3: LLM 语义匹配

### 入口函数

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
    
    if not unresolved_refs or not settings.RESOLVER_LLM_ENABLED:
        return [ResolverDecision(new_entity_ref=r, action="keep") for r in unresolved_refs]
    
    # 构建新实体上下文
    new_entities_ctx = [
        {"id": i, "ref": ref, "desc": unresolved_descs.get(ref, "")}
        for i, ref in enumerate(unresolved_refs)
    ]
    
    # 去重已有实体（按 entity_id）
    seen_eids: set[str] = set()
    deduped_candidates: list[dict] = []
    eid_by_candidate_id: dict[int, str] = {}
    
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
    agent, agent_settings = factory.get_agent("entity-resolver", profile=profile)
    prompts = registry.render_prompts(
        "entity-resolver",
        new_entities_json=json.dumps(new_entities_ctx),
        existing_entities_json=json.dumps(deduped_candidates),
    )
    
    result = await run_agent_with_retry(...)
    
    # 解析决策
    decisions = []
    for res in result.output.resolutions:
        if res.matched_entity_id == -1:
            decisions.append(ResolverDecision(new_entity_ref=ref, action="keep"))
        else:
            target_eid = eid_by_candidate_id[res.matched_entity_id]
            decisions.append(ResolverDecision(
                new_entity_ref=ref,
                action="merge",
                target_entity_id=target_eid,
                resolved_via="llm",
            ))
    
    return decisions
```

### LLM 提示词示例

```json
{
  "new_entities": [
    {"id": 0, "ref": "Cathy", "desc": "met at the party"},
    {"id": 1, "ref": "Catherine", "desc": "college friend"}
  ],
  "existing_entities": [
    {"id": 0, "ref": "Catherine", "aliases": ["Cathy", "Cat"], "desc": "known since 2020"}
  ]
}
```

**LLM 决策示例**:
```json
{
  "resolutions": [
    {"new_entity_id": 0, "matched_entity_id": 0, "reason": "Cathy is an alias of Catherine"},
    {"new_entity_id": 1, "matched_entity_id": 0, "reason": "Cathy and Catherine are the same person"}
  ]
}
```

## 索引构建

```python
@dataclass
class ResolverIndexes:
    entries: list[Any]                      # 候选实体条目
    by_entity_id: dict[str, Any]           # entity_id → EntityModel
    normalized_map: defaultdict[str, list[Any]]  # 标准化名称 → 条目
    lsh_buckets: defaultdict[tuple, list[str]]   # (band_idx, band) → [entity_id]
    aliases_by_entity: dict[str, list[str]]      # entity_id → 别名列表


def build_resolver_indexes(
    entries: list[Any],
    by_entity_id: dict[str, Any],
    aliases_by_entity: dict[str, list[str]],
) -> ResolverIndexes:
    normalized_map: defaultdict[str, list[Any]] = defaultdict(list)
    lsh_buckets: defaultdict[tuple, list[str]] = defaultdict(list)

    for entry in entries:
        # Layer 1 索引
        norm = _normalize(entry.name)
        normalized_map[norm].append(entry)

        # Layer 2 索引
        sig = _minhash_signature(entry.shingles)
        for band_idx, band in enumerate(_lsh_bands(sig)):
            lsh_buckets[(band_idx, band)].append(entry.entity_id)

    return ResolverIndexes(
        entries=entries,
        by_entity_id=by_entity_id,
        normalized_map=normalized_map,
        lsh_buckets=lsh_buckets,
        aliases_by_entity=aliases_by_entity,
    )
```

## 完整处理流程示例

### 输入

```python
# 从 Stage 2 生成的实体
entities = [
    {"ref": "Caroline", "desc": "user's mother"},
    {"ref": "Carol", "desc": "met at family gathering"},
    {"ref": "John", "desc": "colleague at work"},
]

# 已有实体数据库
existing_entities = [
    {"entity_id": "uuid-1", "canonical_ref": "Caroline", "aliases": ["Carol", "C"]},
    {"entity_id": "uuid-2", "canonical_ref": "John", "aliases": ["Johnny"]},
]
```

### 消重过程

```
Step 1: 构建决策
decisions = [
    {"batch_ref": "Caroline", "action": "create", "canonical_ref": "Caroline"},
    {"batch_ref": "Carol", "action": "create", "canonical_ref": "Carol"},
    {"batch_ref": "John", "action": "create", "canonical_ref": "John"},
]

Step 2: 消重 "Caroline"
  Layer 1: 精确匹配
    - normalize("Caroline") = "caroline"
    - normalized_map["caroline"] = [Entry(uuid-1, "Caroline")]
    - 匹配: uuid-1
  结果: merge to uuid-1 (resolved_via="exact")

Step 3: 消重 "Carol"
  Layer 1: 精确匹配
    - normalize("Carol") = "carol"
    - normalized_map["carol"] = []
    - 无匹配 → Layer 2
  
  Layer 2: MinHash + Jaccard
    - fuzzy: "carol"
    - shingles: {"car", "aro", "rol"}
    - LSH 查找 → 候选: [uuid-1]
    - Jaccard: 1/5 = 0.2 < 0.9 → 无匹配 → Layer 3
  
  Layer 3: LLM
    - 发送: new="Carol", existing=[{ref: "Caroline", aliases: ["Carol", "C"]}]
    - LLM 判断: "Carol" 是 "Caroline" 的别名
  结果: merge to uuid-1 (resolved_via="llm")

Step 4: 消重 "John"
  Layer 1: 精确匹配
    - normalize("John") = "john"
    - normalized_map["john"] = [Entry(uuid-2, "John")]
    - 匹配: uuid-2
  结果: merge to uuid-2 (resolved_via="exact")

Step 5: 最终决策
decisions = [
    {"action": "merge", "batch_ref": "Caroline", "target_entity_id": "uuid-1"},
    {"action": "merge", "batch_ref": "Carol", "target_entity_id": "uuid-1"},
    {"action": "merge", "batch_ref": "John", "target_entity_id": "uuid-2"},
]
```

## 配置参数

```python
# membrain/config.py

# 精确匹配配置
RESOLVER_MIN_NAME_LENGTH = 6         # 最小名称长度
RESOLVER_MIN_TOKEN_COUNT = 2         # 最少 token 数

# MinHash 配置
RESOLVER_MINHASH_PERMUTATIONS = 32   # MinHash 签名数量
RESOLVER_MINHASH_BAND_SIZE = 4       # LSH band 大小

# 模糊匹配阈值
RESOLVER_JACCARD_THRESHOLD = 0.9     # Jaccard 相似度阈值
RESOLVER_ENTROPY_THRESHOLD = 1.5     # 熵阈值

# LLM 配置
RESOLVER_LLM_ENABLED = True          # 是否启用 LLM 消重

# 候选检索
RESOLVER_CANDIDATE_TOP_K = 10        # 候选实体数量
```

## 性能优化

### 1. 索引缓存

```python
# 索引在一次批处理中构建并缓存
indexes = build_resolver_indexes(entries, by_entity_id, aliases_by_entity)
# 多个新实体共享同一索引
```

### 2. LSH 减少比较

```
传统方法: O(n²) 比较所有实体对
LSH 方法: O(n) 只比较同一桶内的实体
```

### 3. 早期退出

```
Layer 1 精确匹配 → O(1)
Layer 2 失败 → 跳过 Layer 3（可选）
```

## 总结

实体消重阶段的核心逻辑：

| 层级 | 方法 | 时间复杂度 | 阈值 |
|------|------|------------|------|
| Layer 1 | 精确匹配 | O(1) | 完全相等 |
| Layer 2 | MinHash + Jaccard | O(k) | ≥ 0.9 |
| Layer 3 | LLM 语义 | O(n) LLM调用 | 语义相似 |

**设计亮点**:

1. **三层递进**: 精确 → 模糊 → LLM，平衡效果和成本
2. **MinHash + LSH**: 高效的近似相似度搜索
3. **熵过滤**: 跳过过于简单的名称
4. **LLM 兜底**: 处理复杂消重场景
5. **别名追踪**: 记录和利用别名信息

这一阶段确保了实体被正确地合并到已有的记忆网络中，避免了重复和冲突。