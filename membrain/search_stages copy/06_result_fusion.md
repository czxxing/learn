# Stage 6: 结果融合 - Result Fusion (RRF / Rerank)

## 概述

结果融合是 MemBrain 搜索过程中的核心步骤之一，负责将多路径检索得到的结果进行排序和筛选。MemBrain 提供了两种融合策略：

1. **RRF (Reciprocal Rank Fusion)**: 互惠排名融合 - 默认策略
2. **Rerank**: 交叉编码器重排 - 可选策略

## 代码位置

- **RRF 实现**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L58-L75)
- **Rerank 实现**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L77-L100)
- **融合入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L480-L487)

## 详细代码分析

### 6.1 RRF（互惠排名融合）- 默认策略

#### 6.1.1 核心思想

RRF 的核心思想很简单：**一个文档在多个排名列表中排名越高，其最终得分越高**。

公式如下：
```
RRF_Score(d) = Σ 1 / (k + rank(d))
```

其中：
- `d` 是文档（事实）
- `rank(d)` 是文档在某个排名列表中的排名（从1开始）
- `k` 是一个常数（通常设为60）
- `Σ` 表示对所有排名列表求和

#### 6.1.2 代码实现

```python
# membrain/retrieval/application/retrieval.py

def _fuse_rrf(
    pool: list[RetrievedFact],
    ranked_lists: list[list[int]],
) -> None:
    """Reciprocal Rank Fusion — assigns rerank_score to each fact in pool."""
    
    _RRF_K = 60  # 常数
    
    # 构建排名映射
    rank_maps: list[dict[int, int]] = []
    for lst in ranked_lists:
        rank_maps.append({fid: i for i, fid in enumerate(lst)})
    
    # 计算每个事实的 RRF 分数
    for fact in pool:
        score = 0.0
        for rm in rank_maps:
            rank = rm.get(fact.fact_id)
            if rank is not None:
                score += 1.0 / (_RRF_K + rank + 1)
        fact.rerank_score = score
```

#### 6.1.3 RRF 公式详解

```python
# 核心公式
score += 1.0 / (_RRF_K + rank + 1)

# 展开:
# rank=0 → score += 1/61 ≈ 0.0164
# rank=1 → score += 1/62 ≈ 0.0161
# rank=2 → score += 1/63 ≈ 0.0159
# rank=10 → score += 1/71 ≈ 0.0141
# rank=50 → score += 1/111 ≈ 0.0090
```

**为什么使用 RRF**:

1. **无参数**: 不需要训练或调参
2. **鲁棒**: 对单一路径的噪声不敏感
3. **简单有效**: 已被证明在多路径检索中效果良好

#### 6.1.4 RRF 计算示例

假设有三条检索路径，返回以下排名：

```
Path A: [fact_1, fact_2, fact_3, fact_4, fact_5]
Path B: [fact_2, fact_1, fact_5, fact_3, fact_4]
Path C: [fact_1, fact_3, fact_2, fact_4, fact_5]
```

计算每个事实的 RRF 分数（k=60）：

```python
rank_maps = [
    {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},  # Path A
    {2: 0, 1: 1, 5: 2, 3: 3, 4: 4},  # Path B
    {1: 0, 3: 1, 2: 2, 4: 3, 5: 4},  # Path C
]

# fact_1: 1/(60+0+1) + 1/(60+1+1) + 1/(60+0+1)
#        = 1/61 + 1/62 + 1/61
#        = 0.01639 + 0.01613 + 0.01639
#        = 0.04891

# fact_2: 1/(60+1+1) + 1/(60+0+1) + 1/(60+2+1)
#        = 1/62 + 1/61 + 1/63
#        = 0.01613 + 0.01639 + 0.01587
#        = 0.04839

# fact_3: 1/(60+2+1) + 1/(60+3+1) + 1/(60+1+1)
#        = 1/63 + 1/63 + 1/62
#        = 0.01587 + 0.01587 + 0.01613
#        = 0.04787

# fact_4: 1/(60+3+1) + 1/(60+4+1) + 1/(60+3+1)
#        = 1/64 + 1/65 + 1/64
#        = 0.01563 + 0.01538 + 0.01563
#        = 0.04664

# fact_5: 1/(60+4+1) + 1/(60+2+1) + 1/(60+4+1)
#        = 1/65 + 1/62 + 1/65
#        = 0.01538 + 0.01613 + 0.01538
#        = 04689
```

**最终排名**: fact_1 > fact_2 > fact_3 > fact_5 > fact_4

### 6.2 Rerank（交叉编码器重排）

#### 6.2.1 核心思想

Rerank 使用交叉编码器（Cross-Encoder）模型对候选事实进行精确的相关性评分。与 RRF 不同，Rerank 需要对每个候选进行前向计算，但可以获得更精确的相关性分数。

#### 6.2.2 代码实现

```python
# membrain/retrieval/application/retrieval.py

def _fuse_rerank(
    question: str,
    pool: list[RetrievedFact],
    aspect_infos: dict[int, AspectInfo],
    top_k: int,
) -> list[RetrievedFact]:
    """Cross-encoder reranking — returns top_k facts sorted by relevance score."""
    
    rerank_client = RerankClient()
    try:
        # 1. 富化每个事实的文本
        enriched = [
            enrich_for_rerank(f.text, aspect_infos.get(f.fact_id), f.time_info)
            for f in pool
        ]
        
        # 2. 使用 rerank 模型排序
        ranked = rerank_client.rerank(question, enriched, top_n=top_k)
        
        # 3. 构建结果
        results = []
        for item in ranked:
            f = pool[item["index"]]
            f.rerank_score = item["relevance_score"]
            results.append(f)
        return results
        
    except Exception:
        log.warning("Rerank failed, falling back to pool order")
        return pool[:top_k]
    finally:
        rerank_client.close()
```

#### 6.2.3 事实富化

```python
# membrain/infra/retrieval/aspect_enrichment.py

def enrich_for_rerank(
    fact_text: str,
    info: AspectInfo | None,
    time_info: str = "",
) -> str:
    """Build enriched passage for reranker input."""
    
    if info is None:
        return f"{fact_text} (date: {time_info})" if time_info else fact_text
    
    # 构建实体标签
    entity_label = (
        f"{info.entity_ref} ({info.entity_desc})"
        if info.entity_desc
        else info.entity_ref
    )
    
    # 构建头部
    header = entity_label
    if info.path:
        header = f"{header} > {info.path}"
    
    # 构建摘要
    summary_part = f": {info.leaf_desc}" if info.leaf_desc else ""
    
    # 富化文本
    enriched = f"[{header}{summary_part}] {fact_text}"
    if time_info:
        enriched += f" (date: {time_info})"
    
    return enriched
```

**富化示例**:

```
原始事实: "Caroline worked at Google as a software engineer"

富化后: "[Caroline (user's mother) > Career > Work history: Software engineer] 
         Caroline worked at Google as a software engineer (date: 2023-01-15)"
```

#### 6.2.4 RerankClient 实现

```python
# 假设的实现（实际在 membrain/infra/clients/rerank.py）

class RerankClient:
    def __init__(self):
        self.client = httpx.Client(
            base_url=settings.RERANK_SERVICE_URL,
            headers={"Authorization": f"Bearer {settings.RERANK_API_KEY}"}
        )
    
    def rerank(self, query: str, documents: list[str], top_n: int):
        resp = self.client.post(
            "/v1/rerank",
            json={
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "model": settings.RERANK_MODEL,
            }
        )
        resp.raise_for_status()
        return resp.json()["results"]
```

### 6.3 融合策略选择

```python
# membrain/retrieval/application/retrieval.py

# ── 5. Fusion ─────────────────────────────────────────────────────────
if strategy == "rerank":
    round1_facts = _fuse_rerank(question, pool, aspect_infos, top_k)
else:
    # 默认使用 RRF
    _fuse_rrf(pool, ranked_lists)
    pool.sort(key=lambda f: f.rerank_score, reverse=True)
    round1_facts = pool[:top_k]
```

**策略选择**:

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| RRF | 快速、无需额外模型调用 | 精度有限 | 大多数场景（默认） |
| Rerank | 更精确的相关性评分 | 慢、需要额外模型 | 需要高精度的场景 |

### 6.4 排序和截断

```python
# RRF 策略：排序并截断
pool.sort(key=lambda f: f.rerank_score, reverse=True)
round1_facts = pool[:top_k]  # 默认 top_k=12
```

**为什么需要截断**:

1. **Token 预算**: 后续需要将事实打包到 token 预算中
2. **质量保证**: 排名较低的事实可能相关性不够
3. **计算效率**: 减少后续处理的数据量

## 完整示例

### 输入

```python
# 候选池
pool = [
    RetrievedFact(fact_id=101, text="Caroline worked at Google", source="bm25"),
    RetrievedFact(fact_id=102, text="Caroline was at Google for 3 years", source="embed"),
    RetrievedFact(fact_id=103, text="Google is in Mountain View", source="bm25"),
    RetrievedFact(fact_id=104, text="Caroline's manager was John", source="tree"),
    RetrievedFact(fact_id=105, text="Caroline left Google in 2023", source="embed"),
]

# 排名列表（用于 RRF）
ranked_lists = [
    [101, 102, 103, 104, 105],  # Path A
    [102, 105, 101, 104, 103],  # Path B
    [101, 104, 102, 103, 105],  # Path C
]

# 策略
strategy = "rrf"  # 或 "rerank"
top_k = 3
```

### RRF 处理过程

```python
# 构建排名映射
rank_maps = [
    {101: 0, 102: 1, 103: 2, 104: 3, 105: 4},
    {102: 0, 105: 1, 101: 2, 104: 3, 103: 4},
    {101: 0, 104: 1, 102: 2, 103: 3, 105: 4},
]

# 计算分数
for fact in pool:
    score = 0.0
    for rm in rank_maps:
        rank = rm.get(fact.fact_id)
        if rank is not None:
            score += 1.0 / (60 + rank + 1)
    fact.rerank_score = score

# 排序
pool.sort(key=lambda f: f.rerank_score, reverse=True)

# 截断
round1_facts = pool[:top_k]
```

**分数计算结果**:

```
fact_101: 1/61 + 1/62 + 1/61 = 0.01639 + 0.01613 + 0.01639 = 0.04891
fact_102: 1/62 + 1/61 + 1/62 = 0.01613 + 0.01639 + 0.01613 = 0.04865
fact_103: 1/63 + 1/65 + 1/64 = 0.01587 + 0.01538 + 0.01563 = 0.04688
fact_104: 1/64 + 1/64 + 1/62 = 0.01563 + 0.01563 + 0.01613 = 0.04739
fact_105: 1/65 + 1/62 + 1/65 = 0.01538 + 0.01613 + 0.01538 = 0.04689
```

**排序结果**:
1. fact_101 (0.04891)
2. fact_102 (0.04865)
3. fact_104 (0.04739)

### 输出

```python
round1_facts = [
    RetrievedFact(
        fact_id=101,
        text="Caroline worked at Google",
        source="bm25",
        rerank_score=0.04891,
        ...
    ),
    RetrievedFact(
        fact_id=102,
        text="Caroline was at Google for 3 years",
        source="embed",
        rerank_score=0.04865,
        ...
    ),
    RetrievedFact(
        fact_id=104,
        text="Caroline's manager was John",
        source="tree",
        rerank_score=0.04739,
        ...
    ),
]
```

## 配置参数

```python
# membrain/config.py

QA_RERANK_TOP_K = 12    # 融合后保留的结果数量
```

## 总结

结果融合阶段的核心目标是对多路径检索的结果进行排序：

| 策略 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| RRF | `Σ 1/(k+rank)` | 快速、鲁棒、无参数 | 精度有限 |
| Rerank | Cross-encoder 评分 | 更精确 | 慢、需要额外模型 |

**关键设计决策**:

1. **RRF 作为默认**: 平衡了效果和性能
2. **k=60**: 经验值，减少排名差异的影响
3. **top_k=12**: 在召回和质量之间取得平衡
4. **富化输入**: 为 Rerank 提供更丰富的上下文信息

这一阶段为后续的 Agentic Round 2（如果启用）和上下文打包提供了排序后的候选集。