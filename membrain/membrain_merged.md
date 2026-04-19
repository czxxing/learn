
# MemBrain 技术文档

## 目录

1. 搜索过程分析
2. 搜索详细分析
3. 写入过程分析
4. 01 api entry
5. 02 query expansion
6. 03 multi path retrieval
7. 04 result merging
8. 05 post processing
9. 06 result fusion
10. 07 agentic round2
11. 08 entity resolution
12. 09 session retrieval
13. 10 context packing
14. 11 return results
15. README
16. 01 entity extraction
17. 02 fact generation
18. 03 entity resolution
19. 04 database persistence
20. 05 entity tree update
21. README
22. 01 message input session create
23. 02 async digest session summary
24. 03 batch extraction entity fact generation
25. 04 entity resolution
26. 05 database persistence
27. 06 entity tree update
28. README
29. search write interaction
30. 01 system architecture
31. 02 api endpoints
32. 03 write data
33. 04 search data
34. 05 complete examples
35. 06 advanced features
36. 07 configuration
37. 08 error handling
38. 09 summary
39. README



## 1. 搜索过程分析

# MemBrain 搜索过程详细分析

## 概述

MemBrain 采用**多路径检索（Multi-Path Retrieval）**策略，通过六条独立的检索路径获取记忆事实，然后使用**融合策略（Fusion Strategy）**将结果合并，最终生成用于回答问题的上下文。

## 核心入口

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L334-L575)

```python
def search(
    question: str,
    task_id: int,
    db: Session,
    embed_client: EmbeddingClient,
    http_client: httpx.Client,
    top_k: int = settings.QA_RERANK_TOP_K,
    strategy: Literal["rrf", "rerank"] = "rrf",
    mode: Literal["direct", "expand", "reflect"] = "expand",
) -> dict
```

## 检索模式（Mode）

MemBrain 支持三种检索模式，通过 `mode` 参数选择：

| 模式 | 路径数量 | 说明 |
|------|---------|------|
| `direct` | 3路径 (A+B+C) | 无 LLM 查询重写，直接使用原始问题 |
| `expand` | 6路径 (A+B+B2+B3+C+D) | **默认模式**，使用 LLM 查询扩展 |
| `reflect` | 6路径 + 第二轮反思 | 在 expand 基础上增加 Agentic Round 2 |

## 六条检索路径

### Path A: BM25 事实搜索（关键词）

**文件**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L44-L70)

使用 BM25 算法对事实进行关键词搜索。

```python
def bm25_search_facts(query, task_id, db, limit=20):
    # 使用 ParadeDB 的 BM25 索引
    sql = """
        SELECT id, text, pdb.score(id) AS score
        FROM facts
        WHERE search_text ||| :query
          AND task_id = :task_id
          AND status = 'active'
        ORDER BY score DESC
        LIMIT :limit
    """
```

**查询处理**: 对原始问题进行清洗，移除特殊字符和非 ASCII 字符。

### Path B: 嵌入向量搜索（原始问题）

**文件**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L73-L95)

使用 embedding 模型将问题转换为向量，然后在向量空间中搜索相似的事实。

```python
def embedding_search_facts(query_vec, task_id, db, limit=20):
    # 使用 pgvector 的余弦相似度搜索
    sql = """
        SELECT id, text,
               -(text_embedding <#> CAST(:vec AS halfvec)) AS score
        FROM facts
        WHERE text_embedding IS NOT NULL
          AND status = 'active'
        ORDER BY text_embedding <#> CAST(:vec AS halfvec)
    """
```

### Path B2: HyDE 声明式查询嵌入

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L398-L405)

使用 **HyDE (Hypothetical Document Embedding)** 技术：让 LLM 生成一个假设的"如果答案存在会是什么样子"的声明式句子，然后对这个声明进行嵌入搜索。

**示例**:
- 问题: "When did Caroline have a picnic?"
- HyDE 查询: "Caroline and her friends had a picnic together."

### Path B3: 事件聚焦查询嵌入

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L407-L418)

针对时间问题（如 "When did X?"），移除时间方面，聚焦于事件本身。

**示例**:
- 问题: "When did Caroline have a picnic?"
- 事件查询: "What did Caroline do with friends outdoors?"

### Path C: 实体树光束搜索

**文件**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L97-L268)

这是 MemBrain 最独特的检索路径：

1. **实体匹配**: 通过 BM25 和 embedding 找到与问题相关的实体
2. **树结构光束搜索**: 在实体树结构中沿路径收集叶子节点的事实

```python
def entity_tree_search(query, query_vec, task_id, db):
    # 1. 匹配实体
    entity_ids = _match_entities(query, query_vec, task_id, db, top_n=5)
    
    # 2. 树结构光束搜索
    for eid in entity_ids:
        fact_ids = _tree_beam_search(eid, task_id, db, query_vec, beam_width=3)
    
    # 3. 按嵌入相似度排序返回事实
    return ranked_facts
```

**光束搜索过程**:
- 从根节点开始
- 逐层向下遍历，根据与问题的相似度保留 top-k 节点
- 收集叶子节点关联的事实

### Path D: BM25 解析查询（ Tantivy 语法）

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L97-L124)

使用 LLM 将问题转换为 **Tantivy 查询语法**，支持 AND/OR 语义。

**示例**:
- 问题: "How many children does Emily have?"
- Tantivy 查询: `+emily (child kid son daughter one two three four five)`

## 查询扩展（Query Expansion）

### 1. 查询重写（Query Rewrite）

**文件**: [query_rewrite.py](file:///home/project/MemBrain/membrain/infra/clients/query_rewrite.py)

将问题转换为 3-6 个关键词短语。

```python
# 系统提示
"Extract 3-6 search keywords from the question."
# 示例
"Q: When did Melanie paint a sunrise? → Melanie paint sunrise"
```

### 2. 多查询生成（Multi-Query Generation）

**文件**: [multi_query.py](file:///home/project/MemBrain/membrain/infra/clients/multi_query.py)

LLM 同时生成三个互补查询：

| 查询类型 | 用途 | 示例 |
|---------|------|------|
| Event-focused | 嵌入搜索，聚焦事件 | "What did Caroline do with friends outdoors?" |
| HyDE declarative | 嵌入搜索，假设性声明 | "Caroline and her friends had a picnic together." |
| BM25 keyword | 关键词搜索 | "Caroline friends picnic" |

### 3. BM25 查询生成

**文件**: [bm25_query_gen.py](file:///home/project/MemBrain/membrain/infra/clients/bm25_query_gen.py)

生成 Tantivy 查询字符串，包含：
- `+term`: 必须匹配（AND）
- `term`: 应该匹配（OR）
- `(term1 term2)`: 分组

## 融合策略（Fusion Strategy）

### 1. RRF（Reciprocal Rank Fusion）- 默认

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L58-L75)

```python
def _fuse_rrf(pool, ranked_lists):
    # RRF 公式: score = Σ 1/(k + rank + 1)
    # k = 60 (常数)
    for fact in pool:
        score = 0.0
        for rm in rank_maps:
            rank = rm.get(fact.fact_id)
            if rank is not None:
                score += 1.0 / (_RRF_K + rank + 1)
        fact.rerank_score = score
```

### 2. Rerank（交叉编码器重排）

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L77-L100)

使用 cross-encoder 模型对候选事实进行精确的相关性评分。

```python
def _fuse_rerank(question, pool, aspect_infos, top_k):
    # 1. 为每个事实构建富化的输入
    enriched = [enrich_for_rerank(f.text, aspect_info, f.time_info) for f in pool]
    
    # 2. 使用 rerank 模型排序
    ranked = rerank_client.rerank(question, enriched, top_n=top_k)
    
    # 3. 返回 top-k 事实
    return results
```

## 后处理（Post-processing）

### 1. 时间注解注入

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L163-L197)

```python
def _inject_time_annotations(pool, db):
    # 从 time_annotations 表获取时间信息
    # 从 facts 表的 fact_ts 字段获取时间戳
```

### 2. 实体引用解析

**文件**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L25-L45)

解析事实中的 `[alias]` 引用，替换为实体规范名称。

```python
def _resolve_entity_refs(text, ref_map):
    # [Caroline] → "Caroline (user's mother)"
    return _ENTITY_BRACKET_RE.sub(_replace, text)
```

### 3. Aspect 路径富化

**文件**: [aspect_enrichment.py](file:///home/project/MemBrain/membrain/infra/retrieval/aspect_enrichment.py)

为每个事实构建实体树的上下文路径。

```python
# 示例输出
# Entity: Caroline (user's mother)
# Path: Career > Work history
# Leaf: Software engineer at Google
```

### 4. Aspect 级别去重

**文件**: [aspect_enrichment.py](file:///home/project/MemBrain/membrain/infra/retrieval/aspect_enrichment.py#L121-L156)

限制每个叶子方面最多 3 个事实，每个中间方面最多 8 个事实。

```python
def aspect_dedup(fact_ids, aspect_infos, max_per_leaf=3, max_per_mid=8):
    # 保持输入顺序的同时，应用 Aspect 级别去重
```

## 会话检索（Session Retrieval）

### 1. BM25 会话搜索

**文件**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L328-L361)

使用 BM25 在 session_summaries 表中搜索相关会话。

### 2. 事实聚合评分

**文件**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L363-L418)

根据检索到的事实所属的会话进行聚合评分。

```python
def aggregate_session_scores(facts, task_id, db):
    # 1. 从事实中获取 session_number
    # 2. 聚合每个会话的事实得分
    # 3. 返回排序后的会话列表
```

### 3. 会话上下文格式化

**文件**: [budget_pack.py](file:///home/project/MemBrain/membrain/retrieval/core/budget_pack.py#L86-L112)

```python
def format_session_section(sessions, max_tokens):
    # 格式: ## Relevant Episodes
    # **Subject**: Content
    # ---
```

## 上下文打包（Context Packing）

**文件**: [budget_pack.py](file:///home/project/MemBrain/membrain/retrieval/core/budget_pack.py#L29-L67)

将检索到的事实打包到 token 预算中。

```python
def budget_pack(facts, max_tokens=2000):
    # 1. 按 rerank_score 排序
    # 2. 贪婪填充预算
    # 3. 按时间顺序输出
    # 格式: ## Additional Facts
    #       - fact text (known from message on DATE)
```

## Agentic Round 2（仅 reflect 模式）

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L220-L268)

在第一轮检索后，让 LLM 分析是否需要更多信息，并生成补充查询。

```python
def _reflect_and_refine(question, facts, http_client):
    # 1. 分析第一轮事实是否足够回答问题
    # 2. 如果不足，生成 1-2 个精炼查询
    # 3. 使用精炼查询再次检索
```

## 配置参数

**文件**: [config.py](file:///home/project/MemBrain/membrain/config.py#L107-L125)

```python
# 检索相关配置
QA_BM25_FACT_TOP_N = 20          # Path A 返回数量
QA_EMBED_FACT_TOP_N = 20         # Path B 返回数量
QA_ENTITY_TOP_N = 5               # Path C 实体数量
QA_TREE_BEAM_WIDTH = 3           # Path C 光束宽度
QA_TREE_FACT_TOP_N = 20          # Path C 事实数量
QA_BUDGET_MAX_TOKENS = 2000      # 事实 token 预算
QA_RERANK_TOP_K = 12             # Rerank 后保留数量
QA_SESSION_BM25_TOP_N = 10       # 会话 BM25 数量
QA_SESSION_FINAL_TOP_N = 5       # 最终会话数量
```

## 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户问题                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1. 查询扩展 (mode ≠ direct)                    │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Query Rewrite  │  │ Multi-Query Gen │  │ BM25 Query Gen  │  │
│  │ 问题→关键词     │  │ 3个互补查询      │  │ Tantivy 查询    │  │
│  └───────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└──────────┼───────────────────┼──────────────────────┼────────────┘
           │                   │                      │
           ▼                   ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      2. 六条检索路径                               │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐  │
│  │Path A │ │Path B │ │Path B2│ │Path B3│ │Path C │ │Path D │  │
│  │ BM25  │ │Embed  │ │ HyDE  │ │Event  │ │ Tree  │ │Parsed │  │
│  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      3. 去重合并为 Pool                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      4. 后处理                                    │
│  • 时间注解注入                                                 │
│  • 实体引用解析                                                 │
│  • Aspect 路径富化                                             │
│  • Aspect 级别去重                                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      5. 融合 (RRF / Rerank)                      │
│  按 rerank_score 排序，取 top-k                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
┌─────────────────────┐      ┌─────────────────────────┐
│    mode=reflect?    │      │   会话检索               │
│    (Agentic R2)    │      │ • BM25 搜索             │
│  • 分析是否足够     │      │ • 事实聚合评分           │
│  • 补充查询检索     │      │ • 上下文格式化           │
└─────────┬───────────┘      └────────────┬────────────┘
          │                              │
          └──────────────┬───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      6. 上下文打包                                 │
│  • 事实按 rerank_score 填充预算                                  │
│  • 按时间顺序输出                                               │
│  • 附加会话章节                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      返回结果                                     │
│  • packed_context: 打包的上下文                                  │
│  • packed_token_count: token 数量                               │
│  • fact_ids: 事实 ID 列表                                        │
│  • facts: 事实详情                                               │
│  • sessions: 会话列表                                            │
└─────────────────────────────────────────────────────────────────┘
```

## 总结

MemBrain 的搜索过程是一个高度优化的多路径检索系统：

1. **多路径检索**: 同时使用 BM25、Embedding、实体树搜索，覆盖不同类型的查询
2. **查询扩展**: 利用 LLM 生成多种查询变体，提高召回率
3. **智能融合**: RRF 或 Rerank 将多路径结果有效合并
4. **结构化上下文**: 通过实体树和 Aspect 路径提供丰富的上下文信息
5. **预算感知**: 在 token 预算限制下最大化信息量


## 2. 搜索详细分析

# MemBrain 搜索过程详细分析

## 概述

MemBrain 的搜索过程是一个复杂的多阶段检索系统，旨在从记忆存储中获取与用户问题最相关的信息。整个流程可以概括为以下几个主要阶段：

```
用户问题 → 查询扩展 → 多路径检索 → 结果融合 → 后处理 → 上下文打包 → 返回结果
```

下面将逐步详细分析每个阶段。

---

## 第一阶段：API 入口

### 1.1 搜索入口点

**文件**: [memory.py](file:///home/project/MemBrain/membrain/api/routes/memory.py#L281-L316)

```python
@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(req: MemorySearchRequest):
    # 1. 解析任务标识
    task_pk = resolve_task(req.dataset, req.task)
    
    # 2. 获取必要的客户端
    sf = search_mgr.get_session_factory()      # 数据库会话工厂
    embed_client = search_mgr.get_embed_client()  # 嵌入客户端
    http_client = search_mgr.get_http_client()   # HTTP 客户端（用于 LLM 调用）
    
    # 3. 设置数据库 schema
    schema = f"task_{int(task_pk)}__{_RUN_TAG}"
    db.execute(sa_text(f"SET LOCAL search_path TO {schema}, public"))
    
    # 4. 调用核心搜索函数
    result = _retrieval.search(
        question=req.question,
        task_id=task_pk,
        db=db,
        embed_client=embed_client,
        http_client=http_client,
        top_k=req.top_k or settings.QA_RERANK_TOP_K,
        strategy=req.strategy,
        mode=req.mode,
    )
```

### 1.2 为什么这样设计

- **Schema 隔离**: 每个任务有独立的 database schema，确保数据隔离
- **客户端复用**: 使用共享的 EmbeddingClient 和 HTTPClient，避免重复创建开销
- **参数化**: 支持 `top_k`、`strategy`、`mode` 参数，允许调用方灵活控制

---

## 第二阶段：查询扩展（Query Expansion）

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L348-L370)

### 2.1 直接模式（mode="direct"）

```python
if mode == "direct":
    rewritten = question
    q_event = q_hyde = q_bm25_kw = q_bm25_parsed = ""
```

**原因**: 直接模式跳过所有 LLM 查询扩展，仅使用原始问题进行检索。这适用于：
- 低延迟场景
- 问题本身已经很简洁明确
- 不想调用额外 LLM 的情况

### 2.2 扩展模式（mode="expand" 或 "reflect"）

当 mode 不是 "direct" 时，执行以下查询扩展：

#### 2.2.1 查询重写（Query Rewrite）

**文件**: [query_rewrite.py](file:///home/project/MemBrain/membrain/infra/clients/query_rewrite.py)

```python
rewritten = rewrite_query(question, http_client)
```

**重写示例**:
- 输入: "When did Melanie paint a sunrise?"
- 输出: "Melanie paint sunrise"

**为什么需要查询重写**:
1. **去除冗余词**: 移除 "When did" 等疑问词，减少噪声
2. **提取关键词**: 保留核心实体和动词，提高检索命中率
3. **标准化**: 统一使用词根形式（如 "painting" → "paint"）

**系统提示**:
```
Extract 3-6 search keywords from the question. 
Keep proper nouns exactly as written. 
Use base/infinitive verb forms (e.g. 'research' not 'researching'). 
Remove question words (what/when/did/who/how/is/are). 
Output only the keywords, space-separated, no punctuation.
```

#### 2.2.2 多查询生成（Multi-Query Generation）

**文件**: [multi_query.py](file:///home/project/MemBrain/membrain/infra/clients/multi_query.py)

```python
extra_queries = generate_multi_queries(question, http_client)
q_event = extra_queries[0]  # 事件聚焦查询
q_hyde = extra_queries[1]   # HyDE 声明式查询
q_bm25_kw = extra_queries[2]  # BM25 关键词查询
```

**为什么需要多查询**:

不同的查询形式针对不同类型的检索方法优化：

| 查询类型 | 目的 | 示例 |
|---------|------|------|
| **事件聚焦** | 嵌入搜索专用 | "What did Caroline do with friends outdoors?" |
| **HyDE 声明式** | 嵌入搜索专用 | "Caroline and her friends had a picnic together." |
| **BM25 关键词** | 关键词搜索专用 | "Caroline friends picnic" |

**原因分析**:

1. **时间问题的特殊性**: 
   - 问题 "When did X happen?" 中的时间词会干扰嵌入匹配
   - 事件聚焦查询移除时间，保留核心事件

2. **HyDE 技术**:
   - 假设如果答案存在，它可能以什么形式出现在记忆记录中
   - 让 LLM 生成一个"声明式句子"，这个句子更接近记忆的表述方式

3. **关键词提取**:
   - 进一步的简化，保留最核心的名词和动词
   - 更适合 BM25 这种基于词频的算法

#### 2.2.3 BM25 查询生成

**文件**: [bm25_query_gen.py](file:///home/project/MemBrain/membrain/infra/clients/bm25_query_gen.py)

```python
q_bm25_parsed = generate_bm25_query(question, http_client)
```

**生成的查询示例**:
- 输入: "How many children does Emily have?"
- 输出: `+emily (child kid son daughter one two three four five)`

**为什么需要专门的 BM25 查询**:

1. **利用 BM25 特性**:
   - `+term`: 必须匹配（AND 语义）
   - `(term1 term2)`: 可选匹配（OR 语义）
   
2. **同义词扩展**:
   - "child" 可能以 "kid", "son", "daughter" 形式出现
   - 同时包含这些词提高召回率

3. **词根匹配**:
   - 使用词根形式（"child" 而不是 "children"）匹配词干化的索引

### 2.3 向量嵌入

```python
# 原始问题向量
orig_vec = embed_client.embed_single(question)

# HyDE 向量
hyde_vec = embed_client.embed_single(q_hyde) if q_hyde else orig_vec

# 重写后向量
rewrite_vec = embed_client.embed_single(rewritten) if rewritten != question else orig_vec
```

**为什么需要多个向量**:

1. **orig_vec**: 用于 Path B（原始问题嵌入搜索）
2. **hyde_vec**: 用于 Path B2（HyDE 嵌入搜索）
3. **rewrite_vec**: 用于 Path C（实体树搜索）

每条路径使用不同的查询向量，以获得最佳的检索效果。

---

## 第三阶段：多路径检索

### 3.1 Path A: BM25 关键词搜索

**文件**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L44-L70)

```python
bm25_query = q_bm25_kw or rewritten
path_a = bm25_search_facts(bm25_query, task_id, db, limit=20)
```

**SQL 查询**:
```sql
SELECT id, text, pdb.score(id) AS score
FROM facts
WHERE search_text ||| :query
  AND task_id = :task_id
  AND status = 'active'
ORDER BY score DESC
LIMIT 20
```

**为什么使用 BM25**:

1. **成熟可靠**: BM25 是信息检索领域最经典的算法
2. **关键词匹配**: 擅长精确匹配实体名称、时间等结构化信息
3. **高效**: 基于倒排索引，查询速度快

**补充搜索**:
```python
if rewritten != bm25_query:
    extra_bm25 = bm25_search_facts(rewritten, task_id, db, limit=10)
    # 合并结果，去除重复
```

**原因**: 如果重写后的关键词与 BM25 关键词不同，额外搜索一次以提高召回率。

### 3.2 Path B: 嵌入向量搜索（原始问题）

**文件**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L73-L95)

```python
path_b = embedding_search_facts(orig_vec, task_id, db) if orig_vec else []
```

**SQL 查询**:
```sql
SELECT id, text,
       -(text_embedding <#> CAST(:vec AS halfvec)) AS score
FROM facts
WHERE text_embedding IS NOT NULL
  AND status = 'active'
ORDER BY text_embedding <#> CAST(:vec AS halfvec)
LIMIT 20
```

**为什么使用嵌入向量搜索**:

1. **语义匹配**: 可以找到语义相似但不包含相同关键词的事实
2. **同义词处理**: "car" 和 "automobile" 会被视为相似
3. **上下文理解**: 可以捕捉问题的整体含义

**技术细节**:
- 使用 pgvector 的余弦距离（`<#>` 操作符）
- 使用 halfvec 类型优化存储和计算

### 3.3 Path B2: HyDE 声明式查询嵌入

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L398-L405)

```python
if hyde_vec and hyde_vec is not orig_vec:
    path_b2 = embedding_search_facts(hyde_vec, task_id, db, limit=20)
```

**为什么使用 HyDE**:

1. **声明式查询**: 让 LLM 生成"如果答案存在，它会是什么样子"
2. **更接近记忆形式**: 记忆记录通常以陈述句形式存储，而非问题形式
3. **双重嵌入**: 如果 HyDE 查询与原始问题不同，单独嵌入搜索

**示例**:
- 原始问题: "What does sunflowers represent to Caroline?"
- HyDE 查询: "Sunflowers represent warmth and happiness to Caroline."

### 3.4 Path B3: 事件聚焦查询嵌入

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L407-L418)

```python
if q_event:
    event_vec = embed_client.embed_single(q_event)
    if event_vec:
        path_b3 = embedding_search_facts(event_vec, task_id, db, limit=15)
```

**为什么需要事件聚焦**:

1. **时间问题特殊处理**: "When did X?" 中的时间词会干扰语义匹配
2. **去除噪音**: 移除 "when", "did" 等词，保留核心事件
3. **互补**: 与 Path B 和 B2 互补，覆盖不同类型的查询

### 3.5 Path C: 实体树光束搜索

**文件**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L213-L268)

这是 MemBrain 最独特的检索路径，包含三个步骤：

#### 3.5.1 实体匹配

```python
entity_ids = _match_entities(query, query_vec, task_id, db, top_n=5)
```

**过程**:
1. **BM25 搜索 fact_refs**: 查找包含实体别名的引用
2. **Embedding 搜索 entities**: 在实体描述中搜索语义相似的内容
3. **融合结果**: 合并两种搜索的结果，按得分排序

**为什么需要实体匹配**:
- 先找到问题涉及哪些实体，缩小搜索范围
- 实体是记忆组织的重要维度

#### 3.5.2 树结构光束搜索

```python
fact_ids = _tree_beam_search(entity_id, task_id, db, query_vec, beam_width=3)
```

**光束搜索过程**:

```
根节点
  ├── 子节点 A (相似度: 0.9)
  │     ├── 叶子 1 (fact_id: 101)
  │     └── 叶子 2 (fact_id: 102)
  └── 子节点 B (相似度: 0.7)
        └── 叶子 3 (fact_id: 103)
```

- 从根节点开始
- 每层保留 top-3（beam_width=3）最相似的子节点
- 收集所有叶子节点关联的事实

**为什么使用光束搜索**:

1. **结构化组织**: 记忆按实体-方面（Entity → Aspect）树结构组织
2. **层级导航**: 通过实体树可以找到相关方面的记忆
3. **相似度排序**: 利用嵌入相似度在树中导航

#### 3.5.3 去重和排序

```python
# 按嵌入相似度重新排序
sql = """
    SELECT id, text,
           -(text_embedding <#> CAST(:vec AS halfvec)) AS score
    FROM facts
    WHERE id = ANY(:ids)
    ORDER BY text_embedding <#> CAST(:vec AS halfvec)
"""
```

### 3.6 Path D: BM25 解析查询（Tantivy 语法）

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L427-L432)

```python
if q_bm25_parsed:
    path_d = _bm25_parsed_search_facts(q_bm25_parsed, task_id, db, limit=20)
```

**SQL 查询**:
```sql
SELECT id, text, pdb.score(id) AS score
FROM facts
WHERE id @@@ pdb.parse(:query, lenient => true)
  AND task_id = :task_id
  AND status = 'active'
ORDER BY score DESC
```

**为什么需要 Path D**:

1. **结构化查询**: 使用 Tantivy 查询语法，支持 AND/OR
2. **精确控制**: `+entity` 确保核心实体必须出现
3. **同义词支持**: `(term1 term2)` 形式支持 OR 匹配

---

## 第四阶段：结果合并与去重

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L434-L455)

```python
# 收集每条路径的结果 ID 列表
ranked_lists = [
    [f.fact_id for f in path_a],
    [f.fact_id for f in path_b],
    [f.fact_id for f in path_b2],
    [f.fact_id for f in path_b3],
    [f.fact_id for f in path_c],
    [f.fact_id for f in path_d],
]

# 去重，保留首次出现的版本
seen = {}
for lst in (path_a, path_b, path_b2, path_b3, path_c, path_d):
    for f in lst:
        if f.fact_id not in seen:
            seen[f.fact_id] = f

pool = list(seen.values())
```

**为什么需要合并去重**:

1. **多路径可能返回相同结果**: 同一个事实可能被多条路径检索到
2. **保留首次出现**: 使用字典保持首次出现的顺序和内容
3. **构建候选池**: 为后续融合做准备

---

## 第五阶段：后处理

### 5.1 时间注解注入

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L163-L197)

```python
def _inject_time_annotations(pool, db):
    # 从 time_annotations 表获取时间信息
    # 从 facts 表的 fact_ts 字段获取时间戳
```

**为什么需要时间信息**:

1. **时间上下文**: 帮助 LLM 理解事实发生的时间背景
2. **排序依据**: 事实可以按时间顺序组织
3. **时态理解**: 区分过去、现在、未来的事件

### 5.2 会话编号注入

```python
def _inject_session_numbers(pool, db):
    # 为每个事实关联会话编号
```

**原因**: 事实来源于特定会话，关联会话编号可以：
- 追踪记忆来源
- 后续可以检索相关会话摘要

### 5.3 Aspect 路径富化

**文件**: [aspect_enrichment.py](file:///home/project/MemBrain/membrain/infra/retrieval/aspect_enrichment.py#L23-L75)

```python
aspect_infos = build_aspect_paths(all_fact_ids, task_id, db)

for fact in pool:
    info = aspect_infos.get(fact.fact_id)
    if info:
        fact.entity_ref = info.entity_ref
        fact.aspect_path = info.path
        fact.aspect_summary = info.leaf_desc
```

**Aspect 信息结构**:

```
Entity: Caroline (user's mother)
  └── Root: Person
        └── Aspect: Career
              └── Leaf: Work history
                    └── Fact: "Caroline worked at Google as a software engineer"
```

**为什么需要 Aspect 路径**:

1. **结构化上下文**: 提供事实的层级上下文
2. **去重依据**: 基于 Aspect 进行去重，避免信息冗余
3. **丰富提示**: 帮助 LLM 更好地理解事实的含义

### 5.4 Aspect 级别去重

**文件**: [aspect_enrichment.py](file:///home/project/MemBrain/membrain/infra/retrieval/aspect_enrichment.py#L121-L156)

```python
def aspect_dedup(fact_ids_ordered, aspect_infos, 
                 max_per_leaf=3, max_per_mid=8):
    # 每个叶子方面最多 3 个事实
    # 每个中间方面最多 8 个事实
```

**为什么需要 Aspect 去重**:

1. **避免冗余**: 同一个方面的多个事实可能重复
2. **多样性**: 限制每个方面的数量，确保覆盖多个方面
3. **可控性**: 通过参数控制去重粒度

---

## 第六阶段：结果融合

### 6.1 RRF（互惠排名融合）- 默认策略

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L58-L75)

```python
def _fuse_rrf(pool, ranked_lists):
    """Reciprocal Rank Fusion"""
    k = 60  # 常数
    
    for fact in pool:
        score = 0.0
        for rm in rank_maps:  # 每条路径的排名
            rank = rm.get(fact.fact_id)
            if rank is not None:
                score += 1.0 / (k + rank + 1)
        fact.rerank_score = score
```

**RRF 公式**:
```
score(f) = Σ 1 / (k + rank_i(f))
```

**为什么使用 RRF**:

1. **无参数**: 不需要训练模型
2. **鲁棒**: 对单一路径的噪声不敏感
3. **简单有效**: 已被证明在多路径检索中效果良好

**k=60 的选择**:
- 较大的 k 值可以减少排名差异的影响
- 60 是 BM25/IR 领域的常用值

### 6.2 Rerank（交叉编码器重排）- 可选策略

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L77-L100)

```python
def _fuse_rerank(question, pool, aspect_infos, top_k):
    # 1. 富化每个事实的文本
    enriched = [
        enrich_for_rerank(f.text, aspect_info, f.time_info)
        for f in pool
    ]
    
    # 2. 使用 rerank 模型排序
    ranked = rerank_client.rerank(question, enriched, top_n=top_k)
    
    # 3. 返回 top-k
    return results
```

**为什么使用 Rerank**:

1. **更精确**: Cross-encoder 可以直接计算问题-文档相关性
2. **考虑整体**: 可以理解问题-答案的整体语义
3. **但更慢**: 需要对每个候选进行前向计算

**富化输入**:
```python
enriched = "[Caroline (user's mother) > Career > Work history: Software engineer] "
           "Caroline worked at Google (date: 2023-05-15)"
```

### 6.3 排序和截断

```python
if strategy == "rerank":
    round1_facts = _fuse_rerank(...)
else:
    _fuse_rrf(pool, ranked_lists)
    pool.sort(key=lambda f: f.rerank_score, reverse=True)
    round1_facts = pool[:top_k]  # 默认 top_k=12
```

---

## 第七阶段：Agentic Round 2（仅 reflect 模式）

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L448-L498)

```python
if mode == "reflect" and http_client:
    # 1. LLM 分析第一轮结果是否足够
    refined_queries = _reflect_and_refine(question, round1_facts, http_client)
    
    # 2. 如果不够，生成补充查询
    if refined_queries:
        extra_pool = []
        for rq in refined_queries:
            # BM25 搜索
            for f in bm25_search_facts(rq, task_id, db, limit=15):
                if f.fact_id not in seen_ids:
                    extra_pool.append(f)
            
            # Embedding 搜索
            rq_vec = embed_client.embed_single(rq)
            for f in embedding_search_facts(rq_vec, task_id, db, limit=15):
                if f.fact_id not in seen_ids:
                    extra_pool.append(f)
        
        # 3. 合并额外结果
        round1_facts = round1_facts + extra_pool
```

**反思系统提示**:
```python
_REFLECT_SYSTEM = """\
You are analysing retrieved memory facts to identify what is still missing.
Given a question and the top retrieved facts, output ONLY valid JSON:
{
  "sufficient": true/false,
  "refined_queries": ["query1", "query2"]
}
- Set sufficient=true if the facts clearly contain enough to answer the question.
- Otherwise provide 1-2 targeted queries that search for the missing information.
"""
```

**为什么需要 Agentic Round 2**:

1. **自我纠正**: 让系统评估检索结果是否足够
2. **针对性补充**: 如果不够，生成精确的补充查询
3. **迭代改进**: 类似人类逐步完善搜索的过程

---

## 第八阶段：实体引用解析

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L501-L503)

```python
_resolve_pool_entity_refs(round1_facts, db)
```

**示例**:
- 事实文本: "Caroline [Caroline] went to Paris."
- 解析后: "Caroline (user's mother) went to Paris."

**为什么需要解析**:

1. **可读性**: LLM 需要知道实体 bracket 指的是什么
2. **消歧**: 区分同名实体
3. **上下文**: 提供实体的描述信息

---

## 第九阶段：会话检索

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L505-L522)

### 9.1 BM25 会话搜索

```python
for sq in [bm25_query, q_hyde, question]:
    for s in bm25_search_sessions(sq, task_id, db, limit=8):
        if s.session_id not in seen_sess:
            seen_sess[s.session_id] = s
```

### 9.2 解析查询会话搜索

```python
if q_bm25_parsed:
    for s in _bm25_parsed_search_sessions(q_bm25_parsed, task_id, db, limit=5):
        if s.session_id not in seen_sess:
            seen_sess[s.session_id] = s
```

### 9.3 事实聚合评分

**文件**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L363-L418)

```python
for s in retrieve_sessions(question, task_id, db, round1_facts, limit=6):
    if s.session_id not in seen_sess:
        seen_sess[s.session_id] = s
```

**为什么需要会话检索**:

1. **对话历史**: 检索相关的会话上下文
2. **补充信息**: 会话摘要可能包含事实中没有的全局信息
3. **多维度**: 从事实和会话两个维度获取信息

---

## 第十阶段：上下文打包

**文件**: [budget_pack.py](file:///home/project/MemBrain/membrain/retrieval/core/budget_pack.py#L29-L67)

### 10.1 事实打包

```python
def budget_pack(facts, max_tokens=4500):
    # 1. 按 rerank_score 排序
    sorted_facts = sorted(facts, key=lambda f: f.rerank_score, reverse=True)
    
    # 2. 贪婪填充预算
    selected = []
    total_tokens = 0
    
    for fact in sorted_facts:
        line = _format_fact_line(fact)
        line_tokens = estimate_tokens(line)
        if total_tokens + line_tokens > max_tokens:
            continue
        selected.append(fact)
        total_tokens += line_tokens
    
    # 3. 按时间顺序输出
    sorted_selected = sorted(selected, key=_sort_key_time)
    lines = ["## Additional Facts"] + [_format_fact_line(f) for f in sorted_selected]
    
    return PackedContext(text="\n".join(lines), ...)
```

**为什么这样设计**:

1. **分数优先**: 先选择最相关的事实
2. **时间排序**: 输出时按时间顺序，便于阅读
3. **Token 预算**: 确保上下文不超过 LLM 限制

### 10.2 事实格式化

```python
def _format_fact_line(fact):
    # 处理内联日期标注
    has_inline = bool(_RELATIVE_DATE_RE.search(fact.text))
    line = f"- {_resolve_inline_dates(fact.text)}"
    
    # 添加时间上下文（如果没有内联日期）
    if fact.time_info and not has_inline:
        line += f" (known from message on {_clean_time_info(fact.time_info)})"
    
    return line
```

### 10.3 会话章节格式化

**文件**: [budget_pack.py](file:///home/project/MemBrain/membrain/retrieval/core/budget_pack.py#L86-L112)

```python
def format_session_section(sessions, max_tokens=1500):
    if not sessions:
        return ""
    
    header = "## Relevant Episodes"
    lines = [header]
    budget = max_tokens - estimate_tokens(header)
    
    for s in sessions:
        entry = f"**{s.subject}**: {s.content}\n---"
        cost = estimate_tokens(entry)
        if budget - cost < 0:
            break
        lines.append(entry)
        budget -= cost
    
    return "\n\n".join(lines)
```

### 10.4 合并上下文

```python
packed = budget_pack(round1_facts, max_tokens=4500)

session_section = format_session_section(sessions, 1500)
if session_section:
    packed.text = session_section + "\n\n" + packed.text
    packed.token_count += estimate_tokens(session_section)
```

---

## 第十一阶段：返回结果

**文件**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L524-L559)

```python
return {
    "packed_context": packed.text,           # 打包的上下文文本
    "packed_token_count": packed.token_count,  # Token 数量
    "fact_ids": packed.fact_ids,              # 事实 ID 列表
    "facts": [                                 # 事实详情
        {
            "fact_id": f.fact_id,
            "text": f.text,
            "source": f.source,
            "rerank_score": f.rerank_score,
            "time_info": f.time_info,
            "entity_ref": f.entity_ref,
            "aspect_path": f.aspect_path,
        }
        for f in round1_facts
    ],
    "sessions": [...],  # 会话列表
    "raw_messages": [],  # 原始消息（预留）
}
```

---

## 配置参数汇总

**文件**: [config.py](file:///home/project/MemBrain/membrain/config.py#L107-L125)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QA_BM25_FACT_TOP_N` | 20 | Path A (BM25) 返回数量 |
| `QA_EMBED_FACT_TOP_N` | 20 | Path B (Embed) 返回数量 |
| `QA_ENTITY_TOP_N` | 5 | Path C 实体匹配数量 |
| `QA_TREE_BEAM_WIDTH` | 3 | Path C 光束宽度 |
| `QA_TREE_FACT_TOP_N` | 20 | Path C 事实上限 |
| `QA_BUDGET_MAX_TOKENS` | 2000 | 事实 token 预算 |
| `QA_RERANK_TOP_K` | 12 | Rerank 后保留数量 |
| `QA_SESSION_BM25_TOP_N` | 10 | 会话 BM25 搜索数量 |
| `QA_SESSION_FINAL_TOP_N` | 5 | 最终会话数量 |
| `QA_BM25_MSG_TOP_N` | 5 | 消息 BM25 搜索数量 |

---

## 完整流程图

```
┌──────────────────────────────────────────────────────────────────────┐
│                          用户问题                                      │
│                  "When did Caroline have a picnic?"                  │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    第一步：查询扩展                                     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 1. Query Rewrite (LLM)                                      │   │
│  │    "When did Caroline have a picnic?" → "Caroline picnic"  │   │
│  │                                                              │   │
│  │ 2. Multi-Query Generation (LLM)                            │   │
│  │    • Event: "What did Caroline do outdoors?"              │   │
│  │    • HyDE: "Caroline had a picnic with friends."           │   │
│  │    • BM25: "Caroline picnic friends"                       │   │
│  │                                                              │   │
│  │ 3. BM25 Query Gen (LLM)                                    │   │
│  │    → "+caroline +picnic (friend outdoor summer)"           │   │
│  │                                                              │   │
│  │ 4. Embedding                                                │   │
│  │    • orig_vec = embed(question)                            │   │
│  │    • hyde_vec = embed(hyde_query)                          │   │
│  │    • rewrite_vec = embed(rewritten)                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    第二步：六条检索路径                                 │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Path A   │  │ Path B   │  │ Path B2  │  │ Path B3  │            │
│  │ BM25     │  │ Embed    │  │ HyDE     │  │ Event    │            │
│  │ 关键词    │  │ 原始问题  │  │ 声明式   │  │ 事件聚焦  │            │
│  │ 20条     │  │ 20条     │  │ 20条     │  │ 15条     │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │             │             │             │                   │
│  ┌────┴─────────────┴─────────────┴─────────────┴────┐             │
│  │ Path C: 实体树光束搜索                              │             │
│  │ 1. 匹配实体 (BM25 + Embedding)                    │             │
│  │ 2. 树结构光束搜索 (beam_width=3)                  │             │
│  │ 3. 收集叶子事实                                    │             │
│  └────────────────────┬─────────────────────────────┘             │
│                        │                                             │
│  ┌────────────────────┴─────────────────────────────┐             │
│  │ Path D: Tantivy 解析查询                         │             │
│  │ "+caroline +picnic (friend outdoor summer)"       │             │
│  └──────────────────────────────────────────────────┘             │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    第三步：合并去重                                   │
│                                                                      │
│  合并 6 条路径的结果，按 fact_id 去重                                 │
│  构建候选池 (pool)                                                    │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    第四步：后处理                                     │
│                                                                      │
│  1. 时间注解注入 ───→ 添加事实的时间信息                              │
│  2. 会话编号注入 ──→ 关联事实到会话                                   │
│  3. Aspect 富化 ───→ 添加实体树路径上下文                            │
│  4. Aspect 去重 ───→ 每叶子最多3个，每中间最多8个                   │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    第五步：融合排序                                   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 策略 1: RRF (Reciprocal Rank Fusion) - 默认                │   │
│  │   score = Σ 1/(60 + rank)                                  │   │
│  │   优点: 无参数、鲁棒                                         │   │
│  │                                                              │   │
│  │ 策略 2: Cross-encoder Rerank                                │   │
│  │   输入: [Entity > Aspect] fact (date: time)                │   │
│  │   输出: 相关性分数                                          │   │
│  │   优点: 更精确，但更慢                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  取 top_k (默认 12) 条事实                                           │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐    ┌──────────────────────────────────────┐
│  mode="reflect"?        │    │   会话检索                            │
│  (Agentic Round 2)      │    │                                      │
│                         │    │  1. BM25 搜索会话摘要                 │
│  1. LLM 分析是否足够    │    │  2. 解析查询搜索                      │
│  2. 如不足，生成补充查询 │    │  3. 事实聚合评分                      │
│  3. 补充检索            │    │                                      │
└─────────┬───────────────┘    └──────────────────┬───────────────────┘
          │                                        │
          └──────────────────┬─────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    第六步：实体引用解析                                 │
│                                                                      │
│  "Caroline [Caroline] went to Paris"                                 │
│         ↓                                                             │
│  "Caroline (user's mother) went to Paris"                            │
└─────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    第七步：上下文打包                                   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ## Relevant Episodes                                        │   │
│  │ **Summer 2023**: Caroline organized a picnic...              │   │
│  │ ---                                                          │   │
│  │                                                               │   │
│  │ ## Additional Facts                                         │   │
│  │ - Caroline and friends had a picnic (known from 2023-07-15) │   │
│  │ - The picnic was at Central Park                            │   │
│  │ ...                                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Token 预算: 事实 4500 + 会话 1500 = 6000                            │
└─────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         返回结果                                       │
│                                                                      │
│  • packed_context: 打包的上下文文本                                    │
│  • packed_token_count: 6023                                         │
│  • fact_ids: [101, 203, 87, ...]                                    │
│  • facts: [{fact_id, text, source, rerank_score, ...}, ...]       │
│  • sessions: [{session_id, subject, content, score, ...}, ...]     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 设计理念总结

### 1. 多样性与精确性的平衡

- **多路径检索**: 不同的检索方法擅长不同类型的问题
- **融合策略**: 将多路径结果有效合并

### 2. 层次化的记忆组织

- **实体树结构**: 记忆按实体-方面层级组织
- **光束搜索**: 利用树结构进行导航式检索

### 3. 查询理解与扩展

- **LLM 辅助**: 使用 LLM 理解和扩展查询
- **多样化查询**: 同一问题生成多种查询变体

### 4. 可控性和灵活性

- **模式选择**: direct / expand / reflect
- **策略选择**: rrf / rerank
- **参数调优**: 通过配置控制各阶段行为

### 5. 实用性优先

- **Token 预算**: 确保输出在 LLM 限制内
- **时间排序**: 输出按时间顺序，便于阅读
- **上下文丰富**: 包含时间、实体、会话等多维度信息


## 3. 写入过程分析

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


## 4. 01 api entry

# Stage 1: API 入口 - 搜索请求的接收与初始化

## 概述

本阶段是 MemBrain 搜索过程的入口点，负责接收外部 API 请求，解析参数，获取必要的客户端资源，并设置数据库 schema 上下文。

## 代码位置

- **API 路由**: [memory.py](file:///home/project/MemBrain/membrain/api/routes/memory.py#L281-L316)
- **服务管理器**: [manager.py](file:///home/project/MemBrain/membrain/api/manager.py)

## 详细代码分析

### 1.1 API 路由定义

```python
# membrain/api/routes/memory.py
@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(req: MemorySearchRequest):
```

**关键点**:
- 使用 FastAPI 的 POST 路由装饰器
- 响应模型为 `MemorySearchResponse`

### 1.2 请求参数解析

```python
async def search_memory(req: MemorySearchRequest):
    # 1. 解析任务标识
    resolved = get_task_pk(req.dataset, req.task)
    if resolved is None:
        raise HTTPException(
            404, f"Task '{req.task}' not found in dataset '{req.dataset}'"
        )
    task_pk = resolved
```

**为什么需要任务解析**:
- MemBrain 支持多任务（多用户/多会话）
- 每个任务有独立的数据库 schema
- `task_pk` 是任务的唯一标识符

**请求参数结构**（查看 MemorySearchRequest）:
```python
# 假设的请求结构
class MemorySearchRequest:
    dataset: str      # 数据集名称，如 "personamem_v2"
    task: str         # 任务标识，如 "user_123"
    question: str     # 用户问题
    top_k: int | None = None      # 返回结果数量
    strategy: Literal["rrf", "rerank"] = "rrf"  # 融合策略
    mode: Literal["direct", "expand", "reflect"] = "expand"  # 检索模式
```

### 1.3 获取服务客户端

```python
# 获取数据库会话工厂
sf = search_mgr.get_session_factory()

# 获取嵌入客户端（用于向量搜索）
embed_client = search_mgr.get_embed_client()

# 获取 HTTP 客户端（用于 LLM 调用）
http_client = search_mgr.get_http_client()

# 获取 top_k 参数
top_k = req.top_k or settings.QA_RERANK_TOP_K  # 默认 12
```

**为什么需要三个客户端**:

| 客户端 | 用途 | 关键操作 |
|--------|------|----------|
| `session_factory` | 数据库操作 | 执行 SQL 查询 |
| `embed_client` | 向量化 | 将文本转换为向量 |
| `http_client` | LLM 调用 | 查询重写、多查询生成 |

### 1.4 SearchServiceManager 实现

```python
# membrain/api/manager.py

class SearchServiceManager:
    """Shared engine + global clients for read-only memory search."""

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._sf: sessionmaker | None = None
        self._embed_client: EmbeddingClient | None = None
        self._http_client: httpx.Client | None = None

    def _ensure_engine(self) -> None:
        if self._engine is None:
            self._engine = sa_create_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                pool_size=settings.QA_SEARCH_POOL_SIZE,  # 默认 20
                max_overflow=10,
                pool_timeout=settings.DB_POOL_TIMEOUT,   # 默认 60s
                pool_recycle=settings.DB_POOL_RECYCLE,   # 默认 3600s
            )
            self._sf = sessionmaker(bind=self._engine)
```

**连接池配置说明**:
- `pool_pre_ping=True`: 每次使用前检测连接是否有效
- `pool_size=20`: 保持 20 个活跃连接
- `max_overflow=10`: 允许额外 10 个临时连接
- `pool_recycle=3600`: 1小时后回收连接

### 1.5 客户端获取方法

```python
def get_session_factory(self) -> sessionmaker:
    self._ensure_engine()
    return self._sf

def get_embed_client(self) -> EmbeddingClient:
    if self._embed_client is None:
        self._embed_client = EmbeddingClient()
    return self._embed_client

def get_http_client(self) -> httpx.Client:
    if self._http_client is None:
        self._http_client = httpx.Client(timeout=60.0)
    return self._http_client
```

**单例模式**: 客户端采用延迟初始化和单例模式，避免重复创建。

### 1.6 设置数据库 Schema

```python
# 设置搜索路径到特定任务的 schema
schema = f"task_{int(task_pk)}__{_RUN_TAG}"
with sf() as db:
    db.execute(sa_text(f"SET LOCAL search_path TO {schema}, public"))
```

**为什么需要 Schema 隔离**:
- 每个任务有独立的 schema: `task_1__default`, `task_2__default`
- 数据完全隔离，避免不同任务的数据混淆
- `search_path` 允许同时访问任务 schema 和 public schema

**Schema 命名规则**:
```
task_{task_pk}__{run_tag}
例如: task_1__default
```

### 1.7 调用核心搜索函数

```python
# 在设置的 schema 上下文中执行搜索
result = _retrieval.search(
    question=req.question,
    task_id=task_pk,
    db=db,
    embed_client=embed_client,
    http_client=http_client,
    top_k=top_k,
    strategy=req.strategy,
    mode=req.mode,
)
```

### 1.8 响应构建

```python
return MemorySearchResponse(
    packed_context=result["packed_context"],
    packed_token_count=result["packed_token_count"],
    fact_ids=result["fact_ids"],
    facts=[RetrievedFactOut(**f) for f in result["facts"]],
    sessions=[RetrievedSessionOut(**s) for s in result["sessions"]],
    raw_messages=[],
)
```

## 完整流程示例

### 示例请求

```json
{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "When did Caroline have a picnic?",
    "top_k": 15,
    "strategy": "rrf",
    "mode": "expand"
}
```

### 执行流程

```
1. HTTP POST /api/memory/search
       │
       ▼
2. 解析请求参数
   dataset = "personamem_v2"
   task = "user_001"
   question = "When did Caroline have a picnic?"
   top_k = 15
   strategy = "rrf"
   mode = "expand"
       │
       ▼
3. 获取 task_pk = 1 (假设)
       │
       ▼
4. 获取/创建客户端
   - session_factory (数据库)
   - embed_client (嵌入服务)
   - http_client (LLM API)
       │
       ▼
5. 创建数据库会话
   设置 schema = "task_1__default"
       │
       ▼
6. 调用 _retrieval.search(...)
       │
       ▼
7. 返回 MemorySearchResponse
```

## 关键配置参数

```python
# membrain/config.py

# 搜索服务连接池
QA_SEARCH_POOL_SIZE = 20        # 连接池大小
QA_RERANK_TOP_K = 12           # 默认返回结果数

# 数据库连接
DB_POOL_SIZE = 5               # 默认池大小
DB_POOL_TIMEOUT = 60           # 超时时间(秒)
DB_POOL_RECYCLE = 3600        # 回收时间(秒)
```

## 错误处理

```python
# 1. 任务不存在
if resolved is None:
    raise HTTPException(404, f"Task '{req.task}' not found in dataset '{req.dataset}'")

# 2. 客户端创建失败
# 由调用方（_retrieval.search）捕获和处理

# 3. 数据库连接失败
# 由 SQLAlchemy 异常处理机制处理
```

## 总结

本阶段的核心职责:

| 职责 | 说明 |
|------|------|
| **参数解析** | 解析请求中的 dataset、task、question 等参数 |
| **任务解析** | 将 dataset + task 转换为 task_pk |
| **资源获取** | 获取/创建数据库会话、嵌入客户端、HTTP 客户端 |
| **上下文设置** | 设置数据库 schema 隔离 |
| **结果返回** | 将搜索结果转换为 API 响应格式 |

本阶段为后续的查询扩展和检索阶段做好了准备，是整个搜索流程的"守门人"。


## 5. 02 query expansion

# Stage 2: 查询扩展 - Query Expansion

## 概述

查询扩展是 MemBrain 搜索过程中最关键的预处理阶段。当用户提出一个问题时，直接使用原始问题进行检索往往效果不佳，因为：
1. 问题中包含大量不用于检索的词（如 "When", "did", "what"）
2. 不同形式的查询适用于不同的检索方法
3. 需要利用 LLM 的能力理解查询意图

MemBrain 通过三种查询扩展技术来解决这些问题：
1. **查询重写 (Query Rewrite)**: 将问题转换为关键词短语
2. **多查询生成 (Multi-Query)**: 生成三个互补查询
3. **BM25 查询生成**: 生成 Tantivy 语法查询

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L348-L370)
- **查询重写**: [query_rewrite.py](file:///home/project/MemBrain/membrain/infra/clients/query_rewrite.py)
- **多查询生成**: [multi_query.py](file:///home/project/MemBrain/membrain/infra/clients/multi_query.py)
- **BM25 查询生成**: [bm25_query_gen.py](file:///home/project/MemBrain/membrain/infra/clients/bm25_query_gen.py)

## 详细代码分析

### 2.1 入口：模式判断

```python
# membrain/retrieval/application/retrieval.py

# ── 1. Query rewrite + multi-query expansion ─────────────────────────
if mode == "direct":
    # 直接模式：不使用任何 LLM 查询扩展
    rewritten = question
    q_event = q_hyde = q_bm25_kw = q_bm25_parsed = ""
else:
    # 扩展模式：使用 LLM 进行查询扩展
    rewritten = rewrite_query(question, http_client)
    extra_queries = generate_multi_queries(question, http_client)
    q_event = extra_queries[0] if len(extra_queries) > 0 else ""
    q_hyde = extra_queries[1] if len(extra_queries) > 1 else ""
    q_bm25_kw = extra_queries[2] if len(extra_queries) > 2 else rewritten
    q_bm25_parsed = generate_bm25_query(question, http_client)
```

**模式选择说明**:

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| `direct` | 不使用 LLM 扩展 | 低延迟、问题已简洁明确 |
| `expand` | 使用全部扩展 | **默认**，大多数场景 |
| `expand` | 使用全部扩展 + Agentic Round 2 | 需要更高召回率 |

### 2.2 查询重写 (Query Rewrite)

**文件**: [query_rewrite.py](file:///home/project/MemBrain/membrain/infra/clients/query_rewrite.py)

```python
def rewrite_query(
    question: str,
    http_client: httpx.Client,
    model: str = "",
) -> str:
    """Convert question → 3-6 keyword phrase for BM25/embedding retrieval."""
    
    m = model or settings.QA_LLM_MODEL  # 默认 "gpt-4.1-mini"
    
    try:
        resp = http_client.post(
            f"{settings.LLM_API_URL.rstrip('/')}/chat/completions",
            json={
                "model": m,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _EXAMPLES + f"Q: {question} →"},
                ],
                "max_tokens": 40,
                "temperature": 0.0,  # 确定性输出
            },
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            timeout=15.0,
        )
        resp.raise_for_status()
        keywords = resp.json()["choices"][0]["message"]["content"].strip()
        return keywords if keywords else question
    except Exception:
        return question
```

**系统提示 (Prompt)**:

```python
_SYSTEM = (
    "Extract 3-6 search keywords from the question. "
    "Keep proper nouns exactly as written. "
    "Use base/infinitive verb forms (e.g. 'research' not 'researching'). "
    "Remove question words (what/when/did/who/how/is/are). "
    "Output only the keywords, space-separated, no punctuation."
)
```

**示例 (Examples)**:

```python
_EXAMPLES = (
    "Q: When did Melanie paint a sunrise? → Melanie paint sunrise\n"
    "Q: What did Caroline research? → Caroline research\n"
    "Q: What is Caroline's identity? → Caroline identity\n"
    "Q: Where did they go for their anniversary dinner? → anniversary dinner location\n"
)
```

### 2.3 查询重写示例

| 输入问题 | 输出关键词 |
|----------|-----------|
| "When did Melanie paint a sunrise?" | "Melanie paint sunrise" |
| "What did Caroline research?" | "Caroline research" |
| "What is Caroline's identity?" | "Caroline identity" |
| "Where did they go for their anniversary dinner?" | "anniversary dinner location" |
| "How many children does Emily have?" | "Emily children" |
| "What color did Anna paint her kitchen?" | "Anna kitchen paint color" |

**为什么这样设计**:
1. **去除疑问词**: "When", "What", "Where" 等词对检索无帮助
2. **词根形式**: "painting" → "paint"，提高匹配概率
3. **保留专有名词**: "Caroline" 保持原样
4. **简洁**: 只保留 3-6 个核心关键词

### 2.4 多查询生成 (Multi-Query Generation)

**文件**: [multi_query.py](file:///home/project/MemBrain/membrain/infra/clients/multi_query.py)

```python
def generate_multi_queries(
    question: str,
    http_client: httpx.Client,
    model: str = "",
) -> list[str]:
    """Generate 3 complementary search queries from a question."""
    
    if not settings.QA_MULTI_QUERY_ENABLED:  # 默认 True
        return []
    
    m = model or settings.QA_LLM_MODEL
    try:
        resp = http_client.post(
            f"{settings.LLM_API_URL.rstrip('/')}/chat/completions",
            json={
                "model": m,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"Question: {question}"},
                ],
                "max_tokens": 250,
                "temperature": 0.0,
            },
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            timeout=20.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        
        # 解析 JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        queries = data.get("queries", [])
        
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:3]
        return []
    except Exception:
        log.debug("multi_query generation failed", exc_info=True)
        return []
```

**系统提示**:

```python
_SYSTEM = """\
You are an expert at query reformulation for long-term conversational memory retrieval.
Generate EXACTLY 3 complementary search queries. Each query has a fixed role:

Query 1 — Event-focused (for embedding):
  For temporal questions ("when did X?", "how long ago?"), drop the time aspect
  entirely and focus on the EVENT itself.
  Example: "When did Caroline have a picnic?" → "What did Caroline do with friends outdoors?"
  For non-temporal questions, write a specific direct question as-is.

Query 2 — HyDE declarative (for embedding):
  Write the sentence that WOULD appear verbatim in a memory record if the answer existed.
  Example: "Caroline and her friends had a picnic together."
  Example: "Sunflowers represent warmth and happiness to Caroline."

Query 3 — BM25 keyword strip (for keyword search):
  Keep ONLY entity names + core noun/verb base forms.
  Example: "When did Caroline have a picnic?" → "Caroline friends picnic"

Output ONLY valid JSON: {"queries": ["...", "...", "..."]}\
"""
```

### 2.5 多查询生成示例

**示例 1**: 时间问题

| 输入 | 输出 |
|------|------|
| 问题 | "When did Caroline have a picnic?" |
| Event-focused | "What did Caroline do with friends outdoors?" |
| HyDE | "Caroline and her friends had a picnic together." |
| BM25 | "Caroline picnic friends" |

**示例 2**: 象征意义问题

| 输入 | 输出 |
|------|------|
| 问题 | "What do sunflowers represent to Caroline?" |
| Event-focused | "What is the meaning of sunflowers to Caroline?" |
| HyDE | "Sunflowers represent warmth and happiness to Caroline." |
| BM25 | "Caroline sunflower meaning" |

**示例 3**: 原因问题

| 输入 | 输出 |
|------|------|
| 问题 | "Why did Melanie use colors in her pottery?" |
| Event-focused | "What is the reason behind Melanie's colorful pottery?" |
| HyDE | "Melanie used vibrant colors in her pottery to express creativity." |
| BM25 | "Melanie pottery colors reason" |

**为什么需要三个不同的查询**:

1. **Event-focused (事件聚焦)**:
   - 移除时间词 "when"，聚焦于事件本身
   - 嵌入模型更容易理解核心事件

2. **HyDE (假设性声明)**:
   - 让 LLM 生成"如果答案存在，它会以什么形式出现"
   - 更接近记忆存储的实际形式
   - 嵌入模型可以找到语义相似的内容

3. **BM25 keyword**:
   - 最简洁的关键词形式
   - 适合基于词频的 BM25 算法

### 2.6 BM25 查询生成 (Tantivy Query)

**文件**: [bm25_query_gen.py](file:///home/project/MemBrain/membrain/infra/clients/bm25_query_gen.py)

```python
def generate_bm25_query(
    question: str,
    http_client: httpx.Client,
    model: str = "",
) -> str:
    """Generate a single Tantivy query string from a user question."""
    
    m = model or settings.QA_LLM_MODEL
    
    try:
        resp = http_client.post(
            f"{settings.LLM_API_URL.rstrip('/')}/chat/completions",
            json={
                "model": m,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"Q: {question}"},
                ],
                "max_tokens": 100,
                "temperature": 0.0,
            },
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            timeout=15.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        
        # 解析 JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        query = data.get("query", "")
        
        if query:
            return _sanitize_tantivy_query(query)
        return ""
    except Exception:
        log.debug("BM25 query generation failed", exc_info=True)
        return ""
```

**系统提示**:

```python
_SYSTEM = """\
You are a search query generator for a BM25 full-text index on personal memory records.

# Task
Given a question about someone's life, generate a single Tantivy query string.

# Tantivy Query Syntax
- `+term` — MUST appear (AND)
- `term` — SHOULD appear (OR)
- `(term1 term2 term3)` — Grouping (OR)
- Combine: `+entity (synonym1 synonym2)`

# Query Construction Strategy
1. **Mark core entities with `+`**: the person MUST appear
2. **Expand with synonyms**: child→(kid son daughter), job→(work career)
3. **Use base verb forms**: swim not swimming
4. **Include 6-12 terms total**

# Output
Output ONLY valid JSON: {"query": "+entity (term1 term2 ...)"}\
"""
```

### 2.7 BM25 查询示例

| 输入问题 | 生成的 Tantivy 查询 |
|----------|-------------------|
| "How many children does Emily have?" | `+emily (child kid son daughter one two three four five)` |
| "Where did Max hide his bone?" | `+max +bone (hide bury slipper couch under secret spot)` |
| "What is Sarah's job?" | `+sarah (job work career employ office profess company)` |
| "When did Jake go on a road trip?" | `+jake +road +trip (drive car highway motel camp travel)` |
| "Who is Rachel's husband?" | `+rachel (husband spouse partner marry wedding ring)` |
| "What books has David read recently?" | `+david (book read novel author literature story recent)` |
| "What color did Anna paint her kitchen?" | `+anna +kitchen (paint color wall red blue green white yellow)` |
| "Does Tom have any pets?" | `+tom (pet dog cat animal puppy kitten fish bird)` |

**Tantivy 查询语法解释**:

- `+emily`: "emily" 必须出现
- `(child kid son daughter)`: 这些词至少出现一个
- 结果: "emily" **必须出现**，并且至少出现一个同义词组的词

### 2.8 向量嵌入

```python
# ── Embedding generation ───────────────────────────────────────────────
try:
    orig_vec = embed_client.embed_single(question)
except Exception:
    log.warning("Embedding failed for original question")
    orig_vec = None

try:
    hyde_vec = embed_client.embed_single(q_hyde) if q_hyde else orig_vec
except Exception:
    hyde_vec = orig_vec

try:
    rewrite_vec = (
        embed_client.embed_single(rewritten) if rewritten != question else orig_vec
    )
except Exception:
    rewrite_vec = orig_vec
```

**为什么需要多个向量**:

| 向量 | 用途 | 对应路径 |
|------|------|----------|
| `orig_vec` | 原始问题 | Path B |
| `hyde_vec` | HyDE 查询 | Path B2 |
| `rewrite_vec` | 重写后查询 | Path C |

**容错机制**: 如果任何嵌入失败，回退到原始问题向量。

## 完整示例

### 输入

```python
question = "When did Caroline have a picnic with her friends?"
mode = "expand"
```

### 处理过程

```
Step 1: 查询重写
  输入: "When did Caroline have a picnic with her friends?"
  输出: "Caroline picnic friends"

Step 2: 多查询生成
  输入: "When did Caroline have a picnic with her friends?"
  输出: 
    - q_event: "What did Caroline do with friends outdoors?"
    - q_hyde: "Caroline and her friends had a picnic together."
    - q_bm25_kw: "Caroline picnic friends"

Step 3: BM25 查询生成
  输入: "When did Caroline have a picnic with her friends?"
  输出: "+caroline +picnic (friend outdoor summer party gathering)"

Step 4: 向量嵌入
  - orig_vec = embed("When did Caroline have a picnic with her friends?")
  - hyde_vec = embed("Caroline and her friends had a picnic together.")
  - rewrite_vec = embed("Caroline picnic friends")
```

### 输出变量

| 变量 | 值 | 用途 |
|------|-----|------|
| `rewritten` | "Caroline picnic friends" | Path A BM25 |
| `q_event` | "What did Caroline do with friends outdoors?" | Path B3 |
| `q_hyde` | "Caroline and her friends had a picnic together." | Path B2 |
| `q_bm25_kw` | "Caroline picnic friends" | Path A 补充 |
| `q_bm25_parsed` | "+caroline +picnic (friend outdoor summer party)" | Path D |
| `orig_vec` | [0.1, -0.3, ...] | Path B |
| `hyde_vec` | [0.2, -0.1, ...] | Path B2 |
| `rewrite_vec` | [0.15, -0.25, ...] | Path C |

## 配置参数

```python
# membrain/config.py

QA_MULTI_QUERY_ENABLED = True   # 是否启用多查询生成
QA_LLM_MODEL = "gpt-4.1-mini"   # 使用的 LLM 模型
```

## 错误处理

```python
# 1. LLM 调用失败 - 回退到原始问题
except Exception:
    return question  # rewrite_query

# 2. 多查询生成失败 - 返回空列表
except Exception:
    log.debug("multi_query generation failed", exc_info=True)
    return []

# 3. BM25 查询生成失败 - 返回空字符串
except Exception:
    log.debug("BM25 query generation failed", exc_info=True)
    return ""

# 4. 嵌入失败 - 回退到 None
try:
    orig_vec = embed_client.embed_single(question)
except Exception:
    log.warning("Embedding failed for original question")
    orig_vec = None
```

## 总结

查询扩展阶段的核心目标是将用户问题转换为多种适合不同检索方法的查询形式：

| 技术 | 输入 | 输出 | 适用检索 |
|------|------|------|----------|
| Query Rewrite | 问题 | 关键词短语 | Path A |
| Multi-Query Event | 问题 | 事件描述 | Path B3 |
| Multi-Query HyDE | 问题 | 声明式句子 | Path B2 |
| Multi-Query BM25 | 问题 | 关键词 | Path A 补充 |
| BM25 Query Gen | 问题 | Tantivy 查询 | Path D |

这一阶段充分利用了 LLM 的能力，将原始问题转化为更适合检索的形式，显著提高了后续检索阶段的效果。


## 6. 03 multi path retrieval

# Stage 3: 多路径检索 - Multi-Path Retrieval

## 概述

多路径检索是 MemBrain 搜索过程的核心环节。通过同时执行六条独立的检索路径，可以从不同角度捕捉与问题相关的记忆信息。

这六条路径分别是：
- **Path A**: BM25 关键词搜索
- **Path B**: 嵌入向量搜索（原始问题）
- **Path B2**: HyDE 声明式查询嵌入
- **Path B3**: 事件聚焦查询嵌入
- **Path C**: 实体树光束搜索
- **Path D**: BM25 解析查询（Tantivy）

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L372-L432)
- **BM25 搜索**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L44-L95)
- **嵌入搜索**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L73-L95)
- **实体树搜索**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L213-L268)
- **候选检索**: [candidate_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/candidate_retrieval.py)

## 详细代码分析

### 3.1 Path A: BM25 关键词搜索

```python
# membrain/retrieval/application/retrieval.py

# ── 2. Six retrieval paths ────────────────────────────────────────────
bm25_query = q_bm25_kw or rewritten
path_a = bm25_search_facts(
    bm25_query, task_id, db, limit=settings.QA_BM25_FACT_TOP_N  # 默认 20
)

# 补充搜索：如果重写后的关键词与 BM25 关键词不同
if rewritten != bm25_query:
    extra_bm25 = bm25_search_facts(rewritten, task_id, db, limit=10)
    a_ids = {f.fact_id for f in path_a}
    path_a = path_a + [f for f in extra_bm25 if f.fact_id not in a_ids]
```

**实现代码**:

```python
# membrain/infra/retrieval/fact_retrieval.py

def bm25_search_facts(
    query: str,
    task_id: int,
    db: Session,
    limit: int = settings.QA_BM25_FACT_TOP_N,
) -> list[RetrievedFact]:
    """BM25 keyword search on facts."""
    
    # 清洗查询：移除特殊字符
    safe_query = _sanitize_bm25_query(query)
    if not safe_query:
        return []
    
    # SQL 查询
    sql = sa_text("""
        SELECT id, text, pdb.score(id) AS score
        FROM facts
        WHERE search_text ||| :query
          AND task_id = :task_id
          AND status = 'active'
        ORDER BY score DESC
        LIMIT :limit
    """)
    
    try:
        rows = db.execute(
            sql,
            {"query": safe_query, "task_id": task_id, "limit": limit},
        ).fetchall()
    except Exception:
        log.warning("BM25 fact search failed for query %r", query, exc_info=True)
        return []
    
    return [RetrievedFact(fact_id=r[0], text=r[1], source="bm25") for r in rows]
```

**查询清洗函数**:

```python
# membrain/infra/retrieval/candidate_retrieval.py

_TANTIVY_SPECIAL = re.compile(r"""[+\-&|!(){}\[\]^"~*?:\\/'.,$;`]""")

def _sanitize_bm25_query(raw: str) -> str:
    """Escape special chars for ParadeDB / Tantivy."""
    # 1. 移除非 ASCII 字符
    ascii_only = raw.encode("ascii", errors="ignore").decode("ascii")
    # 2. 移除 NUL 字节
    ascii_only = ascii_only.replace("\x00", " ")
    # 3. 移除 Tantivy 特殊字符
    cleaned = _TANTIVY_SPECIAL.sub(" ", ascii_only)
    # 4. 合并空格
    return " ".join(cleaned.split())
```

**为什么使用 BM25**:
1. **成熟的词项匹配**: 基于词频和逆文档频率
2. **高效**: 使用倒排索引
3. **精确**: 擅长精确匹配实体名称和关键词

### 3.2 Path B: 嵌入向量搜索（原始问题）

```python
# 使用原始问题向量搜索
path_b = embedding_search_facts(
    orig_vec, task_id, db, limit=settings.QA_EMBED_FACT_TOP_N
) if orig_vec else []
```

**实现代码**:

```python
# membrain/infra/retrieval/fact_retrieval.py

def embedding_search_facts(
    query_vec: list[float],
    task_id: int,
    db: Session,
    limit: int = settings.QA_EMBED_FACT_TOP_N,
) -> list[RetrievedFact]:
    """Vector search on facts using pgvector."""
    
    # 转换为 PostgreSQL 向量格式
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
    
    sql = sa_text("""
        SELECT id, text,
               -(text_embedding <#> CAST(:vec AS halfvec)) AS score
        FROM facts
        WHERE task_id = :task_id
          AND text_embedding IS NOT NULL
          AND status = 'active'
        ORDER BY text_embedding <#> CAST(:vec AS halfvec)
        LIMIT :limit
    """)
    
    rows = db.execute(
        sql,
        {"vec": vec_str, "task_id": task_id, "limit": limit},
    ).fetchall()
    
    return [RetrievedFact(fact_id=r[0], text=r[1], source="embed") for r in rows]
```

**技术细节**:
- 使用 pgvector 的 `<#>` 操作符计算余弦距离
- `halfvec` 类型是 float32 向量的存储优化
- 负号 `-` 用于将距离转换为相似度分数（距离越小，相似度越高）

**为什么使用嵌入搜索**:
1. **语义匹配**: 可以找到语义相似但不含相同关键词的事实
2. **同义词处理**: "car" 和 "automobile" 会被视为相似
3. **上下文理解**: 理解问题的整体含义

### 3.3 Path B2: HyDE 声明式查询嵌入

```python
# 使用 HyDE 向量搜索
path_b2: list[RetrievedFact] = []
if hyde_vec and hyde_vec is not orig_vec:
    path_b2 = embedding_search_facts(
        hyde_vec, task_id, db, limit=settings.QA_EMBED_FACT_TOP_N
    )
```

**HyDE 技术解释**:

HyDE (Hypothetical Document Embedding) 的核心思想是：
1. 让 LLM 生成一个"假设的答案文档"
2. 对这个假设文档进行嵌入
3. 用假设文档的向量搜索真实文档

**为什么有效**:
- 记忆存储形式通常是陈述句，而非问句
- HyDE 查询更接近记忆的表述方式
- 可以捕捉问题的深层语义

**示例**:
- 问题: "What do sunflowers represent to Caroline?"
- HyDE: "Sunflowers represent warmth and happiness to Caroline."
- 这个声明式句子更接近记忆中的表述

### 3.4 Path B3: 事件聚焦查询嵌入

```python
# 使用事件聚焦查询向量搜索
path_b3: list[RetrievedFact] = []
if q_event:
    try:
        event_vec = embed_client.embed_single(q_event)
        if event_vec:
            path_b3 = embedding_search_facts(event_vec, task_id, db, limit=15)
    except Exception:
        pass
```

**为什么需要事件聚焦查询**:

对于时间问题（如 "When did X?"），时间词会干扰嵌入匹配：
- 问题: "When did Caroline have a picnic?"
- 嵌入 "When did Caroline..." 时，"when" 这个词会带来噪声
- 事件聚焦查询: "What did Caroline do with friends outdoors?"
- 移除了 "when"，保留了核心事件

**限制条件**:
- 只在 `q_event` 非空时执行
- 限制为 15 条（略少于其他路径的 20 条）

### 3.5 Path C: 实体树光束搜索（最复杂）

这是 MemBrain 最独特的检索路径，包含三个步骤。

#### 3.5.1 实体匹配

```python
# 实体树搜索
path_c: list[RetrievedFact] = []
if rewrite_vec:
    path_c = entity_tree_search(question, rewrite_vec, task_id, db)
    if path_c:
        # 应用 Aspect 级别去重
        c_ids = [f.fact_id for f in path_c]
        c_aspects = build_aspect_paths(c_ids, task_id, db)
        kept = set(aspect_dedup(c_ids, c_aspects))
        path_c = [f for f in path_c if f.fact_id in kept]
```

**实体树搜索实现**:

```python
# membrain/infra/retrieval/fact_retrieval.py

def entity_tree_search(
    query: str,
    query_vec: list[float],
    task_id: int,
    db: Session,
    entity_top_n: int = settings.QA_ENTITY_TOP_N,  # 默认 5
    beam_width: int = settings.QA_TREE_BEAM_WIDTH,  # 默认 3
    limit: int = settings.QA_TREE_FACT_TOP_N,  # 默认 20
) -> list[RetrievedFact]:
    """Path C: entity match → tree beam search → collect leaf facts."""
    
    # Step 1: 匹配实体
    entity_ids = _match_entities(query, query_vec, task_id, db, entity_top_n)
    if not entity_ids:
        return []
    
    # Step 2: 对每个实体进行树搜索
    all_fact_ids: list[int] = []
    for eid in entity_ids:
        fids = _tree_beam_search(eid, task_id, db, query_vec, beam_width)
        all_fact_ids.extend(fids)
    
    # Step 3: 去重并排序
    unique_ids = list(dict.fromkeys(all_fact_ids))
    if not unique_ids:
        return []
    
    # 按嵌入相似度排序
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
    sql = sa_text("""
        SELECT id, text,
               -(text_embedding <#> CAST(:vec AS halfvec)) AS score
        FROM facts
        WHERE id = ANY(:ids)
          AND text_embedding IS NOT NULL
          AND status = 'active'
        ORDER BY text_embedding <#> CAST(:vec AS halfvec)
        LIMIT :limit
    """)
    
    rows = db.execute(
        sql, {"vec": vec_str, "ids": unique_ids, "limit": limit}
    ).fetchall()
    
    return [RetrievedFact(fact_id=r[0], text=r[1], source="tree") for r in rows]
```

#### 3.5.2 实体匹配实现

```python
# membrain/infra/retrieval/fact_retrieval.py

def _match_entities(
    query: str,
    query_vec: list[float],
    task_id: int,
    db: Session,
    top_n: int,
) -> list[str]:
    """Find top entities via BM25 on fact_refs + embedding on entity desc."""
    
    # BM25 搜索 fact_refs（实体别名）
    bm25_rows = _bm25_search(query, task_id, db, limit=top_n * 2)
    
    # Embedding 搜索实体描述
    embed_rows = _embedding_search(query_vec, task_id, db, limit=top_n * 2)
    
    # 合并结果
    seen: dict[str, float] = {}
    
    for row in bm25_rows:
        # _bm25_search returns (alias_text, entity_id)
        eid = row[1]
        seen[eid] = max(seen.get(eid, 0.0), 1.0)  # BM25 分数归一化
    
    for row in embed_rows:
        # _embedding_search returns (entity_id, canonical_ref, desc, score)
        eid = row[0]
        seen[eid] = max(seen.get(eid, 0.0), float(row[3]))
    
    # 按分数排序
    ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    return [eid for eid, _ in ranked[:top_n]]
```

**为什么需要实体匹配**:
1. 先缩小范围：找到与问题相关的实体
2. 实体是记忆组织的重要维度
3. 从实体出发，可以导航到相关的方面（Aspects）

#### 3.5.3 树结构光束搜索

```python
# membrain/infra/retrieval/fact_retrieval.py

def _tree_beam_search(
    entity_id: str,
    task_id: int,
    db: Session,
    question_vec: list[float],
    beam_width: int,
) -> list[int]:
    """Walk the entity tree with beam search, return collected fact_ids."""
    
    # 获取该实体的所有树节点
    sql = sa_text("""
        SELECT id, parent_id, node_type, fact_id, description_embedding
        FROM entity_tree_nodes
        WHERE entity_id = :entity_id AND task_id = :task_id
    """)
    
    rows = db.execute(
        sql,
        {"entity_id": entity_id, "task_id": task_id},
    ).fetchall()
    
    if not rows:
        return []
    
    # 构建树结构
    children_map: dict[int | None, list] = defaultdict(list)
    node_map: dict[int, dict] = {}
    
    for r in rows:
        node = {
            "id": r[0],
            "parent_id": r[1],
            "node_type": r[2],
            "fact_id": r[3],
            "embedding": r[4],
        }
        node_map[node["id"]] = node
        children_map[r[1]].append(node)
    
    # 找到根节点
    roots = children_map[None]
    if not roots:
        return []
    
    # 光束搜索
    collected_fact_ids: list[int] = []
    beam = list(roots)
    
    while beam:
        next_beam = []
        
        for node in beam:
            if node["node_type"] == "leaf":
                # 叶子节点：收集事实
                if node["fact_id"] is not None:
                    collected_fact_ids.append(node["fact_id"])
                continue
            
            # 非叶子节点：获取子节点
            kids = children_map.get(node["id"], [])
            if not kids:
                continue
            
            # 检查是否所有子节点都是叶子
            all_leaves = all(k["node_type"] == "leaf" for k in kids)
            
            if all_leaves:
                # 叶子子节点：按相似度排序，取 top beam_width
                scored = [
                    (
                        _cosine_sim(question_vec, _parse_vec(k["embedding"]))
                        if k["embedding"] else 0.0,
                        k
                    )
                    for k in kids
                ]
                scored.sort(key=lambda x: x[0], reverse=True)
                
                for _, k in scored[:beam_width]:
                    if k["fact_id"] is not None:
                        collected_fact_ids.append(k["fact_id"])
            else:
                # 非叶子子节点：继续搜索
                scored = []
                for k in kids:
                    if k["embedding"] is not None:
                        sim = _cosine_sim(question_vec, _parse_vec(k["embedding"]))
                    else:
                        sim = 0.0
                    scored.append((sim, k))
                
                scored.sort(key=lambda x: x[0], reverse=True)
                next_beam.extend(k for _, k in scored[:beam_width])
        
        beam = next_beam
    
    return collected_fact_ids
```

**光束搜索图解**:

```
实体: Caroline

根节点 (root)
  │
  ├── 子节点 A: Career (相似度: 0.9) ──┬── 叶子: Work history (0.85) ── fact_1
  │                                      ├── 叶子: Education (0.80) ── fact_2
  │                                      └── 叶子: Skills (0.75)
  │
  ├── 子节点 B: Personal Life (0.7) ────┼── 叶子: Family (0.9) ── fact_3
  │                                      └── 叶子: Hobbies (0.6)
  │
  └── 子节点 C: Relationships (0.5) ────
        (beam_width=3, 保留 top-3)

结果: [fact_1, fact_2, fact_3]
```

**为什么使用光束搜索**:
1. **层级导航**: 沿着实体树结构向下搜索
2. **相似度导向**: 根据与问题的相似度选择路径
3. **效率**: beam_width 限制搜索宽度，避免指数爆炸

### 3.6 Path D: BM25 解析查询（Tantivy 语法）

```python
# 使用 LLM 生成的 Tantivy 查询
path_d: list[RetrievedFact] = []
if q_bm25_parsed:
    path_d = _bm25_parsed_search_facts(q_bm25_parsed, task_id, db, limit=20)
```

**实现代码**:

```python
# membrain/retrieval/application/retrieval.py

def _bm25_parsed_search_facts(
    query: str,
    task_id: int,
    db: Session,
    limit: int = 20,
) -> list[RetrievedFact]:
    """Search facts using a Tantivy query string via pdb.parse()."""
    
    if not query:
        return []
    
    sql = sa_text("""
        SELECT id, text, pdb.score(id) AS score
        FROM facts
        WHERE id @@@ pdb.parse(:query, lenient => true)
          AND task_id = :task_id
          AND status = 'active'
        ORDER BY score DESC
        LIMIT :limit
    """)
    
    try:
        rows = db.execute(
            sql, {"query": query, "task_id": task_id, "limit": limit}
        ).fetchall()
    except Exception:
        log.debug("BM25 parsed fact search failed", exc_info=True)
        return []
    
    return [RetrievedFact(fact_id=r[0], text=r[1], source="bm25_parsed") for r in rows]
```

**Tantivy 查询示例**:
- 输入: `+caroline +picnic (friend outdoor summer)`
- 含义: "caroline" **必须出现**，且 "picnic" **必须出现**，同时至少一个同义词出现

**为什么使用 Tantivy 查询**:
1. **精确控制**: 支持 AND/OR 语义
2. **同义词扩展**: `(term1 term2)` 形式支持 OR 匹配
3. **强制实体**: `+entity` 确保核心实体必须出现

## 完整示例

### 输入

```python
question = "When did Caroline have a picnic?"
rewrite_vec = [0.1, -0.3, 0.5, ...]  # 嵌入向量
q_event = "What did Caroline do with friends outdoors?"
q_hyde = "Caroline and her friends had a picnic together."
q_bm25_kw = "Caroline picnic friends"
q_bm25_parsed = "+caroline +picnic (friend outdoor summer)"
```

### 执行结果

| 路径 | 方法 | 返回数量 | 示例事实 |
|------|------|----------|----------|
| Path A | BM25 | 20 | "Caroline had a picnic in July" |
| Path B | Embedding | 20 | "Picnic at Central Park" |
| Path B2 | HyDE | 20 | "Caroline organized a picnic" |
| Path B3 | Event | 15 | "Summer outdoor activity" |
| Path C | Tree | ~10 | "Career > Work" 相关 |
| Path D | Tantivy | 20 | "Caroline + picnic" 严格匹配 |

## 配置参数

```python
# membrain/config.py

QA_BM25_FACT_TOP_N = 20      # Path A 返回数量
QA_EMBED_FACT_TOP_N = 20     # Path B/B2 返回数量
QA_ENTITY_TOP_N = 5          # Path C 实体数量
QA_TREE_BEAM_WIDTH = 3       # Path C 光束宽度
QA_TREE_FACT_TOP_N = 20      # Path C 事实上限
```

## 总结

多路径检索的核心思想：

| 路径 | 方法 | 优势 | 劣势 |
|------|------|------|------|
| A | BM25 | 精确匹配关键词 | 无法处理同义词 |
| B | Embedding | 语义匹配 | 可能偏离关键词 |
| B2 | HyDE | 接近记忆形式 | 需要额外 LLM 调用 |
| B3 | Event | 处理时间问题 | 只针对特定类型 |
| C | Tree | 利用结构化知识 | 依赖实体树 |
| D | Tantivy | 精确 + 同义词 | 需要 LLM 生成 |

通过同时执行这六条路径，MemBrain 可以从不同角度捕捉相关信息，显著提高检索召回率。


## 7. 04 result merging

# Stage 4: 结果合并与去重 - Result Merging & Deduplication

## 概述

在执行六条检索路径后，会得到六个结果列表。这些列表中可能包含重复的事实（因为同一事实可能被多条路径检索到）。本阶段的目标是：

1. **合并所有路径的结果**
2. **按 fact_id 去重**
3. **构建统一的候选池（Pool）**

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L434-L455)

## 详细代码分析

### 4.1 收集排名列表

```python
# membrain/retrieval/application/retrieval.py

# ── 3. Dedup into pool ───────────────────────────────────────────────
# 收集每条路径的结果 ID 列表，用于后续融合
ranked_lists = [
    [f.fact_id for f in path_a],
    [f.fact_id for f in path_b],
    [f.fact_id for f in path_b2],
    [f.fact_id for f in path_b3],
    [f.fact_id for f in path_c],
    [f.fact_id for f in path_d],
]
```

**为什么需要收集排名列表**:

1. **用于 RRF 融合**: 需要知道每个事实在每条路径中的排名
2. **保留顺序信息**: 不仅知道哪些路径返回了该事实，还知道其排名

**示例**:
```python
ranked_lists = [
    [101, 203, 87, 45, ...],  # Path A
    [203, 101, 56, 78, ...],  # Path B
    [87, 203, 101, ...],       # Path B2
    [45, 67, 89, ...],         # Path B3
    [101, 203, ...],           # Path C
    [78, 56, 101, ...],        # Path D
]
```

### 4.2 去重合并

```python
# 使用字典保持首次出现的顺序
seen: dict[int, RetrievedFact] = {}

# 遍历所有路径的结果
for lst in (path_a, path_b, path_b2, path_b3, path_c, path_d):
    for f in lst:
        if f.fact_id not in seen:
            # 首次出现：添加到字典
            seen[f.fact_id] = f

# 转换为列表
pool = list(seen.values())
```

**为什么使用字典去重**:

1. **O(1) 查找**: 字典查找效率高
2. **保持顺序**: 字典保持插入顺序（Python 3.7+）
3. **保留首次出现**: 首次出现的事实对象被保留

**示例图解**:

```
Path A: [fact_1, fact_2, fact_3, fact_4, fact_5]
Path B: [fact_2, fact_1, fact_6, fact_7]
Path B2: [fact_3, fact_2, fact_1, fact_8]
Path B3: [fact_4, fact_9, fact_10]
Path C:  [fact_1, fact_5, fact_11]
Path D:  [fact_6, fact_2, fact_12]

合并去重后:
pool = [fact_1, fact_2, fact_3, fact_4, fact_5, fact_6, fact_7, fact_8, fact_9, fact_10, fact_11, fact_12]
```

### 4.3 空池检查

```python
if not pool:
    return _empty_result()
```

**空池定义**:

```python
def _empty_result() -> dict:
    return {
        "packed_context": "",
        "packed_token_count": 0,
        "fact_ids": [],
        "facts": [],
        "sessions": [],
        "raw_messages": [],
    }
```

**为什么需要空池检查**:

1. **避免后续错误**: 如果没有检索到任何事实，直接返回空结果
2. **快速路径**: 无需执行后续昂贵的后处理操作

## 完整示例

### 输入

假设六条路径返回以下结果：

```python
path_a = [
    RetrievedFact(fact_id=101, text="Caroline had a picnic", source="bm25"),
    RetrievedFact(fact_id=203, text="Picnic was at Central Park", source="bm25"),
    RetrievedFact(fact_id=87, text="Summer picnic event", source="bm25"),
]

path_b = [
    RetrievedFact(fact_id=203, text="Picnic was at Central Park", source="embed"),
    RetrievedFact(fact_id=101, text="Caroline had a picnic", source="embed"),
    RetrievedFact(fact_id=56, text="Friends attended the picnic", source="embed"),
]

path_b2 = [
    RetrievedFact(fact_id=87, text="Summer picnic event", source="embed"),
    RetrievedFact(fact_id=203, text="Picnic was at Central Park", source="embed"),
    RetrievedFact(fact_id=101, text="Caroline had a picnic", source="embed"),
]

path_b3 = [
    RetrievedFact(fact_id=45, text="Outdoor gathering", source="embed"),
]

path_c = [
    RetrievedFact(fact_id=101, text="Caroline had a picnic", source="tree"),
]

path_d = [
    RetrievedFact(fact_id=78, text="Caroline organized event", source="bm25_parsed"),
]
```

### 处理过程

```python
# Step 1: 收集排名列表
ranked_lists = [
    [101, 203, 87],           # Path A
    [203, 101, 56],           # Path B
    [87, 203, 101],           # Path B2
    [45],                     # Path B3
    [101],                    # Path C
    [78],                     # Path D
]

# Step 2: 去重合并
seen = {}

# 遍历 Path A
for f in path_a:
    seen[101] = fact_1
    seen[203] = fact_2
    seen[87] = fact_3

# 遍历 Path B
# fact_2, fact_1 已存在，跳过
# fact_6 (id=56) 不存在，添加
seen[56] = fact_6

# 遍历 Path B2
# fact_3, fact_2, fact_1 已存在，跳过

# 遍历 Path B3
# fact_4 (id=45) 不存在，添加
seen[45] = fact_4

# 遍历 Path C
# fact_1 已存在，跳过

# 遍历 Path D
# fact_5 (id=78) 不存在，添加
seen[78] = fact_5

# Step 3: 转换为列表（保持插入顺序）
pool = [fact_1, fact_2, fact_3, fact_6, fact_4, fact_5]
```

### 输出

```python
pool = [
    RetrievedFact(fact_id=101, text="Caroline had a picnic", source="bm25"),
    RetrievedFact(fact_id=203, text="Picnic was at Central Park", source="bm25"),
    RetrievedFact(fact_id=87, text="Summer picnic event", source="bm25"),
    RetrievedFact(fact_id=56, text="Friends attended the picnic", source="embed"),
    RetrievedFact(fact_id=45, text="Outdoor gathering", source="embed"),
    RetrievedFact(fact_id=78, text="Caroline organized event", source="bm25_parsed"),
]
```

## 关键数据结构

### RetrievedFact 类

```python
# membrain/retrieval/core/types.py

@dataclass
class RetrievedFact:
    fact_id: int
    text: str
    source: str  # "bm25" | "embed" | "tree" | "bm25_parsed"
    rerank_score: float = 0.0
    time_info: str = ""
    entity_ref: str = ""
    aspect_path: str = ""
    aspect_summary: str = ""
    session_number: int | None = None
```

**字段说明**:

| 字段 | 说明 |
|------|------|
| `fact_id` | 事实的唯一标识符 |
| `text` | 事实的文本内容 |
| `source` | 首次检索到该事实的路径 |
| `rerank_score` | 融合后的评分（后续阶段填充） |
| `time_info` | 时间信息（后续阶段填充） |
| `entity_ref` | 关联的实体引用（后续阶段填充） |
| `aspect_path` | 实体树路径（后续阶段填充） |
| `aspect_summary` | 方面摘要（后续阶段填充） |
| `session_number` | 所属会话编号（后续阶段填充） |

## 与后续阶段的关联

### 1. 排名列表用于 RRF 融合

```python
# 后续阶段使用 ranked_lists 进行 RRF 融合
ranked_lists = [
    [f.fact_id for f in path_a],
    [f.fact_id for f in path_b],
    # ...
]
```

### 2. Pool 继续后续处理

```python
# 后续阶段在 pool 基础上进行处理
pool = list(seen.values())

# 后处理
_inject_time_annotations(pool, db)
_inject_session_numbers(pool, db)

# 融合
if strategy == "rerank":
    round1_facts = _fuse_rerank(...)
else:
    _fuse_rrf(pool, ranked_lists)
```

## 总结

本阶段的核心逻辑:

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 收集排名列表 | 为 RRF 融合准备数据 |
| 2 | 遍历所有路径 | 合并六个结果集 |
| 3 | 字典去重 | 移除重复事实，保持首次出现 |
| 4 | 空池检查 | 快速路径，无结果时直接返回 |

**关键设计决策**:

1. **使用字典保持顺序**: Python 3.7+ 字典保持插入顺序
2. **保留首次出现**: 首次出现的事实被保留，后续重复被忽略
3. **保留原始 source**: 虽然可能有多条路径返回同一事实，但只保留第一次遇到的 source

这一阶段为后续的融合和排序奠定了基础。


## 8. 05 post processing

# Stage 5: 后处理 - Post-Processing

## 概述

后处理阶段是 MemBrain 搜索过程中至关重要的一步，它对合并后的候选池进行增强和优化，包括：

1. **时间注解注入**: 为每个事实添加时间信息
2. **会话编号注入**: 关联事实到具体会话
3. **Aspect 路径富化**: 添加实体树的层级上下文
4. **Aspect 级别去重**: 基于方面（Aspect）进行去重，避免信息冗余

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L457-L478)
- **时间注解**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L163-L197)
- **Aspect 处理**: [aspect_enrichment.py](file:///home/project/MemBrain/membrain/infra/retrieval/aspect_enrichment.py)

## 详细代码分析

### 5.1 时间注解注入

```python
# membrain/retrieval/application/retrieval.py

# ── 4. Post-processing ───────────────────────────────────────────────
_inject_time_annotations(pool, db)
```

**实现代码**:

```python
# membrain/retrieval/application/retrieval.py

def _inject_time_annotations(pool: list[RetrievedFact], db: Session) -> None:
    if not pool:
        return
    
    # 收集所有 fact_id
    fact_ids = [f.fact_id for f in pool]
    
    # 从 time_annotations 表获取时间信息
    rows = db.execute(
        sa_text(
            "SELECT fact_id, time_raw, time_resolved FROM time_annotations "
            "WHERE fact_id = ANY(:ids)"
        ),
        {"ids": fact_ids},
    ).fetchall()
    
    # 构建时间映射
    time_map: dict[int, str] = {}
    for fid, raw, resolved in rows:
        if not raw and not resolved:
            continue
        parts: list[str] = []
        if raw:
            parts.append(str(raw))
        if resolved:
            start, *rest = resolved.split("/", 1)
            end = rest[0] if rest else None
            parts.append(f"[{start}, {end}]" if end is not None else f"[{start}]")
        time_map[fid] = " ".join(parts)
    
    # 对于没有时间注解的事实，从 facts 表获取时间戳
    no_time_ids = [f.fact_id for f in pool if f.fact_id not in time_map]
    if no_time_ids:
        for r in db.execute(
            sa_text(
                "SELECT id, fact_ts FROM facts WHERE id = ANY(:ids) "
                "AND fact_ts IS NOT NULL"
            ),
            {"ids": no_time_ids},
        ).fetchall():
            time_map[r[0]] = f"[{r[1]}]"
    
    # 注入时间信息到事实对象
    for fact in pool:
        fact.time_info = time_map.get(fact.fact_id, "")
```

**为什么需要时间注解**:

1. **时间上下文**: 帮助 LLM 理解事实发生的时间背景
2. **排序依据**: 事实可以按时间顺序组织输出
3. **时态理解**: 区分过去、现在、未来的事件

**时间信息格式**:

```
示例 1: "yesterday [2023-05-07, ]"
示例 2: "last week [2023-06-02, 2023-06-08]"
示例 3: "now [2023-05-08T13:56:00Z, ]"
示例 4: "[2023-07-15]" (仅时间戳)
```

### 5.2 会话编号注入

```python
# membrain/retrieval/application/retrieval.py

_inject_session_numbers(pool, db)
```

**实现代码**:

```python
# membrain/retrieval/application/retrieval.py

def _inject_session_numbers(pool: list[RetrievedFact], db: Session) -> None:
    if not pool:
        return
    
    # 收集所有 fact_id
    fact_ids = [f.fact_id for f in pool]
    
    # 查询会话编号
    rows = db.execute(
        sa_text("SELECT id, session_number FROM facts WHERE id = ANY(:ids)"),
        {"ids": fact_ids},
    ).fetchall()
    
    # 构建映射
    sn_map = {r[0]: r[1] for r in rows if r[1] is not None}
    
    # 注入会话编号
    for fact in pool:
        fact.session_number = sn_map.get(fact.fact_id)
```

**为什么需要会话编号**:

1. **追踪来源**: 知道事实来自哪个会话
2. **会话检索**: 可以反向检索相关会话
3. **上下文理解**: 会话提供更广泛的对话上下文

### 5.3 Aspect 路径富化

```python
# membrain/retrieval/application/retrieval.py

all_fact_ids = [f.fact_id for f in pool]
aspect_infos = build_aspect_paths(all_fact_ids, task_id, db)

for fact in pool:
    info = aspect_infos.get(fact.fact_id)
    if info:
        fact.entity_ref = info.entity_ref
        fact.aspect_path = info.path
        fact.aspect_summary = info.leaf_desc
```

**实现代码**:

```python
# membrain/infra/retrieval/aspect_enrichment.py

def build_aspect_paths(
    fact_ids: list[int],
    task_id: int,
    db: Session,
) -> dict[int, AspectInfo]:
    """Look up entity tree ancestry for each fact."""
    
    if not fact_ids:
        return {}
    
    # SQL 查询：获取实体树层级信息
    sql = sa_text("""
        SELECT
            leaf.fact_id,
            e.canonical_ref,
            e.desc,
            parent.node_type  AS parent_type,
            parent.description AS parent_desc,
            gp.node_type       AS gp_type,
            gp.description     AS gp_desc
        FROM entity_tree_nodes leaf
        JOIN entity_tree_nodes parent ON parent.id = leaf.parent_id
        LEFT JOIN entity_tree_nodes gp ON gp.id = parent.parent_id
        JOIN entities e
          ON e.entity_id = leaf.entity_id
         AND e.task_id = :task_id
        WHERE leaf.fact_id = ANY(:fact_ids)
          AND leaf.task_id = :task_id
          AND leaf.node_type = 'leaf'
    """)
    
    rows = db.execute(sql, {"fact_ids": fact_ids, "task_id": task_id}).fetchall()
    
    # 构建 AspectInfo
    result: dict[int, AspectInfo] = {}
    
    for row in rows:
        fid = row[0]
        entity_ref = row[1] or ""
        entity_desc = row[2] or ""
        parent_type = row[3]
        parent_desc = row[4] or ""
        gp_type = row[5]
        gp_desc = row[6] or ""
        
        # 判断树结构层级
        if parent_type == "root":
            # 2层树: root → leaf
            path = parent_desc
            leaf_desc = parent_desc
            mid_desc = ""
        elif gp_type == "root":
            # 3层树: root → aspect → leaf
            path = f"{gp_desc} > {parent_desc}" if gp_desc else parent_desc
            leaf_desc = parent_desc
            mid_desc = gp_desc
        else:
            # 更深的树
            path = parent_desc
            leaf_desc = parent_desc
            mid_desc = gp_desc
        
        # 构建去重键
        leaf_key = f"{entity_ref}::{leaf_desc}"
        mid_key = f"{entity_ref}::{mid_desc}" if mid_desc else leaf_key
        
        result[fid] = AspectInfo(
            entity_ref=entity_ref,
            entity_desc=entity_desc,
            leaf_desc=leaf_desc,
            mid_desc=mid_desc,
            path=path,
            leaf_key=leaf_key,
            mid_key=mid_key,
        )
    
    return result
```

**AspectInfo 数据结构**:

```python
@dataclass
class AspectInfo:
    entity_ref: str      # 实体规范名称，如 "Caroline"
    entity_desc: str     # 实体描述，如 "user's mother"
    leaf_desc: str       # 叶子方面描述
    mid_desc: str       # 中间方面描述
    path: str           # 路径，如 "Career > Work history"
    leaf_key: str       # 去重键: "entity::leaf_desc"
    mid_key: str        # 去重键: "entity::mid_desc"
```

**实体树结构示例**:

```
实体: Caroline (user's mother)
│
├── Root: Person
│     │
│     ├── Aspect: Career
│     │     │
│     │     └── Leaf: Work history ── fact_1
│     │           └── Leaf: Education ── fact_2
│     │
│     └── Aspect: Personal Life
│           │
│           └── Leaf: Family ── fact_3
│                 └── Leaf: Hobbies ── fact_4
```

**Aspect 路径示例**:

| 事实 ID | 实体 | Path | Leaf | 格式 |
|---------|------|------|------|------|
| 101 | Caroline | Career > Work history | Work history | "Career > Work history" |
| 102 | Caroline | Career > Education | Education | "Career > Education" |
| 103 | Caroline | Personal Life > Family | Family | "Personal Life > Family" |

### 5.4 Aspect 级别去重

```python
# membrain/retrieval/application/retrieval.py

# 后续对 Path C 应用去重（在 entity_tree_search 之后）
if path_c:
    c_ids = [f.fact_id for f in path_c]
    c_aspects = build_aspect_paths(c_ids, task_id, db)
    kept = set(aspect_dedup(c_ids, c_aspects))
    path_c = [f for f in path_c if f.fact_id in kept]
```

**实现代码**:

```python
# membrain/infra/retrieval/aspect_enrichment.py

def aspect_dedup(
    fact_ids_ordered: list[int],
    aspect_infos: dict[int, AspectInfo],
    max_per_leaf: int = settings.QA_MAX_PER_LEAF_ASPECT,  # 默认 3
    max_per_mid: int = settings.QA_MAX_PER_MID_ASPECT,     # 默认 8
    protected_ids: set[int] | None = None,
) -> list[int]:
    """Filter fact IDs by aspect-level caps, preserving input order."""
    
    leaf_counts: dict[str, int] = defaultdict(int)
    mid_counts: dict[str, int] = defaultdict(int)
    kept: list[int] = []
    
    for fid in fact_ids_ordered:
        info = aspect_infos.get(fid)
        
        # 无 Aspect 信息：直接保留
        if info is None:
            kept.append(fid)
            continue
        
        # 受保护的事实（如 BM25 关键词命中）：直接保留
        if protected_ids and fid in protected_ids:
            kept.append(fid)
            continue
        
        # 检查叶子方面限制
        if leaf_counts[info.leaf_key] >= max_per_leaf:
            continue
        
        # 检查中间方面限制
        if mid_counts[info.mid_key] >= max_per_mid:
            continue
        
        # 计数并保留
        leaf_counts[info.leaf_key] += 1
        mid_counts[info.mid_key] += 1
        kept.append(fid)
    
    return kept
```

**去重逻辑图解**:

```
假设: max_per_leaf = 3, max_per_mid = 8

输入 fact_ids (按顺序):
  - fact_101: leaf="Work history", mid="Career"
  - fact_102: leaf="Work history", mid="Career"  ✓ 计数=2
  - fact_103: leaf="Work history", mid="Career"  ✓ 计数=3 (达到上限)
  - fact_104: leaf="Work history", mid="Career"  ✗ 超过 leaf 上限，跳过
  - fact_105: leaf="Education", mid="Career"     ✓ 新 leaf，计数=1
  - fact_106: leaf="Family", mid="Personal Life" ✓ 新 mid，计数=1

输出: [fact_101, fact_102, fact_103, fact_105, fact_106]
```

**为什么需要 Aspect 去重**:

1. **避免冗余**: 同一方面的多个事实可能内容重复
2. **多样性**: 确保覆盖多个不同方面
3. **可控性**: 通过参数控制去重粒度

## 完整示例

### 输入

```python
pool = [
    RetrievedFact(fact_id=101, text="Caroline worked at Google"),
    RetrievedFact(fact_id=102, text="Caroline was a software engineer at Google"),
    RetrievedFact(fact_id=103, text="Caroline's job at Google lasted 3 years"),
    RetrievedFact(fact_id=104, text="Caroline received promotion at Google"),
    RetrievedFact(fact_id=105, text="Caroline studied at MIT"),
    RetrievedFact(fact_id=106, text="Caroline's mother is Mary"),
]
```

### 处理过程

```
Step 1: 注入时间注解
  fact_101: time_info = "[2023-01-15]"
  fact_102: time_info = "[2023-01-15]"
  fact_103: time_info = "[2023-01-15]"
  fact_104: time_info = "[2023-06-01]"
  fact_105: time_info = "[2020-09-01]"
  fact_106: time_info = ""

Step 2: 注入会话编号
  fact_101: session_number = 1
  fact_102: session_number = 1
  fact_103: session_number = 1
  fact_104: session_number = 2
  fact_105: session_number = 3
  fact_106: session_number = 4

Step 3: 注入 Aspect 信息
  fact_101: entity_ref="Caroline", aspect_path="Career > Work", aspect_summary="Work"
  fact_102: entity_ref="Caroline", aspect_path="Career > Work", aspect_summary="Work"
  fact_103: entity_ref="Caroline", aspect_path="Career > Work", aspect_summary="Work"
  fact_104: entity_ref="Caroline", aspect_path="Career > Work", aspect_summary="Work"
  fact_105: entity_ref="Caroline", aspect_path="Career > Education", aspect_summary="Education"
  fact_106: entity_ref="Caroline", aspect_path="Personal > Family", aspect_summary="Family"

Step 4: Aspect 去重 (max_per_leaf=3)
  保留: fact_101, fact_102, fact_103 (Work leaf 达到3个)
  跳过: fact_104 (Work leaf 超过3个)
  保留: fact_105 (Education, 新 leaf)
  保留: fact_106 (Family, 新 mid)
```

### 输出

```python
pool = [
    RetrievedFact(
        fact_id=101, 
        text="Caroline worked at Google",
        time_info="[2023-01-15]",
        session_number=1,
        entity_ref="Caroline",
        aspect_path="Career > Work",
        aspect_summary="Work"
    ),
    # ... 类似处理其他事实
]
```

## 配置参数

```python
# membrain/config.py

QA_MAX_PER_LEAF_ASPECT = 3   # 每个叶子方面最多保留3个事实
QA_MAX_PER_MID_ASPECT = 8    # 每个中间方面最多保留8个事实
```

## 总结

后处理阶段的核心目标是为候选池中的事实添加丰富的上下文信息：

| 处理步骤 | 操作 | 目的 |
|----------|------|------|
| 时间注解 | 从数据库获取时间信息 | 提供时间上下文 |
| 会话编号 | 关联事实到会话 | 追踪来源 |
| Aspect 富化 | 添加实体树路径 | 结构化上下文 |
| Aspect 去重 | 限制每方面数量 | 避免冗余 |

**关键设计决策**:

1. **多层次去重**: 叶子级别和中间级别双重限制
2. **信息丰富化**: 从简单的事实文本到包含时间、实体、方面信息的复合结构
3. **保护机制**: 允许特定事实（如 BM25 关键词命中）绕过去重限制

这一阶段为后续的融合和排序提供了更丰富的上下文信息。


## 9. 06 result fusion

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


## 10. 07 agentic round2

# Stage 7: Agentic Round 2 - 反思式检索增强

## 概述

Agentic Round 2（反思式检索增强）是 MemBrain 搜索过程中的可选增强阶段，仅在 `mode="reflect"` 模式下启用。

这一阶段的核心思想是让系统具备"自我反思"能力：
1. 分析第一轮检索结果是否足够回答问题
2. 如果不够，生成针对性的补充查询
3. 使用补充查询进行二次检索

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L489-L518)
- **反思函数**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L220-L268)

## 详细代码分析

### 7.1 入口条件判断

```python
# membrain/retrieval/application/retrieval.py

# ── 6. Agentic round 2 (reflect mode only) ────────────────────────────
if mode == "reflect" and http_client:
    # 执行 Agentic Round 2
    refined_queries = _reflect_and_refine(question, round1_facts, http_client)
    # ...
```

**为什么需要条件判断**:

1. **仅 reflect 模式**: 只有显式设置 `mode="reflect"` 才执行
2. **需要 HTTP 客户端**: 需要调用 LLM 进行反思
3. **可选阶段**: 不是默认行为，不影响基础功能

### 7.2 反思与精炼函数

```python
# membrain/retrieval/application/retrieval.py

_REFLECT_SYSTEM = """\
You are analysing retrieved memory facts to identify what is still missing.
Given a question and the top retrieved facts, output ONLY valid JSON:
{
  "sufficient": true/false,
  "refined_queries": ["query1", "query2"]
}
- Set sufficient=true if the facts clearly contain enough to answer the question.
- Otherwise provide 1-2 targeted queries that search for the missing information.
  Focus on specific entity names or events not yet found. Max 20 words each.
No explanation outside the JSON.\
"""


def _reflect_and_refine(
    question: str,
    facts: list[RetrievedFact],
    http_client: httpx.Client,
) -> list[str]:
    """Return 0-2 refined queries if round-1 facts are insufficient."""
    
    # 构造上下文：前 20 个事实的文本
    facts_text = "\n".join(
        f"- {f.text}" + (f" (time: {f.time_info})" if f.time_info else "")
        for f in facts[:20]
    )
    
    try:
        # 调用 LLM 进行反思
        resp = http_client.post(
            f"{settings.LLM_API_URL.rstrip('/')}/chat/completions",
            json={
                "model": settings.QA_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": _REFLECT_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Retrieved facts ({len(facts)}):\n{facts_text}"
                        ),
                    },
                ],
                "max_tokens": 150,
                "temperature": 0.0,
            },
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            timeout=20.0,
        )
        
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        
        # 解析 JSON（处理 markdown 代码块）
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        
        import json
        data = json.loads(raw)
        
        # 判断是否足够
        if data.get("sufficient"):
            return []
        
        # 返回精炼查询
        queries = data.get("refined_queries", [])
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:2]  # 最多 2 个
        
    except Exception:
        log.debug("reflection failed", exc_info=True)
    
    return []
```

### 7.3 反思提示词设计

**系统提示词分析**:

```
你正在分析已检索的记忆事实，以识别仍然缺失的信息。

给定一个问题和已检索的事实，输出 ONLY 有效 JSON：
{
  "sufficient": true/false,
  "refined_queries": ["query1", "query2"]
}

- 如果事实明显包含足够信息回答问题，设置 sufficient=true
- 否则提供 1-2 个有针对性的查询，搜索尚未找到的缺失信息
  聚焦于尚未找到的特定实体名称或事件
  每个查询最多 20 个词

不要在 JSON 之外提供解释。
```

**关键设计**:

1. **JSON 输出格式**: 确保结构化解析
2. **明确的判断标准**: "sufficient" 字段
3. **最多 2 个查询**: 避免过多补充查询导致噪声
4. **20 词限制**: 确保查询简洁精确

### 7.4 使用精炼查询进行补充检索

```python
if mode == "reflect" and http_client:
    refined_queries = _reflect_and_refine(question, round1_facts, http_client)
    
    if refined_queries:
        extra_pool: list[RetrievedFact] = []
        seen_ids = {f.fact_id for f in pool}  # 记录已有事实 ID
        
        # 对每个精炼查询进行检索
        for rq in refined_queries:
            # BM25 搜索
            for f in bm25_search_facts(rq, task_id, db, limit=15):
                if f.fact_id not in seen_ids:
                    extra_pool.append(f)
                    seen_ids.add(f.fact_id)
            
            # Embedding 搜索
            try:
                rq_vec = embed_client.embed_single(rq)
                if rq_vec:
                    for f in embedding_search_facts(rq_vec, task_id, db, limit=15):
                        if f.fact_id not in seen_ids:
                            extra_pool.append(f)
                            seen_ids.add(f.fact_id)
            except Exception:
                pass
        
        # 后处理补充检索结果
        if extra_pool:
            _inject_time_annotations(extra_pool, db)
            _inject_session_numbers(extra_pool, db)
            
            # 构建 Aspect 信息
            extra_ids = [f.fact_id for f in extra_pool]
            extra_aspects = build_aspect_paths(extra_ids, task_id, db)
            aspect_infos.update(extra_aspects)
            
            for fact in extra_pool:
                info = extra_aspects.get(fact.fact_id)
                if info:
                    fact.entity_ref = info.entity_ref
                    fact.aspect_path = info.path
                    fact.aspect_summary = info.leaf_desc
            
            # 合并到最终结果
            round1_facts = round1_facts + extra_pool
            log.debug("agentic round 2: +%d facts", len(extra_pool))
```

**补充检索策略**:

1. **双重检索**: 同时使用 BM25 和 Embedding 搜索
2. **去重**: 避免重复添加已有的事实
3. **后处理**: 同样应用时间、Aspect 等处理

## 完整示例

### 输入

```python
question = "When did Caroline have a picnic with her friends?"
round1_facts = [
    RetrievedFact(fact_id=101, text="Caroline had a picnic in July", time_info="[2023-07-15]"),
    RetrievedFact(fact_id=102, text="The picnic was at Central Park", time_info="[2023-07-15]"),
    RetrievedFact(fact_id=103, text="Caroline's friends attended", time_info="[2023-07-15]"),
]
```

### 反思过程

**LLM 输入**:
```
Question: When did Caroline have a picnic with her friends?

Retrieved facts (3):
- Caroline had a picnic in July (time: [2023-07-15])
- The picnic was at Central Park (time: [2023-07-15])
- Caroline's friends attended (time: [2023-07-15])
```

**可能输出**:
```json
{
  "sufficient": false,
  "refined_queries": [
    "picnic food menu",
    "weather at picnic"
  ]
}
```

### 补充检索

```python
# 精炼查询
refined_queries = ["picnic food menu", "weather at picnic"]

# 补充检索结果
extra_pool = [
    RetrievedFact(fact_id=201, text="They served sandwiches and lemonade"),
    RetrievedFact(fact_id=202, text="The weather was sunny and warm"),
    RetrievedFact(fact_id=203, text="It was a perfect summer day"),
]

# 合并结果
round1_facts = round1_facts + extra_pool

# 最终结果
round1_facts = [
    # 原有结果
    RetrievedFact(fact_id=101, text="Caroline had a picnic in July", ...),
    RetrievedFact(fact_id=102, text="The picnic was at Central Park", ...),
    RetrievedFact(fact_id=103, text="Caroline's friends attended", ...),
    # 补充结果
    RetrievedFact(fact_id=201, text="They served sandwiches and lemonade", ...),
    RetrievedFact(fact_id=202, text="The weather was sunny and warm", ...),
    RetrievedFact(fact_id=203, text="It was a perfect summer day", ...),
]
```

### 另一种可能：足够的情况

如果第一轮检索已经足够，LLM 可能返回：

```json
{
  "sufficient": true,
  "refined_queries": []
}
```

在这种情况下，不会进行补充检索，直接使用第一轮结果。

## 为什么需要 Agentic Round 2

### 问题背景

1. **静态检索的局限性**: 第一轮检索使用固定的查询，无法动态调整
2. **长尾信息**: 某些相关信息可能在第一轮未被捕获
3. **查询意图变化**: 用户问题可能包含多个子意图

### 解决方案

1. **自我评估**: 让系统评估检索结果是否足够
2. **针对性补充**: 生成精确的补充查询
3. **迭代改进**: 类似人类逐步完善搜索的过程

### 适用场景

| 场景 | 是否需要 Round 2 |
|------|------------------|
| 简单事实查询 | ❌ 可能不需要 |
| 复杂问题（多个子问题） | ✅ 需要 |
| 信息不完整 | ✅ 需要 |
| 高召回要求 | ✅ 需要 |

### 权衡

**优点**:
- 提高召回率
- 动态适应查询需求
- 接近人类搜索行为

**缺点**:
- 增加延迟（需要额外 LLM 调用）
- 增加 API 成本
- 可能引入噪声

## 配置参数

```python
# 无特定配置参数
# 由 mode="reflect" 控制是否启用
```

## 错误处理

```python
# 1. LLM 调用失败
except Exception:
    log.debug("reflection failed", exc_info=True)
    return []  # 返回空列表，跳过补充检索

# 2. JSON 解析失败
except json.JSONDecodeError:
    log.debug("reflection JSON parse failed", exc_info=True)
    return []

# 3. Embedding 失败
except Exception:
    pass  # 继续使用其他检索结果
```

## 总结

Agentic Round 2 是 MemBrain 搜索过程的智能增强机制：

| 组件 | 说明 |
|------|------|
| **反思阶段** | 分析第一轮结果是否足够 |
| **查询精炼** | 生成 1-2 个补充查询 |
| **补充检索** | 使用 BM25 + Embedding 搜索 |
| **结果合并** | 将补充结果加入最终结果 |

**关键设计**:

1. **最大 2 个查询**: 避免过多补充查询
2. **20 词限制**: 确保查询简洁
3. **双重检索**: BM25 + Embedding 提高召回
4. **容错机制**: 任何步骤失败都优雅降级

这一阶段使 MemBrain 能够动态适应复杂查询，显著提高检索召回率。


## 11. 08 entity resolution

# Stage 8: 实体引用解析 - Entity Reference Resolution

## 概述

实体引用解析是 MemBrain 搜索过程中将记忆文本转换为可读格式的关键步骤。在记忆存储中，实体通常以简写形式（如 `[Caroline]`）引用，但 LLM 需要知道这些引用的具体含义。

本阶段的目标是将方括号形式的实体引用（如 `[Caroline]`）替换为实体的完整信息（如 `Caroline (user's mother)`）。

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L521-L523)
- **解析函数**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L25-L45)

## 详细代码分析

### 8.1 入口

```python
# membrain/retrieval/application/retrieval.py

# Resolve entity bracket refs on the final selected facts so that the
# first occurrence in the output gets the description appended.
_resolve_pool_entity_refs(round1_facts, db)
```

**注释说明**:
- 只在第一次出现时附加描述
- 避免重复显示相同的描述

### 8.2 实体引用解析函数

```python
# membrain/infra/retrieval/fact_retrieval.py

# 正则表达式：匹配 [alias] 形式的引用
_ENTITY_BRACKET_RE = re.compile(r"\[([^\]:#]+)\]")


def _resolve_entity_refs(
    text: str,
    ref_map: dict[str, str],  # alias → canonical
) -> str:
    """Replace [alias] entity bracket refs with canonical names.

    Leaves time tokens ([word::DATE], [2023-05-07]) untouched.
    """

    def _replace(m: re.Match) -> str:
        alias = m.group(1)
        canonical = ref_map.get(alias)
        if canonical is None:
            return m.group(0)  # 保持原样
        return canonical

    return _ENTITY_BRACKET_RE.sub(_replace, text)
```

**正则表达式解析**:

```python
_ENTITY_BRACKET_RE = re.compile(r"\[([^\]:#]+\]")
```

- `\[` - 匹配左方括号
- `([^\]:#]+)` - 捕获组，匹配非 `]`, `:`, `#` 的字符（一个或多个）
- `\]` - 匹配右方括号

**不匹配的例外**:
- `[word::DATE]` - 包含 `:`，不匹配
- `[2023-05-07]` - 包含 `-`，不匹配
- `[Caroline]` - 匹配 ✓

### 8.3 批量解析函数

```python
# membrain/retrieval/application/retrieval.py

def _resolve_pool_entity_refs(
    pool: list[RetrievedFact],
    db: Session,
) -> None:
    # 1. 收集所有实体引用
    all_aliases: set[str] = set()
    for fact in pool:
        for alias in _ENTITY_BRACKET_RE.findall(fact.text):
            all_aliases.add(alias)
    
    if not all_aliases:
        return
    
    # 2. 查询数据库获取规范名称
    rows = db.execute(
        sa_text("""
            SELECT DISTINCT ON (fr.alias_text)
                   fr.alias_text, e.canonical_ref
            FROM fact_refs fr
            JOIN entities e ON e.entity_id = fr.entity_id
            WHERE fr.alias_text = ANY(:texts)
            ORDER BY fr.alias_text
        """),
        {"texts": list(all_aliases)},
    ).fetchall()
    
    # 3. 构建别名到规范名称的映射
    alias_canonical = {r[0]: r[1] for r in rows}
    
    if not alias_canonical:
        return
    
    # 4. 替换每个事实中的引用
    for fact in pool:
        fact.text = _resolve_entity_refs(fact.text, alias_canonical)
```

**数据库查询解释**:

```sql
SELECT DISTINCT ON (fr.alias_text)
       fr.alias_text, e.canonical_ref
FROM fact_refs fr
JOIN entities e ON e.entity_id = fr.entity_id
WHERE fr.alias_text = ANY(:texts)
ORDER BY fr.alias_text
```

- `fact_refs` 表：存储实体别名与实体的关联
- `entities` 表：存储实体的规范名称和描述
- `DISTINCT ON`: 对每个别名只返回一个结果

## 完整示例

### 输入

```python
# 事实文本
facts_texts = [
    "Caroline [Caroline] went to Paris with her friends.",
    "She met [John] at the café.",
    "The picnic happened on [2023-07-15].",
    "[Caroline] brought sandwiches for everyone.",
]

# 数据库中的别名映射
alias_canonical = {
    "Caroline": "Caroline (user's mother)",
    "John": "John (Caroline's colleague)",
}
```

### 处理过程

```
Step 1: 收集所有别名
  all_aliases = {"Caroline", "John"}

Step 2: 查询数据库
  rows = [
    ("Caroline", "Caroline (user's mother)"),
    ("John", "John (Caroline's colleague)"),
  ]

Step 3: 构建映射
  alias_canonical = {
    "Caroline": "Caroline (user's mother)",
    "John": "John (Caroline's colleague)",
  }

Step 4: 替换引用
  - "Caroline [Caroline] went to Paris..."
    → "Caroline (user's mother) went to Paris..."
  
  - "She met [John] at the café."
    → "She met John (Caroline's colleague) at the café."
  
  - "The picnic happened on [2023-07-15]."
    → "The picnic happened on [2023-07-15]." (不变，日期格式不匹配)
  
  - "[Caroline] brought sandwiches..."
    → "Caroline (user's mother) brought sandwiches..."
```

### 输出

```python
resolved_texts = [
    "Caroline (user's mother) went to Paris with her friends.",
    "She met John (Caroline's colleague) at the café.",
    "The picnic happened on [2023-07-15].",
    "Caroline (user's mother) brought sandwiches for everyone.",
]
```

## 为什么需要实体引用解析

### 1. 记忆存储的简化表示

在记忆提取阶段，实体以简写形式存储：
- 节省存储空间
- 便于批量处理
- 避免重复存储实体信息

### 2. LLM 需要完整信息

LLM 在生成答案时需要知道：
- 实体是谁
- 实体的背景信息
- 实体之间的关系

### 3. 避免重复描述

MemBrain 的设计只在第一次出现时显示完整描述：
- 第一幕: `[Caroline]` → `Caroline (user's mother)`
- 后续出现: 保持简洁，不重复显示描述

## 数据模型

### Entity 表结构

```sql
CREATE TABLE entities (
    entity_id VARCHAR PRIMARY KEY,
    task_id INTEGER NOT NULL,
    canonical_ref VARCHAR NOT NULL,  -- 规范名称，如 "Caroline"
    "desc" TEXT,                      -- 描述，如 "user's mother"
    desc_embedding VECTOR,            -- 描述的嵌入向量
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### FactRef 表结构

```python
class FactRefModel:
    id: Integer
    entity_id: String      -- 关联到 entities 表
    alias_text: String     -- 别名文本
    fact_id: Integer       -- 关联到 facts 表
```

### 引用格式

| 格式 | 示例 | 说明 |
|------|------|------|
| 实体引用 | `[Caroline]` | 需要解析 |
| 时间引用 | `[2023-07-15]` | 日期，不解析 |
| 时间标签 | `[yesterday::DATE]` | 相对时间，不解析 |

## 错误处理

```python
# 1. 没有别名
if not all_aliases:
    return  # 提前返回

# 2. 数据库没有匹配
if not alias_canonical:
    return  # 提前返回

# 3. 找不到的别名
def _replace(m: re.Match) -> str:
    alias = m.group(1)
    canonical = ref_map.get(alias)
    if canonical is None:
        return m.group(0)  # 保持原样
    return canonical
```

## 性能优化

### 1. 批量查询

```python
# 一次查询获取所有别名
rows = db.execute(
    sa_text("""...""")
    {"texts": list(all_aliases)},
).fetchall()
```

### 2. 提前返回

```python
if not all_aliases:
    return

if not alias_canonical:
    return
```

### 3. 正则表达式预编译

```python
_ENTITY_BRACKET_RE = re.compile(r"\[([^\]:#]+\]")
```

## 总结

实体引用解析阶段的核心功能：

| 功能 | 说明 |
|------|------|
| **识别引用** | 使用正则表达式识别 `[alias]` 形式 |
| **批量查询** | 一次性获取所有别名对应的规范名称 |
| **替换文本** | 将简写替换为完整描述 |
| **保留特殊格式** | 跳过时间标记等不需要解析的内容 |

**关键设计决策**:

1. **只解析实体引用**: 时间标记 `[DATE]` 保持原样
2. **只替换存在的别名**: 数据库中没有的别名保持原样
3. **支持重复出现**: 同一事实中多次出现的同一别名都会被替换

这一阶段确保了 LLM 在生成答案时能够理解实体的具体含义，提供更准确的响应。


## 12. 09 session retrieval

# Stage 9: 会话检索 - Session Retrieval

## 概述

会话检索是 MemBrain 搜索过程中获取对话历史上下文的关键步骤。除了检索单个事实外，系统还会检索相关的会话摘要（Session Summaries），为 LLM 提供更广泛的对话背景。

会话检索包含三种策略：
1. **BM25 会话搜索**: 使用关键词在会话摘要中搜索
2. **解析查询会话搜索**: 使用 Tantivy 查询搜索
3. **事实聚合评分**: 根据检索到的事实所属的会话进行聚合评分

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L525-L541)
- **会话搜索**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L328-L418)

## 详细代码分析

### 9.1 入口代码

```python
# membrain/retrieval/application/retrieval.py

# ── 7. Session retrieval ──────────────────────────────────────────────
seen_sess: dict[int, RetrievedSession] = {}

# 策略 1: BM25 会话搜索
for sq in [bm25_query, q_hyde, question]:
    if not sq:
        continue
    for s in bm25_search_sessions(sq, task_id, db, limit=8):
        if s.session_id not in seen_sess:
            seen_sess[s.session_id] = s

# 策略 2: 解析查询会话搜索
if q_bm25_parsed:
    for s in _bm25_parsed_search_sessions(q_bm25_parsed, task_id, db, limit=5):
        if s.session_id not in seen_sess:
            seen_sess[s.session_id] = s

# 策略 3: 事实聚合评分
for s in retrieve_sessions(question, task_id, db, round1_facts, limit=6):
    if s.session_id not in seen_sess:
        seen_sess[s.session_id] = s

# 排序并截断
sessions = sorted(seen_sess.values(), key=lambda s: s.score, reverse=True)[:10]
```

### 9.2 BM25 会话搜索

```python
# membrain/infra/retrieval/fact_retrieval.py

def bm25_search_sessions(
    query: str,
    task_id: int,
    db: Session,
    limit: int = settings.QA_SESSION_BM25_TOP_N,  # 默认 10
) -> list[RetrievedSession]:
    """BM25 search on session_summaries.content."""
    
    safe_query = _sanitize_bm25_query(query)
    if not safe_query:
        return []
    
    sql = sa_text("""
        SELECT ss.id, ss.session_id, ss.subject, ss.content, 
               pdb.score(ss.id) AS score,
               cs.session_number
        FROM session_summaries ss
        LEFT JOIN chat_sessions cs ON cs.id = ss.session_id
        WHERE ss.content ||| :query
          AND ss.task_id = :task_id
        ORDER BY score DESC
        LIMIT :limit
    """)
    
    try:
        rows = db.execute(
            sql,
            {"query": safe_query, "task_id": task_id, "limit": limit},
        ).fetchall()
    except Exception:
        return []
    
    return [
        RetrievedSession(
            session_summary_id=r[0],
            session_id=r[1],
            subject=r[2],
            content=r[3],
            score=r[4],
            source="bm25",
            session_number=r[5],
        )
        for r in rows
    ]
```

**为什么需要会话搜索**:

1. **更广泛的上下文**: 事实是离散的，会话提供连续的历史
2. **全局信息**: 某些信息（如会话主题）不在单个事实中
3. **时间顺序**: 会话按时间顺序组织

### 9.3 解析查询会话搜索

```python
# membrain/retrieval/application/retrieval.py

def _bm25_parsed_search_sessions(
    query: str,
    task_id: int,
    db: Session,
    limit: int = 5,
) -> list[RetrievedSession]:
    """Search session summaries using a Tantivy query string."""
    
    if not query:
        return []
    
    sql = sa_text("""
        SELECT ss.id, ss.session_id, ss.subject, ss.content,
               pdb.score(ss.id) AS score, cs.session_number
        FROM session_summaries ss
        LEFT JOIN chat_sessions cs ON cs.id = ss.session_id
        WHERE ss.id @@@ pdb.parse(:query, lenient => true)
          AND ss.task_id = :task_id
        ORDER BY score DESC
        LIMIT :limit
    """)
    
    try:
        rows = db.execute(
            sql, {"query": query, "task_id": task_id, "limit": limit}
        ).fetchall()
    except Exception:
        return []
    
    return [
        RetrievedSession(
            session_summary_id=r[0],
            session_id=r[1],
            subject=r[2] or "",
            content=r[3],
            score=r[4],
            source="bm25_parsed",
            session_number=r[5],
        )
        for r in rows
    ]
```

### 9.4 事实聚合评分

```python
# membrain/infra/retrieval/fact_retrieval.py

def retrieve_sessions(
    question: str,
    task_id: int,
    db: Session,
    facts: list[RetrievedFact],
    limit: int = settings.QA_SESSION_FINAL_TOP_N,  # 默认 5
) -> list[RetrievedSession]:
    """Retrieve session summaries via BM25 + fact-aggregation."""
    
    # BM25 搜索
    path_bm25 = bm25_search_sessions(question, task_id, db)
    
    # 事实聚合搜索
    path_agg = aggregate_session_scores(facts, task_id, db, limit=limit)
    
    # 去重合并（agg 优先）
    seen: dict[int, RetrievedSession] = {}
    for s in path_agg + path_bm25:
        if s.session_id not in seen:
            seen[s.session_id] = s
    
    pool = sorted(seen.values(), key=lambda s: s.score, reverse=True)
    return pool[:limit]
```

### 9.5 事实聚合评分实现

```python
# membrain/infra/retrieval/fact_retrieval.py

def aggregate_session_scores(
    facts: list[RetrievedFact],
    task_id: int,
    db: Session,
    limit: int = settings.QA_SESSION_FINAL_TOP_N,  # 默认 5
) -> list[RetrievedSession]:
    """Score sessions by aggregating rerank scores of their constituent facts."""
    
    if not facts:
        return []
    
    fact_ids = [f.fact_id for f in facts]
    score_by_fid = {f.fact_id: f.rerank_score for f in facts}
    
    # 获取事实对应的会话编号
    mapping_sql = sa_text("""
        SELECT id, session_number
        FROM facts
        WHERE id = ANY(:fact_ids)
          AND session_number IS NOT NULL
          AND status = 'active'
    """)
    
    mapping_rows = db.execute(
        mapping_sql,
        {"fact_ids": fact_ids},
    ).fetchall()
    
    if not mapping_rows:
        return []
    
    # 聚合分数
    sess_score: dict[int, float] = defaultdict(float)
    sess_count: dict[int, int] = defaultdict(int)
    
    for fid, sn in mapping_rows:
        sess_score[sn] += score_by_fid.get(fid, 0.0)
        sess_count[sn] += 1
    
    # 排序并获取 top 会话
    ranked_sessions = sorted(sess_score.items(), key=lambda x: x[1], reverse=True)
    top_sess_nums = [sn for sn, _ in ranked_sessions[: limit * 2]]
    
    # 获取会话详情
    rows = (
        db.query(
            SessionSummaryModel.id,
            SessionSummaryModel.session_id,
            SessionSummaryModel.subject,
            SessionSummaryModel.content,
            ChatSessionModel.session_number,
        )
        .join(ChatSessionModel, SessionSummaryModel.session_id == ChatSessionModel.id)
        .filter(
            SessionSummaryModel.task_id == task_id,
            ChatSessionModel.session_number.in_(top_sess_nums),
        )
        .all()
    )
    
    results = []
    for r in rows:
        sn = r[4]
        results.append(
            RetrievedSession(
                session_summary_id=r[0],
                session_id=r[1],
                subject=r[2],
                content=r[3],
                score=sess_score.get(sn, 0.0),
                source="fact_agg",
                contributing_facts=sess_count.get(sn, 0),
                session_number=sn,
            )
        )
    
    results.sort(key=lambda s: s.score, reverse=True)
    return results[:limit]
```

**事实聚合评分原理**:

```
会话 1: fact_101 (score=0.9), fact_102 (score=0.8)
        → 聚合分数 = 0.9 + 0.8 = 1.7
        → 贡献事实数 = 2

会话 2: fact_201 (score=0.7)
        → 聚合分数 = 0.7
        → 贡献事实数 = 1

会话 3: fact_301 (score=0.6), fact_302 (score=0.5), fact_303 (score=0.4)
        → 聚合分数 = 0.6 + 0.5 + 0.4 = 1.5
        → 贡献事实数 = 3
```

## RetrievedSession 数据结构

```python
@dataclass
class RetrievedSession:
    session_summary_id: int
    session_id: int
    subject: str
    content: str
    score: float
    source: str  # "bm25" | "bm25_parsed" | "fact_agg"
    contributing_facts: int = 0
    session_number: int | None = None
```

## 完整示例

### 输入

```python
# 查询
bm25_query = "caroline picnic"
q_hyde = "Caroline and her friends had a picnic together."
question = "When did Caroline have a picnic?"
q_bm25_parsed = "+caroline +picnic (friend outdoor summer)"

# 已检索的事实
round1_facts = [
    RetrievedFact(fact_id=101, session_number=1, rerank_score=0.9),
    RetrievedFact(fact_id=102, session_number=1, rerank_score=0.8),
    RetrievedFact(fact_id=201, session_number=2, rerank_score=0.7),
    RetrievedFact(fact_id=301, session_number=3, rerank_score=0.6),
]
```

### 处理过程

```
Step 1: BM25 会话搜索
  查询: ["caroline picnic", "Caroline and her friends had a picnic together.", "When did Caroline have a picnic?"]
  结果:
    - Session 1 (score=0.95): "Picnic at Central Park"
    - Session 2 (score=0.85): "Summer activities"
    - Session 4 (score=0.75): "Weekend plans"

Step 2: 解析查询会话搜索
  查询: "+caroline +picnic (friend outdoor summer)"
  结果:
    - Session 1 (score=0.90): "Picnic at Central Park"
    - Session 3 (score=0.80): "Outdoor gatherings"

Step 3: 事实聚合评分
  事实分布:
    - Session 1: fact_101 (0.9), fact_102 (0.8) → 分数 = 1.7
    - Session 2: fact_201 (0.7) → 分数 = 0.7
    - Session 3: fact_301 (0.6) → 分数 = 0.6
  结果:
    - Session 1 (score=1.7): 2 个贡献事实
    - Session 2 (score=0.7): 1 个贡献事实
    - Session 3 (score=0.6): 1 个贡献事实

Step 4: 合并去重（fact_agg 优先）
  seen_sess = {
    1: Session 1 (source=bm25),
    2: Session 2 (source=bm25),
    3: Session 3 (source=bm25_parsed),  # 覆盖 fact_agg 的结果
    4: Session 4 (source=bm25),
  }

Step 5: 排序并截断
  sessions = sorted by score, top 10
```

### 输出

```python
sessions = [
    RetrievedSession(
        session_summary_id=1001,
        session_id=1,
        subject="Picnic at Central Park",
        content="Caroline organized a picnic with her friends...",
        score=1.7,
        source="fact_agg",
        contributing_facts=2,
        session_number=1,
    ),
    RetrievedSession(
        session_summary_id=1002,
        session_id=1,
        subject="Picnic at Central Park",
        content="Caroline organized a picnic with her friends...",
        score=0.95,
        source="bm25",
        contributing_facts=0,
        session_number=1,
    ),
    RetrievedSession(
        session_summary_id=1003,
        session_id=2,
        subject="Summer activities",
        content="Various summer outdoor activities...",
        score=0.85,
        source="bm25",
        contributing_facts=0,
        session_number=2,
    ),
    # ...
]
```

## 会话摘要内容示例

```markdown
## Relevant Episodes

**Picnic at Central Park**: 
Caroline organized a picnic with her friends at Central Park in July 2023. 
The weather was sunny and warm. They enjoyed sandwiches, fruits, and lemonade.
---
**Summer activities**:
This summer was filled with outdoor activities including picnics, hiking, 
and beach trips. Caroline particularly enjoyed the weekend gatherings.
---
```

## 为什么需要会话检索

### 1. 提供更广泛的上下文

- **事实是离散的**: 每个事实只包含一个信息片段
- **会话是连续的**: 提供完整的时间线和背景

### 2. 检索全局信息

- 会话主题（subject）
- 会话摘要（content）
- 整体叙事

### 3. 补充事实检索

- 某些相关信息可能不在事实中
- 会话摘要可以提供额外线索

## 配置参数

```python
# membrain/config.py

QA_SESSION_BM25_TOP_N = 10    # BM25 会话搜索返回数量
QA_SESSION_FINAL_TOP_N = 5   # 最终会话数量
```

## 总结

会话检索阶段的核心策略：

| 策略 | 方法 | 说明 |
|------|------|------|
| BM25 | 关键词搜索 | 在会话摘要中搜索 |
| 解析查询 | Tantivy | 使用结构化查询 |
| 事实聚合 | 分数汇总 | 根据事实所属会话评分 |

**关键设计决策**:

1. **三种策略并行**: 确保覆盖各种检索场景
2. **fact_agg 优先**: 事实聚合的评分更有信息量
3. **去重合并**: 避免重复显示同一会话
4. **最多 10 个**: 限制返回数量以控制上下文大小

会话检索为 LLM 提供了更广泛的对话背景，是构建完整答案的重要信息来源。


## 13. 10 context packing

# Stage 10: 上下文打包 - Context Packing

## 概述

上下文打包是 MemBrain 搜索过程的最后处理阶段，负责将检索到的事实和会话整理成符合 LLM 输入要求的格式。这一阶段需要考虑：

1. **Token 预算**: 确保上下文不超过 LLM 的上下文窗口限制
2. **信息密度**: 在有限空间内最大化有用信息
3. **格式规范**: 生成结构化、易于理解的输出格式

MemBrain 使用两种预算：
- **事实预算**: 4500 tokens
- **会话预算**: 1500 tokens

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L543-L558)
- **事实打包**: [budget_pack.py](file:///home/project/MemBrain/membrain/retrieval/core/budget_pack.py#L29-L67)
- **会话格式化**: [budget_pack.py](file:///home/project/MemBrain/membrain/retrieval/core/budget_pack.py#L86-L112)

## 详细代码分析

### 10.1 入口代码

```python
# membrain/retrieval/application/retrieval.py

# ── 8. Pack context ───────────────────────────────────────────────────
packed = budget_pack(round1_facts, max_tokens=_FACT_BUDGET_TOKENS)  # 4500

session_section = format_session_section(sessions, _SESSION_BUDGET_TOKENS)  # 1500

# 将会话章节放在前面（更重要的信息）
if session_section:
    packed.text = session_section + "\n\n" + packed.text
    packed.token_count += estimate_tokens(session_section)
```

**预算常量**:

```python
_FACT_BUDGET_TOKENS = 4500    # 事实 token 预算
_SESSION_BUDGET_TOKENS = 1500  # 会话 token 预算
```

### 10.2 事实预算打包

```python
# membrain/retrieval/core/budget_pack.py

def budget_pack(
    facts: list[RetrievedFact],
    max_tokens: int = settings.QA_BUDGET_MAX_TOKENS,
) -> PackedContext:
    """Pack facts into token budget as a flat bullet list."""
    
    # Step 1: 按 rerank_score 排序
    sorted_facts = sorted(facts, key=lambda f: f.rerank_score, reverse=True)
    
    # Step 2: 贪婪填充预算
    selected: list[RetrievedFact] = []
    total_tokens = 0
    
    for fact in sorted_facts:
        line = _format_fact_line(fact)
        line_tokens = estimate_tokens(line)
        if total_tokens + line_tokens > max_tokens:
            continue  # 跳过超出预算的事实
        
        selected.append(fact)
        total_tokens += line_tokens
    
    # Step 3: 按时间顺序输出
    sorted_selected = sorted(selected, key=_sort_key_time)
    
    # Step 4: 格式化
    lines = ["## Additional Facts"] + [_format_fact_line(f) for f in sorted_selected]
    text = "\n".join(lines)
    
    return PackedContext(
        text=text,
        token_count=estimate_tokens(text),
        fact_ids=[f.fact_id for f in sorted_selected],
    )
```

**打包算法图解**:

```
输入: facts = [fact_A(score=0.9), fact_B(score=0.8), fact_C(score=0.7), fact_D(score=0.6), fact_E(score=0.5)]
预算: max_tokens = 4500

Step 1: 按分数排序
sorted_facts = [A(0.9), B(0.8), C(0.7), D(0.6), E(0.5)]

Step 2: 贪婪填充
- fact_A: 100 tokens → total = 100 ✓
- fact_B: 80 tokens → total = 180 ✓
- fact_C: 120 tokens → total = 300 ✓
- fact_D: 90 tokens → total = 390 ✓
- fact_E: 95 tokens → total = 485 ✓
(假设每条约 100 tokens，实际可容纳约 45 条)

Step 3: 按时间排序
假设时间顺序: C, A, B, D, E
sorted_selected = [C, A, B, D, E]

Step 4: 格式化输出
## Additional Facts
- fact_C text (known from 2023-07-15)
- fact_A text (known from 2023-07-15)
- fact_B text (known from 2023-06-01)
- fact_D text
- fact_E text
```

### 10.3 事实格式化

```python
# membrain/retrieval/core/budget_pack.py

def _format_fact_line(fact: RetrievedFact) -> str:
    """Format a single fact as a bullet line with resolved absolute dates."""
    
    # 检查是否有内联日期标注
    has_inline = bool(_RELATIVE_DATE_RE.search(fact.text))
    
    # 处理内联日期标注
    line = f"- {_resolve_inline_dates(fact.text)}"
    
    # 如果没有内联日期，添加时间上下文
    if fact.time_info and not has_inline:
        line += f" (known from message on {_clean_time_info(fact.time_info)})"
    
    return line
```

**格式化示例**:

```
示例 1: 有内联日期的事实
  原始: "Picnic on [last week::DATE] was fun"
  输出: "- Picnic on [2023-07-15] was fun"
  
示例 2: 无内联日期，但有时间信息
  原始: "Caroline worked at Google"
  time_info: "[2023-01-15]"
  输出: "- Caroline worked at Google (known from message on 2023-01-15)

示例 3: 无时间信息
  原始: "Some random fact"
  输出: "- Some random fact
```

### 10.4 Token 估算

```python
# membrain/retrieval/core/budget_pack.py

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4 + 1
```

**估算公式**: `tokens ≈ characters / 4 + 1`

- 这是一个粗略的估算
- 实际 token 数量取决于具体内容
- 英文约 4 字符/token，中文约 1-2 字符/token

### 10.5 会话章节格式化

```python
# membrain/retrieval/core/budget_pack.py

def format_session_section(
    sessions: list[RetrievedSession],
    max_tokens: int,
) -> str:
    """Format session summaries as a context section, respecting token budget."""
    
    if not sessions:
        return ""
    
    header = "## Relevant Episodes"
    lines = [header]
    budget = max_tokens - estimate_tokens(header)
    
    for s in sessions:
        # 格式化每个会话
        entry = f"**{s.subject}**: {s.content}\n---"
        cost = estimate_tokens(entry)
        
        if budget - cost < 0:
            break  # 超出预算，停止添加
        
        lines.append(entry)
        budget -= cost
    
    return "\n\n".join(lines) if len(lines) > 1 else ""
```

**会话格式化示例**:

```markdown
## Relevant Episodes

**Picnic at Central Park**: 
Caroline organized a picnic with her friends at Central Park in July 2023. 
The weather was sunny and warm. They enjoyed sandwiches, fruits, and lemonade.
---
**Summer activities**:
This summer was filled with outdoor activities including picnics, hiking, 
and beach trips. Caroline particularly enjoyed the weekend gatherings.
---
```

### 10.6 上下文合并

```python
# 事实打包
packed = budget_pack(round1_facts, max_tokens=4500)

# 会话格式化
session_section = format_session_section(sessions, 1500)

# 合并（会话在前，事实在后）
if session_section:
    packed.text = session_section + "\n\n" + packed.text
    packed.token_count += estimate_tokens(session_section)
```

**最终输出格式**:

```markdown
## Relevant Episodes

**Session Subject 1**: Session content...
---
**Session Subject 2**: Session content...
---

## Additional Facts

- Fact text (known from YYYY-MM-DD)
- Fact text (known from YYYY-MM-DD)
- Fact text
- ...
```

## PackedContext 数据结构

```python
@dataclass
class PackedContext:
    text: str              # 格式化的文本
    token_count: int       # 估算的 token 数量
    fact_ids: list[int]   # 包含的事实 ID 列表
```

## 完整示例

### 输入

```python
# 已排序的事实
round1_facts = [
    RetrievedFact(
        fact_id=101,
        text="Caroline worked at Google",
        rerank_score=0.9,
        time_info="[2023-01-15]"
    ),
    RetrievedFact(
        fact_id=102,
        text="She was a software engineer",
        rerank_score=0.85,
        time_info="[2023-01-15]"
    ),
    RetrievedFact(
        fact_id=103,
        text="She worked there for 3 years",
        rerank_score=0.8,
        time_info="[2023-01-15]"
    ),
    RetrievedFact(
        fact_id=104,
        text="Her manager was John",
        rerank_score=0.7,
        time_info="[2023-02-01]"
    ),
]

# 会话列表
sessions = [
    RetrievedSession(
        subject="Google Career",
        content="Caroline worked at Google from 2020 to 2023 as a software engineer...",
        score=0.95,
    ),
    RetrievedSession(
        subject="Work Colleagues",
        content="During her time at Google, Caroline worked closely with her manager John...",
        score=0.85,
    ),
]
```

### 处理过程

```
Step 1: 事实打包 (max_tokens=4500)
  排序后: [101, 102, 103, 104]
  
  贪婪填充:
  - fact_101: "Caroline worked at Google (known from 2023-01-15)"
             → 60 tokens → total = 60 ✓
  - fact_102: "She was a software engineer (known from 2023-01-15)"
             → 50 tokens → total = 110 ✓
  - fact_103: "She worked there for 3 years (known from 2023-01-15)"
             → 45 tokens → total = 155 ✓
  - fact_104: "Her manager was John (known from 2023-02-01)"
             → 40 tokens → total = 195 ✓
  
  按时间排序: [101, 102, 103, 104] (假设都是 2023-01/02)

Step 2: 会话格式化 (max_tokens=1500)
  header: "## Relevant Episodes" → 4 tokens
  budget: 1500 - 4 = 1496
  
  - session_1: 200 tokens → budget = 1296 ✓
  - session_2: 180 tokens → budget = 1116 ✓

Step 3: 合并
  session_section + "\n\n" + fact_section
```

### 最终输出

```markdown
## Relevant Episodes

**Google Career**: Caroline worked at Google from 2020 to 2023 as a software engineer. 
She was part of the Search team and worked on various machine learning projects.
---
**Work Colleagues**: During her time at Google, Caroline worked closely with her manager John 
and collaborated with several team members on product launches.
---

## Additional Facts

- Caroline worked at Google (known from 2023-01-15)
- She was a software engineer (known from 2023-01-15)
- She worked there for 3 years (known from 2023-01-15)
- Her manager was John (known from 2023-02-01)
```

## 为什么这样设计

### 1. 贪婪填充策略

- **按分数排序**: 优先选择最相关的事实
- **贪心**: 每个事实一旦加入就不退出
- **效率**: O(n) 时间复杂度

### 2. 时间排序输出

- **可读性**: 按时间顺序更易于理解
- **连贯性**: 相关事实按时间顺序呈现
- **上下文**: 帮助理解事件发展

### 3. 会话在前

- **优先级**: 会话提供更广泛的上下文
- **引导**: 帮助 LLM 理解对话背景
- **结构**: 会话主题可以帮助组织思路

### 4. Token 估算

- **快速**: 简单计算，无需实际分词
- **保守**: 估算偏高，避免超出限制
- **通用**: 适用于各种语言

## 配置参数

```python
# membrain/config.py

QA_BUDGET_MAX_TOKENS = 2000    # 实际上这里只是默认值
# 实际使用:
_FACT_BUDGET_TOKENS = 4500     # 硬编码在 retrieval.py
_SESSION_BUDGET_TOKENS = 1500   # 硬编码在 retrieval.py
```

## 总结

上下文打包阶段的核心逻辑：

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 按 rerank_score 排序 | 优先选择相关事实 |
| 2 | 贪婪填充预算 | 在 token 限制内最大化信息 |
| 3 | 按时间排序输出 | 提供连贯的时间线 |
| 4 | 格式化事实 | 添加时间上下文 |
| 5 | 格式化会话 | 生成会话章节 |
| 6 | 合并输出 | 会话在前，事实在后 |

**关键设计决策**:

1. **贪婪算法**: 简单高效，适合在线服务
2. **时间排序**: 输出更符合人类阅读习惯
3. **会话优先**: 更广泛的上下文在前
4. **双重预算**: 事实和会话分别控制

这一阶段生成最终用于 LLM 回答问题的上下文文本。


## 14. 11 return results

# Stage 11: 返回结果 - Return Results

## 概述

返回结果是 MemBrain 搜索过程的最后一个阶段，负责将处理后的数据打包成结构化的响应格式返回给客户端。这一阶段将所有前面的处理结果整合在一起，形成完整的 API 响应。

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L543-L575)
- **API 响应构建**: [memory.py](file:///home/project/MemBrain/membrain/api/routes/memory.py#L303-L316)

## 详细代码分析

### 11.1 搜索函数返回值

```python
# membrain/retrieval/application/retrieval.py

return {
    "packed_context": packed.text,              # 打包的上下文文本
    "packed_token_count": packed.token_count,    # Token 数量
    "fact_ids": packed.fact_ids,               # 事实 ID 列表
    "facts": [                                  # 事实详情列表
        {
            "fact_id": f.fact_id,
            "text": f.text,
            "source": f.source,
            "rerank_score": f.rerank_score,
            "time_info": f.time_info,
            "entity_ref": f.entity_ref,
            "aspect_path": f.aspect_path,
        }
        for f in round1_facts
    ],
    "sessions": [                               # 会话列表
        {
            "session_summary_id": s.session_summary_id,
            "session_id": s.session_id,
            "subject": s.subject,
            "content": s.content,
            "score": s.score,
            "source": s.source,
            "contributing_facts": s.contributing_facts,
        }
        for s in sessions
    ],
    "raw_messages": [],  # 预留字段，目前为空列表
}
```

### 11.2 API 响应构建

```python
# membrain/api/routes/memory.py

return MemorySearchResponse(
    packed_context=result["packed_context"],
    packed_token_count=result["packed_token_count"],
    fact_ids=result["fact_ids"],
    facts=[RetrievedFactOut(**f) for f in result["facts"]],
    sessions=[RetrievedSessionOut(**s) for s in result["sessions"]],
    raw_messages=[],
)
```

### 11.3 响应数据结构

#### MemorySearchResponse

```python
# 假设的响应模型
class MemorySearchResponse:
    packed_context: str           # 打包的上下文文本
    packed_token_count: int      # 估算的 token 数量
    fact_ids: list[int]         # 事实 ID 列表
    facts: list[RetrievedFactOut]  # 事实详情列表
    sessions: list[RetrievedSessionOut]  # 会话列表
    raw_messages: list           # 预留：原始消息列表
```

#### RetrievedFactOut

```python
class RetrievedFactOut:
    fact_id: int
    text: str
    source: str
    rerank_score: float
    time_info: str
    entity_ref: str
    aspect_path: str
```

#### RetrievedSessionOut

```python
class RetrievedSessionOut:
    session_summary_id: int
    session_id: int
    subject: str
    content: str
    score: float
    source: str
    contributing_facts: int
```

## 完整示例

### 返回值结构

```python
result = {
    "packed_context": """## Relevant Episodes

**Picnic at Central Park**: Caroline organized a picnic with her friends 
at Central Park in July 2023. The weather was sunny and warm.
---

## Additional Facts

- Caroline had a picnic in July (known from 2023-07-15)
- The picnic was at Central Park (known from 2023-07-15)
- Caroline's friends attended (known from 2023-07-15)
- They served sandwiches and lemonade (known from 2023-07-15)
- The weather was sunny and warm (known from 2023-07-15)
""",
    
    "packed_token_count": 523,
    
    "fact_ids": [101, 102, 103, 201, 202],
    
    "facts": [
        {
            "fact_id": 101,
            "text": "Caroline had a picnic in July",
            "source": "bm25",
            "rerank_score": 0.95,
            "time_info": "[2023-07-15]",
            "entity_ref": "Caroline (user's mother)",
            "aspect_path": "Personal Life > Social Events"
        },
        {
            "fact_id": 102,
            "text": "The picnic was at Central Park",
            "source": "embed",
            "rerank_score": 0.92,
            "time_info": "[2023-07-15]",
            "entity_ref": "Caroline (user's mother)",
            "aspect_path": "Personal Life > Social Events"
        },
        {
            "fact_id": 103,
            "text": "Caroline's friends attended",
            "source": "tree",
            "rerank_score": 0.88,
            "time_info": "[2023-07-15]",
            "entity_ref": "Caroline (user's mother)",
            "aspect_path": "Personal Life > Social Events"
        },
        {
            "fact_id": 201,
            "text": "They served sandwiches and lemonade",
            "source": "bm25_parsed",
            "rerank_score": 0.75,
            "time_info": "",
            "entity_ref": "",
            "aspect_path": ""
        },
        {
            "fact_id": 202,
            "text": "The weather was sunny and warm",
            "source": "embed",
            "rerank_score": 0.70,
            "time_info": "",
            "entity_ref": "",
            "aspect_path": ""
        },
    ],
    
    "sessions": [
        {
            "session_summary_id": 1001,
            "session_id": 1,
            "subject": "Picnic at Central Park",
            "content": "Caroline organized a picnic with her friends at Central Park...",
            "score": 0.95,
            "source": "fact_agg",
            "contributing_facts": 3
        },
        {
            "session_summary_id": 1002,
            "session_id": 2,
            "subject": "Summer Activities",
            "content": "This summer was filled with various outdoor activities...",
            "score": 0.85,
            "source": "bm25",
            "contributing_facts": 0
        },
    ],
    
    "raw_messages": []
}
```

### JSON 响应示例

```json
{
    "packed_context": "## Relevant Episodes\n\n**Picnic at Central Park**: Caroline organized...",
    "packed_token_count": 523,
    "fact_ids": [101, 102, 103, 201, 202],
    "facts": [
        {
            "fact_id": 101,
            "text": "Caroline had a picnic in July",
            "source": "bm25",
            "rerank_score": 0.95,
            "time_info": "[2023-07-15]",
            "entity_ref": "Caroline (user's mother)",
            "aspect_path": "Personal Life > Social Events"
        },
        ...
    ],
    "sessions": [
        {
            "session_summary_id": 1001,
            "session_id": 1,
            "subject": "Picnic at Central Park",
            "content": "Caroline organized a picnic...",
            "score": 0.95,
            "source": "fact_agg",
            "contributing_facts": 3
        },
        ...
    ],
    "raw_messages": []
}
```

## 返回值字段详解

### 1. packed_context

**类型**: `str`

**说明**: 格式化的上下文文本，可直接用于 LLM 提示。

**格式**:
```
## Relevant Episodes

**主题**: 内容
---
**主题**: 内容
---

## Additional Facts

- 事实内容 (known from YYYY-MM-DD)
- 事实内容
```

### 2. packed_token_count

**类型**: `int`

**说明**: 估算的 token 数量，基于字符数 / 4 + 1。

**用途**: 
- 客户端可以据此调整 LLM 调用
- 监控上下文大小

### 3. fact_ids

**类型**: `list[int]`

**说明**: 包含在 packed_context 中的事实 ID 列表。

**用途**:
- 追踪哪些事实被使用
- 调试和分析

### 4. facts

**类型**: `list[dict]`

**说明**: 每个事实的详细信息列表。

**字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| fact_id | int | 事实唯一标识 |
| text | str | 事实文本内容 |
| source | str | 来源路径 (bm25/embed/tree/bm25_parsed) |
| rerank_score | float | 融合后的评分 |
| time_info | str | 时间信息 |
| entity_ref | str | 实体引用 |
| aspect_path | str | 实体树路径 |

### 5. sessions

**类型**: `list[dict]`

**说明**: 相关的会话摘要列表。

**字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| session_summary_id | int | 会话摘要 ID |
| session_id | int | 会话 ID |
| subject | str | 会话主题 |
| content | str | 会话内容摘要 |
| score | float | 评分 |
| source | str | 来源 (bm25/fact_agg/bm25_parsed) |
| contributing_facts | int | 贡献的事实数量 |

### 6. raw_messages

**类型**: `list`

**说明**: 预留字段，目前为空列表。用于未来支持返回原始消息。

## 错误处理

### 空结果

```python
def _empty_result() -> dict:
    return {
        "packed_context": "",
        "packed_token_count": 0,
        "fact_ids": [],
        "facts": [],
        "sessions": [],
        "raw_messages": [],
    }
```

### 场景

1. **无检索结果**: 返回空结果
2. **所有路径失败**: 返回空结果
3. **token 超限**: 仍返回已选择的内容

## 与前端/客户端的交互

### 典型使用流程

```
1. 客户端发送 POST /api/memory/search
   {
     "question": "When did Caroline have a picnic?",
     "dataset": "personamem_v2",
     "task": "user_001"
   }

2. 服务器处理（11个阶段）

3. 服务器返回 MemorySearchResponse
   {
     "packed_context": "...",
     "packed_token_count": 523,
     ...
   }

4. 客户端使用 packed_context 调用 LLM
   获得最终答案
```

### 客户端示例

```python
import requests

response = requests.post(
    "http://localhost:9574/api/memory/search",
    json={
        "question": "When did Caroline have a picnic?",
        "dataset": "personamem_v2",
        "task": "user_001"
    }
)

result = response.json()

# 使用打包的上下文
llm_response = call_llm(
    prompt=f"Question: When did Caroline have a picnic?\n\nContext:\n{result['packed_context']}"
)
```

## 总结

返回结果阶段的核心功能：

| 字段 | 说明 | 用途 |
|------|------|------|
| packed_context | 格式化的上下文 | 直接用于 LLM |
| packed_token_count | Token 数量 | 监控和调整 |
| fact_ids | 事实 ID 列表 | 追踪和分析 |
| facts | 事实详情 | 详细调试 |
| sessions | 会话列表 | 背景信息 |
| raw_messages | 原始消息 | 预留 |

**关键设计决策**:

1. **结构化返回**: 多个字段满足不同需求
2. **同时返回详情和打包文本**: 既方便使用，又保留详细信息
3. **Token 计数**: 帮助客户端估算 LLM 调用成本
4. **预留字段**: 为未来扩展留出空间

这一阶段标志着 MemBrain 搜索过程的完成，返回的结果可以直接用于下游的 LLM 问答任务。


## 15. README

# MemBrain 搜索过程 - 完整指南

本文档详细分析 MemBrain 的搜索过程，将其分为 **11 个阶段**，每个阶段都有独立的详细文档。

## 目录

| 阶段 | 名称 | 文档 |
|:----:|------|------|
| 1 | API 入口 | [01_api_entry.md](01_api_entry.md) |
| 2 | 查询扩展 | [02_query_expansion.md](02_query_expansion.md) |
| 3 | 多路径检索 | [03_multi_path_retrieval.md](03_multi_path_retrieval.md) |
| 4 | 结果合并与去重 | [04_result_merging.md](04_result_merging.md) |
| 5 | 后处理 | [05_post_processing.md](05_post_processing.md) |
| 6 | 结果融合 | [06_result_fusion.md](06_result_fusion.md) |
| 7 | Agentic Round 2 | [07_agentic_round2.md](07_agentic_round2.md) |
| 8 | 实体引用解析 | [08_entity_resolution.md](08_entity_resolution.md) |
| 9 | 会话检索 | [09_session_retrieval.md](09_session_retrieval.md) |
| 10 | 上下文打包 | [10_context_packing.md](10_context_packing.md) |
| 11 | 返回结果 | [11_return_results.md](11_return_results.md) |

## 快速概览

```
用户问题
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: API 入口                                  │
│ • 解析请求参数                                     │
│ • 获取客户端资源                                   │
│ • 设置数据库 Schema                                │
└─────────────────────┬───────────────────────────────┘
                      │
    ┌─────────────────┴─────────────────┐
    ▼                                   ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│ Stage 2: 查询扩展       │   │ Mode = "direct"?        │
│ • Query Rewrite        │   │ 跳过 LLM 扩展          │
│ • Multi-Query         │   │ (仅 3 路径)            │
│ • BM25 Query Gen      │   └─────────────────────────┘
│ • 向量嵌入             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: 多路径检索                                │
│ Path A: BM25 关键词搜索                            │
│ Path B: 嵌入向量搜索 (原始问题)                    │
│ Path B2: HyDE 声明式搜索                          │
│ Path B3: 事件聚焦搜索                             │
│ Path C: 实体树光束搜索                             │
│ Path D: Tantivy 解析查询                          │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4: 结果合并与去重                            │
│ • 合并 6 条路径结果                               │
│ • 按 fact_id 去重                                 │
│ • 构建候选池 (Pool)                               │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 5: 后处理                                    │
│ • 时间注解注入                                     │
│ • 会话编号注入                                     │
│ • Aspect 路径富化                                  │
│ • Aspect 级别去重                                  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 6: 结果融合                                  │
│ 策略选择:                                          │
│ • RRF (默认): 互惠排名融合                         │
│ • Rerank: 交叉编码器重排                          │
└─────────────────────┬───────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────────┐   ┌─────────────────────────┐
│ Mode = "reflect"?   │   │ Mode = "expand"/"direct"│
│                     │   │ 跳过 Stage 7            │
│ Stage 7: Agentic   │   └─────────────────────────┘
│ Round 2             │
│ • LLM 反思评估          │
│ • 生成补充查询          │
│ • 补充检索            │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│ Stage 8: 实体引用解析                              │
│ • [Caroline] → Caroline (user's mother)           │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 9: 会话检索                                  │
│ • BM25 会话搜索                                    │
│ • 解析查询搜索                                     │
│ • 事实聚合评分                                     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 10: 上下文打包                              │
│ • 事实: 4500 tokens 预算                          │
│ • 会话: 1500 tokens 预算                          │
│ • Token 估算与格式化                              │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 11: 返回结果                                │
│ • packed_context: 格式化上下文                     │
│ • fact_ids: 事实 ID 列表                         │
│ • facts: 事实详情                                 │
│ • sessions: 会话列表                              │
└─────────────────────────────────────────────────────┘
```

## 关键参数配置

```python
# 检索参数
QA_BM25_FACT_TOP_N = 20          # Path A 返回数量
QA_EMBED_FACT_TOP_N = 20         # Path B 返回数量
QA_ENTITY_TOP_N = 5              # Path C 实体数量
QA_TREE_BEAM_WIDTH = 3           # Path C 光束宽度
QA_TREE_FACT_TOP_N = 20          # Path C 事实上限
QA_RERANK_TOP_K = 12             # 融合后保留数量

# Token 预算
_FACT_BUDGET_TOKENS = 4500       # 事实预算
_SESSION_BUDGET_TOKENS = 1500    # 会话预算

# 模式
# mode: "direct" | "expand" | "reflect"
# strategy: "rrf" | "rerank"
```

## 检索模式对比

| 模式 | LLM 查询扩展 | Agentic Round 2 | 路径数量 | 用途 |
|------|-------------|-----------------|---------|------|
| `direct` | ❌ | ❌ | 3 (A+B+C) | 低延迟场景 |
| `expand` | ✅ | ❌ | 6 (A+B+B2+B3+C+D) | **默认**，大多数场景 |
| `reflect` | ✅ | ✅ | 6 + R2 | 高召回要求 |

## 融合策略对比

| 策略 | 方法 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| `rrf` | 互惠排名融合 | 快 | 中 | **默认**，大多数场景 |
| `rerank` | 交叉编码器 | 慢 | 高 | 高精度要求 |

## 核心数据结构

### RetrievedFact

```python
@dataclass
class RetrievedFact:
    fact_id: int
    text: str
    source: str          # "bm25" | "embed" | "tree" | "bm25_parsed"
    rerank_score: float
    time_info: str
    entity_ref: str
    aspect_path: str
    session_number: int | None
```

### RetrievedSession

```python
@dataclass
class RetrievedSession:
    session_summary_id: int
    session_id: int
    subject: str
    content: str
    score: float
    source: str           # "bm25" | "fact_agg" | "bm25_parsed"
    contributing_facts: int
    session_number: int | None
```

## 搜索流程示例

### 输入

```python
{
    "question": "When did Caroline have a picnic with her friends?",
    "dataset": "personamem_v2",
    "task": "user_001",
    "mode": "expand",      # 使用查询扩展
    "strategy": "rrf"     # 使用 RRF 融合
}
```

### 处理流程

```
1. API 入口
   → 解析: dataset="personamem_v2", task="user_001"
   → task_pk = 1
   → 获取 embed_client, http_client, session_factory

2. 查询扩展
   → Query Rewrite: "Caroline picnic friends"
   → Multi-Query: ["What did Caroline do outdoors?", 
                   "Caroline and her friends had a picnic together.",
                   "Caroline picnic friends"]
   → BM25 Query: "+caroline +picnic (friend outdoor summer)"
   → 向量嵌入: orig_vec, hyde_vec, rewrite_vec

3. 六条检索路径
   → Path A (BM25): 20 条
   → Path B (Embedding): 20 条
   → Path B2 (HyDE): 20 条
   → Path B3 (Event): 15 条
   → Path C (Tree): ~10 条
   → Path D (Tantivy): 20 条

4. 合并去重
   → 去重后: ~50 条候选

5. 后处理
   → 注入时间信息
   → 注入会话编号
   → 构建 Aspect 信息

6. RRF 融合
   → 计算 RRF 分数
   → 排序，取 top-12

7. 实体引用解析
   → [Caroline] → Caroline (user's mother)

8. 会话检索
   → BM25 搜索: 10 个会话
   → 事实聚合: 5 个会话
   → 合并去重: 10 个会话

9. 上下文打包
   → 事实: 4500 tokens
   → 会话: 1500 tokens
   → 合并输出

10. 返回结果
    → JSON 响应
```

### 输出

```python
{
    "packed_context": "## Relevant Episodes\n\n**Picnic at Central Park**: ...",
    "packed_token_count": 523,
    "fact_ids": [101, 102, 103, 201, 202],
    "facts": [...],
    "sessions": [...],
    "raw_messages": []
}
```

## 文档路径

所有详细文档位于: `docs/search_stages/`


## 16. 01 entity extraction

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


## 17. 02 fact generation

# Stage 2: 事实生成 - Fact Generation

## 概述

事实生成是 MemBrain 写入过程的第二阶段，负责从用户对话消息中提取具体的事实陈述。与实体提取不同，事实生成不仅识别实体，还提取关于这些实体的具体信息（如行为、状态、关系等）。

这一阶段使用 `fact-generator` agent，通过 LLM 从消息中提取结构化的事实信息。

## 代码位置

- **主入口**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py#L213-L275)
- **Agent 配置**: [factory.py](file:///home/project/MemBrain/membrain/agents/factory.py)
- **验证器注册**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py#L55-L83)

## 详细代码分析

### 2.1 入口函数

```python
async def _generate_facts(
    self,
    entity_names: list[str],
    messages_text: str,
    context_text: str,
) -> tuple[list[str], list[dict]]:
    """Stage 2: Fact generation with entity coverage validation."""
    
    # ═══════════════════════════════════════════════════════════════
    # Step 1: 获取 fact-generator agent
    # ═══════════════════════════════════════════════════════════════
    
    fact_generator, generator_settings = self._factory.get_agent(
        "fact-generator",
        profile=self._profile,
    )
    
    # 注册实体覆盖验证器（确保事实只引用已知实体）
    _register_entity_coverage_validator(fact_generator)
    
    # ═══════════════════════════════════════════════════════════════
    # Step 2: 构建提示词
    # ═══════════════════════════════════════════════════════════════
    
    # 将实体列表转为 JSON
    entity_list_json = json.dumps(entity_names, ensure_ascii=False)
    
    prompts = self._registry.render_prompts(
        "fact-generator",
        profile=self._profile,
        entity_list_json=entity_list_json,          # 实体列表
        context_messages=context_text,              # 上下文消息
        messages_json=messages_text,                # 待提取的消息
    )
    
    # 允许的实体引用集合（用于验证）
    allowed_refs = set(entity_names)
    
    # ═══════════════════════════════════════════════════════════════
    # Step 3: 调用 LLM 生成事实（带实体约束）
    # ═══════════════════════════════════════════════════════════════
    
    try:
        result = await run_agent_with_retry(
            fact_generator,
            instructions=prompts,
            model_settings=generator_settings,
            deps={"allowed_entity_refs": allowed_refs},  # 传入允许的实体
        )
        facts = [f.model_dump() for f in result.output.facts]
        
    except Exception as exc:
        # 如果验证失败，尝试无约束模式
        log.warning(
            "fact-generator failed (illegal refs or schema violation), retrying without entity constraint: %s",
            exc,
        )
        
        try:
            result = await run_agent_with_retry(
                fact_generator,
                instructions=prompts,
                model_settings=generator_settings,
                deps={"allowed_entity_refs": set()},  # 无约束
            )
            facts = [f.model_dump() for f in result.output.facts]
            
            # 过滤掉包含非法引用的事实
            entity_names, facts = _apply_fact_generator_fallback(
                facts,
                allowed_refs,
            )
        except Exception:
            log.exception("fact-generator retry also failed")
            entity_names, facts = [], []
    
    # ═══════════════════════════════════════════════════════════════
    # Step 4: 记录日志并返回
    # ═══════════════════════════════════════════════════════════════
    
    log.debug(
        "fact-generator -> %d entities, %d facts", len(entity_names), len(facts)
    )
    log.info(
        "    [extract] entities (%d): %s",
        len(entity_names),
        ", ".join(entity_names) if entity_names else "(none)",
    )
    for fact in facts:
        log.info("    [extract] fact: %s", fact["text"])

    return entity_names, facts
```

### 2.2 实体覆盖验证器

这是 MemBrain 的一个关键设计，确保生成的事实只引用已提取的实体：

```python
# 记录已注册的验证器，避免重复注册
_registered: set[int] = set()


def _register_entity_coverage_validator(agent: Agent) -> None:
    """注册实体覆盖验证器，防止事实引用未提取的实体。"""
    
    # 避免重复注册
    if id(agent) in _registered:
        return
    _registered.add(id(agent))

    @agent.output_validator
    async def validate_entity_coverage(ctx: RunContext[dict], result) -> object:
        """验证生成的事实只引用了已知的实体。"""
        
        # 获取允许的实体引用集合
        allowed_refs: set[str] = ctx.deps.get("allowed_entity_refs", set())
        
        # 如果没有约束，直接返回
        if not allowed_refs:
            return result
        
        # 提取事实中的所有括号引用
        fact_refs: set[str] = set()
        for fact in result.facts:
            # 使用正则提取 [entity] 形式的引用
            fact_refs.update(_ENTITY_BRACKET_RE.findall(fact.text))
        
        # 检查是否有非法引用
        illegal = fact_refs - allowed_refs
        
        if illegal:
            # 如果有非法引用，抛出重试异常
            raise ModelRetry(
                f"These bracketed refs in your facts are not in the entity list: {sorted(illegal)}. "
                f"The allowed refs are: {sorted(allowed_refs)}. "
                f"Fix each fact to use only refs from that list, or remove the fact."
            )
        
        return result
```

### 2.3 回退过滤机制

当 LLM 验证失败时，使用回退机制过滤事实：

```python
# 正则表达式：匹配 [entity] 形式的引用
_ENTITY_BRACKET_RE = re.compile(r"\[([^\]]+)\]")


def _apply_fact_generator_fallback(
    facts: list[dict],
    allowed_refs: set[str],
) -> tuple[list[str], list[dict]]:
    """过滤掉包含非法引用的事实，并返回有效的实体列表。"""
    
    # Step 1: 过滤事实
    filtered_facts = []
    for fact in facts:
        # 提取事实中的所有引用
        refs = set(_ENTITY_BRACKET_RE.findall(fact["text"]))
        
        # 只保留引用都在允许列表中的事实
        if refs <= allowed_refs:
            filtered_facts.append(fact)
    
    # Step 2: 收集使用的实体
    used: set[str] = set()
    for fact in filtered_facts:
        used.update(_ENTITY_BRACKET_RE.findall(fact["text"]))
    
    # 返回排序后的实体列表和过滤后的事实列表
    return sorted(used), filtered_facts
```

## 事实数据结构

### 事实输出格式

```python
@dataclass
class GeneratedFact:
    text: str                    # 事实文本，如 "Caroline [went] to [Boston]"
    time: str | None = None     # 时间信息，如 "last week"
    source: str | None = None   # 来源消息（可选）
```

### 生成的事实示例

```python
# 输入消息
messages_text = """
User: I had lunch with John at Luigi's Pizza yesterday.
Assistant: That sounds nice! Was it good?
User: Yes, we had pizza and pasta. John loves their carbonara.
"""

# 实体列表
entity_names = ["John", "Luigi's Pizza", "yesterday"]

# 生成的事实
facts = [
    {
        "text": "[User] had lunch with [John] at [Luigi's Pizza] [yesterday]",
        "time": "yesterday",
    },
    {
        "text": "[John] loves [Luigi's Pizza]'s carbonara",
        "time": None,
    },
    {
        "text": "[Luigi's Pizza] serves pizza and pasta",
        "time": None,
    },
]
```

## 实体引用语法

MemBrain 使用方括号 `[entity]` 语法来标记事实中的实体引用：

```
原始消息: "Caroline met her sister Lisa at the coffee shop."
生成事实: "[Caroline] met [her sister Lisa] at [the coffee shop]"

原始消息: "John works at Google."
生成事实: "[John] works at [Google]"
```

**语法规则**:
- `[EntityName]`: 直接引用实体
- `[her sister Lisa]`: 带修饰语的实体引用
- `[the coffee shop]`: 描述性实体

## 完整处理流程

```
输入:
  entity_names = ["Caroline", "Lisa", "coffee shop"]
  messages_text = """
    User: I met my sister Lisa at the coffee shop yesterday.
    Assistant: Which coffee shop did you go to?
    User: It was the one on Main Street.
  """
  context_text = ""  (可选的上下文消息)

处理流程:

Step 1: 构建提示词
  prompt = """
  Extract facts from the following messages.
  
  Entities to extract facts about: ["Caroline", "Lisa", "coffee shop"]
  
  Messages:
  [messages_text]
  
  Guidelines:
  - Use [entity] bracket syntax to reference entities
  - Only extract facts about the given entities
  - Include time information when available
  """

Step 2: 调用 fact-generator (带约束)
  deps = {"allowed_entity_refs": {"Caroline", "Lisa", "coffee shop"}}
  
  LLM 分析消息，生成事实

Step 3: 验证输出
  - 检查每个事实中的 [entity] 引用
  - 确保所有引用都在 allowed_refs 中
  - 如果验证失败，抛出 ModelRetry

Step 4: 验证成功
  facts = [
      {"text": "[Caroline] met [Lisa] at [coffee shop] [yesterday]", ...},
      {"text": "[coffee shop] is on [Main Street]", ...},
  ]

Step 5: 回退处理（如果验证失败）
  - 重新调用 LLM（无约束）
  - 过滤掉包含非法引用的事实
  - 返回有效的事实列表

输出:
  entity_names = ["Caroline", "Lisa", "coffee shop"]
  facts = [...]
```

## 配置参数

```python
# membrain/config.py

# Agent 配置
AGENT_MAX_RETRIES = 3              # 最大重试次数
AGENT_RETRY_DELAY = 1.0           # 重试延迟（秒）

# 实体引用正则
ENTITY_BRACKET_PATTERN = r"\[([^\]]+)\]"
```

## 验证器工作流程

```
LLM 生成事实
    │
    ▼
┌─────────────────────────────────────────┐
│ 提取所有 [entity] 引用                   │
│ refs = {"Caroline", "Lisa", "coffee shop"}│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ 检查引用是否在 allowed_refs 中          │
│ allowed_refs = {"Caroline", "Lisa"}    │
│ illegal = refs - allowed_refs          │
└──────────────────┬──────────────────────┘
                   │
        ┌─────────┴─────────┐
        ▼                   ▼
   illegal 非空          illegal 为空
        │                   │
        ▼                   ▼
┌───────────────┐    ┌───────────────┐
│ 抛出 ModelRetry│    │ 返回结果      │
│ 要求重试      │    │              │
└───────────────┘    └───────────────┘
```

## 错误处理

### 场景 1: 验证失败后重试成功

```python
# 第一次调用失败（验证器抛出异常）
try:
    result = await run_agent_with_retry(...)
except ModelRetry as e:
    # 验证失败，重新调用（无约束）
    result = await run_agent_with_retry(
        fact_generator,
        deps={"allowed_entity_refs": set()}  # 无约束
    )
    
    # 过滤非法引用
    _, facts = _apply_fact_generator_fallback(facts, allowed_refs)
```

### 场景 2: 完全失败

```python
# 重试也失败
try:
    result = await run_agent_with_retry(...)
except Exception:
    log.exception("fact-generator retry also failed")
    entity_names, facts = [], []  # 返回空
```

## 完整示例

### 输入

```python
# 实体列表（来自 Stage 1）
entity_names = ["John", "Luigi's Pizza", "User"]

# 消息
messages = [
    {"role": "user", "content": "I had lunch with John at Luigi's Pizza yesterday."},
    {"role": "assistant", "content": "That sounds nice!"},
    {"role": "user", "content": "John said it's his favorite restaurant."}
]

# 上下文（可选）
context_messages = []
```

### 生成的事实

```python
# 生成的事实列表
facts = [
    {
        "text": "[User] had lunch with [John] at [Luigi's Pizza] [yesterday]",
        "time": "yesterday",
    },
    {
        "text": "[John] said [Luigi's Pizza] is [his favorite restaurant]",
        "time": None,
    },
    {
        "text": "[John] likes [Luigi's Pizza]",
        "time": None,
    },
]
```

### 验证过程

```
事实 1: "[User] had lunch with [John] at [Luigi's Pizza] [yesterday]"
  提取引用: {"User", "John", "Luigi's Pizza", "yesterday"}
  允许引用: {"John", "Luigi's Pizza", "User"}
  非法引用: {"yesterday"} → 注意: "yesterday" 是时间，不是实体
  
  验证器处理: 实际上 "yesterday" 可能被视为实体，
  取决于 agent 的理解和 allowed_refs 的定义

事实 2: "[John] said [Luigi's Pizza] is [his favorite restaurant]"
  提取引用: {"John", "Luigi's Pizza", "his favorite restaurant"}
  允许引用: {"John", "Luigi's Pizza", "User"}
  非法引用: {"his favorite restaurant"}
  
  结果: 可能被过滤或需要重试

事实 3: "[John] likes [Luigi's Pizza]"
  提取引用: {"John", "Luigi's Pizza"}
  允许引用: {"John", "Luigi's Pizza", "User"}
  非法引用: ∅
  
  结果: ✓ 验证通过
```

## 为什么这样设计

### 1. 实体覆盖验证

- **防止泄漏**: 确保 LLM 只使用已识别的实体
- **数据一致性**: 保持实体列表和事实的一致性
- **质量控制**: 过滤掉引用未知实体的事实

### 2. 回退机制

- **优雅降级**: 验证失败时不完全失败
- **数据恢复**: 尽可能保留有效的事实
- **灵活性**: 允许在约束和灵活之间平衡

### 3. 两阶段验证

```
阶段 1: LLM 输出验证 (实时)
  - 在 LLM 返回后立即验证
  - 验证失败立即重试
  - 保证输出符合要求

阶段 2: 回退过滤 (后备)
  - 如果实时验证完全失败
  - 放宽约束，重新生成
  - 过滤非法引用
```

## 总结

事实生成阶段的核心逻辑：

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 获取 fact-generator agent | 准备 LLM 调用 |
| 2 | 注册实体覆盖验证器 | 实时验证实体引用 |
| 3 | 构建提示词 | 传入实体列表和消息 |
| 4 | 调用 LLM（带约束） | 生成事实 |
| 5 | 验证输出 | 检查实体引用合法性 |
| 6 | 回退处理（可选） | 放宽约束，重新生成 |
| 7 | 记录日志 | 调试和追踪 |

**设计亮点**:

1. **实时验证**: 在 LLM 返回后立即验证
2. **防止数据泄漏**: 确保事实只引用已知实体
3. **回退机制**: 验证失败时优雅降级
4. **详细日志**: 记录生成的事实便于调试
5. **结构化输出**: 生成标准化的事实格式

这一阶段将非结构化的对话消息转化为结构化的事实陈述，为后续的实体消重和持久化提供了基础。


## 18. 03 entity resolution

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


## 19. 04 database persistence

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


## 20. 05 entity tree update

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


## 21. README

# MemBrain 写入过程 - 完整指南

本文档详细分析 MemBrain 的写入（摄取/ingestion）过程，将其分为 **5 个阶段**，每个阶段都有独立的详细文档。

## 目录

| 阶段 | 名称 | 文档 |
|:----:|------|------|
| 1 | 实体提取 | [01_entity_extraction.md](01_entity_extraction.md) |
| 2 | 事实生成 | [02_fact_generation.md](02_fact_generation.md) |
| 3 | 实体消重 | [03_entity_resolution.md](03_entity_resolution.md) |
| 4 | 数据库持久化 | [04_database_persistence.md](04_database_persistence.md) |
| 5 | 实体树更新 | [05_entity_tree_update.md](05_entity_tree_update.md) |

## 快速概览

```
用户消息
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: 实体提取                                  │
│ • 第一轮：基础 LLM 提取                            │
│ • 加载已有实体上下文                               │
│ • 第二轮：上下文增强提取                           │
│ 输出: entity_names (实体名称列表)                 │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: 事实生成                                  │
│ • 调用 fact-generator agent                       │
│ • 实体覆盖验证 (实时)                             │
│ • 回退过滤机制                                    │
│ 输出: facts (事实列表)                            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: 实体消重                                  │
│ • Layer 1: 精确匹配 (小写+空白折叠)               │
│ • Layer 2: MinHash + Jaccard 模糊匹配            │
│ • Layer 3: LLM 语义匹配                          │
│ 输出: decisions (包含 merge/create 动作)          │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4: 数据库持久化                              │
│ • 构建引用映射                                    │
│ • 创建/更新实体                                   │
│ • 插入事实 + 生成向量                            │
│ • 建立事实-实体关联                              │
│ 输出: 数据库记录                                  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 5: 实体树更新                                │
│ • 查找受影响实体                                   │
│ • 加载现有状态                                    │
│ • 提取 Aspects                                    │
│ • 构建/合并树结构                                 │
│ 输出: 实体树结构                                  │
└─────────────────────────────────────────────────────┘
```

## 核心概念

### 1. 实体 (Entity)

```python
class EntityModel:
    entity_id: str       # UUID
    task_id: int
    canonical_ref: str  # 规范名称
    desc: str           # 描述
    desc_embedding: Vector  # 描述向量
```

### 2. 事实 (Fact)

```python
class FactModel:
    id: int
    task_id: int
    batch_id: str       # 批次 ID
    session_number: int # 会话编号
    text: str           # 事实文本
    text_embedding: Vector  # 文本向量
    status: str         # 'active' | 'archived'
```

### 3. 实体树 (Entity Tree)

```
实体: Caroline
└── root
    ├── Career         # 工作
    │   └── 技能
    ├── Family        # 家庭
    │   ├── 父母
    │   └── 兄弟姐妹
    ├── Social        # 社交
    └── Interests     # 兴趣
```

## 三层实体消重策略

| 层级 | 方法 | 复杂度 | 阈值 |
|------|------|--------|------|
| Layer 1 | 精确匹配 | O(1) | 完全相等 |
| Layer 2 | MinHash + Jaccard | O(k) | ≥ 0.9 |
| Layer 3 | LLM 语义匹配 | O(n) LLM | 语义相似 |

## 两种工作流

| 工作流 | 说明 | 适用场景 |
|--------|------|----------|
| `DefaultIngestWorkflow` | 完整 4 阶段管道 | 大多数场景 |
| `PersonaMemIngestWorkflow` | 跳过实体提取，固定 ["User"] | PersonaMem v2 |

## 配置参数

```python
# 实体解析
RESOLVER_JACCARD_THRESHOLD = 0.9
RESOLVER_ENTROPY_THRESHOLD = 1.5
RESOLVER_MINHASH_PERMUTATIONS = 32
RESOLVER_MINHASH_BAND_SIZE = 4

# 提取上下文
EXTRACTION_CONTEXT_TOP_K = 20
EXTRACTION_CONTEXT_PER_QUERY = 5

# 树配置
MAX_FACTS_PER_NODE = 50
```

## 文档路径

所有详细文档位于: `docs/ingest_stages/`

- [01_entity_extraction.md](01_entity_extraction.md) - 实体提取
- [02_fact_generation.md](02_fact_generation.md) - 事实生成
- [03_entity_resolution.md](03_entity_resolution.md) - 实体消重
- [04_database_persistence.md](04_database_persistence.md) - 数据库持久化
- [05_entity_tree_update.md](05_entity_tree_update.md) - 实体树更新


## 22. 01 message input session create

# Stage 1: 消息输入与会话创建

## 概述

这是 MemBrain 系统的入口阶段，负责接收用户对话消息，创建会话记录，并触发后续的记忆提取流程。这一阶段是数据进入系统的第一道关口。

## 代码位置

- **API 入口**: [memory.py](file:///home/project/MemBrain/membrain/api/routes/memory.py#L174-L240)
- **会话模型**: [models.py](file:///home/project/MemBrain/membrain/storage/models.py)
- **数据解析**: [session_memory_workflow.py](file:///home/project/MemBrain/membrain/memory/application/session_memory_workflow.py)

## 详细代码分析

### 1.1 API 入口函数

```python
@router.post("/memory", response_model=MemoryResponse)
async def process_memory(req: MemoryRequest):
    """Unified memory endpoint.

    Modes (controlled by ``store`` and ``digest``):
      store=True,  digest=False  — save raw messages only
      store=True,  digest=True   — save then digest all pending sessions
      store=False, digest=True   — digest all pending sessions (no new data)
    """
    messages = [m.model_dump() for m in req.messages]

    # 参数验证
    if req.store and not messages:
        raise HTTPException(400, "messages required when store=True")
    if not req.store and not req.digest:
        raise HTTPException(400, "at least one of store or digest must be True")
```

### 1.2 数据集与任务解析

```python
# ── Resolve dataset / task ───────────────────────────────────────
with SessionLocal() as db:
    dataset, task = _get_or_create_dataset_task(
        db,
        req.dataset,
        req.task,
        req.agent_profile,
    )
    dataset_id = dataset.id
    task_pk = task.id
    agent_profile = task.agent_profile
    db.commit()
```

**数据集-任务关系**:
- 一个数据集 (Dataset) 可以包含多个任务 (Task)
- 每个任务对应一个用户的记忆空间
- 使用 `task_pk` 作为后续所有操作的主键

### 1.3 会话创建

```python
# ── Store ────────────────────────────────────────────────────────
session_pk: int | None = None
session_number: int | None = None

if req.store:
    with SessionLocal() as db:
        # 获取当前最大的会话编号
        max_sn = (
            db.query(func.max(ChatSessionModel.session_number))
            .filter_by(task_id=task_pk)
            .scalar()
        ) or 0
        session_number = max_sn + 1

        # 解析会话时间
        session_dt = None
        if req.session_time:
            try:
                session_dt = datetime.fromisoformat(req.session_time)
            except ValueError:
                pass

        # 创建会话记录
        session = ChatSessionModel(
            task_id=task_pk,
            session_number=session_number,
            session_time=session_dt,
            session_time_raw=req.session_time or None,
            digested_at=None,  # 初始未消化
        )
        db.add(session)
        db.flush()
        session_pk = session.id
```

### 1.4 消息存储

```python
        # 存储每条消息
        for pos, msg in enumerate(messages):
            msg_dt = None
            if msg.get("message_time"):
                try:
                    msg_dt = datetime.fromisoformat(msg["message_time"])
                except ValueError:
                    pass
            
            db.add(
                ChatMessageModel(
                    session_id=session_pk,
                    position=pos,
                    speaker=msg["speaker"],
                    content=msg["content"],
                    message_time=msg_dt,
                    message_time_raw=msg.get("message_time") or None,
                )
            )
        db.commit()
```

### 1.5 异步消化触发

```python
# ── Digest (async background) ────────────────────────────────────
if req.digest:
    # 创建异步任务处理消化
    t = asyncio.create_task(_run_digest(task_pk, agent_profile))
    _background_digest_tasks.add(t)
    t.add_done_callback(_background_digest_tasks.discard)
```

## 数据模型

### ChatSessionModel

```python
class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    session_number = Column(Integer, nullable=False)
    session_time = Column(DateTime)
    session_time_raw = Column(String)
    digested_at = Column(DateTime)  # 消化完成时间
    created_at = Column(DateTime, default=datetime.utcnow)
```

### ChatMessageModel

```python
class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    position = Column(Integer, nullable=False)
    speaker = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    message_time = Column(DateTime)
    message_time_raw = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
```

## 请求流程图

```
客户端请求
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ POST /api/memory                                                │
│ {                                                                │
│   "dataset": "personamem_v2",                                   │
│   "task": "user_001",                                          │
│   "messages": [...],                                            │
│   "store": true,                                                │
│   "digest": true                                                │
│ }                                                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 参数验证                                                         │
│ • 检查 store=True 时 messages 非空                              │
│ • 检查 store 或 digest 至少有一个为 True                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 数据集/任务解析                                                  │
│ • 查找或创建 dataset + task                                    │
│ • 返回 task_pk                                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 会话创建                                                         │
│ • 获取最大 session_number + 1                                   │
│ • 解析 session_time                                            │
│ • 创建 ChatSessionModel                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 消息存储                                                        │
│ • 遍历 messages                                                │
│ • 解析 message_time                                            │
│ • 创建 ChatMessageModel                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 异步消化触发                                                    │
│ • 创建 asyncio task                                            │
│ • 后台执行 _run_digest                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 返回响应                                                         │
│ {                                                                │
│   "dataset_id": 1,                                             │
│   "task_pk": 1,                                                │
│   "session_id": 1,                                             │
│   "session_number": 1,                                          │
│   "status": "stored_and_digest_queued"                          │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## 输入示例

```python
# 请求
payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": True,
    "digest": True,
    "session_time": "2024-01-15T14:30:00",
    "messages": [
        {
            "speaker": "user",
            "content": "I had lunch with my friend John at Luigi's Pizza yesterday.",
            "message_time": "2024-01-15T12:00:00"
        },
        {
            "speaker": "assistant",
            "content": "That sounds nice!",
            "message_time": "2024-01-15T12:00:30"
        },
        {
            "speaker": "user",
            "content": "Yes, we had a great time. John said it's his favorite restaurant.",
            "message_time": "2024-01-15T12:01:00"
        }
    ]
}

# 响应
{
    "dataset_id": 1,
    "task_pk": 1,
    "session_id": 1,
    "session_number": 1,
    "digested_sessions": 0,
    "status": "stored_and_digest_queued"
}
```

## 存储模式

| 模式 | store | digest | 说明 |
|------|-------|--------|------|
| 存储并消化 | True | True | 保存消息，然后提取记忆（最常用） |
| 仅存储 | True | False | 只保存原始消息 |
| 仅消化 | False | True | 消化已有消息（不保存新消息） |

## 与后续阶段的关联

```
Stage 1: 消息输入与会话创建
    │
    ├──→ ChatSessionModel (会话记录)
    ├──→ ChatMessageModel (消息记录)
    │
    └──→ 触发 Stage 2: 异步消化
              │
              ├──→ SessionMemoryWorkflow
              ├──→ BatchIngester
              └──→ IngestWorkflow (4 阶段)
```

## 关键设计决策

### 1. 异步消化

```python
# 消化是异步执行的，不阻塞主请求
if req.digest:
    t = asyncio.create_task(_run_digest(task_pk, agent_profile))
```

**优点**:
- 用户请求快速响应
- 消化过程可以批量处理
- 不影响主请求延迟

### 2. 会话编号自增

```python
max_sn = db.query(func.max(...)).scalar() or 0
session_number = max_sn + 1
```

**用途**:
- 按时间顺序追踪会话
- 用于消息关联和检索

### 3. 时间解析

```python
# 支持 ISO 格式时间，也支持原始字符串存储
session_dt = datetime.fromisoformat(req.session_time)
session_time_raw = req.session_time  # 保留原始输入
```

**灵活性**:
- 尝试解析为标准时间
- 保留原始字符串（可能有多种格式）

## 错误处理

### 1. 参数验证错误

```python
if req.store and not messages:
    raise HTTPException(400, "messages required when store=True")
```

### 2. 时间解析错误

```python
try:
    session_dt = datetime.fromisoformat(req.session_time)
except ValueError:
    pass  # 忽略错误，时间为 None
```

### 3. 数据集/任务不存在

```python
# _get_or_create_dataset_task 会自动创建
dataset, task = _get_or_create_dataset_task(db, req.dataset, req.task, ...)
```

## 总结

这一阶段的核心功能：

| 功能 | 说明 |
|------|------|
| 参数验证 | 确保必要参数存在 |
| 数据集/任务解析 | 查找或创建数据集和任务 |
| 会话创建 | 为对话创建会话记录 |
| 消息存储 | 保存原始消息到数据库 |
| 异步触发 | 后台启动记忆提取流程 |

**关键输出**:
- `task_pk`: 任务主键，用于后续所有操作
- `session_pk`: 会话主键
- `session_number`: 会话编号
- 异步任务: 触发后续的消化流程


## 23. 02 async digest session summary

# Stage 2: 异步消化触发与会话摘要

## 概述

这一阶段负责在消息存储到数据库后，后台异步触发记忆提取流程。主要包括：
1. 触发异步消化任务
2. 查找需要消化的会话
3. 生成会话摘要
4. 准备消息给后续阶段处理

## 代码位置

- **消化入口**: [memory.py](file:///home/project/MemBrain/membrain/api/routes/memory.py#L320-L360)
- **会话摘要**: [session_summarizer.py](file:///home/project/MemBrain/membrain/memory/application/session_summarizer.py)
- **工作流**: [session_memory_workflow.py](file:///home/project/MemBrain/membrain/memory/application/session_memory_workflow.py)

## 详细代码分析

### 2.1 异步消化任务

```python
# memory.py

async def _run_digest(task_pk: int, agent_profile: str | None):
    """运行消化任务"""
    from membrain.memory.application.session_memory_workflow import SessionMemoryWorkflow
    
    workflow = SessionMemoryWorkflow(
        task_id=task_pk,
        profile=agent_profile,
    )
    await workflow.digest_all()
```

### 2.2 会话摘要工作流

```python
# session_memory_workflow.py

class SessionMemoryWorkflow:
    def __init__(self, task_id: int, profile: str | None = None):
        self.task_id = task_id
        self.profile = profile
        # 初始化各种客户端和存储
        
    async def digest_all(self):
        """消化所有未消化的会话"""
        
        # 1. 查找未消化的会话
        undigested_sessions = self._find_undigested_sessions()
        
        for session in undigested_sessions:
            # 2. 为每个会话生成摘要
            await self._digest_session(session)
            
        # 3. 清理已完成的任务
        self._cleanup()
```

### 2.3 查找未消化的会话

```python
def _find_undigested_sessions(self) -> list[ChatSessionModel]:
    """查找需要消化的会话"""
    
    query = (
        db.query(ChatSessionModel)
        .filter(
            ChatSessionModel.task_id == self.task_id,
            ChatSessionModel.digested_at.is_(None)  # 未消化
        )
        .order_by(ChatSessionModel.session_number)
    )
    return query.all()
```

### 2.4 会话摘要生成

```python
async def _generate_session_summary(
    self,
    session: ChatSessionModel,
    messages: list[ChatMessageModel]
) -> str:
    """为会话生成摘要"""
    
    # 1. 格式化消息文本
    messages_text = self._format_messages(messages)
    
    # 2. 调用 LLM 生成摘要
    agent, settings = self.factory.get_agent("session-summarizer")
    prompts = self.registry.render_prompts(
        "session-summarizer",
        messages_json=messages_text,
    )
    
    result = await run_agent_with_retry(
        agent,
        instructions=prompts,
        model_settings=settings,
    )
    
    return result.output.summary
```

### 2.5 消息格式化

```python
def _format_messages(self, messages: list[ChatMessageModel]) -> str:
    """将消息格式化为文本"""
    
    lines = []
    for msg in messages:
        speaker = msg.speaker
        content = msg.content
        time_str = msg.message_time_raw or ""
        
        if time_str:
            lines.append(f"[{speaker}] ({time_str}): {content}")
        else:
            lines.append(f"[{speaker}]: {content}")
    
    return "\n".join(lines)
```

## 会话摘要数据结构

```python
class SessionSummaryModel(Base):
    __tablename__ = "session_summaries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    session_number = Column(Integer, nullable=False)
    summary = Column(Text, nullable=False)  # 摘要文本
    summary_embedding = Column(Vector(768))  # 摘要向量
    created_at = Column(DateTime, default=datetime.utcnow)
```

## 处理流程图

```
Stage 1: 消息输入
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 异步任务触发                                                     │
│ asyncio.create_task(_run_digest(task_pk, agent_profile))       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 查找未消化的会话                                                 │
│ SELECT * FROM chat_sessions                                     │
│   WHERE task_id = ? AND digested_at IS NULL                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 遍历每个会话                                                     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 步骤 1: 获取会话消息                                      │   │
│   │ SELECT * FROM chat_messages                             │   │
│   │   WHERE session_id = ?                                 │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 步骤 2: 格式化消息文本                                    │   │
│   │ [user]: message                                        │   │
│   │ [assistant]: message                                   │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 步骤 3: 调用 LLM 生成摘要                                │   │
│   │ agent: session-summarizer                               │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 步骤 4: 生成摘要向量                                      │   │
│   │ embed_client.embed(summary)                            │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 步骤 5: 保存摘要                                         │   │
│   │ INSERT INTO session_summaries                          │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                       │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 更新会话状态                                                     │
│ UPDATE chat_sessions SET digested_at = NOW()                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 触发批量提取                                                     │
│ BatchIngester.ingest_batch(...)                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 输入示例

```python
# 未消化的会话
session = {
    "id": 1,
    "task_id": 1,
    "session_number": 1,
    "session_time": "2024-01-15T14:30:00",
    "digested_at": None
}

# 会话消息
messages = [
    {"speaker": "user", "content": "I had lunch with John yesterday.", "position": 0},
    {"speaker": "assistant", "content": "That sounds nice!", "position": 1},
    {"speaker": "user", "content": "We went to Luigi's Pizza. It's his favorite.", "position": 2}
]

# 格式化后的消息
messages_text = """
[user]: I had lunch with John yesterday.
[assistant]: That sounds nice!
[user]: We went to Luigi's Pizza. It's his favorite.
"""

# LLM 生成的摘要
summary = "User had lunch with their friend John at Luigi's Pizza yesterday. John mentioned it's his favorite restaurant."
```

## 与后续阶段的关联

```
Stage 2: 异步消化触发
    │
    ├──→ SessionSummaryModel (会话摘要)
    │         │
    │         └──→ 用于搜索时的会话检索
    │
    └──→ 触发 Stage 3: 批量提取
              │
              ├──→ BatchIngester.ingest_batch
              └──→ IngestWorkflow
                    │
                    ├──→ 实体提取
                    ├──→ 事实生成
                    ├──→ 实体消重
                    └──→ 数据库持久化
```

## 关键设计决策

### 1. 异步处理

```python
# 后台异步执行，不阻塞主请求
t = asyncio.create_task(_run_digest(task_pk, agent_profile))
_background_digest_tasks.add(t)
```

**优点**:
- 用户请求立即返回
- 消化过程可以批量优化
- 支持并发处理多个会话

### 2. 会话摘要的作用

1. **搜索支持**: 提供会话级别的检索
2. **上下文**: 为 LLM 提供会话概览
3. **聚合**: 将多条消息聚合成简洁摘要

### 3. 向量生成

```python
# 为摘要生成向量，用于相似度检索
summary_embedding = embed_client.embed(summary)
```

**用途**:
- 基于向量的会话检索
- 语义相似度计算

## 总结

这一阶段的核心功能：

| 功能 | 说明 |
|------|------|
| 异步触发 | 后台执行消化任务 |
| 会话查找 | 找到未消化的会话 |
| 摘要生成 | LLM 生成会话摘要 |
| 向量生成 | 为摘要生成嵌入向量 |
| 状态更新 | 标记会话为已消化 |

**关键输出**:
- `SessionSummaryModel`: 会话摘要记录
- `summary_embedding`: 摘要向量
- 触发批量提取流程


## 24. 03 batch extraction entity fact generation

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


## 25. 04 entity resolution

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


## 26. 05 database persistence

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


## 27. 06 entity tree update

# Stage 6: 实体树更新

## 概述

这一阶段负责在数据持久化后更新实体树结构。实体树是 MemBrain 的核心数据结构，用于组织和管理实体之间的层级关系。

## 代码位置

- **更新器入口**: [entity_tree_updater.py](file:///home/project/MemBrain/membrain/memory/application/entity_tree_updater.py)
- **树计算管道**: [pipeline.py](file:///home/project/MemBrain/membrain/memory/core/entity_tree/pipeline.py)
- **树存储**: [entity_tree_store.py](file:///home/project/MemBrain/membrain/infra/persistence/entity_tree_store.py)

## 详细代码分析

### 6.1 更新器入口

```python
# entity_tree_updater.py

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
        # 1. 查找受影响的实体
        touched_entity_ids = self._store.find_touched_entities(task_id, batch_id)
        if not touched_entity_ids:
            return []

        # 2. 锁定实体
        with self._store.lock_entities(task_id, touched_entity_ids) as db:
            # 3. 加载状态
            state = self._store.load_update_state(
                task_id, batch_id, touched_entity_ids=touched_entity_ids, db=db
            )
            
            if not state.targets:
                return []

            # 4. 计算更新
            result = await compute_entity_tree_updates(
                task_id=state.task_id,
                targets=state.targets,
                embed_client=embed_client,
                registry=registry,
                factory=factory,
            )
            
            # 5. 应用更新
            self._store.apply_updates(task_id, state, result, db=db)
            db.commit()

        return result.profiled_entities
```

### 6.2 查找受影响实体

```python
# entity_tree_store.py

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

### 6.3 树更新计算

```python
# pipeline.py

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
        
        # 合并新事实到现有集合
        if existing_tree:
            existing_fact_ids = set(existing_tree.fact_ids)
            all_fact_ids = existing_fact_ids | {f.id for f in new_facts}
        else:
            all_fact_ids = {f.id for f in new_facts}
        
        # 提取 Aspect
        aspects = extract_aspects_from_facts(new_facts)
        
        # 构建/更新树
        if existing_tree:
            tree = merge_aspects(existing_tree, aspects)
        else:
            tree = create_entity_tree(entity_id, aspects)
        
        # 审计重组
        tree = audit_and_rebalance(tree)
        
        all_updates.append(tree)
        
        if len(aspects) > 1:
            profiled_entities.append(entity_id)
    
    return TreeUpdateResult(
        profiled_entities=profiled_entities,
        updates=all_updates,
    )
```

## 实体树结构示例

```
实体: Caroline (uuid-1)
树结构:
└── root
    ├── Career (工作)
    │   └── fact_ids: [1, 5, 10]
    ├── Family (家庭)
    │   └── fact_ids: [2, 7]
    │   ├── 父母
    │   │   └── fact_ids: [2]
    │   └── 兄弟姐妹
    │       └── fact_ids: [7]
    ├── Social (社交)
    │   └── fact_ids: [3, 8, 9]
    └── Interests (兴趣)
        └── fact_ids: [4, 6]
```

## 处理流程

```
输入:
  batch_id = "batch-123"
  touched_entity_ids = ["uuid-1", "uuid-2"]

Step 1: 查找受影响实体
  touched = [entity_id for fact in new_facts]

Step 2: 锁定实体 (防止并发)
  SELECT * FROM entity_trees WHERE entity_id IN (?) FOR UPDATE

Step 3: 加载状态
  - 获取新事实列表
  - 获取现有树结构

Step 4: 计算更新
  - 合并事实
  - 提取 Aspects
  - 构建树结构

Step 5: 应用更新
  - INSERT / UPDATE entity_trees

输出:
  profiled_entities: 已剖析的实体列表
```

## 总结

这一阶段完成写入过程的核心功能：

| 功能 | 说明 |
|------|------|
| 查找受影响实体 | 找出需要更新的实体 |
| 锁定机制 | 防止并发冲突 |
| Aspect 提取 | 按维度组织事实 |
| 树构建/合并 | 更新树结构 |

**关键输出**:
- `EntityTreeModel`: 更新的实体树
- `profiled_entities`: 包含多个 Aspect 的实体


## 28. README

# 交互过程文档

本文档详细分析 MemBrain 的写入过程与搜索过程的交互机制。

## 文档目录

| 阶段 | 名称 | 文档 |
|:----:|------|------|
| 1 | 消息输入与会话创建 | [01_message_input_session_create.md](01_message_input_session_create.md) |
| 2 | 异步消化触发与会话摘要 | [02_async_digest_session_summary.md](02_async_digest_session_summary.md) |
| 3 | 批量提取与实体事实生成 | [03_batch_extraction_entity_fact_generation.md](03_batch_extraction_entity_fact_generation.md) |
| 4 | 实体消重与规范化 | [04_entity_resolution.md](04_entity_resolution.md) |
| 5 | 数据库持久化 | [05_database_persistence.md](05_database_persistence.md) |
| 6 | 实体树更新 | [06_entity_tree_update.md](06_entity_tree_update.md) |

## 快速概览

```
用户消息
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: 消息输入与会话创建                        │
│ • 验证参数                                        │
│ • 创建/查找数据集和任务                           │
│ • 创建会话记录                                    │
│ • 存储消息                                        │
│ • 触发异步消化                                    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: 异步消化触发与会话摘要                    │
│ • 查找未消化的会话                                │
│ • 生成会话摘要                                    │
│ • 生成摘要向量                                    │
│ • 更新会话状态                                    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: 批量提取与实体事实生成                   │
│ • 格式化消息                                      │
│ • 两轮实体提取                                    │
│ • 事实生成 + 验证                                │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4: 实体消重与规范化                         │
│ • Layer 1: 精确匹配                              │
│ • Layer 2: MinHash + Jaccard                     │
│ • Layer 3: LLM 语义匹配                         │
│ • 实体规范化                                      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 5: 数据库持久化                             │
│ • 构建引用映射                                    │
│ • 创建/更新实体                                   │
│ • 插入事实                                        │
│ • 建立关联                                        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 6: 实体树更新                               │
│ • 查找受影响实体                                  │
│ • 锁定实体                                        │
│ • 提取 Aspects                                   │
│ • 更新树结构                                      │
└─────────────────────────────────────────────────────┘
```

## 核心数据结构

### 会话 (Session)

```
ChatSessionModel:
  - id: 会话主键
  - task_id: 任务 ID
  - session_number: 会话编号
  - session_time: 会话时间
  - digested_at: 消化完成时间
```

### 消息 (Message)

```
ChatMessageModel:
  - id: 消息主键
  - session_id: 所属会话
  - speaker: 发言者 (user/assistant)
  - content: 消息内容
  - position: 位置顺序
```

### 实体 (Entity)

```
EntityModel:
  - entity_id: UUID
  - task_id: 任务 ID
  - canonical_ref: 规范名称
  - desc: 描述
  - desc_embedding: 描述向量
```

### 事实 (Fact)

```
FactModel:
  - id: 事实主键
  - task_id: 任务 ID
  - batch_id: 批次 ID
  - session_number: 会话编号
  - text: 事实文本
  - text_embedding: 文本向量
  - status: 状态 (active/archived)
```

### 实体树 (Entity Tree)

```
EntityTreeModel:
  - id: 树主键
  - task_id: 任务 ID
  - entity_id: 实体 ID
  - fact_ids: 事实 ID 列表 (JSON)
  - structure_json: 树结构 (JSON)
```


## 29. search write interaction

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


## 30. 01 system architecture

# Step 1: 系统架构与数据模型

## 概述

本步骤介绍 MemBrain 的整体系统架构和核心数据模型，帮助理解数据在系统中的流转方式。

## 1.1 核心组件

MemBrain 是一个基于 FastAPI 的记忆系统，主要由以下组件构成：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MemBrain 系统                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│   │   客户端    │────▶│  FastAPI   │────▶│   数据库    │              │
│   │  (App/UI)   │     │   Server    │     │ (PostgreSQL)│              │
│   └─────────────┘     └─────────────┘     └─────────────┘              │
│                               │                                                │
│                               ▼                                                │
│                        ┌─────────────┐                                      │
│                        │   LLM       │                                      │
│                        │ (OpenAI/    │                                      │
│                        │  Anthropic) │                                      │
│                        └─────────────┘                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 组件说明

| 组件 | 说明 |
|------|------|
| 客户端 (Client) | 发起 API 请求的应用，如 Web App、移动端 |
| FastAPI Server | 处理 API 请求，协调各个模块 |
| 数据库 (PostgreSQL) | 存储实体、事实、会话等数据 |
| LLM | 生成实体、事实，处理自然语言 |

## 1.2 数据模型关系

MemBrain 使用以下核心数据模型：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据模型关系                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐         ┌──────────────┐                               │
│   │   Dataset    │ 1:N     │    Task      │                               │
│   │   (数据集)   │────────▶│   (任务)    │                               │
│   └──────────────┘         └──────┬───────┘                               │
│                                    │                                        │
│                                    │ 1:N                                    │
│                                    ▼                                        │
│   ┌──────────────┐         ┌──────────────┐                               │
│   │   Session    │ 1:N     │   Message    │                               │
│   │  (会话)      │────────▶│   (消息)     │                               │
│   └──────┬───────┘         └──────────────┘                               │
│          │                                                                   │
│          │ 1:N                                                                 │
│          ▼                                                                    │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐   │
│   │   Entity     │ 1:N     │    Fact      │ N:1     │  FactRef     │   │
│   │   (实体)     │◀────────│   (事实)     │────────▶│  (事实引用)  │   │
│   └──────────────┘         └──────────────┘         └──────────────┘   │
│          │                                                                   │
│          │ 1:1                                                                  │
│          ▼                                                                    │
│   ┌──────────────┐                                                          │
│   │ EntityTree   │                                                          │
│   │  (实体树)    │                                                          │
│   └──────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1.3 数据模型详解

### Dataset (数据集)

```python
class DatasetModel(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**示例**:
```json
{
    "id": 1,
    "name": "personamem_v2",
    "created_at": "2024-01-01T00:00:00"
}
```

### Task (任务)

```python
class TaskModel(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    name = Column(String, nullable=False)
    agent_profile = Column(String)
```

**示例**:
```json
{
    "id": 1,
    "dataset_id": 1,
    "name": "user_001",
    "agent_profile": "personamemv2"
}
```

### Session (会话)

```python
class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    session_number = Column(Integer)
    session_time = Column(DateTime)
    digested_at = Column(DateTime)
```

**示例**:
```json
{
    "id": 1,
    "task_id": 1,
    "session_number": 1,
    "session_time": "2024-01-15T14:30:00",
    "digested_at": "2024-01-15T14:30:05"
}
```

### Message (消息)

```python
class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    speaker = Column(String)  # "user" | "assistant"
    content = Column(Text)
    position = Column(Integer)
```

**示例**:
```json
{
    "id": 1,
    "session_id": 1,
    "speaker": "user",
    "content": "I had lunch with John yesterday.",
    "position": 0
}
```

### Entity (实体)

```python
class EntityModel(Base):
    __tablename__ = "entities"
    
    entity_id = Column(String, primary_key=True)
    task_id = Column(Integer)
    canonical_ref = Column(String)  # 规范名称
    desc = Column(Text)            # 描述
    desc_embedding = Column(Vector(768))  # 向量
```

**示例**:
```json
{
    "entity_id": "550e8400-e29b-41d4-a716-446655440000",
    "task_id": 1,
    "canonical_ref": "John",
    "desc": "User's friend, colleague at work",
    "desc_embedding": [0.123, -0.456, ...]
}
```

### Fact (事实)

```python
class FactModel(Base):
    __tablename__ = "facts"
    
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer)
    batch_id = Column(String)
    text = Column(Text)  # 事实文本
    text_embedding = Column(Vector(768))
    status = Column(String)  # "active" | "archived"
```

**示例**:
```json
{
    "id": 1,
    "task_id": 1,
    "batch_id": "batch-001",
    "text": "[User] had lunch with [John] at [Luigi's Pizza] [yesterday]",
    "text_embedding": [0.789, -0.123, ...],
    "status": "active"
}
```

### EntityTree (实体树)

```python
class EntityTreeModel(Base):
    __tablename__ = "entity_trees"
    
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer)
    entity_id = Column(String)
    fact_ids = Column(JSON)  # 事实 ID 列表
    structure_json = Column(JSON)  # 树结构
```

**示例**:
```json
{
    "id": 1,
    "task_id": 1,
    "entity_id": "user-entity-id",
    "fact_ids": [1, 2, 3, 4, 5],
    "structure_json": {
        "root": {
            "children": [
                {"label": "Social", "fact_ids": [1, 2]},
                {"label": "Career", "fact_ids": [3, 4, 5]}
            ]
        }
    }
}
```

## 1.4 数据流转示例

以一个具体的对话为例，展示数据流转：

### 用户对话

```
用户: I had lunch with my friend John at Luigi's Pizza yesterday.
助手: That sounds nice! What did you have?
用户: I had the carbonara and John ordered a pizza. He said it's his favorite restaurant.
```

### 数据流转过程

```
Step 1: 创建数据集/任务
  Dataset: personamem_v2
  Task: user_001

Step 2: 创建会话
  Session: session_number=1, session_time=2024-01-15

Step 3: 存储消息
  Message 0: [user] I had lunch with my friend John...
  Message 1: [assistant] That sounds nice!...
  Message 2: [user] I had the carbonara...

Step 4: 提取实体
  Entity: John
  Entity: Luigi's Pizza
  Entity: carbonara

Step 5: 生成事实
  Fact: [User] had lunch with [John] at [Luigi's Pizza] [yesterday]
  Fact: [John] likes [Luigi's Pizza]'s carbonara

Step 6: 建立关联
  FactRef: (fact_id=1, entity_id=John, alias_text="John")
  FactRef: (fact_id=1, entity_id=Luigi's Pizza, alias_text="Luigi's Pizza")

Step 7: 构建实体树
  EntityTree(User):
    - Social: fact_ids=[1, 2]
    - Interests: fact_ids=[3]
```

## 1.5 记忆使用场景示例

### 场景 1: 个人生活记忆

```python
# 用户输入对话
messages = [
    {"speaker": "user", "content": "I met my sister Lisa at the coffee shop yesterday."},
    {"speaker": "assistant", "content": "Which coffee shop did you go to?"},
    {"speaker": "user", "content": "It was the one on Main Street. We talked about our summer plans."}
]

# 提取的记忆
# Entities: Lisa, coffee shop, Main Street, summer plans
# Facts:
#   - [User] met [Lisa] at [coffee shop] [yesterday]
#   - [coffee shop] is on [Main Street]
#   - [User] and [Lisa] talked about [summer plans]
# EntityTree:
#   - Social: Lisa, coffee shop
#   - Plans: summer plans
```

### 场景 2: 工作经历记忆

```python
# 用户输入对话
messages = [
    {"speaker": "user", "content": "I started working at Google last month."},
    {"speaker": "assistant", "content": "That's great! What role did you get?"},
    {"speaker": "user", "content": "I'm a software engineer on the Search team. My manager is Sarah."}
]

# 提取的记忆
# Entities: Google, software engineer, Search team, Sarah
# Facts:
#   - [User] started working at [Google] [last month]
#   - [User] is a [software engineer] at [Google]
#   - [User] works on [Search team] at [Google]
#   - [Sarah] is [User]'s manager at [Google]
# EntityTree:
#   - Career: Google, software engineer, Search team
#   - Relationships: Sarah (manager)
```

### 场景 3: 兴趣爱好记忆

```python
# 用户输入对话
messages = [
    {"speaker": "user", "content": "I've been learning piano for 3 years."},
    {"speaker": "assistant", "content": "That's impressive! Do you practice daily?"},
    {"speaker": "user", "content": "Yes, I practice every evening. My favorite piece is Moonlight Sonata."}
]

# 提取的记忆
# Entities: piano, Moonlight Sonata
# Facts:
#   - [User] has been learning [piano] for [3 years]
#   - [User] practices [piano] every evening
#   - [User]'s favorite piece is [Moonlight Sonata]
# EntityTree:
#   - Interests: piano, Moonlight Sonata
#   - Habits: practicing every evening
```

## 总结

本步骤介绍了：
- MemBrain 的核心系统架构
- 6 个主要数据模型及其关系
- 数据的完整流转过程
- 3 个具体的记忆使用场景示例

理解这些基础概念后，可以更好地理解后续的 API 使用和开发流程。


## 31. 02 api endpoints

# Step 2: API 接口与服务器启动

## 概述

本步骤介绍 MemBrain 提供的 API 接口，包括端点说明、请求/响应格式，以及如何启动服务器。

## 2.1 API 端点列表

MemBrain 提供以下主要 API 端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/memory` | POST | 存储和/或消化对话 |
| `/api/memory/search` | POST | 搜索记忆 |

## 2.2 服务器启动

### 启动命令

```bash
# 进入项目目录
cd /home/project/MemBrain

# 启动 MemBrain 服务器
python -m uvicorn membrain.api.server:app --host 0.0.0.0 --port 9574
```

### 启动参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 监听地址 | `0.0.0.0` (所有地址) |
| `--port` | 监听端口 | `9574` |
| `--reload` | 热重载 (开发用) | 关闭 |

### 启动示例

```bash
# 开发模式 (带热重载)
python -m uvicorn membrain.api.server:app --reload

# 生产模式
python -m uvicorn membrain.api.server:app --host 0.0.0.0 --port 9574 --workers 4
```

## 2.3 API 端点详解

### 端点 1: POST /api/memory

**功能**: 存储对话消息并/或提取记忆

**请求体格式**:

```python
class MemoryRequest(BaseModel):
    dataset: str                          # 数据集名称
    task: str                             # 任务标识
    messages: list[dict]                  # 消息列表
    store: bool = True                   # 是否存储原始消息
    digest: bool = True                   # 是否消化（提取记忆）
    session_time: Optional[str] = None   # 会话时间 (ISO 格式)
    agent_profile: Optional[str] = None  # Agent 配置档案
```

**消息格式**:

```python
message = {
    "speaker": "user" | "assistant",     # 发言者
    "content": "消息内容",               # 消息文本
    "message_time": "2024-01-15T10:30:00"  # 可选：消息时间
}
```

**响应格式**:

```python
class MemoryResponse(BaseModel):
    dataset_id: int
    task_pk: int
    session_id: int | None
    session_number: int | None
    digested_sessions: int
    status: str  # "stored" | "stored_and_digest_queued" | "digest_queued"
```

### 端点 2: POST /api/memory/search

**功能**: 搜索记忆，返回用于 LLM 回答的上下文

**请求体格式**:

```python
class MemorySearchRequest(BaseModel):
    dataset: str                          # 数据集名称
    task: str                             # 任务标识
    question: str                         # 搜索问题
    top_k: Optional[int] = None          # 返回结果数量
    strategy: Optional[str] = None       # 融合策略: "rrf" | "rerank"
    mode: Optional[str] = None           # 搜索模式: "direct" | "expand" | "reflect"
```

**响应格式**:

```python
class MemorySearchResponse(BaseModel):
    packed_context: str           # 格式化上下文
    packed_token_count: int      # Token 数量
    fact_ids: list[int]         # 事实 ID 列表
    facts: list[RetrievedFactOut]  # 事实详情
    sessions: list[RetrievedSessionOut]  # 会话列表
    raw_messages: list          # 预留：原始消息
```

## 2.4 搜索模式详解

| 模式 | 说明 | 检索路径数 | 延迟 | 适用场景 |
|------|------|------------|------|----------|
| `direct` | 直接模式，跳过 LLM 扩展 | 3 (A+B+C) | 低 | 低延迟要求 |
| `expand` | 扩展模式，使用 LLM 扩展 | 6 (默认) | 中 | 大多数场景 |
| `reflect` | 反思模式，使用 LLM 反思增强 | 6 + R2 | 高 | 高召回要求 |

**模式选择建议**:

```python
# 低延迟场景
mode = "direct"  # 跳过 LLM 扩展，3 条检索路径

# 大多数场景 (默认)
mode = "expand"  # 使用 LLM 扩展，6 条检索路径

# 高召回场景
mode = "reflect"  # 使用反思增强，可能触发二次检索
```

## 2.5 融合策略详解

| 策略 | 方法 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| `rrf` | 互惠排名融合 | 快 | 中 | 大多数场景 (默认) |
| `rerank` | 交叉编码器重排 | 慢 | 高 | 高精度要求 |

**策略选择建议**:

```python
# 大多数场景 (默认)
strategy = "rrf"  # 快速融合

# 高精度要求
strategy = "rerank"  # 使用交叉编码器重排
```

## 2.6 使用示例

### Python requests 示例

```python
import requests

# MemBrain 服务器地址
BASE_URL = "http://localhost:9574/api"

# 写入数据
response = requests.post(
    f"{BASE_URL}/memory",
    json={
        "dataset": "personamem_v2",
        "task": "user_001",
        "messages": [
            {"speaker": "user", "content": "Hello"}
        ]
    }
)
print(response.json())

# 搜索数据
response = requests.post(
    f"{BASE_URL}/memory/search",
    json={
        "dataset": "personamem_v2",
        "task": "user_001",
        "question": "What did I do?"
    }
)
print(response.json())
```

### curl 示例

```bash
# 写入数据
curl -X POST "http://localhost:9574/api/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "messages": [
      {"speaker": "user", "content": "Hello"}
    ]
  }'

# 搜索数据
curl -X POST "http://localhost:9574/api/memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What did I do?"
  }'
```

## 2.7 记忆使用场景示例

### 场景 1: 基本对话存储

```python
# 存储用户与 AI 的对话
url = "http://localhost:9574/api/memory"

payload = {
    "dataset": "my_app",
    "task": "user_123",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "What's the weather today?"},
        {"speaker": "assistant", "content": "The weather is sunny, 25 degrees."},
        {"speaker": "user", "content": "Great! I'm planning a picnic."},
        {"speaker": "assistant", "content": "Sounds fun! Where are you planning to go?"}
    ]
}

response = requests.post(url, json=payload)
# 响应: {"status": "stored_and_digest_queued", ...}
```

### 场景 2: 带时间的对话

```python
# 存储带有时间戳的对话
payload = {
    "dataset": "my_app",
    "task": "user_123",
    "session_time": "2024-01-15T14:30:00",  # 会话时间
    "store": True,
    "digest": True,
    "messages": [
        {
            "speaker": "user", 
            "content": "I went to the gym this morning.",
            "message_time": "2024-01-15T08:00:00"  # 消息时间
        },
        {
            "speaker": "assistant", 
            "content": "Good job! How long did you workout?",
            "message_time": "2024-01-15T08:01:00"
        }
    ]
}

response = requests.post(url, json=payload)
```

### 场景 3: 不同搜索模式

```python
# 快速搜索 (低延迟)
payload = {
    "dataset": "my_app",
    "task": "user_123",
    "question": "What did I do today?",
    "mode": "direct",      # 跳过 LLM 扩展
    "strategy": "rrf"      # 快速融合
}

# 高精度搜索
payload = {
    "dataset": "my_app",
    "task": "user_123",
    "question": "What did I do today?",
    "mode": "reflect",     # 使用反思模式
    "strategy": "rerank"  # 使用重排序
}

# 调整返回数量
payload = {
    "dataset": "my_app",
    "task": "user_123",
    "question": "What did I do?",
    "top_k": 20  # 返回更多结果
}
```

## 2.8 错误处理

### 常见错误

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| 400 | 参数错误 | 检查必填参数 |
| 404 | 数据集/任务不存在 | 确认 dataset 和 task |
| 500 | 服务器错误 | 检查服务器日志 |

### 错误处理示例

```python
import requests
from requests.exceptions import HTTPError

def safe_request(url, payload):
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except HTTPError as e:
        print(f"HTTP 错误: {e}")
        print(f"响应: {e.response.json()}")
    except Exception as e:
        print(f"错误: {e}")
    return None

# 使用
result = safe_request("http://localhost:9574/api/memory", payload)
```

## 总结

本步骤介绍了：
- 2 个主要 API 端点及其功能
- 服务器启动命令和参数
- 请求/响应格式
- 搜索模式和融合策略的选择
- 3 个具体的使用场景示例
- 错误处理方法

理解 API 接口后，可以开始实际的开发和使用。


## 32. 03 write data

# Step 3: 写入数据 (Ingest)

## 概述

本步骤详细介绍如何通过 API 将用户对话写入 MemBrain 系统，包括请求格式、写入模式、参数说明，以及具体的记忆使用场景示例。

## 3.1 API 请求格式

### 端点

```
POST /api/memory
```

### 请求体

```python
class MemoryRequest(BaseModel):
    dataset: str                          # 数据集名称
    task: str                             # 任务标识
    messages: list[dict]                  # 消息列表
    store: bool = True                   # 是否存储原始消息
    digest: bool = True                   # 是否消化（提取记忆）
    session_time: Optional[str] = None   # 会话时间 (ISO 格式)
    agent_profile: Optional[str] = None  # Agent 配置档案
```

## 3.2 消息格式

### 消息结构

```python
message = {
    "speaker": "user" | "assistant",       # 发言者类型
    "content": "消息内容",                 # 消息文本
    "message_time": "2024-01-15T10:30:00"  # 可选：消息时间
}
```

### speaker 选项

| 值 | 说明 |
|----|------|
| `user` | 用户消息 |
| `assistant` | AI 助手消息 |
| `system` | 系统消息（如果有） |

### 示例

```python
messages = [
    {"speaker": "user", "content": "I had lunch with John yesterday."},
    {"speaker": "assistant", "content": "That sounds nice!"},
    {"speaker": "user", "content": "We went to Luigi's Pizza. It's his favorite restaurant."}
]
```

## 3.3 写入模式

MemBrain 支持三种写入模式：

### 模式 1: 存储并消化 (最常用)

```python
payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": True,     # 存储消息
    "digest": True,    # 提取记忆
    "messages": [...]
}
```

**适用场景**:
- 新的用户对话
- 完整的会话记录
- 需要立即提取记忆

### 模式 2: 仅存储

```python
payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": True,     # 存储消息
    "digest": False,   # 不提取记忆
    "messages": [...]
}
```

**适用场景**:
- 临时保存对话
- 稍后批量消化
- 调试模式

### 模式 3: 仅消化

```python
payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": False,    # 不存储新消息
    "digest": True,    # 消化已有消息
    "messages": []
}
```

**适用场景**:
- 批量处理历史会话
- 重新消化失败的会话

## 3.4 参数详解

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `dataset` | string | 是 | 数据集名称，如 "personamem_v2" |
| `task` | string | 是 | 任务标识，如 "user_001" |
| `messages` | array | 是 | 消息列表 |
| `store` | boolean | 否 | 是否存储原始消息，默认 True |
| `digest` | boolean | 否 | 是否提取记忆，默认 True |
| `session_time` | string | 否 | 会话时间 (ISO 格式) |
| `agent_profile` | string | 否 | Agent 配置档案 |

## 3.5 请求示例

### 基本请求

```python
import requests

url = "http://localhost:9574/api/memory"

payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": [
        {
            "speaker": "user",
            "content": "I had lunch with my friend John at Luigi's Pizza yesterday. It was really good!"
        },
        {
            "speaker": "assistant", 
            "content": "That sounds nice! What did you have?"
        },
        {
            "speaker": "user",
            "content": "I had the carbonara and John ordered a pizza. He said it's his favorite restaurant."
        }
    ]
}

response = requests.post(url, json=payload)
result = response.json()

print(json.dumps(result, indent=2))
```

### 响应格式

```json
{
    "dataset_id": 1,
    "task_pk": 1,
    "session_id": 1,
    "session_number": 1,
    "digested_sessions": 0,
    "status": "stored_and_digest_queued"
}
```

### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `dataset_id` | int | 数据集 ID |
| `task_pk` | int | 任务主键 |
| `session_id` | int | 会话 ID |
| `session_number` | int | 会话编号 |
| `digested_sessions` | int | 已消化的会话数 |
| `status` | string | 状态: "stored" / "stored_and_digest_queued" / "digest_queued" |

## 3.6 记忆使用场景示例

### 场景 1: 社交活动记忆

```python
# 输入: 用户分享社交活动
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "session_time": "2024-01-15T14:30:00",
    "messages": [
        {"speaker": "user", "content": "I had lunch with my friend John at Luigi's Pizza yesterday. It was really good!"},
        {"speaker": "assistant", "content": "That sounds nice! What did you have?"},
        {"speaker": "user", "content": "I had the carbonara and John ordered a pizza. He said it's his favorite restaurant. We've been going there for years."},
        {"speaker": "assistant", "content": "It must be a special place for you both!"},
        {"speaker": "user", "content": "Yes! We actually met there for the first time three years ago. Great memories."}
    ]
}

response = requests.post(url, json=payload)
```

**提取的记忆**:

```
Entities:
  - John (朋友)
  - Luigi's Pizza (餐厅)
  - carbonara (食物)

Facts:
  - [User] had lunch with [John] at [Luigi's Pizza] [yesterday]
  - [John] ordered [pizza] at [Luigi's Pizza]
  - [Luigi's Pizza] is [John]'s favorite restaurant
  - [User] and [John] have been going to [Luigi's Pizza] for [years]
  - [User] and [John] met at [Luigi's Pizza] for the first time [three years ago]

EntityTree:
  - Social: John, Luigi's Pizza
  - Relationships: John (friend)
  - History: first meeting at Luigi's Pizza
```

### 场景 2: 工作相关记忆

```python
# 输入: 用户谈论工作
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "I started working at Google last month."},
        {"speaker": "assistant", "content": "That's great! What role did you get?"},
        {"speaker": "user", "content": "I'm a software engineer on the Search team. My manager is Sarah."},
        {"speaker": "assistant", "content": "Sounds like an exciting opportunity!"},
        {"speaker": "user", "content": "Yes! We work on the core search ranking algorithm. It's challenging but rewarding."}
    ]
}

response = requests.post(url, json=payload)
```

**提取的记忆**:

```
Entities:
  - Google (公司)
  - software engineer (职位)
  - Search team (团队)
  - Sarah (经理)

Facts:
  - [User] started working at [Google] [last month]
  - [User] is a [software engineer] at [Google]
  - [User] works on [Search team] at [Google]
  - [Sarah] is [User]'s manager at [Google]
  - [Search team] works on [core search ranking algorithm]

EntityTree:
  - Career: Google, software engineer, Search team
  - Team: Sarah (manager)
  - Work: core search ranking algorithm
```

### 场景 3: 兴趣偏好记忆

```python
# 输入: 用户谈论兴趣爱好
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "I've been learning piano for 3 years."},
        {"speaker": "assistant", "content": "That's impressive! Do you practice daily?"},
        {"speaker": "user", "content": "Yes, I practice every evening. My favorite piece is Moonlight Sonata."},
        {"speaker": "assistant", "content": "Beethoven's Moonlight Sonata is a beautiful piece!"},
        {"speaker": "user", "content": "I also enjoy playing jazz. My favorite jazz artist is Bill Evans."}
    ]
}

response = requests.post(url, json=payload)
```

**提取的记忆**:

```
Entities:
  - piano (乐器)
  - Moonlight Sonata (曲目)
  - jazz (音乐风格)
  - Bill Evans (音乐人)

Facts:
  - [User] has been learning [piano] for [3 years]
  - [User] practices [piano] every evening
  - [User]'s favorite piece is [Moonlight Sonata]
  - [User] enjoys playing [jazz]
  - [User]'s favorite jazz artist is [Bill Evans]

EntityTree:
  - Interests: piano, Moonlight Sonata, jazz, Bill Evans
  - Habits: practicing every evening
  - Duration: 3 years
```

### 场景 4: 家庭关系记忆

```python
# 输入: 用户谈论家庭
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "My sister Lisa is visiting next week."},
        {"speaker": "assistant", "content": "That's nice! How long will she stay?"},
        {"speaker": "user", "content": "She's staying for a week. We're planning to go hiking together."},
        {"speaker": "assistant", "content": "Great! Any specific trails?"},
        {"speaker": "user", "content": "We're thinking about the Mountain Trail. It's a bit challenging but the view is amazing."}
    ]
}

response = requests.post(url, json=payload)
```

**提取的记忆**:

```
Entities:
  - Lisa (妹妹)
  - Mountain Trail (徒步路线)

Facts:
  - [Lisa] is [User]'s sister
  - [Lisa] is visiting [next week]
  - [Lisa] is staying for [a week]
  - [User] and [Lisa] are planning to go [hiking] together
  - [Mountain Trail] is a challenging hike with amazing views

EntityTree:
  - Family: Lisa (sister)
  - Plans: hiking with Lisa
  - Location: Mountain Trail
```

### 场景 5: 健康生活记忆

```python
# 输入: 用户谈论健康习惯
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "I've started a new diet this week."},
        {"speaker": "assistant", "content": "What kind of diet?"},
        {"speaker": "user", "content": "I'm doing keto. I'm trying to reduce my carb intake to under 50 grams per day."},
        {"speaker": "assistant", "content": "That can be effective! How are you feeling?"},
        {"speaker": "user", "content": "Actually, I feel great! Much more energy than before."}
    ]
}

response = requests.post(url, json=payload)
```

**提取的记忆**:

```
Entities:
  - keto (饮食方式)
  - carbs (营养素)

Facts:
  - [User] started [keto] diet [this week]
  - [User] is trying to reduce [carbs] to under [50 grams] per day
  - [User] feels more energetic on keto

EntityTree:
  - Health: keto diet
  - Goals: reduce carbs to 50g/day
  - Effects: more energy
```

## 3.7 curl 命令示例

```bash
# 存储并消化
curl -X POST "http://localhost:9574/api/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": true,
    "digest": true,
    "messages": [
      {"speaker": "user", "content": "I had lunch with John yesterday."},
      {"speaker": "assistant", "content": "That sounds nice!"},
      {"speaker": "user", "content": "We went to Luigis Pizza. Its his favorite."}
    ]
  }'

# 仅存储
curl -X POST "http://localhost:9574/api/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": true,
    "digest": false,
    "messages": [
      {"speaker": "user", "content": "Test message"}
    ]
  }'

# 仅消化
curl -X POST "http://localhost:9574/api/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": false,
    "digest": true
  }'
```

## 3.8 注意事项

1. **异步处理**: `digest=True` 时，消化是异步进行的，返回状态为 `stored_and_digest_queued`
2. **等待时间**: 建议在写入后等待几秒再搜索，以确保消化完成
3. **消息顺序**: 消息按 `position` 顺序处理，确保对话顺序正确
4. **时间戳**: 建议提供 `session_time` 和 `message_time`，有助于时序记忆

## 3.9 错误处理

```python
import requests
from requests.exceptions import HTTPError

def store_message(url, payload):
    try:
        response = requests.post(url, json=payload)
        
        # 检查状态码
        if response.status_code == 400:
            error = response.json()
            print(f"参数错误: {error.get('detail')}")
            return None
            
        response.raise_for_status()
        return response.json()
        
    except HTTPError as e:
        print(f"HTTP 错误: {e}")
    except Exception as e:
        print(f"错误: {e}")
    
    return None

# 使用
result = store_message("http://localhost:9574/api/memory", payload)
if result:
    print(f"成功! Session ID: {result['session_id']}")
```

## 总结

本步骤介绍了：
- API 请求格式和参数说明
- 消息格式和 speaker 选项
- 三种写入模式及其适用场景
- 5 个具体的记忆使用场景示例
- curl 命令示例
- 错误处理方法

写入数据后，可以进行搜索来检索记忆。


## 33. 04 search data

# Step 4: 搜索数据 (Search)

## 概述

本步骤详细介绍如何通过 API 搜索 MemBrain 中存储的记忆，包括请求格式、搜索模式、融合策略，以及具体的记忆使用场景示例。

## 4.1 API 请求格式

### 端点

```
POST /api/memory/search
```

### 请求体

```python
class MemorySearchRequest(BaseModel):
    dataset: str                          # 数据集名称
    task: str                             # 任务标识
    question: str                         # 搜索问题
    top_k: Optional[int] = None          # 返回结果数量
    strategy: Optional[str] = None       # 融合策略: "rrf" | "rerank"
    mode: Optional[str] = None           # 搜索模式: "direct" | "expand" | "reflect"
```

## 4.2 参数详解

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `dataset` | string | 是 | 数据集名称 |
| `task` | string | 是 | 任务标识 |
| `question` | string | 是 | 搜索问题 |
| `top_k` | int | 否 | 返回结果数量，默认系统配置 |
| `strategy` | string | 否 | 融合策略: "rrf" / "rerank" |
| `mode` | string | 否 | 搜索模式: "direct" / "expand" / "reflect" |

## 4.3 搜索模式

| 模式 | 说明 | 检索路径数 | 延迟 | 适用场景 |
|------|------|------------|------|----------|
| `direct` | 直接模式，跳过 LLM 扩展 | 3 | 低 | 低延迟要求 |
| `expand` | 扩展模式（默认），使用 LLM 扩展 | 6 | 中 | 大多数场景 |
| `reflect` | 反思模式，使用 LLM 反思增强 | 6 + R2 | 高 | 高召回要求 |

### 模式选择建议

```python
# 低延迟场景
mode = "direct"

# 大多数场景 (默认)
mode = "expand"

# 高召回场景
mode = "reflect"
```

## 4.4 融合策略

| 策略 | 方法 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| `rrf` | 互惠排名融合 (默认) | 快 | 中 | 大多数场景 |
| `rerank` | 交叉编码器重排 | 慢 | 高 | 高精度要求 |

### 策略选择建议

```python
# 大多数场景 (默认)
strategy = "rrf"

# 高精度要求
strategy = "rerank"
```

## 4.5 响应格式

```json
{
    "packed_context": "## Relevant Episodes\n\n**Subject**: Content...\n\n## Additional Facts\n\n- Fact text (known from 2024-01-15)\n...",
    "packed_token_count": 523,
    "fact_ids": [1, 2, 3, 4, 5],
    "facts": [
        {
            "fact_id": 1,
            "text": "User had lunch with John",
            "source": "bm25",
            "rerank_score": 0.95,
            "time_info": "2024-01-14",
            "entity_ref": "User",
            "aspect_path": "Social"
        },
        ...
    ],
    "sessions": [
        {
            "session_summary_id": 1,
            "session_id": 1,
            "subject": "Lunch with John",
            "content": "User had lunch...",
            "score": 0.90,
            "source": "bm25",
            "contributing_facts": 3
        },
        ...
    ],
    "raw_messages": []
}
```

### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `packed_context` | string | 格式化上下文，可直接用于 LLM |
| `packed_token_count` | int | 估算的 token 数量 |
| `fact_ids` | array | 包含在 packed_context 中的事实 ID 列表 |
| `facts` | array | 事实详情列表 |
| `sessions` | array | 相关的会话摘要列表 |
| `raw_messages` | array | 预留：原始消息列表 |

## 4.6 请求示例

### 基本请求

```python
import requests
import json

url = "http://localhost:9574/api/memory/search"

payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What did the user do yesterday?",
    "top_k": 10,
    "strategy": "rrf",
    "mode": "expand"
}

response = requests.post(url, json=payload)
result = response.json()

print(json.dumps(result, indent=2))
```

### 使用 LLM 生成答案

```python
# 步骤 1: 搜索记忆
search_result = search_memory("What did I do yesterday?")

# 步骤 2: 提取上下文
context = search_result["packed_context"]

# 步骤 3: 调用 LLM 生成答案
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: What did I do yesterday?"}
    ]
)

answer = response.choices[0].message.content
print(answer)
```

## 4.7 记忆使用场景示例

### 场景 1: 查询过去的社交活动

**场景**: 用户想知道之前和 John 一起做了什么

```python
# 搜索问题
question = "What did I do with John?"

payload = {
    "dataset": "my_app",
    "task": "user_001",
    "question": question,
    "top_k": 10,
    "mode": "expand",
    "strategy": "rrf"
}

response = requests.post(url, json=payload)
result = response.json()

# 输出结果
print("=== 搜索结果 ===")
print(f"Token Count: {result['packed_token_count']}")
print(f"Fact Count: {len(result['facts'])}")

print("\n--- Facts ---")
for fact in result["facts"]:
    print(f"  - {fact['text']}")

print("\n--- Packed Context ---")
print(result["packed_context"])
```

**预期输出**:

```json
{
    "packed_context": "## Relevant Episodes\n\n**Subject**: Lunch with John\n\nUser had lunch with John at Luigi's Pizza yesterday. They have been going there for years. They actually met there for the first time three years ago.\n\n## Additional Facts\n\n- [User] had lunch with [John] at [Luigi's Pizza] [yesterday]\n- [John] ordered [pizza] at [Luigi's Pizza]\n- [Luigi's Pizza] is [John]'s favorite restaurant\n- [User] and [John] have been going to [Luigi's Pizza] for [years]\n- [User] and [John] met at [Luigi's Pizza] for the first time [three years ago]\n",
    "facts": [
        {
            "fact_id": 1,
            "text": "[User] had lunch with [John] at [Luigi's Pizza] [yesterday]",
            "source": "bm25",
            "rerank_score": 0.95,
            "time_info": "yesterday"
        },
        ...
    ]
}
```

### 场景 2: 查询工作信息

**场景**: 用户想知道当前的工作情况

```python
# 搜索问题
question = "What is my current job?"

payload = {
    "dataset": "my_app",
    "task": "user_001",
    "question": question,
    "top_k": 10,
    "mode": "expand",
    "strategy": "rerank"  # 使用高精度模式
}

response = requests.post(url, json=payload)
result = response.json()

print("=== 搜索结果 ===")
print(f"Token Count: {result['packed_token_count']}")

print("\n--- Facts ---")
for fact in result["facts"]:
    print(f"  - {fact['text']}")

print("\n--- Packed Context ---")
print(result["packed_context"])
```

**预期输出**:

```json
{
    "packed_context": "## Relevant Episodes\n\n**Subject**: Work at Google\n\nUser started working at Google last month as a software engineer on the Search team.\n\n## Additional Facts\n\n- [User] started working at [Google] [last month]\n- [User] is a [software engineer] at [Google]\n- [User] works on [Search team] at [Google]\n- [Sarah] is [User]'s manager at [Google]\n- [Search team] works on [core search ranking algorithm]\n",
    "facts": [
        {
            "fact_id": 10,
            "text": "[User] started working at [Google] [last month]",
            "source": "semantic",
            "rerank_score": 0.98
        },
        ...
    ]
}
```

### 场景 3: 查询兴趣爱好

**场景**: 用户想知道自己的兴趣爱好

```python
# 搜索问题
question = "What are my hobbies and interests?"

payload = {
    "dataset": "my_app",
    "task": "user_001",
    "question": question,
    "top_k": 10,
    "mode": "expand",
    "strategy": "rrf"
}

response = requests.post(url, json=payload)
result = response.json()

print("=== 搜索结果 ===")
print(result["packed_context"])
```

**预期输出**:

```json
{
    "packed_context": "## Relevant Episodes\n\n**Subject**: Hobbies and Interests\n\nUser has several hobbies including playing piano and jazz.\n\n## Additional Facts\n\n- [User] has been learning [piano] for [3 years]\n- [User] practices [piano] every evening\n- [User]'s favorite piece is [Moonlight Sonata]\n- [User] enjoys playing [jazz]\n- [User]'s favorite jazz artist is [Bill Evans]\n",
    "facts": [
        {"fact_id": 20, "text": "[User] has been learning [piano] for [3 years]", ...},
        {"fact_id": 21, "text": "[User] practices [piano] every evening", ...},
        ...
    ]
}
```

### 场景 4: 查询家庭关系

**场景**: 用户想知道家庭成员的信息

```python
# 搜索问题
question = "Tell me about my family"

payload = {
    "dataset": "my_app",
    "task": "user_001",
    "question": question,
    "top_k": 10,
    "mode": "expand",
    "strategy": "rrf"
}

response = requests.post(url, json=payload)
result = response.json()

print(result["packed_context"])
```

**预期输出**:

```json
{
    "packed_context": "## Relevant Episodes\n\n**Subject**: Family\n\nUser's sister Lisa is visiting next week for a week. They plan to go hiking together at Mountain Trail.\n\n## Additional Facts\n\n- [Lisa] is [User]'s sister\n- [Lisa] is visiting [next week]\n- [Lisa] is staying for [a week]\n- [User] and [Lisa] are planning to go [hiking] together\n- [Mountain Trail] is a challenging hike with amazing views\n",
    "facts": [
        {"fact_id": 30, "text": "[Lisa] is [User]'s sister", ...},
        ...
    ]
}
```

### 场景 5: 查询健康习惯

**场景**: 用户想知道自己的健康习惯

```python
# 搜索问题
question = "What is my current diet?"

payload = {
    "dataset": "my_app",
    "task": "user_001",
    "question": question,
    "top_k": 10,
    "mode": "expand",
    "strategy": "rrf"
}

response = requests.post(url, json=payload)
result = response.json()

print(result["packed_context"])
```

**预期输出**:

```json
{
    "packed_context": "## Relevant Episodes\n\n**Subject**: Diet\n\nUser started a keto diet this week. They're trying to reduce carbs to under 50 grams per day and feel more energetic.\n\n## Additional Facts\n\n- [User] started [keto] diet [this week]\n- [User] is trying to reduce [carbs] to under [50 grams] per day\n- [User] feels more energetic on keto\n",
    "facts": [
        {"fact_id": 40, "text": "[User] started [keto] diet [this week]", ...},
        ...
    ]
}
```

## 4.8 curl 命令示例

```bash
# 基本搜索
curl -X POST "http://localhost:9574/api/memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What did I do yesterday?"
  }'

# 指定搜索模式和策略
curl -X POST "http://localhost:9574/api/memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What did I do yesterday?",
    "top_k": 10,
    "strategy": "rerank",
    "mode": "reflect"
  }'

# 低延迟搜索
curl -X POST "http://localhost:9574/api/memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What is my job?",
    "mode": "direct",
    "strategy": "rrf"
  }'
```

## 4.9 注意事项

1. **等待消化完成**: 写入数据后，建议等待几秒再搜索，确保消化完成
2. **搜索模式选择**:
   - 需要快速响应: 使用 `direct` 模式
   - 需要完整结果: 使用 `expand` 模式 (默认)
   - 需要高召回: 使用 `reflect` 模式
3. **融合策略选择**:
   - 大多数场景: 使用 `rrf` (默认)
   - 高精度要求: 使用 `rerank`
4. **top_k 调整**: 根据需要调整返回结果数量

## 4.10 完整使用流程

```python
import requests
import time
from openai import OpenAI

BASE_URL = "http://localhost:9574/api"

def store_and_digest():
    """步骤 1: 存储并消化对话"""
    url = f"{BASE_URL}/memory"
    
    payload = {
        "dataset": "my_app",
        "task": "user_001",
        "store": True,
        "digest": True,
        "messages": [
            {"speaker": "user", "content": "I had lunch with John yesterday."},
            {"speaker": "assistant", "content": "That sounds nice!"},
            {"speaker": "user", "content": "We went to Luigi's Pizza. It's his favorite."}
        ]
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    print(f"Status: {result['status']}")
    return result


def search_memory(question):
    """步骤 2: 搜索记忆"""
    url = f"{BASE_URL}/memory/search"
    
    payload = {
        "dataset": "my_app",
        "task": "user_001",
        "question": question,
        "top_k": 10,
        "mode": "expand",
        "strategy": "rrf"
    }
    
    response = requests.post(url, json=payload)
    return response.json()


def generate_answer(question, search_result):
    """步骤 3: 使用 LLM 生成答案"""
    client = OpenAI(api_key="your-api-key")
    
    context = search_result["packed_context"]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    
    return response.choices[0].message.content


# 执行完整流程
if __name__ == "__main__":
    # 1. 存储并消化对话
    store_result = store_and_digest()
    
    # 等待消化完成
    print("等待消化完成...")
    time.sleep(5)
    
    # 2. 搜索记忆
    question = "What did I do with John?"
    search_result = search_memory(question)
    print(f"找到 {len(search_result['facts'])} 个事实")
    
    # 3. 生成答案
    answer = generate_answer(question, search_result)
    print(f"\n答案: {answer}")
```

## 总结

本步骤介绍了：
- API 请求格式和参数说明
- 搜索模式的选择 (direct / expand / reflect)
- 融合策略的选择 (rrf / rerank)
- 响应格式和字段说明
- 5 个具体的记忆使用场景示例
- curl 命令示例
- 完整使用流程

通过搜索 API，可以检索存储的记忆，并用于 LLM 生成答案。


## 34. 05 complete examples

# Step 5: 完整使用示例

## 概述

本步骤提供 MemBrain 的完整使用示例，包括完整的 Python 脚本、curl 命令，以及端到端的使用流程。

## 5.1 完整 Python 脚本

### MemBrainClient 类

```python
#!/usr/bin/env python3
"""
MemBrain 完整使用示例
"""

import requests
import json
import time
from datetime import datetime
from typing import Optional

class MemBrainClient:
    """MemBrain 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:9574/api"):
        self.base_url = base_url
        
    def store_and_digest(
        self,
        dataset: str,
        task: str,
        messages: list[dict],
        session_time: str = None,
        store: bool = True,
        digest: bool = True
    ) -> dict:
        """存储并消化对话
        
        Args:
            dataset: 数据集名称
            task: 任务标识
            messages: 消息列表
            session_time: 会话时间
            store: 是否存储消息
            digest: 是否提取记忆
            
        Returns:
            响应结果字典
        """
        
        url = f"{self.base_url}/memory"
        
        payload = {
            "dataset": dataset,
            "task": task,
            "store": store,
            "digest": digest,
            "session_time": session_time,
            "messages": messages
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def search(
        self,
        dataset: str,
        task: str,
        question: str,
        top_k: int = 10,
        strategy: str = "rrf",
        mode: str = "expand"
    ) -> dict:
        """搜索记忆
        
        Args:
            dataset: 数据集名称
            task: 任务标识
            question: 搜索问题
            top_k: 返回结果数量
            strategy: 融合策略 (rrf/rerank)
            mode: 搜索模式 (direct/expand/reflect)
            
        Returns:
            搜索结果字典
        """
        
        url = f"{self.base_url}/memory/search"
        
        payload = {
            "dataset": dataset,
            "task": task,
            "question": question,
            "top_k": top_k,
            "strategy": strategy,
            "mode": mode
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def store_only(
        self,
        dataset: str,
        task: str,
        messages: list[dict],
        session_time: str = None
    ) -> dict:
        """仅存储消息，不提取记忆"""
        return self.store_and_digest(
            dataset, task, messages, session_time,
            store=True, digest=False
        )
    
    def digest_only(
        self,
        dataset: str,
        task: str
    ) -> dict:
        """仅消化已有消息，不存储新消息"""
        return self.store_and_digest(
            dataset, task, [], None,
            store=False, digest=True
        )
```

### 主函数示例

```python
def main():
    """主函数"""
    
    # 初始化客户端
    client = MemBrainClient()
    
    # 示例数据
    messages = [
        {
            "speaker": "user",
            "content": "I had lunch with my friend John at Luigi's Pizza yesterday."
        },
        {
            "speaker": "assistant",
            "content": "That sounds nice!"
        },
        {
            "speaker": "user", 
            "content": "It was great! We've been going there for years. John said it's his favorite restaurant."
        }
    ]
    
    # 步骤 1: 存储并消化
    print("=== 步骤 1: 存储并消化 ===")
    result = client.store_and_digest(
        dataset="personamem_v2",
        task="user_001",
        messages=messages,
        session_time=datetime.now().isoformat()
    )
    print(f"Status: {result['status']}")
    print(f"Session: {result['session_number']}")
    
    # 等待异步处理
    print("\n等待处理完成...")
    time.sleep(5)
    
    # 步骤 2: 搜索
    print("\n=== 步骤 2: 搜索 ===")
    search_result = client.search(
        dataset="personamem_v2",
        task="user_001",
        question="What did the user do yesterday?",
        top_k=10
    )
    
    print(f"Token Count: {search_result['packed_token_count']}")
    print(f"Facts: {len(search_result['facts'])}")
    print(f"Sessions: {len(search_result['sessions'])}")
    
    print("\n--- Packed Context ---")
    print(search_result['packed_context'])


if __name__ == "__main__":
    main()
```

## 5.2 curl 命令示例

### 存储并消化

```bash
# 基本存储并消化
curl -X POST "http://localhost:9574/api/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": true,
    "digest": true,
    "messages": [
      {"speaker": "user", "content": "I had lunch with John yesterday."},
      {"speaker": "assistant", "content": "That sounds nice!"},
      {"speaker": "user", "content": "We went to Luigis Pizza. Its his favorite."}
    ]
  }'

# 带会话时间
curl -X POST "http://localhost:9574/api/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": true,
    "digest": true,
    "session_time": "2024-01-15T14:30:00",
    "messages": [
      {"speaker": "user", "content": "I had lunch with John yesterday."}
    ]
  }'
```

### 搜索

```bash
# 基本搜索
curl -X POST "http://localhost:9574/api/memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What did the user do yesterday?",
    "top_k": 10,
    "strategy": "rrf",
    "mode": "expand"
  }'

# 高精度搜索
curl -X POST "http://localhost:9574/api/memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What did the user do yesterday?",
    "top_k": 10,
    "strategy": "rerank",
    "mode": "reflect"
  }'

# 低延迟搜索
curl -X POST "http://localhost:9574/api/memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What is my job?",
    "mode": "direct",
    "strategy": "rrf"
  }'
```

## 5.3 端到端使用流程

### 流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           完整使用流程                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
│  │   1. 存储    │────▶│   2. 消化    │────▶│   3. 搜索    │             │
│  │  写入消息    │     │  提取记忆    │     │  检索上下文  │             │
│  └──────────────┘     └──────────────┘     └──────────────┘             │
│                                                    │                       │
│                                                    ▼                       │
│                                            ┌──────────────┐             │
│                                            │  4. 生成答案 │             │
│                                            │  调用 LLM    │             │
│                                            └──────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 完整代码示例

```python
import requests
import time
from openai import OpenAI

BASE_URL = "http://localhost:9574/api"

def store_and_digest(messages, dataset="my_app", task="user_001"):
    """步骤 1: 存储并消化对话"""
    url = f"{BASE_URL}/memory"
    
    payload = {
        "dataset": dataset,
        "task": task,
        "store": True,
        "digest": True,
        "messages": messages
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    print(f"[Store] Status: {result['status']}")
    print(f"[Store] Session ID: {result.get('session_id')}")
    
    return result


def search_memory(question, dataset="my_app", task="user_001"):
    """步骤 2: 搜索记忆"""
    url = f"{BASE_URL}/memory/search"
    
    payload = {
        "dataset": dataset,
        "task": task,
        "question": question,
        "top_k": 10,
        "strategy": "rrf",
        "mode": "expand"
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    print(f"[Search] Token Count: {result['packed_token_count']}")
    print(f"[Search] Fact Count: {len(result['facts'])}")
    print(f"[Search] Session Count: {len(result['sessions'])}")
    
    return result


def generate_answer(question, search_result):
    """步骤 3: 使用 LLM 生成答案"""
    # 这里使用 OpenAI API 作为示例
    client = OpenAI(api_key="your-api-key")
    
    context = search_result["packed_context"]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant that answers questions based on the provided context."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    
    answer = response.choices[0].message.content
    return answer


def main():
    """完整流程示例"""
    
    # 准备对话数据
    messages = [
        {"speaker": "user", "content": "I had lunch with my friend John at Luigi's Pizza yesterday."},
        {"speaker": "assistant", "content": "That sounds nice! What did you have?"},
        {"speaker": "user", "content": "I had the carbonara and John ordered a pizza."},
        {"speaker": "assistant", "content": "Sounds delicious!"},
        {"speaker": "user", "content": "We actually met there for the first time three years ago. Great memories."}
    ]
    
    # 步骤 1: 存储并消化
    print("=" * 50)
    print("步骤 1: 存储并消化对话")
    print("=" * 50)
    store_result = store_and_digest(messages)
    
    # 等待消化完成
    print("\n等待消化完成...")
    time.sleep(5)
    
    # 步骤 2: 搜索记忆
    print("\n" + "=" * 50)
    print("步骤 2: 搜索记忆")
    print("=" * 50)
    question = "What did the user do yesterday?"
    search_result = search_memory(question)
    
    print("\n--- Packed Context ---")
    print(search_result["packed_context"])
    
    # 步骤 3: 生成答案
    print("\n" + "=" * 50)
    print("步骤 3: 生成答案")
    print("=" * 50)
    answer = generate_answer(question, search_result)
    print(answer)


if __name__ == "__main__":
    main()
```

## 5.4 记忆使用场景完整示例

### 场景: 完整的用户画像构建

```python
"""
场景: 为用户构建完整的画像
包括: 基本信息、兴趣爱好、工作经历、家庭关系
"""

client = MemBrainClient()

# 对话 1: 基本信息
messages_1 = [
    {"speaker": "user", "content": "Hi, I'm Alice. I'm 28 years old and I live in San Francisco."},
    {"speaker": "assistant", "content": "Nice to meet you, Alice!"},
]

# 对话 2: 兴趣爱好
messages_2 = [
    {"speaker": "user", "content": "I've been playing guitar for 5 years."},
    {"speaker": "assistant", "content": "That's great! What kind of music do you like?"},
    {"speaker": "user", "content": "I love rock and jazz. My favorite band is Queen."},
]

# 对话 3: 工作经历
messages_3 = [
    {"speaker": "user", "content": "I work as a software engineer at a startup."},
    {"speaker": "assistant", "content": "What do you work on?"},
    {"speaker": "user", "content": "We build AI products. My team lead is Bob."},
]

# 对话 4: 家庭
messages_4 = [
    {"speaker": "user", "content": "My brother is visiting me next month."},
    {"speaker": "assistant", "content": "How exciting! Any plans?"},
    {"speaker": "user", "content": "We're going to hike at Yosemite. It's his favorite place."},
]

# 依次存储所有对话
conversations = [
    ("基本信息", messages_1),
    ("兴趣爱好", messages_2),
    ("工作经历", messages_3),
    ("家庭关系", messages_4),
]

for title, messages in conversations:
    print(f"\n存储: {title}")
    result = client.store_and_digest(
        dataset="user_profile",
        task="alice_001",
        messages=messages
    )
    print(f"  Status: {result['status']}")
    time.sleep(3)  # 等待消化

# 等待所有消化完成
print("\n等待所有消化完成...")
time.sleep(10)

# 搜索用户画像
questions = [
    "Tell me about Alice",
    "What are Alice's hobbies?",
    "What is Alice's job?",
    "Tell me about Alice's family",
]

print("\n" + "=" * 50)
print("用户画像搜索结果")
print("=" * 50)

for question in questions:
    print(f"\n问题: {question}")
    result = client.search(
        dataset="user_profile",
        task="alice_001",
        question=question
    )
    print(f"上下文: {result['packed_context'][:200]}...")
```

**预期输出**:

```
问题: Tell me about Alice
上下文:
## Relevant Episodes

**Subject**: User Profile

Alice is 28 years old and lives in San Francisco. She has been playing guitar for 5 years and loves rock and jazz. She works as a software engineer at a startup and her brother is visiting next month.

## Additional Facts

- [Alice] is [28] years old
- [Alice] lives in [San Francisco]
- [Alice] has been playing [guitar] for [5 years]
- [Alice] loves [rock] and [jazz]
- [Alice]'s favorite band is [Queen]
- [Alice] works as a [software engineer] at a [startup]
- [Alice]'s team lead is [Bob]
- [Alice]'s brother is visiting [next month]
- [Alice] and [brother] are going to [Yosemite]
...
```

## 5.5 使用注意事项

### 1. 异步处理

```python
# digest=True 时，消化是异步的
# 返回后需要等待一段时间再搜索

result = client.store_and_digest(...)
print(result['status'])  # "stored_and_digest_queued"

# 等待建议
time.sleep(5)  # 简单等待
# 或轮询检查
while not is_digest_complete(...):
    time.sleep(1)
```

### 2. 错误处理

```python
import requests
from requests.exceptions import HTTPError

def safe_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPError as e:
            print(f"HTTP 错误: {e.response.status_code}")
            print(f"详情: {e.response.json()}")
            return None
        except Exception as e:
            print(f"错误: {e}")
            return None
    return wrapper

@safe_operation
def safe_search(question):
    return client.search(dataset="my_app", task="user_001", question=question)
```

### 3. 批量处理

```python
def batch_store(client, conversation_list):
    """批量存储多个会话"""
    results = []
    for i, messages in enumerate(conversation_list):
        print(f"处理会话 {i+1}/{len(conversation_list)}")
        result = client.store_and_digest(
            dataset="my_app",
            task="user_001",
            messages=messages
        )
        results.append(result)
        time.sleep(1)  # 避免过快请求
    return results
```

## 总结

本步骤提供了：
- 完整的 Python MemBrainClient 类
- curl 命令示例
- 端到端的使用流程
- 完整的记忆使用场景示例
- 使用注意事项

通过这些示例，可以快速上手使用 MemBrain 进行记忆存储和检索。


## 35. 06 advanced features

# Step 6: 高级功能

## 概述

本步骤介绍 MemBrain 的高级功能，包括多数据集管理、批量处理、自定义搜索策略、异步处理等。

## 6.1 多数据集管理

### 为不同用户创建独立数据集

```python
# 为不同用户创建不同的数据集
client = MemBrainClient()

# 用户列表
users = ["user_001", "user_002", "user_003"]

# 为每个用户存储数据
for user_id in users:
    messages = [
        {"speaker": "user", "content": f"This is {user_id}'s personal data."},
    ]
    
    result = client.store_and_digest(
        dataset="personamem_v2",
        task=user_id,
        messages=messages
    )
    print(f"Stored for {user_id}: {result['status']}")
```

### 使用场景

| 场景 | dataset | task | 说明 |
|------|---------|------|------|
| 多用户应用 | "app_data" | user_id | 每个用户独立 task |
| 多设备应用 | "app_data" | device_id | 每个设备独立 task |
| 多角色应用 | "app_data" | role_name | 每个角色独立 task |

### 数据隔离

```python
# 搜索时指定用户，确保数据隔离
result_user1 = client.search(
    dataset="personamem_v2",
    task="user_001",  # 只搜索 user_001 的数据
    question="What did I do?"
)

result_user2 = client.search(
    dataset="personamem_v2",
    task="user_002",  # 只搜索 user_002 的数据
    question="What did I do?"
)
```

## 6.2 批量写入

### 基本批量写入

```python
# 批量处理多个会话
def batch_ingest(client, sessions, dataset="my_app", task="user_001"):
    """批量写入多个会话
    
    Args:
        client: MemBrainClient 实例
        sessions: 会话列表，每个会话是消息列表
        dataset: 数据集名称
        task: 任务标识
    """
    results = []
    
    for i, messages in enumerate(sessions):
        print(f"处理会话 {i+1}/{len(sessions)}")
        
        result = client.store_and_digest(
            dataset=dataset,
            task=task,
            messages=messages
        )
        
        results.append(result)
        
        # 避免过快请求
        time.sleep(1)
    
    return results


# 使用
sessions = [
    [  # 会话 1
        {"speaker": "user", "content": "Hello"},
        {"speaker": "assistant", "content": "Hi!"}
    ],
    [  # 会话 2
        {"speaker": "user", "content": "How are you?"},
        {"speaker": "assistant", "content": "I'm good!"}
    ],
    [  # 会话 3
        {"speaker": "user", "content": "What's the weather?"},
        {"speaker": "assistant", "content": "It's sunny."}
    ],
]

results = batch_ingest(client, sessions)
```

### 并发批量写入

```python
import asyncio
import aiohttp

async def async_batch_ingest(sessions, dataset="my_app", task="user_001"):
    """异步批量写入"""
    
    async def send_session(session_data, session_id):
        async with aiohttp.ClientSession() as session:
            url = "http://localhost:9574/api/memory"
            payload = {
                "dataset": dataset,
                "task": task,
                "store": True,
                "digest": True,
                "messages": session_data
            }
            async with session.post(url, json=payload) as response:
                return await response.json()
    
    tasks = [
        send_session(messages, i) 
        for i, messages in enumerate(sessions)
    ]
    
    results = await asyncio.gather(*tasks)
    return results


# 使用
sessions = [
    [{"speaker": "user", "content": "Message 1"}],
    [{"speaker": "user", "content": "Message 2"}],
    [{"speaker": "user", "content": "Message 3"}],
]

results = asyncio.run(async_batch_ingest(sessions))
```

## 6.3 自定义搜索策略

### 快速搜索 (低延迟)

```python
# 适用于实时对话场景，需要快速响应

result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do today?",
    mode="direct",      # 跳过 LLM 扩展，3 条检索路径
    strategy="rrf"     # 快速融合
)

# 特点:
# - 延迟最低
# - 跳过 LLM 扩展步骤
# - 只使用 3 条检索路径 (A+B+C)
# - 适用于实时对话、语音助手等场景
```

### 高精度搜索

```python
# 适用于需要高质量答案的场景

result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do today?",
    mode="reflect",     # 使用反思模式
    strategy="rerank"  # 使用交叉编码器重排
)

# 特点:
# - 延迟较高
# - 使用 LLM 反思增强
# - 使用交叉编码器重排
# - 适用于分析报告、总结等场景
```

### 平衡搜索

```python
# 适用于大多数场景

result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do today?",
    mode="expand",      # 默认扩展模式 (默认)
    strategy="rrf"     # 默认融合策略 (默认)
)

# 特点:
# - 延迟适中
# - 使用 LLM 扩展
# - 6 条检索路径
# - 适用于大多数场景
```

### 搜索策略对比

| 场景 | mode | strategy | 延迟 | 质量 |
|------|------|----------|------|------|
| 实时对话 | direct | rrf | 低 | 中 |
| 大多数场景 | expand | rrf | 中 | 高 |
| 分析报告 | reflect | rerank | 高 | 最高 |

## 6.4 异步处理

### 异步写入

```python
import asyncio

async def async_ingest(client, messages):
    """异步写入"""
    
    # 使用 aiohttp 进行异步请求
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        url = f"{client.base_url}/memory"
        payload = {
            "dataset": "my_app",
            "task": "user_001",
            "store": True,
            "digest": True,
            "messages": messages
        }
        
        async with session.post(url, json=payload) as response:
            result = await response.json()
            return result


async def main():
    # 异步执行多个写入
    messages_list = [
        [{"speaker": "user", "content": "Message 1"}],
        [{"speaker": "user", "content": "Message 2"}],
        [{"speaker": "user", "content": "Message 3"}],
    ]
    
    tasks = [async_ingest(client, msgs) for msgs in messages_list]
    results = await asyncio.gather(*tasks)
    
    for r in results:
        print(r)


asyncio.run(main())
```

### 异步搜索

```python
import asyncio
import aiohttp

async def async_search(client, question):
    """异步搜索"""
    
    async with aiohttp.ClientSession() as session:
        url = f"{client.base_url}/memory/search"
        payload = {
            "dataset": "my_app",
            "task": "user_001",
            "question": question,
            "mode": "expand",
            "strategy": "rrf"
        }
        
        async with session.post(url, json=payload) as response:
            result = await response.json()
            return result


async def main():
    # 同时搜索多个问题
    questions = [
        "What did I do yesterday?",
        "What is my job?",
        "What are my hobbies?"
    ]
    
    tasks = [async_search(client, q) for q in questions]
    results = await asyncio.gather(*tasks)
    
    for q, r in zip(questions, results):
        print(f"Question: {q}")
        print(f"Context: {r['packed_context'][:100]}...")
        print()


asyncio.run(main())
```

## 6.5 自定义 Agent Profile

### 使用不同的 Agent Profile

```python
# Agent Profile 可以影响记忆提取的行为
# 不同的 profile 适用于不同的场景

# Profile 1: 详细模式
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "agent_profile": "detailed",  # 详细提取
    "store": True,
    "digest": True,
    "messages": [...]
}

# Profile 2: 简洁模式
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "agent_profile": "concise",  # 简洁提取
    "store": True,
    "digest": True,
    "messages": [...]
}

# Profile 3: 隐私模式
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "agent_profile": "privacy",  # 隐私保护
    "store": True,
    "digest": True,
    "messages": [...]
}
```

### 创建自定义 Profile

```python
# 在配置文件中定义自定义 profile
# 配置文件: config/agent_profiles.yaml

profiles:
  my_custom_profile:
    extraction:
      entity_types: ["person", "place", "event", "activity"]
      max_entities: 50
      min_confidence: 0.8
    
    fact_generation:
      max_facts: 100
      include_context: true
    
    entity_resolution:
      use_llm: true
      threshold: 0.9
```

## 6.6 记忆更新与删除

### 更新记忆

```python
# MemBrain 不支持直接更新
# 需要删除后重新写入

# 方法 1: 删除会话
def delete_session(session_id):
    """删除特定会话"""
    # 直接删除数据库记录 (需要管理员权限)
    pass

# 方法 2: 归档事实
def archive_fact(fact_id):
    """归档特定事实"""
    # 将 fact 状态改为 archived
    # 需要直接操作数据库
    pass

# 方法 3: 重新消化会话
def redigest_session(session_id):
    """重新消化会话"""
    # 设置会话为未消化状态，然后重新消化
    pass
```

## 6.7 监控与调试

### 监控消化状态

```python
def check_digest_status(dataset, task):
    """检查消化状态"""
    
    # 查询数据库获取状态
    # 需要直接访问数据库
    
    pass


def get_statistics(dataset, task):
    """获取统计信息"""
    
    # 获取实体数量、事实数量、会话数量等
    # 需要直接访问数据库
    
    pass
```

### 调试搜索结果

```python
def debug_search(question, dataset="my_app", task="user_001"):
    """调试搜索结果"""
    
    result = client.search(
        dataset=dataset,
        task=task,
        question=question,
        mode="expand",
        strategy="rerank"  # 使用 rerank 获取更详细的信息
    )
    
    print("=== 搜索结果调试 ===")
    print(f"问题: {question}")
    print(f"Token 数量: {result['packed_token_count']}")
    print(f"事实数量: {len(result['facts'])}")
    print(f"会话数量: {len(result['sessions'])}")
    
    print("\n--- 事实详情 ---")
    for fact in result["facts"]:
        print(f"  ID: {fact['fact_id']}")
        print(f"  文本: {fact['text']}")
        print(f"  来源: {fact['source']}")
        print(f"  分数: {fact['rerank_score']}")
        print(f"  时间: {fact.get('time_info')}")
        print()
    
    print("\n--- 会话详情 ---")
    for session in result["sessions"]:
        print(f"  ID: {session['session_id']}")
        print(f"  主题: {session['subject']}")
        print(f"  分数: {session['score']}")
        print()
```

## 6.8 集成示例

### 与 LangChain 集成

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

def langchain_example():
    """与 LangChain 集成的示例"""
    
    # 1. 搜索记忆
    search_result = client.search(
        dataset="my_app",
        task="user_001",
        question="What is my background?",
        mode="expand",
        strategy="rrf"
    )
    
    context = search_result["packed_context"]
    
    # 2. 创建 LangChain 提示
    prompt = PromptTemplate(
        template="Based on the following context, answer the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["context", "question"]
    )
    
    # 3. 创建链
    llm = ChatOpenAI(model_name="gpt-4")
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 4. 执行
    result = chain.run(context=context, question="What is my background?")
    
    return result
```

### 与 LlamaIndex 集成

```python
from llama_index import GPTSimpleVectorIndex, Document

def llamaindex_example():
    """与 LlamaIndex 集成的示例"""
    
    # 1. 搜索记忆
    search_result = client.search(
        dataset="my_app",
        task="user_001",
        question="What did I do?",
        mode="expand",
        strategy="rrf"
    )
    
    # 2. 转换为 Document
    facts_text = "\n".join([f["text"] for f in search_result["facts"]])
    documents = [Document(facts_text)]
    
    # 3. 创建索引
    index = GPTSimpleVectorIndex(documents)
    
    # 4. 查询
    response = index.query("What did I do?")
    
    return response
```

## 6.9 性能优化建议

### 1. 批量写入

```python
# 批量写入比单条写入更高效
batch_size = 10  # 每批 10 条会话
```

### 2. 选择合适的搜索模式

```python
# 实时对话使用 direct 模式
# 报告生成使用 expand 或 reflect 模式
```

### 3. 调整 top_k

```python
# 根据实际需要调整
# top_k 越大，token 越多，延迟越高
```

### 4. 使用连接池

```python
# 使用 requests.Session() 复用连接
session = requests.Session()
session.post(url, json=payload)
```

## 总结

本步骤介绍了：
- 多数据集管理方法
- 批量写入和并发处理
- 自定义搜索策略的选择
- 异步处理方法
- Agent Profile 的使用
- 记忆更新与删除
- 监控与调试方法
- 与 LangChain、LlamaIndex 的集成
- 性能优化建议

这些高级功能可以帮助构建更复杂、更高效的 MemBrain 应用。


## 36. 07 configuration

# Step 7: 配置参数

## 概述

本步骤详细介绍 MemBrain 的各种配置参数，包括搜索参数、Token 预算、实体解析参数等，帮助用户根据实际需求进行调整。

## 7.1 搜索参数

### 基本搜索配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QA_BM25_FACT_TOP_N` | 20 | BM25 路径返回的事实数量 |
| `QA_EMBED_FACT_TOP_N` | 20 | 向量路径返回的事实数量 |
| `QA_ENTITY_TOP_N` | 5 | 实体树返回的实体数量 |
| `QA_TREE_BEAM_WIDTH` | 3 | 实体树光束宽度 |
| `QA_RERANK_TOP_K` | 12 | 融合后保留的结果数量 |

### 配置示例

```python
# 调整搜索参数
config = {
    "QA_BM25_FACT_TOP_N": 30,      # 增加 BM25 返回数量
    "QA_EMBED_FACT_TOP_N": 30,     # 增加向量返回数量
    "QA_ENTITY_TOP_N": 10,         # 增加实体树返回数量
    "QA_TREE_BEAM_WIDTH": 5,       # 增加光束宽度
    "QA_RERANK_TOP_K": 20,         # 增加融合后保留数量
}
```

### 参数影响

```
参数值对搜索结果的影响:

QA_BM25_FACT_TOP_N
    │
    ├── 值越大 → 初始结果越多 → 可能召回更多相关内容
    │
    └── 值越小 → 初始结果越少 → 速度更快但可能遗漏

QA_EMBED_FACT_TOP_N
    │
    ├── 值越大 → 向量检索结果越多 → 语义相似内容更多
    │
    └── 值越小 → 向量检索结果越少 → 速度更快

QA_ENTITY_TOP_N
    │
    ├── 值越大 → 实体树展开越广 → 相关信息更多
    │
    └── 值越小 → 实体树展开越窄 → 速度更快

QA_RERANK_TOP_K
    │
    ├── 值越大 → 最终结果越多 → 上下文更丰富
    │
    └── 值越小 → 最终结果越少 → 上下文更精简
```

## 7.2 Token 预算

### 上下文预算配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `_FACT_BUDGET_TOKENS` | 4500 | 事实上下文预算 (tokens) |
| `_SESSION_BUDGET_TOKENS` | 1500 | 会话摘要上下文预算 (tokens) |
| `_TOTAL_BUDGET_TOKENS` | 6000 | 总上下文预算 (tokens) |

### Token 预算分配

```
总预算: 6000 tokens
    │
    ├── 事实上下文: 4500 tokens (75%)
    │   │
    │   └── 用于存储从记忆中检索的事实
    │
    └── 会话摘要: 1500 tokens (25%)
        │
        └── 用于存储相关的会话摘要
```

### 调整示例

```python
# 增大上下文预算
config = {
    "_FACT_BUDGET_TOKENS": 5000,      # 事实上下文增加到 5000
    "_SESSION_BUDGET_TOKENS": 2000,   # 会话摘要增加到 2000
    "_TOTAL_BUDGET_TOKENS": 7000,     # 总预算增加到 7000
}

# 减小上下文预算 (用于更快的响应)
config = {
    "_FACT_BUDGET_TOKENS": 3000,      # 事实上下文减少到 3000
    "_SESSION_BUDGET_TOKENS": 1000,   # 会话摘要减少到 1000
    "_TOTAL_BUDGET_TOKENS": 4000,     # 总预算减少到 4000
}
```

### 根据模型选择

```python
# GPT-4 (8K 上下文)
config = {
    "_TOTAL_BUDGET_TOKENS": 6000,  # 留出空间给系统消息
}

# GPT-3.5 Turbo (16K 上下文)
config = {
    "_TOTAL_BUDGET_TOKENS": 12000,  # 可以使用更多上下文
}

# GPT-3.5 Turbo (4K 上下文)
config = {
    "_TOTAL_BUDGET_TOKENS": 3000,  # 减少以适应小上下文
}
```

## 7.3 实体解析参数

### 实体消重配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RESOLVER_JACCARD_THRESHOLD` | 0.9 | Jaccard 相似度阈值 |
| `RESOLVER_ENTROPY_THRESHOLD` | 1.5 | 熵阈值 |
| `RESOLVER_MINHASH_PERMUTATIONS` | 32 | MinHash 置换次数 |
| `RESOLVER_LLM_ENABLED` | True | 是否启用 LLM 语义匹配 |

### 参数详解

```python
# Jaccard 阈值
# 值越接近 1.0，匹配越严格
# 值越小，匹配越宽松，但可能产生误匹配

RESOLVER_JACCARD_THRESHOLD = 0.9  # 严格匹配 (默认)
RESOLVER_JACCARD_THRESHOLD = 0.8  # 宽松匹配

# MinHash 置换次数
# 值越大，结果越准确，但计算越慢
# 值越小，计算越快，但结果可能不够准确

RESOLVER_MINHASH_PERMUTATIONS = 32   # 默认
RESOLVER_MINHASH_PERMUTATIONS = 64   # 更准确
RESOLVER_MINHASH_PERMUTATIONS = 16   # 更快

# LLM 语义匹配
# 启用后，使用 LLM 进行语义级别的实体匹配
# 精度更高，但成本更高

RESOLVER_LLM_ENABLED = True   # 启用 (默认)
RESOLVER_LLM_ENABLED = False # 禁用 (更快)
```

### 三层实体消重策略

```
实体消重流程:

Layer 1: 精确匹配
    │
    ├── 方法: 字符串完全匹配 (忽略大小写)
    ├── 阈值: 100% 匹配
    └── 用途: 消除重复的实体名称

    ▼

Layer 2: MinHash + Jaccard
    │
    ├── 方法: 局部敏感哈希 + Jaccard 相似度
    ├── 阈值: RESOLVER_JACCARD_THRESHOLD (默认 0.9)
    └── 用途: 消除拼写变体、近似名称

    ▼

Layer 3: LLM 语义匹配
    │
    ├── 方法: 使用 LLM 判断语义等价
    ├── 阈值: 语义相似
    └── 用途: 消除同义词、别名

    控制: RESOLVER_LLM_ENABLED
```

## 7.4 提取参数

### 实体提取配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EXTRACTION_MAX_RETRIES` | 3 | 提取最大重试次数 |
| `EXTRACTION_TIMEOUT` | 60 | 提取超时时间 (秒) |

### 事实生成配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FACT_GEN_MAX_RETRIES` | 3 | 事实生成最大重试次数 |
| `FACT_GEN_TIMEOUT` | 120 | 事实生成超时时间 (秒) |

## 7.5 数据库配置

### 连接池配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DB_POOL_SIZE` | 10 | 连接池大小 |
| `DB_MAX_OVERFLOW` | 20 | 最大溢出连接数 |
| `DB_POOL_TIMEOUT` | 30 | 连接超时时间 (秒) |

### 配置示例

```python
# 数据库连接配置
db_config = {
    "DB_POOL_SIZE": 10,         # 10 个连接
    "DB_MAX_OVERFLOW": 20,      # 最多 20 个溢出
    "DB_POOL_TIMEOUT": 30,      # 30 秒超时
}
```

## 7.6 LLM 配置

### 模型选择

```python
# 可用的 LLM 提供商

# OpenAI
llm_config = {
    "provider": "openai",
    "model": "gpt-4",           # 或 "gpt-3.5-turbo"
    "temperature": 0.0,         # 温度参数
}

# Anthropic
llm_config = {
    "provider": "anthropic",
    "model": "claude-3-opus",   # 或 "claude-3-sonnet"
    "temperature": 0.0,
}

# Azure OpenAI
llm_config = {
    "provider": "azure",
    "model": "gpt-4",
    "deployment_name": "gpt-4-deployment",
}
```

### 嵌入模型配置

```python
# 嵌入模型配置
embed_config = {
    "provider": "openai",
    "model": "text-embedding-ada-002",  # 或其他嵌入模型
    "dimensions": 1536,                  # 向量维度
}
```

## 7.7 日志配置

### 日志级别

```python
import logging

# 设置日志级别
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG / INFO / WARNING / ERROR
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 组件日志级别
logging.getLogger("membrain.search").setLevel(logging.INFO)
logging.getLogger("membrain.ingest").setLevel(logging.DEBUG)
logging.getLogger("membrain.entity").setLevel(logging.WARNING)
```

### 日志输出

```python
# 输出到文件
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("membrain.log"),
        logging.StreamHandler()
    ]
)
```

## 7.8 环境变量配置

### 常用环境变量

```bash
# 数据库配置
export DATABASE_URL="postgresql://user:password@localhost:5432/membrain"

# LLM 配置
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# 服务器配置
export MEMBRAIN_HOST="0.0.0.0"
export MEMBRAIN_PORT="9574"

# 日志配置
export LOG_LEVEL="INFO"
```

### 配置加载

```python
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 读取配置
database_url = os.getenv("DATABASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")
membrain_port = os.getenv("MEMBRAIN_PORT", "9574")
```

## 7.9 生产环境配置建议

### 开发环境

```python
# 开发环境配置
config = {
    "DEBUG": True,
    "LOG_LEVEL": "DEBUG",
    "DB_POOL_SIZE": 5,
    "QA_BM25_FACT_TOP_N": 10,
    "QA_EMBED_FACT_TOP_N": 10,
    "RESOLVER_LLM_ENABLED": False,  # 开发时禁用 LLM
}
```

### 测试环境

```python
# 测试环境配置
config = {
    "DEBUG": False,
    "LOG_LEVEL": "INFO",
    "DB_POOL_SIZE": 10,
    "QA_BM25_FACT_TOP_N": 20,
    "QA_EMBED_FACT_TOP_N": 20,
    "RESOLVER_LLM_ENABLED": True,
}
```

### 生产环境

```python
# 生产环境配置
config = {
    "DEBUG": False,
    "LOG_LEVEL": "WARNING",
    "DB_POOL_SIZE": 20,
    "DB_MAX_OVERFLOW": 40,
    "QA_BM25_FACT_TOP_N": 30,
    "QA_EMBED_FACT_TOP_N": 30,
    "QA_RERANK_TOP_K": 15,
    "RESOLVER_LLM_ENABLED": True,
    "_TOTAL_BUDGET_TOKENS": 6000,
}
```

## 7.10 配置优先级

```
配置优先级 (从高到低):

1. 环境变量
   │
   └── 最高优先级，直接覆盖所有配置
   
2. 配置文件
   │
   └── config.yaml / config.json
   
3. 代码默认配置
   │
   └── config/default.py
   
4. 硬编码默认值
   │
   └── 最低优先级
```

## 总结

本步骤介绍了：
- 搜索参数的配置和影响
- Token 预算的分配和调整
- 实体解析参数的配置
- 提取参数的配置
- 数据库和 LLM 的配置
- 日志配置
- 环境变量配置
- 不同环境的配置建议
- 配置优先级

通过合理配置这些参数，可以根据实际需求优化 MemBrain 的性能和功能。


## 37. 08 error handling

# Step 8: 错误处理与故障排除

## 概述

本步骤详细介绍 MemBrain 的常见错误处理方法、故障排除技巧，以及最佳实践，帮助用户快速定位和解决问题。

## 8.1 常见错误

### API 错误

#### 错误 1: 参数错误 (400)

```json
{
    "detail": "messages required when store=True"
}
```

**原因**: 当 `store=True` 时，必须提供 `messages` 参数

**解决方案**:

```python
# 错误示例
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": []  # 空消息列表
}

# 正确示例
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "Hello"}
    ]
}
```

#### 错误 2: 缺少必要参数 (400)

```json
{
    "detail": "at least one of store or digest must be True"
}
```

**原因**: `store` 和 `digest` 不能同时为 `False`

**解决方案**:

```python
# 错误示例
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": False,
    "digest": False,
    "messages": [...]
}

# 正确示例 - 至少一个为 True
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": False,  # 或 True
    "messages": [...]
}
```

#### 错误 3: 数据集/任务不存在 (404)

```json
{
    "detail": "Task 'user_999' not found in dataset 'personamem_v2'"
}
```

**原因**: 搜索时指定的 task 不存在

**解决方案**:

```python
# 确保先写入数据，再搜索
# 或者使用 store=True 创建新数据

payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": True,
    "digest": False,
    "messages": [{"speaker": "user", "content": "init"}]
}

# 先创建数据
result = requests.post(url, json=payload)

# 然后再搜索
search_result = client.search(
    dataset="personamem_v2",
    task="user_001",
    question="What did I do?"
)
```

### 网络错误

#### 错误 4: 连接失败

```python
# 错误
requests.exceptions.ConnectionError: [Errno 111] Connection refused

# 原因: MemBrain 服务器未启动
# 解决方案:
# 1. 启动服务器
# python -m uvicorn membrain.api.server:app --host 0.0.0.0 --port 9574

# 2. 检查端口是否被占用
# netstat -tulpn | grep 9574
```

#### 错误 5: 请求超时

```python
# 错误
requests.exceptions.Timeout: Request timed out

# 原因: 请求处理时间过长
# 解决方案:
# 1. 增加超时时间
response = requests.post(url, json=payload, timeout=60)

# 2. 使用更快的搜索模式
payload = {
    ...
    "mode": "direct",  # 跳过 LLM 扩展
    "strategy": "rrf"  # 快速融合
}
```

### LLM 错误

#### 错误 6: API 密钥无效

```python
# 错误
openai.error.AuthenticationError: Invalid API Key

# 解决方案:
# 1. 检查环境变量
import os
print(os.getenv("OPENAI_API_KEY"))

# 2. 设置正确的 API 密钥
os.environ["OPENAI_API_KEY"] = "sk-..."
```

#### 错误 7: API 配额耗尽

```python
# 错误
openai.error.RateLimitError: You exceeded your current quota

# 解决方案:
# 1. 检查 API 使用量
# 2. 升级订阅计划
# 3. 等待配额重置
```

## 8.2 错误处理示例

### 基础错误处理

```python
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout

def safe_request(url, payload, max_retries=3):
    """带重试的请求"""
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            
            # 检查 HTTP 错误
            if response.status_code == 400:
                error = response.json()
                print(f"参数错误: {error.get('detail')}")
                return None
                
            elif response.status_code == 404:
                error = response.json()
                print(f"资源不存在: {error.get('detail')}")
                return None
                
            elif response.status_code >= 500:
                print(f"服务器错误: {response.status_code}")
                continue
                
            response.raise_for_status()
            return response.json()
            
        except ConnectionError:
            print(f"连接失败 (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
                
        except Timeout:
            print(f"请求超时 (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
                
        except HTTPError as e:
            print(f"HTTP 错误: {e}")
            break
            
    return None
```

### 高级错误处理类

```python
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout
from typing import Optional, Dict, Any
import time

class MemBrainError(Exception):
    """MemBrain 基础异常"""
    pass

class MemBrainAPIError(MemBrainError):
    """API 错误"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")

class MemBrainConnectionError(MemBrainError):
    """连接错误"""
    pass

class MemBrainTimeoutError(MemBrainError):
    """超时错误"""
    pass


class MemBrainClient:
    """带错误处理的 MemBrain 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:9574/api"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        payload: Optional[Dict] = None,
        max_retries: int = 3,
        timeout: int = 30
    ) -> Optional[Dict]:
        """发送请求，带错误处理"""
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                if method == "POST":
                    response = self.session.post(
                        url, 
                        json=payload, 
                        timeout=timeout
                    )
                else:
                    raise ValueError(f"不支持的方法: {method}")
                
                # 检查状态码
                if response.status_code == 400:
                    error = response.json()
                    raise MemBrainAPIError(400, error.get('detail', '参数错误'))
                    
                elif response.status_code == 404:
                    error = response.json()
                    raise MemBrainAPIError(404, error.get('detail', '资源不存在'))
                    
                elif response.status_code >= 500:
                    print(f"服务器错误 {response.status_code}，重试 {attempt + 1}/{max_retries}")
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.ConnectionError as e:
                if attempt == max_retries - 1:
                    raise MemBrainConnectionError(f"连接失败: {e}")
                time.sleep(2)
                
            except requests.exceptions.Timeout as e:
                if attempt == max_retries - 1:
                    raise MemBrainTimeoutError(f"请求超时: {e}")
                time.sleep(2)
                
            except HTTPError as e:
                raise MemBrainAPIError(e.response.status_code, str(e))
                
        return None
    
    def store_and_digest(self, dataset: str, task: str, messages: list) -> Optional[Dict]:
        """存储并消化"""
        return self._request(
            "POST", 
            "memory",
            {
                "dataset": dataset,
                "task": task,
                "store": True,
                "digest": True,
                "messages": messages
            }
        )
    
    def search(
        self, 
        dataset: str, 
        task: str, 
        question: str,
        **kwargs
    ) -> Optional[Dict]:
        """搜索"""
        return self._request(
            "POST",
            "memory/search",
            {
                "dataset": dataset,
                "task": task,
                "question": question,
                **kwargs
            }
        )


# 使用示例
try:
    client = MemBrainClient()
    
    # 存储
    result = client.store_and_digest(
        dataset="my_app",
        task="user_001",
        messages=[{"speaker": "user", "content": "Hello"}]
    )
    print(f"存储成功: {result}")
    
    # 搜索
    import time
    time.sleep(5)  # 等待消化
    
    result = client.search(
        dataset="my_app",
        task="user_001",
        question="What did I say?"
    )
    print(f"搜索成功: {result}")
    
except MemBrainAPIError as e:
    print(f"API 错误: {e.status_code} - {e.message}")
    
except MemBrainConnectionError as e:
    print(f"连接错误: {e}")
    
except MemBrainTimeoutError as e:
    print(f"超时错误: {e}")
    
except MemBrainError as e:
    print(f"MemBrain 错误: {e}")
```

## 8.3 故障排除

### 问题 1: 搜索结果为空

**症状**: 搜索返回空结果

**可能原因**:
1. 数据未被消化
2. 数据集/任务不匹配
3. 搜索问题与存储内容不相关

**排查步骤**:

```python
# 步骤 1: 检查数据是否存储
result = client.store_and_digest(
    dataset="my_app",
    task="user_001",
    messages=[{"speaker": "user", "content": "I love playing guitar"}]
)
print(f"存储状态: {result['status']}")

# 步骤 2: 等待消化完成
import time
time.sleep(5)

# 步骤 3: 使用更广泛的搜索词
result = client.search(
    dataset="my_app",
    task="user_001",
    question="music",  # 尝试更广泛的关键词
    mode="expand",
    top_k=20
)
print(f"结果数量: {len(result['facts'])}")
```

### 问题 2: 消化失败

**症状**: 写入成功但搜索无结果

**可能原因**:
1. LLM API 问题
2. 提取超时
3. 实体/事实生成失败

**排查步骤**:

```python
import logging

# 开启详细日志
logging.basicConfig(level=logging.DEBUG)

# 检查服务器日志
# 查找 "extract" 或 "digest" 相关的日志

# 手动触发消化
result = client.store_and_digest(
    dataset="my_app",
    task="user_001",
    messages=[{"speaker": "user", "content": "Test"}]
)

# 检查是否有 digest_queued
print(f"状态: {result.get('status')}")
```

### 问题 3: 搜索结果不相关

**症状**: 返回的事实与问题不相关

**可能原因**:
1. 搜索模式选择不当
2. top_k 太小
3. 融合策略不适当

**解决方案**:

```python
# 使用更精确的搜索模式
result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do yesterday?",
    mode="reflect",     # 使用反思模式
    strategy="rerank",  # 使用重排序
    top_k=20            # 增加返回数量
)
```

### 问题 4: 响应速度慢

**症状**: 搜索响应时间过长

**可能原因**:
1. LLM 调用延迟
2. 数据库查询慢
3. 网络延迟

**解决方案**:

```python
# 使用更快的搜索模式
result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do?",
    mode="direct",   # 跳过 LLM 扩展
    strategy="rrf"   # 快速融合
)
```

### 问题 5: Token 预算超限

**症状**: 返回结果被截断

**可能原因**:
1. top_k 太大
2. 事实文本太长

**解决方案**:

```python
# 减少 top_k
result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do?",
    top_k=5  # 减少返回数量
)

# 使用 direct 模式
result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do?",
    mode="direct"
)
```

## 8.4 日志分析

### 开启调试日志

```python
import logging

# 开启所有模块的调试日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 开启特定模块的日志
logging.getLogger("membrain.search").setLevel(logging.DEBUG)
logging.getLogger("membrain.ingest").setLevel(logging.DEBUG)
logging.getLogger("membrain.entity").setLevel(logging.DEBUG)
```

### 常见日志关键词

| 关键词 | 说明 |
|--------|------|
| `entity-extractor` | 实体提取日志 |
| `fact-generator` | 事实生成日志 |
| `entity-resolver` | 实体消重日志 |
| `search-query` | 搜索查询日志 |
| `retrieval` | 检索日志 |

### 分析日志示例

```bash
# 搜索实体提取相关日志
grep "entity-extractor" membrain.log

# 搜索错误
grep "ERROR" membrain.log

# 搜索特定时间段的日志
grep "2024-01-15 14:3" membrain.log
```

## 8.5 健康检查

### API 健康检查

```python
import requests

def health_check(base_url="http://localhost:9574"):
    """检查 MemBrain 服务健康状态"""
    
    try:
        # 检查根路径
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"根路径状态: {response.status_code}")
        
        # 检查 docs
        response = requests.get(f"{base_url}/docs", timeout=5)
        print(f"文档路径状态: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False


# 执行健康检查
if health_check():
    print("MemBrain 服务正常")
else:
    print("MemBrain 服务异常")
```

### 数据库健康检查

```python
# 需要直接访问数据库
# 检查表是否存在

def check_database():
    """检查数据库健康状态"""
    
    # 检查表是否存在
    tables = ["datasets", "tasks", "chat_sessions", "chat_messages", 
              "entities", "facts", "fact_refs", "entity_trees"]
    
    for table in tables:
        # 查询表是否存在
        # SELECT * FROM information_schema.tables WHERE table_name = ?
        pass
```

## 8.6 性能监控

### 监控指标

```python
import time
from functools import wraps

def monitor_performance(func):
    """性能监控装饰器"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"{func.__name__} 执行时间: {duration:.2f}秒")
        
        return result
    
    return wrapper


# 使用示例
@monitor_performance
def slow_search():
    result = client.search(
        dataset="my_app",
        task="user_001",
        question="What did I do?",
        mode="expand"
    )
    return result


result = slow_search()
```

## 8.7 常见问题 FAQ

### Q1: 为什么搜索不到刚写入的数据?

**A**: 写入是异步消化的，需要等待几秒后再搜索。建议:
```python
result = client.store_and_digest(...)
time.sleep(5)  # 等待消化
result = client.search(...)
```

### Q2: 如何提高搜索精度?

**A**: 
1. 使用 `mode="reflect"` 模式
2. 使用 `strategy="rerank"` 策略
3. 调整 `top_k` 参数

### Q3: 如何加快搜索速度?

**A**:
1. 使用 `mode="direct"` 模式
2. 使用 `strategy="rrf"` 策略
3. 减少 `top_k` 参数

### Q4: 实体消重不起作用?

**A**: 检查配置:
```python
RESOLVER_LLM_ENABLED = True  # 确保启用 LLM
RESOLVER_JACCARD_THRESHOLD = 0.9  # 确保阈值合理
```

### Q5: 内存占用过高?

**A**:
1. 减少 `top_k` 参数
2. 减少 token 预算
3. 使用更小的嵌入模型

## 总结

本步骤介绍了：
- API 常见错误及解决方案
- 网络错误处理
- LLM 错误处理
- 完整的错误处理示例
- 常见问题的故障排除步骤
- 日志分析方法
- 健康检查方法
- 性能监控技巧
- 常见问题 FAQ

通过这些错误处理和故障排除方法，可以快速定位和解决 MemBrain 使用中的问题。


## 38. 09 summary

# Step 9: 总结与最佳实践

## 概述

本步骤是 MemBrain 使用指南的最终总结，涵盖核心概念、最佳实践、性能优化建议，以及进一步学习的资源。

## 9.1 核心概念总结

### 数据模型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据模型关系                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Dataset ──────────▶ Task ──────────▶ Session ──────────▶ Message       │
│                                                                             │
│   Entity ◀────────── Fact ◀─────────── FactRef                            │
│                                                                             │
│   EntityTree (实体按 Aspect 组织成树结构)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 核心流程

```
写入流程:
  Messages → Session → Extraction → Entity Resolution → Persistence → Entity Tree
  
搜索流程:
  Question → Query Expansion → Multi-Path Retrieval → RRF/Rerank → Context Packing → LLM Answer
```

### 搜索模式

| 模式 | 说明 | 检索路径 | 适用场景 |
|------|------|----------|----------|
| `direct` | 直接模式 | 3 条 | 低延迟场景 |
| `expand` | 扩展模式 | 6 条 | 大多数场景 |
| `reflect` | 反思模式 | 6+2 条 | 高召回场景 |

### 融合策略

| 策略 | 方法 | 速度 | 精度 |
|------|------|------|------|
| `rrf` | 互惠排名融合 | 快 | 中 |
| `rerank` | 交叉编码器重排 | 慢 | 高 |

## 9.2 最佳实践

### 写入最佳实践

#### 1. 合理的消息格式

```python
# ✅ 好的示例: 完整的上下文信息
messages = [
    {"speaker": "user", "content": "I had lunch with John at Luigi's Pizza yesterday."},
    {"speaker": "assistant", "content": "That sounds nice!"},
    {"speaker": "user", "content": "We've been going there for years. It's his favorite."}
]

# ❌ 差的示例: 缺少上下文
messages = [
    {"speaker": "user", "content": "Had lunch"}
]
```

#### 2. 提供时间信息

```python
# ✅ 好的示例: 提供时间信息
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "session_time": "2024-01-15T14:30:00",
    "messages": [
        {"speaker": "user", "content": "Went to the gym", "message_time": "2024-01-15T08:00:00"}
    ]
}
```

#### 3. 批量写入

```python
# ✅ 好的示例: 批量处理
for session in sessions:
    client.store_and_digest(dataset, task, session)
    time.sleep(1)  # 避免过快
```

### 搜索最佳实践

#### 1. 选择合适的搜索模式

```python
# 实时对话 - 低延迟
result = client.search(
    dataset, task, question,
    mode="direct",
    strategy="rrf"
)

# 大多数场景 - 平衡
result = client.search(
    dataset, task, question,
    mode="expand",
    strategy="rrf"
)

# 分析报告 - 高精度
result = client.search(
    dataset, task, question,
    mode="reflect",
    strategy="rerank"
)
```

#### 2. 优化问题表述

```python
# ✅ 好的示例: 具体明确
"What did I do last weekend?"

# ❌ 差的示例: 模糊笼统
"What did I do?"
```

#### 3. 合理设置 top_k

```python
# 需要详细信息
top_k = 20

# 快速摘要
top_k = 5
```

### 性能优化最佳实践

#### 1. 使用连接池

```python
# ✅ 好的示例
session = requests.Session()
for _ in range(100):
    session.post(url, json=payload)

# ❌ 差的示例
for _ in range(100):
    requests.post(url, json=payload)
```

#### 2. 异步处理

```python
# ✅ 好的示例: 批量异步处理
async def batch_search(questions):
    tasks = [async_search(q) for q in questions]
    return await asyncio.gather(*tasks)
```

#### 3. 缓存常用查询

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(dataset, task, question):
    return client.search(dataset, task, question)
```

## 9.3 架构建议

### 单用户应用

```
┌─────────────┐
│   客户端    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    MemBrain 服务器                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   API      │  │   搜索     │  │   写入     │     │
│  │   服务     │  │   引擎     │  │   引擎     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                           │                │               │
│                           └────────┬───────┘               │
│                                    ▼                       │
│                         ┌─────────────────┐               │
│                         │   PostgreSQL   │               │
│                         │   + pgvector   │               │
│                         └─────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 多租户应用

```
┌─────────────────────────────────────────────────────────────────┐
│                         负载均衡器                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │实例 1   │   │实例 2   │   │实例 3   │
   │         │   │         │   │         │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              ┌─────────────────┐
              │  PostgreSQL     │
              │  (多租户隔离)   │
              └─────────────────┘
```

## 9.4 安全建议

### 1. API 密钥保护

```python
# ✅ 好的示例: 使用环境变量
import os
api_key = os.getenv("OPENAI_API_KEY")

# ❌ 差的示例: 硬编码密钥
api_key = "sk-xxx..."  # 切勿这样做
```

### 2. 数据隔离

```python
# 确保用户只能访问自己的数据
def search(user_id, question):
    return client.search(
        dataset="app_data",
        task=user_id,  # 使用用户 ID 作为 task
        question=question
    )
```

### 3. 输入验证

```python
import re

def validate_input(data):
    # 验证数据集名称
    if not re.match(r'^[a-zA-Z0-9_]+$', data['dataset']):
        raise ValueError("Invalid dataset name")
    
    # 验证任务名称
    if not re.match(r'^[a-zA-Z0-9_]+$', data['task']):
        raise ValueError("Invalid task name")
    
    return True
```

## 9.5 监控与运维

### 关键指标

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| API 响应时间 | 搜索/写入延迟 | > 5s |
| 错误率 | 失败请求比例 | > 1% |
| CPU 使用率 | 服务器负载 | > 80% |
| 内存使用率 | 内存占用 | > 85% |
| 数据库连接 | 连接池使用 | > 90% |

### 日志监控

```python
# 设置日志告警
import logging
from logging.handlers import SMTPHandler

logger = logging.getLogger("membrain")

# 错误告警
error_handler = SMTPHandler(
    fromaddr="membrain@example.com",
    toaddrs=["admin@example.com"],
    subject="MemBrain Error Alert"
)
logger.addHandler(error_handler)
logger.setLevel(logging.ERROR)
```

## 9.6 常见陷阱

### 陷阱 1: 忽略异步消化

```python
# ❌ 错误: 写入后立即搜索
result = client.store_and_digest(...)
result = client.search(...)  # 可能返回空

# ✅ 正确: 等待消化完成
result = client.store_and_digest(...)
time.sleep(5)  # 等待消化
result = client.search(...)
```

### 陷阱 2: 不正确的任务隔离

```python
# ❌ 错误: 多个用户共享 task
client.store_and_digest("app_data", "shared_task", messages)

# ✅ 正确: 每个用户独立 task
client.store_and_digest("app_data", f"user_{user_id}", messages)
```

### 陷阱 3: 过度使用资源

```python
# ❌ 错误: 大量请求
for question in questions:
    client.search(...)  # 可能触发速率限制

# ✅ 正确: 控制请求频率
for question in questions:
    client.search(...)
    time.sleep(1)
```

### 陷阱 4: 不验证输入

```python
# ❌ 错误: 直接使用用户输入
client.store_and_digest(user_input["dataset"], ...)

# ✅ 正确: 验证输入
if not validate_dataset(user_input["dataset"]):
    raise ValueError("Invalid dataset")
```

## 9.7 扩展阅读

### 相关文档

| 文档 | 说明 |
|------|------|
| [搜索过程详解](../search_stages/README.md) | 11 个阶段的搜索过程 |
| [写入过程详解](../ingest_stages/README.md) | 5 个阶段的写入过程 |
| [交互过程详解](../interaction/README.md) | 6 个阶段的交互过程 |

### API 参考

- `POST /api/memory` - 存储和消化
- `POST /api/memory/search` - 搜索记忆

### 代码位置

- API 入口: `membrain/api/server.py`
- 搜索核心: `membrain/memory/application/search/`
- 写入核心: `membrain/memory/application/ingest/`

## 9.8 快速参考

### 常用操作

```python
# 1. 存储并消化
client.store_and_digest(dataset, task, messages)

# 2. 搜索
client.search(dataset, task, question)

# 3. 等待消化
time.sleep(5)

# 4. 使用 LLM 生成答案
response = llm.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}]
)
```

### 参数速查

```python
# 搜索参数
{
    "mode": "direct" | "expand" | "reflect",  # 搜索模式
    "strategy": "rrf" | "rerank",              # 融合策略
    "top_k": 10                                # 返回数量
}
```

## 9.9 总结

MemBrain 是一个功能强大的记忆系统，通过以下核心能力实现智能记忆管理：

1. **结构化存储**: 将非结构化对话转化为实体、事实和实体树
2. **智能检索**: 多路径检索 + RRF/Rerank 融合
3. **上下文感知**: 根据问题动态构建上下文
4. **可扩展性**: 支持多种搜索模式和策略

### 关键要点

```
✅ 写入数据时提供完整上下文
✅ 搜索前等待消化完成
✅ 选择合适的搜索模式
✅ 使用错误处理
✅ 监控性能指标
✅ 遵循安全最佳实践
```

### 下一步

- 尝试运行示例代码
- 根据实际需求调整配置
- 阅读详细的阶段文档
- 参与社区讨论

---

**文档版本**: 1.0  
**最后更新**: 2024-01-15


## 39. README

# 使用指南 - 分步教程

本文档提供 MemBrain 的分步使用教程，从基础概念到高级功能。

## 目录

| 步骤 | 名称 | 说明 |
|:----:|------|------|
| 1 | [系统架构与数据模型](01_system_architecture.md) | 理解系统架构和核心数据模型 |
| 2 | [API 接口与服务器启动](02_api_endpoints.md) | 了解 API 端点和启动服务器 |
| 3 | [写入数据](03_write_data.md) | 学习如何写入数据到 MemBrain |
| 4 | [搜索数据](04_search_data.md) | 学习如何搜索和检索记忆 |
| 5 | [完整使用示例](05_complete_examples.md) | 完整的代码示例和流程 |
| 6 | [高级功能](06_advanced_features.md) | 多数据集、批量处理、异步等 |
| 7 | [配置参数](07_configuration.md) | 配置参数详解 |
| 8 | [错误处理与故障排除](08_error_handling.md) | 常见错误和解决方案 |
| 9 | [总结与最佳实践](09_summary.md) | 总结和最佳实践指南 |

## 快速开始

### 1. 启动服务器

```bash
cd /home/project/MemBrain
python -m uvicorn membrain.api.server:app --host 0.0.0.0 --port 9574
```

### 2. 写入数据

```python
import requests

response = requests.post(
    "http://localhost:9574/api/memory",
    json={
        "dataset": "my_app",
        "task": "user_001",
        "store": True,
        "digest": True,
        "messages": [
            {"speaker": "user", "content": "I had lunch with John yesterday."}
        ]
    }
)
print(response.json())
```

### 3. 等待消化

```python
import time
time.sleep(5)  # 等待异步消化完成
```

### 4. 搜索记忆

```python
response = requests.post(
    "http://localhost:9574/api/memory/search",
    json={
        "dataset": "my_app",
        "task": "user_001",
        "question": "What did I do yesterday?"
    }
)
result = response.json()
print(result["packed_context"])
```

## 学习路径

```
Step 1: 系统架构
    │
    ▼
Step 2: API 接口
    │
    ▼
Step 3: 写入数据 ──────▶ Step 5: 完整示例
    │                         │
    ▼                         ▼
Step 4: 搜索数据        Step 6: 高级功能
    │                         │
    ▼                         ▼
Step 7: 配置参数        Step 8: 错误处理
    │                         │
    └─────────▶ Step 9: 总结与最佳实践 ◀─────────┘
```

## 每步重点

| 步骤 | 重点 |
|------|------|
| Step 1 | 理解数据模型关系 |
| Step 2 | 掌握 API 端点 |
| Step 3 | 学会写入数据 |
| Step 4 | 掌握搜索技巧 |
| Step 5 | 实践完整流程 |
| Step 6 | 了解高级功能 |
| Step 7 | 学会配置优化 |
| Step 8 | 学会故障排除 |
| Step 9 | 遵循最佳实践 |

