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