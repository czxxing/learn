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