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