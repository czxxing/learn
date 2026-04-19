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