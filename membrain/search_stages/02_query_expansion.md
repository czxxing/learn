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