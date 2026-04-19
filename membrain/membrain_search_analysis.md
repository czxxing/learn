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