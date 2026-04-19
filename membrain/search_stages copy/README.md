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