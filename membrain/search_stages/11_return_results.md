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