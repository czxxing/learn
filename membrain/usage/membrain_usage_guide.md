# MemBrain 完整使用指南

## 概述

本指南详细介绍 MemBrain 的完整使用过程，包括 API 接口、数据格式、以及典型使用场景。MemBrain 是一个记忆系统，用于存储和检索用户对话中提取的结构化记忆。

## 目录

1. [系统架构概览](#1-系统架构概览)
2. [API 接口](#2-api-接口)
3. [写入数据 (Ingest)](#3-写入数据-ingest)
4. [搜索数据 (Search)](#4-搜索数据-search)
5. [完整使用示例](#5-完整使用示例)
6. [高级功能](#6-高级功能)
7. [配置参数](#7-配置参数)

---

## 1. 系统架构概览

### 1.1 核心组件

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

### 1.2 数据模型

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

---

## 2. API 接口

MemBrain 提供以下主要 API 端点：

### 2.1 接口列表

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/memory` | POST | 存储和/或消化对话 |
| `/api/memory/search` | POST | 搜索记忆 |

### 2.2 服务器启动

```bash
# 启动 MemBrain 服务器
cd /home/project/MemBrain
python -m uvicorn membrain.api.server:app --host 0.0.0.0 --port 9574
```

---

## 3. 写入数据 (Ingest)

### 3.1 API 请求格式

**端点**: `POST /api/memory`

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

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

### 3.2 请求示例

```python
import requests
import json

url = "http://localhost:9574/api/memory"

# 准备请求数据
payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": True,
    "digest": True,
    "session_time": "2024-01-15T10:30:00",
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

# 发送请求
response = requests.post(url, json=payload)
result = response.json()

print(json.dumps(result, indent=2))
```

### 3.3 响应格式

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

### 3.4 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `dataset` | string | 是 | 数据集名称，如 "personamem_v2" |
| `task` | string | 是 | 任务标识，如 "user_001" |
| `messages` | array | 是 | 消息列表 |
| `store` | boolean | 否 | 是否存储原始消息，默认 True |
| `digest` | boolean | 否 | 是否提取记忆，默认 True |
| `session_time` | string | 否 | 会话时间 (ISO 格式) |
| `agent_profile` | string | 否 | Agent 配置档案 |

### 3.5 消息speaker选项

```python
# 有效的 speaker 值
"user"       # 用户消息
"assistant"  # AI 助手消息
"system"    # 系统消息（如果有）
```

### 3.6 写入模式

```python
# 模式 1: 存储并消化 (最常用)
payload = {
    "store": True,
    "digest": True,
}

# 模式 2: 仅存储
payload = {
    "store": True,
    "digest": False,
}

# 模式 3: 仅消化 (处理已有消息)
payload = {
    "store": False,
    "digest": True,
}
```

---

## 4. 搜索数据 (Search)

### 4.1 API 请求格式

**端点**: `POST /api/memory/search`

```python
class MemorySearchRequest(BaseModel):
    dataset: str                          # 数据集名称
    task: str                             # 任务标识
    question: str                         # 搜索问题
    top_k: Optional[int] = None          # 返回结果数量
    strategy: Optional[str] = None       # 融合策略: "rrf" | "rerank"
    mode: Optional[str] = None           # 搜索模式: "direct" | "expand" | "reflect"
```

### 4.2 请求示例

```python
import requests
import json

url = "http://localhost:9574/api/memory/search"

# 准备请求数据
payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What did the user do yesterday?",
    "top_k": 10,
    "strategy": "rrf",
    "mode": "expand"
}

# 发送请求
response = requests.post(url, json=payload)
result = response.json()

# 打印结果
print("Packed Context:")
print(result["packed_context"])
print("\n" + "="*50)
print(f"Token Count: {result['packed_token_count']}")
print(f"Fact Count: {len(result['facts'])}")
```

### 4.3 响应格式

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

### 4.4 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `packed_context` | string | 格式化上下文，可直接用于 LLM |
| `packed_token_count` | int | 估算的 token 数量 |
| `fact_ids` | array | 包含在 packed_context 中的事实 ID 列表 |
| `facts` | array | 事实详情列表 |
| `sessions` | array | 相关的会话摘要列表 |
| `raw_messages` | array | 预留：原始消息列表 |

### 4.5 搜索模式

```python
# 模式说明
mode = "direct"   # 直接模式，跳过 LLM 扩展，3 条检索路径
mode = "expand"   # 扩展模式（默认），使用 LLM 扩展，6 条检索路径
mode = "reflect"  # 反思模式，使用 LLM 反思增强
```

### 4.6 融合策略

```python
# 策略说明
strategy = "rrf"     # Reciprocal Rank Fusion (默认)，快速但精度中等
strategy = "rerank"  # Cross-encoder Rerank，慢但精度高
```

---

## 5. 完整使用示例

### 5.1 完整流程示例

```python
import requests
import time

BASE_URL = "http://localhost:9574/api"

def store_and_digest():
    """步骤 1: 存储并消化对话"""
    
    url = f"{BASE_URL}/memory"
    
    payload = {
        "dataset": "personamem_v2",
        "task": "user_001",
        "store": True,
        "digest": True,
        "session_time": "2024-01-15T14:30:00",
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
                "content": "I had the carbonara and John ordered a pizza. He said it's his favorite restaurant. We've been going there for years."
            },
            {
                "speaker": "assistant",
                "content": "It must be a special place for you both!"
            },
            {
                "speaker": "user",
                "content": "Yes! We actually met there for the first time three years ago. Great memories."
            }
        ]
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    print("=== 存储结果 ===")
    print(f"Status: {result['status']}")
    print(f"Session ID: {result['session_id']}")
    print(f"Task PK: {result['task_pk']}")
    
    return result


def search_memory():
    """步骤 2: 搜索记忆"""
    
    url = f"{BASE_URL}/memory/search"
    
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
    
    print("\n=== 搜索结果 ===")
    print(f"Token Count: {result['packed_token_count']}")
    print(f"Fact Count: {len(result['facts'])}")
    print(f"Session Count: {len(result['sessions'])}")
    
    print("\n--- Facts ---")
    for fact in result["facts"]:
        print(f"  - {fact['text']} (score: {fact['rerank_score']:.2f})")
    
    print("\n--- Sessions ---")
    for session in result["sessions"]:
        print(f"  - {session['subject']}: {session['content'][:50]}...")
    
    print("\n--- Packed Context ---")
    print(result["packed_context"])
    
    return result


def generate_answer(search_result):
    """步骤 3: 使用 LLM 生成答案"""
    
    # 这里使用 OpenAI API 作为示例
    from openai import OpenAI
    
    client = OpenAI(api_key="your-api-key")
    
    context = search_result["packed_context"]
    question = "What did the user do yesterday?"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    
    answer = response.choices[0].message.content
    print("\n=== LLM 答案 ===")
    print(answer)
    
    return answer


# 执行完整流程
if __name__ == "__main__":
    # 1. 存储并消化对话
    store_result = store_and_digest()
    
    # 等待消化完成（异步）
    print("\n等待消化完成...")
    time.sleep(5)
    
    # 2. 搜索记忆
    search_result = search_memory()
    
    # 3. 生成答案
    answer = generate_answer(search_result)
```

### 5.2 curl 命令示例

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

# 搜索
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
```

### 5.3 Python 完整脚本

```python
#!/usr/bin/env python3
"""
MemBrain 完整使用示例
"""

import requests
import json
import time
from datetime import datetime

class MemBrainClient:
    """MemBrain 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:9574/api"):
        self.base_url = base_url
        
    def store_and_digest(
        self,
        dataset: str,
        task: str,
        messages: list[dict],
        session_time: str = None
    ) -> dict:
        """存储并消化对话"""
        
        url = f"{self.base_url}/memory"
        
        payload = {
            "dataset": dataset,
            "task": task,
            "store": True,
            "digest": True,
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
        """搜索记忆"""
        
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

---

## 6. 高级功能

### 6.1 多数据集管理

```python
# 为不同用户创建不同的数据集
datasets = ["user_001", "user_002", "user_003"]

for user_id in datasets:
    payload = {
        "dataset": "personamem_v2",
        "task": user_id,
        "messages": [...]
    }
    requests.post(url, json=payload)
```

### 6.2 批量写入

```python
# 批量处理多个会话
def batch_ingest(client, sessions):
    results = []
    for session in sessions:
        result = client.store_and_digest(
            dataset="personamem_v2",
            task="user_001",
            messages=session
        )
        results.append(result)
    return results
```

### 6.3 自定义搜索策略

```python
# 使用不同的搜索模式

# 快速搜索（低延迟）
result = client.search(
    dataset="personamem_v2",
    task="user_001",
    question="What did I do?",
    mode="direct",      # 跳过 LLM 扩展
    strategy="rrf"     # 快速融合
)

# 高精度搜索
result = client.search(
    dataset="personamem_v2",
    task="user_001",
    question="What did I do?",
    mode="reflect",     # 使用反思模式
    strategy="rerank"  # 使用重排序
)
```

### 6.4 异步处理

```python
import asyncio

async def async_ingest():
    """异步写入"""
    
    # 写入请求会立即返回，消化在后台进行
    response = await client.post_async("/memory", json=payload)
    
    # 可以在后台任务中处理
    background_task = asyncio.create_task(process_digest())
    
    return response.json()


async def async_search():
    """异步搜索"""
    
    response = await client.post_async("/memory/search", json=payload)
    return response.json()
```

---

## 7. 配置参数

### 7.1 搜索参数

```python
# 搜索相关配置
QA_BM25_FACT_TOP_N = 20          # BM25 路径返回数量
QA_EMBED_FACT_TOP_N = 20         # 向量路径返回数量
QA_ENTITY_TOP_N = 5              # 实体树实体数量
QA_TREE_BEAM_WIDTH = 3           # 实体树光束宽度
QA_RERANK_TOP_K = 12             # 融合后保留数量
```

### 7.2 Token 预算

```python
_FACT_BUDGET_TOKENS = 4500       # 事实上下文预算
_SESSION_BUDGET_TOKENS = 1500    # 会话上下文预算
```

### 7.3 实体解析参数

```python
RESOLVER_JACCARD_THRESHOLD = 0.9
RESOLVER_ENTROPY_THRESHOLD = 1.5
RESOLVER_MINHASH_PERMUTATIONS = 32
RESOLVER_LLM_ENABLED = True
```

---

## 8. 错误处理

### 8.1 常见错误

```python
# 错误 1: 数据集不存在
{
    "detail": "Task 'user_999' not found in dataset 'personamem_v2'"
}

# 错误 2: 消息格式错误
{
    "detail": "messages required when store=True"
}

# 错误 3: 参数错误
{
    "detail": "at least one of store or digest must be True"
}
```

### 8.2 错误处理示例

```python
import requests
from requests.exceptions import HTTPError

def safe_search(client, dataset, task, question):
    try:
        result = client.search(dataset, task, question)
        return result
    except HTTPError as e:
        if e.response.status_code == 404:
            print(f"数据集或任务不存在: {dataset}/{task}")
        else:
            print(f"HTTP 错误: {e}")
        return None
    except Exception as e:
        print(f"错误: {e}")
        return None
```

---

## 9. 总结

MemBrain 使用流程总结：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MemBrain 使用流程                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 启动服务器                                                             │
│     python -m uvicorn membrain.api.server:app --port 9574                  │
│                                                                             │
│  2. 写入数据                                                               │
│     POST /api/memory                                                        │
│     - store=True, digest=True                                               │
│     - 消息会自动提取为实体和事实                                            │
│                                                                             │
│  3. 搜索数据                                                               │
│     POST /api/memory/search                                                │
│     - 返回 packed_context 可直接用于 LLM                                    │
│                                                                             │
│  4. 生成答案                                                               │
│     - 使用 packed_context 调用 LLM                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键要点**:

1. **存储**: 使用 `/api/memory` 存储对话
2. **消化**: `digest=True` 会自动提取记忆（实体、事实）
3. **搜索**: 使用 `/api/memory/search` 检索记忆
4. **回答**: 使用返回的 `packed_context` 调用 LLM