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