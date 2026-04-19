# Step 4: 搜索数据 (Search)

## 概述

本步骤详细介绍如何通过 API 搜索 MemBrain 中存储的记忆，包括请求格式、搜索模式、融合策略，以及具体的记忆使用场景示例。

## 4.1 API 请求格式

### 端点

```
POST /api/memory/search
```

### 请求体

```python
class MemorySearchRequest(BaseModel):
    dataset: str                          # 数据集名称
    task: str                             # 任务标识
    question: str                         # 搜索问题
    top_k: Optional[int] = None          # 返回结果数量
    strategy: Optional[str] = None       # 融合策略: "rrf" | "rerank"
    mode: Optional[str] = None           # 搜索模式: "direct" | "expand" | "reflect"
```

## 4.2 参数详解

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `dataset` | string | 是 | 数据集名称 |
| `task` | string | 是 | 任务标识 |
| `question` | string | 是 | 搜索问题 |
| `top_k` | int | 否 | 返回结果数量，默认系统配置 |
| `strategy` | string | 否 | 融合策略: "rrf" / "rerank" |
| `mode` | string | 否 | 搜索模式: "direct" / "expand" / "reflect" |

## 4.3 搜索模式

| 模式 | 说明 | 检索路径数 | 延迟 | 适用场景 |
|------|------|------------|------|----------|
| `direct` | 直接模式，跳过 LLM 扩展 | 3 | 低 | 低延迟要求 |
| `expand` | 扩展模式（默认），使用 LLM 扩展 | 6 | 中 | 大多数场景 |
| `reflect` | 反思模式，使用 LLM 反思增强 | 6 + R2 | 高 | 高召回要求 |

### 模式选择建议

```python
# 低延迟场景
mode = "direct"

# 大多数场景 (默认)
mode = "expand"

# 高召回场景
mode = "reflect"
```

## 4.4 融合策略

| 策略 | 方法 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| `rrf` | 互惠排名融合 (默认) | 快 | 中 | 大多数场景 |
| `rerank` | 交叉编码器重排 | 慢 | 高 | 高精度要求 |

### 策略选择建议

```python
# 大多数场景 (默认)
strategy = "rrf"

# 高精度要求
strategy = "rerank"
```

## 4.5 响应格式

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

### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `packed_context` | string | 格式化上下文，可直接用于 LLM |
| `packed_token_count` | int | 估算的 token 数量 |
| `fact_ids` | array | 包含在 packed_context 中的事实 ID 列表 |
| `facts` | array | 事实详情列表 |
| `sessions` | array | 相关的会话摘要列表 |
| `raw_messages` | array | 预留：原始消息列表 |

## 4.6 请求示例

### 基本请求

```python
import requests
import json

url = "http://localhost:9574/api/memory/search"

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

print(json.dumps(result, indent=2))
```

### 使用 LLM 生成答案

```python
# 步骤 1: 搜索记忆
search_result = search_memory("What did I do yesterday?")

# 步骤 2: 提取上下文
context = search_result["packed_context"]

# 步骤 3: 调用 LLM 生成答案
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: What did I do yesterday?"}
    ]
)

answer = response.choices[0].message.content
print(answer)
```

## 4.7 记忆使用场景示例

### 场景 1: 查询过去的社交活动

**场景**: 用户想知道之前和 John 一起做了什么

```python
# 搜索问题
question = "What did I do with John?"

payload = {
    "dataset": "my_app",
    "task": "user_001",
    "question": question,
    "top_k": 10,
    "mode": "expand",
    "strategy": "rrf"
}

response = requests.post(url, json=payload)
result = response.json()

# 输出结果
print("=== 搜索结果 ===")
print(f"Token Count: {result['packed_token_count']}")
print(f"Fact Count: {len(result['facts'])}")

print("\n--- Facts ---")
for fact in result["facts"]:
    print(f"  - {fact['text']}")

print("\n--- Packed Context ---")
print(result["packed_context"])
```

**预期输出**:

```json
{
    "packed_context": "## Relevant Episodes\n\n**Subject**: Lunch with John\n\nUser had lunch with John at Luigi's Pizza yesterday. They have been going there for years. They actually met there for the first time three years ago.\n\n## Additional Facts\n\n- [User] had lunch with [John] at [Luigi's Pizza] [yesterday]\n- [John] ordered [pizza] at [Luigi's Pizza]\n- [Luigi's Pizza] is [John]'s favorite restaurant\n- [User] and [John] have been going to [Luigi's Pizza] for [years]\n- [User] and [John] met at [Luigi's Pizza] for the first time [three years ago]\n",
    "facts": [
        {
            "fact_id": 1,
            "text": "[User] had lunch with [John] at [Luigi's Pizza] [yesterday]",
            "source": "bm25",
            "rerank_score": 0.95,
            "time_info": "yesterday"
        },
        ...
    ]
}
```

### 场景 2: 查询工作信息

**场景**: 用户想知道当前的工作情况

```python
# 搜索问题
question = "What is my current job?"

payload = {
    "dataset": "my_app",
    "task": "user_001",
    "question": question,
    "top_k": 10,
    "mode": "expand",
    "strategy": "rerank"  # 使用高精度模式
}

response = requests.post(url, json=payload)
result = response.json()

print("=== 搜索结果 ===")
print(f"Token Count: {result['packed_token_count']}")

print("\n--- Facts ---")
for fact in result["facts"]:
    print(f"  - {fact['text']}")

print("\n--- Packed Context ---")
print(result["packed_context"])
```

**预期输出**:

```json
{
    "packed_context": "## Relevant Episodes\n\n**Subject**: Work at Google\n\nUser started working at Google last month as a software engineer on the Search team.\n\n## Additional Facts\n\n- [User] started working at [Google] [last month]\n- [User] is a [software engineer] at [Google]\n- [User] works on [Search team] at [Google]\n- [Sarah] is [User]'s manager at [Google]\n- [Search team] works on [core search ranking algorithm]\n",
    "facts": [
        {
            "fact_id": 10,
            "text": "[User] started working at [Google] [last month]",
            "source": "semantic",
            "rerank_score": 0.98
        },
        ...
    ]
}
```

### 场景 3: 查询兴趣爱好

**场景**: 用户想知道自己的兴趣爱好

```python
# 搜索问题
question = "What are my hobbies and interests?"

payload = {
    "dataset": "my_app",
    "task": "user_001",
    "question": question,
    "top_k": 10,
    "mode": "expand",
    "strategy": "rrf"
}

response = requests.post(url, json=payload)
result = response.json()

print("=== 搜索结果 ===")
print(result["packed_context"])
```

**预期输出**:

```json
{
    "packed_context": "## Relevant Episodes\n\n**Subject**: Hobbies and Interests\n\nUser has several hobbies including playing piano and jazz.\n\n## Additional Facts\n\n- [User] has been learning [piano] for [3 years]\n- [User] practices [piano] every evening\n- [User]'s favorite piece is [Moonlight Sonata]\n- [User] enjoys playing [jazz]\n- [User]'s favorite jazz artist is [Bill Evans]\n",
    "facts": [
        {"fact_id": 20, "text": "[User] has been learning [piano] for [3 years]", ...},
        {"fact_id": 21, "text": "[User] practices [piano] every evening", ...},
        ...
    ]
}
```

### 场景 4: 查询家庭关系

**场景**: 用户想知道家庭成员的信息

```python
# 搜索问题
question = "Tell me about my family"

payload = {
    "dataset": "my_app",
    "task": "user_001",
    "question": question,
    "top_k": 10,
    "mode": "expand",
    "strategy": "rrf"
}

response = requests.post(url, json=payload)
result = response.json()

print(result["packed_context"])
```

**预期输出**:

```json
{
    "packed_context": "## Relevant Episodes\n\n**Subject**: Family\n\nUser's sister Lisa is visiting next week for a week. They plan to go hiking together at Mountain Trail.\n\n## Additional Facts\n\n- [Lisa] is [User]'s sister\n- [Lisa] is visiting [next week]\n- [Lisa] is staying for [a week]\n- [User] and [Lisa] are planning to go [hiking] together\n- [Mountain Trail] is a challenging hike with amazing views\n",
    "facts": [
        {"fact_id": 30, "text": "[Lisa] is [User]'s sister", ...},
        ...
    ]
}
```

### 场景 5: 查询健康习惯

**场景**: 用户想知道自己的健康习惯

```python
# 搜索问题
question = "What is my current diet?"

payload = {
    "dataset": "my_app",
    "task": "user_001",
    "question": question,
    "top_k": 10,
    "mode": "expand",
    "strategy": "rrf"
}

response = requests.post(url, json=payload)
result = response.json()

print(result["packed_context"])
```

**预期输出**:

```json
{
    "packed_context": "## Relevant Episodes\n\n**Subject**: Diet\n\nUser started a keto diet this week. They're trying to reduce carbs to under 50 grams per day and feel more energetic.\n\n## Additional Facts\n\n- [User] started [keto] diet [this week]\n- [User] is trying to reduce [carbs] to under [50 grams] per day\n- [User] feels more energetic on keto\n",
    "facts": [
        {"fact_id": 40, "text": "[User] started [keto] diet [this week]", ...},
        ...
    ]
}
```

## 4.8 curl 命令示例

```bash
# 基本搜索
curl -X POST "http://localhost:9574/api/memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What did I do yesterday?"
  }'

# 指定搜索模式和策略
curl -X POST "http://localhost:9574/api/memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What did I do yesterday?",
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

## 4.9 注意事项

1. **等待消化完成**: 写入数据后，建议等待几秒再搜索，确保消化完成
2. **搜索模式选择**:
   - 需要快速响应: 使用 `direct` 模式
   - 需要完整结果: 使用 `expand` 模式 (默认)
   - 需要高召回: 使用 `reflect` 模式
3. **融合策略选择**:
   - 大多数场景: 使用 `rrf` (默认)
   - 高精度要求: 使用 `rerank`
4. **top_k 调整**: 根据需要调整返回结果数量

## 4.10 完整使用流程

```python
import requests
import time
from openai import OpenAI

BASE_URL = "http://localhost:9574/api"

def store_and_digest():
    """步骤 1: 存储并消化对话"""
    url = f"{BASE_URL}/memory"
    
    payload = {
        "dataset": "my_app",
        "task": "user_001",
        "store": True,
        "digest": True,
        "messages": [
            {"speaker": "user", "content": "I had lunch with John yesterday."},
            {"speaker": "assistant", "content": "That sounds nice!"},
            {"speaker": "user", "content": "We went to Luigi's Pizza. It's his favorite."}
        ]
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    print(f"Status: {result['status']}")
    return result


def search_memory(question):
    """步骤 2: 搜索记忆"""
    url = f"{BASE_URL}/memory/search"
    
    payload = {
        "dataset": "my_app",
        "task": "user_001",
        "question": question,
        "top_k": 10,
        "mode": "expand",
        "strategy": "rrf"
    }
    
    response = requests.post(url, json=payload)
    return response.json()


def generate_answer(question, search_result):
    """步骤 3: 使用 LLM 生成答案"""
    client = OpenAI(api_key="your-api-key")
    
    context = search_result["packed_context"]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    
    return response.choices[0].message.content


# 执行完整流程
if __name__ == "__main__":
    # 1. 存储并消化对话
    store_result = store_and_digest()
    
    # 等待消化完成
    print("等待消化完成...")
    time.sleep(5)
    
    # 2. 搜索记忆
    question = "What did I do with John?"
    search_result = search_memory(question)
    print(f"找到 {len(search_result['facts'])} 个事实")
    
    # 3. 生成答案
    answer = generate_answer(question, search_result)
    print(f"\n答案: {answer}")
```

## 总结

本步骤介绍了：
- API 请求格式和参数说明
- 搜索模式的选择 (direct / expand / reflect)
- 融合策略的选择 (rrf / rerank)
- 响应格式和字段说明
- 5 个具体的记忆使用场景示例
- curl 命令示例
- 完整使用流程

通过搜索 API，可以检索存储的记忆，并用于 LLM 生成答案。