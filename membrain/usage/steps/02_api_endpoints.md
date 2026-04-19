# Step 2: API 接口与服务器启动

## 概述

本步骤介绍 MemBrain 提供的 API 接口，包括端点说明、请求/响应格式，以及如何启动服务器。

## 2.1 API 端点列表

MemBrain 提供以下主要 API 端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/memory` | POST | 存储和/或消化对话 |
| `/api/memory/search` | POST | 搜索记忆 |

## 2.2 服务器启动

### 启动命令

```bash
# 进入项目目录
cd /home/project/MemBrain

# 启动 MemBrain 服务器
python -m uvicorn membrain.api.server:app --host 0.0.0.0 --port 9574
```

### 启动参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 监听地址 | `0.0.0.0` (所有地址) |
| `--port` | 监听端口 | `9574` |
| `--reload` | 热重载 (开发用) | 关闭 |

### 启动示例

```bash
# 开发模式 (带热重载)
python -m uvicorn membrain.api.server:app --reload

# 生产模式
python -m uvicorn membrain.api.server:app --host 0.0.0.0 --port 9574 --workers 4
```

## 2.3 API 端点详解

### 端点 1: POST /api/memory

**功能**: 存储对话消息并/或提取记忆

**请求体格式**:

```python
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

**响应格式**:

```python
class MemoryResponse(BaseModel):
    dataset_id: int
    task_pk: int
    session_id: int | None
    session_number: int | None
    digested_sessions: int
    status: str  # "stored" | "stored_and_digest_queued" | "digest_queued"
```

### 端点 2: POST /api/memory/search

**功能**: 搜索记忆，返回用于 LLM 回答的上下文

**请求体格式**:

```python
class MemorySearchRequest(BaseModel):
    dataset: str                          # 数据集名称
    task: str                             # 任务标识
    question: str                         # 搜索问题
    top_k: Optional[int] = None          # 返回结果数量
    strategy: Optional[str] = None       # 融合策略: "rrf" | "rerank"
    mode: Optional[str] = None           # 搜索模式: "direct" | "expand" | "reflect"
```

**响应格式**:

```python
class MemorySearchResponse(BaseModel):
    packed_context: str           # 格式化上下文
    packed_token_count: int      # Token 数量
    fact_ids: list[int]         # 事实 ID 列表
    facts: list[RetrievedFactOut]  # 事实详情
    sessions: list[RetrievedSessionOut]  # 会话列表
    raw_messages: list          # 预留：原始消息
```

## 2.4 搜索模式详解

| 模式 | 说明 | 检索路径数 | 延迟 | 适用场景 |
|------|------|------------|------|----------|
| `direct` | 直接模式，跳过 LLM 扩展 | 3 (A+B+C) | 低 | 低延迟要求 |
| `expand` | 扩展模式，使用 LLM 扩展 | 6 (默认) | 中 | 大多数场景 |
| `reflect` | 反思模式，使用 LLM 反思增强 | 6 + R2 | 高 | 高召回要求 |

**模式选择建议**:

```python
# 低延迟场景
mode = "direct"  # 跳过 LLM 扩展，3 条检索路径

# 大多数场景 (默认)
mode = "expand"  # 使用 LLM 扩展，6 条检索路径

# 高召回场景
mode = "reflect"  # 使用反思增强，可能触发二次检索
```

## 2.5 融合策略详解

| 策略 | 方法 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| `rrf` | 互惠排名融合 | 快 | 中 | 大多数场景 (默认) |
| `rerank` | 交叉编码器重排 | 慢 | 高 | 高精度要求 |

**策略选择建议**:

```python
# 大多数场景 (默认)
strategy = "rrf"  # 快速融合

# 高精度要求
strategy = "rerank"  # 使用交叉编码器重排
```

## 2.6 使用示例

### Python requests 示例

```python
import requests

# MemBrain 服务器地址
BASE_URL = "http://localhost:9574/api"

# 写入数据
response = requests.post(
    f"{BASE_URL}/memory",
    json={
        "dataset": "personamem_v2",
        "task": "user_001",
        "messages": [
            {"speaker": "user", "content": "Hello"}
        ]
    }
)
print(response.json())

# 搜索数据
response = requests.post(
    f"{BASE_URL}/memory/search",
    json={
        "dataset": "personamem_v2",
        "task": "user_001",
        "question": "What did I do?"
    }
)
print(response.json())
```

### curl 示例

```bash
# 写入数据
curl -X POST "http://localhost:9574/api/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "messages": [
      {"speaker": "user", "content": "Hello"}
    ]
  }'

# 搜索数据
curl -X POST "http://localhost:9574/api/memory/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "What did I do?"
  }'
```

## 2.7 记忆使用场景示例

### 场景 1: 基本对话存储

```python
# 存储用户与 AI 的对话
url = "http://localhost:9574/api/memory"

payload = {
    "dataset": "my_app",
    "task": "user_123",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "What's the weather today?"},
        {"speaker": "assistant", "content": "The weather is sunny, 25 degrees."},
        {"speaker": "user", "content": "Great! I'm planning a picnic."},
        {"speaker": "assistant", "content": "Sounds fun! Where are you planning to go?"}
    ]
}

response = requests.post(url, json=payload)
# 响应: {"status": "stored_and_digest_queued", ...}
```

### 场景 2: 带时间的对话

```python
# 存储带有时间戳的对话
payload = {
    "dataset": "my_app",
    "task": "user_123",
    "session_time": "2024-01-15T14:30:00",  # 会话时间
    "store": True,
    "digest": True,
    "messages": [
        {
            "speaker": "user", 
            "content": "I went to the gym this morning.",
            "message_time": "2024-01-15T08:00:00"  # 消息时间
        },
        {
            "speaker": "assistant", 
            "content": "Good job! How long did you workout?",
            "message_time": "2024-01-15T08:01:00"
        }
    ]
}

response = requests.post(url, json=payload)
```

### 场景 3: 不同搜索模式

```python
# 快速搜索 (低延迟)
payload = {
    "dataset": "my_app",
    "task": "user_123",
    "question": "What did I do today?",
    "mode": "direct",      # 跳过 LLM 扩展
    "strategy": "rrf"      # 快速融合
}

# 高精度搜索
payload = {
    "dataset": "my_app",
    "task": "user_123",
    "question": "What did I do today?",
    "mode": "reflect",     # 使用反思模式
    "strategy": "rerank"  # 使用重排序
}

# 调整返回数量
payload = {
    "dataset": "my_app",
    "task": "user_123",
    "question": "What did I do?",
    "top_k": 20  # 返回更多结果
}
```

## 2.8 错误处理

### 常见错误

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| 400 | 参数错误 | 检查必填参数 |
| 404 | 数据集/任务不存在 | 确认 dataset 和 task |
| 500 | 服务器错误 | 检查服务器日志 |

### 错误处理示例

```python
import requests
from requests.exceptions import HTTPError

def safe_request(url, payload):
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except HTTPError as e:
        print(f"HTTP 错误: {e}")
        print(f"响应: {e.response.json()}")
    except Exception as e:
        print(f"错误: {e}")
    return None

# 使用
result = safe_request("http://localhost:9574/api/memory", payload)
```

## 总结

本步骤介绍了：
- 2 个主要 API 端点及其功能
- 服务器启动命令和参数
- 请求/响应格式
- 搜索模式和融合策略的选择
- 3 个具体的使用场景示例
- 错误处理方法

理解 API 接口后，可以开始实际的开发和使用。