# Step 3: 写入数据 (Ingest)

## 概述

本步骤详细介绍如何通过 API 将用户对话写入 MemBrain 系统，包括请求格式、写入模式、参数说明，以及具体的记忆使用场景示例。

## 3.1 API 请求格式

### 端点

```
POST /api/memory
```

### 请求体

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

## 3.2 消息格式

### 消息结构

```python
message = {
    "speaker": "user" | "assistant",       # 发言者类型
    "content": "消息内容",                 # 消息文本
    "message_time": "2024-01-15T10:30:00"  # 可选：消息时间
}
```

### speaker 选项

| 值 | 说明 |
|----|------|
| `user` | 用户消息 |
| `assistant` | AI 助手消息 |
| `system` | 系统消息（如果有） |

### 示例

```python
messages = [
    {"speaker": "user", "content": "I had lunch with John yesterday."},
    {"speaker": "assistant", "content": "That sounds nice!"},
    {"speaker": "user", "content": "We went to Luigi's Pizza. It's his favorite restaurant."}
]
```

## 3.3 写入模式

MemBrain 支持三种写入模式：

### 模式 1: 存储并消化 (最常用)

```python
payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": True,     # 存储消息
    "digest": True,    # 提取记忆
    "messages": [...]
}
```

**适用场景**:
- 新的用户对话
- 完整的会话记录
- 需要立即提取记忆

### 模式 2: 仅存储

```python
payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": True,     # 存储消息
    "digest": False,   # 不提取记忆
    "messages": [...]
}
```

**适用场景**:
- 临时保存对话
- 稍后批量消化
- 调试模式

### 模式 3: 仅消化

```python
payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": False,    # 不存储新消息
    "digest": True,    # 消化已有消息
    "messages": []
}
```

**适用场景**:
- 批量处理历史会话
- 重新消化失败的会话

## 3.4 参数详解

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `dataset` | string | 是 | 数据集名称，如 "personamem_v2" |
| `task` | string | 是 | 任务标识，如 "user_001" |
| `messages` | array | 是 | 消息列表 |
| `store` | boolean | 否 | 是否存储原始消息，默认 True |
| `digest` | boolean | 否 | 是否提取记忆，默认 True |
| `session_time` | string | 否 | 会话时间 (ISO 格式) |
| `agent_profile` | string | 否 | Agent 配置档案 |

## 3.5 请求示例

### 基本请求

```python
import requests

url = "http://localhost:9574/api/memory"

payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": True,
    "digest": True,
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

response = requests.post(url, json=payload)
result = response.json()

print(json.dumps(result, indent=2))
```

### 响应格式

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

### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `dataset_id` | int | 数据集 ID |
| `task_pk` | int | 任务主键 |
| `session_id` | int | 会话 ID |
| `session_number` | int | 会话编号 |
| `digested_sessions` | int | 已消化的会话数 |
| `status` | string | 状态: "stored" / "stored_and_digest_queued" / "digest_queued" |

## 3.6 记忆使用场景示例

### 场景 1: 社交活动记忆

```python
# 输入: 用户分享社交活动
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "session_time": "2024-01-15T14:30:00",
    "messages": [
        {"speaker": "user", "content": "I had lunch with my friend John at Luigi's Pizza yesterday. It was really good!"},
        {"speaker": "assistant", "content": "That sounds nice! What did you have?"},
        {"speaker": "user", "content": "I had the carbonara and John ordered a pizza. He said it's his favorite restaurant. We've been going there for years."},
        {"speaker": "assistant", "content": "It must be a special place for you both!"},
        {"speaker": "user", "content": "Yes! We actually met there for the first time three years ago. Great memories."}
    ]
}

response = requests.post(url, json=payload)
```

**提取的记忆**:

```
Entities:
  - John (朋友)
  - Luigi's Pizza (餐厅)
  - carbonara (食物)

Facts:
  - [User] had lunch with [John] at [Luigi's Pizza] [yesterday]
  - [John] ordered [pizza] at [Luigi's Pizza]
  - [Luigi's Pizza] is [John]'s favorite restaurant
  - [User] and [John] have been going to [Luigi's Pizza] for [years]
  - [User] and [John] met at [Luigi's Pizza] for the first time [three years ago]

EntityTree:
  - Social: John, Luigi's Pizza
  - Relationships: John (friend)
  - History: first meeting at Luigi's Pizza
```

### 场景 2: 工作相关记忆

```python
# 输入: 用户谈论工作
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "I started working at Google last month."},
        {"speaker": "assistant", "content": "That's great! What role did you get?"},
        {"speaker": "user", "content": "I'm a software engineer on the Search team. My manager is Sarah."},
        {"speaker": "assistant", "content": "Sounds like an exciting opportunity!"},
        {"speaker": "user", "content": "Yes! We work on the core search ranking algorithm. It's challenging but rewarding."}
    ]
}

response = requests.post(url, json=payload)
```

**提取的记忆**:

```
Entities:
  - Google (公司)
  - software engineer (职位)
  - Search team (团队)
  - Sarah (经理)

Facts:
  - [User] started working at [Google] [last month]
  - [User] is a [software engineer] at [Google]
  - [User] works on [Search team] at [Google]
  - [Sarah] is [User]'s manager at [Google]
  - [Search team] works on [core search ranking algorithm]

EntityTree:
  - Career: Google, software engineer, Search team
  - Team: Sarah (manager)
  - Work: core search ranking algorithm
```

### 场景 3: 兴趣偏好记忆

```python
# 输入: 用户谈论兴趣爱好
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "I've been learning piano for 3 years."},
        {"speaker": "assistant", "content": "That's impressive! Do you practice daily?"},
        {"speaker": "user", "content": "Yes, I practice every evening. My favorite piece is Moonlight Sonata."},
        {"speaker": "assistant", "content": "Beethoven's Moonlight Sonata is a beautiful piece!"},
        {"speaker": "user", "content": "I also enjoy playing jazz. My favorite jazz artist is Bill Evans."}
    ]
}

response = requests.post(url, json=payload)
```

**提取的记忆**:

```
Entities:
  - piano (乐器)
  - Moonlight Sonata (曲目)
  - jazz (音乐风格)
  - Bill Evans (音乐人)

Facts:
  - [User] has been learning [piano] for [3 years]
  - [User] practices [piano] every evening
  - [User]'s favorite piece is [Moonlight Sonata]
  - [User] enjoys playing [jazz]
  - [User]'s favorite jazz artist is [Bill Evans]

EntityTree:
  - Interests: piano, Moonlight Sonata, jazz, Bill Evans
  - Habits: practicing every evening
  - Duration: 3 years
```

### 场景 4: 家庭关系记忆

```python
# 输入: 用户谈论家庭
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "My sister Lisa is visiting next week."},
        {"speaker": "assistant", "content": "That's nice! How long will she stay?"},
        {"speaker": "user", "content": "She's staying for a week. We're planning to go hiking together."},
        {"speaker": "assistant", "content": "Great! Any specific trails?"},
        {"speaker": "user", "content": "We're thinking about the Mountain Trail. It's a bit challenging but the view is amazing."}
    ]
}

response = requests.post(url, json=payload)
```

**提取的记忆**:

```
Entities:
  - Lisa (妹妹)
  - Mountain Trail (徒步路线)

Facts:
  - [Lisa] is [User]'s sister
  - [Lisa] is visiting [next week]
  - [Lisa] is staying for [a week]
  - [User] and [Lisa] are planning to go [hiking] together
  - [Mountain Trail] is a challenging hike with amazing views

EntityTree:
  - Family: Lisa (sister)
  - Plans: hiking with Lisa
  - Location: Mountain Trail
```

### 场景 5: 健康生活记忆

```python
# 输入: 用户谈论健康习惯
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "I've started a new diet this week."},
        {"speaker": "assistant", "content": "What kind of diet?"},
        {"speaker": "user", "content": "I'm doing keto. I'm trying to reduce my carb intake to under 50 grams per day."},
        {"speaker": "assistant", "content": "That can be effective! How are you feeling?"},
        {"speaker": "user", "content": "Actually, I feel great! Much more energy than before."}
    ]
}

response = requests.post(url, json=payload)
```

**提取的记忆**:

```
Entities:
  - keto (饮食方式)
  - carbs (营养素)

Facts:
  - [User] started [keto] diet [this week]
  - [User] is trying to reduce [carbs] to under [50 grams] per day
  - [User] feels more energetic on keto

EntityTree:
  - Health: keto diet
  - Goals: reduce carbs to 50g/day
  - Effects: more energy
```

## 3.7 curl 命令示例

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

# 仅存储
curl -X POST "http://localhost:9574/api/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": true,
    "digest": false,
    "messages": [
      {"speaker": "user", "content": "Test message"}
    ]
  }'

# 仅消化
curl -X POST "http://localhost:9574/api/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": false,
    "digest": true
  }'
```

## 3.8 注意事项

1. **异步处理**: `digest=True` 时，消化是异步进行的，返回状态为 `stored_and_digest_queued`
2. **等待时间**: 建议在写入后等待几秒再搜索，以确保消化完成
3. **消息顺序**: 消息按 `position` 顺序处理，确保对话顺序正确
4. **时间戳**: 建议提供 `session_time` 和 `message_time`，有助于时序记忆

## 3.9 错误处理

```python
import requests
from requests.exceptions import HTTPError

def store_message(url, payload):
    try:
        response = requests.post(url, json=payload)
        
        # 检查状态码
        if response.status_code == 400:
            error = response.json()
            print(f"参数错误: {error.get('detail')}")
            return None
            
        response.raise_for_status()
        return response.json()
        
    except HTTPError as e:
        print(f"HTTP 错误: {e}")
    except Exception as e:
        print(f"错误: {e}")
    
    return None

# 使用
result = store_message("http://localhost:9574/api/memory", payload)
if result:
    print(f"成功! Session ID: {result['session_id']}")
```

## 总结

本步骤介绍了：
- API 请求格式和参数说明
- 消息格式和 speaker 选项
- 三种写入模式及其适用场景
- 5 个具体的记忆使用场景示例
- curl 命令示例
- 错误处理方法

写入数据后，可以进行搜索来检索记忆。