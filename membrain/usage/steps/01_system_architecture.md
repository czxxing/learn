# Step 1: 系统架构与数据模型

## 概述

本步骤介绍 MemBrain 的整体系统架构和核心数据模型，帮助理解数据在系统中的流转方式。

## 1.1 核心组件

MemBrain 是一个基于 FastAPI 的记忆系统，主要由以下组件构成：

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

### 组件说明

| 组件 | 说明 |
|------|------|
| 客户端 (Client) | 发起 API 请求的应用，如 Web App、移动端 |
| FastAPI Server | 处理 API 请求，协调各个模块 |
| 数据库 (PostgreSQL) | 存储实体、事实、会话等数据 |
| LLM | 生成实体、事实，处理自然语言 |

## 1.2 数据模型关系

MemBrain 使用以下核心数据模型：

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

## 1.3 数据模型详解

### Dataset (数据集)

```python
class DatasetModel(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**示例**:
```json
{
    "id": 1,
    "name": "personamem_v2",
    "created_at": "2024-01-01T00:00:00"
}
```

### Task (任务)

```python
class TaskModel(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    name = Column(String, nullable=False)
    agent_profile = Column(String)
```

**示例**:
```json
{
    "id": 1,
    "dataset_id": 1,
    "name": "user_001",
    "agent_profile": "personamemv2"
}
```

### Session (会话)

```python
class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    session_number = Column(Integer)
    session_time = Column(DateTime)
    digested_at = Column(DateTime)
```

**示例**:
```json
{
    "id": 1,
    "task_id": 1,
    "session_number": 1,
    "session_time": "2024-01-15T14:30:00",
    "digested_at": "2024-01-15T14:30:05"
}
```

### Message (消息)

```python
class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    speaker = Column(String)  # "user" | "assistant"
    content = Column(Text)
    position = Column(Integer)
```

**示例**:
```json
{
    "id": 1,
    "session_id": 1,
    "speaker": "user",
    "content": "I had lunch with John yesterday.",
    "position": 0
}
```

### Entity (实体)

```python
class EntityModel(Base):
    __tablename__ = "entities"
    
    entity_id = Column(String, primary_key=True)
    task_id = Column(Integer)
    canonical_ref = Column(String)  # 规范名称
    desc = Column(Text)            # 描述
    desc_embedding = Column(Vector(768))  # 向量
```

**示例**:
```json
{
    "entity_id": "550e8400-e29b-41d4-a716-446655440000",
    "task_id": 1,
    "canonical_ref": "John",
    "desc": "User's friend, colleague at work",
    "desc_embedding": [0.123, -0.456, ...]
}
```

### Fact (事实)

```python
class FactModel(Base):
    __tablename__ = "facts"
    
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer)
    batch_id = Column(String)
    text = Column(Text)  # 事实文本
    text_embedding = Column(Vector(768))
    status = Column(String)  # "active" | "archived"
```

**示例**:
```json
{
    "id": 1,
    "task_id": 1,
    "batch_id": "batch-001",
    "text": "[User] had lunch with [John] at [Luigi's Pizza] [yesterday]",
    "text_embedding": [0.789, -0.123, ...],
    "status": "active"
}
```

### EntityTree (实体树)

```python
class EntityTreeModel(Base):
    __tablename__ = "entity_trees"
    
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer)
    entity_id = Column(String)
    fact_ids = Column(JSON)  # 事实 ID 列表
    structure_json = Column(JSON)  # 树结构
```

**示例**:
```json
{
    "id": 1,
    "task_id": 1,
    "entity_id": "user-entity-id",
    "fact_ids": [1, 2, 3, 4, 5],
    "structure_json": {
        "root": {
            "children": [
                {"label": "Social", "fact_ids": [1, 2]},
                {"label": "Career", "fact_ids": [3, 4, 5]}
            ]
        }
    }
}
```

## 1.4 数据流转示例

以一个具体的对话为例，展示数据流转：

### 用户对话

```
用户: I had lunch with my friend John at Luigi's Pizza yesterday.
助手: That sounds nice! What did you have?
用户: I had the carbonara and John ordered a pizza. He said it's his favorite restaurant.
```

### 数据流转过程

```
Step 1: 创建数据集/任务
  Dataset: personamem_v2
  Task: user_001

Step 2: 创建会话
  Session: session_number=1, session_time=2024-01-15

Step 3: 存储消息
  Message 0: [user] I had lunch with my friend John...
  Message 1: [assistant] That sounds nice!...
  Message 2: [user] I had the carbonara...

Step 4: 提取实体
  Entity: John
  Entity: Luigi's Pizza
  Entity: carbonara

Step 5: 生成事实
  Fact: [User] had lunch with [John] at [Luigi's Pizza] [yesterday]
  Fact: [John] likes [Luigi's Pizza]'s carbonara

Step 6: 建立关联
  FactRef: (fact_id=1, entity_id=John, alias_text="John")
  FactRef: (fact_id=1, entity_id=Luigi's Pizza, alias_text="Luigi's Pizza")

Step 7: 构建实体树
  EntityTree(User):
    - Social: fact_ids=[1, 2]
    - Interests: fact_ids=[3]
```

## 1.5 记忆使用场景示例

### 场景 1: 个人生活记忆

```python
# 用户输入对话
messages = [
    {"speaker": "user", "content": "I met my sister Lisa at the coffee shop yesterday."},
    {"speaker": "assistant", "content": "Which coffee shop did you go to?"},
    {"speaker": "user", "content": "It was the one on Main Street. We talked about our summer plans."}
]

# 提取的记忆
# Entities: Lisa, coffee shop, Main Street, summer plans
# Facts:
#   - [User] met [Lisa] at [coffee shop] [yesterday]
#   - [coffee shop] is on [Main Street]
#   - [User] and [Lisa] talked about [summer plans]
# EntityTree:
#   - Social: Lisa, coffee shop
#   - Plans: summer plans
```

### 场景 2: 工作经历记忆

```python
# 用户输入对话
messages = [
    {"speaker": "user", "content": "I started working at Google last month."},
    {"speaker": "assistant", "content": "That's great! What role did you get?"},
    {"speaker": "user", "content": "I'm a software engineer on the Search team. My manager is Sarah."}
]

# 提取的记忆
# Entities: Google, software engineer, Search team, Sarah
# Facts:
#   - [User] started working at [Google] [last month]
#   - [User] is a [software engineer] at [Google]
#   - [User] works on [Search team] at [Google]
#   - [Sarah] is [User]'s manager at [Google]
# EntityTree:
#   - Career: Google, software engineer, Search team
#   - Relationships: Sarah (manager)
```

### 场景 3: 兴趣爱好记忆

```python
# 用户输入对话
messages = [
    {"speaker": "user", "content": "I've been learning piano for 3 years."},
    {"speaker": "assistant", "content": "That's impressive! Do you practice daily?"},
    {"speaker": "user", "content": "Yes, I practice every evening. My favorite piece is Moonlight Sonata."}
]

# 提取的记忆
# Entities: piano, Moonlight Sonata
# Facts:
#   - [User] has been learning [piano] for [3 years]
#   - [User] practices [piano] every evening
#   - [User]'s favorite piece is [Moonlight Sonata]
# EntityTree:
#   - Interests: piano, Moonlight Sonata
#   - Habits: practicing every evening
```

## 总结

本步骤介绍了：
- MemBrain 的核心系统架构
- 6 个主要数据模型及其关系
- 数据的完整流转过程
- 3 个具体的记忆使用场景示例

理解这些基础概念后，可以更好地理解后续的 API 使用和开发流程。