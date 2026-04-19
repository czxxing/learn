# Stage 1: 消息输入与会话创建

## 概述

这是 MemBrain 系统的入口阶段，负责接收用户对话消息，创建会话记录，并触发后续的记忆提取流程。这一阶段是数据进入系统的第一道关口。

## 代码位置

- **API 入口**: [memory.py](file:///home/project/MemBrain/membrain/api/routes/memory.py#L174-L240)
- **会话模型**: [models.py](file:///home/project/MemBrain/membrain/storage/models.py)
- **数据解析**: [session_memory_workflow.py](file:///home/project/MemBrain/membrain/memory/application/session_memory_workflow.py)

## 详细代码分析

### 1.1 API 入口函数

```python
@router.post("/memory", response_model=MemoryResponse)
async def process_memory(req: MemoryRequest):
    """Unified memory endpoint.

    Modes (controlled by ``store`` and ``digest``):
      store=True,  digest=False  — save raw messages only
      store=True,  digest=True   — save then digest all pending sessions
      store=False, digest=True   — digest all pending sessions (no new data)
    """
    messages = [m.model_dump() for m in req.messages]

    # 参数验证
    if req.store and not messages:
        raise HTTPException(400, "messages required when store=True")
    if not req.store and not req.digest:
        raise HTTPException(400, "at least one of store or digest must be True")
```

### 1.2 数据集与任务解析

```python
# ── Resolve dataset / task ───────────────────────────────────────
with SessionLocal() as db:
    dataset, task = _get_or_create_dataset_task(
        db,
        req.dataset,
        req.task,
        req.agent_profile,
    )
    dataset_id = dataset.id
    task_pk = task.id
    agent_profile = task.agent_profile
    db.commit()
```

**数据集-任务关系**:
- 一个数据集 (Dataset) 可以包含多个任务 (Task)
- 每个任务对应一个用户的记忆空间
- 使用 `task_pk` 作为后续所有操作的主键

### 1.3 会话创建

```python
# ── Store ────────────────────────────────────────────────────────
session_pk: int | None = None
session_number: int | None = None

if req.store:
    with SessionLocal() as db:
        # 获取当前最大的会话编号
        max_sn = (
            db.query(func.max(ChatSessionModel.session_number))
            .filter_by(task_id=task_pk)
            .scalar()
        ) or 0
        session_number = max_sn + 1

        # 解析会话时间
        session_dt = None
        if req.session_time:
            try:
                session_dt = datetime.fromisoformat(req.session_time)
            except ValueError:
                pass

        # 创建会话记录
        session = ChatSessionModel(
            task_id=task_pk,
            session_number=session_number,
            session_time=session_dt,
            session_time_raw=req.session_time or None,
            digested_at=None,  # 初始未消化
        )
        db.add(session)
        db.flush()
        session_pk = session.id
```

### 1.4 消息存储

```python
        # 存储每条消息
        for pos, msg in enumerate(messages):
            msg_dt = None
            if msg.get("message_time"):
                try:
                    msg_dt = datetime.fromisoformat(msg["message_time"])
                except ValueError:
                    pass
            
            db.add(
                ChatMessageModel(
                    session_id=session_pk,
                    position=pos,
                    speaker=msg["speaker"],
                    content=msg["content"],
                    message_time=msg_dt,
                    message_time_raw=msg.get("message_time") or None,
                )
            )
        db.commit()
```

### 1.5 异步消化触发

```python
# ── Digest (async background) ────────────────────────────────────
if req.digest:
    # 创建异步任务处理消化
    t = asyncio.create_task(_run_digest(task_pk, agent_profile))
    _background_digest_tasks.add(t)
    t.add_done_callback(_background_digest_tasks.discard)
```

## 数据模型

### ChatSessionModel

```python
class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    session_number = Column(Integer, nullable=False)
    session_time = Column(DateTime)
    session_time_raw = Column(String)
    digested_at = Column(DateTime)  # 消化完成时间
    created_at = Column(DateTime, default=datetime.utcnow)
```

### ChatMessageModel

```python
class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    position = Column(Integer, nullable=False)
    speaker = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    message_time = Column(DateTime)
    message_time_raw = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
```

## 请求流程图

```
客户端请求
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ POST /api/memory                                                │
│ {                                                                │
│   "dataset": "personamem_v2",                                   │
│   "task": "user_001",                                          │
│   "messages": [...],                                            │
│   "store": true,                                                │
│   "digest": true                                                │
│ }                                                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 参数验证                                                         │
│ • 检查 store=True 时 messages 非空                              │
│ • 检查 store 或 digest 至少有一个为 True                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 数据集/任务解析                                                  │
│ • 查找或创建 dataset + task                                    │
│ • 返回 task_pk                                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 会话创建                                                         │
│ • 获取最大 session_number + 1                                   │
│ • 解析 session_time                                            │
│ • 创建 ChatSessionModel                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 消息存储                                                        │
│ • 遍历 messages                                                │
│ • 解析 message_time                                            │
│ • 创建 ChatMessageModel                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 异步消化触发                                                    │
│ • 创建 asyncio task                                            │
│ • 后台执行 _run_digest                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 返回响应                                                         │
│ {                                                                │
│   "dataset_id": 1,                                             │
│   "task_pk": 1,                                                │
│   "session_id": 1,                                             │
│   "session_number": 1,                                          │
│   "status": "stored_and_digest_queued"                          │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## 输入示例

```python
# 请求
payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": True,
    "digest": True,
    "session_time": "2024-01-15T14:30:00",
    "messages": [
        {
            "speaker": "user",
            "content": "I had lunch with my friend John at Luigi's Pizza yesterday.",
            "message_time": "2024-01-15T12:00:00"
        },
        {
            "speaker": "assistant",
            "content": "That sounds nice!",
            "message_time": "2024-01-15T12:00:30"
        },
        {
            "speaker": "user",
            "content": "Yes, we had a great time. John said it's his favorite restaurant.",
            "message_time": "2024-01-15T12:01:00"
        }
    ]
}

# 响应
{
    "dataset_id": 1,
    "task_pk": 1,
    "session_id": 1,
    "session_number": 1,
    "digested_sessions": 0,
    "status": "stored_and_digest_queued"
}
```

## 存储模式

| 模式 | store | digest | 说明 |
|------|-------|--------|------|
| 存储并消化 | True | True | 保存消息，然后提取记忆（最常用） |
| 仅存储 | True | False | 只保存原始消息 |
| 仅消化 | False | True | 消化已有消息（不保存新消息） |

## 与后续阶段的关联

```
Stage 1: 消息输入与会话创建
    │
    ├──→ ChatSessionModel (会话记录)
    ├──→ ChatMessageModel (消息记录)
    │
    └──→ 触发 Stage 2: 异步消化
              │
              ├──→ SessionMemoryWorkflow
              ├──→ BatchIngester
              └──→ IngestWorkflow (4 阶段)
```

## 关键设计决策

### 1. 异步消化

```python
# 消化是异步执行的，不阻塞主请求
if req.digest:
    t = asyncio.create_task(_run_digest(task_pk, agent_profile))
```

**优点**:
- 用户请求快速响应
- 消化过程可以批量处理
- 不影响主请求延迟

### 2. 会话编号自增

```python
max_sn = db.query(func.max(...)).scalar() or 0
session_number = max_sn + 1
```

**用途**:
- 按时间顺序追踪会话
- 用于消息关联和检索

### 3. 时间解析

```python
# 支持 ISO 格式时间，也支持原始字符串存储
session_dt = datetime.fromisoformat(req.session_time)
session_time_raw = req.session_time  # 保留原始输入
```

**灵活性**:
- 尝试解析为标准时间
- 保留原始字符串（可能有多种格式）

## 错误处理

### 1. 参数验证错误

```python
if req.store and not messages:
    raise HTTPException(400, "messages required when store=True")
```

### 2. 时间解析错误

```python
try:
    session_dt = datetime.fromisoformat(req.session_time)
except ValueError:
    pass  # 忽略错误，时间为 None
```

### 3. 数据集/任务不存在

```python
# _get_or_create_dataset_task 会自动创建
dataset, task = _get_or_create_dataset_task(db, req.dataset, req.task, ...)
```

## 总结

这一阶段的核心功能：

| 功能 | 说明 |
|------|------|
| 参数验证 | 确保必要参数存在 |
| 数据集/任务解析 | 查找或创建数据集和任务 |
| 会话创建 | 为对话创建会话记录 |
| 消息存储 | 保存原始消息到数据库 |
| 异步触发 | 后台启动记忆提取流程 |

**关键输出**:
- `task_pk`: 任务主键，用于后续所有操作
- `session_pk`: 会话主键
- `session_number`: 会话编号
- 异步任务: 触发后续的消化流程