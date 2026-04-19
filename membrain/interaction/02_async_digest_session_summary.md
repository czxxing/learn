# Stage 2: 异步消化触发与会话摘要

## 概述

这一阶段负责在消息存储到数据库后，后台异步触发记忆提取流程。主要包括：
1. 触发异步消化任务
2. 查找需要消化的会话
3. 生成会话摘要
4. 准备消息给后续阶段处理

## 代码位置

- **消化入口**: [memory.py](file:///home/project/MemBrain/membrain/api/routes/memory.py#L320-L360)
- **会话摘要**: [session_summarizer.py](file:///home/project/MemBrain/membrain/memory/application/session_summarizer.py)
- **工作流**: [session_memory_workflow.py](file:///home/project/MemBrain/membrain/memory/application/session_memory_workflow.py)

## 详细代码分析

### 2.1 异步消化任务

```python
# memory.py

async def _run_digest(task_pk: int, agent_profile: str | None):
    """运行消化任务"""
    from membrain.memory.application.session_memory_workflow import SessionMemoryWorkflow
    
    workflow = SessionMemoryWorkflow(
        task_id=task_pk,
        profile=agent_profile,
    )
    await workflow.digest_all()
```

### 2.2 会话摘要工作流

```python
# session_memory_workflow.py

class SessionMemoryWorkflow:
    def __init__(self, task_id: int, profile: str | None = None):
        self.task_id = task_id
        self.profile = profile
        # 初始化各种客户端和存储
        
    async def digest_all(self):
        """消化所有未消化的会话"""
        
        # 1. 查找未消化的会话
        undigested_sessions = self._find_undigested_sessions()
        
        for session in undigested_sessions:
            # 2. 为每个会话生成摘要
            await self._digest_session(session)
            
        # 3. 清理已完成的任务
        self._cleanup()
```

### 2.3 查找未消化的会话

```python
def _find_undigested_sessions(self) -> list[ChatSessionModel]:
    """查找需要消化的会话"""
    
    query = (
        db.query(ChatSessionModel)
        .filter(
            ChatSessionModel.task_id == self.task_id,
            ChatSessionModel.digested_at.is_(None)  # 未消化
        )
        .order_by(ChatSessionModel.session_number)
    )
    return query.all()
```

### 2.4 会话摘要生成

```python
async def _generate_session_summary(
    self,
    session: ChatSessionModel,
    messages: list[ChatMessageModel]
) -> str:
    """为会话生成摘要"""
    
    # 1. 格式化消息文本
    messages_text = self._format_messages(messages)
    
    # 2. 调用 LLM 生成摘要
    agent, settings = self.factory.get_agent("session-summarizer")
    prompts = self.registry.render_prompts(
        "session-summarizer",
        messages_json=messages_text,
    )
    
    result = await run_agent_with_retry(
        agent,
        instructions=prompts,
        model_settings=settings,
    )
    
    return result.output.summary
```

### 2.5 消息格式化

```python
def _format_messages(self, messages: list[ChatMessageModel]) -> str:
    """将消息格式化为文本"""
    
    lines = []
    for msg in messages:
        speaker = msg.speaker
        content = msg.content
        time_str = msg.message_time_raw or ""
        
        if time_str:
            lines.append(f"[{speaker}] ({time_str}): {content}")
        else:
            lines.append(f"[{speaker}]: {content}")
    
    return "\n".join(lines)
```

## 会话摘要数据结构

```python
class SessionSummaryModel(Base):
    __tablename__ = "session_summaries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    session_number = Column(Integer, nullable=False)
    summary = Column(Text, nullable=False)  # 摘要文本
    summary_embedding = Column(Vector(768))  # 摘要向量
    created_at = Column(DateTime, default=datetime.utcnow)
```

## 处理流程图

```
Stage 1: 消息输入
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 异步任务触发                                                     │
│ asyncio.create_task(_run_digest(task_pk, agent_profile))       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 查找未消化的会话                                                 │
│ SELECT * FROM chat_sessions                                     │
│   WHERE task_id = ? AND digested_at IS NULL                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 遍历每个会话                                                     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 步骤 1: 获取会话消息                                      │   │
│   │ SELECT * FROM chat_messages                             │   │
│   │   WHERE session_id = ?                                 │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 步骤 2: 格式化消息文本                                    │   │
│   │ [user]: message                                        │   │
│   │ [assistant]: message                                   │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 步骤 3: 调用 LLM 生成摘要                                │   │
│   │ agent: session-summarizer                               │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 步骤 4: 生成摘要向量                                      │   │
│   │ embed_client.embed(summary)                            │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 步骤 5: 保存摘要                                         │   │
│   │ INSERT INTO session_summaries                          │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                       │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 更新会话状态                                                     │
│ UPDATE chat_sessions SET digested_at = NOW()                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 触发批量提取                                                     │
│ BatchIngester.ingest_batch(...)                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 输入示例

```python
# 未消化的会话
session = {
    "id": 1,
    "task_id": 1,
    "session_number": 1,
    "session_time": "2024-01-15T14:30:00",
    "digested_at": None
}

# 会话消息
messages = [
    {"speaker": "user", "content": "I had lunch with John yesterday.", "position": 0},
    {"speaker": "assistant", "content": "That sounds nice!", "position": 1},
    {"speaker": "user", "content": "We went to Luigi's Pizza. It's his favorite.", "position": 2}
]

# 格式化后的消息
messages_text = """
[user]: I had lunch with John yesterday.
[assistant]: That sounds nice!
[user]: We went to Luigi's Pizza. It's his favorite.
"""

# LLM 生成的摘要
summary = "User had lunch with their friend John at Luigi's Pizza yesterday. John mentioned it's his favorite restaurant."
```

## 与后续阶段的关联

```
Stage 2: 异步消化触发
    │
    ├──→ SessionSummaryModel (会话摘要)
    │         │
    │         └──→ 用于搜索时的会话检索
    │
    └──→ 触发 Stage 3: 批量提取
              │
              ├──→ BatchIngester.ingest_batch
              └──→ IngestWorkflow
                    │
                    ├──→ 实体提取
                    ├──→ 事实生成
                    ├──→ 实体消重
                    └──→ 数据库持久化
```

## 关键设计决策

### 1. 异步处理

```python
# 后台异步执行，不阻塞主请求
t = asyncio.create_task(_run_digest(task_pk, agent_profile))
_background_digest_tasks.add(t)
```

**优点**:
- 用户请求立即返回
- 消化过程可以批量优化
- 支持并发处理多个会话

### 2. 会话摘要的作用

1. **搜索支持**: 提供会话级别的检索
2. **上下文**: 为 LLM 提供会话概览
3. **聚合**: 将多条消息聚合成简洁摘要

### 3. 向量生成

```python
# 为摘要生成向量，用于相似度检索
summary_embedding = embed_client.embed(summary)
```

**用途**:
- 基于向量的会话检索
- 语义相似度计算

## 总结

这一阶段的核心功能：

| 功能 | 说明 |
|------|------|
| 异步触发 | 后台执行消化任务 |
| 会话查找 | 找到未消化的会话 |
| 摘要生成 | LLM 生成会话摘要 |
| 向量生成 | 为摘要生成嵌入向量 |
| 状态更新 | 标记会话为已消化 |

**关键输出**:
- `SessionSummaryModel`: 会话摘要记录
- `summary_embedding`: 摘要向量
- 触发批量提取流程