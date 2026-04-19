# 交互过程文档

本文档详细分析 MemBrain 的写入过程与搜索过程的交互机制。

## 文档目录

| 阶段 | 名称 | 文档 |
|:----:|------|------|
| 1 | 消息输入与会话创建 | [01_message_input_session_create.md](01_message_input_session_create.md) |
| 2 | 异步消化触发与会话摘要 | [02_async_digest_session_summary.md](02_async_digest_session_summary.md) |
| 3 | 批量提取与实体事实生成 | [03_batch_extraction_entity_fact_generation.md](03_batch_extraction_entity_fact_generation.md) |
| 4 | 实体消重与规范化 | [04_entity_resolution.md](04_entity_resolution.md) |
| 5 | 数据库持久化 | [05_database_persistence.md](05_database_persistence.md) |
| 6 | 实体树更新 | [06_entity_tree_update.md](06_entity_tree_update.md) |

## 快速概览

```
用户消息
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: 消息输入与会话创建                        │
│ • 验证参数                                        │
│ • 创建/查找数据集和任务                           │
│ • 创建会话记录                                    │
│ • 存储消息                                        │
│ • 触发异步消化                                    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: 异步消化触发与会话摘要                    │
│ • 查找未消化的会话                                │
│ • 生成会话摘要                                    │
│ • 生成摘要向量                                    │
│ • 更新会话状态                                    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: 批量提取与实体事实生成                   │
│ • 格式化消息                                      │
│ • 两轮实体提取                                    │
│ • 事实生成 + 验证                                │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4: 实体消重与规范化                         │
│ • Layer 1: 精确匹配                              │
│ • Layer 2: MinHash + Jaccard                     │
│ • Layer 3: LLM 语义匹配                         │
│ • 实体规范化                                      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 5: 数据库持久化                             │
│ • 构建引用映射                                    │
│ • 创建/更新实体                                   │
│ • 插入事实                                        │
│ • 建立关联                                        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 6: 实体树更新                               │
│ • 查找受影响实体                                  │
│ • 锁定实体                                        │
│ • 提取 Aspects                                   │
│ • 更新树结构                                      │
└─────────────────────────────────────────────────────┘
```

## 核心数据结构

### 会话 (Session)

```
ChatSessionModel:
  - id: 会话主键
  - task_id: 任务 ID
  - session_number: 会话编号
  - session_time: 会话时间
  - digested_at: 消化完成时间
```

### 消息 (Message)

```
ChatMessageModel:
  - id: 消息主键
  - session_id: 所属会话
  - speaker: 发言者 (user/assistant)
  - content: 消息内容
  - position: 位置顺序
```

### 实体 (Entity)

```
EntityModel:
  - entity_id: UUID
  - task_id: 任务 ID
  - canonical_ref: 规范名称
  - desc: 描述
  - desc_embedding: 描述向量
```

### 事实 (Fact)

```
FactModel:
  - id: 事实主键
  - task_id: 任务 ID
  - batch_id: 批次 ID
  - session_number: 会话编号
  - text: 事实文本
  - text_embedding: 文本向量
  - status: 状态 (active/archived)
```

### 实体树 (Entity Tree)

```
EntityTreeModel:
  - id: 树主键
  - task_id: 任务 ID
  - entity_id: 实体 ID
  - fact_ids: 事实 ID 列表 (JSON)
  - structure_json: 树结构 (JSON)
```