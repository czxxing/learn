# MemBrain 写入过程 - 完整指南

本文档详细分析 MemBrain 的写入（摄取/ingestion）过程，将其分为 **5 个阶段**，每个阶段都有独立的详细文档。

## 目录

| 阶段 | 名称 | 文档 |
|:----:|------|------|
| 1 | 实体提取 | [01_entity_extraction.md](01_entity_extraction.md) |
| 2 | 事实生成 | [02_fact_generation.md](02_fact_generation.md) |
| 3 | 实体消重 | [03_entity_resolution.md](03_entity_resolution.md) |
| 4 | 数据库持久化 | [04_database_persistence.md](04_database_persistence.md) |
| 5 | 实体树更新 | [05_entity_tree_update.md](05_entity_tree_update.md) |

## 快速概览

```
用户消息
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: 实体提取                                  │
│ • 第一轮：基础 LLM 提取                            │
│ • 加载已有实体上下文                               │
│ • 第二轮：上下文增强提取                           │
│ 输出: entity_names (实体名称列表)                 │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: 事实生成                                  │
│ • 调用 fact-generator agent                       │
│ • 实体覆盖验证 (实时)                             │
│ • 回退过滤机制                                    │
│ 输出: facts (事实列表)                            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: 实体消重                                  │
│ • Layer 1: 精确匹配 (小写+空白折叠)               │
│ • Layer 2: MinHash + Jaccard 模糊匹配            │
│ • Layer 3: LLM 语义匹配                          │
│ 输出: decisions (包含 merge/create 动作)          │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4: 数据库持久化                              │
│ • 构建引用映射                                    │
│ • 创建/更新实体                                   │
│ • 插入事实 + 生成向量                            │
│ • 建立事实-实体关联                              │
│ 输出: 数据库记录                                  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 5: 实体树更新                                │
│ • 查找受影响实体                                   │
│ • 加载现有状态                                    │
│ • 提取 Aspects                                    │
│ • 构建/合并树结构                                 │
│ 输出: 实体树结构                                  │
└─────────────────────────────────────────────────────┘
```

## 核心概念

### 1. 实体 (Entity)

```python
class EntityModel:
    entity_id: str       # UUID
    task_id: int
    canonical_ref: str  # 规范名称
    desc: str           # 描述
    desc_embedding: Vector  # 描述向量
```

### 2. 事实 (Fact)

```python
class FactModel:
    id: int
    task_id: int
    batch_id: str       # 批次 ID
    session_number: int # 会话编号
    text: str           # 事实文本
    text_embedding: Vector  # 文本向量
    status: str         # 'active' | 'archived'
```

### 3. 实体树 (Entity Tree)

```
实体: Caroline
└── root
    ├── Career         # 工作
    │   └── 技能
    ├── Family        # 家庭
    │   ├── 父母
    │   └── 兄弟姐妹
    ├── Social        # 社交
    └── Interests     # 兴趣
```

## 三层实体消重策略

| 层级 | 方法 | 复杂度 | 阈值 |
|------|------|--------|------|
| Layer 1 | 精确匹配 | O(1) | 完全相等 |
| Layer 2 | MinHash + Jaccard | O(k) | ≥ 0.9 |
| Layer 3 | LLM 语义匹配 | O(n) LLM | 语义相似 |

## 两种工作流

| 工作流 | 说明 | 适用场景 |
|--------|------|----------|
| `DefaultIngestWorkflow` | 完整 4 阶段管道 | 大多数场景 |
| `PersonaMemIngestWorkflow` | 跳过实体提取，固定 ["User"] | PersonaMem v2 |

## 配置参数

```python
# 实体解析
RESOLVER_JACCARD_THRESHOLD = 0.9
RESOLVER_ENTROPY_THRESHOLD = 1.5
RESOLVER_MINHASH_PERMUTATIONS = 32
RESOLVER_MINHASH_BAND_SIZE = 4

# 提取上下文
EXTRACTION_CONTEXT_TOP_K = 20
EXTRACTION_CONTEXT_PER_QUERY = 5

# 树配置
MAX_FACTS_PER_NODE = 50
```

## 文档路径

所有详细文档位于: `docs/ingest_stages/`

- [01_entity_extraction.md](01_entity_extraction.md) - 实体提取
- [02_fact_generation.md](02_fact_generation.md) - 事实生成
- [03_entity_resolution.md](03_entity_resolution.md) - 实体消重
- [04_database_persistence.md](04_database_persistence.md) - 数据库持久化
- [05_entity_tree_update.md](05_entity_tree_update.md) - 实体树更新