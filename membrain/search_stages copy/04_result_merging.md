# Stage 4: 结果合并与去重 - Result Merging & Deduplication

## 概述

在执行六条检索路径后，会得到六个结果列表。这些列表中可能包含重复的事实（因为同一事实可能被多条路径检索到）。本阶段的目标是：

1. **合并所有路径的结果**
2. **按 fact_id 去重**
3. **构建统一的候选池（Pool）**

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L434-L455)

## 详细代码分析

### 4.1 收集排名列表

```python
# membrain/retrieval/application/retrieval.py

# ── 3. Dedup into pool ───────────────────────────────────────────────
# 收集每条路径的结果 ID 列表，用于后续融合
ranked_lists = [
    [f.fact_id for f in path_a],
    [f.fact_id for f in path_b],
    [f.fact_id for f in path_b2],
    [f.fact_id for f in path_b3],
    [f.fact_id for f in path_c],
    [f.fact_id for f in path_d],
]
```

**为什么需要收集排名列表**:

1. **用于 RRF 融合**: 需要知道每个事实在每条路径中的排名
2. **保留顺序信息**: 不仅知道哪些路径返回了该事实，还知道其排名

**示例**:
```python
ranked_lists = [
    [101, 203, 87, 45, ...],  # Path A
    [203, 101, 56, 78, ...],  # Path B
    [87, 203, 101, ...],       # Path B2
    [45, 67, 89, ...],         # Path B3
    [101, 203, ...],           # Path C
    [78, 56, 101, ...],        # Path D
]
```

### 4.2 去重合并

```python
# 使用字典保持首次出现的顺序
seen: dict[int, RetrievedFact] = {}

# 遍历所有路径的结果
for lst in (path_a, path_b, path_b2, path_b3, path_c, path_d):
    for f in lst:
        if f.fact_id not in seen:
            # 首次出现：添加到字典
            seen[f.fact_id] = f

# 转换为列表
pool = list(seen.values())
```

**为什么使用字典去重**:

1. **O(1) 查找**: 字典查找效率高
2. **保持顺序**: 字典保持插入顺序（Python 3.7+）
3. **保留首次出现**: 首次出现的事实对象被保留

**示例图解**:

```
Path A: [fact_1, fact_2, fact_3, fact_4, fact_5]
Path B: [fact_2, fact_1, fact_6, fact_7]
Path B2: [fact_3, fact_2, fact_1, fact_8]
Path B3: [fact_4, fact_9, fact_10]
Path C:  [fact_1, fact_5, fact_11]
Path D:  [fact_6, fact_2, fact_12]

合并去重后:
pool = [fact_1, fact_2, fact_3, fact_4, fact_5, fact_6, fact_7, fact_8, fact_9, fact_10, fact_11, fact_12]
```

### 4.3 空池检查

```python
if not pool:
    return _empty_result()
```

**空池定义**:

```python
def _empty_result() -> dict:
    return {
        "packed_context": "",
        "packed_token_count": 0,
        "fact_ids": [],
        "facts": [],
        "sessions": [],
        "raw_messages": [],
    }
```

**为什么需要空池检查**:

1. **避免后续错误**: 如果没有检索到任何事实，直接返回空结果
2. **快速路径**: 无需执行后续昂贵的后处理操作

## 完整示例

### 输入

假设六条路径返回以下结果：

```python
path_a = [
    RetrievedFact(fact_id=101, text="Caroline had a picnic", source="bm25"),
    RetrievedFact(fact_id=203, text="Picnic was at Central Park", source="bm25"),
    RetrievedFact(fact_id=87, text="Summer picnic event", source="bm25"),
]

path_b = [
    RetrievedFact(fact_id=203, text="Picnic was at Central Park", source="embed"),
    RetrievedFact(fact_id=101, text="Caroline had a picnic", source="embed"),
    RetrievedFact(fact_id=56, text="Friends attended the picnic", source="embed"),
]

path_b2 = [
    RetrievedFact(fact_id=87, text="Summer picnic event", source="embed"),
    RetrievedFact(fact_id=203, text="Picnic was at Central Park", source="embed"),
    RetrievedFact(fact_id=101, text="Caroline had a picnic", source="embed"),
]

path_b3 = [
    RetrievedFact(fact_id=45, text="Outdoor gathering", source="embed"),
]

path_c = [
    RetrievedFact(fact_id=101, text="Caroline had a picnic", source="tree"),
]

path_d = [
    RetrievedFact(fact_id=78, text="Caroline organized event", source="bm25_parsed"),
]
```

### 处理过程

```python
# Step 1: 收集排名列表
ranked_lists = [
    [101, 203, 87],           # Path A
    [203, 101, 56],           # Path B
    [87, 203, 101],           # Path B2
    [45],                     # Path B3
    [101],                    # Path C
    [78],                     # Path D
]

# Step 2: 去重合并
seen = {}

# 遍历 Path A
for f in path_a:
    seen[101] = fact_1
    seen[203] = fact_2
    seen[87] = fact_3

# 遍历 Path B
# fact_2, fact_1 已存在，跳过
# fact_6 (id=56) 不存在，添加
seen[56] = fact_6

# 遍历 Path B2
# fact_3, fact_2, fact_1 已存在，跳过

# 遍历 Path B3
# fact_4 (id=45) 不存在，添加
seen[45] = fact_4

# 遍历 Path C
# fact_1 已存在，跳过

# 遍历 Path D
# fact_5 (id=78) 不存在，添加
seen[78] = fact_5

# Step 3: 转换为列表（保持插入顺序）
pool = [fact_1, fact_2, fact_3, fact_6, fact_4, fact_5]
```

### 输出

```python
pool = [
    RetrievedFact(fact_id=101, text="Caroline had a picnic", source="bm25"),
    RetrievedFact(fact_id=203, text="Picnic was at Central Park", source="bm25"),
    RetrievedFact(fact_id=87, text="Summer picnic event", source="bm25"),
    RetrievedFact(fact_id=56, text="Friends attended the picnic", source="embed"),
    RetrievedFact(fact_id=45, text="Outdoor gathering", source="embed"),
    RetrievedFact(fact_id=78, text="Caroline organized event", source="bm25_parsed"),
]
```

## 关键数据结构

### RetrievedFact 类

```python
# membrain/retrieval/core/types.py

@dataclass
class RetrievedFact:
    fact_id: int
    text: str
    source: str  # "bm25" | "embed" | "tree" | "bm25_parsed"
    rerank_score: float = 0.0
    time_info: str = ""
    entity_ref: str = ""
    aspect_path: str = ""
    aspect_summary: str = ""
    session_number: int | None = None
```

**字段说明**:

| 字段 | 说明 |
|------|------|
| `fact_id` | 事实的唯一标识符 |
| `text` | 事实的文本内容 |
| `source` | 首次检索到该事实的路径 |
| `rerank_score` | 融合后的评分（后续阶段填充） |
| `time_info` | 时间信息（后续阶段填充） |
| `entity_ref` | 关联的实体引用（后续阶段填充） |
| `aspect_path` | 实体树路径（后续阶段填充） |
| `aspect_summary` | 方面摘要（后续阶段填充） |
| `session_number` | 所属会话编号（后续阶段填充） |

## 与后续阶段的关联

### 1. 排名列表用于 RRF 融合

```python
# 后续阶段使用 ranked_lists 进行 RRF 融合
ranked_lists = [
    [f.fact_id for f in path_a],
    [f.fact_id for f in path_b],
    # ...
]
```

### 2. Pool 继续后续处理

```python
# 后续阶段在 pool 基础上进行处理
pool = list(seen.values())

# 后处理
_inject_time_annotations(pool, db)
_inject_session_numbers(pool, db)

# 融合
if strategy == "rerank":
    round1_facts = _fuse_rerank(...)
else:
    _fuse_rrf(pool, ranked_lists)
```

## 总结

本阶段的核心逻辑:

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 收集排名列表 | 为 RRF 融合准备数据 |
| 2 | 遍历所有路径 | 合并六个结果集 |
| 3 | 字典去重 | 移除重复事实，保持首次出现 |
| 4 | 空池检查 | 快速路径，无结果时直接返回 |

**关键设计决策**:

1. **使用字典保持顺序**: Python 3.7+ 字典保持插入顺序
2. **保留首次出现**: 首次出现的事实被保留，后续重复被忽略
3. **保留原始 source**: 虽然可能有多条路径返回同一事实，但只保留第一次遇到的 source

这一阶段为后续的融合和排序奠定了基础。