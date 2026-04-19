# Stage 5: 后处理 - Post-Processing

## 概述

后处理阶段是 MemBrain 搜索过程中至关重要的一步，它对合并后的候选池进行增强和优化，包括：

1. **时间注解注入**: 为每个事实添加时间信息
2. **会话编号注入**: 关联事实到具体会话
3. **Aspect 路径富化**: 添加实体树的层级上下文
4. **Aspect 级别去重**: 基于方面（Aspect）进行去重，避免信息冗余

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L457-L478)
- **时间注解**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L163-L197)
- **Aspect 处理**: [aspect_enrichment.py](file:///home/project/MemBrain/membrain/infra/retrieval/aspect_enrichment.py)

## 详细代码分析

### 5.1 时间注解注入

```python
# membrain/retrieval/application/retrieval.py

# ── 4. Post-processing ───────────────────────────────────────────────
_inject_time_annotations(pool, db)
```

**实现代码**:

```python
# membrain/retrieval/application/retrieval.py

def _inject_time_annotations(pool: list[RetrievedFact], db: Session) -> None:
    if not pool:
        return
    
    # 收集所有 fact_id
    fact_ids = [f.fact_id for f in pool]
    
    # 从 time_annotations 表获取时间信息
    rows = db.execute(
        sa_text(
            "SELECT fact_id, time_raw, time_resolved FROM time_annotations "
            "WHERE fact_id = ANY(:ids)"
        ),
        {"ids": fact_ids},
    ).fetchall()
    
    # 构建时间映射
    time_map: dict[int, str] = {}
    for fid, raw, resolved in rows:
        if not raw and not resolved:
            continue
        parts: list[str] = []
        if raw:
            parts.append(str(raw))
        if resolved:
            start, *rest = resolved.split("/", 1)
            end = rest[0] if rest else None
            parts.append(f"[{start}, {end}]" if end is not None else f"[{start}]")
        time_map[fid] = " ".join(parts)
    
    # 对于没有时间注解的事实，从 facts 表获取时间戳
    no_time_ids = [f.fact_id for f in pool if f.fact_id not in time_map]
    if no_time_ids:
        for r in db.execute(
            sa_text(
                "SELECT id, fact_ts FROM facts WHERE id = ANY(:ids) "
                "AND fact_ts IS NOT NULL"
            ),
            {"ids": no_time_ids},
        ).fetchall():
            time_map[r[0]] = f"[{r[1]}]"
    
    # 注入时间信息到事实对象
    for fact in pool:
        fact.time_info = time_map.get(fact.fact_id, "")
```

**为什么需要时间注解**:

1. **时间上下文**: 帮助 LLM 理解事实发生的时间背景
2. **排序依据**: 事实可以按时间顺序组织输出
3. **时态理解**: 区分过去、现在、未来的事件

**时间信息格式**:

```
示例 1: "yesterday [2023-05-07, ]"
示例 2: "last week [2023-06-02, 2023-06-08]"
示例 3: "now [2023-05-08T13:56:00Z, ]"
示例 4: "[2023-07-15]" (仅时间戳)
```

### 5.2 会话编号注入

```python
# membrain/retrieval/application/retrieval.py

_inject_session_numbers(pool, db)
```

**实现代码**:

```python
# membrain/retrieval/application/retrieval.py

def _inject_session_numbers(pool: list[RetrievedFact], db: Session) -> None:
    if not pool:
        return
    
    # 收集所有 fact_id
    fact_ids = [f.fact_id for f in pool]
    
    # 查询会话编号
    rows = db.execute(
        sa_text("SELECT id, session_number FROM facts WHERE id = ANY(:ids)"),
        {"ids": fact_ids},
    ).fetchall()
    
    # 构建映射
    sn_map = {r[0]: r[1] for r in rows if r[1] is not None}
    
    # 注入会话编号
    for fact in pool:
        fact.session_number = sn_map.get(fact.fact_id)
```

**为什么需要会话编号**:

1. **追踪来源**: 知道事实来自哪个会话
2. **会话检索**: 可以反向检索相关会话
3. **上下文理解**: 会话提供更广泛的对话上下文

### 5.3 Aspect 路径富化

```python
# membrain/retrieval/application/retrieval.py

all_fact_ids = [f.fact_id for f in pool]
aspect_infos = build_aspect_paths(all_fact_ids, task_id, db)

for fact in pool:
    info = aspect_infos.get(fact.fact_id)
    if info:
        fact.entity_ref = info.entity_ref
        fact.aspect_path = info.path
        fact.aspect_summary = info.leaf_desc
```

**实现代码**:

```python
# membrain/infra/retrieval/aspect_enrichment.py

def build_aspect_paths(
    fact_ids: list[int],
    task_id: int,
    db: Session,
) -> dict[int, AspectInfo]:
    """Look up entity tree ancestry for each fact."""
    
    if not fact_ids:
        return {}
    
    # SQL 查询：获取实体树层级信息
    sql = sa_text("""
        SELECT
            leaf.fact_id,
            e.canonical_ref,
            e.desc,
            parent.node_type  AS parent_type,
            parent.description AS parent_desc,
            gp.node_type       AS gp_type,
            gp.description     AS gp_desc
        FROM entity_tree_nodes leaf
        JOIN entity_tree_nodes parent ON parent.id = leaf.parent_id
        LEFT JOIN entity_tree_nodes gp ON gp.id = parent.parent_id
        JOIN entities e
          ON e.entity_id = leaf.entity_id
         AND e.task_id = :task_id
        WHERE leaf.fact_id = ANY(:fact_ids)
          AND leaf.task_id = :task_id
          AND leaf.node_type = 'leaf'
    """)
    
    rows = db.execute(sql, {"fact_ids": fact_ids, "task_id": task_id}).fetchall()
    
    # 构建 AspectInfo
    result: dict[int, AspectInfo] = {}
    
    for row in rows:
        fid = row[0]
        entity_ref = row[1] or ""
        entity_desc = row[2] or ""
        parent_type = row[3]
        parent_desc = row[4] or ""
        gp_type = row[5]
        gp_desc = row[6] or ""
        
        # 判断树结构层级
        if parent_type == "root":
            # 2层树: root → leaf
            path = parent_desc
            leaf_desc = parent_desc
            mid_desc = ""
        elif gp_type == "root":
            # 3层树: root → aspect → leaf
            path = f"{gp_desc} > {parent_desc}" if gp_desc else parent_desc
            leaf_desc = parent_desc
            mid_desc = gp_desc
        else:
            # 更深的树
            path = parent_desc
            leaf_desc = parent_desc
            mid_desc = gp_desc
        
        # 构建去重键
        leaf_key = f"{entity_ref}::{leaf_desc}"
        mid_key = f"{entity_ref}::{mid_desc}" if mid_desc else leaf_key
        
        result[fid] = AspectInfo(
            entity_ref=entity_ref,
            entity_desc=entity_desc,
            leaf_desc=leaf_desc,
            mid_desc=mid_desc,
            path=path,
            leaf_key=leaf_key,
            mid_key=mid_key,
        )
    
    return result
```

**AspectInfo 数据结构**:

```python
@dataclass
class AspectInfo:
    entity_ref: str      # 实体规范名称，如 "Caroline"
    entity_desc: str     # 实体描述，如 "user's mother"
    leaf_desc: str       # 叶子方面描述
    mid_desc: str       # 中间方面描述
    path: str           # 路径，如 "Career > Work history"
    leaf_key: str       # 去重键: "entity::leaf_desc"
    mid_key: str        # 去重键: "entity::mid_desc"
```

**实体树结构示例**:

```
实体: Caroline (user's mother)
│
├── Root: Person
│     │
│     ├── Aspect: Career
│     │     │
│     │     └── Leaf: Work history ── fact_1
│     │           └── Leaf: Education ── fact_2
│     │
│     └── Aspect: Personal Life
│           │
│           └── Leaf: Family ── fact_3
│                 └── Leaf: Hobbies ── fact_4
```

**Aspect 路径示例**:

| 事实 ID | 实体 | Path | Leaf | 格式 |
|---------|------|------|------|------|
| 101 | Caroline | Career > Work history | Work history | "Career > Work history" |
| 102 | Caroline | Career > Education | Education | "Career > Education" |
| 103 | Caroline | Personal Life > Family | Family | "Personal Life > Family" |

### 5.4 Aspect 级别去重

```python
# membrain/retrieval/application/retrieval.py

# 后续对 Path C 应用去重（在 entity_tree_search 之后）
if path_c:
    c_ids = [f.fact_id for f in path_c]
    c_aspects = build_aspect_paths(c_ids, task_id, db)
    kept = set(aspect_dedup(c_ids, c_aspects))
    path_c = [f for f in path_c if f.fact_id in kept]
```

**实现代码**:

```python
# membrain/infra/retrieval/aspect_enrichment.py

def aspect_dedup(
    fact_ids_ordered: list[int],
    aspect_infos: dict[int, AspectInfo],
    max_per_leaf: int = settings.QA_MAX_PER_LEAF_ASPECT,  # 默认 3
    max_per_mid: int = settings.QA_MAX_PER_MID_ASPECT,     # 默认 8
    protected_ids: set[int] | None = None,
) -> list[int]:
    """Filter fact IDs by aspect-level caps, preserving input order."""
    
    leaf_counts: dict[str, int] = defaultdict(int)
    mid_counts: dict[str, int] = defaultdict(int)
    kept: list[int] = []
    
    for fid in fact_ids_ordered:
        info = aspect_infos.get(fid)
        
        # 无 Aspect 信息：直接保留
        if info is None:
            kept.append(fid)
            continue
        
        # 受保护的事实（如 BM25 关键词命中）：直接保留
        if protected_ids and fid in protected_ids:
            kept.append(fid)
            continue
        
        # 检查叶子方面限制
        if leaf_counts[info.leaf_key] >= max_per_leaf:
            continue
        
        # 检查中间方面限制
        if mid_counts[info.mid_key] >= max_per_mid:
            continue
        
        # 计数并保留
        leaf_counts[info.leaf_key] += 1
        mid_counts[info.mid_key] += 1
        kept.append(fid)
    
    return kept
```

**去重逻辑图解**:

```
假设: max_per_leaf = 3, max_per_mid = 8

输入 fact_ids (按顺序):
  - fact_101: leaf="Work history", mid="Career"
  - fact_102: leaf="Work history", mid="Career"  ✓ 计数=2
  - fact_103: leaf="Work history", mid="Career"  ✓ 计数=3 (达到上限)
  - fact_104: leaf="Work history", mid="Career"  ✗ 超过 leaf 上限，跳过
  - fact_105: leaf="Education", mid="Career"     ✓ 新 leaf，计数=1
  - fact_106: leaf="Family", mid="Personal Life" ✓ 新 mid，计数=1

输出: [fact_101, fact_102, fact_103, fact_105, fact_106]
```

**为什么需要 Aspect 去重**:

1. **避免冗余**: 同一方面的多个事实可能内容重复
2. **多样性**: 确保覆盖多个不同方面
3. **可控性**: 通过参数控制去重粒度

## 完整示例

### 输入

```python
pool = [
    RetrievedFact(fact_id=101, text="Caroline worked at Google"),
    RetrievedFact(fact_id=102, text="Caroline was a software engineer at Google"),
    RetrievedFact(fact_id=103, text="Caroline's job at Google lasted 3 years"),
    RetrievedFact(fact_id=104, text="Caroline received promotion at Google"),
    RetrievedFact(fact_id=105, text="Caroline studied at MIT"),
    RetrievedFact(fact_id=106, text="Caroline's mother is Mary"),
]
```

### 处理过程

```
Step 1: 注入时间注解
  fact_101: time_info = "[2023-01-15]"
  fact_102: time_info = "[2023-01-15]"
  fact_103: time_info = "[2023-01-15]"
  fact_104: time_info = "[2023-06-01]"
  fact_105: time_info = "[2020-09-01]"
  fact_106: time_info = ""

Step 2: 注入会话编号
  fact_101: session_number = 1
  fact_102: session_number = 1
  fact_103: session_number = 1
  fact_104: session_number = 2
  fact_105: session_number = 3
  fact_106: session_number = 4

Step 3: 注入 Aspect 信息
  fact_101: entity_ref="Caroline", aspect_path="Career > Work", aspect_summary="Work"
  fact_102: entity_ref="Caroline", aspect_path="Career > Work", aspect_summary="Work"
  fact_103: entity_ref="Caroline", aspect_path="Career > Work", aspect_summary="Work"
  fact_104: entity_ref="Caroline", aspect_path="Career > Work", aspect_summary="Work"
  fact_105: entity_ref="Caroline", aspect_path="Career > Education", aspect_summary="Education"
  fact_106: entity_ref="Caroline", aspect_path="Personal > Family", aspect_summary="Family"

Step 4: Aspect 去重 (max_per_leaf=3)
  保留: fact_101, fact_102, fact_103 (Work leaf 达到3个)
  跳过: fact_104 (Work leaf 超过3个)
  保留: fact_105 (Education, 新 leaf)
  保留: fact_106 (Family, 新 mid)
```

### 输出

```python
pool = [
    RetrievedFact(
        fact_id=101, 
        text="Caroline worked at Google",
        time_info="[2023-01-15]",
        session_number=1,
        entity_ref="Caroline",
        aspect_path="Career > Work",
        aspect_summary="Work"
    ),
    # ... 类似处理其他事实
]
```

## 配置参数

```python
# membrain/config.py

QA_MAX_PER_LEAF_ASPECT = 3   # 每个叶子方面最多保留3个事实
QA_MAX_PER_MID_ASPECT = 8    # 每个中间方面最多保留8个事实
```

## 总结

后处理阶段的核心目标是为候选池中的事实添加丰富的上下文信息：

| 处理步骤 | 操作 | 目的 |
|----------|------|------|
| 时间注解 | 从数据库获取时间信息 | 提供时间上下文 |
| 会话编号 | 关联事实到会话 | 追踪来源 |
| Aspect 富化 | 添加实体树路径 | 结构化上下文 |
| Aspect 去重 | 限制每方面数量 | 避免冗余 |

**关键设计决策**:

1. **多层次去重**: 叶子级别和中间级别双重限制
2. **信息丰富化**: 从简单的事实文本到包含时间、实体、方面信息的复合结构
3. **保护机制**: 允许特定事实（如 BM25 关键词命中）绕过去重限制

这一阶段为后续的融合和排序提供了更丰富的上下文信息。