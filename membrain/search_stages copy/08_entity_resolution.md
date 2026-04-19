# Stage 8: 实体引用解析 - Entity Reference Resolution

## 概述

实体引用解析是 MemBrain 搜索过程中将记忆文本转换为可读格式的关键步骤。在记忆存储中，实体通常以简写形式（如 `[Caroline]`）引用，但 LLM 需要知道这些引用的具体含义。

本阶段的目标是将方括号形式的实体引用（如 `[Caroline]`）替换为实体的完整信息（如 `Caroline (user's mother)`）。

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L521-L523)
- **解析函数**: [fact_retrieval.py](file:///home/project/MemBrain/membrain/infra/retrieval/fact_retrieval.py#L25-L45)

## 详细代码分析

### 8.1 入口

```python
# membrain/retrieval/application/retrieval.py

# Resolve entity bracket refs on the final selected facts so that the
# first occurrence in the output gets the description appended.
_resolve_pool_entity_refs(round1_facts, db)
```

**注释说明**:
- 只在第一次出现时附加描述
- 避免重复显示相同的描述

### 8.2 实体引用解析函数

```python
# membrain/infra/retrieval/fact_retrieval.py

# 正则表达式：匹配 [alias] 形式的引用
_ENTITY_BRACKET_RE = re.compile(r"\[([^\]:#]+)\]")


def _resolve_entity_refs(
    text: str,
    ref_map: dict[str, str],  # alias → canonical
) -> str:
    """Replace [alias] entity bracket refs with canonical names.

    Leaves time tokens ([word::DATE], [2023-05-07]) untouched.
    """

    def _replace(m: re.Match) -> str:
        alias = m.group(1)
        canonical = ref_map.get(alias)
        if canonical is None:
            return m.group(0)  # 保持原样
        return canonical

    return _ENTITY_BRACKET_RE.sub(_replace, text)
```

**正则表达式解析**:

```python
_ENTITY_BRACKET_RE = re.compile(r"\[([^\]:#]+\]")
```

- `\[` - 匹配左方括号
- `([^\]:#]+)` - 捕获组，匹配非 `]`, `:`, `#` 的字符（一个或多个）
- `\]` - 匹配右方括号

**不匹配的例外**:
- `[word::DATE]` - 包含 `:`，不匹配
- `[2023-05-07]` - 包含 `-`，不匹配
- `[Caroline]` - 匹配 ✓

### 8.3 批量解析函数

```python
# membrain/retrieval/application/retrieval.py

def _resolve_pool_entity_refs(
    pool: list[RetrievedFact],
    db: Session,
) -> None:
    # 1. 收集所有实体引用
    all_aliases: set[str] = set()
    for fact in pool:
        for alias in _ENTITY_BRACKET_RE.findall(fact.text):
            all_aliases.add(alias)
    
    if not all_aliases:
        return
    
    # 2. 查询数据库获取规范名称
    rows = db.execute(
        sa_text("""
            SELECT DISTINCT ON (fr.alias_text)
                   fr.alias_text, e.canonical_ref
            FROM fact_refs fr
            JOIN entities e ON e.entity_id = fr.entity_id
            WHERE fr.alias_text = ANY(:texts)
            ORDER BY fr.alias_text
        """),
        {"texts": list(all_aliases)},
    ).fetchall()
    
    # 3. 构建别名到规范名称的映射
    alias_canonical = {r[0]: r[1] for r in rows}
    
    if not alias_canonical:
        return
    
    # 4. 替换每个事实中的引用
    for fact in pool:
        fact.text = _resolve_entity_refs(fact.text, alias_canonical)
```

**数据库查询解释**:

```sql
SELECT DISTINCT ON (fr.alias_text)
       fr.alias_text, e.canonical_ref
FROM fact_refs fr
JOIN entities e ON e.entity_id = fr.entity_id
WHERE fr.alias_text = ANY(:texts)
ORDER BY fr.alias_text
```

- `fact_refs` 表：存储实体别名与实体的关联
- `entities` 表：存储实体的规范名称和描述
- `DISTINCT ON`: 对每个别名只返回一个结果

## 完整示例

### 输入

```python
# 事实文本
facts_texts = [
    "Caroline [Caroline] went to Paris with her friends.",
    "She met [John] at the café.",
    "The picnic happened on [2023-07-15].",
    "[Caroline] brought sandwiches for everyone.",
]

# 数据库中的别名映射
alias_canonical = {
    "Caroline": "Caroline (user's mother)",
    "John": "John (Caroline's colleague)",
}
```

### 处理过程

```
Step 1: 收集所有别名
  all_aliases = {"Caroline", "John"}

Step 2: 查询数据库
  rows = [
    ("Caroline", "Caroline (user's mother)"),
    ("John", "John (Caroline's colleague)"),
  ]

Step 3: 构建映射
  alias_canonical = {
    "Caroline": "Caroline (user's mother)",
    "John": "John (Caroline's colleague)",
  }

Step 4: 替换引用
  - "Caroline [Caroline] went to Paris..."
    → "Caroline (user's mother) went to Paris..."
  
  - "She met [John] at the café."
    → "She met John (Caroline's colleague) at the café."
  
  - "The picnic happened on [2023-07-15]."
    → "The picnic happened on [2023-07-15]." (不变，日期格式不匹配)
  
  - "[Caroline] brought sandwiches..."
    → "Caroline (user's mother) brought sandwiches..."
```

### 输出

```python
resolved_texts = [
    "Caroline (user's mother) went to Paris with her friends.",
    "She met John (Caroline's colleague) at the café.",
    "The picnic happened on [2023-07-15].",
    "Caroline (user's mother) brought sandwiches for everyone.",
]
```

## 为什么需要实体引用解析

### 1. 记忆存储的简化表示

在记忆提取阶段，实体以简写形式存储：
- 节省存储空间
- 便于批量处理
- 避免重复存储实体信息

### 2. LLM 需要完整信息

LLM 在生成答案时需要知道：
- 实体是谁
- 实体的背景信息
- 实体之间的关系

### 3. 避免重复描述

MemBrain 的设计只在第一次出现时显示完整描述：
- 第一幕: `[Caroline]` → `Caroline (user's mother)`
- 后续出现: 保持简洁，不重复显示描述

## 数据模型

### Entity 表结构

```sql
CREATE TABLE entities (
    entity_id VARCHAR PRIMARY KEY,
    task_id INTEGER NOT NULL,
    canonical_ref VARCHAR NOT NULL,  -- 规范名称，如 "Caroline"
    "desc" TEXT,                      -- 描述，如 "user's mother"
    desc_embedding VECTOR,            -- 描述的嵌入向量
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### FactRef 表结构

```python
class FactRefModel:
    id: Integer
    entity_id: String      -- 关联到 entities 表
    alias_text: String     -- 别名文本
    fact_id: Integer       -- 关联到 facts 表
```

### 引用格式

| 格式 | 示例 | 说明 |
|------|------|------|
| 实体引用 | `[Caroline]` | 需要解析 |
| 时间引用 | `[2023-07-15]` | 日期，不解析 |
| 时间标签 | `[yesterday::DATE]` | 相对时间，不解析 |

## 错误处理

```python
# 1. 没有别名
if not all_aliases:
    return  # 提前返回

# 2. 数据库没有匹配
if not alias_canonical:
    return  # 提前返回

# 3. 找不到的别名
def _replace(m: re.Match) -> str:
    alias = m.group(1)
    canonical = ref_map.get(alias)
    if canonical is None:
        return m.group(0)  # 保持原样
    return canonical
```

## 性能优化

### 1. 批量查询

```python
# 一次查询获取所有别名
rows = db.execute(
    sa_text("""...""")
    {"texts": list(all_aliases)},
).fetchall()
```

### 2. 提前返回

```python
if not all_aliases:
    return

if not alias_canonical:
    return
```

### 3. 正则表达式预编译

```python
_ENTITY_BRACKET_RE = re.compile(r"\[([^\]:#]+\]")
```

## 总结

实体引用解析阶段的核心功能：

| 功能 | 说明 |
|------|------|
| **识别引用** | 使用正则表达式识别 `[alias]` 形式 |
| **批量查询** | 一次性获取所有别名对应的规范名称 |
| **替换文本** | 将简写替换为完整描述 |
| **保留特殊格式** | 跳过时间标记等不需要解析的内容 |

**关键设计决策**:

1. **只解析实体引用**: 时间标记 `[DATE]` 保持原样
2. **只替换存在的别名**: 数据库中没有的别名保持原样
3. **支持重复出现**: 同一事实中多次出现的同一别名都会被替换

这一阶段确保了 LLM 在生成答案时能够理解实体的具体含义，提供更准确的响应。