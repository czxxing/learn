# Stage 10: 上下文打包 - Context Packing

## 概述

上下文打包是 MemBrain 搜索过程的最后处理阶段，负责将检索到的事实和会话整理成符合 LLM 输入要求的格式。这一阶段需要考虑：

1. **Token 预算**: 确保上下文不超过 LLM 的上下文窗口限制
2. **信息密度**: 在有限空间内最大化有用信息
3. **格式规范**: 生成结构化、易于理解的输出格式

MemBrain 使用两种预算：
- **事实预算**: 4500 tokens
- **会话预算**: 1500 tokens

## 代码位置

- **主入口**: [retrieval.py](file:///home/project/MemBrain/membrain/retrieval/application/retrieval.py#L543-L558)
- **事实打包**: [budget_pack.py](file:///home/project/MemBrain/membrain/retrieval/core/budget_pack.py#L29-L67)
- **会话格式化**: [budget_pack.py](file:///home/project/MemBrain/membrain/retrieval/core/budget_pack.py#L86-L112)

## 详细代码分析

### 10.1 入口代码

```python
# membrain/retrieval/application/retrieval.py

# ── 8. Pack context ───────────────────────────────────────────────────
packed = budget_pack(round1_facts, max_tokens=_FACT_BUDGET_TOKENS)  # 4500

session_section = format_session_section(sessions, _SESSION_BUDGET_TOKENS)  # 1500

# 将会话章节放在前面（更重要的信息）
if session_section:
    packed.text = session_section + "\n\n" + packed.text
    packed.token_count += estimate_tokens(session_section)
```

**预算常量**:

```python
_FACT_BUDGET_TOKENS = 4500    # 事实 token 预算
_SESSION_BUDGET_TOKENS = 1500  # 会话 token 预算
```

### 10.2 事实预算打包

```python
# membrain/retrieval/core/budget_pack.py

def budget_pack(
    facts: list[RetrievedFact],
    max_tokens: int = settings.QA_BUDGET_MAX_TOKENS,
) -> PackedContext:
    """Pack facts into token budget as a flat bullet list."""
    
    # Step 1: 按 rerank_score 排序
    sorted_facts = sorted(facts, key=lambda f: f.rerank_score, reverse=True)
    
    # Step 2: 贪婪填充预算
    selected: list[RetrievedFact] = []
    total_tokens = 0
    
    for fact in sorted_facts:
        line = _format_fact_line(fact)
        line_tokens = estimate_tokens(line)
        if total_tokens + line_tokens > max_tokens:
            continue  # 跳过超出预算的事实
        
        selected.append(fact)
        total_tokens += line_tokens
    
    # Step 3: 按时间顺序输出
    sorted_selected = sorted(selected, key=_sort_key_time)
    
    # Step 4: 格式化
    lines = ["## Additional Facts"] + [_format_fact_line(f) for f in sorted_selected]
    text = "\n".join(lines)
    
    return PackedContext(
        text=text,
        token_count=estimate_tokens(text),
        fact_ids=[f.fact_id for f in sorted_selected],
    )
```

**打包算法图解**:

```
输入: facts = [fact_A(score=0.9), fact_B(score=0.8), fact_C(score=0.7), fact_D(score=0.6), fact_E(score=0.5)]
预算: max_tokens = 4500

Step 1: 按分数排序
sorted_facts = [A(0.9), B(0.8), C(0.7), D(0.6), E(0.5)]

Step 2: 贪婪填充
- fact_A: 100 tokens → total = 100 ✓
- fact_B: 80 tokens → total = 180 ✓
- fact_C: 120 tokens → total = 300 ✓
- fact_D: 90 tokens → total = 390 ✓
- fact_E: 95 tokens → total = 485 ✓
(假设每条约 100 tokens，实际可容纳约 45 条)

Step 3: 按时间排序
假设时间顺序: C, A, B, D, E
sorted_selected = [C, A, B, D, E]

Step 4: 格式化输出
## Additional Facts
- fact_C text (known from 2023-07-15)
- fact_A text (known from 2023-07-15)
- fact_B text (known from 2023-06-01)
- fact_D text
- fact_E text
```

### 10.3 事实格式化

```python
# membrain/retrieval/core/budget_pack.py

def _format_fact_line(fact: RetrievedFact) -> str:
    """Format a single fact as a bullet line with resolved absolute dates."""
    
    # 检查是否有内联日期标注
    has_inline = bool(_RELATIVE_DATE_RE.search(fact.text))
    
    # 处理内联日期标注
    line = f"- {_resolve_inline_dates(fact.text)}"
    
    # 如果没有内联日期，添加时间上下文
    if fact.time_info and not has_inline:
        line += f" (known from message on {_clean_time_info(fact.time_info)})"
    
    return line
```

**格式化示例**:

```
示例 1: 有内联日期的事实
  原始: "Picnic on [last week::DATE] was fun"
  输出: "- Picnic on [2023-07-15] was fun"
  
示例 2: 无内联日期，但有时间信息
  原始: "Caroline worked at Google"
  time_info: "[2023-01-15]"
  输出: "- Caroline worked at Google (known from message on 2023-01-15)

示例 3: 无时间信息
  原始: "Some random fact"
  输出: "- Some random fact
```

### 10.4 Token 估算

```python
# membrain/retrieval/core/budget_pack.py

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4 + 1
```

**估算公式**: `tokens ≈ characters / 4 + 1`

- 这是一个粗略的估算
- 实际 token 数量取决于具体内容
- 英文约 4 字符/token，中文约 1-2 字符/token

### 10.5 会话章节格式化

```python
# membrain/retrieval/core/budget_pack.py

def format_session_section(
    sessions: list[RetrievedSession],
    max_tokens: int,
) -> str:
    """Format session summaries as a context section, respecting token budget."""
    
    if not sessions:
        return ""
    
    header = "## Relevant Episodes"
    lines = [header]
    budget = max_tokens - estimate_tokens(header)
    
    for s in sessions:
        # 格式化每个会话
        entry = f"**{s.subject}**: {s.content}\n---"
        cost = estimate_tokens(entry)
        
        if budget - cost < 0:
            break  # 超出预算，停止添加
        
        lines.append(entry)
        budget -= cost
    
    return "\n\n".join(lines) if len(lines) > 1 else ""
```

**会话格式化示例**:

```markdown
## Relevant Episodes

**Picnic at Central Park**: 
Caroline organized a picnic with her friends at Central Park in July 2023. 
The weather was sunny and warm. They enjoyed sandwiches, fruits, and lemonade.
---
**Summer activities**:
This summer was filled with outdoor activities including picnics, hiking, 
and beach trips. Caroline particularly enjoyed the weekend gatherings.
---
```

### 10.6 上下文合并

```python
# 事实打包
packed = budget_pack(round1_facts, max_tokens=4500)

# 会话格式化
session_section = format_session_section(sessions, 1500)

# 合并（会话在前，事实在后）
if session_section:
    packed.text = session_section + "\n\n" + packed.text
    packed.token_count += estimate_tokens(session_section)
```

**最终输出格式**:

```markdown
## Relevant Episodes

**Session Subject 1**: Session content...
---
**Session Subject 2**: Session content...
---

## Additional Facts

- Fact text (known from YYYY-MM-DD)
- Fact text (known from YYYY-MM-DD)
- Fact text
- ...
```

## PackedContext 数据结构

```python
@dataclass
class PackedContext:
    text: str              # 格式化的文本
    token_count: int       # 估算的 token 数量
    fact_ids: list[int]   # 包含的事实 ID 列表
```

## 完整示例

### 输入

```python
# 已排序的事实
round1_facts = [
    RetrievedFact(
        fact_id=101,
        text="Caroline worked at Google",
        rerank_score=0.9,
        time_info="[2023-01-15]"
    ),
    RetrievedFact(
        fact_id=102,
        text="She was a software engineer",
        rerank_score=0.85,
        time_info="[2023-01-15]"
    ),
    RetrievedFact(
        fact_id=103,
        text="She worked there for 3 years",
        rerank_score=0.8,
        time_info="[2023-01-15]"
    ),
    RetrievedFact(
        fact_id=104,
        text="Her manager was John",
        rerank_score=0.7,
        time_info="[2023-02-01]"
    ),
]

# 会话列表
sessions = [
    RetrievedSession(
        subject="Google Career",
        content="Caroline worked at Google from 2020 to 2023 as a software engineer...",
        score=0.95,
    ),
    RetrievedSession(
        subject="Work Colleagues",
        content="During her time at Google, Caroline worked closely with her manager John...",
        score=0.85,
    ),
]
```

### 处理过程

```
Step 1: 事实打包 (max_tokens=4500)
  排序后: [101, 102, 103, 104]
  
  贪婪填充:
  - fact_101: "Caroline worked at Google (known from 2023-01-15)"
             → 60 tokens → total = 60 ✓
  - fact_102: "She was a software engineer (known from 2023-01-15)"
             → 50 tokens → total = 110 ✓
  - fact_103: "She worked there for 3 years (known from 2023-01-15)"
             → 45 tokens → total = 155 ✓
  - fact_104: "Her manager was John (known from 2023-02-01)"
             → 40 tokens → total = 195 ✓
  
  按时间排序: [101, 102, 103, 104] (假设都是 2023-01/02)

Step 2: 会话格式化 (max_tokens=1500)
  header: "## Relevant Episodes" → 4 tokens
  budget: 1500 - 4 = 1496
  
  - session_1: 200 tokens → budget = 1296 ✓
  - session_2: 180 tokens → budget = 1116 ✓

Step 3: 合并
  session_section + "\n\n" + fact_section
```

### 最终输出

```markdown
## Relevant Episodes

**Google Career**: Caroline worked at Google from 2020 to 2023 as a software engineer. 
She was part of the Search team and worked on various machine learning projects.
---
**Work Colleagues**: During her time at Google, Caroline worked closely with her manager John 
and collaborated with several team members on product launches.
---

## Additional Facts

- Caroline worked at Google (known from 2023-01-15)
- She was a software engineer (known from 2023-01-15)
- She worked there for 3 years (known from 2023-01-15)
- Her manager was John (known from 2023-02-01)
```

## 为什么这样设计

### 1. 贪婪填充策略

- **按分数排序**: 优先选择最相关的事实
- **贪心**: 每个事实一旦加入就不退出
- **效率**: O(n) 时间复杂度

### 2. 时间排序输出

- **可读性**: 按时间顺序更易于理解
- **连贯性**: 相关事实按时间顺序呈现
- **上下文**: 帮助理解事件发展

### 3. 会话在前

- **优先级**: 会话提供更广泛的上下文
- **引导**: 帮助 LLM 理解对话背景
- **结构**: 会话主题可以帮助组织思路

### 4. Token 估算

- **快速**: 简单计算，无需实际分词
- **保守**: 估算偏高，避免超出限制
- **通用**: 适用于各种语言

## 配置参数

```python
# membrain/config.py

QA_BUDGET_MAX_TOKENS = 2000    # 实际上这里只是默认值
# 实际使用:
_FACT_BUDGET_TOKENS = 4500     # 硬编码在 retrieval.py
_SESSION_BUDGET_TOKENS = 1500   # 硬编码在 retrieval.py
```

## 总结

上下文打包阶段的核心逻辑：

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 按 rerank_score 排序 | 优先选择相关事实 |
| 2 | 贪婪填充预算 | 在 token 限制内最大化信息 |
| 3 | 按时间排序输出 | 提供连贯的时间线 |
| 4 | 格式化事实 | 添加时间上下文 |
| 5 | 格式化会话 | 生成会话章节 |
| 6 | 合并输出 | 会话在前，事实在后 |

**关键设计决策**:

1. **贪婪算法**: 简单高效，适合在线服务
2. **时间排序**: 输出更符合人类阅读习惯
3. **会话优先**: 更广泛的上下文在前
4. **双重预算**: 事实和会话分别控制

这一阶段生成最终用于 LLM 回答问题的上下文文本。