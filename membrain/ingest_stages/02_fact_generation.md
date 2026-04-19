# Stage 2: 事实生成 - Fact Generation

## 概述

事实生成是 MemBrain 写入过程的第二阶段，负责从用户对话消息中提取具体的事实陈述。与实体提取不同，事实生成不仅识别实体，还提取关于这些实体的具体信息（如行为、状态、关系等）。

这一阶段使用 `fact-generator` agent，通过 LLM 从消息中提取结构化的事实信息。

## 代码位置

- **主入口**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py#L213-L275)
- **Agent 配置**: [factory.py](file:///home/project/MemBrain/membrain/agents/factory.py)
- **验证器注册**: [ingest_workflow.py](file:///home/project/MemBrain/membrain/memory/application/ingest_workflow.py#L55-L83)

## 详细代码分析

### 2.1 入口函数

```python
async def _generate_facts(
    self,
    entity_names: list[str],
    messages_text: str,
    context_text: str,
) -> tuple[list[str], list[dict]]:
    """Stage 2: Fact generation with entity coverage validation."""
    
    # ═══════════════════════════════════════════════════════════════
    # Step 1: 获取 fact-generator agent
    # ═══════════════════════════════════════════════════════════════
    
    fact_generator, generator_settings = self._factory.get_agent(
        "fact-generator",
        profile=self._profile,
    )
    
    # 注册实体覆盖验证器（确保事实只引用已知实体）
    _register_entity_coverage_validator(fact_generator)
    
    # ═══════════════════════════════════════════════════════════════
    # Step 2: 构建提示词
    # ═══════════════════════════════════════════════════════════════
    
    # 将实体列表转为 JSON
    entity_list_json = json.dumps(entity_names, ensure_ascii=False)
    
    prompts = self._registry.render_prompts(
        "fact-generator",
        profile=self._profile,
        entity_list_json=entity_list_json,          # 实体列表
        context_messages=context_text,              # 上下文消息
        messages_json=messages_text,                # 待提取的消息
    )
    
    # 允许的实体引用集合（用于验证）
    allowed_refs = set(entity_names)
    
    # ═══════════════════════════════════════════════════════════════
    # Step 3: 调用 LLM 生成事实（带实体约束）
    # ═══════════════════════════════════════════════════════════════
    
    try:
        result = await run_agent_with_retry(
            fact_generator,
            instructions=prompts,
            model_settings=generator_settings,
            deps={"allowed_entity_refs": allowed_refs},  # 传入允许的实体
        )
        facts = [f.model_dump() for f in result.output.facts]
        
    except Exception as exc:
        # 如果验证失败，尝试无约束模式
        log.warning(
            "fact-generator failed (illegal refs or schema violation), retrying without entity constraint: %s",
            exc,
        )
        
        try:
            result = await run_agent_with_retry(
                fact_generator,
                instructions=prompts,
                model_settings=generator_settings,
                deps={"allowed_entity_refs": set()},  # 无约束
            )
            facts = [f.model_dump() for f in result.output.facts]
            
            # 过滤掉包含非法引用的事实
            entity_names, facts = _apply_fact_generator_fallback(
                facts,
                allowed_refs,
            )
        except Exception:
            log.exception("fact-generator retry also failed")
            entity_names, facts = [], []
    
    # ═══════════════════════════════════════════════════════════════
    # Step 4: 记录日志并返回
    # ═══════════════════════════════════════════════════════════════
    
    log.debug(
        "fact-generator -> %d entities, %d facts", len(entity_names), len(facts)
    )
    log.info(
        "    [extract] entities (%d): %s",
        len(entity_names),
        ", ".join(entity_names) if entity_names else "(none)",
    )
    for fact in facts:
        log.info("    [extract] fact: %s", fact["text"])

    return entity_names, facts
```

### 2.2 实体覆盖验证器

这是 MemBrain 的一个关键设计，确保生成的事实只引用已提取的实体：

```python
# 记录已注册的验证器，避免重复注册
_registered: set[int] = set()


def _register_entity_coverage_validator(agent: Agent) -> None:
    """注册实体覆盖验证器，防止事实引用未提取的实体。"""
    
    # 避免重复注册
    if id(agent) in _registered:
        return
    _registered.add(id(agent))

    @agent.output_validator
    async def validate_entity_coverage(ctx: RunContext[dict], result) -> object:
        """验证生成的事实只引用了已知的实体。"""
        
        # 获取允许的实体引用集合
        allowed_refs: set[str] = ctx.deps.get("allowed_entity_refs", set())
        
        # 如果没有约束，直接返回
        if not allowed_refs:
            return result
        
        # 提取事实中的所有括号引用
        fact_refs: set[str] = set()
        for fact in result.facts:
            # 使用正则提取 [entity] 形式的引用
            fact_refs.update(_ENTITY_BRACKET_RE.findall(fact.text))
        
        # 检查是否有非法引用
        illegal = fact_refs - allowed_refs
        
        if illegal:
            # 如果有非法引用，抛出重试异常
            raise ModelRetry(
                f"These bracketed refs in your facts are not in the entity list: {sorted(illegal)}. "
                f"The allowed refs are: {sorted(allowed_refs)}. "
                f"Fix each fact to use only refs from that list, or remove the fact."
            )
        
        return result
```

### 2.3 回退过滤机制

当 LLM 验证失败时，使用回退机制过滤事实：

```python
# 正则表达式：匹配 [entity] 形式的引用
_ENTITY_BRACKET_RE = re.compile(r"\[([^\]]+)\]")


def _apply_fact_generator_fallback(
    facts: list[dict],
    allowed_refs: set[str],
) -> tuple[list[str], list[dict]]:
    """过滤掉包含非法引用的事实，并返回有效的实体列表。"""
    
    # Step 1: 过滤事实
    filtered_facts = []
    for fact in facts:
        # 提取事实中的所有引用
        refs = set(_ENTITY_BRACKET_RE.findall(fact["text"]))
        
        # 只保留引用都在允许列表中的事实
        if refs <= allowed_refs:
            filtered_facts.append(fact)
    
    # Step 2: 收集使用的实体
    used: set[str] = set()
    for fact in filtered_facts:
        used.update(_ENTITY_BRACKET_RE.findall(fact["text"]))
    
    # 返回排序后的实体列表和过滤后的事实列表
    return sorted(used), filtered_facts
```

## 事实数据结构

### 事实输出格式

```python
@dataclass
class GeneratedFact:
    text: str                    # 事实文本，如 "Caroline [went] to [Boston]"
    time: str | None = None     # 时间信息，如 "last week"
    source: str | None = None   # 来源消息（可选）
```

### 生成的事实示例

```python
# 输入消息
messages_text = """
User: I had lunch with John at Luigi's Pizza yesterday.
Assistant: That sounds nice! Was it good?
User: Yes, we had pizza and pasta. John loves their carbonara.
"""

# 实体列表
entity_names = ["John", "Luigi's Pizza", "yesterday"]

# 生成的事实
facts = [
    {
        "text": "[User] had lunch with [John] at [Luigi's Pizza] [yesterday]",
        "time": "yesterday",
    },
    {
        "text": "[John] loves [Luigi's Pizza]'s carbonara",
        "time": None,
    },
    {
        "text": "[Luigi's Pizza] serves pizza and pasta",
        "time": None,
    },
]
```

## 实体引用语法

MemBrain 使用方括号 `[entity]` 语法来标记事实中的实体引用：

```
原始消息: "Caroline met her sister Lisa at the coffee shop."
生成事实: "[Caroline] met [her sister Lisa] at [the coffee shop]"

原始消息: "John works at Google."
生成事实: "[John] works at [Google]"
```

**语法规则**:
- `[EntityName]`: 直接引用实体
- `[her sister Lisa]`: 带修饰语的实体引用
- `[the coffee shop]`: 描述性实体

## 完整处理流程

```
输入:
  entity_names = ["Caroline", "Lisa", "coffee shop"]
  messages_text = """
    User: I met my sister Lisa at the coffee shop yesterday.
    Assistant: Which coffee shop did you go to?
    User: It was the one on Main Street.
  """
  context_text = ""  (可选的上下文消息)

处理流程:

Step 1: 构建提示词
  prompt = """
  Extract facts from the following messages.
  
  Entities to extract facts about: ["Caroline", "Lisa", "coffee shop"]
  
  Messages:
  [messages_text]
  
  Guidelines:
  - Use [entity] bracket syntax to reference entities
  - Only extract facts about the given entities
  - Include time information when available
  """

Step 2: 调用 fact-generator (带约束)
  deps = {"allowed_entity_refs": {"Caroline", "Lisa", "coffee shop"}}
  
  LLM 分析消息，生成事实

Step 3: 验证输出
  - 检查每个事实中的 [entity] 引用
  - 确保所有引用都在 allowed_refs 中
  - 如果验证失败，抛出 ModelRetry

Step 4: 验证成功
  facts = [
      {"text": "[Caroline] met [Lisa] at [coffee shop] [yesterday]", ...},
      {"text": "[coffee shop] is on [Main Street]", ...},
  ]

Step 5: 回退处理（如果验证失败）
  - 重新调用 LLM（无约束）
  - 过滤掉包含非法引用的事实
  - 返回有效的事实列表

输出:
  entity_names = ["Caroline", "Lisa", "coffee shop"]
  facts = [...]
```

## 配置参数

```python
# membrain/config.py

# Agent 配置
AGENT_MAX_RETRIES = 3              # 最大重试次数
AGENT_RETRY_DELAY = 1.0           # 重试延迟（秒）

# 实体引用正则
ENTITY_BRACKET_PATTERN = r"\[([^\]]+)\]"
```

## 验证器工作流程

```
LLM 生成事实
    │
    ▼
┌─────────────────────────────────────────┐
│ 提取所有 [entity] 引用                   │
│ refs = {"Caroline", "Lisa", "coffee shop"}│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ 检查引用是否在 allowed_refs 中          │
│ allowed_refs = {"Caroline", "Lisa"}    │
│ illegal = refs - allowed_refs          │
└──────────────────┬──────────────────────┘
                   │
        ┌─────────┴─────────┐
        ▼                   ▼
   illegal 非空          illegal 为空
        │                   │
        ▼                   ▼
┌───────────────┐    ┌───────────────┐
│ 抛出 ModelRetry│    │ 返回结果      │
│ 要求重试      │    │              │
└───────────────┘    └───────────────┘
```

## 错误处理

### 场景 1: 验证失败后重试成功

```python
# 第一次调用失败（验证器抛出异常）
try:
    result = await run_agent_with_retry(...)
except ModelRetry as e:
    # 验证失败，重新调用（无约束）
    result = await run_agent_with_retry(
        fact_generator,
        deps={"allowed_entity_refs": set()}  # 无约束
    )
    
    # 过滤非法引用
    _, facts = _apply_fact_generator_fallback(facts, allowed_refs)
```

### 场景 2: 完全失败

```python
# 重试也失败
try:
    result = await run_agent_with_retry(...)
except Exception:
    log.exception("fact-generator retry also failed")
    entity_names, facts = [], []  # 返回空
```

## 完整示例

### 输入

```python
# 实体列表（来自 Stage 1）
entity_names = ["John", "Luigi's Pizza", "User"]

# 消息
messages = [
    {"role": "user", "content": "I had lunch with John at Luigi's Pizza yesterday."},
    {"role": "assistant", "content": "That sounds nice!"},
    {"role": "user", "content": "John said it's his favorite restaurant."}
]

# 上下文（可选）
context_messages = []
```

### 生成的事实

```python
# 生成的事实列表
facts = [
    {
        "text": "[User] had lunch with [John] at [Luigi's Pizza] [yesterday]",
        "time": "yesterday",
    },
    {
        "text": "[John] said [Luigi's Pizza] is [his favorite restaurant]",
        "time": None,
    },
    {
        "text": "[John] likes [Luigi's Pizza]",
        "time": None,
    },
]
```

### 验证过程

```
事实 1: "[User] had lunch with [John] at [Luigi's Pizza] [yesterday]"
  提取引用: {"User", "John", "Luigi's Pizza", "yesterday"}
  允许引用: {"John", "Luigi's Pizza", "User"}
  非法引用: {"yesterday"} → 注意: "yesterday" 是时间，不是实体
  
  验证器处理: 实际上 "yesterday" 可能被视为实体，
  取决于 agent 的理解和 allowed_refs 的定义

事实 2: "[John] said [Luigi's Pizza] is [his favorite restaurant]"
  提取引用: {"John", "Luigi's Pizza", "his favorite restaurant"}
  允许引用: {"John", "Luigi's Pizza", "User"}
  非法引用: {"his favorite restaurant"}
  
  结果: 可能被过滤或需要重试

事实 3: "[John] likes [Luigi's Pizza]"
  提取引用: {"John", "Luigi's Pizza"}
  允许引用: {"John", "Luigi's Pizza", "User"}
  非法引用: ∅
  
  结果: ✓ 验证通过
```

## 为什么这样设计

### 1. 实体覆盖验证

- **防止泄漏**: 确保 LLM 只使用已识别的实体
- **数据一致性**: 保持实体列表和事实的一致性
- **质量控制**: 过滤掉引用未知实体的事实

### 2. 回退机制

- **优雅降级**: 验证失败时不完全失败
- **数据恢复**: 尽可能保留有效的事实
- **灵活性**: 允许在约束和灵活之间平衡

### 3. 两阶段验证

```
阶段 1: LLM 输出验证 (实时)
  - 在 LLM 返回后立即验证
  - 验证失败立即重试
  - 保证输出符合要求

阶段 2: 回退过滤 (后备)
  - 如果实时验证完全失败
  - 放宽约束，重新生成
  - 过滤非法引用
```

## 总结

事实生成阶段的核心逻辑：

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 获取 fact-generator agent | 准备 LLM 调用 |
| 2 | 注册实体覆盖验证器 | 实时验证实体引用 |
| 3 | 构建提示词 | 传入实体列表和消息 |
| 4 | 调用 LLM（带约束） | 生成事实 |
| 5 | 验证输出 | 检查实体引用合法性 |
| 6 | 回退处理（可选） | 放宽约束，重新生成 |
| 7 | 记录日志 | 调试和追踪 |

**设计亮点**:

1. **实时验证**: 在 LLM 返回后立即验证
2. **防止数据泄漏**: 确保事实只引用已知实体
3. **回退机制**: 验证失败时优雅降级
4. **详细日志**: 记录生成的事实便于调试
5. **结构化输出**: 生成标准化的事实格式

这一阶段将非结构化的对话消息转化为结构化的事实陈述，为后续的实体消重和持久化提供了基础。