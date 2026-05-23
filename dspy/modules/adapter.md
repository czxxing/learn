# DSPy Adapter 模块详解

> 本文档详细分析 DSPy Adapter 适配器的实现原理和使用方法。

## 目录

- [一、概述](#一概述)
- [二、架构设计](#二架构设计)
- [三、BaseAdapter 核心实现](#三baseadapter-核心实现)
- [四、ChatAdapter 实现](#四chatadapter-实现)
- [五、JSONAdapter 实现](#五jsonadapter-实现)
- [六、消息格式化流程](#六消息格式化流程)
- [七、原生功能支持](#七原生功能支持)
- [八、使用示例](#八使用示例)

---

## 一、概述

Adapter 是 DSPy 的核心抽象层，负责在模块/签名和语言模型之间进行格式转换。

### 1.1 核心职责

```
┌─────────────────────────────────────────────────────────────────┐
│                     输入端                                      │
│                                                                 │
│  Module.forward()                                               │
│       │                                                        │
│       ▼                                                        │
│  Signature + Inputs + Demos                                     │
│       │                                                        │
└───────┼─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Adapter 层                                 │
│                                                                 │
│  1. format(): 签名 → LM 可理解的格式                             │
│  2. LM 调用: 发送请求到语言模型                                  │
│  3. parse(): LM 响应 → 结构化输出                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      输出端                                      │
│                                                                 │
│  Prediction 对象                                                 │
│       │                                                        │
└───────┼─────────────────────────────────────────────────────────┘
```

### 1.2 可用适配器

| 适配器 | 描述 | 适用场景 |
|--------|------|----------|
| `ChatAdapter` | 默认聊天格式 | 通用场景 |
| `JSONAdapter` | JSON 格式 | 需要结构化输出 |
| `XMLAdapter` | XML 格式 | 兼容性需求 |
| `TwoStepAdapter` | 两步处理 | 复杂任务 |
| `Image` | 图像类型 | 多模态输入 |

---

## 二、架构设计

### 2.1 类层次结构

```
Adapter (base.py)  ← 基类
    │
    ├── ChatAdapter (chat_adapter.py)  ← 默认
    ├── JSONAdapter (json_adapter.py)
    ├── XMLAdapter (xml_adapter.py)
    └── TwoStepAdapter (two_step_adapter.py)
```

### 2.2 调用流程

```python
class Adapter:
    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        # 1. 预处理：处理原生功能
        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)

        # 2. 格式化输入
        messages = self.format(processed_signature, demos, inputs)

        # 3. LM 调用
        outputs = lm(messages=messages, **lm_kwargs)

        # 4. 后处理：解析输出
        return self._call_postprocess(processed_signature, signature, outputs, lm, lm_kwargs)
```

---

## 三、BaseAdapter 核心实现

### 3.1 初始化

```python
class Adapter:
    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[Type]] | None = None,
    ):
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.native_response_types = native_response_types or _DEFAULT_NATIVE_RESPONSE_TYPES
```

### 3.2 预处理 (_call_preprocess)

处理原生 LM 功能，如函数调用、引用等：

```python
def _call_preprocess(self, lm, lm_kwargs, signature, inputs):
    # 1. 检查函数调用
    if self.use_native_function_calling:
        tool_call_input_field_name = self._get_tool_call_input_field_name(signature)
        tool_call_output_field_name = self._get_tool_call_output_field_name(signature)

        if tool_call_output_field_name and lm.supports_function_calling:
            # 获取工具定义
            tools = inputs[tool_call_input_field_name]
            tools = tools if isinstance(tools, list) else [tools]

            # 转换为 litellm 格式
            lm_tools = [tool.format_as_litellm_function_call() for tool in tools]
            lm_kwargs["tools"] = lm_tools

            # 从签名中移除工具相关字段
            signature = signature.delete(tool_call_output_field_name)
            signature = signature.delete(tool_call_input_field_name)
            return signature

    # 2. 处理原生响应类型 (Citations, Reasoning)
    for name, field in signature.output_fields.items():
        if field.annotation in self.native_response_types:
            signature = field.annotation.adapt_to_native_lm_feature(
                signature, name, lm, lm_kwargs
            )

    return signature
```

### 3.3 后处理 (_call_postprocess)

解析 LM 响应为结构化数据：

```python
def _call_postprocess(self, processed_signature, original_signature, outputs, lm, lm_kwargs):
    values = []

    for output in outputs:
        # 1. 提取文本、logprobs、tool_calls
        if isinstance(output, dict):
            text = output["text"]
            logprobs = output.get("logprobs")
            tool_calls = output.get("tool_calls")
        else:
            text = output

        # 2. 解析文本
        if text:
            value = self.parse(processed_signature, text)
            # 补全原始签名中可能缺失的字段
            for field_name in original_signature.output_fields.keys():
                if field_name not in value:
                    value[field_name] = None

        # 3. 处理函数调用
        if tool_calls:
            tool_calls = [
                {
                    "name": v["function"]["name"],
                    "args": json_repair.loads(v["function"]["arguments"]),
                }
                for v in tool_calls
            ]
            value[tool_call_output_field_name] = ToolCalls.from_dict_list(tool_calls)

        # 4. 处理原生响应类型
        for name, field in original_signature.output_fields.items():
            if field.annotation in self.native_response_types:
                parsed_value = field.annotation.parse_lm_response(output)
                if parsed_value is not None:
                    value[name] = parsed_value

        values.append(value)

    return values
```

---

## 四、ChatAdapter 实现

### 4.1 概述

`ChatAdapter` 是默认适配器，使用分隔符格式与语言模型交互。

### 4.2 核心方法

#### format_system_message()

构建系统消息，包含三个部分：

```python
def format_system_message(self, signature):
    return (
        f"{self.format_field_description(signature)}\n"
        f"{self.format_field_structure(signature)}\n"
        f"{self.format_task_description(signature)}"
    )
```

**生成示例**：

```
Your input fields are:
- question: str

Your output fields are:
- answer: str

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]

[[ ## answer ## ]]
[[ ## completed ## ]]

In adhering to this structure, your objective is: Answer the question.
```

#### format_field_description()

```python
def format_field_description(self, signature):
    return (
        f"Your input fields are:\n{get_field_description_string(signature.input_fields)}\n"
        f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"
    )
```

#### format_field_structure()

使用分隔符定义输出格式：

```python
def format_field_structure(self, signature):
    parts = []
    parts.append("All interactions will be structured in the following way...")

    def format_fields(fields):
        return self.format_field_with_value({...})

    parts.append(format_fields(signature.input_fields))
    parts.append(format_fields(signature.output_fields))
    parts.append("[[ ## completed ## ]]\n")

    return "\n\n".join(parts)
```

#### format_demos()

构建 Few-shot 示例：

```python
def format_demos(self, signature, demos):
    messages = []
    for demo in demos:
        # User message
        user_content = self.format_user_message_content(
            signature, demo, prefix="Input:", suffix="\n\nOutput:"
        )
        messages.append({"role": "user", "content": user_content})

        # Assistant message
        assistant_content = self.format_assistant_message_content(
            signature, demo
        )
        messages.append({"role": "assistant", "content": assistant_content})

    return messages
```

---

## 五、JSONAdapter 实现

### 5.1 概述

`JSONAdapter` 使用 JSON Schema 来指导 LM 输出结构化数据。

### 5.2 核心差异

与 ChatAdapter 不同，JSONAdapter：
- 在 system message 中包含 JSON Schema
- 使用 `response_format` 参数（如果支持）
- 解析时直接提取 JSON

### 5.3 使用场景

```python
import dspy

# 使用 JSONAdapter
dspy.configure(adapter=dspy.JSONAdapter())

extract = dspy.Predict("text -> entities")
result = extract(text="John lives in NYC")

# 输出会被解析为 JSON 格式
print(result.entities)  # {"names": ["John"], "cities": ["NYC"]}
```

---

## 六、消息格式化流程

### 6.1 完整消息构建

```
Adapter.format(signature, demos, inputs)
    │
    ├── 1. System Message
    │         │
    │         ├── format_field_description()
    │         │         └── "Your input fields are:\n- question: str\n..."
    │         │
    │         ├── format_field_structure()
    │         │         └── "[[ ## field ## ]]" 分隔符
    │         │
    │         └── format_task_description()
    │                   └── "In adhering to this structure..."
    │
    ├── 2. Demo Messages (可选)
    │         │
    │         ├── User: format_user_message_content(demo)
    │         └── Assistant: format_assistant_message_content(demo)
    │
    └── 3. Current Input
              │
              └── User: format_user_message_content(inputs)
```

### 6.2 最终消息格式

```python
[
    {
        "role": "system",
        "content": """Your input fields are:
- question: str

Your output fields are:
- answer: str

All interactions will be structured in the following way...

[[ ## question ## ]]

[[ ## answer ## ]]
[[ ## completed ## ]]

In adhering to this structure, your objective is: Answer the question."""
    },
    {
        "role": "user",
        "content": "Input:\n\nquestion: What is 2+2?\n\nOutput:\n\n[[ ## answer ## ]]\n\n4"
    },
    {
        "role": "assistant",
        "content": "[[ ## answer ## ]]\n\n4"
    },
    {
        "role": "user",
        "content": "Input:\n\nquestion: What is 3+3?\n\nOutput:\n\n[[ ## answer ## ]]"
    }
]
```

---

## 七、原生功能支持

### 7.1 函数调用

```python
class ToolCallingAdapter(dspy.ChatAdapter):
    def __init__(self):
        super().__init__(use_native_function_calling=True)

# 使用
dspy.configure(adapter=ToolCallingAdapter())

class Calculator(dspy.Signature):
    expression: str = dspy.InputField(desc="Math expression")
    result: float = dspy.OutputField()
    tool_calls: list[dspy.Tool] = dspy.OutputField()

calc = dspy.Predict(Calculator)
result = calc(expression="2 + 2")
print(result.tool_calls)  # [ToolCall(name="calculate", args={"expr": "2+2"})]
```

### 7.2 引用解析

```python
class CitationAdapter(dspy.ChatAdapter):
    def __init__(self):
        super().__init__(
            native_response_types=[dspy.Citations]
        )

class RAG(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: dspy.Citations = dspy.OutputField()

rag = dspy.Predict(RAG)
result = rag(question="...", context="...")
print(result.citations)  # Citations对象，包含引文信息
```

---

## 八、使用示例

### 8.1 全局配置适配器

```python
import dspy

# 配置全局适配器
dspy.configure(adapter=dspy.JSONAdapter())

# 所有 Predict 都使用 JSONAdapter
predict = dspy.Predict("question -> answer")
```

### 8.2 创建自定义适配器

```python
import re

class MyAdapter(dspy.Adapter):
    def format_field_description(self, signature):
        # 自定义字段描述格式
        parts = []
        for name, field in signature.input_fields.items():
            desc = field.json_schema_extra.get("desc", "")
            parts.append(f"{name}: {desc}")
        return "Inputs:\n" + "\n".join(parts)

    def format_field_structure(self, signature):
        # 使用自定义分隔符
        return "Output format: <output>...</output>"

    def format_task_description(self, signature):
        return signature.instructions or "Complete the task."

    def parse(self, signature, text):
        # 自定义解析逻辑
        # 提取 <output> 标签内容
        match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
        if match:
            return {"answer": match.group(1)}
        return {"answer": text}

# 使用自定义适配器
dspy.configure(adapter=MyAdapter())
```

### 8.3 适配器切换

```python
# 方法1: 全局配置
dspy.configure(adapter=dspy.XMLAdapter())

# 方法2: 运行时设置
with dspy.context(adapter=dspy.JSONAdapter()):
    result = predict(question="...")

# 方法3: 创建带适配器的 LM
lm = dspy.LM("openai/gpt-4o", adapter=dspy.XMLAdapter())
```

---

## 九、源码文件

| 文件 | 路径 |
|------|------|
| BaseAdapter | [adapters/base.py](file:///home/project/dspy/dspy/adapters/base.py) |
| ChatAdapter | [adapters/chat_adapter.py](file:///home/project/dspy/dspy/adapters/chat_adapter.py) |
| JSONAdapter | [adapters/json_adapter.py](file:///home/project/dspy/dspy/adapters/json_adapter.py) |
| XMLAdapter | [adapters/xml_adapter.py](file:///home/project/dspy/dspy/adapters/xml_adapter.py) |
| TwoStepAdapter | [adapters/two_step_adapter.py](file:///home/project/dspy/dspy/adapters/two_step_adapter.py) |
| Types | [adapters/types/](file:///home/project/dspy/dspy/adapters/types/) |

---

*本文档基于 DSPy v2.x 源码分析生成*