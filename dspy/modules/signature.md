# DSPy Signature 模块详解

> 本文档详细分析 `dspy.Signature` 的实现原理和使用方法。

## 目录

- [一、概述](#一概述)
- [二、核心概念](#二核心概念)
- [三、核心实现](#三核心实现)
- [四、Field 定义](#四field-定义)
- [五、签名操作](#五签名操作)
- [六、使用示例](#六使用示例)

---

## 一、概述

Signature 是 DSPy 的核心抽象，用于定义模块的输入输出规范。

### 1.1 核心职责

```
┌─────────────────────────────────────────┐
│            Signature                    │
│                                         │
│  输入字段:                              │
│    - question: str                     │
│                                         │
│  输出字段:                              │
│    - answer: str                       │
│                                         │
│  指令:                                 │
│    "Answer the question."              │
└─────────────────────────────────────────┘
```

| 职责 | 描述 |
|------|------|
| **类型安全** | 定义输入输出字段及其类型 |
| **文档生成** | 自动生成字段描述 |
| **指令管理** | 管理任务指令 |

---

## 二、核心概念

### 2.1 短格式签名

```python
# 使用字符串定义签名
signature = dspy.Signature("question -> answer")

# 多字段
signature = dspy.Signature("input1, input2 -> output1, output2")

# 带指令
signature = dspy.Signature("question -> answer", "Answer clearly.")
```

### 2.2 长格式签名

```python
class MySignature(dspy.Signature):
    """自定义签名类"""
    input_field: str = dspy.InputField(desc="输入描述")
    output_field: str = dspy.OutputField(desc="输出描述")
```

### 2.3 InputField / OutputField

```python
class InputField(FieldInfo):
    """输入字段定义"""
    def __init__(
        self,
        desc: str = None,      # 字段描述
        default=None,           # 默认值
        format=None,            # 格式要求
    ): ...

class OutputField(FieldInfo):
    """输出字段定义"""
    def __init__(
        self,
        desc: str = None,      # 字段描述
        default=None,           # 默认值
    ): ...
```

---

## 三、核心实现

### 3.1 元类设计

```python
class SignatureMeta(type(BaseModel)):
    def __call__(cls, *args, **kwargs):
        if cls is Signature:
            # 调用 Signature(...) 时创建新类
            return make_signature(*args, **kwargs)
        return super().__call__(*args, **kwargs)
```

### 3.2 make_signature()

```python
def make_signature(signature_str, instructions=None, custom_types=None):
    """从字符串创建签名类"""
    # 1. 解析签名字符串
    inputs_str, outputs_str = signature_str.split("->")
    inputs = parse_fields(inputs_str)
    outputs = parse_fields(outputs_str)

    # 2. 构建类体
    class_dict = {
        "__doc__": instructions or _default_instructions(...),
    }

    # 3. 添加输入字段
    for name, type_hint in inputs:
        class_dict[name] = InputField()

    # 4. 添加输出字段
    for name, type_hint in outputs:
        class_dict[name] = OutputField()

    # 5. 创建签名类
    return SignatureMeta.__new__(
        SignatureMeta,
        "DynamicSignature",
        (Signature,),
        class_dict,
    )
```

### 3.3 签名类结构

```python
class Signature(BaseModel, metaclass=SignatureMeta):
    """签名基类"""
    # 子类需要定义字段
    pass

class DynamicSignature(Signature):
    """动态创建的签名"""
    question: str = InputField()
    answer: str = OutputField()
```

---

## 四、Field 定义

### 4.1 InputField

```python
class InputField(FieldInfo):
    def __init__(
        self,
        desc: str = None,
        default=None,
        format=None,
        annotation=None,
        **kwargs,
    ):
        super().__init__(
            default=default,
            json_schema_extra={"desc": desc, "format": format, **kwargs}
        )
        self._dspy_info = {"desc": desc, "format": format}
```

### 4.2 OutputField

```python
class OutputField(FieldInfo):
    def __init__(
        self,
        desc: str = None,
        default=None,
        **kwargs,
    ):
        super().__init__(
            default=default,
            json_schema_extra={"desc": desc, **kwargs}
        )
```

---

## 五、签名操作

### 5.1 prepend() - 添加字段

```python
class ExtendedSignature(Signature):
    reasoning: str = OutputField(desc="${reasoning}")
    answer: str = OutputField()

# 添加 reasoning 字段
extended = base.prepend(name="reasoning", field=OutputField(desc="${reasoning}"))
```

### 5.2 delete() - 删除字段

```python
# 从签名中移除字段
reduced = signature.delete("temporary_field")
```

### 5.3 修改指令

```python
class CustomSignature(Signature):
    """自定义指令"""
    input: str
    output: str
```

### 5.4 字段访问

```python
signature = dspy.Signature("question -> answer")

# 输入字段
signature.input_fields  # {"question": FieldInfo}

# 输出字段
signature.output_fields  # {"answer": FieldInfo}

# 指令
signature.instructions  # "Given the fields `question`, produce the fields `answer`."

# 所有字段
signature.field_names()  # ["question", "answer"]
```

---

## 六、使用示例

### 6.1 字符串格式

```python
# 简单签名
sig1 = dspy.Signature("question -> answer")

# 多字段
sig2 = dspy.Signature("input1, input2 -> output1, output2")

# 带指令
sig3 = dspy.Signature(
    "question -> answer",
    "Answer the question clearly and concisely."
)
```

### 6.2 类定义格式

```python
class QuestionAnswering(dspy.Signature):
    """回答用户问题"""
    question: str = dspy.InputField(desc="用户的问题")
    answer: str = dspy.OutputField(desc="简洁的答案")
```

### 6.3 带类型注解

```python
from typing import List

class EntityExtraction(dspy.Signature):
    """提取命名实体"""
    text: str = dspy.InputField(desc="输入文本")
    entities: List[str] = dspy.OutputField(desc="提取的实体列表")
```

### 6.4 自定义描述

```python
class SentimentAnalysis(dspy.Signature):
    """情感分析"""
    text: str = dspy.InputField(
        desc="待分析的文本",
        format="UTF-8"
    )
    sentiment: str = dspy.OutputField(
        desc="情感极性: positive, negative, 或 neutral"
    )
    confidence: float = dspy.OutputField(
        desc="置信度 (0-1)"
    )
```

### 6.5 动态字段

```python
# 基于模板创建签名
def create_extraction_sig(fields: List[str]):
    inputs = ", ".join(fields)
    return dspy.Signature(f"{inputs} -> summary")

# 使用
sig = create_extraction_sig(["title", "author", "date"])
predictor = dspy.Predict(sig)
```

### 6.6 签名组合

```python
# 在模块中组合多个签名
class MultiTaskPipeline(dspy.Module):
    def __init__(self):
        self.extract = dspy.Predict("text -> entities")
        self.classify = dspy.Predict("entities -> category")
        self.summarize = dspy.Predict("text, category -> summary")
```

### 6.7 类型检测

```python
sig = dspy.Signature("question -> answer")

# 获取字段类型
sig.input_fields["question"].annotation  # <class 'str'>

# 获取描述
sig.input_fields["question"].json_schema_extra.get("desc")  # 描述文字
```

---

## 七、源码文件

| 文件 | 路径 |
|------|------|
| Signature | [signatures/signature.py](file:///home/project/dspy/dspy/signatures/signature.py) |
| Field | [signatures/field.py](file:///home/project/dspy/dspy/signatures/field.py) |
| Utils | [signatures/utils.py](file:///home/project/dspy/dspy/signatures/utils.py) |

---

## 八、高级用法

### 8.1 自动类型推断

```python
# DSPy 会自动推断字段类型为 str
sig = dspy.Signature("question -> answer")
# 等同于:
# question: str = InputField()
# answer: str = OutputField()
```

### 8.2 默认值

```python
class WithDefaults(dspy.Signature):
    required_field: str = InputField()
    optional_field: str = InputField(default="default_value")
```

### 8.3 嵌套签名

```python
class ContextQA(dspy.Signature):
    context: str = InputField(desc="相关上下文")
    question: str = InputField(desc="问题")
    answer: str = OutputField(desc="基于上下文的答案")
```

---

*本文档基于 DSPy v2.x 源码分析生成*