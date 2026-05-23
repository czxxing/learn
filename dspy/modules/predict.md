# DSPy Predict 模块详解

> 本文档详细分析 `dspy.Predict` 的实现原理和使用方法。

## 目录

- [一、概述](#一概述)
- [二、核心实现](#二核心实现)
- [三、执行流程](#三执行流程)
- [四、状态管理](#四状态管理)
- [五、配置系统](#五配置系统)
- [六、使用示例](#六使用示例)

---

## 一、概述

`dspy.Predict` 是 DSPy 最基础的预测模块，负责将输入转换为输出，是构建复杂流水线的基础组件。

### 1.1 类定义

```python
class Predict(Module, Parameter):
    """Basic DSPy module that maps inputs to outputs using a language model."""
```

### 1.2 核心职责

| 职责 | 描述 |
|------|------|
| **签名管理** | 管理输入输出签名 |
| **预处理** | 验证输入、获取 LM、准备配置 |
| **适配调用** | 调用 Adapter 进行格式化与解析 |
| **后处理** | 构建 Prediction 对象 |
| **状态持久化** | 支持序列化与反序列化 |

---

## 二、核心实现

### 2.1 初始化

```python
def __init__(self, signature: str | type[Signature], callbacks=None, **config):
    super().__init__(callbacks=callbacks)
    self.stage = random.randbytes(8).hex()  # 唯一标识
    self.signature = ensure_signature(signature)  # 确保签名有效
    self.config = config  # 默认配置
    self.reset()  # 重置状态
```

### 2.2 状态重置

```python
def reset(self):
    self.lm = None      # 语言模型（可在模块或全局设置）
    self.traces = []    # 追踪记录（用于优化器）
    self.train = []     # 训练数据
    self.demos = []     # Few-shot 示例
```

---

## 三、执行流程

### 3.1 完整流程图

```
Predict.forward(**kwargs)
    │
    ├─── _forward_preprocess(**kwargs)
    │         │
    │         ├── 1. 提取配置参数
    │         │         ├── signature (可选覆盖)
    │         │         ├── demos (示例数据)
    │         │         └── config (LM参数)
    │         │
    │         ├── 2. 获取 LM 实例
    │         │         ├── kwargs.pop("lm") 优先
    │         │         ├── self.lm 其次
    │         │         └── settings.lm 最后兜底
    │         │
    │         ├── 3. 处理温度和生成数量
    │         │         └── 如果 n>1 且 temperature<=0.15，自动设为 0.7
    │         │
    │         ├── 4. 填充默认值
    │         ├── 5. 类型检查
    │         └── 返回: (lm, config, signature, demos, kwargs)
    │
    ├─── adapter.__call__(lm, config, signature, demos, inputs)
    │         │
    │         ├── Adapter.format() → 构建消息
    │         ├── LM 调用
    │         └── Adapter.parse() → 解析输出
    │
    └─── _forward_postprocess(completions, signature)
              │
              ├── 记录到 settings.trace
              └── 返回 Prediction.from_completions()
```

### 3.2 预处理详解 (_forward_preprocess)

```python
def _forward_preprocess(self, **kwargs):
    # 1. 提取特权参数
    signature = kwargs.pop("signature", self.signature)  # 可选覆盖签名
    demos = kwargs.pop("demos", self.demos)             # 示例数据
    config = {**self.config, **kwargs.pop("config", {})} # 合并配置

    # 2. 获取 LM 实例（优先级：参数 > 实例 > 全局）
    lm = kwargs.pop("lm", self.lm) or settings.lm

    if lm is None:
        raise ValueError("No LM is loaded...")

    # 3. 处理温度和生成数量
    temperature = config.get("temperature") or lm.kwargs.get("temperature")
    num_generations = config.get("n") or lm.kwargs.get("n") or 1

    if (temperature is None or temperature <= 0.15) and num_generations > 1:
        config["temperature"] = 0.7  # 自动增加随机性

    # 4. 填充默认值
    for k, v in signature.input_fields.items():
        if k not in kwargs and v.default is not PydanticUndefined:
            kwargs[k] = v.default

    # 5. 类型检查
    if settings.warn_on_type_mismatch:
        for field_name, field_info in signature.input_fields.items():
            if field_name in kwargs:
                if not _is_value_compatible_with_type(kwargs[field_name], field_info.annotation):
                    logger.warning(f"Type mismatch for field '{field_name}'...")

    return lm, config, signature, demos, kwargs
```

### 3.3 前向执行 (forward)

```python
def forward(self, **kwargs):
    # 1. 预处理
    lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)

    # 2. 获取适配器
    adapter = settings.adapter or ChatAdapter()

    # 3. 流式处理检查
    if self._should_stream():
        with settings.context(caller_predict=self):
            completions = adapter(
                lm, lm_kwargs=config, signature=signature, 
                demos=demos, inputs=kwargs
            )
    else:
        completions = adapter(
            lm, lm_kwargs=config, signature=signature, 
            demos=demos, inputs=kwargs
        )

    # 4. 后处理
    return self._forward_postprocess(completions, signature, **kwargs)
```

---

## 四、状态管理

### 4.1 序列化

```python
def dump_state(self, json_mode=True):
    state = {
        "traces": self.traces,
        "train": self.train,
        "demos": self._serialize_demos(json_mode),
        "signature": self.signature.dump_state(),
        "lm": self.lm.dump_state() if self.lm else None,
    }
    return state

def load_state(self, state, allow_unsafe_lm_state=False):
    for name, value in state.items():
        if name not in ["signature", "extended_signature", "lm"]:
            setattr(self, name, value)

    self.signature = self.signature.load_state(state["signature"])
    self.lm = LM(**sanitized_lm_state) if state["lm"] else None
    return self
```

### 4.2 状态流转

```
创建 Predict
    │
    ▼
reset() → 初始状态
    │
    ▼
forward() 调用
    │
    ├─── demos 被消费
    ├─── traces 被记录
    └─── 返回 Prediction
    │
    ▼
可通过 dump_state() 保存
    │
    ▼
可通过 load_state() 恢复
```

---

## 五、配置系统

### 5.1 初始化配置

```python
# 创建时设置
predict = dspy.Predict(
    "question -> answer",
    temperature=0.7,
    max_tokens=100,
)
```

### 5.2 调用时覆盖

```python
# 覆盖初始化配置
predict(
    question="...",
    config={"temperature": 1.0}
)
```

### 5.3 update_config / get_config

```python
predict.update_config(temperature=0.5)
config = predict.get_config()
```

### 5.4 支持的参数

| 参数 | 描述 | 示例 |
|------|------|------|
| `temperature` | 采样温度 | `0.7` |
| `max_tokens` | 最大生成长度 | `100` |
| `n` | 生成数量 | `3` |
| `rollout_id` | 缓存隔离ID | `1` |

---

## 六、使用示例

### 6.1 基本使用

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 简单预测
predict = dspy.Predict("question -> answer")
result = predict(question="What is the capital of France?")
print(result.answer)  # Paris
```

### 6.2 带类型注解

```python
from typing import List

class ExtractSignature(dspy.Signature):
    """Extract named entities from text."""
    text: str = dspy.InputField(desc="Input text")
    entities: List[str] = dspy.OutputField(desc="Extracted entities")

extract = dspy.Predict(ExtractSignature)
result = extract(text="John lives in New York and works at Google.")
print(result.entities)  # ["John", "New York", "Google"]
```

### 6.3 带 Few-shot 示例

```python
predict = dspy.Predict("question -> answer")

# 添加示例
predict.demos = [
    {"question": "2+2=?", "answer": "4"},
    {"question": "3*3=?", "answer": "9"},
]

result = predict(question="5*5=?")
print(result.answer)
```

### 6.4 运行时指定 demos

```python
demos = [
    {"question": "What is 1+1?", "answer": "2"},
    {"question": "What is 2+2?", "answer": "4"},
]

result = predict(demos=demos, question="What is 3+3?")
```

### 6.5 多输出字段

```python
class SentimentAnalysis(dspy.Signature):
    text: str = dspy.InputField()
    sentiment: str = dspy.OutputField()
    confidence: float = dspy.OutputField()

analyzer = dspy.Predict(SentimentAnalysis)
result = analyzer(text="I love this product!")
print(result.sentiment)   # positive
print(result.confidence)  # 0.95
```

### 6.6 调用时指定 LM

```python
# 为单个调用指定不同的 LM
result = predict(
    question="...",
    lm=dspy.LM("anthropic/claude-3-sonnet")
)
```

---

## 七、源码文件

| 文件 | 路径 |
|------|------|
| Predict 类 | [predict/predict.py](file:///home/project/dspy/dspy/predict/predict.py) |
| Parameter 基类 | [predict/parameter.py](file:///home/project/dspy/dspy/predict/parameter.py) |
| Signature | [signatures/signature.py](file:///home/project/dspy/dspy/signatures/signature.py) |
| Prediction | [primitives/prediction.py](file:///home/project/dspy/dspy/primitives/prediction.py) |

---

*本文档基于 DSPy v2.x 源码分析生成*