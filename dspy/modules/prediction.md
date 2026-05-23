# DSPy Prediction 模块详解

> 本文档详细分析 `dspy.Prediction` 的实现原理和使用方法。

## 目录

- [一、概述](#一概述)
- [二、核心实现](#二核心实现)
- [三、Completions 类](#三completions-类)
- [四、运算支持](#四运算支持)
- [五、使用示例](#五使用示例)

---

## 一、概述

`Prediction` 是 DSPy 模块执行后返回的结果对象，封装了语言模型的输出。

### 1.1 类层次结构

```
Example (primitives/example.py)  ← 基类
    │
    └── Prediction (primitives/prediction.py)  ← 结果类
            │
            └── Completions (内部类)
```

### 1.2 核心职责

| 职责 | 描述 |
|------|------|
| **结果封装** | 存储 LM 输出字段 |
| **多结果管理** | 支持 n>1 的多次生成 |
| **算术运算** | 支持 +、-、/、* 等运算 |
| **比较运算** | 支持 <、>、<=、>= 比较 |

---

## 二、核心实现

### 2.1 类定义

```python
class Prediction(Example):
    """A prediction object that contains the output of a DSPy module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self._demos      # Prediction 不需要 demos
        del self._input_keys # 不需要输入键

        self._completions = None  # 多次生成结果
        self._lm_usage = None     # API 使用量
```

### 2.2 从 completions 创建

```python
@classmethod
def from_completions(cls, list_or_dict, signature=None):
    obj = cls()
    obj._completions = Completions(list_or_dict, signature=signature)
    obj._store = {k: v[0] for k, v in obj._completions.items()}
    return obj
```

### 2.3 字段访问

```python
# 访问第一个 completion 的字段
result = predict(question="What is 2+2?")
result.answer  # "4" (第一个结果)

# 访问多次生成
result.completions[0].answer  # 第一个结果
result.completions[1].answer  # 第二个结果
result.completions[2].answer  # 第三个结果
```

---

## 三、Completions 类

### 3.1 类定义

```python
class Completions:
    def __init__(self, list_or_dict, signature=None):
        self.signature = signature

        if isinstance(list_or_dict, list):
            # 转换为 {field: [values...]}
            kwargs = {}
            for arg in list_or_dict:
                for k, v in arg.items():
                    kwargs.setdefault(k, []).append(v)
        else:
            kwargs = list_or_dict

        self._completions = kwargs
```

### 3.2 索引访问

```python
completions = result.completions

# 按索引访问
first = completions[0]      # Prediction 对象
second = completions[1]     # Prediction 对象

# 按字段名访问
answers = completions.answer  # ["4", "Four", "四"]
```

---

## 四、运算支持

### 4.1 算术运算

```python
# Prediction 支持 +、-、*、/ 运算（用于带分数的结果）
result1 = Prediction(score=0.8)
result2 = Prediction(score=0.9)

# 与数值运算
avg = (result1 + result2) / 2  # 0.85

# 自身运算
total = result1 + result2  # 1.7
```

### 4.2 比较运算

```python
result1 = Prediction(score=0.8)
result2 = Prediction(score=0.9)

result1 < result2   # True
result1 > result2   # False
result1 <= result2  # True
result1 >= result2  # False
```

### 4.3 类型转换

```python
result = Prediction(score=0.95)

# 转为 float
float(result)  # 0.95
```

---

## 五、使用示例

### 5.1 基本使用

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question -> answer")
result = predict(question="What is the capital of France?")

# 访问结果
print(result.answer)  # "Paris"

# 检查 completions
print(result.completions)  # Completions(answer=["Paris"])
```

### 5.2 多结果生成

```python
predict = dspy.Predict("question -> answer")

# 生成多个答案
result = predict(
    question="What is 2+2?",
    config={"n": 3, "temperature": 0.8}
)

# 访问所有结果
for i, completion in enumerate(result.completions):
    print(f"答案 {i+1}: {completion.answer}")

# 访问单个字段的所有结果
print(result.completions.answer)  # ["4", "Four", "四"]
```

### 5.3 多字段

```python
class SentimentAnalysis(dspy.Signature):
    text: str = dspy.InputField()
    sentiment: str = dspy.OutputField()
    confidence: float = dspy.OutputField()

analyzer = dspy.Predict(SentimentAnalysis)
result = analyzer(text="I love DSPy!")

print(result.sentiment)    # "positive"
print(result.confidence)  # 0.95
```

### 5.4 用于评估

```python
def evaluate(example, pred, trace=None):
    # 比较预测和期望
    return pred.answer.lower() == example.answer.lower()

# 在评估中使用
example = Example(question="2+2=?", answer="4")
pred = predict(question=example.question)

score = evaluate(example, pred)  # True 或 False
```

### 5.5 排序和选择

```python
# 多结果排序
result = predict(question="...", config={"n": 5})

# 按置信度排序（假设有 confidence 字段）
ranked = sorted(
    result.completions,
    key=lambda x: getattr(x, "confidence", 0),
    reverse=True
)

# 选择最佳结果
best = ranked[0]
```

---

## 六、源码文件

| 文件 | 路径 |
|------|------|
| Prediction | [primitives/prediction.py](file:///home/project/dspy/dspy/primitives/prediction.py) |
| Example 基类 | [primitives/example.py](file:///home/project/dspy/dspy/primitives/example.py) |

---

*本文档基于 DSPy v2.x 源码分析生成*