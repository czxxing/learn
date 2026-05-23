# DSPy Module 模块详解

> 本文档详细分析 `dspy.Module` 的实现原理和使用方法。

## 目录

- [一、概述](#一概述)
- [二、核心实现](#二核心实现)
- [三、调用流程](#三调用流程)
- [四、核心方法详解](#四核心方法详解)
- [五、状态管理](#五状态管理)
- [六、并行处理](#六并行处理)
- [七、使用示例](#七使用示例)

---

## 一、概述

`dspy.Module` 是所有 DSPy 模块的基类，提供了语言模型调用的统一入口和基础架构支持。

### 1.1 类层次结构

```
BaseModule (primitives/base_module.py)
    │
    └── Module (primitives/module.py)  ← 基类
            │
            ├── Predict (predict/predict.py)
            ├── ChainOfThought (predict/chain_of_thought.py)
            ├── ReAct (predict/react.py)
            ├── ProgramOfThought (predict/program_of_thought.py)
            ├── Parallel (predict/parallel.py)
            ├── Retrieve (retrievers/retrieve.py)
            └── 用户自定义模块
```

### 1.2 核心职责

| 职责 | 描述 |
|------|------|
| **生命周期管理** | 通过元类确保模块正确初始化 |
| **调用分发** | `__call__` 方法统一处理所有调用 |
| **回调机制** | 支持 `BaseCallback` 钩子 |
| **状态追踪** | 维护 `history` 记录 LM 调用 |
| **使用量统计** | 跟踪 API 使用量 |
| **序列化** | 支持 pickle 序列化/反序列化 |

---

## 二、核心实现

### 2.1 元类：ProgramMeta

```python
class ProgramMeta(type):
    """Metaclass ensuring every ``dspy.Module`` instance is properly initialised."""

    def __call__(cls, *args, **kwargs):
        # 1. 创建实例但不调用 __init__
        obj = cls.__new__(cls, *args, **kwargs)
        
        if isinstance(obj, cls):
            # 2. 先执行基础初始化
            Module._base_init(obj)
            # 3. 再调用用户的 __init__
            cls.__init__(obj, *args, **kwargs)
            
            # 4. 确保关键属性存在
            if not hasattr(obj, "callbacks"):
                obj.callbacks = []
            if not hasattr(obj, "history"):
                obj.history = []
        return obj
```

**设计要点**：
- 即使子类忘记调用 `super().__init__()`，也能保证基础属性存在
- 统一了所有模块的初始化流程

### 2.2 基础初始化

```python
class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False  # 是否经过编译优化
        self.callbacks = []      # 回调处理器列表
        self.history = []        # LM 调用历史

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
        self._compiled = False
        self.history = []
```

---

## 三、调用流程

### 3.1 __call__ 方法

```python
@with_callbacks
def __call__(self, *args, **kwargs) -> Prediction:
    from dspy.dsp.utils.settings import thread_local_overrides

    # 1. 维护调用者模块栈
    caller_modules = settings.caller_modules or []
    caller_modules = list(caller_modules)
    caller_modules.append(self)

    # 2. 设置线程上下文
    with settings.context(caller_modules=caller_modules):
        # 3. 追踪使用量（如果启用）
        if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
            with track_usage() as usage_tracker:
                output = self.forward(*args, **kwargs)
            tokens = usage_tracker.get_total_tokens()
            self._set_lm_usage(tokens, output)
            return output

        # 4. 执行 forward
        return self.forward(*args, **kwargs)
```

### 3.2 调用流程图

```
用户调用: program(question="...")
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. @with_callbacks 装饰器                                   │
│    - 触发 on_module_start 回调                               │
│    - 包装整个调用过程                                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. settings.context(caller_modules=[...])                    │
│    - 设置线程本地配置                                        │
│    - 维护模块调用栈                                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. track_usage() (可选)                                     │
│    - 统计 API 使用量                                        │
│    - 附加到 Prediction 对象                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. self.forward(*args, **kwargs)                            │
│    - 执行用户定义的业务逻辑                                  │
│    - 返回 Prediction 对象                                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. on_module_end 回调触发                                   │
│    - 返回结果或异常信息                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、核心方法详解

### 4.1 named_predictors()

遍历模块中的所有 Predict 实例：

```python
def named_predictors(self):
    """Return all named Predict modules in this module."""
    from dspy.predict.predict import Predict
    
    return [
        (name, param) 
        for name, param in self.named_parameters() 
        if isinstance(param, Predict)
    ]
```

**使用场景**：
- 设置统一的 LM
- 收集所有 Predict 的状态
- 优化器遍历

### 4.2 set_lm() / get_lm()

```python
def set_lm(self, lm):
    """Set the language model for all predictors in this module."""
    for _, param in self.named_predictors():
        param.lm = lm

def get_lm(self):
    """Get the language model used by this module's predictors."""
    all_used_lms = [param.lm for _, param in self.named_predictors()]
    
    if len(set(all_used_lms)) == 1:
        return all_used_lms[0]
    
    raise ValueError("Multiple LMs are being used...")
```

### 4.3 map_named_predictors()

对所有 Predict 应用转换函数：

```python
def map_named_predictors(self, func):
    """Apply a function to all named predictors."""
    for name, predictor in self.named_predictors():
        set_attribute_by_name(self, name, func(predictor))
    return self

# 使用示例：将所有 Predict 的 temperature 设置为 0.7
program.map_named_predictors(lambda p: p.update_config(temperature=0.7))
```

### 4.4 inspect_history()

查看 LM 调用历史：

```python
def inspect_history(self, n: int = 1, file=None):
    """Display the LM call history for this module."""
    pretty_print_history(self.history, n, file=file)
```

---

## 五、状态管理

### 5.1 序列化支持

```python
def __getstate__(self):
    """排除运行时状态以便序列化"""
    state = self.__dict__.copy()
    state.pop("history", None)   # 不保存调用历史
    state.pop("callbacks", None) # 不保存回调
    return state

def __setstate__(self, state):
    """恢复状态时重新初始化"""
    self.__dict__.update(state)
    if not hasattr(self, "history"):
        self.history = []
    if not hasattr(self, "callbacks"):
        self.callbacks = []
```

### 5.2 使用量追踪

```python
def _set_lm_usage(self, tokens: dict, output: Any):
    """将使用量附加到 Prediction 对象"""
    prediction_in_output = None
    
    if isinstance(output, Prediction):
        prediction_in_output = output
    elif isinstance(output, tuple) and len(output) > 0:
        if isinstance(output[0], Prediction):
            prediction_in_output = output[0]
    
    if prediction_in_output:
        prediction_in_output.set_lm_usage(tokens)
```

---

## 六、并行处理

### 6.1 batch() 方法

```python
def batch(
    self,
    examples: list[Example],
    num_threads: int | None = None,
    max_errors: int | None = None,
    return_failed_examples: bool = False,
    timeout: int = 120,
) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]:
    """处理多个样本的并行执行"""
    exec_pairs = [(self, example.inputs()) for example in examples]
    
    parallel_executor = Parallel(
        num_threads=num_threads,
        max_errors=max_errors,
        return_failed_examples=return_failed_examples,
        timeout=timeout,
    )
    
    return parallel_executor.forward(exec_pairs)
```

### 6.2 使用示例

```python
# 并行处理多个样本
results = program.batch(examples)
for result in results:
    print(result.answer)

# 获取失败样本
results, failed_examples, exceptions = program.batch(
    examples, 
    return_failed_examples=True
)
```

---

## 七、使用示例

### 7.1 简单模块

```python
import dspy

class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")
    
    def forward(self, question):
        return self.predict(question=question)

# 使用
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
qa = SimpleQA()
result = qa(question="What is 2+2?")
print(result.answer)
```

### 7.2 多阶段流水线

```python
class MultiStagePipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.analyze = dspy.ChainOfThought("context, question -> analysis")
        self.answer = dspy.Predict("analysis -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        analysis = self.analyze(context=context, question=question)
        return self.answer(analysis=analysis.analysis)

pipeline = MultiStagePipeline()
result = pipeline(question="...")
```

### 7.3 带回调的模块

```python
class LoggingCallback(dspy.utils.callback.BaseCallback):
    def on_module_start(self, call_id, instance, inputs):
        print(f"模块开始: {instance.__class__.__name__}")
    
    def on_module_end(self, call_id, outputs, exception):
        if exception:
            print(f"错误: {exception}")
        else:
            print(f"完成: {outputs}")

qa = SimpleQA(callbacks=[LoggingCallback()])
result = qa(question="...")
```

---

## 八、源码文件

| 文件 | 路径 |
|------|------|
| Module 类 | [primitives/module.py](file:///home/project/dspy/dspy/primitives/module.py) |
| BaseModule 基类 | [primitives/base_module.py](file:///home/project/dspy/dspy/primitives/base_module.py) |
| 回调机制 | [utils/callback.py](file:///home/project/dspy/dspy/utils/callback.py) |
| 使用量追踪 | [utils/usage_tracker.py](file:///home/project/dspy/dspy/utils/usage_tracker.py) |

---

*本文档基于 DSPy v2.x 源码分析生成*