# DSPy LM 模块详解

> 本文档详细分析 `dspy.LM` 的实现原理和使用方法。

## 目录

- [一、概述](#一概述)
- [二、核心实现](#二核心实现)
- [三、调用机制](#三调用机制)
- [四、缓存系统](#四缓存系统)
- [五、模型支持](#五模型支持)
- [六、微调与强化学习](#六微调与强化学习)
- [七、使用示例](#七使用示例)

---

## 一、概述

`dspy.LM` 是 DSPy 的语言模型统一封装层，通过集成 LiteLLM 支持 100+ 种语言模型。

### 1.1 类层次结构

```
BaseLM (clients/base_lm.py)  ← 基类
    │
    └── LM (clients/lm.py)  ← 主要实现
            │
            ├── OpenAIProvider
            ├── AnthropicProvider
            ├── VertexProvider
            └── 自定义 Provider
```

### 1.2 支持的模型类型

| 类型 | 描述 | 适用场景 |
|------|------|----------|
| `chat` | 聊天补全 | 大多数场景 |
| `text` | 文本补全 | 简单任务 |
| `responses` | OpenAI Responses API | 新版 API |

---

## 二、核心实现

### 2.1 初始化

```python
class LM(BaseLM):
    def __init__(
        self,
        model: str,                           # "provider/model-name"
        model_type: Literal["chat", "text", "responses"] = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,                   # 是否启用缓存
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,                # 重试次数
        provider: Provider | None = None,     # 自定义 Provider
        finetuning_model: str | None = None,
        **kwargs,                             # 额外参数
    ):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.provider = provider or self.infer_provider()
        self.callbacks = callbacks or []
        self.history = []                    # 调用历史
        self.num_retries = num_retries
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
```

### 2.2 模型识别

```python
@property
def _provider_name(self) -> str:
    """从模型字符串提取提供商名称"""
    if "/" in self.model:
        return self.model.split("/", 1)[0]
    return "openai"  # 默认为 OpenAI

def infer_provider(self) -> Provider:
    """根据模型名称推断 Provider"""
    if OpenAIProvider.is_provider_model(self.model):
        return OpenAIProvider()
    return Provider()  # 默认 Provider
```

### 2.3 特殊模型处理

```python
# 识别 OpenAI 推理模型 (o1, o3, gpt-5 等)
model_pattern = re.match(
    r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$",
    model_family,
)

if model_pattern:
    # 推理模型有特殊要求
    if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
        raise ValueError("推理模型需要 temperature=1.0 和 max_tokens>=16000")
    self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
```

---

## 三、调用机制

### 3.1 forward() 方法

```python
def forward(self, prompt=None, messages=None, **kwargs):
    # 1. 准备消息
    messages = messages or [{"role": "user", "content": prompt}]
    kwargs = {**self.kwargs, **kwargs}

    # 2. 根据类型选择 completion 函数
    if self.model_type == "chat":
        completion = litellm_completion
    elif self.model_type == "text":
        completion = litellm_text_completion
    elif self.model_type == "responses":
        completion = litellm_responses_completion

    # 3. 获取缓存函数
    completion, litellm_cache_args = self._get_cached_completion_fn(completion, self.cache)

    # 4. 调用 litellm
    results = completion(
        request=dict(model=self.model, messages=messages, **kwargs),
        num_retries=self.num_retries,
        cache=litellm_cache_args,
    )

    # 5. 截断检查
    self._check_truncation(results)

    # 6. 使用量追踪
    if not getattr(results, "cache_hit", False):
        settings.usage_tracker.add_usage(self.model, dict(getattr(results, "usage", {})))

    return results
```

### 3.2 异步调用 (aforward)

```python
async def aforward(self, prompt=None, messages=None, **kwargs):
    # 异步版本，使用 alitellm_completion 等
    if self.model_type == "chat":
        completion = alitellm_completion
    elif self.model_type == "text":
        completion = alitellm_text_completion
    elif self.model_type == "responses":
        completion = alitellm_responses_completion

    results = await completion(
        request=dict(model=self.model, messages=messages, **kwargs),
        num_retries=self.num_retries,
    )

    return results
```

### 3.3 调用流程图

```
LM.forward(messages, **kwargs)
    │
    ├─── 1. 合并参数: {**self.kwargs, **kwargs}
    │
    ├─── 2. 选择 completion 函数
    │         ├── "chat" → litellm_completion
    │         ├── "text" → litellm_text_completion
    │         └── "responses" → litellm_responses_completion
    │
    ├─── 3. 缓存处理
    │         └── request_cache() 包装
    │
    ├─── 4. litellm 调用
    │         └── completion(request, num_retries=3)
    │
    ├─── 5. 截断检查
    │         └── _check_truncation()
    │
    └─── 6. 使用量追踪
              └── usage_tracker.add_usage()
```

---

## 四、缓存系统

### 4.1 缓存机制

```python
def _get_cached_completion_fn(self, completion_fn, cache):
    ignored_args_for_cache_key = ["api_key", "api_base", "base_url"]

    if cache:
        # 使用 request_cache 装饰器
        completion_fn = request_cache(
            cache_arg_name="request",
            ignored_args_for_cache_key=ignored_args_for_cache_key,
        )(completion_fn)

    # litellm 缓存参数
    litellm_cache_args = {"no-cache": True, "no-store": True}

    return completion_fn, litellm_cache_args
```

### 4.2 缓存策略

| 策略 | 描述 |
|------|------|
| **DSPy 缓存** | 基于请求内容的哈希缓存 |
| **忽略参数** | `api_key`, `api_base` 等不参与缓存键 |
| **rollout_id** | 不同 ID 可绕过相同请求的缓存 |

### 4.3 rollout_id 用法

```python
# 相同输入，不同 rollout_id → 不共享缓存
result1 = lm(messages=[...], rollout_id=1)
result2 = lm(messages=[...], rollout_id=2)  # 不会命中 result1 的缓存

# 相同输入，rollout_id=None 但 temperature>0 → 不使用缓存
result = lm(messages=[...], temperature=0.7)
```

---

## 五、模型支持

### 5.1 能力检测

```python
@property
def supports_function_calling(self) -> bool:
    """检测是否支持函数调用"""
    return _get_litellm().supports_function_calling(model=self.model)

@property
def supports_reasoning(self) -> bool:
    """检测是否支持推理模式"""
    return _get_litellm().supports_reasoning(self.model)

@property
def supports_response_schema(self) -> bool:
    """检测是否支持响应 Schema"""
    return _get_litellm().supports_response_schema(
        model=self.model,
        custom_llm_provider=self._provider_name
    )

@property
def supported_params(self) -> set[str]:
    """获取支持的参数"""
    params = _get_litellm().get_supported_openai_params(
        model=self.model,
        custom_llm_provider=self._provider_name
    )
    return set(params) if params else set()
```

### 5.2 支持的提供商

```python
# OpenAI
dspy.LM("openai/gpt-4o")
dspy.LM("openai/gpt-4o-mini")
dspy.LM("openai/o1-preview")
dspy.LM("openai/o3-mini")

# Anthropic
dspy.LM("anthropic/claude-3-5-sonnet-20241022")

# Google
dspy.LM("gemini/gemini-pro")

# Azure
dspy.LM("azure/gpt-4o")

# 自定义
dspy.LM("openrouter/anthropic/claude-3.5-sonnet")
```

### 5.3 Provider 系统

```python
class Provider:
    """基础 Provider"""

class OpenAIProvider(Provider):
    """OpenAI 系列模型"""

class DatabricksProvider(Provider):
    """Databricks 模型服务"""
```

---

## 六、微调与强化学习

### 6.1 微调 (finetune)

```python
def finetune(
    self,
    train_data: list[dict[str, Any]],
    train_data_format: TrainDataFormat | None,
    train_kwargs: dict[str, Any] | None = None,
) -> TrainingJob:
    """启动微调任务"""
    if not self.provider.finetunable:
        raise ValueError(f"Provider {self.provider} does not support fine-tuning...")

    # 后台线程执行
    thread = threading.Thread(target=self._run_finetune_job)
    job = self.provider.TrainingJob(
        thread=thread,
        model=self.finetuning_model or self.model,
        train_data=train_data,
        train_data_format=train_data_format,
        train_kwargs=train_kwargs,
    )
    thread.start()
    return job
```

### 6.2 强化学习 (reinforce)

```python
def reinforce(self, train_kwargs) -> ReinforceJob:
    """启动强化学习任务 (GRPO)"""
    if not self.provider.reinforceable:
        raise ValueError(...)

    job = self.provider.ReinforceJob(lm=self, train_kwargs=train_kwargs)
    job.initialize()
    return job
```

---

## 七、使用示例

### 7.1 基本使用

```python
import dspy

# 配置全局 LM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 或创建后使用
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
```

### 7.2 带参数配置

```python
lm = dspy.LM(
    "openai/gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    cache=True,  # 默认启用
    num_retries=3,
)
```

### 7.3 自定义 Provider

```python
from dspy.clients.provider import Provider

class MyProvider(Provider):
    @property
    def finetunable(self) -> bool:
        return True

    def finetune(self, job, model, train_data, train_data_format, train_kwargs):
        # 实现微调逻辑
        ...

# 使用
lm = dspy.LM(
    "myprovider/custom-model",
    provider=MyProvider()
)
```

### 7.4 批量调用

```python
lm = dspy.LM("openai/gpt-4o-mini")

# 多次调用共享历史
response1 = lm(messages=[{"role": "user", "content": "Hello"}])
response2 = lm(messages=[{"role": "user", "content": "How are you?"}])

# 清除历史
lm.history.clear()
```

### 7.5 模型部署

```python
# 启动模型
lm = dspy.LM("openai/gpt-4o")
lm.launch(launch_kwargs={"num_tokens": 1000})

# 终止模型
lm.kill()
```

### 7.6 自定义 LM

```python
from openai import OpenAI
import dspy

class MyCustomLM(dspy.BaseLM):
    def __init__(self, model="gpt-4o-mini", **kwargs):
        super().__init__(model=model, **kwargs)
        self.client = OpenAI()

    @property
    def supports_function_calling(self):
        return "gpt-4o" in self.model

    def forward(self, prompt=None, messages=None, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages or [{"role": "user", "content": prompt}],
            **self.kwargs,
            **kwargs,
        )
        return response

# 使用
dspy.configure(lm=MyCustomLM(model="gpt-4o"))
```

---

## 八、源码文件

| 文件 | 路径 |
|------|------|
| LM 类 | [clients/lm.py](file:///home/project/dspy/dspy/clients/lm.py) |
| BaseLM 基类 | [clients/base_lm.py](file:///home/project/dspy/dspy/clients/base_lm.py) |
| Provider | [clients/provider.py](file:///home/project/dspy/dspy/clients/provider.py) |
| 缓存 | [clients/cache.py](file:///home/project/dspy/dspy/clients/cache.py) |
| LiteLLM 集成 | [clients/_litellm.py](file:///home/project/dspy/dspy/clients/_litellm.py) |

---

*本文档基于 DSPy v2.x 源码分析生成*