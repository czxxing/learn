# DSPy Teleprompter 模块详解

> 本文档详细分析 DSPy 优化器（Teleprompter）的实现原理和使用方法。

## 目录

- [一、概述](#一概述)
- [二、架构设计](#二架构设计)
- [三、BootstrapFewShot](#三bootstrapfewshot)
- [四、MIPROv2](#四miprov2)
- [五、GEPA](#五gepa)
- [六、使用示例](#六使用示例)

---

## 一、概述

Teleprompter 是 DSPy 的自动优化模块，负责优化程序的提示词和参数。

### 1.1 优化流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      Teleprompter 优化流程                        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   trainset   │ -> │   student    │ -> │   valset     │     │
│  │  (训练集)     │    │   (学生程序)  │    │  (验证集)    │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                   │                   │               │
│         v                   v                   v               │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                     优化循环                           │    │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────┐  │    │
│  │  │ 生成提示 │ -> │ 运行程序 │ -> │ 评估结果 │ -> │ 更新 │  │    │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────┘  │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                 │
│                              v                                 │
│                      compiled_student                          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 可用优化器

| 优化器 | 描述 | 适用场景 |
|--------|------|----------|
| `BootstrapFewShot` | Few-Shot 引导 | 小数据集 |
| `BootstrapFewShotWithRandomSearch` | 随机搜索 | 快速迭代 |
| `MIPROv2` | 多保真度提示优化 | 中等数据集 |
| `GEPA` | 梯度提示优化 | 大数据集 |
| `BootstrapFinetune` | 微调优化 | 需要微调 |
| `Ensemble` | 集成优化 | 提升稳定性 |

---

## 二、架构设计

### 2.1 Teleprompter 基类

```python
class Teleprompter:
    def __init__(self):
        pass

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        valset: list[Example] | None = None,
        **kwargs
    ) -> Module:
        """
        优化学生程序。

        Args:
            student: 要优化的学生程序
            trainset: 训练集
            teacher: 教师程序（可选）
            valset: 验证集

        Returns:
            优化后的程序
        """
        raise NotImplementedError
```

### 2.2 优化器接口

```python
class Optimizer(ABC):
    """优化器抽象基类"""

    def compile(self, student, trainset, valset, metric, **config):
        """执行优化"""
        ...

    def get_params(self):
        """获取优化器参数"""
        return self.__dict__
```

---

## 三、BootstrapFewShot

### 3.1 原理

BootstrapFewShot 通过教师模型生成高质量的 Few-Shot 示例来引导学生模型。

```
┌─────────────────────────────────────────────────────────────┐
│                  BootstrapFewShot 流程                      │
│                                                             │
│  trainset                                                  │
│     │                                                      │
│     ├──> 教师模型运行 -> 收集 traces                        │
│     │           │                                          │
│     │           v                                          │
│     │     高质量 traces -> demos                            │
│     │                                                      │
│     └──> 学生程序 + demos -> 预测                           │
│                                                             │
│  valset                                                    │
│     │                                                      │
│     └──> 评估预测                                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 源码实现

```python
class BootstrapFewShot(Teleprompter):
    def __init__(
        self,
        metric=None,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        max_rounds=1,
        **kwargs
    ):
        self.metric = metric
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds

    def compile(self, student, trainset, teacher=None, valset=None, **kwargs):
        # 1. 如果没有教师，使用学生自己生成 demos
        if teacher is None:
            teacher = student

        # 2. 运行教师模型生成 traces
        traces = self._bootstrap_traces(teacher, trainset)

        # 3. 选择高质量 demos
        demos = self._select_demos(traces, self.max_bootstrapped_demos)

        # 4. 为学生设置 demos
        for predictor in student.predictors():
            predictor.demos = demos

        return student
```

---

## 四、MIPROv2

### 4.1 原理

MIPROv2 (Multi-Instruction Program Optimization) 使用贝叶斯优化在多个保真度上优化提示。

```
┌─────────────────────────────────────────────────────────────┐
│                      MIPROv2 流程                           │
│                                                             │
│  ┌─────────────┐                                            │
│  │  指令候选   │                                            │
│  │ (demos候选) │                                            │
│  └─────────────┘                                            │
│        │                                                    │
│        v                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              贝叶斯优化循环                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │
│  │  │ 采样指令 │  │ 运行程序  │  │ 评估结果 │        │   │
│  │  └──────────┘  └──────────┘  └──────────┘        │   │
│  │       ^                                            │   │
│  │       └────────────────────────────────────────────┘   │
│  └─────────────────────────────────────────────────────┘   │
│        │                                                    │
│        v                                                    │
│  ┌─────────────┐                                            │
│  │  最佳指令   │                                            │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 多保真度评估

```python
class MIPROv2(Teleprompter):
    def __init__(
        self,
        metric=None,
        num_trials=20,
        max_bootstrapped_demos=4,
        min_demos=1,
        max_demos=8,
        **kwargs
    ):
        self.metric = metric
        self.num_trials = num_trials
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.min_demos = min_demos
        self.max_demos = max_demos

    def compile(self, student, trainset, valset, **kwargs):
        # 1. 生成指令候选
        instruction_candidates = generate_instruction_candidates(trainset)

        # 2. 贝叶斯优化
        best_score = -float('inf')
        for trial in range(self.num_trials):
            # 采样指令和 demo 数量
            instruction = sample_instruction(instruction_candidates)
            num_demos = sample_num_demos(self.min_demos, self.max_demos)

            # 设置学生
            student = self._apply_candidate(student, instruction, num_demos)

            # 评估（多保真度）
            score = self._evaluate_multi_fidelity(student, valset)

            # 更新最佳
            if score > best_score:
                best_score = score
                best_student = student

        return best_student
```

---

## 五、GEPA

### 5.1 原理

GEPA (Gradient-Based Prompt Optimizer) 使用梯度下降来优化提示参数。

### 5.2 特点

| 特点 | 描述 |
|------|------|
| **基于梯度** | 使用梯度信息指导优化 |
| **大规模数据** | 适合大数据集 |
| **参数高效** | 优化提示参数而非权重 |

---

## 六、使用示例

### 6.1 BootstrapFewShot

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 定义学生程序
student = dspy.ChainOfThought("question -> answer")

# 定义评估指标
def metric(example, pred, trace=None):
    return pred.answer.lower() == example.answer.lower()

# 创建优化器
optimizer = dspy.BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=8,
)

# 编译优化
optimized = optimizer.compile(
    student,
    trainset=train_examples,
    valset=val_examples,
)
```

### 6.2 BootstrapFewShotWithRandomSearch

```python
optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=8,
    num_searches=10,  # 尝试 10 种配置
    temp=[0.3, 0.5, 0.7, 0.9],  # 温度搜索
)

optimized = optimizer.compile(student, trainset=train_examples)
```

### 6.3 MIPROv2

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=metric,
    num_trials=20,
    max_bootstrapped_demos=4,
)

optimized = optimizer.compile(
    student,
    trainset=train_examples,
    valset=val_examples,
)
```

### 6.4 使用教师模型

```python
# 定义教师程序（通常更强大）
teacher = dspy.ChainOfThought("question -> answer")

# 使用教师优化学生
optimizer = dspy.BootstrapFewShot(metric=metric)
optimized = optimizer.compile(
    student,
    teacher=teacher,  # 提供教师模型
    trainset=train_examples,
)
```

### 6.5 优化后使用

```python
# 优化后的程序使用方式与普通程序相同
result = optimized(question="What is the meaning of life?")
print(result.answer)

# 保存优化后的程序
import joblib
joblib.dump(optimized, "optimized_program.pkl")

# 加载
loaded = joblib.load("optimized_program.pkl")
```

---

## 七、源码文件

| 文件 | 路径 |
|------|------|
| Teleprompter 基类 | [teleprompt/teleprompt.py](file:///home/project/dspy/dspy/teleprompt/teleprompt.py) |
| BootstrapFewShot | [teleprompt/bootstrap.py](file:///home/project/dspy/dspy/teleprompt/bootstrap.py) |
| MIPROv2 | [teleprompt/mipro_optimizer_v2.py](file:///home/project/dspy/dspy/teleprompt/mipro_optimizer_v2.py) |
| GEPA | [teleprompt/gepa/gepa.py](file:///home/project/dspy/dspy/teleprompt/gepa/gepa.py) |

---

*本文档基于 DSPy v2.x 源码分析生成*