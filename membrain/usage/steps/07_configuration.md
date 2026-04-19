# Step 7: 配置参数

## 概述

本步骤详细介绍 MemBrain 的各种配置参数，包括搜索参数、Token 预算、实体解析参数等，帮助用户根据实际需求进行调整。

## 7.1 搜索参数

### 基本搜索配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QA_BM25_FACT_TOP_N` | 20 | BM25 路径返回的事实数量 |
| `QA_EMBED_FACT_TOP_N` | 20 | 向量路径返回的事实数量 |
| `QA_ENTITY_TOP_N` | 5 | 实体树返回的实体数量 |
| `QA_TREE_BEAM_WIDTH` | 3 | 实体树光束宽度 |
| `QA_RERANK_TOP_K` | 12 | 融合后保留的结果数量 |

### 配置示例

```python
# 调整搜索参数
config = {
    "QA_BM25_FACT_TOP_N": 30,      # 增加 BM25 返回数量
    "QA_EMBED_FACT_TOP_N": 30,     # 增加向量返回数量
    "QA_ENTITY_TOP_N": 10,         # 增加实体树返回数量
    "QA_TREE_BEAM_WIDTH": 5,       # 增加光束宽度
    "QA_RERANK_TOP_K": 20,         # 增加融合后保留数量
}
```

### 参数影响

```
参数值对搜索结果的影响:

QA_BM25_FACT_TOP_N
    │
    ├── 值越大 → 初始结果越多 → 可能召回更多相关内容
    │
    └── 值越小 → 初始结果越少 → 速度更快但可能遗漏

QA_EMBED_FACT_TOP_N
    │
    ├── 值越大 → 向量检索结果越多 → 语义相似内容更多
    │
    └── 值越小 → 向量检索结果越少 → 速度更快

QA_ENTITY_TOP_N
    │
    ├── 值越大 → 实体树展开越广 → 相关信息更多
    │
    └── 值越小 → 实体树展开越窄 → 速度更快

QA_RERANK_TOP_K
    │
    ├── 值越大 → 最终结果越多 → 上下文更丰富
    │
    └── 值越小 → 最终结果越少 → 上下文更精简
```

## 7.2 Token 预算

### 上下文预算配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `_FACT_BUDGET_TOKENS` | 4500 | 事实上下文预算 (tokens) |
| `_SESSION_BUDGET_TOKENS` | 1500 | 会话摘要上下文预算 (tokens) |
| `_TOTAL_BUDGET_TOKENS` | 6000 | 总上下文预算 (tokens) |

### Token 预算分配

```
总预算: 6000 tokens
    │
    ├── 事实上下文: 4500 tokens (75%)
    │   │
    │   └── 用于存储从记忆中检索的事实
    │
    └── 会话摘要: 1500 tokens (25%)
        │
        └── 用于存储相关的会话摘要
```

### 调整示例

```python
# 增大上下文预算
config = {
    "_FACT_BUDGET_TOKENS": 5000,      # 事实上下文增加到 5000
    "_SESSION_BUDGET_TOKENS": 2000,   # 会话摘要增加到 2000
    "_TOTAL_BUDGET_TOKENS": 7000,     # 总预算增加到 7000
}

# 减小上下文预算 (用于更快的响应)
config = {
    "_FACT_BUDGET_TOKENS": 3000,      # 事实上下文减少到 3000
    "_SESSION_BUDGET_TOKENS": 1000,   # 会话摘要减少到 1000
    "_TOTAL_BUDGET_TOKENS": 4000,     # 总预算减少到 4000
}
```

### 根据模型选择

```python
# GPT-4 (8K 上下文)
config = {
    "_TOTAL_BUDGET_TOKENS": 6000,  # 留出空间给系统消息
}

# GPT-3.5 Turbo (16K 上下文)
config = {
    "_TOTAL_BUDGET_TOKENS": 12000,  # 可以使用更多上下文
}

# GPT-3.5 Turbo (4K 上下文)
config = {
    "_TOTAL_BUDGET_TOKENS": 3000,  # 减少以适应小上下文
}
```

## 7.3 实体解析参数

### 实体消重配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RESOLVER_JACCARD_THRESHOLD` | 0.9 | Jaccard 相似度阈值 |
| `RESOLVER_ENTROPY_THRESHOLD` | 1.5 | 熵阈值 |
| `RESOLVER_MINHASH_PERMUTATIONS` | 32 | MinHash 置换次数 |
| `RESOLVER_LLM_ENABLED` | True | 是否启用 LLM 语义匹配 |

### 参数详解

```python
# Jaccard 阈值
# 值越接近 1.0，匹配越严格
# 值越小，匹配越宽松，但可能产生误匹配

RESOLVER_JACCARD_THRESHOLD = 0.9  # 严格匹配 (默认)
RESOLVER_JACCARD_THRESHOLD = 0.8  # 宽松匹配

# MinHash 置换次数
# 值越大，结果越准确，但计算越慢
# 值越小，计算越快，但结果可能不够准确

RESOLVER_MINHASH_PERMUTATIONS = 32   # 默认
RESOLVER_MINHASH_PERMUTATIONS = 64   # 更准确
RESOLVER_MINHASH_PERMUTATIONS = 16   # 更快

# LLM 语义匹配
# 启用后，使用 LLM 进行语义级别的实体匹配
# 精度更高，但成本更高

RESOLVER_LLM_ENABLED = True   # 启用 (默认)
RESOLVER_LLM_ENABLED = False # 禁用 (更快)
```

### 三层实体消重策略

```
实体消重流程:

Layer 1: 精确匹配
    │
    ├── 方法: 字符串完全匹配 (忽略大小写)
    ├── 阈值: 100% 匹配
    └── 用途: 消除重复的实体名称

    ▼

Layer 2: MinHash + Jaccard
    │
    ├── 方法: 局部敏感哈希 + Jaccard 相似度
    ├── 阈值: RESOLVER_JACCARD_THRESHOLD (默认 0.9)
    └── 用途: 消除拼写变体、近似名称

    ▼

Layer 3: LLM 语义匹配
    │
    ├── 方法: 使用 LLM 判断语义等价
    ├── 阈值: 语义相似
    └── 用途: 消除同义词、别名

    控制: RESOLVER_LLM_ENABLED
```

## 7.4 提取参数

### 实体提取配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EXTRACTION_MAX_RETRIES` | 3 | 提取最大重试次数 |
| `EXTRACTION_TIMEOUT` | 60 | 提取超时时间 (秒) |

### 事实生成配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FACT_GEN_MAX_RETRIES` | 3 | 事实生成最大重试次数 |
| `FACT_GEN_TIMEOUT` | 120 | 事实生成超时时间 (秒) |

## 7.5 数据库配置

### 连接池配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DB_POOL_SIZE` | 10 | 连接池大小 |
| `DB_MAX_OVERFLOW` | 20 | 最大溢出连接数 |
| `DB_POOL_TIMEOUT` | 30 | 连接超时时间 (秒) |

### 配置示例

```python
# 数据库连接配置
db_config = {
    "DB_POOL_SIZE": 10,         # 10 个连接
    "DB_MAX_OVERFLOW": 20,      # 最多 20 个溢出
    "DB_POOL_TIMEOUT": 30,      # 30 秒超时
}
```

## 7.6 LLM 配置

### 模型选择

```python
# 可用的 LLM 提供商

# OpenAI
llm_config = {
    "provider": "openai",
    "model": "gpt-4",           # 或 "gpt-3.5-turbo"
    "temperature": 0.0,         # 温度参数
}

# Anthropic
llm_config = {
    "provider": "anthropic",
    "model": "claude-3-opus",   # 或 "claude-3-sonnet"
    "temperature": 0.0,
}

# Azure OpenAI
llm_config = {
    "provider": "azure",
    "model": "gpt-4",
    "deployment_name": "gpt-4-deployment",
}
```

### 嵌入模型配置

```python
# 嵌入模型配置
embed_config = {
    "provider": "openai",
    "model": "text-embedding-ada-002",  # 或其他嵌入模型
    "dimensions": 1536,                  # 向量维度
}
```

## 7.7 日志配置

### 日志级别

```python
import logging

# 设置日志级别
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG / INFO / WARNING / ERROR
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 组件日志级别
logging.getLogger("membrain.search").setLevel(logging.INFO)
logging.getLogger("membrain.ingest").setLevel(logging.DEBUG)
logging.getLogger("membrain.entity").setLevel(logging.WARNING)
```

### 日志输出

```python
# 输出到文件
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("membrain.log"),
        logging.StreamHandler()
    ]
)
```

## 7.8 环境变量配置

### 常用环境变量

```bash
# 数据库配置
export DATABASE_URL="postgresql://user:password@localhost:5432/membrain"

# LLM 配置
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# 服务器配置
export MEMBRAIN_HOST="0.0.0.0"
export MEMBRAIN_PORT="9574"

# 日志配置
export LOG_LEVEL="INFO"
```

### 配置加载

```python
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 读取配置
database_url = os.getenv("DATABASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")
membrain_port = os.getenv("MEMBRAIN_PORT", "9574")
```

## 7.9 生产环境配置建议

### 开发环境

```python
# 开发环境配置
config = {
    "DEBUG": True,
    "LOG_LEVEL": "DEBUG",
    "DB_POOL_SIZE": 5,
    "QA_BM25_FACT_TOP_N": 10,
    "QA_EMBED_FACT_TOP_N": 10,
    "RESOLVER_LLM_ENABLED": False,  # 开发时禁用 LLM
}
```

### 测试环境

```python
# 测试环境配置
config = {
    "DEBUG": False,
    "LOG_LEVEL": "INFO",
    "DB_POOL_SIZE": 10,
    "QA_BM25_FACT_TOP_N": 20,
    "QA_EMBED_FACT_TOP_N": 20,
    "RESOLVER_LLM_ENABLED": True,
}
```

### 生产环境

```python
# 生产环境配置
config = {
    "DEBUG": False,
    "LOG_LEVEL": "WARNING",
    "DB_POOL_SIZE": 20,
    "DB_MAX_OVERFLOW": 40,
    "QA_BM25_FACT_TOP_N": 30,
    "QA_EMBED_FACT_TOP_N": 30,
    "QA_RERANK_TOP_K": 15,
    "RESOLVER_LLM_ENABLED": True,
    "_TOTAL_BUDGET_TOKENS": 6000,
}
```

## 7.10 配置优先级

```
配置优先级 (从高到低):

1. 环境变量
   │
   └── 最高优先级，直接覆盖所有配置
   
2. 配置文件
   │
   └── config.yaml / config.json
   
3. 代码默认配置
   │
   └── config/default.py
   
4. 硬编码默认值
   │
   └── 最低优先级
```

## 总结

本步骤介绍了：
- 搜索参数的配置和影响
- Token 预算的分配和调整
- 实体解析参数的配置
- 提取参数的配置
- 数据库和 LLM 的配置
- 日志配置
- 环境变量配置
- 不同环境的配置建议
- 配置优先级

通过合理配置这些参数，可以根据实际需求优化 MemBrain 的性能和功能。