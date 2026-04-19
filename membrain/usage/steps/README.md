# 使用指南 - 分步教程

本文档提供 MemBrain 的分步使用教程，从基础概念到高级功能。

## 目录

| 步骤 | 名称 | 说明 |
|:----:|------|------|
| 1 | [系统架构与数据模型](01_system_architecture.md) | 理解系统架构和核心数据模型 |
| 2 | [API 接口与服务器启动](02_api_endpoints.md) | 了解 API 端点和启动服务器 |
| 3 | [写入数据](03_write_data.md) | 学习如何写入数据到 MemBrain |
| 4 | [搜索数据](04_search_data.md) | 学习如何搜索和检索记忆 |
| 5 | [完整使用示例](05_complete_examples.md) | 完整的代码示例和流程 |
| 6 | [高级功能](06_advanced_features.md) | 多数据集、批量处理、异步等 |
| 7 | [配置参数](07_configuration.md) | 配置参数详解 |
| 8 | [错误处理与故障排除](08_error_handling.md) | 常见错误和解决方案 |
| 9 | [总结与最佳实践](09_summary.md) | 总结和最佳实践指南 |

## 快速开始

### 1. 启动服务器

```bash
cd /home/project/MemBrain
python -m uvicorn membrain.api.server:app --host 0.0.0.0 --port 9574
```

### 2. 写入数据

```python
import requests

response = requests.post(
    "http://localhost:9574/api/memory",
    json={
        "dataset": "my_app",
        "task": "user_001",
        "store": True,
        "digest": True,
        "messages": [
            {"speaker": "user", "content": "I had lunch with John yesterday."}
        ]
    }
)
print(response.json())
```

### 3. 等待消化

```python
import time
time.sleep(5)  # 等待异步消化完成
```

### 4. 搜索记忆

```python
response = requests.post(
    "http://localhost:9574/api/memory/search",
    json={
        "dataset": "my_app",
        "task": "user_001",
        "question": "What did I do yesterday?"
    }
)
result = response.json()
print(result["packed_context"])
```

## 学习路径

```
Step 1: 系统架构
    │
    ▼
Step 2: API 接口
    │
    ▼
Step 3: 写入数据 ──────▶ Step 5: 完整示例
    │                         │
    ▼                         ▼
Step 4: 搜索数据        Step 6: 高级功能
    │                         │
    ▼                         ▼
Step 7: 配置参数        Step 8: 错误处理
    │                         │
    └─────────▶ Step 9: 总结与最佳实践 ◀─────────┘
```

## 每步重点

| 步骤 | 重点 |
|------|------|
| Step 1 | 理解数据模型关系 |
| Step 2 | 掌握 API 端点 |
| Step 3 | 学会写入数据 |
| Step 4 | 掌握搜索技巧 |
| Step 5 | 实践完整流程 |
| Step 6 | 了解高级功能 |
| Step 7 | 学会配置优化 |
| Step 8 | 学会故障排除 |
| Step 9 | 遵循最佳实践 |