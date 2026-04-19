# 文档索引

本文档是 MemBrain 项目所有技术文档的索引。

## 文档目录

### 搜索过程文档 (11 个阶段)
位置: `docs/search_stages/`

| 文档 | 说明 |
|------|------|
| [README.md](search_stages/README.md) | 搜索过程汇总 |
| [01_api_entry.md](search_stages/01_api_entry.md) | Stage 1: API 入口 |
| [02_query_expansion.md](search_stages/02_query_expansion.md) | Stage 2: 查询扩展 |
| [03_multi_path_retrieval.md](search_stages/03_multi_path_retrieval.md) | Stage 3: 多路径检索 |
| [04_result_merging.md](search_stages/04_result_merging.md) | Stage 4: 结果合并与去重 |
| [05_post_processing.md](search_stages/05_post_processing.md) | Stage 5: 后处理 |
| [06_result_fusion.md](search_stages/06_result_fusion.md) | Stage 6: 结果融合 |
| [07_agentic_round2.md](search_stages/07_agentic_round2.md) | Stage 7: Agentic Round 2 |
| [08_entity_resolution.md](search_stages/08_entity_resolution.md) | Stage 8: 实体引用解析 |
| [09_session_retrieval.md](search_stages/09_session_retrieval.md) | Stage 9: 会话检索 |
| [10_context_packing.md](search_stages/10_context_packing.md) | Stage 10: 上下文打包 |
| [11_return_results.md](search_stages/11_return_results.md) | Stage 11: 返回结果 |

### 写入过程文档 (5 个阶段)
位置: `docs/ingest_stages/`

| 文档 | 说明 |
|------|------|
| [README.md](ingest_stages/README.md) | 写入过程汇总 |
| [01_entity_extraction.md](ingest_stages/01_entity_extraction.md) | Stage 1: 实体提取 |
| [02_fact_generation.md](ingest_stages/02_fact_generation.md) | Stage 2: 事实生成 |
| [03_entity_resolution.md](ingest_stages/03_entity_resolution.md) | Stage 3: 实体消重 |
| [04_database_persistence.md](ingest_stages/04_database_persistence.md) | Stage 4: 数据库持久化 |
| [05_entity_tree_update.md](ingest_stages/05_entity_tree_update.md) | Stage 5: 实体树更新 |

### 交互过程文档 (6 个阶段)
位置: `docs/interaction/`

| 文档 | 说明 |
|------|------|
| [README.md](interaction/README.md) | 交互过程汇总 |
| [01_message_input_session_create.md](interaction/01_message_input_session_create.md) | Stage 1: 消息输入与会话创建 |
| [02_async_digest_session_summary.md](interaction/02_async_digest_session_summary.md) | Stage 2: 异步消化触发与会话摘要 |
| [03_batch_extraction_entity_fact_generation.md](interaction/03_batch_extraction_entity_fact_generation.md) | Stage 3: 批量提取与实体事实生成 |
| [04_entity_resolution.md](interaction/04_entity_resolution.md) | Stage 4: 实体消重与规范化 |
| [05_database_persistence.md](interaction/05_database_persistence.md) | Stage 5: 数据库持久化 |
| [06_entity_tree_update.md](interaction/06_entity_tree_update.md) | Stage 6: 实体树更新 |

### 使用指南
位置: `docs/usage/`

| 文档 | 说明 |
|------|------|
| [membrain_usage_guide.md](usage/membrain_usage_guide.md) | 完整使用指南 |
| [steps/README.md](usage/steps/README.md) | 分步使用教程 |

### 分步教程 (9 个步骤)
位置: `docs/usage/steps/`

| 步骤 | 名称 | 说明 |
|:----:|------|------|
| 1 | [系统架构与数据模型](usage/steps/01_system_architecture.md) | 理解系统架构和核心数据模型 |
| 2 | [API 接口与服务器启动](usage/steps/02_api_endpoints.md) | 了解 API 端点和启动服务器 |
| 3 | [写入数据](usage/steps/03_write_data.md) | 学习如何写入数据到 MemBrain |
| 4 | [搜索数据](usage/steps/04_search_data.md) | 学习如何搜索和检索记忆 |
| 5 | [完整使用示例](usage/steps/05_complete_examples.md) | 完整的代码示例和流程 |
| 6 | [高级功能](usage/steps/06_advanced_features.md) | 多数据集、批量处理、异步等 |
| 7 | [配置参数](usage/steps/07_configuration.md) | 配置参数详解 |
| 8 | [错误处理与故障排除](usage/steps/08_error_handling.md) | 常见错误和解决方案 |
| 9 | [总结与最佳实践](usage/steps/09_summary.md) | 总结和最佳实践指南 |

---

## 文档结构

```
docs/
├── search_stages/          # 搜索过程 (11 个阶段)
│   └── README.md, 01-11_*.md
│
├── ingest_stages/          # 写入过程 (5 个阶段)
│   └── README.md, 01-05_*.md
│
├── interaction/            # 交互过程 (6 个阶段)
│   └── README.md, 01-06_*.md
│
├── usage/                 # 使用指南
│   ├── membrain_usage_guide.md
│   └── steps/             # 分步教程 (9 个步骤)
│       ├── README.md
│       ├── 01_system_architecture.md
│       ├── 02_api_endpoints.md
│       ├── 03_write_data.md
│       ├── 04_search_data.md
│       ├── 05_complete_examples.md
│       ├── 06_advanced_features.md
│       ├── 07_configuration.md
│       ├── 08_error_handling.md
│       └── 09_summary.md
│
└── README.md              # 本文档
```

---

## 快速导航

### 搜索过程 (11 阶段)
搜索过程从用户问题出发，通过多路径检索和融合，返回用于 LLM 回答的上下文。

### 写入过程 (5 阶段)
写入过程将用户对话转化为结构化记忆，包括实体提取、事实生成、实体消重等。

### 交互过程 (6 阶段)
交互过程详细描述从消息输入到数据持久化的完整流程，是写入过程的细化版本。

### 使用指南
包含完整的 API 使用示例、代码示例和配置参数说明。