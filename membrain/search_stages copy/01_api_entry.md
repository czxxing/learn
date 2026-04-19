# Stage 1: API 入口 - 搜索请求的接收与初始化

## 概述

本阶段是 MemBrain 搜索过程的入口点，负责接收外部 API 请求，解析参数，获取必要的客户端资源，并设置数据库 schema 上下文。

## 代码位置

- **API 路由**: [memory.py](file:///home/project/MemBrain/membrain/api/routes/memory.py#L281-L316)
- **服务管理器**: [manager.py](file:///home/project/MemBrain/membrain/api/manager.py)

## 详细代码分析

### 1.1 API 路由定义

```python
# membrain/api/routes/memory.py
@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(req: MemorySearchRequest):
```

**关键点**:
- 使用 FastAPI 的 POST 路由装饰器
- 响应模型为 `MemorySearchResponse`

### 1.2 请求参数解析

```python
async def search_memory(req: MemorySearchRequest):
    # 1. 解析任务标识
    resolved = get_task_pk(req.dataset, req.task)
    if resolved is None:
        raise HTTPException(
            404, f"Task '{req.task}' not found in dataset '{req.dataset}'"
        )
    task_pk = resolved
```

**为什么需要任务解析**:
- MemBrain 支持多任务（多用户/多会话）
- 每个任务有独立的数据库 schema
- `task_pk` 是任务的唯一标识符

**请求参数结构**（查看 MemorySearchRequest）:
```python
# 假设的请求结构
class MemorySearchRequest:
    dataset: str      # 数据集名称，如 "personamem_v2"
    task: str         # 任务标识，如 "user_123"
    question: str     # 用户问题
    top_k: int | None = None      # 返回结果数量
    strategy: Literal["rrf", "rerank"] = "rrf"  # 融合策略
    mode: Literal["direct", "expand", "reflect"] = "expand"  # 检索模式
```

### 1.3 获取服务客户端

```python
# 获取数据库会话工厂
sf = search_mgr.get_session_factory()

# 获取嵌入客户端（用于向量搜索）
embed_client = search_mgr.get_embed_client()

# 获取 HTTP 客户端（用于 LLM 调用）
http_client = search_mgr.get_http_client()

# 获取 top_k 参数
top_k = req.top_k or settings.QA_RERANK_TOP_K  # 默认 12
```

**为什么需要三个客户端**:

| 客户端 | 用途 | 关键操作 |
|--------|------|----------|
| `session_factory` | 数据库操作 | 执行 SQL 查询 |
| `embed_client` | 向量化 | 将文本转换为向量 |
| `http_client` | LLM 调用 | 查询重写、多查询生成 |

### 1.4 SearchServiceManager 实现

```python
# membrain/api/manager.py

class SearchServiceManager:
    """Shared engine + global clients for read-only memory search."""

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._sf: sessionmaker | None = None
        self._embed_client: EmbeddingClient | None = None
        self._http_client: httpx.Client | None = None

    def _ensure_engine(self) -> None:
        if self._engine is None:
            self._engine = sa_create_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                pool_size=settings.QA_SEARCH_POOL_SIZE,  # 默认 20
                max_overflow=10,
                pool_timeout=settings.DB_POOL_TIMEOUT,   # 默认 60s
                pool_recycle=settings.DB_POOL_RECYCLE,   # 默认 3600s
            )
            self._sf = sessionmaker(bind=self._engine)
```

**连接池配置说明**:
- `pool_pre_ping=True`: 每次使用前检测连接是否有效
- `pool_size=20`: 保持 20 个活跃连接
- `max_overflow=10`: 允许额外 10 个临时连接
- `pool_recycle=3600`: 1小时后回收连接

### 1.5 客户端获取方法

```python
def get_session_factory(self) -> sessionmaker:
    self._ensure_engine()
    return self._sf

def get_embed_client(self) -> EmbeddingClient:
    if self._embed_client is None:
        self._embed_client = EmbeddingClient()
    return self._embed_client

def get_http_client(self) -> httpx.Client:
    if self._http_client is None:
        self._http_client = httpx.Client(timeout=60.0)
    return self._http_client
```

**单例模式**: 客户端采用延迟初始化和单例模式，避免重复创建。

### 1.6 设置数据库 Schema

```python
# 设置搜索路径到特定任务的 schema
schema = f"task_{int(task_pk)}__{_RUN_TAG}"
with sf() as db:
    db.execute(sa_text(f"SET LOCAL search_path TO {schema}, public"))
```

**为什么需要 Schema 隔离**:
- 每个任务有独立的 schema: `task_1__default`, `task_2__default`
- 数据完全隔离，避免不同任务的数据混淆
- `search_path` 允许同时访问任务 schema 和 public schema

**Schema 命名规则**:
```
task_{task_pk}__{run_tag}
例如: task_1__default
```

### 1.7 调用核心搜索函数

```python
# 在设置的 schema 上下文中执行搜索
result = _retrieval.search(
    question=req.question,
    task_id=task_pk,
    db=db,
    embed_client=embed_client,
    http_client=http_client,
    top_k=top_k,
    strategy=req.strategy,
    mode=req.mode,
)
```

### 1.8 响应构建

```python
return MemorySearchResponse(
    packed_context=result["packed_context"],
    packed_token_count=result["packed_token_count"],
    fact_ids=result["fact_ids"],
    facts=[RetrievedFactOut(**f) for f in result["facts"]],
    sessions=[RetrievedSessionOut(**s) for s in result["sessions"]],
    raw_messages=[],
)
```

## 完整流程示例

### 示例请求

```json
{
    "dataset": "personamem_v2",
    "task": "user_001",
    "question": "When did Caroline have a picnic?",
    "top_k": 15,
    "strategy": "rrf",
    "mode": "expand"
}
```

### 执行流程

```
1. HTTP POST /api/memory/search
       │
       ▼
2. 解析请求参数
   dataset = "personamem_v2"
   task = "user_001"
   question = "When did Caroline have a picnic?"
   top_k = 15
   strategy = "rrf"
   mode = "expand"
       │
       ▼
3. 获取 task_pk = 1 (假设)
       │
       ▼
4. 获取/创建客户端
   - session_factory (数据库)
   - embed_client (嵌入服务)
   - http_client (LLM API)
       │
       ▼
5. 创建数据库会话
   设置 schema = "task_1__default"
       │
       ▼
6. 调用 _retrieval.search(...)
       │
       ▼
7. 返回 MemorySearchResponse
```

## 关键配置参数

```python
# membrain/config.py

# 搜索服务连接池
QA_SEARCH_POOL_SIZE = 20        # 连接池大小
QA_RERANK_TOP_K = 12           # 默认返回结果数

# 数据库连接
DB_POOL_SIZE = 5               # 默认池大小
DB_POOL_TIMEOUT = 60           # 超时时间(秒)
DB_POOL_RECYCLE = 3600        # 回收时间(秒)
```

## 错误处理

```python
# 1. 任务不存在
if resolved is None:
    raise HTTPException(404, f"Task '{req.task}' not found in dataset '{req.dataset}'")

# 2. 客户端创建失败
# 由调用方（_retrieval.search）捕获和处理

# 3. 数据库连接失败
# 由 SQLAlchemy 异常处理机制处理
```

## 总结

本阶段的核心职责:

| 职责 | 说明 |
|------|------|
| **参数解析** | 解析请求中的 dataset、task、question 等参数 |
| **任务解析** | 将 dataset + task 转换为 task_pk |
| **资源获取** | 获取/创建数据库会话、嵌入客户端、HTTP 客户端 |
| **上下文设置** | 设置数据库 schema 隔离 |
| **结果返回** | 将搜索结果转换为 API 响应格式 |

本阶段为后续的查询扩展和检索阶段做好了准备，是整个搜索流程的"守门人"。