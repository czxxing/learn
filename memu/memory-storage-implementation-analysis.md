# MEMU 内存存储实现分析

## 1. 核心数据模型设计

### 1.1 基础数据结构

MEMU 使用 Pydantic 模型定义核心数据结构，确保类型安全和数据验证：

```python
class BaseRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: pendulum.now("UTC"))
    updated_at: datetime = Field(default_factory=lambda: pendulum.now("UTC"))
```

所有数据记录都继承自 `BaseRecord`，包含：
- 自动生成的 UUID 作为唯一标识符
- 创建和更新时间戳

### 1.2 核心内存模型

#### MemoryItem - 内存项
```python
class MemoryItem(BaseRecord):
    resource_id: str | None
    memory_type: str  # profile, event, knowledge, behavior, skill, tool
    summary: str
    embedding: list[float] | None = None
    happened_at: datetime | None = None
    extra: dict[str, Any] = {}
    # extra 字段包含：
    # - content_hash: str (去重用)
    # - reinforcement_count: int (强化次数)
    # - last_reinforced_at: str (最后强化时间)
    # - ref_id: str (引用ID)
    # - when_to_use: str (工具内存提示)
    # - metadata: dict (类型特定元数据)
    # - tool_calls: list[dict] (工具调用历史)
```

#### MemoryCategory - 内存分类
```python
class MemoryCategory(BaseRecord):
    name: str
    description: str
    embedding: list[float] | None = None
    summary: str | None = None
```

#### Resource - 资源引用
```python
class Resource(BaseRecord):
    url: str
    modality: str
    local_path: str
    caption: str | None = None
    embedding: list[float] | None = None
```

#### CategoryItem - 分类关系
```python
class CategoryItem(BaseRecord):
    item_id: str
    category_id: str
```

#### ToolCallResult - 工具调用记录
```python
class ToolCallResult(BaseModel):
    tool_name: str
    input: dict[str, Any] | str
    output: str
    success: bool = True
    time_cost: float = 0.0
    token_cost: int = -1
    score: float = 0.0
    call_hash: str = ""
    created_at: datetime = Field(default_factory=lambda: pendulum.now("UTC"))
```

## 2. 存储机制实现

### 2.1 数据库状态管理

MEMU 使用 `DatabaseState` 数据类管理内存中的所有数据：

```python
@dataclass
class DatabaseState:
    resources: dict[str, Resource] = field(default_factory=dict)
    items: dict[str, MemoryItem] = field(default_factory=dict)
    categories: dict[str, MemoryCategory] = field(default_factory=dict)
    relations: list[CategoryItem] = field(default_factory=list)
```

所有数据都存储在内存字典中，通过 ID 进行快速访问。

### 2.2 存储仓库接口

MEMU 使用 Python Protocol 定义存储仓库接口，实现了高度抽象的存储架构：

```python
@runtime_checkable
class MemoryItemRepo(Protocol):
    """Repository contract for memory items."""

    items: dict[str, MemoryItem]

    def get_item(self, item_id: str) -> MemoryItem | None: ...

    def list_items(self, where: Mapping[str, Any] | None = None) -> dict[str, MemoryItem]: ...

    def clear_items(self, where: Mapping[str, Any] | None = None) -> dict[str, MemoryItem]: ...

    def create_item(
        self,
        *,  # keyword-only parameters
        resource_id: str,
        memory_type: MemoryType,
        summary: str,
        embedding: list[float],
        user_data: dict[str, Any],
        reinforce: bool = False,
        tool_record: dict[str, Any] | None = None,
    ) -> MemoryItem: ...

    def update_item(
        self,
        *,  # keyword-only parameters
        item_id: str,
        memory_type: MemoryType | None = None,
        summary: str | None = None,
        embedding: list[float] | None = None,
        extra: dict[str, Any] | None = None,
        tool_record: dict[str, Any] | None = None,
    ) -> MemoryItem: ...

    def delete_item(self, item_id: str) -> None: ...

    def list_items_by_ref_ids(
        self, ref_ids: list[str], where: Mapping[str, Any] | None = None
    ) -> dict[str, MemoryItem]: ...

    def vector_search_items(
        self, query_vec: list[float], top_k: int, where: Mapping[str, Any] | None = None
    ) -> list[tuple[str, float]]: ...

    def load_existing(self) -> None: ...
```

**设计特点：**
- 使用 `@runtime_checkable` 实现运行时类型检查
- 强制使用 keyword-only 参数 (`*`)，提高代码可读性
- 完整的 CRUD 操作支持
- 向量搜索功能
- 引用 ID 列表查询支持
- 条件过滤查询

#### 2.2.1 内存项创建

`InMemoryMemoryItemRepository` 是该接口的具体实现：

```python
def create_item(
    self,
    *,  # keyword-only parameters
    resource_id: str,
    memory_type: MemoryType,
    summary: str,
    embedding: list[float],
    user_data: dict[str, Any],
    reinforce: bool = False,
    tool_record: dict[str, Any] | None = None,
) -> MemoryItem:
    # 处理强化逻辑
    if reinforce and memory_type != "tool":
        return self.create_item_reinforce(...)
    
    # 构建额外数据字典
    extra: dict[str, Any] = {}
    if tool_record:
        # 将工具记录字段添加到extra顶层
        for key in ("when_to_use", "metadata", "tool_calls"):
            if tool_record.get(key) is not None:
                extra[key] = tool_record[key]
    
    # 创建并存储新内存项
    mid = str(uuid.uuid4())
    it = self.memory_item_model(
        id=mid,
        resource_id=resource_id,
        memory_type=memory_type,
        summary=summary,
        embedding=embedding,
        extra=extra if extra else {},
        **user_data,
    )
    self.items[mid] = it
    return it
```

## 3. 去重与强化机制

### 3.1 内容哈希计算

```python
def compute_content_hash(summary: str, memory_type: str) -> str:
    # 标准化：小写、去除空格、压缩空白字符
    normalized = " ".join(summary.lower().split())
    content = f"{memory_type}:{normalized}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

该函数通过以下步骤生成哈希：
1. 将摘要转换为小写
2. 压缩多个空格为单个空格
3. 与内存类型组合
4. 使用 SHA-256 生成哈希并取前16个字符

### 3.2 基于哈希的去重

```python
def _find_by_hash(self, content_hash: str, user_data: dict[str, Any]) -> MemoryItem | None:
    """
    在同一用户范围内通过内容哈希查找现有项
    实现去重：如果相同内容存在于同一用户，我们强化它而不是创建重复项
    """
    for item in self.items.values():
        # 从extra字典读取content_hash
        item_hash = (item.extra or {}).get("content_hash")
        if item_hash != content_hash:
            continue
        # 检查范围匹配（user_id, agent_id等）
        if matches_where(item, user_data):
            return item
    return None
```

### 3.3 内存强化

```python
def create_item_reinforce(
    self,
    *,  # keyword-only parameters
    resource_id: str,
    memory_type: MemoryType,
    summary: str,
    embedding: list[float],
    user_data: dict[str, Any],
    reinforce: bool = False,
) -> MemoryItem:
    content_hash = compute_content_hash(summary, memory_type)
    
    # 检查是否存在相同哈希的项
    existing = self._find_by_hash(content_hash, user_data)
    if existing:
        # 强化现有内存而非创建重复项
        current_extra = existing.extra or {}
        current_count = current_extra.get("reinforcement_count", 1)
        existing.extra = {
            **current_extra,
            "reinforcement_count": current_count + 1,
            "last_reinforced_at": pendulum.now("UTC").isoformat(),
        }
        existing.updated_at = pendulum.now("UTC")
        return existing
    
    # 创建新项并添加显著性跟踪
    mid = str(uuid.uuid4())
    now = pendulum.now("UTC")
    item_extra = user_data.pop("extra", {}) if "extra" in user_data else {}
    item_extra.update({
        "content_hash": content_hash,
        "reinforcement_count": 1,
        "last_reinforced_at": now.isoformat(),
    })
    it = self.memory_item_model(
        id=mid,
        resource_id=resource_id,
        memory_type=memory_type,
        summary=summary,
        embedding=embedding,
        extra=item_extra,
        **user_data,
    )
    self.items[mid] = it
    return it
```

强化机制的关键特点：
- 相同内容只存储一次
- 每次出现时增加强化计数
- 记录最后强化时间
- 用于后续显著性排序

## 4. 向量搜索功能

### 4.1 搜索接口

```python
def vector_search_items(
    self,
    query_vec: list[float],
    top_k: int,
    where: Mapping[str, Any] | None = None,
    *,  # keyword-only parameters
    ranking: str = "similarity",
    recency_decay_days: float = 30.0,
) -> list[tuple[str, float]]:
    pool = self.list_items(where)
    
    if ranking == "salience":
        # 显著性感知排序：相似度 × 强化次数 × 时效性
        corpus = [
            (
                i.id,
                i.embedding,
                (i.extra or {}).get("reinforcement_count", 1),
                self._parse_datetime((i.extra or {}).get("last_reinforced_at")),
            )
            for i in pool.values()
        ]
        return cosine_topk_salience(query_vec, corpus, k=top_k, recency_decay_days=recency_decay_days)
    
    # 默认：纯余弦相似度（向后兼容）
    hits = cosine_topk(query_vec, [(i.id, i.embedding) for i in pool.values()], k=top_k)
    return hits
```

### 4.2 排序策略

MEMU 支持两种排序策略：

1. **纯余弦相似度**：基于向量空间中的距离直接排序

2. **显著性排序**：结合三个因素：
   - 向量相似度
   - 强化次数（`reinforcement_count`）
   - 时效性（`last_reinforced_at`），使用30天衰减

### 4.3 工具内存的特殊处理

工具内存有特殊的字段和行为：

```python
# 工具内存的extra字段可能包含：
# - when_to_use: str - 提示何时应检索此内存
# - metadata: dict - 类型特定元数据（如tool_name, avg_success_rate）
# - tool_calls: list[dict] - 工具调用历史（序列化的ToolCallResult）
```

工具调用记录有自己的哈希生成逻辑：

```python
def generate_hash(self) -> str:
    """从工具输入和输出生成MD5哈希用于去重。"""
    input_str = json.dumps(self.input, sort_keys=True) if isinstance(self.input, dict) else str(self.input)
    combined = f"{self.tool_name}|{input_str}|{self.output}"
    return hashlib.md5(combined.encode("utf-8"), usedforsecurity=False).hexdigest()
```

## 5. 架构设计分析

### 5.1 优点

1. **模块化设计**：清晰的分层结构，数据模型与存储实现分离
2. **类型安全**：使用 Pydantic 确保数据验证和类型安全
3. **灵活性**：支持多种内存类型和存储后端
4. **高效去重**：基于内容哈希的去重机制避免重复存储
5. **智能强化**：重复出现的内容会被强化而非重复存储
6. **高级搜索**：支持向量搜索和显著性排序
7. **可扩展性**：仓库模式使添加新的存储后端变得容易

### 5.2 优化空间

1. **存储效率**：
   - 当前实现将所有数据存储在内存中，不适用于大规模部署
   - 建议添加持久化存储支持（已在代码中预留接口）

2. **索引优化**：
   - 对于大规模数据集，向量搜索效率可能成为瓶颈
   - 建议集成专业向量数据库（如 Pinecone、Milvus）

3. **哈希冲突**：
   - 当前使用 SHA-256 的前16个字符，存在理论上的冲突风险
   - 可以考虑使用完整哈希或结合其他字段

4. **并发处理**：
   - 内存存储在并发环境下可能存在线程安全问题
   - 建议添加适当的锁机制

5. **数据过期**：
   - 当前没有实现内存项的自动过期机制
   - 可以添加基于时间或访问频率的过期策略

## 6. 代码优化建议

### 6.1 性能优化

```python
# 当前实现（可能存在性能问题）
def _find_by_hash(self, content_hash: str, user_data: dict[str, Any]) -> MemoryItem | None:
    for item in self.items.values():
        item_hash = (item.extra or {}).get("content_hash")
        if item_hash != content_hash:
            continue
        if matches_where(item, user_data):
            return item
    return None

# 优化建议：添加哈希索引
def __init__(self, *, state: InMemoryState, memory_item_model: type[MemoryItem]) -> None:
    self._state = state
    self.memory_item_model = memory_item_model
    self.items: dict[str, MemoryItem] = self._state.items
    self._content_hash_index: dict[str, list[str]] = defaultdict(list)  # 哈希到item_id的映射

# 更新create_item和create_item_reinforce方法以维护索引

# 优化后的查找方法
def _find_by_hash(self, content_hash: str, user_data: dict[str, Any]) -> MemoryItem | None:
    if content_hash not in self._content_hash_index:
        return None
    for item_id in self._content_hash_index[content_hash]:
        item = self.items.get(item_id)
        if item and matches_where(item, user_data):
            return item
    return None
```

### 6.2 数据持久化

建议实现基于文件或数据库的持久化存储：

```python
# 示例：添加JSON文件持久化支持
def save_to_file(self, file_path: str) -> None:
    """将所有内存项保存到JSON文件"""
    data = {
        "resources": {k: v.model_dump() for k, v in self._state.resources.items()},
        "items": {k: v.model_dump() for k, v in self._state.items.items()},
        "categories": {k: v.model_dump() for k, v in self._state.categories.items()},
        "relations": [r.model_dump() for r in self._state.relations],
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str, indent=2)

# 加载功能类似实现
```

### 6.3 并发安全

```python
# 示例：添加线程安全支持
import threading

class ThreadSafeInMemoryMemoryItemRepository(InMemoryMemoryItemRepository):
    def __init__(self, *, state: InMemoryState, memory_item_model: type[MemoryItem]) -> None:
        super().__init__(state=state, memory_item_model=memory_item_model)
        self._lock = threading.RLock()
    
    def create_item(self, **kwargs) -> MemoryItem:
        with self._lock:
            return super().create_item(**kwargs)
    
    # 为其他关键方法添加锁...
```

## 7. 总结

MEMU 的内存存储实现采用了现代软件工程的最佳实践：

- **清晰的数据模型**：使用 Pydantic 确保类型安全和数据验证
- **高效的存储机制**：基于内存的字典存储提供快速访问
- **智能的去重与强化**：避免重复存储并提高重要内容的权重
- **强大的搜索功能**：支持向量搜索和多种排序策略
- **灵活的架构设计**：仓库模式使系统易于扩展

虽然当前实现主要针对内存存储进行了优化，但通过仓库模式的设计，可以轻松扩展到支持持久化存储（如 PostgreSQL、SQLite）和专业向量数据库。

这种设计平衡了性能、灵活性和可扩展性，为 AI 系统提供了高效的内存管理解决方案。