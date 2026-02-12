# MEMU 记忆检索流程分析

## 1. 检索系统概述

MEMU 的记忆检索系统是一个多层次、可配置的组件，支持两种主要检索策略：

- **RAG 模式**：基于嵌入向量的相似性搜索
- **LLM 模式**：利用大语言模型进行排名和筛选

检索流程设计为分层架构，从粗粒度到细粒度依次检索：
1. **分类 (Categories)**：首先检索相关分类
2. **记忆项 (Items)**：然后检索具体记忆项
3. **资源 (Resources)**：最后检索相关资源

## 2. 配置系统

### 2.1 核心配置结构

```python
class RetrieveConfig(BaseModel):
    method: Annotated[Literal["rag", "llm"], Normalize] = "rag"
    route_intention: bool = True
    category: RetrieveCategoryConfig = Field(default=RetrieveCategoryConfig())
    item: RetrieveItemConfig = Field(default=RetrieveItemConfig())
    resource: RetrieveResourceConfig = Field(default=RetrieveResourceConfig())
    sufficiency_check: bool = True
    sufficiency_check_llm_profile: str = "default"
    llm_ranking_llm_profile: str = "default"
```

### 2.2 分类检索配置

```python
class RetrieveCategoryConfig(BaseModel):
    enabled: bool = True
    top_k: int = 5  # 检索的分类数量
```

### 2.3 记忆项检索配置

```python
class RetrieveItemConfig(BaseModel):
    enabled: bool = True
    top_k: int = 5  # 检索的记忆项数量
    use_category_references: bool = False  # 是否使用分类引用
    ranking: Literal["similarity", "salience"] = "similarity"  # 排名策略
    recency_decay_days: float = 30.0  # 时效性衰减天数
```

### 2.4 资源检索配置

```python
class RetrieveResourceConfig(BaseModel):
    enabled: bool = True
    top_k: int = 5  # 检索的资源数量
```

## 3. 检索核心流程

### 3.1 入口点

检索流程从 `RetrieveMixin.retrieve()` 方法开始：

```python
async def retrieve(
    self,
    queries: list[dict[str, Any]],
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not queries:
        raise ValueError("empty_queries")
    ctx = self._get_context()
    store = self._get_database()
    original_query = self._extract_query_text(queries[-1])
    where_filters = self._normalize_where(where)

    # 确定工作流类型
    workflow_name = "retrieve_llm" if self.retrieve_config.method == "llm" else "retrieve_rag"

    # 初始化工作流状态
    state: WorkflowState = {
        "method": self.retrieve_config.method,
        "original_query": original_query,
        "context_queries": context_queries_objs,
        "route_intention": route_intention,
        "skip_rewrite": len(queries) == 1,
        "retrieve_category": retrieve_category,
        "retrieve_item": retrieve_item,
        "retrieve_resource": retrieve_resource,
        "sufficiency_check": sufficiency_check,
        "ctx": ctx,
        "store": store,
        "where": where_filters,
    }

    # 执行工作流
    result = await self._run_workflow(workflow_name, state)
    response = cast(dict[str, Any] | None, result.get("response"))
    if response is None:
        msg = "Retrieve workflow failed to produce a response"
        raise RuntimeError(msg)
    return response
```

### 3.2 工作流设计

MEMU 使用工作流引擎管理检索流程，定义了清晰的步骤依赖关系：

#### 3.2.1 RAG 工作流步骤

```python
def _build_rag_retrieve_workflow(self) -> list[WorkflowStep]:
    steps = [
        WorkflowStep(step_id="route_intention", ...),  # 意图路由
        WorkflowStep(step_id="route_category", ...),   # 分类路由
        WorkflowStep(step_id="sufficiency_after_category", ...),  # 分类充分性检查
        WorkflowStep(step_id="recall_items", ...),     # 记忆项检索
        WorkflowStep(step_id="sufficiency_after_items", ...),  # 记忆项充分性检查
        WorkflowStep(step_id="recall_resources", ...),  # 资源检索
        WorkflowStep(step_id="build_context", ...),     # 构建上下文
    ]
    return steps
```

#### 3.2.2 LLM 工作流步骤

```python
def _build_llm_retrieve_workflow(self) -> list[WorkflowStep]:
    steps = [
        WorkflowStep(step_id="route_intention", ...),  # 意图路由
        WorkflowStep(step_id="route_category", ...),   # 分类路由
        WorkflowStep(step_id="sufficiency_after_category", ...),  # 分类充分性检查
        WorkflowStep(step_id="recall_items", ...),     # 记忆项检索
        WorkflowStep(step_id="sufficiency_after_items", ...),  # 记忆项充分性检查
        WorkflowStep(step_id="recall_resources", ...),  # 资源检索
        WorkflowStep(step_id="build_context", ...),     # 构建上下文
    ]
    return steps
```

## 4. RAG 检索流程详细分析

### 4.1 意图路由

```python
async def _rag_route_intention(self, state: WorkflowState, step_context: Any) -> WorkflowState:
    if not state.get("route_intention"):
        state.update({
            "needs_retrieval": True,
            "rewritten_query": state["original_query"],
            "active_query": state["original_query"],
            "next_step_query": None,
            "proceed_to_items": False,
            "proceed_to_resources": False,
        })
        return state

    llm_client = self._get_step_llm_client(step_context)
    needs_retrieval, rewritten_query = await self._decide_if_retrieval_needed(
        state["original_query"],
        state["context_queries"],
        retrieved_content=None,
        llm_client=llm_client,
    )
    if state.get("skip_rewrite"):
        rewritten_query = state["original_query"]

    state.update({
        "needs_retrieval": needs_retrieval,
        "rewritten_query": rewritten_query,
        "active_query": rewritten_query,
        "next_step_query": None,
        "proceed_to_items": False,
        "proceed_to_resources": False,
    })
    return state
```

### 4.2 分类路由

```python
async def _rag_route_category(self, state: WorkflowState, step_context: Any) -> WorkflowState:
    if not state.get("retrieve_category") or not state.get("needs_retrieval"):
        state["category_hits"] = []
        state["category_summary_lookup"] = {}
        state["query_vector"] = None
        return state

    embed_client = self._get_step_embedding_client(step_context)
    store = state["store"]
    where_filters = state.get("where") or {}
    category_pool = store.memory_category_repo.list_categories(where_filters)
    qvec = (await embed_client.embed([state["active_query"]]))[0]
    hits, summary_lookup = await self._rank_categories_by_summary(
        qvec,
        self.retrieve_config.category.top_k,
        state["ctx"],
        store,
        embed_client=embed_client,
        categories=category_pool,
    )
    state.update({
        "query_vector": qvec,
        "category_hits": hits,
        "category_summary_lookup": summary_lookup,
        "category_pool": category_pool,
    })
    return state
```

### 4.3 分类充分性检查

```python
async def _rag_category_sufficiency(self, state: WorkflowState, step_context: Any) -> WorkflowState:
    if not state.get("needs_retrieval"):
        state["proceed_to_items"] = False
        return state
    if not state.get("retrieve_category") or not state.get("sufficiency_check"):
        state["proceed_to_items"] = True
        return state

    retrieved_content = ""
    store = state["store"]
    where_filters = state.get("where") or {}
    category_pool = state.get("category_pool") or store.memory_category_repo.list_categories(where_filters)
    hits = state.get("category_hits") or []
    if hits:
        retrieved_content = self._format_category_content(
            hits,
            state.get("category_summary_lookup", {}),
            store,
            categories=category_pool,
        )

    llm_client = self._get_step_llm_client(step_context)
    needs_more, rewritten_query = await self._decide_if_retrieval_needed(
        state["active_query"],
        state["context_queries"],
        retrieved_content=retrieved_content or "No content retrieved yet.",
        llm_client=llm_client,
    )
    state["next_step_query"] = rewritten_query
    state["active_query"] = rewritten_query
    state["proceed_to_items"] = needs_more
    if needs_more:
        embed_client = self._get_step_embedding_client(step_context)
        state["query_vector"] = (await embed_client.embed([state["active_query"]]))[0]
    return state
```

### 4.4 记忆项检索

```python
async def _rag_recall_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
    if not state.get("retrieve_item") or not state.get("needs_retrieval") or not state.get("proceed_to_items"):
        state["item_hits"] = []
        return state

    store = state["store"]
    where_filters = state.get("where") or {}
    items_pool = store.memory_item_repo.list_items(where_filters)
    qvec = state.get("query_vector")
    if qvec is None:
        embed_client = self._get_step_embedding_client(step_context)
        qvec = (await embed_client.embed([state["active_query"]]))[0]
        state["query_vector"] = qvec
    state["item_hits"] = store.memory_item_repo.vector_search_items(
        qvec,
        self.retrieve_config.item.top_k,
        where=where_filters,
        ranking=self.retrieve_config.item.ranking,
        recency_decay_days=self.retrieve_config.item.recency_decay_days,
    )
    state["item_pool"] = items_pool
    return state
```

### 4.5 资源检索

```python
async def _rag_recall_resources(self, state: WorkflowState, step_context: Any) -> WorkflowState:
    if (
        not state.get("needs_retrieval")
        or not state.get("retrieve_resource")
        or not state.get("proceed_to_resources")
    ):
        state["resource_hits"] = []
        return state

    store = state["store"]
    where_filters = state.get("where") or {}
    resource_pool = store.resource_repo.list_resources(where_filters)
    state["resource_pool"] = resource_pool
    corpus = self._resource_caption_corpus(store, resources=resource_pool)
    if not corpus:
        state["resource_hits"] = []
        return state

    qvec = state.get("query_vector")
    if qvec is None:
        embed_client = self._get_step_embedding_client(step_context)
        qvec = (await embed_client.embed([state["active_query"]]))[0]
        state["query_vector"] = qvec
    state["resource_hits"] = cosine_topk(qvec, corpus, k=self.retrieve_config.resource.top_k)
    return state
```

## 5. LLM 检索流程

LLM 检索流程与 RAG 流程结构相似，但使用 LLM 进行排名而非向量搜索：

### 5.1 分类路由

```python
async def _llm_route_category(self, state: WorkflowState, step_context: Any) -> WorkflowState:
    if not state.get("needs_retrieval"):
        state["category_hits"] = []
        return state
    llm_client = self._get_step_llm_client(step_context)
    store = state["store"]
    where_filters = state.get("where") or {}
    category_pool = store.memory_category_repo.list_categories(where_filters)
    hits = await self._llm_rank_categories(
        state["active_query"],
        self.retrieve_config.category.top_k,
        state["ctx"],
        store,
        llm_client=llm_client,
        categories=category_pool,
    )
    state["category_hits"] = hits
    state["category_pool"] = category_pool
    return state
```

### 5.2 记忆项检索

```python
async def _llm_recall_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
    if not state.get("needs_retrieval") or not state.get("proceed_to_items"):
        state["item_hits"] = []
        return state

    where_filters = state.get("where") or {}
    category_ids = [cat["id"] for cat in category_hits]
    llm_client = self._get_step_llm_client(step_context)
    store = state["store"]

    use_refs = getattr(self.retrieve_config.item, "use_category_references", False)
    ref_ids: list[str] = []
    if use_refs and category_hits:
        from memu.utils.references import extract_references

        for cat in category_hits:
            summary = cat.get("summary") or ""
            ref_ids.extend(extract_references(summary))
    if ref_ids:
        items_pool = store.memory_item_repo.list_items_by_ref_ids(ref_ids, where_filters)
    else:
        items_pool = store.memory_item_repo.list_items(where_filters)

    relations = store.category_item_repo.list_relations(where_filters)
    category_pool = state.get("category_pool") or store.memory_category_repo.list_categories(where_filters)
    state["item_hits"] = await self._llm_rank_items(
        state["active_query"],
        self.retrieve_config.item.top_k,
        category_ids,
        state.get("category_hits", []),
        state["ctx"],
        store,
        llm_client=llm_client,
        categories=category_pool,
        items=items_pool,
        relations=relations,
    )
    state["item_pool"] = items_pool
    state["relation_pool"] = relations
    return state
```

## 6. 关键组件分析

### 6.1 检索决策

```python
async def _decide_if_retrieval_needed(
    self,
    query: str,
    context_queries: list[dict[str, Any]] | None,
    retrieved_content: str | None = None,
    system_prompt: str | None = None,
    llm_client: Any | None = None,
) -> tuple[bool, str]:
    history_text = self._format_query_context(context_queries)
    content_text = retrieved_content or "No content retrieved yet."

    prompt = self.retrieve_config.sufficiency_check_prompt or PRE_RETRIEVAL_USER_PROMPT
    user_prompt = prompt.format(
        query=self._escape_prompt_value(query),
        conversation_history=self._escape_prompt_value(history_text),
        retrieved_content=self._escape_prompt_value(content_text),
    )

    sys_prompt = system_prompt or PRE_RETRIEVAL_SYSTEM_PROMPT
    client = llm_client or self._get_llm_client()
    response = await client.chat(user_prompt, system_prompt=sys_prompt)
    decision = self._extract_decision(response)
    rewritten = self._extract_rewritten_query(response) or query

    return decision == "RETRIEVE", rewritten
```

### 6.2 向量搜索

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
        # 显著性感知排序：相似度 × 强化 × 时效性
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

### 6.3 分类排名

```python
async def _rank_categories_by_summary(
    self,
    query_vec: list[float],
    top_k: int,
    ctx: Context,
    store: Database,
    embed_client: Any | None = None,
    categories: Mapping[str, Any] | None = None,
) -> tuple[list[tuple[str, float]], dict[str, str]]:
    category_pool = categories if categories is not None else store.memory_category_repo.categories
    entries = [(cid, cat.summary) for cid, cat in category_pool.items() if cat.summary]
    if not entries:
        return [], {}
    summary_texts = [summary for _, summary in entries]
    client = embed_client or self._get_llm_client()
    summary_embeddings = await client.embed(summary_texts)
    corpus = [(cid, emb) for (cid, _), emb in zip(entries, summary_embeddings, strict=True)]
    hits = cosine_topk(query_vec, corpus, k=top_k)
    summary_lookup = dict(entries)
    return hits, summary_lookup
```

## 7. 输出构建

### 7.1 RAG 输出构建

```python
def _rag_build_context(self, state: WorkflowState, _: Any) -> WorkflowState:
    response = {
        "needs_retrieval": bool(state.get("needs_retrieval")),
        "original_query": state["original_query"],
        "rewritten_query": state.get("rewritten_query", state["original_query"]),
        "next_step_query": state.get("next_step_query"),
        "categories": [],
        "items": [],
        "resources": [],
    }
    if state.get("needs_retrieval"):
        store = state["store"]
        where_filters = state.get("where") or {}
        categories_pool = state.get("category_pool") or store.memory_category_repo.list_categories(where_filters)
        items_pool = state.get("item_pool") or store.memory_item_repo.list_items(where_filters)
        resources_pool = state.get("resource_pool") or store.resource_repo.list_resources(where_filters)
        response["categories"] = self._materialize_hits(
            state.get("category_hits", []),
            categories_pool,
        )
        response["items"] = self._materialize_hits(state.get("item_hits", []), items_pool)
        response["resources"] = self._materialize_hits(
            state.get("resource_hits", []),
            resources_pool,
        )
    state["response"] = response
    return state
```

## 8. 架构设计分析

### 8.1 优点

1. **分层检索**：从粗到细的检索策略提高了效率和准确性
2. **可配置性**：丰富的配置选项允许自定义检索行为
3. **两种检索模式**：支持向量搜索(RAG)和LLM排名两种模式
4. **充分性检查**：在每个层级检查检索结果是否足够
5. **引用追踪**：支持通过分类引用直接检索相关记忆项
6. **显著性排序**：考虑记忆项的强化次数和时效性

### 8.2 优化空间

1. **并发处理**：当前流程是串行的，可以考虑并行检索不同层级
2. **缓存机制**：对于频繁查询的内容可以添加缓存
3. **检索策略**：可以增加更复杂的检索策略，如混合模式
4. **性能优化**：对于大规模数据集，向量搜索可能成为瓶颈
5. **错误处理**：增加更完善的错误处理和重试机制

## 9. 代码优化建议

### 9.1 并发检索优化

```python
# 当前串行实现
async def _rag_recall_all(self, state: WorkflowState, step_context: Any) -> WorkflowState:
    # 并行检索分类、记忆项和资源
    tasks = []
    if state.get("retrieve_category") and state.get("needs_retrieval"):
        tasks.append(self._rag_route_category(state.copy(), step_context))
    if state.get("retrieve_item") and state.get("needs_retrieval"):
        tasks.append(self._rag_recall_items(state.copy(), step_context))
    if state.get("retrieve_resource") and state.get("needs_retrieval"):
        tasks.append(self._rag_recall_resources(state.copy(), step_context))
    
    results = await asyncio.gather(*tasks)
    
    # 合并结果
    for result in results:
        state.update(result)
    
    return state
```

### 9.2 缓存实现

```python
# 添加缓存装饰器
from functools import lru_cache

class CachedRetrieveMixin(RetrieveMixin):
    @lru_cache(maxsize=100)
    def _cached_vector_search(self, query_hash: str, where_hash: str, top_k: int, ranking: str) -> list[tuple[str, float]]:
        # 实际的向量搜索逻辑
        pass
    
    async def _rag_recall_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        # 使用缓存的向量搜索
        query_hash = hash(str(state["query_vector"]))
        where_hash = hash(str(state.get("where")))
        hits = self._cached_vector_search(query_hash, where_hash, self.retrieve_config.item.top_k, self.retrieve_config.item.ranking)
        state["item_hits"] = hits
        return state
```

## 10. 总结

MEMU 的记忆检索系统是一个设计精良、功能丰富的组件，具有以下特点：

1. **灵活的配置系统**：支持多种检索策略和参数调整
2. **分层检索架构**：从分类到记忆项再到资源的逐步细化检索
3. **两种检索模式**：向量搜索(RAG)和LLM排名
4. **充分性检查**：在每个层级评估检索结果是否足够
5. **显著性排序**：考虑记忆项的重要性和时效性
6. **引用追踪**：支持通过分类引用直接检索相关记忆项

该系统为AI应用提供了强大的记忆检索能力，支持从海量记忆中快速准确地检索相关信息，为AI的响应提供了丰富的上下文支持。