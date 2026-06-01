


          
让我详细查看向量搜索和映射的完整代码逻辑：

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/project/cognee/cognee/modules/retrieval/utils/node_edge_vector_search.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/project/cognee/cognee/modules/graph/cognee_graph/CogneeGraph.py
          
## 从向量搜索到映射打分的完整详解

---

## 核心概念：什么是"命中"与"未命中"

### 向量搜索的工作原理

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           向量数据库                                      │
│                                                                          │
│  Entity_name 集合 (所有节点的向量)                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  id="n1" → Python      → 向量 [0.1, 0.2, ...]                  │    │
│  │  id="n2" → 机器学习    → 向量 [0.3, 0.4, ...]                  │    │
│  │  id="n3" → NumPy       → 向量 [0.5, 0.6, ...]                  │    │
│  │  id="n4" → TensorFlow  → 向量 [0.7, 0.8, ...]                  │    │
│  │  ... (可能有1000个节点)                                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  EdgeType_relationship_name 集合 (所有边的向量)                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  id="e1" → "是一种"  → 向量 [0.2, 0.1, ...]                     │    │
│  │  id="e2" → "使用"    → 向量 [0.4, 0.3, ...]                     │    │
│  │  id="e3" → "支持"    → 向量 [0.6, 0.5, ...]                     │    │
│  │  id="e4" → "基于"    → 向量 [0.8, 0.7, ...]                     │    │
│  │  ... (可能有500种关系)                                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 搜索过程

```
查询: "Python和机器学习有什么关系？"

1. 嵌入查询 → query_vector = [0.15, 0.25, ...]

2. 向量数据库计算相似度:

   Entity_name 集合:
   ┌─────────────────────────────────────────────────────────────────┐
   │  计算 query_vector 与所有节点向量的余弦相似度                     │
   │                                                                 │
   │  Python      : similarity = 0.85  → 距离 = 0.15  ✅ 命中        │
   │  机器学习    : similarity = 0.75  → 距离 = 0.25  ✅ 命中        │
   │  NumPy       : similarity = 0.45  → 距离 = 0.55  ✅ 命中        │
   │  TensorFlow  : similarity = 0.20  → 距离 = 0.80                   │
   │  ... (其他1000个节点相似度更低)                                   │
   └─────────────────────────────────────────────────────────────────┘

   假设 wide_search_top_k = 100，返回前100个命中的结果：
   → 返回 Python, 机器学习, NumPy, ... 等

3. "是一种" 关系 (e1) 的相似度计算:
   - "是一种" 向量 vs query_vector = 0.30
   - 相似度 = 0.30 → 距离 = 0.70
   - 在边集合中排名第50名以外...

   假设只返回前10个边关系：
   → "使用"(e2) 命中，但 "是一种"(e1) 未命中！
```

---

## 详细执行流程

### 第一步：构建图结构

假设我们有这样一张图：

```
                    ┌─────────────┐
                    │   Python   │
                    │  id: "n1"  │
                    │ imp: 0.9   │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   Edge E1   │
                    │  id: "e1"  │
                    │ 关系: "是一种"
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   机器学习   │
                    │  id: "n2"  │
                    │ imp: 0.8   │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   Edge E2   │
                    │  id: "e2"  │
                    │ 关系: "使用"
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │    NumPy    │
                    │  id: "n3"  │
                    │ imp: 0.7   │
                    └─────────────┘
```

### 第二步：图元素初始状态

```python
# 创建图
graph = CogneeGraph()

# 添加节点
n1 = Node(id="n1", attributes={"name": "Python", "importance_weight": 0.9})
n2 = Node(id="n2", attributes={"name": "机器学习", "importance_weight": 0.8})
n3 = Node(id="n3", attributes={"name": "NumPy", "importance_weight": 0.7})

# 添加边
e1 = Edge(node1=n1, node2=n2, attributes={"relationship": "是一种", "importance_weight": 0.6})
e2 = Edge(node1=n2, node2=n3, attributes={"relationship": "使用", "importance_weight": 0.5})

# 图中的边列表
graph.edges = [e1, e2]

# 图中节点字典
graph.nodes = {"n1": n1, "n2": n2, "n3": n3}

# edges_by_distance_key 用于边匹配
# key = edge_type_id (由 relationship 生成)
graph.edges_by_distance_key = {
    "e1": [e1],  # "是一种" 关系对应的边
    "e2": [e2],  # "使用" 关系对应的边
}
```

**初始状态（向量化前）**：

| 元素 | vector_distance | 说明 |
|------|----------------|------|
| n1 (Python) | `None` | 未初始化 |
| n2 (机器学习) | `None` | 未初始化 |
| n3 (NumPy) | `None` | 未初始化 |
| e1 (是一种) | `None` | 未初始化 |
| e2 (使用) | `None` | 未初始化 |

---

### 第三步：向量搜索执行

#### 3.1 查询嵌入

```python
# 输入
query_text = "Python和机器学习有什么关系？"

# 嵌入模型
query_embedding = await embedding_engine.embed_text([query_text])
# 输出: [0.15, -0.25, 0.35, ...] (1536维向量)
```

#### 3.2 搜索配置

```python
# 要搜索的集合
collections = [
    "Entity_name",              # 实体节点
    "TextSummary_text",         # 文本摘要
    "EntityType_name",          # 实体类型
    "DocumentChunk_text",       # 文档块
    "EdgeType_relationship_name" # 边关系
]

# 初始检索数量
wide_search_top_k = 100  # 首次检索返回100个最相似的元素
```

#### 3.3 并行向量搜索

```python
# 伪代码展示
search_results = await asyncio.gather(
    vector_engine.search(collection="Entity_name", query_vector=embedding, limit=100),
    vector_engine.search(collection="TextSummary_text", query_vector=embedding, limit=100),
    vector_engine.search(collection="EntityType_name", query_vector=embedding, limit=100),
    vector_engine.search(collection="DocumentChunk_text", query_vector=embedding, limit=100),
    vector_engine.search(collection="EdgeType_relationship_name", query_vector=embedding, limit=100),
)
```

#### 3.4 向量搜索结果详解

```python
# search_results 是一个列表，按 collections 顺序存储每个集合的搜索结果

search_results = [
    # [0] Entity_name 集合 - 实体节点
    # 返回的是按相似度排序的 top-100 结果
    [
        ScoredResult(id="n1", score=0.15, payload={...}),   # Python (第1名)
        ScoredResult(id="n2", score=0.25, payload={...}),   # 机器学习 (第2名)
        ScoredResult(id="n3", score=0.55, payload={...}),  # NumPy (第3名)
        ScoredResult(id="n7", score=0.60, payload={...}),  # 编程语言 (第4名)
        ScoredResult(id="n12", score=0.70, payload={...}), # 深度学习 (第5名)
        ...
        # 注意：只返回了命中的100个，不是所有1000个节点
    ],
    
    # [1] TextSummary_text 集合 - 文本摘要
    [],  # 无结果
    
    # [2] EntityType_name 集合 - 实体类型
    [
        ScoredResult(id="concept", score=0.30),  # 概念类型
        ScoredResult(id="technology", score=0.40),  # 技术类型
    ],
    
    # [3] DocumentChunk_text 集合 - 文档块
    [
        ScoredResult(id="chunk_100", score=0.35, payload={...}),
        ...
    ],
    
    # [4] EdgeType_relationship_name 集合 - 边关系 ★重点★
    [
        ScoredResult(id="e2", score=0.35, payload={...}),   # "使用" 关系 ✅ 命中
        # 假设只返回了10个最相似的边关系
        # "e1" (是一种) 没有进入前10名... 
        # 这意味着 "是一种" 边在这次搜索中"未命中"！
    ],
]
```

---

## 第四步：NodeEdgeVectorSearch 处理结果

### 4.1 分离节点和边结果

```python
class NodeEdgeVectorSearch:
    def set_distances_from_results(self, collections, search_results, query_list_length):
        
        # 初始化
        self.node_distances = {}
        self.edge_distances = []
        
        # 遍历每个集合的结果
        for collection_name, results in zip(collections, search_results):
            
            if collection_name == "EdgeType_relationship_name":
                # 边集合单独存储
                self.edge_distances = results  # [ScoredResult(e2, 0.35)]
            else:
                # 节点集合存储到 node_distances
                self.node_distances[collection_name] = results
```

**处理后的数据结构**：

```python
# node_distances: 存储所有节点的向量搜索结果
node_distances = {
    "Entity_name": [
        ScoredResult(id="n1", score=0.15),
        ScoredResult(id="n2", score=0.25),
        ScoredResult(id="n3", score=0.55),
        ...
    ],
    "TextSummary_text": [],
    "EntityType_name": [
        ScoredResult(id="concept", score=0.30),
        ScoredResult(id="technology", score=0.40),
    ],
    "DocumentChunk_text": [
        ScoredResult(id="chunk_100", score=0.35),
        ...
    ],
}

# edge_distances: 边的向量搜索结果
edge_distances = [
    ScoredResult(id="e2", score=0.35),  # "使用" 关系
]
# 注意：e1 ("是一种") 不在 edge_distances 中！它未命中！
```

---

## 第五步：距离映射到图元素

### 5.1 重置所有距离

```python
async def map_vector_distances_to_graph_nodes(node_distances, query_list_length=None):
    query_count = 1
    
    # 第一步：重置所有节点的距离为惩罚值
    # 这确保了没有匹配到向量搜索结果的节点有默认值
    self.reset_distances(self.nodes.values(), query_count)
```

**重置前**：
```python
n1.vector_distance = None
n2.vector_distance = None
n3.vector_distance = None
```

**重置后**：
```python
# reset_distances 实现
for node in graph.nodes.values():
    node.attributes["vector_distance"] = [6.5]  # 默认惩罚值

# 结果
n1.vector_distance = [6.5]
n2.vector_distance = [6.5]
n3.vector_distance = [6.5]
```

### 5.2 映射节点距离

```python
# 第二步：遍历节点集合，用向量搜索结果更新
for collection_name, scored_results in node_distances.items():
    if not scored_results:
        continue  # 跳过空集合
    
    for result in scored_results:
        node_id = result.id          # "n1", "n2", "n3"
        score = result.score         # 0.15, 0.25, 0.55
        
        node = graph.get_node(node_id)  # 查找对应节点
        if node:
            # 更新距离
            node.update_distance_for_query(
                query_index=0,
                score=score,  # 使用向量搜索的分数作为距离
                query_count=1,
                default_penalty=6.5
            )
```

**映射过程**：

```
node_distances 中的结果：
┌─────────────────────────────────────────────────────────┐
│  ScoredResult(id="n1", score=0.15) → Python 节点        │
│  → node.vector_distance = [0.15]                       │
├─────────────────────────────────────────────────────────┤
│  ScoredResult(id="n2", score=0.25) → 机器学习 节点       │
│  → node.vector_distance = [0.25]                       │
├─────────────────────────────────────────────────────────┤
│  ScoredResult(id="n3", score=0.55) → NumPy 节点          │
│  → node.vector_distance = [0.55]                       │
├─────────────────────────────────────────────────────────┤
│  其他节点不在搜索结果中，保持 [6.5]                       │
└─────────────────────────────────────────────────────────┘
```

**映射后节点状态**：

| 节点ID | 名称 | 原始距离 | 映射后 vector_distance |
|--------|------|----------|---------------------|
| n1 | Python | 0.15 | `[0.15]` ✅ 命中 |
| n2 | 机器学习 | 0.25 | `[0.25]` ✅ 命中 |
| n3 | NumPy | 0.55 | `[0.55]` ✅ 命中 |
| n4 | TensorFlow | (不在结果中) | `[6.5]` ❌ 未命中 |
| ... | ... | ... | ... |

### 5.3 映射边距离

```python
async def map_vector_distances_to_graph_edges(edge_distances, query_list_length=None):
    query_count = 1
    
    # 重置所有边的距离
    self.reset_distances(self.edges, query_count)
    
    # 遍历边的搜索结果
    for result in edge_distances:
        edge_id = result.id    # "e2"
        score = result.score   # 0.35
        
        # 关键：通过 edges_by_distance_key 查找对应的边
        matching_edges = graph.edges_by_distance_key.get(edge_id)
        
        if matching_edges:
            for edge in matching_edges:
                edge.update_distance_for_query(
                    query_index=0,
                    score=score,
                    query_count=1,
                    default_penalty=6.5
                )
```

**映射过程详解**：

```
edge_distances 中的结果：
┌─────────────────────────────────────────────────────────┐
│  ScoredResult(id="e2", score=0.35)                      │
│                                                         │
│  查找 graph.edges_by_distance_key["e2"]                  │
│  → 找到 [Edge(id="e2", relationship="使用")]             │
│  → edge.vector_distance = [0.35]                        │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  e1 ("是一种") 不在 edge_distances 中！                  │
│  → 没有匹配操作                                          │
│  → e1.vector_distance 保持 [6.5] (惩罚值)                │
│  → 这就是"未命中"！                                      │
└─────────────────────────────────────────────────────────┘
```

**映射后边状态**：

| 边ID | 关系 | 搜索命中？ | vector_distance | 原因 |
|------|------|-----------|----------------|------|
| e1 | 是一种 | ❌ 未命中 | `[6.5]` | 不在 edge_distances 中 |
| e2 | 使用 | ✅ 命中 | `[0.35]` | 在 edge_distances 中 |

---

## 第六步：完整映射后的图状态

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           映射后的完整图状态                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐ vec_dist=[0.15]   ┌─────────────┐ vec_dist=[0.25]           │
│  │   Python   │ importance=0.9    │   机器学习   │ importance=0.8            │
│  │   id: n1   │ ◀── 命中        │   id: n2   │ ◀── 命中                   │
│  └─────────────┘                  └──────┬──────┘                            │
│         │                               │                                    │
│         │                               │                                    │
│  vec_dist=[6.5]                    vec_dist=[0.35]                            │
│  importance=0.6                    importance=0.5                             │
│         │                               │                                    │
│         ▼                               ▼                                    │
│  ┌─────────────────────────────────────────────────┐                         │
│  │              Edge E1 (是一种)                   │                         │
│  │         ❌ 未命中 → vec_dist=[6.5]              │                         │
│  │         原因: 不在向量搜索结果中                  │                         │
│  └─────────────────────────────────────────────────┘                         │
│                       Edge E2 (使用)                                          │
│              ✅ 命中 → vec_dist=[0.35]                                        │
│                                                                              │
│  ┌─────────────┐ vec_dist=[0.55]                                            │
│  │    NumPy    │ importance=0.7                                              │
│  │   id: n3   │ ◀── 命中                                                    │
│  └─────────────┘                                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 第七步：三元组打分

### 7.1 E1 边打分（包含未命中的边）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         E1 三元组打分                                         │
│                   Python --[e1:是一种]--> 机器学习                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Element 1: Node1 (Python)                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ vector_distance = [0.15]  ✅ 命中                                      │   │
│  │ importance_weight = 0.9                                             │   │
│  │                                                                       │   │
│  │ Step 1: 应用 importance_weight                                        │   │
│  │   distance = (2 - 0.9) × 0.15 = 1.1 × 0.15 = 0.165                   │   │
│  │                                                                       │   │
│  │ Step 2: 反馈权重混合 (feedback_influence=0.2, feedback_weight=0.5)   │   │
│  │   0 <= 0.165 <= 2 → 可以混合                                        │   │
│  │   normalized = 0.165 / 2 = 0.0825                                   │   │
│  │   blended = 0.8 × 0.0825 + 0.2 × 0.5 = 0.066 + 0.1 = 0.166          │   │
│  │   effective = 0.166 × 2 = 0.332                                     │   │
│  │                                                                       │   │
│  │ Result: 0.332                                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Element 2: Edge (是一种) ★★★ 未命中 ★★★                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ vector_distance = [6.5]  ❌ 未命中 → 使用惩罚值                        │   │
│  │ importance_weight = 0.6                                             │   │
│  │                                                                       │   │
│  │ Step 1: 应用 importance_weight                                        │   │
│  │   distance = (2 - 0.6) × 6.5 = 1.4 × 6.5 = 9.1                       │   │
│  │                                                                       │   │
│  │ Step 2: 反馈权重混合检查                                               │   │
│  │   9.1 >= 6.5 (triplet_distance_penalty)                              │   │
│  │   → 跳过混合，保持惩罚值                                               │   │
│  │                                                                       │   │
│  │ Result: 9.1  ⚠️ 惩罚值，大幅增加三元组分数                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Element 3: Node2 (机器学习)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ vector_distance = [0.25]  ✅ 命中                                     │   │
│  │ importance_weight = 0.8                                             │   │
│  │                                                                       │   │
│  │ Step 1: (2 - 0.8) × 0.25 = 0.3                                       │   │
│  │ Step 2: normalized = 0.15, blended = 0.22, effective = 0.44          │   │
│  │                                                                       │   │
│  │ Result: 0.44                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  E1 三元组总分 = 0.332 + 9.1 + 0.44 = 9.872                          │   │
│  │                                                                       │   │
│  │  ⚠️ 由于边"是一种"未命中，总分很高，排名靠后！                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 E2 边打分（边命中）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         E2 三元组打分                                         │
│                 机器学习 --[e2:使用]--> NumPy                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Element 1: Node1 (机器学习)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ vector_distance = [0.25]  ✅ 命中                                    │   │
│  │ Result: 0.44                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Element 2: Edge (使用) ★★★ 命中 ★★★                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ vector_distance = [0.35]  ✅ 命中                                    │   │
│  │ importance_weight = 0.5                                             │   │
│  │                                                                       │   │
│  │ Step 1: (2 - 0.5) × 0.35 = 1.5 × 0.35 = 0.525                       │   │
│  │ Step 2: 0 <= 0.525 <= 2 → 可以混合                                   │   │
│  │   normalized = 0.2625                                               │   │
│  │   blended = 0.8 × 0.2625 + 0.2 × 0.5 = 0.31                         │   │
│  │   effective = 0.31 × 2 = 0.62                                       │   │
│  │                                                                       │   │
│  │ Result: 0.62 ✅ 正常分数                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Element 3: Node2 (NumPy)                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ vector_distance = [0.55]  ✅ 命中                                    │   │
│  │ importance_weight = 0.7                                             │   │
│  │                                                                       │   │
│  │ Step 1: (2 - 0.7) × 0.55 = 1.3 × 0.55 = 0.715                       │   │
│  │ Step 2: 0 <= 0.715 <= 2 → 可以混合                                   │   │
│  │   normalized = 0.3575                                               │   │
│  │   blended = 0.8 × 0.3575 + 0.2 × 0.5 = 0.386                        │   │
│  │   effective = 0.386 × 2 = 0.772                                     │   │
│  │                                                                       │   │
│  │ Result: 0.772                                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  E2 三元组总分 = 0.44 + 0.62 + 0.772 = 1.832                         │   │
│  │                                                                       │   │
│  │  ✅ 所有元素都命中，总分较低，排名靠前！                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 第八步：结果对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              最终结果                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   边       │  节点1   │   边   │  节点2   │  三元组分数  │   排名           │
│ ─────────────────────────────────────────────────────────────────────────── │
│   E2       │ 机器学习 │  使用  │  NumPy   │   1.83     │  🥇 第1名        │
│            │ vec=0.25 │ vec=0.35│ vec=0.55 │  ✅全命中   │                  │
│            │ ✅       │ ✅      │ ✅       │            │                  │
│ ─────────────────────────────────────────────────────────────────────────── │
│   E1       │  Python  │ 是一种  │ 机器学习  │   9.87     │  🥈 第2名        │
│            │ vec=0.15 │ vec=6.5 │ vec=0.25 │  ⚠️边未命中 │                  │
│            │ ✅       │ ❌惩罚  │ ✅       │            │                  │
│                                                                              │
│  分数越低 = 相似度越高 = 排名越前                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 关键理解：为什么"未命中"会导致惩罚？

### 未命中的本质

```
向量搜索返回的是 top-K 个最相似的元素

假设 edge 集合有 500 种关系类型：
- "使用" 相似度排名第 3 → 在 top-10 中 → ✅ 命中
- "是一种" 相似度排名第 50 → 不在 top-10 中 → ❌ 未命中

未命中意味着：
1. 该关系与查询的语义相似度较低
2. 系统没有找到该边的向量分数
3. 必须使用默认惩罚值 6.5
4. 惩罚值经过 importance_weight 放大后变成 9.1
5. 大幅增加三元组总分，导致排名靠后
```

### 惩罚值的作用

```python
# 惩罚值确保未命中的元素不会排在命中的元素前面

# 命中边的距离：0.35
# 未命中边的距离：6.5

# 即使节点分数再好，未命中边也会把总分拉高
# 9.1 (未命中边的贡献) >> 0.35 (命中边的贡献)
```

这就是为什么 **"机器学习--使用-->NumPy"** 排在 **"Python--是一种-->机器学习"** 前面的原因！
        