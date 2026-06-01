


          
让我深入查看图三元组打分机制的完整代码：

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/project/cognee/cognee/modules/graph/cognee_graph/CogneeGraph.py
          

        
            
toolName: search_by_regex
            
status: success
          
query: calculate_top_triplet|map_vector_distances|vector_distance|importance_weight
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/project/cognee/cognee/modules/graph/cognee_graph/CogneeGraphElements.py
          
现在我对图三元组打分机制有了完整的理解，下面是详细分析：

---

## 📐 图三元组打分机制详解

### 核心概念

**三元组 (Triplet)** = `主语节点 (Node1) → 谓词/关系 (Edge) → 宾语节点 (Node2)`

图三元组打分是cognee检索系统的**核心算法**，它将向量搜索的相似度分数转换为**图结构中的三元组重要性分数**，用于返回最相关的知识片段。

---

### 1️⃣ 数据结构 — [CogneeGraphElements.py](file:///home/project/cognee/cognee/modules/graph/cognee_graph/CogneeGraphElements.py)

#### Node 节点结构
```python
class Node(BaseModel):
    id: str                              # 节点唯一标识
    attributes: Dict[str, Any]           # 节点属性
    skeleton_neighbours: List["Node"]   # 邻居节点列表
    skeleton_edges: List["Edge"]         # 连接该节点的边列表
    status: np.ndarray                   # 节点状态向量
    
    # 关键属性初始化
    def __init__(self, node_id, attributes=None, dimension=1, node_penalty=6.5):
        node_attributes = attributes if attributes else {}
        node_attributes["vector_distance"] = None  # 向量距离（打分时填充）
        ...
```

#### Edge 边结构
```python
class Edge(BaseModel):
    node1: "Node"                        # 源节点
    node2: "Node"                        # 目标节点
    attributes: Dict[str, Any]          # 边属性
    directed: bool                       # 是否为有向边
    status: np.ndarray                   # 边状态向量
    
    # 关键属性
    edge_attributes["vector_distance"] = None  # 向量距离
```

---

### 2️⃣ 距离初始化 — [CogneeGraphElements.py#L57-67](file:///home/project/cognee/cognee/modules/graph/cognee_graph/CogneeGraphElements.py#L57-67)

```python
def reset_vector_distances(self, query_count: int, default_penalty: float) -> None:
    """将所有节点/边的距离初始化为惩罚值"""
    self.attributes["vector_distance"] = [default_penalty] * query_count
```

**默认惩罚值**: `triplet_distance_penalty = 6.5`

这确保了没有向量分数的元素排名靠后。

---

### 3️⃣ 向量距离映射 — [CogneeGraph.py#L404-450](file:///home/project/cognee/cognee/modules/graph/cognee_graph/CogneeGraph.py#L404-450)

#### 映射到节点
```python
async def map_vector_distances_to_graph_nodes(self, node_distances, query_list_length=None):
    query_count = query_list_length or 1
    
    # 1. 重置所有节点距离为惩罚值
    self.reset_distances(self.nodes.values(), query_count)
    
    # 2. 遍历每个向量集合的搜索结果
    for collection_name, scored_results in node_distances.items():
        per_query_scored_results = self._normalize_query_distance_lists(...)
        
        for query_index, scored_results in enumerate(per_query_scored_results):
            for result in scored_results:
                node_id = str(getattr(result, "id", None))
                node = self.get_node(node_id)
                
                # 3. 获取向量数据库返回的分数
                score = float(getattr(result, "score", self.triplet_distance_penalty))
                
                # 4. 更新节点的向量距离
                node.update_distance_for_query(
                    query_index=query_index,
                    score=score,
                    query_count=query_count,
                    default_penalty=self.triplet_distance_penalty,
                )
```

#### 映射到边
```python
async def map_vector_distances_to_graph_edges(self, edge_distances, query_list_length=None):
    # 类似逻辑，但通过 edges_by_distance_key 查找匹配的边
    for query_index, scored_results in enumerate(per_query_scored_results):
        for result in scored_results:
            matching_edges = self.edges_by_distance_key.get(str(result.id))
            for edge in matching_edges:
                edge.update_distance_for_query(
                    query_index=query_index,
                    score=float(getattr(result, "score", self.triplet_distance_penalty)),
                    ...
                )
```

---

### 4️⃣ 核心打分算法 — [CogneeGraph.py#L459-540](file:///home/project/cognee/cognee/modules/graph/cognee_graph/CogneeGraph.py#L459-540)

```python
def _calculate_query_top_triplet_importances(self, k: int, query_index: int = 0):
    """
    计算某个查询的 Top-K 三元组重要性
    """
    
    def _effective_distance(distance: float, feedback_weight: float) -> float:
        """计算有效距离（考虑反馈权重）"""
        
        # 如果关闭了反馈影响，直接返回原始距离
        if active_feedback_influence <= 0.0:
            return distance
        
        # 仅对有效的余弦距离进行混合
        # 余弦距离范围: [0, 2]，其中 0=完全相同, 2=完全相反
        if distance >= self.triplet_distance_penalty or distance < 0.0 or distance > 2.0:
            return distance  # 保持惩罚值不变
        
        # 归一化反馈权重到 [0, 1]
        normalized_feedback_weight = max(0.0, min(1.0, float(feedback_weight)))
        
        # 归一化距离到 [0, 1] 空间
        normalized_distance = distance / 2.0
        
        # 混合公式: 
        # blended = (1 - feedback_influence) * 归一化距离 + feedback_influence * (1 - 反馈权重)
        blended_normalized = (1 - feedback_influence) * normalized_distance + \
                            feedback_influence * (1 - normalized_feedback_weight)
        
        # 投影回 [0, 2] 距离空间
        return blended_normalized * 2.0

    def score(edge: Edge) -> float:
        """
        计算一条边的三元组分数
        三元组 = Node1 + Edge + Node2
        """
        
        # 评估三元组的三个组成部分
        elements = (
            (edge.node1, "node1"),           # 主语节点
            (edge.node2, "node2"),           # 宾语节点  
            (edge, "edge")                   # 关系/边
        )
        
        importances = []
        
        for element, label in elements:
            # 1️⃣ 获取该元素的向量距离
            distances = element.attributes.get("vector_distance")
            
            # 2️⃣ 获取重要性权重 (默认0.5)
            importance_weight = element.attributes.get("importance_weight", 0.5)
            
            # 3️⃣ 应用重要性权重
            # importance_weight ∈ [0, 1]
            # weight=1 → 距离乘以(2-1)=1 → 保持不变
            # weight=0.5 → 距离乘以(2-0.5)=1.5 → 距离增加
            # weight=0 → 距离乘以(2-0)=2 → 距离翻倍
            distance = (2 - importance_weight) * distance
            
            # 4️⃣ 获取反馈权重 (默认0.5)
            feedback_weight = element.attributes.get("feedback_weight", 0.5)
            
            # 5️⃣ 计算有效距离（考虑反馈）
            effective_dist = _effective_distance(distance, feedback_weight)
            
            importances.append(effective_dist)
        
        # 6️⃣ 返回三元组总分（三个部分距离之和）
        return sum(importances)
    
    # 7️⃣ 使用最小堆获取 top-k 最小距离的三元组
    return heapq.nsmallest(k, self.edges, key=score)
```

---

### 5️⃣ 完整打分公式

对于一条边（包含两个节点），其三元组分数计算如下：

```
triplet_score(edge) = effective_distance(node1) + effective_distance(node2) + effective_distance(edge)
```

其中每个元素的**有效距离**计算为：

```
# 步骤1: 应用重要性权重
distance1 = (2 - importance_weight) * vector_distance

# 步骤2: 应用反馈权重混合
if feedback_influence > 0:
    normalized_dist = distance1 / 2.0
    blended = (1 - feedback_influence) * normalized_dist + \
              feedback_influence * (1 - feedback_weight)
    effective_distance = blended * 2.0
else:
    effective_distance = distance1
```

---

### 6️⃣ 关键参数详解

| 参数 | 默认值 | 范围 | 作用 |
|------|--------|------|------|
| `triplet_distance_penalty` | 6.5 | [0, ∞) | 默认惩罚值，无向量分数时使用 |
| `importance_weight` | 0.5 | [0, 1] | 元素重要性，影响距离缩放 |
| `feedback_weight` | 0.5 | [0, 1] | 用户反馈权重，提高相关元素排名 |
| `feedback_influence` | 0.0 | [0, 1] | 反馈影响程度 |

---

### 7️⃣ 重要性权重的影响

```python
distance = (2 - importance_weight) * distance
```

| importance_weight | 乘数 | 效果 |
|-------------------|------|------|
| 1.0 | 1.0 | 距离不变（高重要性元素） |
| 0.5 | 1.5 | 距离×1.5 |
| 0.0 | 2.0 | 距离翻倍（低重要性元素排名靠后） |

---

### 8️⃣ 反馈权重混合示意

```
当 feedback_influence = 0.5, feedback_weight = 0.8:

原始距离 = 0.4
normalized = 0.4 / 2 = 0.2

blended_normalized = (1 - 0.5) * 0.2 + 0.5 * (1 - 0.8)
                 = 0.5 * 0.2 + 0.5 * 0.2
                 = 0.1 + 0.1 = 0.2

effective_distance = 0.2 * 2 = 0.4  (距离保持不变，因为高反馈权重抵消)
```

---

### 9️⃣ 流程总览

```
┌─────────────────────────────────────────────────────────────┐
│                    向量搜索阶段                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │ Entity_name │   │Edge_relationship│ │TextSummary │ ...   │
│  │  0.3, 0.5   │   │  0.8, 0.2    │   │  0.6, 0.4   │        │
│  └─────────────┘   └─────────────┘   └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    距离映射阶段                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 重置所有节点的 vector_distance = [6.5]           │   │
│  │ 2. 根据搜索结果更新 vector_distance                  │   │
│  │ 3. 对于边，通过 edge_type_id 查找匹配的边并更新       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    三元组打分阶段                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 对每条边计算:                                         │   │
│  │   triplet_score = node1_dist + edge_dist + node2_dist│   │
│  │                                                     │   │
│  │ 其中每个距离应用:                                     │   │
│  │   - importance_weight 缩放                          │   │
│  │   - feedback_weight 混合                           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Top-K 选择                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 使用 heapq.nsmallest() 获取分数最低的 K 条边        │   │
│  │ (距离越低 = 相似度越高)                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        返回三元组结果
```

---

### 🔟 为什么使用距离而非相似度？

1. **一致性**: 余弦相似度转距离的公式是 `distance = 2 - similarity * 2`
2. **堆优化**: `heapq.nsmallest()` 天然适合找最小值，便于高效获取 Top-K
3. **可加性**: 三元组分数是三个距离之和，便于理解和调试
        