# 第十二章：Scanner 与查询执行

## 🎯 核心概览

Scanner 是 Lance 的查询执行引擎，负责将用户的查询转换为高效的执行计划。本章讲解 Scanner 的构建器模式、投影下推和谓词下推等优化技术。

---

## 📊 第一部分：Scanner 的架构设计

### What - Scanner 是什么？

**定义**：Scanner 是一个构建器模式的查询执行器，支持灵活的查询配置和优化。

```rust
pub struct Scanner {
    dataset: Arc<Dataset>,
    projection: Option<Vec<String>>,    // 选择的列
    filters: Vec<FilterExpr>,           // WHERE 条件
    limit: Option<usize>,               // LIMIT
    offset: Option<usize>,              // OFFSET
    index_hints: Vec<IndexHint>,        // 索引提示
}

impl Scanner {
    pub fn project(mut self, columns: Vec<String>) -> Self {
        self.projection = Some(columns);
        self
    }
    
    pub fn filter(mut self, expr: FilterExpr) -> Self {
        self.filters.push(expr);
        self
    }
    
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
    
    pub async fn try_into_stream(self) -> Result<impl Stream<Item = Result<RecordBatch>>> {
        let plan = self.build_execution_plan()?;
        plan.execute().await
    }
}
```

### Why - 为什么需要 Scanner？

#### 问题：不优化的扫描

```
简单方案：
scan() {
    for each fragment {
        read all columns
        return all rows
    }
}

性能问题：
✗ 读取不需要的列（浪费 IO）
✗ 不过滤行（浪费内存）
✗ 不使用索引（慢速扫描）

示例：
SELECT id, name FROM users WHERE age > 30

简单方案：读取所有 100 列，然后过滤
高效方案：
1. 识别只需要 id, name 列
2. 过滤 age > 30
3. 使用 age 上的索引（如果有）
性能提升：100 倍
```

### How - Scanner 的优化策略

#### 1. 投影下推（Projection Pushdown）

```
传统做法：
SELECT id, name FROM users
    ↓
读取所有列 [id, name, age, email, phone, ...]
    ↓
内存中选择 id, name

问题：浪费 IO

Lance 优化：
SELECT id, name FROM users
    ↓
编译器分析：只需要 id, name
    ↓
告诉文件读取器：只读这两列
    ↓
磁盘上跳过其他列
```

**实现示例**：

```rust
impl Scanner {
    fn build_execution_plan(&self) -> Result<ExecutionPlan> {
        // 识别需要的列
        let required_columns = match &self.projection {
            Some(cols) => cols.clone(),
            None => self.dataset.schema().all_columns(),
        };
        
        // 为了过滤，可能需要额外的列
        let mut columns_for_filter = required_columns.clone();
        for filter in &self.filters {
            columns_for_filter.extend(filter.required_columns());
        }
        
        // 去重
        columns_for_filter.sort();
        columns_for_filter.dedup();
        
        // 创建读取计划，只读这些列
        Ok(ExecutionPlan::Project {
            columns: columns_for_filter,
            filters: self.filters.clone(),
        })
    }
}
```

**性能对比**：

```
1000 万行，100 列的表

查询：SELECT col_1, col_2 FROM table WHERE col_5 > 100

无投影下推：
- 读取所有列：1GB × 100 = 100GB
- 过滤：在内存中过滤
- 性能：需要 100GB I/O

有投影下推：
- 读取 col_1, col_2, col_5：1GB × 3 = 3GB
- 过滤：在磁盘读取时应用
- 性能：只需要 3GB I/O
- 性能提升：33 倍
```

#### 2. 谓词下推（Predicate Pushdown）

```
传统做法：
WHERE age > 30 AND city = 'NYC'
    ↓
读取所有数据
    ↓
内存中应用过滤

问题：读了不需要的行

Lance 优化：
WHERE age > 30 AND city = 'NYC'
    ↓
检查是否有索引：
- age 有 BTree 索引
- city 有 Bitmap 索引
    ↓
使用索引快速获取候选行
    ↓
只读这些候选行的数据
```

**实现示例**：

```rust
impl Scanner {
    async fn apply_predicate_pushdown(&self) -> Result<Vec<u32>> {
        let mut candidates = (0..self.dataset.row_count()).collect::<Vec<_>>();
        
        for filter in &self.filters {
            // 尝试使用索引
            if let Some(index) = self.dataset.get_index(&filter.column) {
                let matching = index.search(&filter.value)?;
                candidates.retain(|id| matching.contains(id));
            } else {
                // 无索引，需要全扫描此列
                // 延迟到实际读取时
            }
        }
        
        Ok(candidates)
    }
}
```

---

## 🏗️ 第二部分：构建器模式详解

### 链式调用

```python
# Python 中的构建器模式
results = dataset.scan() \
    .select(["id", "name", "age"]) \  # 投影
    .where("age > 30") \              # 过滤 1
    .where("city = 'NYC'") \          # 过滤 2
    .limit(10) \                      # 限制
    .offset(5) \                      # 偏移
    .to_pandas()                      # 执行
```

### 内部转换

```
用户调用：scan().select([...]).where(...).limit(10)

内部状态转换：

1. scan()
   Scanner { projection: None, filters: [], limit: None }

2. .select(["id", "name"])
   Scanner { projection: Some(["id", "name"]), filters: [], limit: None }

3. .where("age > 30")
   Scanner { projection: Some(["id", "name"]), filters: [age > 30], limit: None }

4. .limit(10)
   Scanner { projection: Some(["id", "name"]), filters: [age > 30], limit: Some(10) }

5. .to_pandas()  // 执行
   构建执行计划 → 执行 → 转换为 Pandas
```

### 优化重排

```rust
impl Scanner {
    fn optimize_filters(&mut self) {
        // 优化 1：谓词重排（高选择性的放前面）
        self.filters.sort_by_key(|f| {
            // 估计选择性（返回的行数百分比）
            self.estimate_selectivity(f)
        });
        
        // 优化 2：合并相同列的过滤
        self.filters = self.merge_filters_on_same_column(&self.filters);
        
        // 优化 3：检查可以用索引的过滤
        self.identify_index_compatible_filters();
    }
}
```

---

## 💡 第三部分：查询优化实例

### 例子 1：简单查询

```python
import lance

users = lance.open("users.lance")

# 查询
results = users.scan() \
    .select(["id", "name"]) \
    .where("age > 30") \
    .to_pandas()

# Scanner 执行计划：
# 1. 识别需要的列：{id, name, age}
# 2. 检查过滤条件：age > 30
#    - age 有 BTree 索引
#    - 使用索引获取候选行
# 3. 读取计划：
#    - 使用索引找出 age > 30 的行
#    - 从磁盘读取这些行的 id, name, age 列
#    - 过滤 age > 30
#    - 投影 id, name
# 4. 执行

# 性能：
# 100万行表，只返回10万行
# 无优化：读 100万行 × 3列 = 3GB I/O
# 有优化：读 10万行 × 3列 = 300MB I/O
# 提升：10 倍
```

### 例子 2：向量搜索 + 过滤

```python
import numpy as np

products = lance.open("products.lance")

# 向量搜索 + 标量过滤
query = np.random.rand(768).astype(np.float32)

results = products.search(query, k=100) \
    .where("price < 500 AND category = 'electronics'") \
    .to_pandas()

# Scanner 执行计划：
# 1. 向量搜索（使用 IVF_PQ 索引）
#    - 得到 1000 个候选（k × 10）
# 2. 标量过滤
#    - 使用 price BTree 索引
#    - 使用 category Bitmap 索引
#    - 得到 200 个候选
# 3. 精确距离计算和排序
#    - 计算 200 个向量的精确距离
#    - 返回前 100 个

# 性能：
# 1000万商品，返回 100 个
# 无索引：扫描 1000万，计算距离
# 有索引：扫描 1000（向量索引），200（标量），计算 200
# 提升：50 倍（向量）+ 5 倍（标量）= 250 倍
```

---

## 📊 执行流程示意

```
用户查询
    ↓
Scanner 构建
    ↓
优化器分析
├─ 投影分析：需要哪些列
├─ 过滤分析：可用的索引
└─ 成本估计：选择最优执行路径
    ↓
执行计划生成
├─ 索引扫描（如果有索引）
├─ 列读取（投影下推）
├─ 过滤应用（谓词下推）
└─ 结果处理（排序、聚合）
    ↓
并行执行
├─ Fragment 1: [执行计划]
├─ Fragment 2: [执行计划]
└─ Fragment N: [执行计划]
    ↓
合并结果
    ↓
返回给用户
```

---

## 📚 总结

Scanner 通过以下机制实现高效查询：

1. **投影下推**：只读需要的列
2. **谓词下推**：尽早过滤行
3. **索引利用**：使用可用的索引加速
4. **成本优化**：选择最优的执行计划
5. **并行执行**：多 Fragment 并行处理

下一章将讲解数据写入流程，即数据如何进入 Lance。
