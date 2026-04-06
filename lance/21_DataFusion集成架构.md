# 第二十一章：DataFusion 集成架构

## 🎯 核心概览

Lance 通过与 Apache DataFusion 的深度集成，实现了向量搜索与 SQL 查询的无缝融合。DataFusion 是一个高性能的分布式 SQL 查询引擎，Lance 将自己暴露为一个 DataFusion 表源（TableProvider），使 SQL 引擎能够直接访问向量数据并执行向量搜索操作。

**整合效果**：Lance 的向量搜索能力与 DataFusion 的 SQL 能力完美融合，支持 **向量搜索 + SQL 过滤的混合查询**。

---

## 🏗️ DataFusion 架构基础

### DataFusion 核心概念

```
SQL 查询字符串
  ↓
SQL 解析器（Parser）
  ↓
逻辑计划（LogicalPlan）
  ↓
优化器（Optimizer）
  ↓
物理计划（ExecutionPlan）
  ↓
执行器（Executor）
  ↓
结果（RecordBatch）
```

### 核心接口

```rust
// TableProvider：Lance 需要实现这个 trait
#[async_trait]
pub trait TableProvider: Sync + Send {
    // 返回表的 Schema
    fn schema(&self) -> SchemaRef;
    
    // 返回表的所有 ExecutionPlan
    async fn scan(
        &self,
        state: &SessionState,
        projection: &Option<Vec<usize>>,
        filters: &[Expr],
        limit: &Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>>;
    
    // 返回表的统计信息
    fn statistics(&self) -> Option<Statistics>;
    
    // 支持的操作（如排序、分组）
    fn supports_filter_pushdown(&self, filter: &Expr) -> FilterPushdownSupport;
}

// ExecutionPlan：Lance 自定义的执行计划
pub trait ExecutionPlan: Send + Sync {
    // 返回输出 schema
    fn schema(&self) -> SchemaRef;
    
    // 执行查询，返回流
    fn execute(
        &self,
        partition: usize,
        state: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream>;
    
    // 输出分区数
    fn output_partitioning(&self) -> Partitioning;
    
    // 执行树中的子计划
    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>>;
}

// Expr：表达式（由 SQL 解析器生成）
pub enum Expr {
    Column(Column),
    Literal(ScalarValue),
    BinaryOp {
        left: Box<Expr>,
        op: Operator,
        right: Box<Expr>,
    },
    // ... 更多表达式类型
}
```

---

## 🔌 Lance 的 TableProvider 实现

### LanceTableProvider 核心结构

```rust
pub struct LanceTableProvider {
    dataset: Arc<Dataset>,           // Lance 数据集
    schema: SchemaRef,               // 表 Schema
    row_count: usize,                // 总行数
}

#[async_trait]
impl TableProvider for LanceTableProvider {
    fn schema(&self) -> SchemaRef {
        Arc::new(self.schema.clone())
    }
    
    async fn scan(
        &self,
        state: &SessionState,
        projection: &Option<Vec<usize>>,
        filters: &[Expr],
        limit: &Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // 关键逻辑：分析过滤条件，判断是否包含向量搜索
        let (vector_search_expr, scalar_filters) = self.extract_vector_search(filters)?;
        
        if let Some(vector_expr) = vector_search_expr {
            // 有向量搜索：使用 VectorSearchPlan
            Ok(Arc::new(VectorSearchPlan {
                dataset: self.dataset.clone(),
                vector_expr,
                scalar_filters,
                projection: projection.clone(),
                limit: limit.clone(),
            }))
        } else {
            // 无向量搜索：使用普通 ScanPlan
            Ok(Arc::new(ScanPlan {
                dataset: self.dataset.clone(),
                filters: filters.to_vec(),
                projection: projection.clone(),
                limit: limit.clone(),
            }))
        }
    }
    
    fn statistics(&self) -> Option<Statistics> {
        Some(Statistics {
            num_rows: Some(self.row_count),
            total_byte_size: None,
            column_statistics: vec![],
        })
    }
    
    fn supports_filter_pushdown(&self, filter: &Expr) -> FilterPushdownSupport {
        // 判断过滤条件是否支持下推
        match filter {
            // 向量搜索表达式不支持下推（需要特殊处理）
            Expr::VectorSearch { .. } => FilterPushdownSupport::Unsupported,
            // 标量条件支持下推
            Expr::BinaryOp { .. } => FilterPushdownSupport::Partial,
            _ => FilterPushdownSupport::Unsupported,
        }
    }
}
```

### 向量搜索表达式识别

```rust
impl LanceTableProvider {
    fn extract_vector_search(&self, filters: &[Expr]) -> Result<(Option<VectorSearchExpr>, Vec<Expr>)> {
        let mut vector_expr = None;
        let mut scalar_filters = vec![];
        
        for filter in filters {
            // 检查是否是向量搜索表达式
            if let Some(vs_expr) = self.try_parse_vector_search(filter)? {
                // 向量搜索表达式形如：
                // vector = @query_vec (使用 @ 前缀表示查询向量)
                // 或 vector IN (SELECT embedding FROM other_table)
                vector_expr = Some(vs_expr);
            } else {
                // 普通标量过滤
                scalar_filters.push(filter.clone());
            }
        }
        
        Ok((vector_expr, scalar_filters))
    }
    
    fn try_parse_vector_search(&self, expr: &Expr) -> Result<Option<VectorSearchExpr>> {
        match expr {
            // 匹配表达式：column = @parameter
            Expr::BinaryOp {
                left: box Expr::Column(col),
                op: Operator::Eq,
                right: box Expr::Literal(ScalarValue::Utf8(Some(param_name))),
            } if param_name.starts_with('@') => {
                Ok(Some(VectorSearchExpr {
                    column_name: col.name.clone(),
                    query_param: param_name.clone(),
                    metric: "l2".to_string(),
                    k: 10,  // 默认返回 Top-10
                }))
            }
            _ => Ok(None),
        }
    }
}

pub struct VectorSearchExpr {
    pub column_name: String,      // 向量列名
    pub query_param: String,      // 查询向量参数，如 "@query"
    pub metric: String,           // 距离度量：l2, cosine, dot
    pub k: usize,                 // 返回 Top-K
}
```

---

## ⚙️ ExecutionPlan 实现

### 向量搜索执行计划

```rust
pub struct VectorSearchPlan {
    pub dataset: Arc<Dataset>,
    pub vector_expr: VectorSearchExpr,
    pub scalar_filters: Vec<Expr>,
    pub projection: Option<Vec<usize>>,
    pub limit: Option<usize>,
}

#[async_trait]
impl ExecutionPlan for VectorSearchPlan {
    fn schema(&self) -> SchemaRef {
        // 返回投影后的 schema
        if let Some(ref proj) = self.projection {
            let fields: Vec<_> = proj.iter()
                .map(|&i| self.dataset.schema().fields[i].clone())
                .collect();
            Arc::new(Schema::new(fields))
        } else {
            self.dataset.schema()
        }
    }
    
    fn output_partitioning(&self) -> Partitioning {
        // 向量搜索结果由于已排序，不再分区
        Partitioning::UnknownPartitioning(1)
    }
    
    fn output_ordering(&self) -> Option<Vec<PhysicalSortExpr>> {
        // 结果按距离排序
        Some(vec![PhysicalSortExpr {
            expr: Arc::new(Column::new("__distance__", 0)),
            options: SortOptions::default(),
        }])
    }
    
    async fn execute(
        &self,
        partition: usize,
        state: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err("VectorSearchPlan outputs only one partition".into());
        }
        
        // 1. 获取查询向量（从参数中）
        let query_vector = state.get_vector_param(&self.vector_expr.query_param)?;
        
        // 2. 执行向量搜索
        let search_results = self.dataset.search(
            &self.vector_expr.column_name,
            &query_vector,
        )
        .limit(self.vector_expr.k)
        .execute()
        .await?;
        
        // 3. 应用标量过滤（如果有）
        let mut filtered = search_results;
        if !self.scalar_filters.is_empty() {
            filtered = self.apply_scalar_filters(filtered).await?;
        }
        
        // 4. 投影列（只返回需要的列）
        let projected = if let Some(ref proj) = self.projection {
            filtered.project(proj)?
        } else {
            filtered
        };
        
        // 5. 返回为流
        Ok(Box::pin(RecordBatchStream::new(projected)))
    }
    
    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]  // 向量搜索是叶节点
    }
    
    fn statistics(&self) -> Statistics {
        Statistics {
            num_rows: Some(self.vector_expr.k),
            total_byte_size: None,
            column_statistics: vec![],
        }
    }
}

impl VectorSearchPlan {
    async fn apply_scalar_filters(&self, mut batch: RecordBatch) -> Result<RecordBatch> {
        for filter in &self.scalar_filters {
            // 计算过滤表达式
            let mask = filter.evaluate(&batch)?;
            
            // 应用布尔掩码
            batch = batch.filter(&mask)?;
            
            // 如果已经没有行了，提前返回
            if batch.num_rows() == 0 {
                break;
            }
        }
        Ok(batch)
    }
}
```

### 普通扫描执行计划

```rust
pub struct ScanPlan {
    pub dataset: Arc<Dataset>,
    pub filters: Vec<Expr>,
    pub projection: Option<Vec<usize>>,
    pub limit: Option<usize>,
}

#[async_trait]
impl ExecutionPlan for ScanPlan {
    fn schema(&self) -> SchemaRef {
        // ... 与 VectorSearchPlan 类似
    }
    
    async fn execute(
        &self,
        partition: usize,
        state: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // 1. 创建 Scanner
        let mut scanner = self.dataset.scan();
        
        // 2. 应用过滤
        for filter in &self.filters {
            scanner = scanner.filter(filter)?;
        }
        
        // 3. 应用投影
        if let Some(ref proj) = self.projection {
            scanner = scanner.project(proj)?;
        }
        
        // 4. 应用 Limit
        if let Some(lim) = self.limit {
            scanner = scanner.limit(lim);
        }
        
        // 5. 执行扫描
        let batches = scanner.execute().await?;
        Ok(Box::pin(RecordBatchStream::new(batches)))
    }
    
    // ... 其他方法
}
```

---

## 🎯 LogicalPlan 扩展

### 向量搜索逻辑计划

```rust
pub enum LancePlanNode {
    // 向量搜索逻辑节点
    VectorSearch {
        input: Arc<LogicalPlan>,
        column: String,
        query: Vec<f32>,
        k: usize,
        metric: String,
    },
    
    // 向量存储扫描
    Scan {
        table_name: String,
        filters: Vec<Expr>,
        projection: Vec<String>,
    },
}

impl LogicalPlan {
    // 将 SQL 转换为逻辑计划的例子：
    // SELECT * FROM products WHERE embedding = @query LIMIT 10
    // 转换为：
    // VectorSearch(
    //   input: Scan("products"),
    //   column: "embedding",
    //   query: @query,
    //   k: 10,
    //   metric: "l2"
    // )
}
```

### 优化器规则

```rust
// Lance 自定义的优化器规则
pub struct LanceOptimizerRule;

impl OptimizerRule for LanceOptimizerRule {
    fn optimize(&self, plan: &LogicalPlan) -> Result<LogicalPlan> {
        // 规则1：下推标量过滤
        // 将 WHERE price < 100 在向量搜索之前执行
        
        // 规则2：投影下推
        // 只扫描需要的列
        
        // 规则3：Limit 下推
        // 将 LIMIT 推入索引
        
        // 规则4：向量搜索融合
        // 如果有多个向量列搜索，合并成一个计划
        
        Ok(plan.clone())
    }
}
```

---

## 🎨 自定义 UDF（用户定义函数）

### 向量距离 UDF

```rust
pub fn register_lance_udfs(ctx: &mut SessionContext) {
    // UDF1：向量欧氏距离
    ctx.register_udf(
        create_udf(
            "l2_distance",
            vec![
                DataType::FixedSizeList(Box::new(Field::new("item", DataType::Float32, true)), 768),
                DataType::FixedSizeList(Box::new(Field::new("item", DataType::Float32, true)), 768),
            ],
            Arc::new(DataType::Float32),
            Volatility::Immutable,
            Arc::new(|args| {
                // 计算两个向量的 L2 距离
                // ||v1 - v2||^2
                let v1 = args[0].as_fixed_size_list();
                let v2 = args[1].as_fixed_size_list();
                
                let mut distances = Vec::new();
                for i in 0..v1.len() {
                    let vec1 = v1.value(i).as_primitive::<Float32Type>();
                    let vec2 = v2.value(i).as_primitive::<Float32Type>();
                    
                    let dist: f32 = vec1.iter()
                        .zip(vec2.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    
                    distances.push(Some(dist.sqrt()));
                }
                
                Ok(Arc::new(Float32Array::from(distances)))
            }),
        )
    );
    
    // UDF2：向量余弦相似度
    ctx.register_udf(
        create_udf(
            "cosine_similarity",
            vec![
                DataType::FixedSizeList(Box::new(Field::new("item", DataType::Float32, true)), 768),
                DataType::FixedSizeList(Box::new(Field::new("item", DataType::Float32, true)), 768),
            ],
            Arc::new(DataType::Float32),
            Volatility::Immutable,
            Arc::new(|args| {
                // 计算余弦相似度
                // <v1, v2> / (||v1|| * ||v2||)
                let v1 = args[0].as_fixed_size_list();
                let v2 = args[1].as_fixed_size_list();
                
                let mut similarities = Vec::new();
                for i in 0..v1.len() {
                    let vec1 = v1.value(i).as_primitive::<Float32Type>();
                    let vec2 = v2.value(i).as_primitive::<Float32Type>();
                    
                    let dot_product: f32 = vec1.iter()
                        .zip(vec2.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    
                    let norm1: f32 = vec1.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                    let norm2: f32 = vec2.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                    
                    let sim = if norm1 > 0.0 && norm2 > 0.0 {
                        dot_product / (norm1 * norm2)
                    } else {
                        0.0
                    };
                    
                    similarities.push(Some(sim));
                }
                
                Ok(Arc::new(Float32Array::from(similarities)))
            }),
        )
    );
}
```

### 使用自定义 UDF

```python
import datafusion
import pandas as pd
import numpy as np

# 创建 DataFusion 会话
ctx = datafusion.SessionContext()

# 注册 Lance 表
ctx.register_lance("products", "products.lance")

# 使用内置 UDF 进行向量距离计算
query_vec = np.random.randn(768).astype(np.float32)

# 方法1：直接向量搜索
results = ctx.sql("""
    SELECT id, name, l2_distance(embedding, @query_vec) as distance
    FROM products
    WHERE category = 'electronics'
    ORDER BY distance
    LIMIT 10
""").collect()

print(results)

# 方法2：更复杂的混合查询
results = ctx.sql("""
    SELECT p.id, p.name, p.price,
           cosine_similarity(embedding, @query_vec) as similarity,
           CASE WHEN price < 100 THEN 'affordable' ELSE 'premium' END as category
    FROM products p
    WHERE similarity(embedding, @query_vec) > 0.8
    AND price BETWEEN 50 AND 500
    ORDER BY similarity DESC
    LIMIT 20
""").collect()

print(results.to_pandas())
```

---

## 🔄 查询执行流程

### 完整执行过程

```
SQL: SELECT * FROM products WHERE embedding = @query LIMIT 10

  ↓ Step 1: 解析 (Parser)
    LogicalPlan::Filter {
        input: LogicalPlan::Scan(...),
        predicate: Column("embedding") == Param("@query")
    }

  ↓ Step 2: 优化 (Optimizer)
    1. 下推投影到 Scan
    2. 识别向量搜索表达式
    3. 转换为 VectorSearchPlan
    优化后：VectorSearch {
        column: "embedding",
        query: @query,
        k: 10
    }

  ↓ Step 3: 物理规划 (Planner)
    Arc<VectorSearchPlan> {
        dataset: Arc<Dataset>,
        vector_expr: VectorSearchExpr,
        projection: Some([0, 1, 2]),
        limit: Some(10)
    }

  ↓ Step 4: 执行 (Executor)
    1. 获取查询向量：query_vec = state.get_vector_param("@query")
    2. 调用 dataset.search("embedding", query_vec).limit(10)
    3. 返回 10 个最相关的行

  ↓ Step 5: 收集结果 (Collector)
    结果转换为 RecordBatch，返回给用户
```

### Python 示例代码

```python
import datafusion
import numpy as np

# 创建会话
ctx = datafusion.SessionContext()

# 注册 Lance 表
ctx.register_lance("products", uri="products.lance")

# 准备查询向量
query_vec = np.array([0.1, 0.2, -0.3, ...], dtype=np.float32)  # 768 维

# 执行向量搜索 + SQL 混合查询
result = ctx.sql("""
    SELECT 
        id,
        name,
        price,
        category
    FROM products
    WHERE embedding = @query_vec
      AND price < 100
      AND rating > 4.0
    ORDER BY __distance__
    LIMIT 10
""").collect()

# 转换为 Pandas DataFrame
df = result.to_pandas()
print(df)
```

---

## 📊 性能优化

### 1. 统计信息优化

```rust
impl LanceTableProvider {
    fn statistics(&self) -> Option<Statistics> {
        // 精确的统计信息帮助优化器做出更好的决策
        let mut col_stats = Vec::new();
        
        // 为每个列计算统计信息
        for field in self.dataset.schema().fields.iter() {
            let stats = self.compute_column_statistics(field)?;
            col_stats.push(stats);
        }
        
        Some(Statistics {
            num_rows: Some(self.row_count),
            total_byte_size: Some(self.estimate_total_size()),
            column_statistics: col_stats,
        })
    }
    
    fn compute_column_statistics(&self, field: &Field) -> Result<ColumnStatistics> {
        match field.data_type() {
            // 数值列：计算最小值、最大值、空值数
            DataType::Float32 | DataType::Float64 => {
                let (min_val, max_val, null_count) = self.compute_numeric_stats(field)?;
                Ok(ColumnStatistics {
                    min_value: Some(min_val),
                    max_value: Some(max_val),
                    null_count: Some(null_count),
                    distinct_count: None,
                })
            }
            // 字符串列：计算不同值个数
            DataType::Utf8 => {
                let distinct = self.compute_distinct_count(field)?;
                Ok(ColumnStatistics {
                    min_value: None,
                    max_value: None,
                    null_count: Some(0),
                    distinct_count: Some(distinct),
                })
            }
            _ => Ok(ColumnStatistics::default()),
        }
    }
    
    fn estimate_total_size(&self) -> u64 {
        // 估计整个表的大小（用于优化器决策）
        // 计算：行数 × 平均行大小
        let avg_row_size = self.compute_avg_row_size();
        (self.row_count as u64) * (avg_row_size as u64)
    }
}
```

### 2. 过滤下推优化

```rust
impl LanceTableProvider {
    fn supports_filter_pushdown(&self, filter: &Expr) -> FilterPushdownSupport {
        match filter {
            // 向量搜索不支持下推
            Expr::VectorSearch { .. } => FilterPushdownSupport::Unsupported,
            
            // 简单的比较条件支持完全下推
            Expr::BinaryOp {
                left: box Expr::Column { .. },
                op: BinaryOperator::Lt | BinaryOperator::Gt | 
                    BinaryOperator::LtEq | BinaryOperator::GtEq |
                    BinaryOperator::Eq,
                right: box Expr::Literal(_),
            } => FilterPushdownSupport::Full,
            
            // AND 条件：逐个检查
            Expr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => {
                let left_support = self.supports_filter_pushdown(left);
                let right_support = self.supports_filter_pushdown(right);
                
                match (left_support, right_support) {
                    (FilterPushdownSupport::Full, FilterPushdownSupport::Full) => {
                        FilterPushdownSupport::Full
                    }
                    (FilterPushdownSupport::Unsupported, _) | 
                    (_, FilterPushdownSupport::Unsupported) => {
                        FilterPushdownSupport::Unsupported
                    }
                    _ => FilterPushdownSupport::Partial,
                }
            }
            
            // 其他条件部分支持
            _ => FilterPushdownSupport::Partial,
        }
    }
    
    // 实际执行过滤下推
    fn apply_filter_pushdown(&self, filter: &Expr) -> Result<Arc<Dataset>> {
        // 将过滤条件在索引层面处理
        // 返回过滤后的数据集（不加载完整数据）
        match filter {
            Expr::BinaryOp {
                left: box Expr::Column { name, .. },
                op: op @ (BinaryOperator::Lt | BinaryOperator::Gt | 
                         BinaryOperator::LtEq | BinaryOperator::GtEq),
                right: box Expr::Literal(value),
            } => {
                // 使用索引加速范围查询
                self.dataset.range_search(name, op, value)
            }
            _ => Ok(self.dataset.clone()),
        }
    }
}
```

### 3. 投影下推优化

```rust
impl LanceTableProvider {
    fn supports_pushdown_projection(&self) -> bool {
        // Lance 原生支持列选择，不需要扫描不必要的列
        true
    }
    
    fn push_down_projection(
        &self,
        projection: &Option<Vec<usize>>,
    ) -> Arc<Dataset> {
        // 只加载投影中需要的列
        if let Some(cols) = projection {
            let col_names: Vec<_> = cols.iter()
                .map(|&idx| self.dataset.schema().fields[idx].name.clone())
                .collect();
            
            // 创建仅包含这些列的扫描器
            self.dataset.select_columns(col_names)
        } else {
            self.dataset.clone()
        }
    }
}

// 优化器规则：自动应用投影下推
pub struct ProjectionPushdownRule;

impl OptimizerRule for ProjectionPushdownRule {
    fn optimize(&self, plan: &LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Projection { exprs, input } => {
                // 提取投影中引用的列
                let required_cols = extract_column_references(exprs);
                
                // 将 projection 推入 scan
                let optimized_input = match input.as_ref() {
                    LogicalPlan::Scan { .. } => {
                        // 直接在扫描时应用投影
                        self.optimize_scan_with_projection(input, &required_cols)?
                    }
                    _ => self.optimize(input)?,
                };
                
                Ok(LogicalPlan::Projection {
                    exprs: exprs.clone(),
                    input: Box::new(optimized_input),
                })
            }
            _ => Ok(plan.clone()),
        }
    }
}
```

### 4. Limit 下推优化

```rust
impl VectorSearchPlan {
    // Limit 对向量搜索特别有效
    fn push_down_limit(&mut self, limit: Option<usize>) {
        // 向量搜索本身就返回有序结果
        // 只需返回前 K 个即可
        if let Some(l) = limit {
            // 更新搜索的 K 值
            self.vector_expr.k = self.vector_expr.k.min(l);
        }
    }
}

// 优化器规则：自动下推 limit
pub struct LimitPushdownRule;

impl OptimizerRule for LimitPushdownRule {
    fn optimize(&self, plan: &LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Limit { n, input } => {
                // 检查 input 是否是向量搜索
                if let LogicalPlan::VectorSearch { k, .. } = input.as_ref() {
                    // 更新 K 值
                    let optimized_input = self.update_vector_search_k(input, *n)?;
                    return Ok(optimized_input);
                }
                
                // 否则保留原计划
                Ok(plan.clone())
            }
            _ => Ok(plan.clone()),
        }
    }
}
```

### 5. 执行计划优化

```rust
// 完整的优化器实现
pub struct LanceOptimizer {
    rules: Vec<Box<dyn OptimizerRule>>,
}

impl LanceOptimizer {
    pub fn new() -> Self {
        Self {
            rules: vec![
                Box::new(ProjectionPushdownRule),
                Box::new(LimitPushdownRule),
                Box::new(FilterPushdownRule),
                Box::new(VectorSearchFusionRule),
            ],
        }
    }
    
    pub fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let mut current_plan = plan;
        
        // 迭代应用所有规则
        for rule in &self.rules {
            current_plan = rule.optimize(&current_plan)?;
        }
        
        Ok(current_plan)
    }
}

// 向量搜索融合规则（合并多个向量搜索）
pub struct VectorSearchFusionRule;

impl OptimizerRule for VectorSearchFusionRule {
    fn optimize(&self, plan: &LogicalPlan) -> Result<LogicalPlan> {
        // 如果有多个独立的向量搜索，尝试融合成一个
        // 这样可以共享质心距离计算
        match plan {
            LogicalPlan::Union { inputs } => {
                let mut has_vector_search = false;
                for input in inputs {
                    if matches!(input.as_ref(), LogicalPlan::VectorSearch { .. }) {
                        has_vector_search = true;
                        break;
                    }
                }
                
                if has_vector_search && inputs.len() == 2 {
                    // 尝试融合两个向量搜索
                    if let (LogicalPlan::VectorSearch { .. }, LogicalPlan::VectorSearch { .. }) = 
                        (inputs[0].as_ref(), inputs[1].as_ref()) 
                    {
                        return self.fuse_vector_searches(inputs);
                    }
                }
                
                Ok(plan.clone())
            }
            _ => Ok(plan.clone()),
        }
    }
}
```

### 6. 查询成本估计

```rust
pub struct CostEstimator;

impl CostEstimator {
    pub fn estimate_cost(&self, plan: &ExecutionPlan) -> f64 {
        match plan {
            // 向量搜索成本 = 质心距离计算 + 分区扫描
            // 通常在 1-100ms 之间
            ExecutionPlan::VectorSearch { .. } => {
                100.0  // 相对成本单位
            }
            
            // 全表扫描成本 = 扫描所有行
            ExecutionPlan::Scan { row_count, .. } => {
                (*row_count as f64) * 0.001  // 每行 0.001 单位
            }
            
            // 过滤成本 = 输入成本 + 过滤计算
            ExecutionPlan::Filter { input, .. } => {
                self.estimate_cost(input) * 1.1  // 10% 额外开销
            }
            
            // 投影成本 = 输入成本 + 列投影
            ExecutionPlan::Projection { input, cols, .. } => {
                let base_cost = self.estimate_cost(input);
                // 只投影少量列时减少成本
                base_cost * ((*cols as f64) / 10.0).max(0.5)
            }
            
            _ => 1000.0,
        }
    }
}
```

### 7. 完整的查询优化流程

```python
import time
from dataclasses import dataclass
from typing import Dict, Tuple
import datafusion
import numpy as np

@dataclass
class QueryProfile:
    plan: str
    latency_ms: float
    memory_mb: float
    io_ops: int
    vector_searches: int

class QueryOptimizationPipeline:
    def __init__(self, session: datafusion.SessionContext):
        self.session = session
        self.optimizer = LanceOptimizer()
    
    def analyze_query(
        self,
        sql: str,
        show_plan: bool = True,
    ) -> QueryProfile:
        """分析查询性能"""
        
        # 1. 解析 SQL
        start = time.time()
        logical_plan = self.session.parse_sql(sql)
        parse_time = time.time() - start
        
        # 2. 优化逻辑计划
        start = time.time()
        optimized_plan = self.optimizer.optimize(logical_plan)
        optimize_time = time.time() - start
        
        # 3. 转换物理计划
        physical_plan = self.session.planner.create_physical_plan(optimized_plan)
        
        if show_plan:
            print(f"Optimized Plan:\n{physical_plan}")
            print(f"Parse: {parse_time*1000:.1f}ms, Optimize: {optimize_time*1000:.1f}ms")
        
        # 4. 执行并测量
        start = time.time()
        result = self.session.execute(physical_plan)
        execution_time = time.time() - start
        
        # 5. 分析执行计划中的向量搜索数
        vector_search_count = self.count_vector_searches(physical_plan)
        
        return QueryProfile(
            plan=str(optimized_plan),
            latency_ms=execution_time * 1000,
            memory_mb=0,  # 通过内存追踪器获取
            io_ops=0,     # 通过 IO 监控器获取
            vector_searches=vector_search_count,
        )
    
    def benchmark_query(
        self,
        sql: str,
        iterations: int = 10,
    ) -> Dict[str, float]:
        """基准测试查询性能"""
        
        latencies = []
        
        for _ in range(iterations):
            profile = self.analyze_query(sql, show_plan=False)
            latencies.append(profile.latency_ms)
        
        latencies.sort()
        return {
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "avg_ms": np.mean(latencies),
            "p50_ms": latencies[len(latencies) // 2],
            "p95_ms": latencies[int(len(latencies) * 0.95)],
            "p99_ms": latencies[int(len(latencies) * 0.99)],
        }
    
    def recommend_optimizations(self, sql: str) -> list:
        """自动推荐优化建议"""
        
        profile = self.analyze_query(sql)
        recommendations = []
        
        # 检查是否有向量搜索
        if profile.vector_searches == 0:
            recommendations.append(
                "考虑使用向量搜索而不是全表扫描"
            )
        
        # 检查是否有多个向量搜索
        if profile.vector_searches > 1:
            recommendations.append(
                f"检测到 {profile.vector_searches} 个向量搜索，可以融合优化"
            )
        
        # 检查延迟
        if profile.latency_ms > 100:
            recommendations.append(
                "延迟较高，考虑：1) 增加 Nprobes 参数；"
                "2) 使用预过滤减少搜索范围；3) 启用重排"
            )
        
        return recommendations

# 使用示例
ctx = datafusion.SessionContext()
ctx.register_lance("products", "products.lance")

pipeline = QueryOptimizationPipeline(ctx)

# 分析查询
sql = """
SELECT id, name, price
FROM products
WHERE embedding = @query_vec
  AND price < 100
  AND category = 'electronics'
LIMIT 10
"""

profile = pipeline.analyze_query(sql, show_plan=True)
print(f"\n执行时间: {profile.latency_ms:.1f}ms")
print(f"向量搜索数: {profile.vector_searches}")

# 基准测试
benchmark = pipeline.benchmark_query(sql, iterations=20)
print(f"\n基准测试结果:")
for metric, value in benchmark.items():
    print(f"  {metric}: {value:.1f}ms")

# 获得优化建议
recommendations = pipeline.recommend_optimizations(sql)
print(f"\n优化建议:")
for rec in recommendations:
    print(f"  - {rec}")
```

---

## 📚 总结

DataFusion 集成使 Lance 从单纯的向量数据库升级为 **向量 + SQL 的混合查询引擎**：

1. **TableProvider**：Lance 作为数据源被 DataFusion 认可
2. **ExecutionPlan**：自定义执行计划支持向量搜索
3. **LogicalPlan**：优化器理解向量搜索的含义
4. **UDF**：用户可以定义自己的向量操作函数

这种深度集成使得：
- SQL 查询可以直接调用向量搜索
- 向量搜索结果可以与 SQL 条件组合
- 所有操作在一个统一的执行框架内
- 性能得到充分优化

