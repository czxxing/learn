# 第二十二章：SQL 查询支持

## 🎯 核心概览

Lance 通过完整的 SQL 解析、规划和执行框架，提供对向量数据的 SQL 查询支持。用户可以使用熟悉的 SQL 语法进行向量搜索、标量过滤、联接等复杂操作，而不需要了解 Lance 的内部 API。

**能力**：从 SQL 字符串直接执行向量搜索，支持**混合查询** = 向量搜索 + SQL 条件。

---

## 🔍 SQL 解析

### 解析流程

```
SQL 字符串
  ↓
SQL 词法分析（Lexer）
  ↓
SQL 语法分析（Parser）
  ↓
抽象语法树（AST）
  ↓
语义检查和验证
  ↓
逻辑计划（LogicalPlan）
```

### Lance 自定义的 SQL 语法扩展

```sql
-- 标准 SQL 中，向量搜索表示为：
SELECT * FROM table WHERE vector_column MATCH @query_vector

-- 或使用函数形式：
SELECT * FROM table 
WHERE vector_distance(vector_column, @query_vector) < threshold

-- 参数绑定语法：
-- @param_name：表示一个参数，由调用时动态传入
SELECT * FROM products 
WHERE embedding MATCH @query_vec
  AND price < @max_price
  AND category = @category
```

### Rust 实现：自定义 SQL Parser

```rust
use datafusion::sql::parser::{Parser, ParserConfig};
use datafusion::sql::sqlparser::ast::{Statement, Expr as SqlExpr};

pub struct LanceSqlParser {
    inner: Parser,
}

impl LanceSqlParser {
    pub fn new() -> Self {
        Self {
            inner: Parser::new(ParserConfig::default()),
        }
    }
    
    // 扩展 SQL 语法以支持向量搜索
    pub fn parse_statement(&self, sql: &str) -> Result<LanceStatement> {
        // 1. 使用标准 DataFusion 解析器
        let ast = self.inner.parse_sql(sql)?;
        
        // 2. 检查是否包含向量搜索表达式
        let lance_stmt = self.enhance_statement(&ast)?;
        
        Ok(lance_stmt)
    }
    
    fn enhance_statement(&self, ast: &Statement) -> Result<LanceStatement> {
        match ast {
            Statement::Select(select) => {
                // 检查 WHERE 子句中是否有向量搜索
                if let Some(where_clause) = &select.selection {
                    let (vector_expr, scalar_expr) = self.split_vector_and_scalar(where_clause)?;
                    
                    Ok(LanceStatement::Select {
                        columns: select.projection.clone(),
                        from: select.from.clone(),
                        vector_search: vector_expr,
                        where_clause: scalar_expr,
                        limit: select.limit.clone(),
                        order_by: select.order_by.clone(),
                    })
                } else {
                    Ok(LanceStatement::Select {
                        columns: select.projection.clone(),
                        from: select.from.clone(),
                        vector_search: None,
                        where_clause: None,
                        limit: select.limit.clone(),
                        order_by: select.order_by.clone(),
                    })
                }
            }
            _ => Ok(LanceStatement::Other(ast.clone())),
        }
    }
    
    fn split_vector_and_scalar(&self, expr: &SqlExpr) -> Result<(Option<VectorSearchExpr>, Option<SqlExpr>)> {
        // 将表达式分为两类：
        // 1. 向量搜索表达式（MATCH 或 distance < threshold）
        // 2. 普通标量表达式（可用 AND 连接）
        
        match expr {
            // 单个向量搜索表达式
            SqlExpr::Function { name, args, .. } if name.0[0].value == "vector_distance" => {
                let vector_expr = self.parse_vector_distance_expr(expr)?;
                Ok((Some(vector_expr), None))
            }
            
            // AND 表达式：可能两边都有或只有一边有向量搜索
            SqlExpr::BinaryOp { left, op, right } if op.to_string() == "AND" => {
                let (vec_left, scalar_left) = self.split_vector_and_scalar(left)?;
                let (vec_right, scalar_right) = self.split_vector_and_scalar(right)?;
                
                // 最多只能有一个向量搜索表达式
                let vector_expr = match (vec_left, vec_right) {
                    (Some(v), None) => Some(v),
                    (None, Some(v)) => Some(v),
                    (None, None) => None,
                    (Some(_), Some(_)) => return Err("Multiple vector searches not supported".into()),
                };
                
                // 合并标量表达式
                let scalar_expr = match (scalar_left, scalar_right) {
                    (Some(l), Some(r)) => Some(SqlExpr::BinaryOp {
                        left: Box::new(l),
                        op: op.clone(),
                        right: Box::new(r),
                    }),
                    (Some(e), None) | (None, Some(e)) => Some(e),
                    (None, None) => None,
                };
                
                Ok((vector_expr, scalar_expr))
            }
            
            // 其他表达式都认为是标量表达式
            _ => Ok((None, Some(expr.clone()))),
        }
    }
    
    fn parse_vector_distance_expr(&self, expr: &SqlExpr) -> Result<VectorSearchExpr> {
        // 解析：vector_distance(embedding, @query) < 0.5
        // 或：embedding MATCH @query_vec
        
        // 提取列名、查询向量参数、距离阈值等
        
        Ok(VectorSearchExpr {
            column_name: "embedding".to_string(),
            query_param: "@query".to_string(),
            metric: "l2".to_string(),
            threshold: 0.5,
        })
    }
}

pub enum LanceStatement {
    Select {
        columns: Vec<SqlSelectItem>,
        from: Vec<TableWithJoins>,
        vector_search: Option<VectorSearchExpr>,
        where_clause: Option<SqlExpr>,
        limit: Option<Limit>,
        order_by: Vec<OrderByExpr>,
    },
    Other(Statement),
}

pub struct VectorSearchExpr {
    pub column_name: String,
    pub query_param: String,
    pub metric: String,
    pub threshold: f32,
}
```

---

## 📋 规划与优化

### 逻辑计划生成

```rust
pub struct Planner {
    dataset: Arc<Dataset>,
}

impl Planner {
    pub fn plan_select(
        &self,
        select: &SelectStatement,
    ) -> Result<LogicalPlan> {
        // 1. 创建表扫描节点
        let scan = LogicalPlan::Scan {
            table: select.table.clone(),
            filter: None,
            projection: select.columns.clone(),
        };
        
        // 2. 如果有向量搜索，插入向量搜索节点
        let plan = if let Some(ref vector_expr) = select.vector_search {
            LogicalPlan::VectorSearch {
                input: Box::new(scan),
                column: vector_expr.column_name.clone(),
                query: vector_expr.query_param.clone(),
                k: 10,  // 默认返回 Top-10
            }
        } else {
            scan
        };
        
        // 3. 应用标量过滤
        let plan = if let Some(ref where_expr) = select.where_clause {
            LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: where_expr.clone(),
            }
        } else {
            plan
        };
        
        // 4. 应用排序
        let plan = if !select.order_by.is_empty() {
            LogicalPlan::Sort {
                input: Box::new(plan),
                sort_exprs: select.order_by.clone(),
            }
        } else {
            plan
        };
        
        // 5. 应用 Limit
        let plan = if let Some(limit) = &select.limit {
            LogicalPlan::Limit {
                input: Box::new(plan),
                n: limit.rows.len(),
            }
        } else {
            plan
        };
        
        Ok(plan)
    }
}

pub enum LogicalPlan {
    Scan {
        table: String,
        filter: Option<Expr>,
        projection: Vec<String>,
    },
    VectorSearch {
        input: Box<LogicalPlan>,
        column: String,
        query: String,
        k: usize,
    },
    Filter {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    Sort {
        input: Box<LogicalPlan>,
        sort_exprs: Vec<OrderByExpr>,
    },
    Limit {
        input: Box<LogicalPlan>,
        n: usize,
    },
}
```

### 优化规则

```rust
pub struct Optimizer;

impl Optimizer {
    pub fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let mut plan = plan;
        
        // 规则1：下推投影
        plan = self.push_down_projection(plan)?;
        
        // 规则2：下推过滤
        plan = self.push_down_filters(plan)?;
        
        // 规则3：合并向量搜索
        plan = self.merge_vector_searches(plan)?;
        
        // 规则4：消除不必要的排序
        plan = self.eliminate_redundant_sort(plan)?;
        
        Ok(plan)
    }
    
    // 规则1：投影下推
    // SELECT col1, col2 FROM (SELECT col1, col2, col3 FROM table)
    // 优化为：SELECT col1, col2 FROM table
    fn push_down_projection(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Scan { table, filter, .. } => {
                // 只扫描需要的列
                Ok(LogicalPlan::Scan {
                    table,
                    filter,
                    projection: vec!["col1".to_string(), "col2".to_string()],
                })
            }
            _ => Ok(plan),
        }
    }
    
    // 规则2：过滤下推
    // 将标量过滤推入向量搜索之前执行，减少搜索范围
    // SELECT * FROM (
    //   SELECT * FROM products WHERE embedding MATCH @query
    // ) WHERE price < 100
    // 优化为：
    // SELECT * FROM (
    //   SELECT * FROM products WHERE price < 100
    // ) WHERE embedding MATCH @query
    fn push_down_filters(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Filter {
                input: box LogicalPlan::VectorSearch { .. },
                predicate,
            } => {
                // 尝试将标量过滤下推
                // ... 具体逻辑
                Ok(plan)
            }
            _ => Ok(plan),
        }
    }
    
    // 规则3：合并多个向量搜索
    fn merge_vector_searches(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        // 如果有多个向量搜索，可以合并成一个
        // ... 具体逻辑
        Ok(plan)
    }
    
    // 规则4：消除冗余排序
    fn eliminate_redundant_sort(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        // 向量搜索结果已按距离排序，不需要再排序
        match plan {
            LogicalPlan::Sort {
                input: box LogicalPlan::VectorSearch { .. },
                ..
            } => {
                // 消除这个排序节点
                Ok(plan)
            }
            _ => Ok(plan),
        }
    }
}
```

---

## ⚙️ 执行引擎

### 混合查询执行

```rust
pub struct QueryExecutor {
    dataset: Arc<Dataset>,
}

impl QueryExecutor {
    pub async fn execute(
        &self,
        plan: LogicalPlan,
        params: QueryParams,  // 包含 @param 值的映射
    ) -> Result<Vec<RecordBatch>> {
        // 根据逻辑计划构建物理计划
        let physical_plan = self.build_physical_plan(plan)?;
        
        // 执行物理计划
        self.execute_physical_plan(physical_plan, params).await
    }
    
    fn build_physical_plan(&self, plan: LogicalPlan) -> Result<Arc<dyn ExecutionPlan>> {
        match plan {
            LogicalPlan::Scan { table, filter, projection } => {
                Ok(Arc::new(ScanExecutor {
                    table,
                    filter,
                    projection,
                }))
            }
            
            LogicalPlan::VectorSearch { input, column, query, k } => {
                Ok(Arc::new(VectorSearchExecutor {
                    dataset: self.dataset.clone(),
                    input: self.build_physical_plan(*input)?,
                    column,
                    query,
                    k,
                }))
            }
            
            LogicalPlan::Filter { input, predicate } => {
                Ok(Arc::new(FilterExecutor {
                    input: self.build_physical_plan(*input)?,
                    predicate,
                }))
            }
            
            LogicalPlan::Limit { input, n } => {
                Ok(Arc::new(LimitExecutor {
                    input: self.build_physical_plan(*input)?,
                    n,
                }))
            }
            
            _ => Err("Unsupported plan".into()),
        }
    }
    
    async fn execute_physical_plan(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        params: QueryParams,
    ) -> Result<Vec<RecordBatch>> {
        // 执行计划并收集结果
        let mut results = vec![];
        
        for partition in 0..plan.output_partitioning().num_partitions() {
            let context = TaskContext::new(params.clone());
            let stream = plan.execute(partition, context).await?;
            
            // 从流中读取所有 batch
            for batch in stream {
                results.push(batch);
            }
        }
        
        Ok(results)
    }
}
```

### 执行器实现

```rust
// 向量搜索执行器
pub struct VectorSearchExecutor {
    dataset: Arc<Dataset>,
    column: String,
    query: String,        // "@query_vec"
    k: usize,
}

#[async_trait]
impl Executor for VectorSearchExecutor {
    async fn execute(&self, params: &QueryParams) -> Result<Vec<RecordBatch>> {
        // 1. 从参数中获取查询向量
        let query_vec = params.get_vector(&self.query)?;
        
        // 2. 执行向量搜索
        let results = self.dataset.search(
            &self.column,
            &query_vec,
        )
        .limit(self.k)
        .execute()
        .await?;
        
        // 3. 返回结果
        Ok(vec![results])
    }
}

// 过滤执行器
pub struct FilterExecutor {
    input: Arc<dyn ExecutionPlan>,
    predicate: Expr,
}

#[async_trait]
impl Executor for FilterExecutor {
    async fn execute(&self, params: &QueryParams) -> Result<Vec<RecordBatch>> {
        // 1. 执行输入计划
        let mut input_results = self.input.execute(params).await?;
        
        // 2. 对每个 batch 应用过滤
        let mut results = vec![];
        for batch in input_results {
            // 计算谓词表达式
            let mask = self.predicate.evaluate(&batch)?;
            
            // 应用布尔掩码
            let filtered = batch.filter(&mask)?;
            
            results.push(filtered);
        }
        
        Ok(results)
    }
}
```

---

## 📝 SQL 查询示例

### 示例1：基础向量搜索

```python
import lance
import numpy as np

# 注册表
table = lance.write_table(
    {
        "id": np.arange(1000),
        "embedding": np.random.randn(1000, 768).astype(np.float32),
        "text": ["doc " + str(i) for i in range(1000)],
    },
    uri="docs.lance"
)

# 创建索引
table.create_index(column="embedding", index_type="ivf_pq")

# SQL 查询：基础向量搜索
query_vec = np.random.randn(768).astype(np.float32)

results = table.search_sql("""
    SELECT id, text, __distance__
    FROM docs
    WHERE embedding MATCH @query_vec
    LIMIT 10
""", params={"query_vec": query_vec})

for row in results:
    print(f"ID: {row['id']}, Distance: {row['__distance__']:.4f}")
```

### 示例2：混合查询（向量搜索 + SQL 条件）

```python
# SQL 查询：向量搜索 + 标量过滤
results = table.search_sql("""
    SELECT id, text, price
    FROM products
    WHERE embedding MATCH @query_vec
      AND price < @max_price
      AND category = @category
    LIMIT 10
""", params={
    "query_vec": query_vec,
    "max_price": 100.0,
    "category": "electronics",
})
```

### 示例3：聚合查询

```python
# SQL 查询：分组聚合
results = table.search_sql("""
    SELECT 
        category,
        COUNT(*) as count,
        AVG(price) as avg_price
    FROM products
    WHERE embedding MATCH @query_vec
    GROUP BY category
    LIMIT 10
""", params={"query_vec": query_vec})
```

### 示例4：联接查询

```python
# SQL 查询：表联接
results = table.search_sql("""
    SELECT 
        p.id,
        p.name,
        p.price,
        r.rating
    FROM products p
    JOIN reviews r ON p.id = r.product_id
    WHERE p.embedding MATCH @query_vec
      AND r.rating > 4.0
    LIMIT 10
""", params={"query_vec": query_vec})
```

---

## 🔄 完整查询处理流程

```
SQL: SELECT * FROM products 
     WHERE embedding MATCH @query_vec AND price < 100

  ↓ 步骤1：词法分析和语法分析
    - 分析 SELECT、FROM、WHERE 关键字
    - 识别列名、表名、参数

  ↓ 步骤2：语义检查
    - 验证列存在于表中
    - 验证参数类型匹配
    - 检查向量列有索引

  ↓ 步骤3：逻辑计划生成
    LogicalPlan = VectorSearch(
        input: Scan("products"),
        column: "embedding",
        query: "@query_vec",
        k: 10
    ) + Filter(price < 100)

  ↓ 步骤4：优化
    - 下推过滤：在向量搜索前先过滤价格
    - 投影下推：只扫描需要的列
    优化后：
    Filter(price < 100) → VectorSearch(...)

  ↓ 步骤5：物理计划生成
    - 分配具体的执行器
    - 选择索引类型（IVF、HNSW 等）

  ↓ 步骤6：执行
    1. 先对所有产品过滤（price < 100）
    2. 在过滤后的结果上执行向量搜索
    3. 返回 Top-10

  ↓ 步骤7：结果返回
    RecordBatch with columns: [id, name, price, embedding, __distance__]
```

---

## 📚 总结

SQL 查询支持使 Lance 成为一个真正的**混合数据库**：

1. **完整的 SQL 支持**：SELECT、WHERE、JOIN、GROUP BY 等
2. **向量搜索一等公民**：在 SQL 中直接使用向量搜索
3. **无缝集成**：向量操作与标量操作在同一框架内
4. **性能优化**：优化器自动选择最优执行计划
5. **灵活的参数**：支持动态参数绑定

这使得开发者可以用熟悉的 SQL 语言处理向量数据，无需学习新的 API 或编程模式。
