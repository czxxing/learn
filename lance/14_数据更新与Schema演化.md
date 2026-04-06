# 第十四章：数据更新与 Schema 演化

## 🎯 核心概览

数据更新和 Schema 演化是现实系统的关键需求。Lance 支持无重写的列添加、类型转换和回填机制。

---

## 📊 第一部分：列的添加与删除

### 列添加（零成本）

```python
import lance

dataset = lance.open("users.lance")

# 添加新列：location（有默认值）
dataset.add_column("location", "default_city", default_value="Unknown")

# 内部机制：
# 1. 在 Schema 中添加新列定义
# 2. 记录：新列+默认值信息
# 3. 不修改现有数据文件
# 4. 读取时自动填充默认值

# 性能：O(1)，只修改元数据
```

**实现细节**：

```rust
impl Dataset {
    pub async fn add_column(
        &mut self,
        column_name: String,
        data_type: DataType,
        default_value: Option<ScalarValue>,
    ) -> Result<()> {
        // 步骤 1：验证列名不存在
        if self.schema.contains(&column_name) {
            return Err(Error::ColumnAlreadyExists(column_name));
        }
        
        // 步骤 2：在 Schema 中添加新列
        let new_field = Field::new(&column_name, data_type, true);
        self.schema.push(new_field);
        
        // 步骤 3：记录列的默认值
        let mut new_manifest = self.manifest.clone();
        new_manifest.metadata.insert(
            format!("column:{}_default", column_name),
            default_value.serialize()?,
        );
        
        // 步骤 4：原子提交
        self.commit(new_manifest).await?;
        
        Ok(())
    }
}
```

**读取时的处理**：

```rust
pub fn read_batch_with_schema_evolution(
    batch: RecordBatch,
    evolved_schema: &Schema,
) -> Result<RecordBatch> {
    let mut columns = batch.columns().to_vec();
    
    // 为新列添加默认值
    for field in evolved_schema.fields() {
        if !batch.schema().contains(field.name()) {
            // 新列不在原始批次中
            let default_value = field.metadata.get("default")?;
            let default_array = create_array_with_value(
                default_value,
                batch.num_rows(),
                &field.data_type,
            )?;
            columns.push(Arc::new(default_array));
        }
    }
    
    RecordBatch::try_new(evolved_schema.into(), columns)
}
```

### 列删除

```python
# 删除列：不删除数据文件
dataset.drop_column("old_column")

# 内部：
# 1. 在 Schema 中标记列为已删除
# 2. 创建新 Manifest
# 3. 扫描时跳过此列
# 4. 存储文件不变（向后兼容）

# 性能：O(1)
```

---

## 🔄 第二部分：类型转换

### 兼容的转换

```
可以零拷贝或廉价转换的类型：

int32 → int64
✓ 可以：符号扩展

float32 → float64
✓ 可以：精度提升

string → binary
✓ 可以：编码转换

但不可以逆转转换：
int64 → int32
✗ 范围可能溢出

float64 → float32
✗ 精度丢失
```

### 类型转换实现

```rust
impl Dataset {
    pub async fn alter_column_type(
        &mut self,
        column_name: String,
        new_type: DataType,
    ) -> Result<()> {
        // 步骤 1：验证转换可行
        if !can_convert(&self.schema.field(&column_name).data_type, &new_type) {
            return Err(Error::IncompatibleTypeConversion);
        }
        
        // 步骤 2：如果是廉价转换，只修改 Schema
        if is_cheap_conversion(&self.schema.field(&column_name).data_type, &new_type) {
            self.schema.field_mut(&column_name).data_type = new_type;
            let new_manifest = self.manifest.clone();
            return self.commit(new_manifest).await;
        }
        
        // 步骤 3：昂贵转换，需要回填
        self.backfill_column_type(&column_name, &new_type).await?;
        
        Ok(())
    }
}
```

---

## 💫 第三部分：回填（Backfill）机制

### 回填流程

```
场景：添加必需列（非空，无默认值）

步骤 1：扫描现有数据，提取值
old_data: [row_0, row_1, ..., row_N]
新列需要计算值: compute_value(row_i) → value_i

步骤 2：生成新列数据
new_column: [value_0, value_1, ..., value_N]

步骤 3：创建新 Fragment
包含：旧列 + 新列数据

步骤 4：更新 Manifest
指向新 Fragment

步骤 5：后台清理
删除旧 Fragment（可选）

性能：取决于列的计算复杂度
```

### 实现示例

```rust
impl Dataset {
    pub async fn backfill_column_type(
        &mut self,
        column_name: &str,
        new_type: &DataType,
    ) -> Result<()> {
        // 步骤 1：扫描原始列
        let old_column = self.scan()
            .select(vec![column_name.to_string()])
            .try_into_stream()
            .await?
            .collect::<Vec<_>>()
            .await;
        
        // 步骤 2：转换
        let converted_column = self.convert_column(
            old_column,
            &self.schema.field(column_name).data_type,
            new_type,
        ).await?;
        
        // 步骤 3：添加回填标记到 Schema
        self.schema.field_mut(column_name).data_type = new_type.clone();
        
        // 步骤 4：创建新的 Manifest（引用转换后的数据）
        let mut new_manifest = self.manifest.clone();
        new_manifest.metadata.insert(
            format!("column:{}_backfilled", column_name),
            "true".to_string(),
        );
        
        // 步骤 5：提交
        self.commit(new_manifest).await?;
        
        Ok(())
    }
}
```

---

## 📊 Schema 演化的实际案例

### 案例 1：电商产品表演化

```
初始版本（v0）：
{
    product_id: int64,
    name: string,
    price: float32,
}

v1：添加分类
dataset.add_column("category", string, default="general")

v2：添加评分
dataset.add_column("rating", float32, default=0.0)

v3：添加库存
dataset.add_column("stock", int32, default=0)

成本分析：
✓ 每次操作都是 O(1)
✓ 无数据重写
✓ 支持时间旅行（访问 v0 时自动填充默认）

读取时间旅行：
dataset_v0 = lance.open("products.lance", version=0)
# Schema：{product_id, name, price}

dataset_v3 = lance.open("products.lance", version=3)
# Schema：{product_id, name, price, category=default, rating=default, stock=default}
```

### 案例 2：数据类型升级

```
初始版本：
user_count: int32

v1：升级到 int64（支持更大的数字）
dataset.alter_column_type("user_count", int64)

# Lance 自动：
# 1. 检测这是廉价转换（符号扩展）
# 2. 只修改 Schema
# 3. 读取时自动转换

成本：O(1) - 无数据重写
```

---

## 🏗️ 第四部分：批量更新

### 行级别更新

```python
# 更新单行的单个值
dataset.update(
    row_id=12345,
    column="status",
    value="inactive"
)

# 内部：
# 1. 找到 row_id 对应的 Fragment
# 2. 加载该 Fragment
# 3. 修改值
# 4. 重新写入 Fragment
# 5. 创建新 Manifest
```

### 条件更新

```python
# 更新满足条件的所有行
dataset.update_where(
    filter="status = 'active' AND score < 50",
    updates={"status": "inactive"}
)

# 性能考虑：
# 需要扫描所有 Fragment，找出匹配行
# 对大数据集可能比较慢
# 建议：
# - 使用索引加速过滤
# - 批量更新小部分数据
# - 定期重新组织数据（VACUUM）
```

---

## 📚 总结

Lance 的更新和 Schema 演化特性：

1. **零成本列添加**：只修改元数据
2. **灵活的类型转换**：廉价转换自动优化
3. **回填机制**：支持复杂的数据变换
4. **向后兼容**：时间旅行访问旧 Schema
5. **原子性更新**：所有变更都是事务性的

下一章开始讨论索引系统。
