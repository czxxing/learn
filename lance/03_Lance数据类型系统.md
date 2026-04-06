# 第3章：Lance 数据类型系统

## 概述

Lance 的数据类型系统是在 Apache Arrow 的基础上构建的。为了支持多模态 AI 数据，Lance 扩展了 Arrow，添加了向量类型和 Blob 类型等特殊支持。本章讨论 Lance 如何扩展 Arrow 类型系统、Schema 设计与演化机制。

## Arrow 类型简介

### Arrow 的基本数据类型

Arrow 提供了各种标准数据类型：

```rust
pub enum DataType {
    // 原始数据类型
    Null,
    Boolean,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,
    Decimal128(i32, i32),  // 精度、小数位数
    
    // 二进制数据
    Binary,
    LargeBinary,
    FixedSizeBinary(i32),
    Utf8,
    LargeUtf8,
    
    // 复杂结构
    Struct(Vec<Field>),
    List(Box<Field>),
    LargeList(Box<Field>),
    FixedSizeList(Box<Field>, i32),
    Map(Box<Field>, bool),
    
    // 时间类型
    Date32,
    Date64,
    Timestamp(TimeUnit, Option<String>),
    Time32(TimeUnit),
    Time64(TimeUnit),
    Duration(TimeUnit),
    Interval(IntervalUnit),
    
    // 字典类型
    Dictionary(ref Box<DataType>, ref Box<DataType>),
}
```

### Lance 特有扩展类型

Lance 进一步扩展了 Arrow 中不存在的特殊类型：

```rust
// Lance 自定义扩展
pub enum LanceDataType {
    // 向量类型 - Arrow 中没有原生向量支持
    Vector(VectorType),
    
    // Blob 类型 - 大对象存储
    Blob,
    
    // Struct 扩展 - 支持嵌套向量
    Struct(Vec<Field>),  // 其中一个 Field 可能是 Vector
}

pub enum VectorType {
    Float32(u32),      // 预分配维数
    Float64(u32),
    Binary(u32),       // 1-bit 向量
}

pub struct Field {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub metadata: HashMap<String, String>,
}
```

## Schema 设计

### 什么是 Schema？

Schema 是描述整个数据集布局的结构：

```rust
pub struct Schema {
    // 字段列表
    fields: Vec<Field>,
    
    // 元数据（可选）
    metadata: HashMap<String, String>,
}

impl Schema {
    // 常用方法
    pub fn get_field(&self, name: &str) -> Option<&Field>;
    pub fn project(&self, fields: &[usize]) -> Schema;
    pub fn with_metadata(&mut self, key: String, value: String);
}
```

### Schema 元数据

Lance 在 Schema 的 metadata 中存储其他信息：

```rust
pub struct SchemaMetadata {
    // 版本信息
    pub version: u32,
    
    // 业务元数据
    pub tags: HashMap<String, String>,
    
    // 索引信息
    pub indexes: Vec<IndexMetadata>,
    
    // Blob 配置
    pub blob_version: Option<BlobVersion>,
}

pub enum BlobVersion {
    V1,  // 原始内联存储
    V2,  // 延迟加载优化
}
```

### Schema 命名约定

```rust
// 方案 1：手动构建
let schema = Arc::new(Schema::new(vec![
    Field::new("id", DataType::Int64, false),
    Field::new("name", DataType::Utf8, false),
    Field::new("embedding", DataType::List(
        Box::new(Field::new("item", DataType::Float32, false))
    ), true),
    Field::new("image", DataType::Binary, true),  // Blob
]));

// 方案 2：从 Arrow Schema 转换
use arrow_schema::Schema as ArrowSchema;
let arrow_schema = ArrowSchema::new(vec![...]);
let lance_schema = Schema::from(arrow_schema);
```

## 真实世界场景

### 场景 1：电子商务产品搜索

```rust
// 产品数据 Schema
pub struct Product {
    pub id: u64,
    pub name: String,
    pub description: String,
    pub price: f32,
    pub image_url: String,
    pub embedding: Vec<f32>,  // 文本描述的嵌入
    pub image_bytes: Vec<u8>,  // 产品图片
    pub categories: Vec<String>,
    pub updated_at: i64,
}

// Lance Schema 标准定义
let schema = Arc::new(Schema::new(vec![
    Field::new("id", DataType::UInt64, false),
    Field::new("name", DataType::Utf8, false),
    Field::new("description", DataType::Utf8, false),
    Field::new("price", DataType::Float32, false),
    Field::new("image_url", DataType::Utf8, false),
    Field::new("embedding", DataType::List(
        Box::new(Field::new("item", DataType::Float32, false))
    ), false),
    Field::new("image_bytes", DataType::Binary, false),  // Blob
    Field::new("categories", DataType::List(
        Box::new(Field::new("item", DataType::Utf8, false))
    ), false),
    Field::new("updated_at", DataType::Timestamp(TimeUnit::Millisecond, None), false),
]));
```

### 场景 2：生物信息学数据（需要序列化）

```rust
// DNA 序列数据 Schema
pub struct DNASequence {
    pub sequence_id: String,
    pub organism: String,
    pub sequence: String,  // ATCG 数据
    pub quality_scores: Vec<u8>,
    pub gc_content: f32,
    pub metadata_json: String,
}

let schema = Arc::new(Schema::new(vec![
    Field::new("sequence_id", DataType::Utf8, false),
    Field::new("organism", DataType::Utf8, false),
    Field::new("sequence", DataType::Utf8, false),  // 可以编码为二进制
    Field::new("quality_scores", DataType::List(
        Box::new(Field::new("item", DataType::UInt8, false))
    ), false),
    Field::new("gc_content", DataType::Float32, false),
    Field::new("metadata_json", DataType::Utf8, false),
    // 添加向量化描述嵌入
    Field::new("embedding", DataType::List(
        Box::new(Field::new("item", DataType::Float32, false))
    ), true),
]));
```

## Schema 演化

### 演化的第一个性质：添加新列

```rust
// 原始数据需要新列
// 表：[id, name, embedding]

// 添加 semantic_embedding 列
// 可以不立即回填数据，而是逐渐回填
let schema_evolution = SchemaEvolution {
    base_version: 1,
    changes: vec![
        SchemaChange::AddColumn {
            position: 3,
            field: Field::new(
                "semantic_embedding",
                DataType::List(Box::new(Field::new("item", DataType::Float32, false))),
                true,  // 可以为 NULL
            ),
            default_value: None,  // 不添加默认值
        },
    ],
};

// Lance 会采取以下策略：
// 1. 新 Fragment 添加 semantic_embedding
// 2. 旧 Fragment 仍然保存或逐渐回填（影响可选）
```

## 实际综合示例

### 应用：检索与视频推荐

```rust
pub struct MovieMetadata {
    pub id: u64,
    pub title: String,
    pub release_year: u32,
    pub poster_image: Vec<u8>,      // Blob
    pub description: String,
    pub description_embedding: Vec<f32>,  // 文本嵌入
    pub poster_embedding: Vec<u8>,  // 图像嵌入
    pub tags: Vec<String>,
    pub ratings: HashMap<String, f32>,
}

let schema = Arc::new(Schema::new(vec![
    Field::new("id", DataType::UInt64, false),
    Field::new("title", DataType::Utf8, false),
    Field::new("release_year", DataType::UInt32, false),
    Field::new("poster_image", DataType::Binary, false),  // Blob
    Field::new("description", DataType::Utf8, false),
    Field::new("description_embedding", DataType::List(
        Box::new(Field::new("item", DataType::Float32, false))
    ), false),
    Field::new("poster_embedding", DataType::List(
        Box::new(Field::new("item", DataType::Float32, false))
    ), false),
    Field::new("tags", DataType::List(
        Box::new(Field::new("item", DataType::Utf8, false))
    ), false),
    // 图像嵌入提示字典序列化元数据
    Field::new("ratings", DataType::Map(
        Box::new(Field::new("entries", DataType::Struct(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Float32, false),
        ]), false)),
        false,
    ), true),
]));

// 批此下推操作：仅获取需要的列
pub async fn search_by_title(dataset: &Dataset, keyword: &str) -> Result<Vec<u64>> {
    // 仅需要 id、title，不需要 poster_image 和嵌入数据
    let projection = vec!["id", "title"];
    let mut scanner = dataset.scan();
    scanner = scanner.project(projection)?;
    scanner = scanner.filter(format!("title LIKE '%{}%'", keyword))?;
    let batches = scanner.try_into_stream().await?;
    // ...
}
```

## 总结

Lance 的数据类型系统充分覆盖了场景：

1. **基于 Arrow**：利用 Arrow 的丰富数据类型体系
2. **便于扩展**：添加了 Vector、Blob 等类型
3. **性能考虑**：支持批此下推选择，指标过滤
4. **演化支持**：添加列时不需要完整重写
5. **美福考量**：支持索引信息的 Metadata 存储

下一章讨论：容器与缓存机制
