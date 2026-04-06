# 第9章：Dataset 核心实现与生命周期

## 概述

Dataset 是 Lance 的最高层抽象，提供了读写数据、版本管理、查询执行的统一接口。本章讨论 Dataset 的创建、打开、转换、混合整理。

## Dataset 的永久启动

### 创建 Dataset

```rust
pub struct Dataset {
    uri: String,
    manifest: Arc<Manifest>,
    schema: Arc<Schema>,
    fragments: Vec<Fragment>,
    object_store: Arc<dyn ObjectStore>,
    index_cache: Arc<IndexCache>,
}

impl Dataset {
    // 创建新 Dataset
    pub async fn create<T: RecordBatchReader>(
        uri: &str,
        reader: T,
        params: WriteParams,
    ) -> Result<Dataset> {
        // 1. 读取第一个批次获得 schema
        let first_batch = reader.next().transpose()?
            .ok_or("Empty dataset")?;
        let schema = Arc::new(Schema::from(first_batch.schema()));
        
        // 2. 创建物理存储
        let storage = ObjectStoreParams::from(uri).build_store()?;
        
        // 3. 转移数据到数据目录
        let mut writer = DatasetWriter::new(
            storage.clone(),
            schema.clone(),
            params,
        );
        
        // 写入第一个批次
        writer.write_batch(&first_batch).await?;
        
        // 对其余批次进行写入
        for batch in reader {
            let batch = batch?;
            writer.write_batch(&batch).await?;
        }
        
        // 4. 提交事务并获得manifest
        let manifest = writer.finish().await?;
        
        Ok(Dataset {
            uri: uri.to_string(),
            manifest: Arc::new(manifest),
            schema,
            fragments: vec![],
            object_store: storage,
            index_cache: Arc::new(IndexCache::new()),
        })
    }
}
```

### 打开 Dataset

```rust
impl Dataset {
    // 打开现有的 Dataset
    pub async fn open(uri: &str) -> Result<Dataset> {
        // 1. 创建物理存储
        let object_store = ObjectStoreParams::from(uri).build_store()?;
        
        // 2. 读取最新 manifest
        let manifest_path = Path::from(format!("{}/_lance/_latest.txt", uri));
        let latest_version_bytes = object_store.get(&manifest_path).await?;
        let latest_version: u64 = String::from_utf8(latest_version_bytes.to_vec())?
            .trim()
            .parse()?;
        
        // 3. 读取 manifest 文件
        let manifest_file_path = Path::from(
            format!("{}/_lance/_manifests/v{}.manifest", uri, latest_version)
        );
        let manifest_bytes = object_store.get(&manifest_file_path).await?;
        let manifest = Manifest::decode(&manifest_bytes)?;
        
        // 4. 初始化数据集
        Ok(Dataset {
            uri: uri.to_string(),
            manifest: Arc::new(manifest.clone()),
            schema: Arc::new(manifest.schema),
            fragments: manifest.fragments,
            object_store: Arc::new(object_store),
            index_cache: Arc::new(IndexCache::new()),
        })
    }
    
    // 打开指定版本
    pub async fn open_at_version(
        uri: &str,
        version: u64,
    ) -> Result<Dataset> {
        let object_store = ObjectStoreParams::from(uri).build_store()?;
        
        let manifest_file_path = Path::from(
            format!("{}/_lance/_manifests/v{}.manifest", uri, version)
        );
        let manifest_bytes = object_store.get(&manifest_file_path).await?;
        let manifest = Manifest::decode(&manifest_bytes)?;
        
        Ok(Dataset {
            uri: uri.to_string(),
            manifest: Arc::new(manifest.clone()),
            schema: Arc::new(manifest.schema),
            fragments: manifest.fragments,
            object_store: Arc::new(object_store),
            index_cache: Arc::new(IndexCache::new()),
        })
    }
}
```

## 数据转换接口

### Scanner 构建者模式

```rust
pub struct ScannerBuilder {
    dataset: Arc<Dataset>,
    projection: Option<Vec<String>>,
    filter: Option<String>,
    limit: Option<u64>,
    offset: Option<u64>,
}

impl ScannerBuilder {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        Self {
            dataset,
            projection: None,
            filter: None,
            limit: None,
            offset: None,
        }
    }
    
    pub fn project(mut self, columns: Vec<String>) -> Self {
        self.projection = Some(columns);
        self
    }
    
    pub fn filter(mut self, expr: String) -> Self {
        self.filter = Some(expr);
        self
    }
    
    pub fn limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }
    
    pub async fn build(self) -> Result<Scanner> {
        let schema = if let Some(cols) = self.projection {
            self.dataset.schema.project(&cols)?
        } else {
            self.dataset.schema.as_ref().clone()
        };
        
        Ok(Scanner {
            dataset: self.dataset.clone(),
            schema: Arc::new(schema),
            filter: self.filter,
            limit: self.limit,
            offset: self.offset,
        })
    }
}
```

### Scanner 查询执行

```rust
pub struct Scanner {
    dataset: Arc<Dataset>,
    schema: Arc<Schema>,
    filter: Option<String>,
    limit: Option<u64>,
    offset: Option<u64>,
}

impl Scanner {
    pub async fn try_into_stream(self) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let dataset = self.dataset.clone();
        let filter = self.filter.clone();
        let limit = self.limit;
        let offset = self.offset;
        let schema = self.schema.clone();
        
        let stream = stream::iter(dataset.fragments.clone())
            .then(move |fragment| {
                let dataset = dataset.clone();
                let filter = filter.clone();
                let schema = schema.clone();
                
                async move {
                    // 打开 fragment 中的数据文件
                    let reader = dataset.open_fragment(&fragment).await?;
                    
                    // 读取批次
                    let mut batch = reader.read_all(&schema).await?;
                    
                    // 应用过滤
                    if let Some(ref f) = filter {
                        batch = apply_filter(batch, f)?;
                    }
                    
                    Ok::<RecordBatch, Error>(batch)
                }
            })
            .boxed();
        
        Ok(stream)
    }
}
```

## 数据写入流程

```rust
pub struct DatasetWriter {
    object_store: Arc<dyn ObjectStore>,
    schema: Arc<Schema>,
    fragments: Vec<Fragment>,
    params: WriteParams,
}

impl DatasetWriter {
    pub async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        // 1. 编码批次
        let mut encoder = BatchEncoder::new(&self.schema);
        let encoded = encoder.encode_batch(batch).await?;
        
        // 2. 写入数据文件
        let data_file_path = self.generate_data_file_path();
        self.object_store.put(&data_file_path, encoded).await?;
        
        // 3. 创建 Fragment
        let fragment = Fragment {
            id: self.fragments.len() as u32,
            files: vec![DataFile {
                path: data_file_path.to_string(),
                column_indices: (0..self.schema.fields().len()).map(|i| i as u32).collect(),
                row_count: batch.num_rows() as u64,
                byte_size: batch.get_array_memory_size(),
                statistics: None,
            }],
            deletions: vec![],
            row_count: batch.num_rows() as u64,
            byte_size: batch.get_array_memory_size(),
            physical_rows: batch.num_rows() as u64,
        };
        
        self.fragments.push(fragment);
        Ok(())
    }
    
    pub async fn finish(self) -> Result<Manifest> {
        // 1. 创建 manifest
        let manifest = Manifest {
            version: 1,
            writer_version: WriterVersion::V2,
            created_at: chrono::Utc::now().timestamp_millis(),
            updated_at: chrono::Utc::now().timestamp_millis(),
            tag: None,
            description: None,
            schema: self.schema.as_ref().clone(),
            fragments: self.fragments,
            deletions: vec![],
            index_metadata: vec![],
            statistics: DatasetStatistics::from_fragments(&self.fragments),
            tags: vec![],
            branches: vec![],
        };
        
        // 2. 写入 manifest 文件
        let manifest_path = Path::from(
            format!("{}/_lance/_manifests/v{}.manifest", self.uri, manifest.version)
        );
        self.object_store.put(&manifest_path, manifest.encode()?).await?;
        
        // 3. 更新最新版本指针
        let latest_path = Path::from(format!("{}/_lance/_latest.txt", self.uri));
        self.object_store.put(
            &latest_path,
            Bytes::from(manifest.version.to_string()),
        ).await?;
        
        Ok(manifest)
    }
}
```

## 真实世界场景

### 场景 1：大规模数据转换

```rust
pub async fn convert_parquet_to_lance(
    parquet_path: &str,
    lance_uri: &str,
) -> Result<()> {
    // 1. 读入 Parquet 文件
    let file = tokio::fs::File::open(parquet_path).await?;
    let reader = ParquetFileReader::new(file);
    
    // 2. 转换为批次
    let batches = reader.to_record_batch()?;
    let record_batch_reader = RecordBatchIterator::new(
        batches.into_iter().map(Ok),
        reader.schema().clone(),
    );
    
    // 3. 创建 Lance Dataset
    let dataset = Dataset::create(
        lance_uri,
        record_batch_reader,
        WriteParams::default(),
    ).await?;
    
    Ok(())
}
```

### 场景 2：查询数据集

```rust
pub async fn query_dataset(
    uri: &str,
    columns: Vec<&str>,
    filter: &str,
) -> Result<Vec<RecordBatch>> {
    let dataset = Dataset::open(uri).await?;
    
    let scanner = dataset
        .scan()
        .project(columns.iter().map(|s| s.to_string()).collect())
        .filter(filter.to_string())
        .build()
        .await?;
    
    let batches: Vec<RecordBatch> = scanner
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    
    Ok(batches)
}
```

## 总结

Dataset 核心接口：

1. **创建打开**：创建新数据集或打开现有
2. **Scanner 构建**：灵活的查询构建
3. **批次处理**：并列批次处理
4. **数据写入**：控制版本管理
5. **数据查询**：时间旅行查询

下一章讨论索引系统架构与向量搜索。
