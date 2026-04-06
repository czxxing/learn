# 第二十八章：Blob 数据支持

## 🎯 核心概览

Blob（Binary Large Object）是指图像、视频、音频等大型非结构化数据。Lance 支持在同一数据集中混合存储结构化数据（向量、标量）和非结构化数据（Blob），实现真正的多模态数据库。

---

## 📦 Blob 存储架构

### Blob 类型定义

```rust
pub enum BlobType {
    // 内联（小于 64KB）
    Inline(Vec<u8>),
    
    // 外部存储（引用外部文件）
    External {
        path: String,
        offset: u64,
        length: u64,
        format: String,  // "png", "jpg", "mp4" 等
    },
    
    // 对象存储（S3、GCS 等）
    RemoteObject {
        uri: String,  // "s3://bucket/key"
        metadata: BlobMetadata,
    },
}

pub struct BlobMetadata {
    pub size: u64,
    pub mime_type: String,
    pub hash: String,  // SHA256 校验和
    pub created_at: u64,
}
```

### Blob 列定义

```rust
pub struct BlobColumn {
    pub name: String,
    pub blob_type: BlobType,
    pub storage_path: PathBuf,
    pub inline_threshold: u64,  // 小于此值内联存储
}

impl BlobColumn {
    pub fn new(name: &str, storage_path: PathBuf) -> Self {
        Self {
            name: name.to_string(),
            blob_type: BlobType::External {
                path: String::new(),
                offset: 0,
                length: 0,
                format: String::new(),
            },
            storage_path,
            inline_threshold: 64 * 1024,  // 默认 64KB
        }
    }
}
```

---

## 💾 Blob 存储与检索

### Blob 写入

```rust
pub struct BlobWriter {
    blob_dir: PathBuf,
    inline_threshold: u64,
}

impl BlobWriter {
    pub async fn write(&self, data: &[u8], format: &str) -> Result<BlobReference> {
        let hash = self.compute_hash(data);
        
        if data.len() < self.inline_threshold as usize {
            // 内联存储
            Ok(BlobReference::Inline {
                data: data.to_vec(),
                hash,
            })
        } else {
            // 外部存储
            let path = self.blob_dir.join(format!("{}.{}", hash, format));
            tokio::fs::write(&path, data).await?;
            
            Ok(BlobReference::External {
                path: path.to_string_lossy().to_string(),
                size: data.len() as u64,
                hash,
                format: format.to_string(),
            })
        }
    }
    
    async fn write_batch(&self, blobs: Vec<Vec<u8>>) -> Result<Vec<BlobReference>> {
        let mut refs = Vec::new();
        
        for blob_data in blobs {
            refs.push(self.write(&blob_data, "bin").await?);
        }
        
        Ok(refs)
    }
    
    fn compute_hash(&self, data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }
}
```

### Blob 读取（延迟加载）

```rust
pub struct BlobLoader {
    blob_dir: PathBuf,
    cache: Arc<RwLock<LruCache<String, Vec<u8>>>>,
}

impl BlobLoader {
    pub async fn load(&self, blob_ref: &BlobReference) -> Result<Vec<u8>> {
        match blob_ref {
            // 内联 Blob：直接返回
            BlobReference::Inline { data, .. } => {
                Ok(data.clone())
            }
            
            // 外部 Blob：检查缓存，如果没有则从磁盘读取
            BlobReference::External { path, size, hash, .. } => {
                // 检查缓存
                if let Some(data) = self.cache.read().unwrap().peek(hash) {
                    return Ok(data.clone());
                }
                
                // 从磁盘读取
                let data = tokio::fs::read(path).await?;
                
                // 验证大小
                if data.len() != *size as usize {
                    return Err("Blob size mismatch".into());
                }
                
                // 写入缓存
                self.cache.write().unwrap()
                    .put(hash.clone(), data.clone());
                
                Ok(data)
            }
            
            // 远程 Blob：从 S3/GCS 等读取
            BlobReference::Remote { uri, .. } => {
                self.load_remote(uri).await
            }
        }
    }
    
    async fn load_remote(&self, uri: &str) -> Result<Vec<u8>> {
        // 使用对象存储 SDK 读取
        // 这里是伪代码
        let s3_client = create_s3_client()?;
        let (bucket, key) = parse_s3_uri(uri)?;
        s3_client.get_object(&bucket, &key).await
    }
    
    // 批量加载优化：合并 IO
    pub async fn load_batch(&self, blob_refs: &[BlobReference]) -> Result<Vec<Vec<u8>>> {
        // 按存储位置分组
        let mut by_location = HashMap::new();
        
        for (idx, blob_ref) in blob_refs.iter().enumerate() {
            match blob_ref {
                BlobReference::External { path, .. } => {
                    by_location.entry(path.clone())
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
                _ => {}
            }
        }
        
        // 对每个位置的 Blobs 进行批量读取
        let mut results = vec![Vec::new(); blob_refs.len()];
        
        for (path, indices) in by_location {
            let data = tokio::fs::read(&path).await?;
            for idx in indices {
                results[idx] = data.clone();
            }
        }
        
        Ok(results)
    }
}
```

---

## 🎬 多模态数据示例

### Python API

```python
import lance
import numpy as np
from pathlib import Path

# 创建包含向量和图像的多模态数据集
data = {
    "id": [1, 2, 3, 4, 5],
    
    # 向量列
    "embedding": np.random.randn(5, 768).astype(np.float32),
    
    # 标量列
    "title": ["image1", "image2", "image3", "image4", "image5"],
    "category": ["cat", "dog", "cat", "dog", "cat"],
    
    # Blob 列（图像）
    "image": [
        Path("./images/1.jpg").read_bytes(),
        Path("./images/2.jpg").read_bytes(),
        Path("./images/3.jpg").read_bytes(),
        Path("./images/4.jpg").read_bytes(),
        Path("./images/5.jpg").read_bytes(),
    ],
}

# 写入表
table = lance.write_table(data, uri="multimodal.lance")

# 创建向量索引（只在 embedding 列）
table.create_index(
    column="embedding",
    index_type="ivf_pq",
    num_partitions=10,
)

# 向量搜索 + Blob 检索
query_vec = np.random.randn(768).astype(np.float32)

results = (
    table.search(query_vec)
    .limit(10)
    .to_list()
)

# 获取相关的图像
for result in results:
    image_data = result["image"]  # 自动加载 Blob
    title = result["title"]
    print(f"Found: {title}, size: {len(image_data)} bytes")
    
    # 保存图像
    Path(f"./output/{title}.jpg").write_bytes(image_data)
```

### 延迟加载示例

```python
# 延迟加载：不立即加载 Blob
results = table.search(query_vec).limit(10).to_arrow()
print(f"Retrieved {len(results)} results")  # 很快，没有加载图像

# 只在需要时加载
for row in results:
    if row["category"] == "cat":
        image = row["image"]  # 此时才加载
        # 处理图像
```

---

## 🔗 对象存储集成

### S3 存储

```rust
pub struct S3BlobStorage {
    s3_client: S3Client,
    bucket: String,
    prefix: String,
}

impl S3BlobStorage {
    pub async fn write(&self, data: &[u8], format: &str) -> Result<String> {
        let hash = compute_hash(data);
        let key = format!("{}/{}.{}", self.prefix, hash, format);
        
        self.s3_client.put_object(
            PutObjectRequest {
                bucket: self.bucket.clone(),
                key: key.clone(),
                body: Some(data.to_vec()),
                content_type: Some(format!("image/{}", format)),
                ..Default::default()
            }
        ).await?;
        
        Ok(format!("s3://{}/{}", self.bucket, key))
    }
    
    pub async fn read(&self, uri: &str) -> Result<Vec<u8>> {
        let (bucket, key) = parse_s3_uri(uri)?;
        
        let output = self.s3_client.get_object(
            GetObjectRequest {
                bucket,
                key,
                ..Default::default()
            }
        ).await?;
        
        let mut data = Vec::new();
        output.body.read_to_end(&mut data)?;
        Ok(data)
    }
}
```

---

## 📊 性能优化

### Blob 缓存策略

```rust
pub struct BlobCache {
    memory_cache: Arc<RwLock<LruCache<String, Vec<u8>>>>,
    max_memory: u64,
    current_memory: Arc<AtomicU64>,
}

impl BlobCache {
    pub async fn get_or_load(
        &self,
        blob_ref: &BlobReference,
        loader: &BlobLoader,
    ) -> Result<Vec<u8>> {
        // 1. 检查内存缓存
        if let Some(data) = self.memory_cache.read().unwrap().peek(&blob_ref.hash()) {
            return Ok(data.clone());
        }
        
        // 2. 从存储加载
        let data = loader.load(blob_ref).await?;
        
        // 3. 根据大小决定是否缓存
        let data_size = data.len() as u64;
        let current = self.current_memory.load(Ordering::Relaxed);
        
        if current + data_size < self.max_memory {
            self.memory_cache.write().unwrap()
                .put(blob_ref.hash(), data.clone());
            self.current_memory.fetch_add(data_size, Ordering::Relaxed);
        }
        
        Ok(data)
    }
}
```

---

## 📚 总结

Blob 数据支持使 Lance 成为真正的多模态数据库：

1. **混合存储**：结构化 + 非结构化数据
2. **延迟加载**：需要时才加载 Blob
3. **对象存储集成**：支持云存储
4. **智能缓存**：优化内存使用

这使得开发者可以在单一数据库中管理所有数据类型。
