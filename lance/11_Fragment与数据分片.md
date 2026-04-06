# 第十一章：Fragment 与数据分片

## 🎯 核心概览

Fragment 是 Lance 数据集的逻辑分片单位，是在 Manifest 层面组织数据的关键。本章深入讲解 Fragment 的设计理念、组织方式和在实际中的应用。

---

## 📊 第一部分：Fragment 的设计理念

### What - Fragment 是什么？

**定义**：Fragment 是 Lance 数据集中的逻辑分片，代表一个独立的、自包含的数据单元。

```rust
pub struct Fragment {
    pub id: u32,                    // Fragment ID
    pub physical_schema: Schema,    // 物理 Schema
    pub row_count: u64,            // 行数
    pub metadata: FragmentMetadata, // 元数据
    pub deletion_file: Option<String>, // 删除标记文件
}
```

### Why - 为什么需要 Fragment？

#### 问题：单文件的局限

```
不分片方案：
数据集 = 单个 1GB 文件

问题：
✗ 文件太大，加载到内存需要 1GB+
✗ 删除行需要重写整个文件
✗ 并行读取困难
✗ 缓存不友好（无法只缓存部分列）
```

#### 解决方案：Fragment 分片

```
分片方案：
数据集 = 10 个 100MB Fragments

优势：
✓ 每个 Fragment 独立管理
✓ 删除只影响一个 Fragment
✓ 支持并行读取多个 Fragment
✓ 缓存以 Fragment 为单位
✓ Schema 演化更灵活
```

### How - Fragment 的实现策略

#### 分片大小的考虑

```
Fragment 大小的权衡：

太小（10MB）：
✗ 过多小文件，元数据开销大
✗ 列表操作频繁（创建、删除）
✓ 缓存更友好

太大（1GB）：
✓ 文件数少，元数据开销小
✗ 不够灵活，删除成本高
✗ 缓存压力大

Lance 的建议：100-256MB
✓ 平衡文件数和单个文件大小
✓ 实践验证的最优大小
```

---

## 🏗️ 第二部分：数据文件组织

### Fragment 的物理存储

```
一个 Dataset 的文件结构：

dataset/
├── _latest              # 指向最新版本
├── _versions/
│   ├── 0.manifest       # 版本 0 的 Manifest
│   ├── 1.manifest       # 版本 1 的 Manifest
│   └── 2.manifest       # 版本 2（当前）
├── data/
│   ├── 0.lance          # Fragment 0 数据文件
│   ├── 1.lance          # Fragment 1 数据文件
│   ├── 2.lance          # Fragment 2 数据文件（可能跨多个版本使用）
│   └── _deletions_2.txt # Fragment 2 的删除标记（v2 中）
└── indices/
    ├── embedding_0.idx  # Fragment 0 的索引
    └── embedding_1.idx  # Fragment 1 的索引
```

### Fragment 跨版本共享

```
关键特性：不同版本的数据集可以共享 Fragment

v0: Manifest {fragments: [F0, F1, F2], schema: {a, b, c}}
v1: Manifest {fragments: [F0, F1, F2, F3], schema: {a, b, c}}
    （F3 是新增数据）

v2: Manifest {
    fragments: [F0, F1, F2, F3], 
    schema: {a, b, c, d},
    F0_metadata: {new_column: d, default: 0}
}
    （添加列 d，F0-F3 的旧行自动填充默认值）

好处：
✓ 零拷贝列添加（只改元数据）
✓ 版本间共享数据文件（节省存储）
✓ 时间旅行无需数据重复
```

---

## 📝 第三部分：删除文件处理

### 删除机制

```
删除行的处理方式：

不使用删除文件：
row_id_list = [0, 5, 10, 15, ...]
扫描时遍历所有行，逐一检查是否在 row_id_list 中
性能：O(n)，n 为行总数

使用删除文件（Bitmap）：
deletion_file = Bitmap {
    bits: [1, 0, 0, 0, 1, ...]  // 1 表示已删除
    size: 8 bytes / 行
}
扫描时直接检查 bitmap
性能：O(1)，1-2 cycles

Lance 的做法：
- 小删除量（<1%）：inline 在 Fragment 元数据
- 大删除量（>1%）：单独的删除文件
```

### 删除文件的实现

```rust
pub struct DeletionFile {
    pub fragment_id: u32,
    pub version: u64,
    pub bitmap: RoaringBitmap,  // 高效的位图实现
}

impl Dataset {
    pub async fn delete_rows(&mut self, row_ids: &[u32]) -> Result<()> {
        // 步骤 1：按 Fragment 分组
        let mut fragments_to_delete = HashMap::new();
        for &row_id in row_ids {
            let (frag_id, local_id) = resolve_row_id(row_id);
            fragments_to_delete.entry(frag_id)
                .or_insert_with(Vec::new)
                .push(local_id);
        }
        
        // 步骤 2：创建新 Manifest
        let mut new_manifest = self.manifest.clone();
        
        for (frag_id, local_ids) in fragments_to_delete {
            let mut bitmap = RoaringBitmap::new();
            for &id in &local_ids {
                bitmap.insert(id);
            }
            
            // 写入删除文件
            let del_file = DeletionFile {
                fragment_id: frag_id,
                version: new_manifest.version + 1,
                bitmap,
            };
            
            self.store.put(&del_file.path(), del_file.serialize()).await?;
            
            // 更新 Manifest 中的删除文件引用
            new_manifest.fragments[frag_id].deletion_file = Some(del_file.path());
        }
        
        // 步骤 3：原子提交新版本
        self.commit(new_manifest).await?;
        
        Ok(())
    }
}
```

### 扫描时的删除过滤

```rust
pub async fn scan_with_deletions(
    &self,
    fragment_id: u32,
) -> Result<impl Stream<Item = RecordBatch>> {
    let fragment = &self.manifest.fragments[fragment_id];
    
    // 加载删除 bitmap
    let deletion_bitmap = if let Some(del_file) = &fragment.deletion_file {
        let data = self.store.get(del_file).await?;
        DeletionFile::deserialize(&data)?.bitmap
    } else {
        RoaringBitmap::new()  // 空 bitmap，无删除
    };
    
    // 读取数据
    let reader = FragmentReader::open(self.store.clone(), fragment).await?;
    let batches = reader.read_all().await?;
    
    // 应用删除过滤
    Ok(batches.into_iter().map(move |batch| {
        filter_batch_by_deletion(&batch, &deletion_bitmap)
    }))
}

fn filter_batch_by_deletion(
    batch: &RecordBatch,
    deletion_bitmap: &RoaringBitmap,
) -> RecordBatch {
    let mut valid_indices = Vec::new();
    
    for i in 0..batch.num_rows() {
        if !deletion_bitmap.contains(i as u32) {
            valid_indices.push(i);
        }
    }
    
    batch.take(&valid_indices).unwrap()
}
```

---

## 💡 第四部分：Fragment 的实际应用

### 场景 1：时间序列数据

```python
import lance
from datetime import datetime

# 创建时间序列数据集，按天分片
dataset = lance.open("metrics.lance")

# 每天一个 Fragment
# 2024-01-01: Fragment 0
# 2024-01-02: Fragment 1
# ...
# 2024-12-31: Fragment 365

# 查询最近 7 天的数据
results = dataset.scan() \
    .where("timestamp >= '2024-12-25'") \
    .to_pandas()

# Lance 自动：
# 1. 识别涉及的 Fragment（只读 Fragment 359-365）
# 2. 并行读取 7 个 Fragment
# 3. 合并结果
```

### 场景 2：多源数据合并

```python
# 源 1：历史数据（1000万行）
historical = lance.open("historical_data.lance")

# 源 2：新增数据（100万行）
new_data = pd.DataFrame({...})

# 合并：创建新的 Fragment，添加到 Manifest
dataset = lance.open("merged.lance")
dataset.add(new_data)  # 创建新 Fragment

# 结果：
# Fragment 0: 历史数据（1000万行）
# Fragment 1: 新增数据（100万行）
```

### 场景 3：数据删除

```python
# 删除某个用户的所有数据
user_id_to_delete = 12345

dataset = lance.open("users.lance")

# 找出该用户的所有行
rows = dataset.scan() \
    .where(f"user_id = {user_id_to_delete}") \
    .to_pandas()

row_ids = rows.index.tolist()

# 删除这些行
dataset.delete_rows(row_ids)

# 内部操作：
# - 创建删除 bitmap
# - 写入删除文件
# - 更新 Manifest
# - 扫描时自动过滤这些行
```

---

## 📊 性能特征

### Fragment 数量的影响

```
查询性能对比：

1000 万行数据集

配置 A：1 个 Fragment（1GB）
- 扫描时间：1000ms（加载 1GB）
- 缓存空间：1GB
- 删除成本：重写整个文件（高）

配置 B：10 个 Fragment（100MB 每个）
- 扫描时间：100ms（并行读 10 个）
- 缓存空间：100MB（单个缓存）
- 删除成本：只修改相关 Fragment（低）

配置 C：100 个 Fragment（10MB 每个）
- 扫描时间：50ms（并行读 100 个）
- 缓存空间：10MB（极小）
- 元数据开销：大
- 文件管理复杂度：高

推荐：10-50 个 Fragment
```

---

## 📚 总结

Fragment 是 Lance 的分片管理核心：

1. **灵活分片**：支持大数据集的高效管理
2. **版本间共享**：零拷贝的版本控制
3. **独立删除**：删除不影响其他 Fragment
4. **并行处理**：充分利用多核能力
5. **缓存友好**：支持精细粒度的缓存

在下一章中，我们将讨论 Scanner 和查询执行，即如何高效地从这些 Fragment 中读取数据。
