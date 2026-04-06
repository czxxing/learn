# 第二十六章：RowID 系统

## 🎯 核心概览

RowID 是 Lance 中每一行数据的唯一标识符。RowID 系统负责生成、管理和映射行号，支持行级的增删改查操作。RowID 通常是一个 64 位整数，格式为 `fragment_id | row_number`。

---

## 🔢 RowID 生成

### RowID 的结构

```
┌─────────────────────────────────────────────┐
│ Row ID (u64)                                 │
├──────────────────────┬──────────────────────┤
│ Fragment ID (u32)    │ Row Number (u32)     │
│ 32 bits              │ 32 bits              │
└──────────────────────┴──────────────────────┘

例：
- Fragment 0, Row 0: 0x0000_0000_0000_0000
- Fragment 0, Row 1: 0x0000_0000_0000_0001
- Fragment 1, Row 0: 0x0000_0001_0000_0000
- Fragment 1, Row 1: 0x0000_0001_0000_0001
```

### RowID 生成器

```rust
pub struct RowIDGenerator {
    next_fragment_id: Arc<AtomicU32>,
    current_fragment: Arc<RwLock<u32>>,
    row_counter_per_fragment: Arc<DashMap<u32, AtomicU32>>,
}

impl RowIDGenerator {
    pub fn new() -> Self {
        Self {
            next_fragment_id: Arc::new(AtomicU32::new(0)),
            current_fragment: Arc::new(RwLock::new(0)),
            row_counter_per_fragment: Arc::new(DashMap::new()),
        }
    }
    
    // 为新 Fragment 分配 ID
    pub fn allocate_fragment_id(&self) -> u32 {
        let id = self.next_fragment_id.fetch_add(1, Ordering::SeqCst);
        self.row_counter_per_fragment.insert(id, AtomicU32::new(0));
        id
    }
    
    // 为新行生成 RowID
    pub fn generate_row_id(&self, fragment_id: u32) -> u64 {
        let counter = self.row_counter_per_fragment
            .entry(fragment_id)
            .or_insert_with(|| AtomicU32::new(0));
        
        let row_number = counter.fetch_add(1, Ordering::SeqCst);
        
        // 组合 Fragment ID 和 Row Number
        self.encode_row_id(fragment_id, row_number)
    }
    
    // 为一批行生成 RowID
    pub fn generate_row_ids(&self, fragment_id: u32, count: u32) -> Vec<u64> {
        let counter = self.row_counter_per_fragment
            .entry(fragment_id)
            .or_insert_with(|| AtomicU32::new(0));
        
        let start_row = counter.fetch_add(count, Ordering::SeqCst);
        
        (start_row..start_row + count)
            .map(|row_num| self.encode_row_id(fragment_id, row_num))
            .collect()
    }
    
    // 编码 RowID
    #[inline]
    fn encode_row_id(&self, fragment_id: u32, row_number: u32) -> u64 {
        ((fragment_id as u64) << 32) | (row_number as u64)
    }
    
    // 解码 RowID
    #[inline]
    pub fn decode_row_id(row_id: u64) -> (u32, u32) {
        let fragment_id = (row_id >> 32) as u32;
        let row_number = row_id as u32;
        (fragment_id, row_number)
    }
}
```

---

## 🗂️ 行地址映射

### 物理位置定位

RowID 需要快速映射到物理存储位置：

```rust
pub struct RowAddressMap {
    // Fragment ID -> Fragment 信息
    fragments: DashMap<u32, FragmentInfo>,
    
    // RowID -> 物理位置（文件、偏移）
    address_cache: Arc<RwLock<LruCache<u64, RowAddress>>>,
}

pub struct FragmentInfo {
    id: u32,
    location: String,  // 文件路径
    num_rows: u32,
    row_offset: u32,   // Fragment 内的起始行号
}

pub struct RowAddress {
    fragment_path: String,
    row_index: u32,      // Fragment 内的行索引
    byte_offset: u64,    // 文件内的字节偏移
    byte_length: u32,    // 行数据的字节长度
}

impl RowAddressMap {
    // 从 RowID 获取物理位置
    pub fn get_address(&self, row_id: u64) -> Result<RowAddress> {
        // 1. 查询缓存
        if let Some(address) = self.address_cache.read().unwrap().get(&row_id) {
            return Ok(address.clone());
        }
        
        // 2. 解析 RowID
        let (fragment_id, row_number) = RowIDGenerator::decode_row_id(row_id);
        
        // 3. 查询 Fragment 信息
        let fragment = self.fragments.get(&fragment_id)
            .ok_or("Fragment not found")?;
        
        // 4. 构建物理地址
        let address = RowAddress {
            fragment_path: fragment.location.clone(),
            row_index: row_number,
            byte_offset: self.calculate_offset(&fragment, row_number)?,
            byte_length: self.calculate_row_length(&fragment)?,
        };
        
        // 5. 缓存结果
        self.address_cache.write().unwrap()
            .put(row_id, address.clone());
        
        Ok(address)
    }
    
    fn calculate_offset(&self, fragment: &FragmentInfo, row_number: u32) -> Result<u64> {
        // 对于固定大小的行，偏移 = row_number * row_size
        // 对于可变大小的行，需要查询行长度索引
        Ok((row_number as u64) * 100)  // 假设每行 100 字节
    }
}
```

### 批量地址查询优化

```rust
impl RowAddressMap {
    // 一次性查询多个 RowID 的地址，优化 IO
    pub fn get_addresses_batch(&self, row_ids: &[u64]) -> Result<Vec<RowAddress>> {
        // 1. 按 Fragment ID 分组
        let mut by_fragment: HashMap<u32, Vec<(u64, u32)>> = HashMap::new();
        
        for &row_id in row_ids {
            let (frag_id, row_num) = RowIDGenerator::decode_row_id(row_id);
            by_fragment.entry(frag_id)
                .or_insert_with(Vec::new)
                .push((row_id, row_num));
        }
        
        // 2. 对每个 Fragment 进行批量查询
        let mut results = Vec::new();
        
        for (frag_id, rows) in by_fragment {
            let fragment = self.fragments.get(&frag_id)?;
            
            // 批量读取行长度（一次 IO）
            let row_lengths = self.read_row_lengths_batch(
                &fragment,
                rows.iter().map(|(_, row_num)| *row_num).collect::<Vec<_>>(),
            )?;
            
            // 计算地址
            for ((row_id, row_num), length) in rows.iter().zip(row_lengths) {
                results.push(RowAddress {
                    fragment_path: fragment.location.clone(),
                    row_index: *row_num,
                    byte_offset: self.calculate_offset(&fragment, *row_num)?,
                    byte_length: length,
                });
            }
        }
        
        Ok(results)
    }
}
```

---

## 🗑️ 删除标记与行可见性

### 软删除机制

Lance 使用软删除（标记而不是物理删除）以支持版本管理：

```rust
pub struct DeletionVector {
    // 位向量：0 表示存在，1 表示已删除
    bits: BitVec,
    version: u64,
    fragment_id: u32,
}

impl DeletionVector {
    pub fn new(fragment_id: u32, capacity: usize, version: u64) -> Self {
        Self {
            bits: BitVec::from_elem(capacity, false),  // 初始全部存在
            version,
            fragment_id,
        }
    }
    
    // 标记行为已删除
    pub fn delete(&mut self, row_index: u32) {
        self.bits.set(row_index as usize, true);
    }
    
    // 检查行是否已删除
    pub fn is_deleted(&self, row_index: u32) -> bool {
        self.bits.get(row_index as usize)
            .map(|b| b.clone())
            .unwrap_or(false)
    }
    
    // 获取未删除行的位掩码
    pub fn get_valid_rows_mask(&self) -> Vec<bool> {
        self.bits.iter().map(|b| !b).collect()
    }
    
    // 批量检查
    pub fn filter_deleted(&self, row_ids: &[u64]) -> Vec<u64> {
        row_ids.iter()
            .filter(|&&row_id| {
                let (_, row_num) = RowIDGenerator::decode_row_id(row_id);
                !self.is_deleted(row_num)
            })
            .copied()
            .collect()
    }
}
```

### 行可见性管理

```rust
pub struct RowVisibility {
    deletion_vectors: DashMap<u32, DeletionVector>,  // Fragment -> DeletionVector
    version: u64,
}

impl RowVisibility {
    pub fn new(version: u64) -> Self {
        Self {
            deletion_vectors: DashMap::new(),
            version,
        }
    }
    
    // 删除行
    pub fn delete_row(&self, row_id: u64) -> Result<()> {
        let (frag_id, row_num) = RowIDGenerator::decode_row_id(row_id);
        
        if let Some(mut dv) = self.deletion_vectors.get_mut(&frag_id) {
            dv.delete(row_num);
            Ok(())
        } else {
            Err("Fragment not found".into())
        }
    }
    
    // 批量删除
    pub fn delete_rows(&self, row_ids: &[u64]) -> Result<()> {
        // 按 Fragment 分组
        let mut by_fragment: HashMap<u32, Vec<u32>> = HashMap::new();
        
        for &row_id in row_ids {
            let (frag_id, row_num) = RowIDGenerator::decode_row_id(row_id);
            by_fragment.entry(frag_id)
                .or_insert_with(Vec::new)
                .push(row_num);
        }
        
        // 批量更新删除向量
        for (frag_id, row_nums) in by_fragment {
            if let Some(mut dv) = self.deletion_vectors.get_mut(&frag_id) {
                for row_num in row_nums {
                    dv.delete(row_num);
                }
            }
        }
        
        Ok(())
    }
    
    // 获取有效行
    pub fn filter_valid_rows(&self, row_ids: &[u64]) -> Vec<u64> {
        row_ids.iter()
            .filter(|&&row_id| {
                let (frag_id, row_num) = RowIDGenerator::decode_row_id(row_id);
                self.deletion_vectors
                    .get(&frag_id)
                    .map(|dv| !dv.is_deleted(row_num))
                    .unwrap_or(true)
            })
            .copied()
            .collect()
    }
}
```

---

## 🔍 RowID 应用示例

### Python API

```python
import lance

# 创建表并写入数据
table = lance.write_table(
    {
        "id": [1, 2, 3, 4, 5],
        "text": ["a", "b", "c", "d", "e"],
    },
    uri="data.lance"
)

# 获取行 ID
row_ids = table.row_ids()
print(f"All row IDs: {row_ids}")

# 删除特定行
table.delete("id == 2")  # 删除 id=2 的行

# 查看删除后的有效行
remaining_rows = table.search().to_list()
print(f"Rows after deletion: {remaining_rows}")

# 更新特定行
table.update(
    {"text": "updated"},
    where="id == 3"
)

# 查询特定行的地址信息
row_address = table.get_row_address(row_ids[0])
print(f"Row 0 address: {row_address}")
```

### 删除与恢复

```python
# 标记删除（不物理删除）
table.delete("id > 3")

# 查看已删除的行（在历史版本中存在）
historical_table = table.as_of(version=1)
all_rows = historical_table.search().to_list()
print(f"All rows in v1: {all_rows}")

# 恢复删除
table.restore(row_ids=[4, 5])
```

---

## 📊 性能优化

### RowID 缓存

```rust
pub struct RowIDCache {
    lru_cache: Arc<RwLock<LruCache<u32, Vec<u64>>>>,  // Fragment ID -> RowID 列表
}

impl RowIDCache {
    pub fn get_or_load(&self, fragment_id: u32) -> Result<Vec<u64>> {
        // 检查缓存
        if let Some(row_ids) = self.lru_cache.read().unwrap().peek(&fragment_id) {
            return Ok(row_ids.clone());
        }
        
        // 从磁盘加载
        let row_ids = self.load_from_disk(fragment_id)?;
        
        // 写入缓存
        self.lru_cache.write().unwrap()
            .put(fragment_id, row_ids.clone());
        
        Ok(row_ids)
    }
}
```

---

## 📚 总结

RowID 系统是 Lance 进行行级操作的基础：

1. **生成策略**：高效的分布式 ID 生成
2. **地址映射**：快速定位物理存储位置
3. **删除标记**：软删除支持版本管理
4. **行可见性**：多版本下的一致性视图

这些机制共同支撑了 Lance 的更新、删除、恢复等高级操作。
