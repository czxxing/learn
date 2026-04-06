# 第二十四章：Manifest 与版本管理

## 🎯 核心概览

Manifest 是 Lance 的核心版本管理机制，记录了数据集的完整历史。每次提交都会生成一个新的 Manifest，包含指向数据的指针、版本元数据和提交信息。这样使得 Lance 可以支持 **时间旅行（Time Travel）、版本回滚和分支管理**。

---

## 📋 Manifest 文件格式

### 基础结构

```protobuf
// manifest.proto

message ManifestEntry {
    uint64 version = 1;              // 版本号，单调递增
    uint64 timestamp = 2;            // 提交时间戳（毫秒）
    string commit_msg = 3;           // 提交信息
    
    // 数据指针
    repeated Fragment fragments = 4;  // 本版本的 fragments
    repeated string deleted_files = 5; // 删除的文件列表
    
    // 元数据
    Schema schema = 6;               // 表 schema
    map<string, string> metadata = 7; // 自定义元数据
    
    // 版本控制信息
    uint64 parent_version = 8;       // 父版本号（用于回滚）
    string branch = 9;               // 所在分支
    repeated string tags = 10;       // 标签列表
}

message Fragment {
    uint64 id = 1;
    uint64 physical_rows = 2;
    repeated DataFile files = 3;
    
    message DataFile {
        string path = 1;
        string format = 2;  // "lance", "parquet" 等
        uint64 size = 3;
        map<string, string> metadata = 4;
    }
}

message Manifest {
    repeated ManifestEntry versions = 1;
    string current_branch = 2;
    map<string, uint64> branches = 3;  // 分支 -> 版本号映射
    map<string, uint64> tags = 4;      // 标签 -> 版本号映射
}
```

### Manifest 存储示例

```
data.lance/
├── _manifest/
│   ├── v1_manifest  (版本1的 manifest)
│   ├── v2_manifest  (版本2的 manifest)
│   └── v3_manifest  (版本3的 manifest，当前最新)
├── fragments/
│   ├── 0/
│   │   ├── data.lance
│   │   └── indices/
│   └── 1/
│       ├── data.lance
│       └── indices/
└── _current  (指向当前版本的指针，内容：3)
```

---

## 🔢 版本号生成策略

### 版本号管理

```rust
pub struct VersionController {
    dataset_path: PathBuf,
    current_version: Arc<RwLock<u64>>,
}

impl VersionController {
    pub async fn allocate_version(&self) -> Result<u64> {
        let mut version = self.current_version.write().await;
        *version += 1;
        
        // 持久化版本号
        self.persist_current_version(*version).await?;
        
        Ok(*version)
    }
    
    pub async fn persist_current_version(&self, version: u64) -> Result<()> {
        let current_file = self.dataset_path.join("_current");
        tokio::fs::write(
            &current_file,
            version.to_string(),
        ).await?;
        Ok(())
    }
    
    pub async fn load_current_version(&self) -> Result<u64> {
        let current_file = self.dataset_path.join("_current");
        if current_file.exists() {
            let content = tokio::fs::read_to_string(&current_file).await?;
            Ok(content.trim().parse()?)
        } else {
            Ok(0)  // 首次创建时从 0 开始
        }
    }
}
```

### 版本生命周期

```
创建表
  ↓ version = 1
写入数据（10 行）
  ↓ version = 2
更新数据（修改 2 行）
  ↓ version = 3
删除数据（删除 3 行）
  ↓ version = 4
```

---

## 🌿 分支（Branch）与标签（Tag）

### 分支管理

```rust
pub struct BranchManager {
    manifest_dir: PathBuf,
}

impl BranchManager {
    // 创建新分支
    pub async fn create_branch(
        &self,
        branch_name: &str,
        from_version: u64,
    ) -> Result<()> {
        // 记录分支指向的版本
        let branch_file = self.manifest_dir.join(format!("branch_{}", branch_name));
        tokio::fs::write(&branch_file, from_version.to_string()).await?;
        Ok(())
    }
    
    // 切换分支
    pub async fn switch_branch(&self, branch_name: &str) -> Result<()> {
        let branch_file = self.manifest_dir.join(format!("branch_{}", branch_name));
        if !branch_file.exists() {
            return Err(format!("Branch not found: {}", branch_name).into());
        }
        
        let version = tokio::fs::read_to_string(&branch_file)
            .await?
            .trim()
            .parse::<u64>()?;
        
        // 更新当前分支指针
        let current_branch_file = self.manifest_dir.join("_current_branch");
        tokio::fs::write(&current_branch_file, branch_name).await?;
        
        // 更新当前版本
        let current_version_file = self.manifest_dir.join("_current_version");
        tokio::fs::write(&current_version_file, version.to_string()).await?;
        
        Ok(())
    }
    
    // 列出所有分支
    pub async fn list_branches(&self) -> Result<Vec<(String, u64)>> {
        let mut branches = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.manifest_dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();
            
            if file_name_str.starts_with("branch_") {
                let branch_name = file_name_str[7..].to_string();  // 去掉 "branch_" 前缀
                let version: u64 = tokio::fs::read_to_string(entry.path())
                    .await?
                    .trim()
                    .parse()?;
                branches.push((branch_name, version));
            }
        }
        
        Ok(branches)
    }
}
```

### 标签管理

```rust
pub struct TagManager {
    manifest_dir: PathBuf,
}

impl TagManager {
    // 创建标签（轻量级指针，指向特定版本）
    pub async fn create_tag(
        &self,
        tag_name: &str,
        version: u64,
        message: &str,
    ) -> Result<()> {
        let tag_file = self.manifest_dir.join(format!("tag_{}", tag_name));
        
        let tag_info = format!("{}\n{}", version, message);
        tokio::fs::write(&tag_file, tag_info).await?;
        
        Ok(())
    }
    
    // 获取标签指向的版本
    pub async fn get_tag_version(&self, tag_name: &str) -> Result<u64> {
        let tag_file = self.manifest_dir.join(format!("tag_{}", tag_name));
        let content = tokio::fs::read_to_string(&tag_file).await?;
        let first_line = content.lines().next().unwrap_or("0");
        Ok(first_line.parse()?)
    }
    
    // 列出所有标签
    pub async fn list_tags(&self) -> Result<Vec<(String, u64, String)>> {
        let mut tags = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.manifest_dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();
            
            if file_name_str.starts_with("tag_") {
                let tag_name = file_name_str[4..].to_string();
                let content = tokio::fs::read_to_string(entry.path()).await?;
                let mut lines = content.lines();
                let version: u64 = lines.next().unwrap_or("0").parse()?;
                let message = lines.next().unwrap_or("").to_string();
                tags.push((tag_name, version, message));
            }
        }
        
        Ok(tags)
    }
}
```

---

## ⏰ 时间旅行（Time Travel）

### 读取历史版本

```rust
pub struct Dataset {
    path: PathBuf,
    current_version: u64,
    manifest: Arc<Manifest>,
}

impl Dataset {
    // 切换到特定版本
    pub async fn checkout(&mut self, version: u64) -> Result<()> {
        // 验证版本存在
        let manifest_path = self.path.join("_manifest").join(format!("v{}_manifest", version));
        if !manifest_path.exists() {
            return Err(format!("Version {} does not exist", version).into());
        }
        
        // 加载该版本的 manifest
        let manifest = self.load_manifest(version).await?;
        
        // 更新当前版本
        self.current_version = version;
        self.manifest = Arc::new(manifest);
        
        Ok(())
    }
    
    // 加载特定版本的 manifest
    async fn load_manifest(&self, version: u64) -> Result<Manifest> {
        let manifest_path = self.path.join("_manifest").join(format!("v{}_manifest", version));
        let data = tokio::fs::read(&manifest_path).await?;
        
        // 反序列化（使用 Protobuf 或 JSON）
        let manifest = bincode::deserialize(&data)?;
        Ok(manifest)
    }
    
    // 查看历史版本列表
    pub async fn list_versions(&self) -> Result<Vec<VersionInfo>> {
        let manifest_dir = self.path.join("_manifest");
        let mut versions = Vec::new();
        let mut entries = tokio::fs::read_dir(&manifest_dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let file_name = entry.file_name();
            if let Some(version_str) = file_name
                .to_str()
                .and_then(|f| f.strip_prefix("v"))
                .and_then(|f| f.strip_suffix("_manifest"))
            {
                if let Ok(version_num) = version_str.parse::<u64>() {
                    let manifest = self.load_manifest(version_num).await?;
                    versions.push(VersionInfo {
                        version: version_num,
                        timestamp: manifest.versions[0].timestamp,
                        commit_msg: manifest.versions[0].commit_msg.clone(),
                        row_count: self.count_rows_for_version(version_num).await?,
                    });
                }
            }
        }
        
        // 按版本号降序排列
        versions.sort_by(|a, b| b.version.cmp(&a.version));
        Ok(versions)
    }
    
    async fn count_rows_for_version(&self, version: u64) -> Result<u64> {
        let manifest = self.load_manifest(version).await?;
        let total_rows: u64 = manifest.versions[0].fragments.iter()
            .map(|f| f.physical_rows)
            .sum();
        Ok(total_rows)
    }
}

pub struct VersionInfo {
    pub version: u64,
    pub timestamp: u64,
    pub commit_msg: String,
    pub row_count: u64,
}
```

### Python API 使用

```python
import lance
from datetime import datetime, timedelta

# 打开表
table = lance.open("data.lance")

# 查看所有版本
versions = table.list_versions()
for v in versions:
    timestamp = datetime.fromtimestamp(v["timestamp"] / 1000)
    print(f"v{v['version']}: {timestamp} - {v['commit_msg']} ({v['row_count']} rows)")

# 时间旅行：读取 1 小时前的数据
one_hour_ago = datetime.now() - timedelta(hours=1)
table_past = table.as_of(one_hour_ago.timestamp() * 1000)
print(f"1 小时前的行数：{table_past.count_rows()}")

# 回到特定版本
table_v3 = table.checkout(3)
print(f"版本 3 的数据：{table_v3.search(...).to_list()}")

# 分支管理
table.create_branch("experimental", from_version=5)
table.switch_branch("experimental")

# 标签管理
table.create_tag("v1.0-release", version=10, message="First release")
results = table.as_of(tag="v1.0-release").search(...).to_list()
```

---

## 🔄 Manifest 更新协议

### ACID 提交

```rust
pub struct ManifestWriter {
    dataset_path: PathBuf,
}

impl ManifestWriter {
    pub async fn commit(
        &self,
        new_fragments: Vec<Fragment>,
        deleted_fragments: Vec<u64>,
        commit_msg: &str,
    ) -> Result<u64> {
        // Step 1: 获取新版本号
        let new_version = self.allocate_version().await?;
        
        // Step 2: 构建新 manifest
        let mut new_manifest = self.load_current_manifest().await?;
        
        new_manifest.versions.push(ManifestEntry {
            version: new_version,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_millis() as u64,
            commit_msg: commit_msg.to_string(),
            fragments: new_fragments,
            deleted_files: deleted_fragments
                .iter()
                .map(|id| format!("fragments/{}", id))
                .collect(),
            parent_version: new_version - 1,
            ..Default::default()
        });
        
        // Step 3: 原子写入
        // 先写到临时文件
        let temp_path = self.dataset_path
            .join("_manifest")
            .join(format!("v{}_manifest.tmp", new_version));
        
        let serialized = bincode::serialize(&new_manifest)?;
        tokio::fs::write(&temp_path, serialized).await?;
        
        // Step 4: 原子重命名（在 POSIX 系统上是原子操作）
        let final_path = self.dataset_path
            .join("_manifest")
            .join(format!("v{}_manifest", new_version));
        
        tokio::fs::rename(&temp_path, &final_path).await?;
        
        // Step 5: 更新 _current 指针
        let current_file = self.dataset_path.join("_current");
        tokio::fs::write(&current_file, new_version.to_string()).await?;
        
        Ok(new_version)
    }
}
```

---

## 📊 版本压缩

过多的版本会占用大量存储。Lance 支持版本压缩来清理历史。

```rust
pub struct ManifestCompactor {
    dataset_path: PathBuf,
}

impl ManifestCompactor {
    pub async fn compact(&self, keep_versions: usize) -> Result<()> {
        // 1. 获取所有版本
        let versions = self.list_all_versions().await?;
        
        if versions.len() <= keep_versions {
            return Ok(());  // 无需压缩
        }
        
        // 2. 确定要删除的版本
        let to_delete = versions.len() - keep_versions;
        let oldest_kept = versions[to_delete];
        
        // 3. 删除旧 manifest 文件
        for version in &versions[..to_delete] {
            let manifest_file = self.dataset_path
                .join("_manifest")
                .join(format!("v{}_manifest", version));
            tokio::fs::remove_file(manifest_file).await?;
        }
        
        // 4. 更新 manifest 索引
        // ...
        
        Ok(())
    }
}
```

---

## 📚 总结

Manifest 与版本管理提供了 Lance 的关键特性：

1. **完整版本历史**：每次更改都记录
2. **时间旅行**：可以回到任意历史时间点
3. **分支管理**：支持多个开发分支
4. **标签管理**：标记重要版本
5. **ACID 保证**：原子提交确保一致性

这些功能使 Lance 成为一个真正的数据湖，支持完整的版本管理和审计。
