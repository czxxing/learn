## Curvine客户端与Master、Worker交互的完整架构

### 1. 整体交互架构

Curvine采用**主从架构**，客户端通过RPC与Master和Worker进行交互：

- **Master节点**：负责元数据管理、负载均衡、任务调度
- **Worker节点**：负责数据存储和实际的数据读写操作
- **客户端**：通过统一的API接口与集群交互

### 2. 核心交互组件

#### 2.1 FsClient - 文件系统客户端
<mcsymbol name="FsClient" filename="fs_client.rs" path="/home/czx/RustroverProjects/curvine/curvine-client/src/file/fs_client.rs" startline="32" type="class">FsClient</mcsymbol>是客户端与Master/Worker交互的核心组件：

```rust
pub struct FsClient {
    context: Arc<FsContext>,
    connector: Arc<ClusterConnector>,
}
```

#### 2.2 ClusterConnector - 集群连接器
负责管理与Master和Worker的RPC连接，基于ORPC框架实现。

### 3. 客户端与Master的交互

#### 3.1 元数据操作
客户端通过Master进行所有元数据操作：

```rust
// 创建目录
pub async fn mkdir(&self, path: &Path, opts: MkdirOpts) -> FsResult<bool>

// 创建文件
pub async fn create(&self, path: &Path, create_parent: bool) -> FsResult<FileStatus>

// 获取文件状态
pub async fn file_status(&self, path: &Path) -> FsResult<FileStatus>

// 删除文件
pub async fn delete(&self, path: &Path, recursive: bool) -> FsResult<()>

// 重命名文件
pub async fn rename(&self, src: &Path, dst: &Path) -> FsResult<bool>

// 列出目录内容
pub async fn list_status(&self, path: &Path) -> FsResult<Vec<FileStatus>>
```

#### 3.2 块管理
Master负责块的分配和管理：

```rust
// 添加数据块
pub async fn add_block(&self, path: &Path, previous: Option<CommitBlock>, 
                      local_addr: &ClientAddress) -> FsResult<LocatedBlock>

// 完成文件写入
pub async fn complete_file(&self, path: &Path, len: i64, 
                          last: Option<CommitBlock>) -> FsResult<()>

// 获取块位置信息
pub async fn get_block_locations(&self, path: &Path) -> FsResult<FileBlocks>
```

#### 3.3 任务管理
通过JobMasterClient进行任务管理：

```rust
// 提交加载任务
pub async fn submit_load_job(&self, command: LoadJobCommand) -> FsResult<LoadJobResult>

// 获取任务状态
pub async fn get_job_status(&self, job_id: impl AsRef<str>) -> FsResult<JobStatus>

// 取消任务
pub async fn cancel_job(&self, job_id: impl AsRef<str>) -> FsResult<()>
```

### 4. 客户端与Worker的交互

#### 4.1 数据读写操作
客户端直接与Worker进行数据读写：

```rust
// 在block_client.rs中实现
pub async fn read_block(&self, block: &LocatedBlock, offset: i64, 
                       len: i64, buf: &mut [u8]) -> FsResult<()>

pub async fn write_block(&self, block: &LocatedBlock, offset: i64, 
                        data: &[u8]) -> FsResult<()>
```

#### 4.2 数据流处理
Worker负责实际的数据存储和检索：

- **读操作**：客户端从Worker读取数据块
- **写操作**：客户端向Worker写入数据块
- **数据校验**：Worker负责数据完整性和一致性

### 5. RPC通信机制

#### 5.1 统一的RPC接口
所有操作都通过统一的RPC接口：

```rust
pub async fn rpc<T, R>(&self, code: RpcCode, header: T) -> FsResult<R>
where
    T: PMessage + Default,
    R: PMessage + Default,
{
    self.connector
        .proto_rpc::<T, R, FsError>(code, header)
        .await
}
```

#### 5.2 RPC代码定义
在<mcfolder name="curvine-common" path="/home/czx/RustroverProjects/curvine/curvine-common"></mcfolder>中定义了所有RPC操作码：

```rust
pub enum RpcCode {
    // 文件系统操作
    Mkdir = 1,
    CreateFile = 2,
    FileStatus = 3,
    Delete = 4,
    Rename = 5,
    ListStatus = 6,
    
    // 块管理操作
    AddBlock = 10,
    CompleteFile = 11,
    GetBlockLocations = 12,
    
    // 任务管理
    SubmitJob = 20,
    GetJobStatus = 21,
    CancelJob = 22,
    
    // 集群管理
    GetMasterInfo = 30,
    MetricsReport = 31,
}
```

### 6. 交互流程示例

#### 6.1 文件写入流程
1. **客户端**调用`create()`方法创建文件（与Master交互）
2. **Master**分配初始数据块，返回LocatedBlock信息
3. **客户端**调用`add_block()`获取更多数据块（与Master交互）
4. **客户端**直接与Worker进行数据写入
5. **客户端**调用`complete_file()`完成文件写入（与Master交互）

#### 6.2 文件读取流程
1. **客户端**调用`file_status()`获取文件信息（与Master交互）
2. **Master**返回文件的块位置信息
3. **客户端**根据块位置信息直接与对应的Worker进行数据读取
4. **Worker**返回数据给客户端

#### 6.3 任务执行流程
1. **客户端**通过JobMasterClient提交任务（与Master交互）
2. **Master**将任务分发给合适的Worker
3. **Worker**执行任务并报告进度
4. **客户端**通过Master查询任务状态

### 7. 性能优化特性

#### 7.1 连接复用
- 使用<mcsymbol name="ClusterConnector" filename="cluster_connector.rs" path="/home/czx/RustroverProjects/curvine/orpc/src/client/cluster_connector.rs" startline="1" type="class">ClusterConnector</mcsymbol>管理连接池
- 支持连接复用，减少连接建立开销

#### 7.2 负载均衡
- Master负责Worker的负载均衡
- 客户端自动选择最优的Worker进行数据操作

#### 7.3 错误恢复
- 自动重试失败的RPC调用
- Worker故障时自动切换到其他可用Worker

### 8. 配置管理

客户端配置在<mcfolder name="curvine-common" path="/home/czx/RustroverProjects/curvine/curvine-common/src/conf"></mcfolder>中定义：

```toml
[client]
master_addrs = [
    { hostname = "localhost", port = 8995 },
    { hostname = "cv-master-1", port = 8995 },
    { hostname = "cv-master-2", port = 8995 }
]
```

### 9. 总结

Curvine客户端与Master、Worker的交互体现了**分层架构**的设计思想：

1. **控制平面**（与Master交互）：负责元数据管理、任务调度
2. **数据平面**（与Worker交互）：负责实际的数据读写操作
3. **统一接口**：通过FsClient提供简洁的API

这种设计使得：
- **高可用性**：Master集群提供故障转移
- **高性能**：数据操作直接与Worker进行，减少中间环节
- **易用性**：统一的API接口简化客户端开发
- **可扩展性**：支持动态添加Worker节点

客户端通过智能的路由和负载均衡机制，确保与Master和Worker的高效交互，为上层应用提供稳定可靠的分布式存储服务。
        
