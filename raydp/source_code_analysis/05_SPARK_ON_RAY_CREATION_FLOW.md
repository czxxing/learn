# RayDP项目中Spark在Ray中运行的创建流程分析

## 概述

本文详细分析RayDP项目如何实现Spark在Ray环境中运行的完整创建流程。RayDP通过巧妙的架构设计，将Spark的计算和资源管理能力与Ray的分布式调度能力相结合，实现了Spark-on-Ray的运行模式。

## 核心组件

### 1. Python层组件
- `context.py`: 提供用户API入口点
- `spark/ray_cluster.py`: Spark集群管理器
- `spark/ray_cluster_master.py`: Spark应用主节点管理器

### 2. Java层组件
- `AppMasterEntryPoint.java`: Java应用主节点入口点
- `RayAppMaster.java`: Spark应用主节点
- `RayCoarseGrainedSchedulerBackend.java`: Spark调度后端
- `RayExecutorUtils.java`: Ray执行器工具类

### 3. 跨语言通信组件
- Py4J: Python和Java之间的通信桥梁
- Ray Java Client: Ray和Java之间的通信

## 创建流程详细分析

### 第一步：用户API调用

用户通过`init_spark`函数启动Spark集群：

```python
import raydp

# 初始化Ray（如果尚未初始化）
if not ray.is_initialized():
    ray.init()

# 创建Spark集群
spark = raydp.init_spark(
    app_name="MySparkApp",
    num_executors=2,
    executor_cores=2,
    executor_memory="2G",
    configs={"spark.sql.adaptive.enabled": "true"}
)
```

### 第二步：上下文初始化 (`context.py`)

#### 2.1 全局锁检查
```python
with _spark_context_lock:
    global _global_spark_context
    if _global_spark_context is None:
        # 创建新的Spark上下文
        _global_spark_context = _SparkContext(...)
    else:
        raise Exception("The spark environment has inited.")
```

#### 2.2 创建SparkContext实例
```python
def __init__(self, ...):
    # 初始化配置参数
    self._app_name = app_name
    self._num_executors = num_executors
    self._executor_cores = executor_cores
    self._executor_memory = executor_memory
    # ...
    
    # 初始化集群和会话引用
    self._spark_cluster: Optional[SparkCluster] = None
    self._spark_session: Optional[SparkSession] = None
```

### 第三步：准备Placement Group资源 (`context.py`)

```python
def _prepare_placement_group(self):
    if self._placement_group_strategy is not None:
        # 解析内存大小
        if isinstance(self._executor_memory, str):
            memory = parse_memory_size(self._executor_memory)
        else:
            memory = self._executor_memory
            
        # 为每个executor创建资源bundle
        bundles = []
        for _ in range(self._num_executors):
            bundles.append({
                "CPU": self._executor_cores,
                "memory": memory
            })
        
        # 创建Ray Placement Group
        pg = ray.util.placement_group(bundles, strategy=self._placement_group_strategy)
        ray.get(pg.ready())
        
        # 将Placement Group信息添加到Spark配置中
        self._configs[self._PLACEMENT_GROUP_CONF] = self._placement_group.id.hex()
```

### 第四步：创建Spark集群 (`spark/ray_cluster.py`)

#### 4.1 初始化SparkCluster
```python
def __init__(self, ...):
    super().__init__(None)  # 继承自Cluster基类
    self._app_name = app_name
    self._spark_master = None
    # ...
    self._prepare_spark_configs()  # 准备Spark配置
    self._set_up_master(resources=self._get_master_resources(self._configs), kwargs=None)  # 设置主节点
```

#### 4.2 准备Spark配置
```python
def _prepare_spark_configs(self):
    # 设置基本Spark配置
    self._configs["spark.executor.instances"] = str(self._num_executors)
    self._configs["spark.executor.cores"] = str(self._executor_cores)
    self._configs["spark.executor.memory"] = str(self._executor_memory)
    
    # 设置驱动节点IP（非macOS）
    if platform.system() != "Darwin":
        driver_node_ip = ray.util.get_node_ip_address()
        if "spark.driver.host" not in self._configs:
            self._configs["spark.driver.host"] = str(driver_node_ip)
            self._configs["spark.driver.bindAddress"] = str(driver_node_ip)
    
    # 设置类路径
    raydp_cp = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../jars/*"))
    ray_cp = os.path.abspath(os.path.join(os.path.dirname(ray.__file__), "jars/*"))
    spark_home = os.environ.get("SPARK_HOME", os.path.dirname(pyspark.__file__))
    spark_jars_dir = os.path.abspath(os.path.join(spark_home, "jars/*"))
    
    # 设置Java agent
    raydp_agent_path = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                                    "../../jars/raydp-agent*.jar"))
    raydp_agent_jar = glob.glob(raydp_agent_path)[0]
    self._configs[SPARK_JAVAAGENT] = raydp_agent_jar
    
    # 设置日志配置
    self._configs[SPARK_RAY_LOG4J_FACTORY_CLASS_KEY] = versions.RAY_LOG4J_VERSION
```

#### 4.3 设置Spark主节点
```python
def _set_up_master(self, resources: Dict[str, float], kwargs: Dict[Any, Any]):
    spark_master_name = self._app_name + RAYDP_SPARK_MASTER_SUFFIX
    
    if resources:
        # 提取CPU和内存资源
        num_cpu = resources.get("CPU", 1)
        memory = resources.pop("memory", None)
        
        # 创建Ray actor作为Spark主节点
        self._spark_master_handle = RayDPSparkMaster.options(
            name=spark_master_name,
            num_cpus=num_cpu,
            memory=memory,
            resources=resources
        ).remote(self._configs)
    else:
        self._spark_master_handle = RayDPSparkMaster.options(name=spark_master_name) \
            .remote(self._configs)
    
    # 启动主节点
    ray.get(self._spark_master_handle.start_up.remote())
```

### 第五步：启动RayDPSparkMaster (`spark/ray_cluster_master.py`)

#### 5.1 RayDPSparkMaster初始化
```python
@ray.remote
class RayDPSparkMaster:
    def __init__(self, configs):
        self._gateway = None
        self._app_master_java_bridge = None
        self._host = None
        self._started_up = False
        self._configs = configs
        self._spark_home = None
        self._objects = {}
        self._actor_id = None
```

#### 5.2 启动Java网关
```python
def start_up(self, popen_kwargs=None):
    if self._started_up:
        logger.warning("The RayClusterMaster has started already. Do not call it twice")
        return
    
    # 准备JVM类路径
    extra_classpath = os.pathsep.join(self._prepare_jvm_classpath())
    
    # 启动Java网关
    self._gateway = self._launch_gateway(extra_classpath, popen_kwargs)
    
    # 获取应用主节点Java桥接器
    self._app_master_java_bridge = self._gateway.entry_point.getAppMasterBridge()
    
    # 生成Ray配置并传递给Java应用主节点
    ray_properties = self._generate_ray_configs()
    self._gateway.jvm.org.apache.spark.deploy.raydp.RayAppMaster.setProperties(ray_properties)
    
    # 获取当前节点IP
    self._host = ray.util.get_node_ip_address()
    
    # 创建应用主节点
    self._create_app_master(extra_classpath)
    
    self._started_up = True
```

#### 5.3 启动Java进程
```python
def _launch_gateway(self, class_path, popen_kwargs=None):
    env = dict(os.environ)
    
    # 构建Java命令
    command = ["java"]
    
    # 添加额外Java选项
    if RAYDP_APPMASTER_EXTRA_JAVA_OPTIONS in self._configs:
        extra_opts = self._configs[RAYDP_APPMASTER_EXTRA_JAVA_OPTIONS].strip()
        if extra_opts:
            command.extend(shlex.split(extra_opts))
    
    # 添加Java agent和其他JVM选项
    command.append("-javaagent:" + self._configs[SPARK_JAVAAGENT])
    command.append("-Dray.logging.dir=" + logging_dir)
    command.append("-D" + SPARK_RAY_LOG4J_FACTORY_CLASS_KEY + "=" + self._configs[SPARK_RAY_LOG4J_FACTORY_CLASS_KEY])
    command.append("-cp")
    command.append(class_path)
    command.append("org.apache.spark.deploy.raydp.AppMasterEntryPoint")  # Java主类
    
    # 创建临时目录存储连接信息
    conn_info_dir = tempfile.mkdtemp()
    try:
        fd, conn_info_file = tempfile.mkstemp(dir=conn_info_dir)
        os.close(fd)
        os.unlink(conn_info_file)
        env["_RAYDP_APPMASTER_CONN_INFO_PATH"] = conn_info_file
        
        # 启动Java进程
        popen_kwargs = {} if popen_kwargs is None else popen_kwargs
        popen_kwargs["stdin"] = PIPE
        popen_kwargs["env"] = env
        
        def preexec_func():
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        popen_kwargs["preexec_fn"] = preexec_func
        proc = Popen(command, **popen_kwargs)
        
        # 等待连接信息文件出现
        while not proc.poll() and not os.path.isfile(conn_info_file):
            time.sleep(0.1)
        
        # 读取端口号
        with open(conn_info_file, "rb") as info:
            length = info.read(4)
            if not length:
                raise EOFError
            gateway_port = struct.unpack("!i", length)[0]
    
    finally:
        shutil.rmtree(conn_info_dir)
    
    # 创建Py4J网关
    gateway = JavaGateway(gateway_parameters=GatewayParameters(
        port=gateway_port, auto_convert=True))
    gateway.proc = proc
    return gateway
```

#### 5.4 生成Ray集群配置
```python
def _generate_ray_configs(self):
    assert ray.is_initialized()
    options = {}
    
    # 获取Ray节点信息
    node = ray._private.worker._global_node
    options["ray.run-mode"] = "CLUSTER"
    options["ray.node-ip"] = node.node_ip_address
    options["ray.address"] = node.address
    options["ray.logging.dir"] = node.get_logs_dir_path()
    options["ray.session-dir"] = node.get_session_dir_path()
    options["ray.raylet.node-manager-port"] = node.node_manager_port
    options["ray.raylet.socket-name"] = node.raylet_socket_name
    options["ray.object-store.socket-name"] = node.plasma_store_socket_name
    options["ray.job.namespace"] = ray.get_runtime_context().namespace
    
    return json.dumps(options)
```

### 第六步：Java端启动 (`AppMasterEntryPoint.java`)

#### 6.1 Java应用主节点入口
```java
public class AppMasterEntryPoint {
    public static void main(String[] args) throws Exception {
        // 从环境变量获取连接信息路径
        String connInfoPath = System.getenv("_RAYDP_APPMASTER_CONN_INFO_PATH");
        
        // 创建Py4J网关服务器
        GatewayServer server = new GatewayServer(null, 0,
            new InetSocketAddress(InetAddress.getLocalHost(), 0),
            GatewayServer.DEFAULT_CONNECT_TIMEOUT,
            GatewayServer.DEFAULT_READ_TIMEOUT,
            new SparkAppGatewayEntry());
        
        server.startup();
        
        // 将端口写入连接信息文件
        try (RandomAccessFile file = new RandomAccessFile(connInfoPath, "rw");
             FileChannel channel = file.getChannel()) {
            // 写入端口号
            byte[] portBytes = ByteBuffer.allocate(4).putInt(server.getListeningPort()).array();
            channel.write(ByteBuffer.wrap(portBytes));
        }
        
        // 初始化RayAppMaster
        RayAppMaster appMaster = new RayAppMaster();
        appMaster.initialize();
    }
}
```

#### 6.2 RayAppMaster初始化
```java
public class RayAppMaster {
    public void initialize() {
        // 从系统属性获取Ray配置
        Properties props = System.getProperties();
        String rayAddress = props.getProperty("ray.address");
        String nodeIp = props.getProperty("ray.node-ip");
        
        // 初始化Ray运行时
        Ray.init(new RayConfig()
            .setRedisAddress(rayAddress)
            .setNodeIpAddress(nodeIp));
        
        // 设置Spark外部集群管理器
        SparkConf conf = new SparkConf();
        conf.set("spark.master", "ray");  // 使用Ray作为主节点
        conf.set("spark.submit.deployMode", "cluster");
        
        // 创建Spark环境
        initializeSparkEnvironment(conf);
    }
}
```

### 第七步：Spark调度后端集成 (`RayCoarseGrainedSchedulerBackend.java`)

#### 7.1 自定义调度后端
```java
public class RayCoarseGrainedSchedulerBackend extends CoarseGrainedSchedulerBackend {
    @Override
    public void start() {
        // 请求Ray资源来启动Spark执行器
        for (int i = 0; i < totalExpectedExecutors; i++) {
            requestExecutorOnRay();
        }
    }
    
    private void requestExecutorOnRay() {
        // 创建Ray actor作为Spark执行器
        RayExecutor actor = Ray.actor(RayExecutor.class)
            .setResource("CPU", executorCores)
            .setMemory(executorMemory)
            .remote();
        
        // 获取actor引用并注册到Spark调度器
        RayExecutorRef executorRef = actor.get();
        registerExecutorWithSpark(executorRef);
    }
}
```

#### 7.2 Ray执行器实现 (`RayExecutor.java`)
```java
@RayRemote
public class RayExecutor {
    public void launchTask(TaskDescription taskDesc) {
        // 在Ray actor中执行Spark任务
        Executor executor = new Executor(...);
        executor.launchTask(taskDesc);
    }
    
    public void killTask(TaskDescription taskDesc) {
        // 杀死指定的任务
    }
}
```

### 第八步：创建SparkSession (`context.py`)

#### 8.1 获取SparkSession
```python
def get_spark_session(self) -> SparkSession:
    if self._spark_session is not None:
        return self._spark_session
    
    spark_builder = SparkSession.builder
    # 应用所有配置
    for k, v in self._configs.items():
        spark_builder.config(k, v)
    
    if self._enable_hive:
        spark_builder.enableHiveSupport()
    
    # 设置应用名称和主节点URL
    self._spark_session = spark_builder \
        .appName(self._app_name) \
        .master(self.get_cluster_url()) \
        .getOrCreate()
    
    return self._spark_session
```

### 第九步：建立Spark与Ray的连接 (`context.py`)

```python
def connect_spark_driver_to_ray(self):
    # 获取Ray配置
    jvm_properties_ref = self._spark_master_handle._generate_ray_configs.remote()
    jvm_properties = ray.get(jvm_properties_ref)
    
    # 设置Java虚拟机属性
    jvm = self._spark_session._jvm
    jvm.org.apache.spark.deploy.raydp.RayAppMaster.setProperties(jvm_properties)
    
    # 连接到Ray对象存储
    jvm.org.apache.spark.sql.raydp.ObjectStoreWriter.connectToRay()
```

## 关键技术实现

### 1. 资源管理集成
- **Ray Placement Groups**: 用于预分配Spark执行器所需的资源
- **Ray Actor Resources**: 为每个Spark执行器分配独立的Ray actor
- **动态资源分配**: 支持根据需求动态创建和销毁执行器

### 2. 跨语言通信
- **Py4J**: Python和Java之间的双向通信
- **Ray Java Client**: Java代码与Ray集群的通信
- **JVM Properties**: 传递Ray集群配置给Java进程

### 3. 执行器管理
- **Ray Actors as Executors**: 每个Spark执行器运行在独立的Ray actor中
- **资源隔离**: 每个执行器有独立的CPU、内存资源
- **生命周期管理**: 与Ray的actor生命周期一致

### 4. 调度集成
- **自定义Scheduler Backend**: 替换Spark的默认调度后端
- **资源感知调度**: 利用Ray的资源调度能力
- **故障恢复**: 利用Ray的容错机制

## 创建流程图

```
用户调用init_spark()
         ↓
检查Ray是否初始化，如未初始化则启动
         ↓
获取全局锁，创建_SparkContext实例
         ↓
准备Placement Group资源（如需要）
         ↓
创建SparkCluster实例
         ↓
准备Spark配置（executor数量、内存、类路径等）
         ↓
创建RayDPSparkMaster actor
         ↓
启动Java网关进程(AppMasterEntryPoint)
         ↓
生成Ray集群配置并传递给Java
         ↓
创建Spark应用主节点
         ↓
获取集群URL
         ↓
创建SparkSession，连接到Ray集群
         ↓
建立Spark驱动与Ray的连接
         ↓
返回SparkSession给用户使用
```

## 优势分析

### 1. 简化部署
- 无需独立的Spark集群
- 利用现有的Ray集群
- 自动资源管理

### 2. 资源效率
- 统一的资源调度
- 更好的资源利用率
- 支持混合工作负载

### 3. 弹性扩展
- 动态扩缩容
- 与Ray autoscaler集成
- 支持多租户

### 4. 生态整合
- 与Ray生态系统无缝集成
- 支持Ray的分布式训练
- 统一的监控和管理

## 总结

RayDP通过精巧的架构设计，成功地将Spark运行在Ray之上。其核心创新在于：

1. **分层抽象**：Python层提供用户接口，Java层实现Spark逻辑
2. **跨语言通信**：利用Py4J和Ray Java Client实现高效通信
3. **资源融合**：将Spark的计算模型映射到Ray的资源模型
4. **调度集成**：自定义调度后端实现与Ray的深度集成

这种设计使得用户可以在Ray集群上运行Spark应用，享受Ray的资源管理和调度优势，同时保持Spark编程模型的熟悉性。

## 补充章节：Spark在Ray中运行的原理分析

### 1. 架构原理

#### 1.1 Spark与Ray的架构对比

**传统Spark架构**：
- Master/Slave架构
- 独立的资源管理器（Standalone/YARN/Kubernetes）
- Spark Driver和Executor独立部署
- 静态资源分配

**RayDP中的Spark架构**：
- 基于Ray的Actor模型
- Ray作为统一的资源管理器
- Spark组件作为Ray Actor运行
- 动态资源分配

#### 1.2 核心架构组件

```
┌─────────────────────────────────────────────────────────────┐
│                    Ray Cluster                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Ray Driver     │  │ Ray AppMaster  │  │ Ray Worker  │ │
│  │  (Spark Driver) │  │  (Spark AM)    │  │ (Spark Exe) │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│         │                      │                  │        │
│         └──────────────────────┼──────────────────┘        │
│                                │                           │
│                    ┌───────────▼───────────┐               │
│                    │   Ray Object Store    │               │
│                    │   (Data Exchange)     │               │
│                    └───────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 2. 运行原理详解

#### 2.1 Spark Driver在Ray中的运行原理

**传统模式**：
- Spark Driver运行在独立进程中
- 与集群管理器通信获取资源
- 直接与Executor通信

**RayDP模式**：
- Spark Driver作为Ray Driver运行
- 通过Ray的资源管理获取计算资源
- 通过Ray的通信机制与Executor通信

**实现机制**：
1. **Driver启动**：Python代码调用`raydp.init_spark()`创建SparkSession
2. **资源请求**：Driver向Ray集群请求计算资源
3. **通信建立**：建立Driver与Ray集群的通信通道

#### 2.2 Spark Executor在Ray中的运行原理

**核心概念**：Ray Actor作为Spark Executor容器

**实现步骤**：
1. **Actor创建**：RayDP为每个Spark Executor创建一个Ray Actor
2. **资源分配**：每个Actor获得独立的CPU、内存资源
3. **Executor启动**：在Actor内部启动Spark Executor进程
4. **任务执行**：Spark Task在Actor内执行

**代码实现原理**：
```java
// 伪代码：Ray Actor作为Executor容器
@RayRemote
public class RayExecutorContainer {
    private Executor sparkExecutor;  // Spark执行器实例
    
    public void launchTask(TaskDescription taskDesc) {
        // 在当前Actor中执行Spark任务
        sparkExecutor.launchTask(taskDesc);
    }
    
    public void killTask(TaskDescription taskDesc) {
        sparkExecutor.killTask(taskDesc);
    }
}
```

#### 2.3 调度机制原理

**传统调度**：
- Spark CoarseGrainedSchedulerBackend
- 直接管理Executor生命周期
- 通过RPC与Executor通信

**RayDP调度**：
- 自定义RayCoarseGrainedSchedulerBackend
- 通过Ray管理Actor生命周期
- 通过Ray通信机制与Executor通信

**调度流程**：
```
Spark Driver
     ↓ (请求Executor)
Ray Scheduler
     ↓ (创建Actor)
Ray Executor Actor
     ↓ (启动Spark Executor)
Spark Executor Process
     ↓ (执行Task)
Ray Object Store (数据交换)
```

### 3. 资源管理原理

#### 3.1 资源分配机制

**Placement Groups**：
- 预分配资源组，确保Executor资源可用
- 支持不同的资源分配策略（STRICT_PACK/STRICT_SPREAD等）
- 保证资源的确定性分配

**资源配置**：
```python
# 用户指定的资源需求
num_executors = 4
executor_cores = 2
executor_memory = "4GB"

# 转换为Ray资源需求
bundles = [
    {"CPU": 2, "memory": 4*1024*1024*1024},  # 每个executor的资源
    {"CPU": 2, "memory": 4*1024*1024*1024},
    {"CPU": 2, "memory": 4*1024*1024*1024},
    {"CPU": 2, "memory": 4*1024*1024*1024}
]
placement_group = ray.util.placement_group(bundles, strategy="STRICT_SPREAD")
```

#### 3.2 资源隔离机制

**进程级隔离**：
- 每个Ray Actor运行在独立的进程中
- 操作系统级别的资源隔离

**网络隔离**：
- 每个Actor有独立的网络标识
- 避免网络资源竞争

**内存隔离**：
- Ray的内存管理机制
- 防止内存溢出影响其他组件

### 4. 数据交换原理

#### 4.1 Ray对象存储

**对象存储集成**：
- Spark DataFrame ↔ Ray Dataset 的转换
- 零拷贝数据交换
- 高效的序列化机制

**转换接口**：
```python
# Spark DataFrame to Ray Dataset
ray_ds = ray.data.from_spark(spark_df)

# Ray Dataset to Spark DataFrame  
spark_df = ray_ds.to_spark()
```

#### 4.2 数据本地性优化

**位置感知调度**：
- Ray调度器了解数据位置
- 优先将计算调度到数据所在的节点

**缓存机制**：
- Ray对象存储的缓存
- Spark的缓存与Ray缓存的协同

### 5. 通信机制原理

#### 5.1 内部通信

**Python-Java通信**：
- Py4J网关机制
- 远程方法调用
- 高效的数据传输

**Java-Java通信**：
- Spark内部RPC机制
- 保持原有的通信模式

#### 5.2 跨组件通信

**Ray Actor通信**：
- Ray的分布式Actor通信
- 异步消息传递
- 高吞吐量低延迟

### 6. 容错机制原理

#### 6.1 Ray的容错机制

**Actor容错**：
- Ray Actor的重启机制
- 状态恢复（如果启用了checkpoint）

**节点容错**：
- Ray集群的节点故障检测
- 自动任务迁移

#### 6.2 Spark的容错机制

**RDD容错**：
- RDD血统信息（Lineage）
- 任务重新执行机制

**集成容错**：
- Ray的节点容错 + Spark的RDD容错
- 双重容错保障

### 7. 性能优化原理

#### 7.1 资源利用率优化

**弹性伸缩**：
- 根据工作负载动态调整Executor数量
- 与Ray Autoscaler集成

**资源共享**：
- Ray集群中的资源共享
- 混合工作负载支持

#### 7.2 数据处理优化

**流水线执行**：
- Spark Stage的流水线执行
- 减少中间数据落地

**内存管理**：
- Ray对象存储的内存管理
- Spark内存管理的优化

### 8. 优势与挑战

#### 8.1 架构优势

**统一资源管理**：
- 单一资源池，避免资源碎片
- 统一的资源调度策略

**简化运维**：
- 单一集群管理
- 统一的监控和调试

**生态集成**：
- 与Ray生态系统无缝集成
- 支持复杂的AI/ML工作流

#### 8.2 技术挑战

**复杂性管理**：
- 两套系统的集成复杂性
- 调试和故障排查难度

**性能平衡**：
- 跨系统通信开销
- 资源调度的精细化控制

**兼容性维护**：
- 多版本Spark的兼容
- Ray版本的演进适配

### 9. 应用场景分析

#### 9.1 数据处理与AI集成

**端到端Pipeline**：
```
数据预处理(Spark) → 模型训练(Ray Train) → 模型服务(Ray Serve)
```

**数据共享**：
- Spark处理的数据直接供给AI框架
- 避免数据复制和转换开销

#### 9.2 混合工作负载

**批处理 + 实时处理**：
- Spark SQL进行批处理
- Ray实时推理服务
- 统一资源池管理

通过以上原理分析可以看出，RayDP通过将Spark运行在Ray之上，实现了资源的统一管理和生态系统的深度融合，为构建现代数据AI管道提供了强大的基础设施支持。