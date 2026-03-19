# 03 Java AppMaster 初始化

## 概述
本文档分析了 RayDP 中的 Java AppMaster 初始化过程，这是负责管理 Ray 集群上 Spark 执行器的核心组件。AppMaster 处理资源分配、执行器生命周期管理以及 Spark 驱动程序和执行器之间的通信。

## AppMaster Actor 创建过程

### Scala AppMaster 类定义
RayAppMaster 实现为管理 Spark 应用程序生命周期的 Scala 类：

```scala
class RayAppMaster(
    host: String,
    port: Int,
    actorExtraClasspath: String,
    sparkConf: SparkConf) extends Logging with Serializable {

  private val appId = generateApplicationId()
  private val executors = mutable.Map[String, RayTaskActor]()
  private val executorIdToWorker = mutable.Map[String, String]()
  private val pendingTasks = mutable.Queue[TaskRequest]()
  
  // 初始化内部状态
  private var isInitialized = false
  private var resourceManager: Option[ResourceManager] = None
  
  def initialize(): Unit = {
    logInfo(s"正在为应用程序初始化 RayAppMaster: $appId")
    
    // 初始化资源管理器
    resourceManager = Some(new ResourceManager(sparkConf))
    
    // 开始心跳监控
    startHeartbeatMonitoring()
    
    // 初始化通信通道
    initializeCommunicationChannels()
    
    isInitialized = true
    logInfo(s"RayAppMaster 成功为应用程序初始化: $appId")
  }
}
```

### 资源管理器初始化
AppMaster 初始化 ResourceManager 来处理 Ray 资源分配：

```scala
class ResourceManager(sparkConf: SparkConf) extends Logging {
  
  private val rayResources = mutable.Map[String, ResourceAllocation]()
  private val resourceLock = new Object()
  
  def allocateResources(request: ResourceRequest): Future[ResourceAllocation] = {
    resourceLock.synchronized {
      logDebug(s"正在处理资源请求: ${request.executorId}")
      
      // 从 Spark 配置解析资源要求
      val cpuRequirement = sparkConf.getInt(s"spark.executor.cores", 1)
      val memoryRequirement = parseMemoryRequirement(
        sparkConf.get("spark.executor.memory", "1g"))
      
      // 分配 Ray 资源
      val resources = Map(
        "CPU" -> cpuRequirement.toDouble,
        "memory" -> memoryRequirement
      )
      
      // 使用指定资源创建 Ray actor
      val actorOptions = RayActorOptions.newBuilder()
        .setResources(resources)
        .build()
      
      val allocation = ResourceAllocation(
        executorId = request.executorId,
        resources = resources,
        actorOptions = actorOptions,
        allocatedAt = System.currentTimeMillis()
      )
      
      rayResources += (request.executorId -> allocation)
      Future.successful(allocation)
    }
  }
  
  private def parseMemoryRequirement(memoryStr: String): Double = {
    val normalized = memoryStr.toLowerCase.trim
    if (normalized.endsWith("g")) {
      normalized.substring(0, normalized.length - 1).toDouble * 1024 * 1024 * 1024
    } else if (normalized.endsWith("m")) {
      normalized.substring(0, normalized.length - 1).toDouble * 1024 * 1024
    } else {
      normalized.toDouble // 如果没有单位则假设为字节
    }
  }
}
```

## 通信通道设置

### Py4J 网关集成
AppMaster 通过 Py4J 与 Python 驱动程序建立通信：

```scala
class Py4JServer(host: String, port: Int) extends Logging {
  
  private var gatewayServer: Option[GatewayServer] = None
  private val registeredCallbacks = mutable.Set[Object]()
  
  def start(): Unit = {
    logInfo(s"正在启动 Py4J 服务器于 $host:$port")
    
    val entryPoint = new AppMasterEntryPoint(this)
    val gatewayServerInstance = new GatewayServer(
      entryPoint,
      port,
      0, // 最大连接数
      GatewayServer.DEFAULT_CONNECT_TIMEOUT,
      GatewayServer.DEFAULT_READ_TIMEOUT,
      host
    )
    
    gatewayServerInstance.start()
    gatewayServer = Some(gatewayServerInstance)
    
    logInfo(s"Py4J 服务器成功启动于 $host:$port")
  }
  
  def registerCallback(obj: Object): Unit = {
    registeredCallbacks += obj
  }
  
  def stop(): Unit = {
    gatewayServer.foreach(_.shutdown())
    gatewayServer = None
  }
}

class AppMasterEntryPoint(server: Py4JServer) extends Serializable {
  
  def getAppMaster(): RayAppMaster = {
    // 返回对主 AppMaster 实例的引用
    RayAppMaster.getInstance()
  }
  
  def submitTask(taskRequest: TaskRequest): TaskResponse = {
    // 处理来自 Python 的任务提交
    RayAppMaster.getInstance().handleTaskSubmission(taskRequest)
  }
  
  def registerExecutor(executorInfo: ExecutorInfo): Boolean = {
    // 处理来自 Python 的执行器注册
    RayAppMaster.getInstance().registerExecutor(executorInfo)
  }
}
```

### 执行器管理协议
AppMaster 实现执行器管理协议：

```scala
class RayAppMaster(/* ... */) {
  
  def registerExecutor(executorInfo: ExecutorInfo): Boolean = synchronized {
    logInfo(s"正在注册执行器: ${executorInfo.executorId}")
    
    if (executors.contains(executorInfo.executorId)) {
      logWarning(s"执行器 ${executorInfo.executorId} 已经注册")
      return false
    }
    
    // 验证执行器信息
    if (!validateExecutorInfo(executorInfo)) {
      logError(s"执行器信息无效: ${executorInfo.executorId}")
      return false
    }
    
    // 存储执行器引用
    executors += (executorInfo.executorId -> executorInfo.rayActor)
    executorIdToWorker += (executorInfo.executorId -> executorInfo.workerId)
    
    // 更新资源跟踪
    trackExecutorResources(executorInfo)
    
    logInfo(s"成功注册执行器: ${executorInfo.executorId}")
    true
  }
  
  private def validateExecutorInfo(info: ExecutorInfo): Boolean = {
    // 验证必需字段
    if (info.executorId == null || info.executorId.isEmpty) {
      logError("执行器 ID 不能为空")
      return false
    }
    
    if (info.rayActor == null) {
      logError("Ray actor 引用不能为空")
      return false
    }
    
    if (info.workerId == null || info.workerId.isEmpty) {
      logError("工作节点 ID 不能为空")
      return false
    }
    
    // 验证资源分配
    val expectedCpu = sparkConf.getInt("spark.executor.cores", 1)
    val expectedMemory = parseMemoryString(sparkConf.get("spark.executor.memory", "1g"))
    
    if (info.cpuAllocated != expectedCpu) {
      logError(s"CPU 分配不匹配: 期望 $expectedCpu, 得到 ${info.cpuAllocated}")
      return false
    }
    
    if (info.memoryAllocated != expectedMemory) {
      logError(s"内存分配不匹配: 期望 $expectedMemory, 得到 ${info.memoryAllocated}")
      return false
    }
    
    true
  }
  
  def handleTaskSubmission(taskRequest: TaskRequest): TaskResponse = synchronized {
    logDebug(s"正在处理任务提交: ${taskRequest.taskId}")
    
    // 为任务找到合适的执行器
    val targetExecutor = selectExecutorForTask(taskRequest)
    
    if (targetExecutor.isDefined) {
      // 将任务提交到选定的执行器
      submitTaskToExecutor(targetExecutor.get, taskRequest)
    } else {
      // 没有可用的合适执行器，将任务排队
      pendingTasks.enqueue(taskRequest)
      TaskResponse(
        taskId = taskRequest.taskId,
        status = TaskStatus.PENDING,
        message = "没有可用的执行器，任务已排队"
      )
    }
  }
  
  private def selectExecutorForTask(task: TaskRequest): Option[String] = {
    // 演示用的简单轮询选择
    val availableExecutors = executors.keys.filter { executorId =>
      !isExecutorBusy(executorId)
    }.toList
    
    if (availableExecutors.nonEmpty) {
      Some(availableExecutors.head)
    } else {
      None
    }
  }
}
```

## 心跳监控系统

### 执行器健康跟踪
AppMaster 实现心跳监控系统来跟踪执行器健康状况：

```scala
class HeartbeatMonitor(appMaster: RayAppMaster) extends Logging {
  
  private val executorHeartbeats = mutable.Map[String, Long]()
  private val monitorThread: Option[Thread] = None
  private var isRunning = false
  
  def start(): Unit = {
    isRunning = true
    val thread = new Thread(() => monitorLoop(), "HeartbeatMonitor")
    thread.setDaemon(true)
    thread.start()
    monitorThread = Some(thread)
  }
  
  private def monitorLoop(): Unit = {
    while (isRunning) {
      try {
        val currentTime = System.currentTimeMillis()
        val timeoutThreshold = currentTime - 30000 // 30秒超时
        
        // 检查超时的执行器
        val timedOutExecutors = executorHeartbeats.filter(_._2 < timeoutThreshold)
        
        timedOutExecutors.foreach { case (executorId, lastHeartbeat) =>
          logWarning(s"执行器 $executorId 超时，标记为失败")
          handleExecutorFailure(executorId)
          executorHeartbeats.remove(executorId)
        }
        
        // 清理失败执行器的引用
        appMaster.cleanupFailedExecutors(timedOutExecutors.keys.toSet)
        
        Thread.sleep(5000) // 每5秒检查一次
        
      } catch {
        case e: InterruptedException =>
          logInfo("心跳监控被中断，正在停止")
          isRunning = false
        case t: Throwable =>
          logError("心跳监控错误", t)
          Thread.sleep(5000) // 等待后重试
      }
    }
  }
  
  def recordHeartbeat(executorId: String): Unit = {
    executorHeartbeats.put(executorId, System.currentTimeMillis())
  }
  
  def handleExecutorFailure(executorId: String): Unit = {
    logWarning(s"正在处理执行器故障: $executorId")
    
    // 通知 Spark 上下文关于执行器故障
    appMaster.notifyExecutorFailure(executorId)
    
    // 如果配置了重启，则尝试重启执行器
    if (shouldRestartExecutor(executorId)) {
      restartExecutor(executorId)
    }
  }
  
  private def shouldRestartExecutor(executorId: String): Boolean = {
    // 检查是否启用了重启并在重试限制内
    val maxRetries = appMaster.sparkConf.getInt("spark.raydp.executor.maxRetries", 3)
    val retryCount = appMaster.getExecutorRetryCount(executorId)
    retryCount < maxRetries
  }
  
  def stop(): Unit = {
    isRunning = false
    monitorThread.foreach(_.interrupt())
  }
}
```

## 应用程序生命周期管理

### AppMaster 状态管理
AppMaster 维护自己的生命周期状态：

```scala
sealed trait AppState
case object INITIALIZING extends AppState
case object RUNNING extends AppState
case object STOPPING extends AppState
case object STOPPED extends AppState

class RayAppMaster(/* ... */) {
  private var currentState: AppState = INITIALIZING
  
  def startApplication(): Unit = synchronized {
    if (currentState != INITIALIZING) {
      throw new IllegalStateException(s"无法在状态 $currentState 下启动应用程序")
    }
    
    logInfo("正在启动 RayAppMaster 应用程序")
    
    // 初始化组件
    initialize()
    
    // 启动监控服务
    startHeartbeatMonitoring()
    
    // 标记为运行中
    currentState = RUNNING
    logInfo("RayAppMaster 启动成功")
  }
  
  def stopApplication(): Unit = synchronized {
    if (currentState == STOPPED || currentState == STOPPING) {
      logInfo("AppMaster 已停止或正在停止")
      return
    }
    
    logInfo("正在停止 RayAppMaster")
    currentState = STOPPING
    
    // 停止监控
    stopHeartbeatMonitoring()
    
    // 优雅地关闭执行器
    shutdownAllExecutors()
    
    // 停止 Py4J 服务器
    py4JServer.foreach(_.stop())
    
    currentState = STOPPED
    logInfo("RayAppMaster 停止成功")
  }
  
  private def shutdownAllExecutors(): Unit = synchronized {
    logInfo("正在关闭所有执行器")
    
    executors.values.foreach { executor =>
      try {
        // 发送关闭信号给执行器
        executor.shutdown()
      } catch {
        case e: Exception =>
          logWarning(s"关闭执行器时出错: ${e.getMessage}")
      }
    }
    
    executors.clear()
    executorIdToWorker.clear()
  }
}
```

## 总结
RayDP 中的 Java AppMaster 初始化过程涉及：
1. 创建具有资源管理功能的 RayAppMaster actor
2. 与 Python 驱动程序建立 Py4J 通信通道
3. 实现执行器注册和管理协议
4. 设置心跳监控以跟踪执行器健康状况
5. 管理从初始化到关闭的完整应用程序生命周期

此初始化建立了在 Ray 上运行的 Spark 应用程序的中央协调点，实现高效的资源管理和可靠的执行器生命周期管理。