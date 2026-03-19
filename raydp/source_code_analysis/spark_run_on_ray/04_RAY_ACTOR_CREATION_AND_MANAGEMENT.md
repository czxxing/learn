# 04 Ray Actor 创建和管理

## 概述
本文档分析了 RayDP 中的 Ray Actor 创建和管理过程，这对于 Spark 执行器如何在 Ray 集群中实例化和管理至关重要。Ray Actor 模型为 Spark 工作负载提供了分布式执行基础设施。

## Ray Actor 基础层

### RayDP Actor 基类
所有 RayDP actor 的基础始于处理通用功能的基类：

```scala
abstract class RayDPActorBase extends Logging with Serializable {
  
  protected val actorId: String = generateUniqueActorId()
  protected val createdAt: Long = System.currentTimeMillis()
  protected var isInitialized: Boolean = false
  protected val stateLock: Object = new Object()
  
  def initialize(): Unit = {
    stateLock.synchronized {
      if (!isInitialized) {
        doInitialize()
        isInitialized = true
        logInfo(s"${this.getClass.getSimpleName} 以 ID 初始化: $actorId")
      }
    }
  }
  
  protected def doInitialize(): Unit
  
  def getId: String = actorId
  
  def getCreationTime: Long = createdAt
  
  def isAlive: Boolean = isInitialized
}

class RayTaskActor(
    val executorId: String,
    val workerId: String,
    val resources: Map[String, Double],
    val sparkConf: SparkConf) extends RayDPActorBase {
  
  private var taskQueue: mutable.Queue[TaskRequest] = mutable.Queue.empty
  private var currentTask: Option[TaskRequest] = None
  private val taskLock = new Object()
  
  override protected def doInitialize(): Unit = {
    logInfo(s"正在为执行器初始化 RayTaskActor: $executorId")
    
    // 为此执行器初始化 Spark 上下文
    initializeSparkExecutor()
    
    // 开始任务处理循环
    startTaskProcessing()
  }
  
  private def initializeSparkExecutor(): Unit = {
    // 创建 Spark 执行器实例
    val executorBackend = new CoarseGrainedExecutorBackend(
      driverUrl = s"ray://${Ray.getRuntimeContext.getNodeIpAddress}:10001",
      executorId = executorId,
      hostname = Ray.getRuntimeContext.getNodeIpAddress,
      cores = sparkConf.getInt("spark.executor.cores", 1),
      appId = sparkConf.getAppId,
      workerId = workerId,
      resources = resources
    )
    
    // 初始化后端
    executorBackend.initialize()
  }
  
  private def startTaskProcessing(): Unit = {
    // 开始后台线程进行任务处理
    val taskProcessor = new Thread(() => {
      while (isInitialized) {
        try {
          processNextTask()
          Thread.sleep(10) // 小延迟以防止忙等待
        } catch {
          case e: InterruptedException =>
            logInfo("任务处理器被中断，退出")
            return
          case t: Throwable =>
            logError("任务处理器错误", t)
        }
      }
    }, s"TaskProcessor-$executorId")
    
    taskProcessor.setDaemon(true)
    taskProcessor.start()
  }
}
```

## 执行器 Actor 创建过程

### 动态 Actor 实例化
基于资源分配动态创建执行器 actor：

```scala
class ActorFactory(sparkConf: SparkConf) extends Logging {
  
  def createExecutorActor(
      executorId: String,
      workerId: String,
      resources: Map[String, Double]): RayTaskActor = {
    
    logInfo(s"正在创建工作节点上创建执行器 actor: $executorId : $workerId")
    
    // 使用资源规范准备 actor 选项
    val rayActorOptions = RayActorOptions.newBuilder()
      .setResources(resources.asJava)
      .setMaxConcurrency(1) // 单线程以兼容 Spark 执行器
      .build()
    
    // 使用 Ray 的 actor 创建机制创建 actor
    val rayActorRef = Ray.actor(
      classOf[RayTaskActor],
      executorId,
      workerId,
      resources,
      sparkConf
    ).setOptions(rayActorOptions).remote()
    
    // 在我们的托管包装器中包装 Ray actor 引用
    val managedActor = new RayTaskActor(
      executorId = executorId,
      workerId = workerId,
      resources = resources,
      sparkConf = sparkConf
    )
    
    // 初始化 actor
    managedActor.initialize()
    
    logInfo(s"成功创建执行器 actor: $executorId")
    managedActor
  }
  
  def createAppMasterActor(
      host: String,
      port: Int,
      actorExtraClasspath: String,
      sparkConf: SparkConf): RayAppMaster = {
    
    logInfo("正在创建 AppMaster actor")
    
    // 确定 AppMaster 的资源要求
    val appMasterResources = Map(
      "CPU" -> sparkConf.getDouble("spark.raydp.appmaster.cores", 0.5),
      "memory" -> parseMemoryRequirement(
        sparkConf.get("spark.raydp.appmaster.memory", "512m"))
    )
    
    val rayActorOptions = RayActorOptions.newBuilder()
      .setResources(appMasterResources.asJava)
      .setMaxConcurrency(1)
      .build()
    
    val appMaster = new RayAppMaster(
      host = host,
      port = port,
      actorExtraClasspath = actorExtraClasspath,
      sparkConf = sparkConf
    )
    
    appMaster.initialize()
    
    logInfo("成功创建 AppMaster actor")
    appMaster
  }
  
  private def parseMemoryRequirement(memoryStr: String): Double = {
    val normalized = memoryStr.toLowerCase.trim
    if (normalized.endsWith("g")) {
      normalized.substring(0, normalized.length - 1).toDouble * 1024 * 1024 * 1024
    } else if (normalized.endsWith("m")) {
      normalized.substring(0, normalized.length - 1).toDouble * 1024 * 1024
    } else {
      normalized.toDouble
    }
  }
}
```

## Actor 生命周期管理

### Actor 状态机
每个 actor 维护自己的生命周期状态并进行适当的转换：

```scala
sealed trait ActorState
case object CREATED extends ActorState
case object INITIALIZING extends ActorState
case object RUNNING extends ActorState
case object SHUTTING_DOWN extends ActorState
case object TERMINATED extends ActorState

trait ActorLifecycle {
  protected var state: ActorState = CREATED
  protected val stateChangeLock = new Object()
  
  def getCurrentState: ActorState = state
  
  protected def transitionTo(newState: ActorState): Boolean = {
    stateChangeLock.synchronized {
      val validTransition = isValidStateTransition(state, newState)
      if (validTransition) {
        logDebug(s"从 $state 转换到 $newState")
        state = newState
        true
      } else {
        logWarning(s"从 $state 到 $newState 的状态转换无效")
        false
      }
    }
  }
  
  private def isValidStateTransition(current: ActorState, next: ActorState): Boolean = {
    (current, next) match {
      case (CREATED, INITIALIZING) => true
      case (INITIALIZING, RUNNING) => true
      case (RUNNING, SHUTTING_DOWN) => true
      case (SHUTTING_DOWN, TERMINATED) => true
      case (CREATED | INITIALIZING, SHUTTING_DOWN) => true // 允许提前关闭
      case _ => false
    }
  }
}

class ManagedRayActor(actorId: String) extends ActorLifecycle with Logging {
  
  private var shutdownHooks: List[() => Unit] = List.empty
  
  def initialize(): Boolean = {
    if (transitionTo(INITIALIZING)) {
      try {
        doInitialize()
        transitionTo(RUNNING)
        logInfo(s"Actor $actorId 初始化成功")
        true
      } catch {
        case e: Exception =>
          logError(s"初始化 actor $actorId 失败", e)
          transitionTo(SHUTTING_DOWN)
          false
      }
    } else {
      false
    }
  }
  
  protected def doInitialize(): Unit = {
    // 子类实现特定初始化
  }
  
  def shutdown(): Boolean = {
    if (transitionTo(SHUTTING_DOWN)) {
      try {
        // 执行所有关闭钩子
        shutdownHooks.foreach(hook => {
          try {
            hook()
          } catch {
            case e: Exception =>
              logWarning("执行关闭钩子时出错", e)
          }
        })
        
        doShutdown()
        transitionTo(TERMINATED)
        logInfo(s"Actor $actorId 关闭成功")
        true
      } catch {
        case e: Exception =>
          logError(s"actor $actorId 关闭期间出错", e)
          false
      }
    } else {
      false
    }
  }
  
  protected def doShutdown(): Unit = {
    // 子类实现特定关闭逻辑
  }
  
  def addShutdownHook(hook: () => Unit): Unit = {
    shutdownHooks = hook :: shutdownHooks
  }
}
```

## 基于资源的 Actor 调度

### 智能 Actor 放置
RayDP 根据资源可用性实现智能 actor 放置：

```scala
class ActorScheduler(rpcClient: RpcClient) extends Logging {
  
  private val nodeResourceTracker = mutable.Map[String, NodeResources]()
  private val actorPlacementCache = mutable.Map[String, String]() // actorId -> nodeId
  
  case class NodeResources(
    availableCpu: Double,
    availableMemory: Double,
    availableCustomResources: Map[String, Double],
    nodeId: String
  )
  
  def scheduleActor(
      actorId: String,
      requiredResources: Map[String, Double],
      placementStrategy: PlacementStrategy): String = {
    
    logDebug(s"调度 actor $actorId 具有资源: $requiredResources")
    
    // 从 Ray 集群更新资源信息
    updateNodeResources()
    
    // 根据策略选择适当的节点
    val targetNodeId = selectTargetNode(requiredResources, placementStrategy)
    
    if (targetNodeId.nonEmpty) {
      // 在选定的节点上预留资源
      reserveResourcesOnNode(targetNodeId, requiredResources)
      
      // 缓存放置决策
      actorPlacementCache += (actorId -> targetNodeId)
      
      logInfo(s"在节点上调度 actor $actorId: $targetNodeId")
    } else {
      throw new RuntimeException(s"找不到适合 actor $actorId 的节点")
    }
  }
  
  private def selectTargetNode(
      requiredResources: Map[String, Double],
      strategy: PlacementStrategy): String = {
    
    val availableNodes = nodeResourceTracker.filter { case (_, resources) =>
      hasSufficientResources(resources, requiredResources)
    }
    
    strategy match {
      case PlacementStrategy.SPREAD =>
        selectNodeWithMostAvailableResources(availableNodes)
      case PlacementStrategy.LOCALITY =>
        selectNodeWithBestLocality(availableNodes)
      case PlacementStrategy.MIN_IDLE =>
        selectNodeWithLeastLoad(availableNodes)
    }
  }
  
  private def hasSufficientResources(
      nodeResources: NodeResources,
      required: Map[String, Double]): Boolean = {
    
    required.forall { case (resourceType, requiredAmount) =>
      resourceType match {
        case "CPU" => nodeResources.availableCpu >= requiredAmount
        case "memory" => nodeResources.availableMemory >= requiredAmount
        case customResource => 
          nodeResources.availableCustomResources.getOrElse(customResource, 0.0) >= requiredAmount
      }
    }
  }
  
  private def updateNodeResources(): Unit = {
    try {
      // 查询 Ray 集群的当前资源可用性
      val clusterResources = rpcClient.getClusterResources()
      
      nodeResourceTracker.clear()
      clusterResources.foreach { case (nodeId, resources) =>
        val nodeRes = NodeResources(
          availableCpu = resources.getOrElse("CPU", 0.0),
          availableMemory = resources.getOrElse("memory", 0.0),
          availableCustomResources = resources.filterKeys(!Set("CPU", "memory").contains(_)),
          nodeId = nodeId
        )
        nodeResourceTracker += (nodeId -> nodeRes)
      }
    } catch {
      case e: Exception =>
        logWarning("更新节点资源失败", e)
    }
  }
  
  private def reserveResourcesOnNode(nodeId: String, resources: Map[String, Double]): Unit = {
    val nodeResources = nodeResourceTracker(nodeId)
    
    val updatedResources = NodeResources(
      availableCpu = nodeResources.availableCpu - resources.getOrElse("CPU", 0.0),
      availableMemory = nodeResources.availableMemory - resources.getOrElse("memory", 0.0),
      availableCustomResources = resources.foldLeft(nodeResources.availableCustomResources) {
        case (acc, (resourceType, amount)) if resourceType != "CPU" && resourceType != "memory" =>
          acc.updated(resourceType, acc.getOrElse(resourceType, 0.0) - amount)
        case (acc, _) => acc
      },
      nodeId = nodeId
    )
    
    nodeResourceTracker.update(nodeId, updatedResources)
  }
}

sealed trait PlacementStrategy
object PlacementStrategy {
  case object SPREAD extends PlacementStrategy
  case object LOCALITY extends PlacementStrategy
  case object MIN_IDLE extends PlacementStrategy
}
```

## Actor 通信框架

### Actor 间通信
Actor 通过 Ray 的通信机制相互通信：

```scala
trait ActorCommunicator {
  
  def sendMessage[T](targetActor: RayActor[T], message: Any): Future[Any]
  
  def broadcastMessage[T](actors: Seq[RayActor[T]], message: Any): Future[Seq[Any]]
  
  def registerMessageHandler(messageType: Class[_], handler: Any => Unit): Unit
}

class RayActorCommunicator extends ActorCommunicator with Logging {
  
  override def sendMessage[T](targetActor: RayActor[T], message: Any): Future[Any] = {
    logDebug(s"发送消息到 actor: ${targetActor.getId}")
    
    // 使用 Ray 的远程调用机制
    val future = targetActor.handleMessage(message)
    
    future.onComplete {
      case Success(result) => 
        logDebug(s"消息由 actor 成功处理: ${targetActor.getId}")
      case Failure(exception) => 
        logError(s"发送消息到 actor 失败: ${targetActor.getId}", exception)
    }
    
    future.asInstanceOf[Future[Any]]
  }
  
  override def broadcastMessage[T](actors: Seq[RayActor[T]], message: Any): Future[Seq[Any]] = {
    logDebug(s"向 ${actors.length} 个 actor 广播消息")
    
    val futures = actors.map(actor => {
      try {
        actor.handleMessage(message)
      } catch {
        case e: Exception =>
          logError(s"发送消息到 actor 失败: ${actor.getId}", e)
          Future.failed(e)
      }
    })
    
    Future.sequence(futures).asInstanceOf[Future[Seq[Any]]]
  }
  
  override def registerMessageHandler(messageType: Class[_], handler: Any => Unit): Unit = {
    // 将处理程序注册到消息分发器
    MessageDispatcher.registerHandler(messageType, handler)
  }
}

class MessageDispatcher extends Logging {
  
  private val handlers = mutable.Map[Class[_], Any => Unit]()
  
  def dispatchMessage(message: Any): Unit = {
    val messageType = message.getClass
    
    handlers.get(messageType) match {
      case Some(handler) =>
        try {
          handler(message)
        } catch {
          case e: Exception =>
            logError(s"处理类型为的消息时出错: $messageType", e)
        }
      case None =>
        logWarning(s"没有为消息类型注册处理程序: $messageType")
    }
  }
  
  def registerHandler(messageType: Class[_], handler: Any => Unit): Unit = {
    handlers += (messageType -> handler)
    logDebug(s"为消息类型注册处理程序: $messageType")
  }
}
```

## Actor 故障恢复

### 容错机制
RayDP 为 actor 实现全面的故障恢复：

```scala
class ActorFailureRecovery(appMaster: RayAppMaster) extends Logging {
  
  private val failedActorRegistry = mutable.Set[String]()
  private val recoveryQueue = mutable.Queue[RecoveryRequest]()
  private val maxRecoveryAttempts = 3
  
  case class RecoveryRequest(
    actorId: String,
    actorType: String,
    creationParams: Map[String, Any],
    attemptCount: Int = 0
  )
  
  def handleActorFailure(actorId: String, failureReason: String): Unit = {
    logWarning(s"Actor $actorId 失败: $failureReason")
    
    failedActorRegistry += actorId
    
    // 确定是否应恢复 actor
    if (shouldRecoverActor(actorId)) {
      val recoveryRequest = createRecoveryRequest(actorId)
      addToRecoveryQueue(recoveryRequest)
      
      logInfo(s"将 actor $actorId 添加到恢复队列")
    } else {
      logWarning(s"不恢复 actor $actorId，超出最大尝试次数")
      notifyApplicationFailure(actorId, failureReason)
    }
  }
  
  private def shouldRecoverActor(actorId: String): Boolean = {
    val currentAttempts = getRecoveryAttempts(actorId)
    currentAttempts < maxRecoveryAttempts
  }
  
  private def createRecoveryRequest(actorId: String): RecoveryRequest = {
    // 检索原始 actor 创建参数
    val creationParams = appMaster.getActorCreationParams(actorId)
    
    RecoveryRequest(
      actorId = actorId,
      actorType = appMaster.getActorType(actorId),
      creationParams = creationParams,
      attemptCount = getRecoveryAttempts(actorId) + 1
    )
  }
  
  def processRecoveryQueue(): Unit = {
    val recoverableRequests = recoveryQueue.dequeueAll(_ => true)
    
    recoverableRequests.foreach { request =>
      try {
        val newActorId = recoverActor(request)
        if (newActorId.nonEmpty) {
          logInfo(s"成功恢复 actor: ${request.actorId} 作为: $newActorId")
          
          // 在 AppMaster 中更新映射
          appMaster.updateActorMapping(request.actorId, newActorId)
        } else {
          logError(s"恢复 actor 失败: ${request.actorId}")
          if (request.attemptCount >= maxRecoveryAttempts) {
            logError(s"达到 actor 的最大恢复尝试次数: ${request.actorId}")
          } else {
            // 重新排队以供稍后再次尝试
            recoveryQueue.enqueue(request.copy(attemptCount = request.attemptCount + 1))
          }
        }
      } catch {
        case e: Exception =>
          logError(s"恢复 actor 期间出现异常: ${request.actorId}", e)
          // 重新排队以供另一次尝试
          recoveryQueue.enqueue(request)
      }
    }
  }
  
  private def recoverActor(request: RecoveryRequest): String = {
    try {
      request.actorType match {
        case "EXECUTOR" =>
          val executorId = request.creationParams("executorId").asInstanceOf[String]
          val workerId = request.creationParams("workerId").asInstanceOf[String]
          val resources = request.creationParams("resources").asInstanceOf[Map[String, Double]]
          val sparkConf = request.creationParams("sparkConf").asInstanceOf[SparkConf]
          
          // 使用相同配置创建新执行器 actor
          val newExecutorId = s"${executorId}_recovered_${System.currentTimeMillis()}"
          val factory = new ActorFactory(sparkConf)
          val newActor = factory.createExecutorActor(newExecutorId, workerId, resources)
          
          // 与 AppMaster 注册
          appMaster.registerRecoveredExecutor(newActor, request.actorId)
          
          newExecutorId
          
        case _ =>
          logError(s"未知恢复 actor 类型: ${request.actorType}")
          ""
      }
    } catch {
      case e: Exception =>
        logError(s"恢复 actor 失败: ${request.actorId}", e)
        ""
    }
  }
  
  private def addToRecoveryQueue(request: RecoveryRequest): Unit = {
    recoveryQueue.enqueue(request)
  }
  
  private def getRecoveryAttempts(actorId: String): Int = {
    // 跟踪恢复尝试的实现
    0 // 为示例简化
  }
}
```

## 总结
RayDP 中的 Ray Actor 创建和管理过程涉及：
1. 使用基础 actor 类和初始化模式创建基础层
2. 实现带有适当资源分配的动态 actor 实例化
3. 使用状态机和适当转换管理 actor 生命周期
4. 基于资源可用性实现智能 actor 放置
5. 在 actor 之间建立通信框架
6. 提供全面的故障恢复机制

此 actor 管理系统构成了 RayDP 分布式执行模型的骨干，使 Ray 集群上的 Spark 工作负载能够高效可靠地执行。