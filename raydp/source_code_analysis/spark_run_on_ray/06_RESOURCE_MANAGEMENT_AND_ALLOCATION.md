# 06 资源管理和分配

## 概述
本文档分析了RayDP中的资源管理和分配机制，这对于在运行Spark应用程序时有效利用Ray集群资源至关重要。这包括为Spark执行器和AppMaster分配CPU、内存和自定义资源。

## Ray资源抽象层

### 资源描述符类
RayDP通过专用类抽象Ray资源：

```scala
case class ResourceRequest(
  requestId: String,
  executorId: String,
  resources: Map[String, Double],  // 例如，Map("CPU" -> 2.0, "memory" -> 4096.0)
  priority: Int = 0,
  timeoutMs: Long = 30000
)

case class ResourceAllocation(
  allocationId: String,
  executorId: String,
  resources: Map[String, Double],
  allocatedAt: Long,
  expiresAt: Option[Long] = None,
  nodeAffinity: Option[String] = None
)

case class ResourceLimits(
  maxCpu: Double,
  maxMemory: Double,
  maxCustomResources: Map[String, Double]
)

class ResourceManager(sparkConf: SparkConf) extends Logging {
  
  private val resourcePool = mutable.Map[String, ResourceAllocation]()
  private val pendingRequests = mutable.Queue[ResourceRequest]()
  private val resourceLock = new Object()
  private val resourceLimits = configureResourceLimits()
  
  private def configureResourceLimits(): ResourceLimits = {
    ResourceLimits(
      maxCpu = sparkConf.getDouble("spark.raydp.resource.maxCpu", 100.0),
      maxMemory = parseMemoryString(sparkConf.get("spark.raydp.resource.maxMemory", "100g")),
      maxCustomResources = Map.empty  // 可扩展支持GPU、存储等
    )
  }
  
  def requestResources(request: ResourceRequest): Future[ResourceAllocation] = {
    resourceLock.synchronized {
      // 根据限制验证请求
      if (!validateResourceRequest(request)) {
        return Future.failed(new IllegalArgumentException(
          s"资源请求超出限制: ${request.resources}"))
      }
      
      // 检查即时可用性
      val immediateAllocation = tryImmediateAllocation(request)
      
      if (immediateAllocation.isDefined) {
        // 即时分配成功
        val allocation = immediateAllocation.get
        resourcePool += (allocation.allocationId -> allocation)
        Future.successful(allocation)
      } else {
        // 排队等待后续分配
        pendingRequests.enqueue(request)
        
        // 启动异步分配
        Future {
          waitForAllocation(request)
        }(ExecutionContext.global)
      }
    }
  }
  
  private def validateResourceRequest(request: ResourceRequest): Boolean = {
    val totalRequested = calculateTotalResources()
    val requestResources = request.resources
    
    // 检查CPU限制
    val requestedCpu = requestResources.getOrElse("CPU", 0.0)
    val totalCpu = totalRequested.getOrElse("CPU", 0.0) + requestedCpu
    if (totalCpu > resourceLimits.maxCpu) return false
    
    // 检查内存限制
    val requestedMemory = requestResources.getOrElse("memory", 0.0)
    val totalMemory = totalRequested.getOrElse("memory", 0.0) + requestedMemory
    if (totalMemory > resourceLimits.maxMemory) return false
    
    true
  }
  
  private def calculateTotalResources(): Map[String, Double] = {
    val total = mutable.Map[String, Double]()
    
    resourcePool.values.foreach { allocation =>
      allocation.resources.foreach { case (resourceType, amount) =>
        total(resourceType) = total.getOrElse(resourceType, 0.0) + amount
      }
    }
    
    total.toMap
  }
}
```

## 资源分配算法

### 首次适应递减算法
RayDP实现了高效的资源分配算法：

```scala
class ResourceAllocator extends Logging {
  
  def allocateResources(
      availableResources: Map[String, Double],
      resourceRequests: Seq[ResourceRequest]): Seq[(ResourceRequest, Option[ResourceAllocation])] = {
    
    // 按优先级和资源大小（从大到小）排序请求
    val sortedRequests = resourceRequests.sortBy { req =>
      (-req.priority, -calculateTotalResourceSize(req.resources))
    }
    
    val allocations = mutable.ArrayBuffer[(ResourceRequest, Option[ResourceAllocation])]()
    var currentResources = availableResources
    
    sortedRequests.foreach { request =>
      val allocation = tryAllocateToResources(currentResources, request)
      
      allocation match {
        case Some(newAllocation) =>
          // 更新可用资源
          currentResources = subtractResources(currentResources, request.resources)
          allocations += ((request, Some(newAllocation)))
        case None =>
          allocations += ((request, None))
      }
    }
    
    allocations.toSeq
  }
  
  private def tryAllocateToResources(
      available: Map[String, Double],
      request: ResourceRequest): Option[ResourceAllocation] = {
    
    // 检查是否所有请求的资源都可用
    val hasEnoughResources = request.resources.forall { case (resourceType, requestedAmount) =>
      val availableAmount = available.getOrElse(resourceType, 0.0)
      availableAmount >= requestedAmount
    }
    
    if (hasEnoughResources) {
      Some(ResourceAllocation(
        allocationId = s"alloc_${System.currentTimeMillis()}_${request.requestId}",
        executorId = request.executorId,
        resources = request.resources,
        allocatedAt = System.currentTimeMillis(),
        expiresAt = if (request.timeoutMs > 0) 
          Some(System.currentTimeMillis() + request.timeoutMs) 
        else 
          None
      ))
    } else {
      None
    }
  }
  
  private def calculateTotalResourceSize(resources: Map[String, Double]): Double = {
    resources.values.sum
  }
  
  private def subtractResources(
      base: Map[String, Double],
      toSubtract: Map[String, Double]): Map[String, Double] = {
    
    val result = base.to[mutable.Map]
    
    toSubtract.foreach { case (resourceType, amount) =>
      val currentAmount = result.getOrElse(resourceType, 0.0)
      result(resourceType) = math.max(0.0, currentAmount - amount)
    }
    
    result.toMap
  }
}
```

## 动态资源扩展

### 基于需求的自动扩展
RayDP支持Spark应用程序的动态资源扩展：

```scala
class ResourceAutoScaler(appMaster: RayAppMaster, sparkConf: SparkConf) extends Logging {
  
  private val minExecutors = sparkConf.getInt("spark.raydp.minExecutors", 1)
  private val maxExecutors = sparkConf.getInt("spark.raydp.maxExecutors", 100)
  private val scalingFactor = sparkConf.getDouble("spark.raydp.scaling.factor", 1.2)
  private val scalingCooldownMs = sparkConf.getLong("spark.raydp.scaling.cooldown", 30000)
  
  private var lastScaleTime = 0L
  private val scalingLock = new Object()
  
  def evaluateAndScale(): Unit = {
    scalingLock.synchronized {
      val currentTime = System.currentTimeMillis()
      
      if (currentTime - lastScaleTime < scalingCooldownMs) {
        return  // 尊重冷却期
      }
      
      val currentMetrics = collectCurrentMetrics()
      val scalingDecision = makeScalingDecision(currentMetrics)
      
      scalingDecision match {
        case ScaleUp(count) =>
          scaleUp(count)
          lastScaleTime = currentTime
        case ScaleDown(count) =>
          scaleDown(count)
          lastScaleTime = currentTime
        case NoAction =>
          // 无需扩展
      }
    }
  }
  
  private def collectCurrentMetrics(): ResourceMetrics = {
    val activeExecutors = appMaster.getActiveExecutors()
    val pendingTasks = appMaster.getPendingTaskCount()
    val avgCpuUtilization = calculateAverageCpuUtilization(activeExecutors)
    val avgMemoryUtilization = calculateAverageMemoryUtilization(activeExecutors)
    
    ResourceMetrics(
      currentExecutorCount = activeExecutors.size,
      pendingTaskCount = pendingTasks,
      averageCpuUtilization = avgCpuUtilization,
      averageMemoryUtilization = avgMemoryUtilization,
      clusterResourceAvailability = getClusterResourceAvailability()
    )
  }
  
  private def makeScalingDecision(metrics: ResourceMetrics): ScalingAction = {
    // 扩展条件
    if (metrics.currentExecutorCount < minExecutors) {
      val needed = minExecutors - metrics.currentExecutorCount
      return ScaleUp(math.min(needed, maxExecutors - metrics.currentExecutorCount))
    }
    
    if (metrics.pendingTaskCount > 0 && 
        metrics.averageCpuUtilization > 0.8) {
      // 高CPU利用率且有待处理任务
      val scaleCount = math.max(1, (metrics.currentExecutorCount * scalingFactor).toInt - metrics.currentExecutorCount)
      return ScaleUp(math.min(scaleCount, maxExecutors - metrics.currentExecutorCount))
    }
    
    // 缩减条件
    if (metrics.currentExecutorCount > minExecutors &&
        metrics.averageCpuUtilization < 0.3 &&
        metrics.pendingTaskCount == 0) {
      // 低利用率，无待处理任务
      val scaleCount = math.max(1, (metrics.currentExecutorCount / scalingFactor).toInt - metrics.currentExecutorCount)
      return ScaleDown(math.min(-scaleCount, metrics.currentExecutorCount - minExecutors))
    }
    
    NoAction
  }
  
  private def scaleUp(count: Int): Unit = {
    logInfo(s"扩展 $count 个执行器")
    
    val executorRequests = (1 to count).map { i =>
      val executorId = s"executor_${System.currentTimeMillis()}_auto_$i"
      ResourceRequest(
        requestId = s"scaleup_${System.currentTimeMillis()}_$i",
        executorId = executorId,
        resources = Map(
          "CPU" -> sparkConf.getDouble("spark.executor.cores", 1.0),
          "memory" -> parseMemoryString(sparkConf.get("spark.executor.memory", "1g"))
        )
      )
    }
    
    // 为新执行器请求资源
    executorRequests.foreach { request =>
      appMaster.requestExecutor(request)
    }
  }
  
  private def scaleDown(count: Int): Unit = {
    logInfo(s"缩减 $count 个执行器")
    
    // 识别要移除的利用率不足的执行器
    val executorsToRemove = identifyExecutorsForRemoval(count)
    
    executorsToRemove.foreach { executorId =>
      appMaster.removeExecutor(executorId)
    }
  }
  
  private def identifyExecutorsForRemoval(count: Int): Seq[String] = {
    val activeExecutors = appMaster.getActiveExecutors()
    
    // 按利用率（从低到高）排序并选择前'count'个
    activeExecutors
      .sortBy(executorId => getExecutorUtilization(executorId))
      .take(count)
  }
}

sealed trait ScalingAction
case class ScaleUp(count: Int) extends ScalingAction
case class ScaleDown(count: Int) extends ScalingAction
case object NoAction extends ScalingAction

case class ResourceMetrics(
  currentExecutorCount: Int,
  pendingTaskCount: Int,
  averageCpuUtilization: Double,
  averageMemoryUtilization: Double,
  clusterResourceAvailability: Map[String, Double]
)
```

## 资源池管理

### 共享资源池实现
RayDP实现了一个共享资源池以实现高效的资源利用：

```scala
class ResourcePoolManager(clusterResources: RayClusterResources) extends Logging {
  
  private val resourcePools = mutable.Map[String, ResourcePool]()
  private val poolLock = new Object()
  
  def createResourcePool(
      poolName: String,
      resourceSpec: Map[String, Double],
      sharingPolicy: ResourceSharingPolicy): ResourcePool = {
    
    poolLock.synchronized {
      if (resourcePools.contains(poolName)) {
        throw new IllegalArgumentException(s"资源池 $poolName 已存在")
      }
      
      val pool = new ResourcePool(
        name = poolName,
        totalResources = resourceSpec,
        sharingPolicy = sharingPolicy,
        clusterResources = clusterResources
      )
      
      resourcePools += (poolName -> pool)
      pool
    }
  }
  
  def getPool(poolName: String): Option[ResourcePool] = {
    resourcePools.get(poolName)
  }
  
  def getAllPools(): Map[String, ResourcePool] = {
    resourcePools.toMap
  }
  
  def releasePool(poolName: String): Unit = {
    poolLock.synchronized {
      resourcePools.get(poolName) match {
        case Some(pool) =>
          pool.releaseAllResources()
          resourcePools.remove(poolName)
          logInfo(s"释放了资源池: $poolName")
        case None =>
          logWarning(s"尝试释放不存在的池: $poolName")
      }
    }
  }
}

class ResourcePool(
    val name: String,
    val totalResources: Map[String, Double],
    val sharingPolicy: ResourceSharingPolicy,
    clusterResources: RayClusterResources) extends Logging {
  
  private val allocatedResources = mutable.Map[String, ResourceAllocation]()
  private val resourceLock = new Object()
  private var usedResources = Map[String, Double]().withDefaultValue(0.0)
  
  def allocateResources(request: ResourceRequest): Option[ResourceAllocation] = {
    resourceLock.synchronized {
      // 检查请求的资源在此池中是否可用
      val hasAvailableResources = request.resources.forall { case (resourceType, amount) =>
        val totalAvailable = totalResources.getOrElse(resourceType, 0.0)
        val currentlyUsed = usedResources(resourceType)
        val requested = amount
        
        totalAvailable >= (currentlyUsed + requested)
      }
      
      if (hasAvailableResources) {
        // 更新已用资源
        request.resources.foreach { case (resourceType, amount) =>
          usedResources = usedResources.updated(resourceType, usedResources(resourceType) + amount)
        }
        
        // 创建分配
        val allocation = ResourceAllocation(
          allocationId = s"${name}_alloc_${System.currentTimeMillis()}_${request.requestId}",
          executorId = request.executorId,
          resources = request.resources,
          allocatedAt = System.currentTimeMillis()
        )
        
        allocatedResources += (allocation.allocationId -> allocation)
        
        logDebug(s"在池 $name 中分配了资源: ${request.resources}")
        Some(allocation)
      } else {
        logDebug(s"池 $name 中的资源不足以满足请求: ${request.resources}")
        None
      }
    }
  }
  
  def releaseResources(allocationId: String): Boolean = {
    resourceLock.synchronized {
      allocatedResources.get(allocationId) match {
        case Some(allocation) =>
          // 将分配的资源释放回池中
          allocation.resources.foreach { case (resourceType, amount) =>
            val currentUsed = usedResources(resourceType)
            usedResources = usedResources.updated(resourceType, math.max(0, currentUsed - amount))
          }
          
          allocatedResources.remove(allocationId)
          logDebug(s"从池 $name 释放了分配 $allocationId")
          true
          
        case None =>
          logWarning(s"尝试释放不存在的分配: $allocationId")
          false
      }
    }
  }
  
  def getResourceUtilization(): ResourceUtilization = {
    resourceLock.synchronized {
      val utilization = totalResources.map { case (resourceType, total) =>
        val used = usedResources(resourceType)
        val percentage = if (total > 0) (used / total) * 100 else 0.0
        (resourceType, ResourceUsage(used, total, percentage))
      }
      
      ResourceUtilization(name, utilization.toMap)
    }
  }
  
  def releaseAllResources(): Unit = {
    resourceLock.synchronized {
      val allocationsToRelease = allocatedResources.keys.toList
      allocationsToRelease.foreach(releaseResources)
      usedResources = Map[String, Double]().withDefaultValue(0.0)
    }
  }
}

sealed trait ResourceSharingPolicy
object ResourceSharingPolicy {
  case object FAIR_SHARE extends ResourceSharingPolicy
  case object PRIORITY_BASED extends ResourceSharingPolicy
  case object ROUND_ROBIN extends ResourceSharingPolicy
}

case class ResourceUsage(used: Double, total: Double, utilizationPercentage: Double)
case class ResourceUtilization(poolName: String, usage: Map[String, ResourceUsage])
```

## 内存管理

### JVM和堆外内存管理
RayDP管理Spark执行器的JVM堆和堆外内存：

```scala
class MemoryManager(sparkConf: SparkConf) extends Logging {
  
  // 从Spark设置中获取内存配置
  private val executorMemory = parseMemoryString(sparkConf.get("spark.executor.memory", "1g"))
  private val executorMemoryFraction = sparkConf.getDouble("spark.rdd.compress", 0.6) // 简化映射
  private val storageFraction = sparkConf.getDouble("spark.storage.memoryFraction", 0.5)
  private val executionFraction = 1.0 - storageFraction
  
  // 堆外内存设置
  private val offHeapEnabled = sparkConf.getBoolean("spark.memory.offHeap.enabled", false)
  private val offHeapSize = if (offHeapEnabled) {
    parseMemoryString(sparkConf.get("spark.memory.offHeap.size", "0"))
  } else 0L
  
  private val memoryStore = new UnifiedMemoryManager(
    onHeapStorageRegionSize = (executorMemory * storageFraction).toLong,
    onHeapExecutionRegionSize = (executorMemory * executionFraction).toLong,
    offHeapStorageRegionSize = if (offHeapEnabled) offHeapSize / 2 else 0L,
    offHeapExecutionRegionSize = if (offHeapEnabled) offHeapSize / 2 else 0L
  )
  
  def acquireExecutionMemory(size: Long, taskAttemptId: Long): Long = {
    memoryStore.acquireExecutionMemory(size, taskAttemptId, MemoryMode.ON_HEAP)
  }
  
  def acquireStorageMemory(size: Long, blockId: BlockId, memoryMode: MemoryMode): Boolean = {
    memoryStore.acquireStorageMemory(blockId, size, memoryMode)
  }
  
  def releaseExecutionMemory(size: Long, taskAttemptId: Long, memoryMode: MemoryMode): Unit = {
    memoryStore.releaseExecutionMemory(size, taskAttemptId, memoryMode)
  }
  
  def getMaxMemory(): Long = {
    executorMemory + (if (offHeapEnabled) offHeapSize else 0L)
  }
  
  def getUsedMemory(): Long = {
    memoryStore.getUsedOnHeapMemory() + memoryStore.getUsedOffHeapMemory()
  }
  
  def getFreeMemory(): Long = {
    getMaxMemory() - getUsedMemory()
  }
}

class UnifiedMemoryManager(
    onHeapStorageRegionSize: Long,
    onHeapExecutionRegionSize: Long,
    offHeapStorageRegionSize: Long,
    offHeapExecutionRegionSize: Long) extends Logging {
  
  private val onHeapStoragePool = new MemoryPool("on_heap_storage", onHeapStorageRegionSize)
  private val onHeapExecutionPool = new MemoryPool("on_heap_execution", onHeapExecutionRegionSize)
  private val offHeapStoragePool = new MemoryPool("off_heap_storage", offHeapStorageRegionSize)
  private val offHeapExecutionPool = new MemoryPool("off_heap_execution", offHeapExecutionRegionSize)
  
  private val taskMemoryManager = new TaskMemoryManager(this)
  
  def acquireExecutionMemory(size: Long, taskAttemptId: Long, memoryMode: MemoryMode): Long = {
    memoryMode match {
      case MemoryMode.ON_HEAP => onHeapExecutionPool.borrowMemory(size, taskAttemptId)
      case MemoryMode.OFF_HEAP => offHeapExecutionPool.borrowMemory(size, taskAttemptId)
    }
  }
  
  def acquireStorageMemory(blockId: BlockId, size: Long, memoryMode: MemoryMode): Boolean = {
    memoryMode match {
      case MemoryMode.ON_HEAP => onHeapStoragePool.tryBorrowMemory(size, blockId.toString)
      case MemoryMode.OFF_HEAP => offHeapStoragePool.tryBorrowMemory(size, blockId.toString)
    }
  }
  
  def releaseExecutionMemory(size: Long, taskAttemptId: Long, memoryMode: MemoryMode): Unit = {
    memoryMode match {
      case MemoryMode.ON_HEAP => onHeapExecutionPool.releaseMemory(size, taskAttemptId)
      case MemoryMode.OFF_HEAP => offHeapExecutionPool.releaseMemory(size, taskAttemptId)
    }
  }
  
  def getUsedOnHeapMemory(): Long = {
    onHeapStoragePool.getUsedMemory() + onHeapExecutionPool.getUsedMemory()
  }
  
  def getUsedOffHeapMemory(): Long = {
    offHeapStoragePool.getUsedMemory() + offHeapExecutionPool.getUsedMemory()
  }
}

class MemoryPool(name: String, maxSize: Long) extends Logging {
  
  private var usedMemory = 0L
  private val lock = new Object()
  private val borrowerMap = mutable.Map[Any, Long]() // borrowerId -> borrowedAmount
  
  def borrowMemory(size: Long, borrowerId: Any): Long = {
    lock.synchronized {
      if (size > 0 && (usedMemory + size) <= maxSize) {
        usedMemory += size
        val currentBorrowed = borrowerMap.getOrElse(borrowerId, 0L)
        borrowerMap(borrowerId) = currentBorrowed + size
        size
      } else {
        // 尝试从其他借用人处回收内存（如果可能）
        val available = maxSize - usedMemory
        math.min(size, available)
      }
    }
  }
  
  def tryBorrowMemory(size: Long, borrowerId: Any): Boolean = {
    lock.synchronized {
      if (size > 0 && (usedMemory + size) <= maxSize) {
        usedMemory += size
        val currentBorrowed = borrowerMap.getOrElse(borrowerId, 0L)
        borrowerMap(borrowerId) = currentBorrowed + size
        true
      } else {
        false
      }
    }
  }
  
  def releaseMemory(size: Long, borrowerId: Any): Unit = {
    lock.synchronized {
      val currentBorrowed = borrowerMap.getOrElse(borrowerId, 0L)
      val toRelease = math.min(size, currentBorrowed)
      
      usedMemory -= toRelease
      borrowerMap(borrowerId) = currentBorrowed - toRelease
      
      if (borrowerMap(borrowerId) <= 0) {
        borrowerMap.remove(borrowerId)
      }
    }
  }
  
  def getUsedMemory(): Long = usedMemory
  def getMaxSize(): Long = maxSize
  def getAvailableMemory(): Long = maxSize - usedMemory
}

object MemoryMode extends Enumeration {
  type MemoryMode = Value
  val ON_HEAP, OFF_HEAP = Value
}
```

## 资源监控和分析

### 实时资源监控
RayDP包含全面的资源监控功能：

```scala
class ResourceMonitor(appMaster: RayAppMaster) extends Logging {
  
  private val metricsCollector = new MetricsCollector()
  private val resourceReporters = mutable.Set[ResourceMetricsReporter]()
  private var monitoringThread: Option[Thread] = None
  private var isMonitoring = false
  
  def startMonitoring(): Unit = {
    if (!isMonitoring) {
      isMonitoring = true
      monitoringThread = Some(new Thread(() => {
        while (isMonitoring) {
          try {
            val snapshot = collectResourceSnapshot()
            resourceReporters.foreach(_.reportMetrics(snapshot))
            Thread.sleep(5000) // 每5秒报告一次
          } catch {
            case e: InterruptedException =>
              logInfo("资源监控被中断")
              isMonitoring = false
            case t: Throwable =>
              logError("资源监控错误", t)
              Thread.sleep(5000) // 继续尽管有错误
          }
        }
      }, "ResourceMonitor"))
      
      monitoringThread.foreach(_.start())
      logInfo("资源监控已启动")
    }
  }
  
  private def collectResourceSnapshot(): ResourceSnapshot = {
    val currentTime = System.currentTimeMillis()
    
    val executorMetrics = appMaster.getAllExecutors().map { executorId =>
      val executorMetrics = appMaster.getExecutorMetrics(executorId)
      executorId -> ExecutorResourceMetrics(
        cpuUsage = executorMetrics.cpuUsage,
        memoryUsage = executorMetrics.memoryUsage,
        activeTasks = executorMetrics.activeTasks,
        pendingTasks = executorMetrics.pendingTasks,
        bytesRead = executorMetrics.bytesRead,
        bytesWritten = executorMetrics.bytesWritten
      )
    }.toMap
    
    val clusterMetrics = getClusterMetrics()
    
    ResourceSnapshot(
      timestamp = currentTime,
      executorMetrics = executorMetrics,
      clusterMetrics = clusterMetrics
    )
  }
  
  private def getClusterMetrics(): ClusterResourceMetrics = {
    // 从Ray获取集群级别的资源信息
    val clusterResources = Ray.clusterResources()
    val availableResources = Ray.availableResources()
    
    ClusterResourceMetrics(
      totalCpu = clusterResources.getOrDefault("CPU", 0.0),
      availableCpu = availableResources.getOrDefault("CPU", 0.0),
      totalMemory = clusterResources.getOrDefault("memory", 0.0),
      availableMemory = availableResources.getOrDefault("memory", 0.0),
      totalCustomResources = clusterResources.asScala.filterKeys(_ != "CPU" && _ != "memory").toMap,
      availableCustomResources = availableResources.asScala.filterKeys(_ != "CPU" && _ != "memory").toMap
    )
  }
  
  def addMetricsReporter(reporter: ResourceMetricsReporter): Unit = {
    resourceReporters += reporter
  }
  
  def stopMonitoring(): Unit = {
    isMonitoring = false
    monitoringThread.foreach(_.interrupt())
    logInfo("资源监控已停止")
  }
}

trait ResourceMetricsReporter {
  def reportMetrics(snapshot: ResourceSnapshot): Unit
}

class LoggingResourceReporter extends ResourceMetricsReporter with Logging {
  
  override def reportMetrics(snapshot: ResourceSnapshot): Unit = {
    val totalCpuUsage = snapshot.executorMetrics.values.map(_.cpuUsage).sum
    val avgCpuUsage = if (snapshot.executorMetrics.nonEmpty) {
      totalCpuUsage / snapshot.executorMetrics.size
    } else 0.0
    
    logInfo(f"资源快照 - 时间: ${snapshot.timestamp} - " +
            f"执行器: ${snapshot.executorMetrics.size} - " +
            f"平均CPU使用率: $avgCpuUsage%.2f - " +
            f"集群CPU可用: ${snapshot.clusterMetrics.availableCpu}%.2f")
  }
}

case class ResourceSnapshot(
  timestamp: Long,
  executorMetrics: Map[String, ExecutorResourceMetrics],
  clusterMetrics: ClusterResourceMetrics
)

case class ExecutorResourceMetrics(
  cpuUsage: Double,
  memoryUsage: Double,
  activeTasks: Int,
  pendingTasks: Int,
  bytesRead: Long,
  bytesWritten: Long
)

case class ClusterResourceMetrics(
  totalCpu: Double,
  availableCpu: Double,
  totalMemory: Double,
  availableMemory: Double,
  totalCustomResources: Map[String, Double],
  availableCustomResources: Map[String, Double]
)
```

## 总结
RayDP中的资源管理和分配系统包括：
1. 具有请求和分配描述符的资源抽象层
2. 复杂的分配算法，如首次适应递减算法
3. 基于需求和利用率指标的动态自动扩展
4. 具有不同共享策略的共享资源池管理
5. 用于堆内和堆外内存的综合内存管理
6. 实时资源监控和分析功能

该资源管理系统确保了Ray集群资源的有效利用，同时为可靠和可扩展地运行Spark应用程序提供了必要的保证。