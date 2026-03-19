# 07 容错与恢复

## 概述
本文档分析了RayDP中的容错和恢复机制，这些机制对于确保在Ray集群上运行的Spark应用程序的可靠性和弹性至关重要。这些机制处理各种级别的故障，包括执行器故障、网络分区和资源不可用。

## 执行器故障检测和处理

### 基于心跳的故障检测
RayDP实现了强大的心跳机制来检测执行器故障：

```scala
class ExecutorFailureDetector(appMaster: RayAppMaster) extends Logging {
  
  private val executorHeartbeats = mutable.Map[String, Long]()
  private val executorLastKnownStatus = mutable.Map[String, ExecutorStatus]()
  private val failureListeners = mutable.Set[FailureListener]()
  private val detectorThread: Option[Thread] = None
  private var isRunning = false
  private val heartbeatTimeoutMs = 30000 // 30秒
  private val checkIntervalMs = 5000     // 5秒
  
  def startDetection(): Unit = {
    isRunning = true
    val thread = new Thread(() => {
      while (isRunning) {
        try {
          detectFailedExecutors()
          Thread.sleep(checkIntervalMs)
        } catch {
          case e: InterruptedException =>
            logInfo("执行器故障检测器被中断")
            isRunning = false
          case t: Throwable =>
            logError("执行器故障检测错误", t)
            Thread.sleep(checkIntervalMs) // 继续执行尽管有错误
        }
      }
    }, "ExecutorFailureDetector")
    
    thread.setDaemon(true)
    thread.start()
    detectorThread = Some(thread)
    logInfo("执行器故障检测已启动")
  }
  
  private def detectFailedExecutors(): Unit = {
    val currentTime = System.currentTimeMillis()
    val timeoutThreshold = currentTime - heartbeatTimeoutMs
    
    val timedOutExecutors = executorHeartbeats.filter { case (_, lastHeartbeat) =>
      lastHeartbeat < timeoutThreshold
    }
    
    timedOutExecutors.foreach { case (executorId, lastHeartbeat) =>
      logWarning(s"执行器 $executorId 超时 (最后心跳: ${new Date(lastHeartbeat)})")
      
      // 通知故障监听器
      failureListeners.foreach(_.onExecutorFailureDetected(executorId, "HEARTBEAT_TIMEOUT"))
      
      // 从活动跟踪中移除
      executorHeartbeats.remove(executorId)
      executorLastKnownStatus.remove(executorId)
      
      // 触发恢复过程
      appMaster.handleExecutorFailure(executorId, "HEARTBEAT_TIMEOUT")
    }
  }
  
  def recordHeartbeat(executorId: String, status: ExecutorStatus): Unit = {
    executorHeartbeats.put(executorId, System.currentTimeMillis())
    executorLastKnownStatus.put(executorId, status)
  }
  
  def registerFailureListener(listener: FailureListener): Unit = {
    failureListeners += listener
  }
  
  def unregisterFailureListener(listener: FailureListener): Unit = {
    failureListeners -= listener
  }
  
  def stopDetection(): Unit = {
    isRunning = false
    detectorThread.foreach(_.interrupt())
    logInfo("执行器故障检测已停止")
  }
}

trait FailureListener {
  def onExecutorFailureDetected(executorId: String, reason: String): Unit
}

class TaskReassignmentHandler(appMaster: RayAppMaster) extends FailureListener with Logging {
  
  override def onExecutorFailureDetected(executorId: String, reason: String): Unit = {
    logInfo(s"处理失败执行器的任务重新分配: $executorId")
    
    // 获取在失败执行器上运行的任务
    val affectedTasks = appMaster.getTasksOnExecutor(executorId)
    
    if (affectedTasks.nonEmpty) {
      logInfo(s"发现 ${affectedTasks.size} 个任务受执行器故障影响")
      
      // 标记任务为失败
      affectedTasks.foreach { taskId =>
        appMaster.markTaskAsFailed(taskId, executorId, reason)
      }
      
      // 尝试重新调度失败的任务
      rescheduleAffectedTasks(affectedTasks)
    }
  }
  
  private def rescheduleAffectedTasks(taskIds: Seq[String]): Unit = {
    val rescheduledCount = taskIds.count { taskId =>
      try {
        val taskInfo = appMaster.getTaskInfo(taskId)
        val newExecutorId = findSuitableExecutor(taskInfo)
        
        if (newExecutorId.isDefined) {
          appMaster.rescheduleTask(taskId, newExecutorId.get)
          true
        } else {
          logWarning(s"未找到适合任务的执行器: $taskId")
          appMaster.queueTaskForLater(taskId)
          false
        }
      } catch {
        case e: Exception =>
          logError(s"重新调度任务错误: $taskId", e)
          false
      }
    }
    
    logInfo(s"成功重新调度了 ${taskIds.size} 个任务中的 $rescheduledCount 个")
  }
  
  private def findSuitableExecutor(taskInfo: TaskInfo): Option[String] = {
    val availableExecutors = appMaster.getAvailableExecutors()
    
    // 查找具有足够资源用于任务的执行器
    availableExecutors.find { executorId =>
      val executorResources = appMaster.getExecutorResources(executorId)
      val taskRequirements = taskInfo.resourceRequirements
      
      taskRequirements.forall { case (resourceType, requiredAmount) =>
        executorResources.getOrElse(resourceType, 0.0) >= requiredAmount
      }
    }
  }
}
```

## Checkpoint and Recovery Mechanisms

### Distributed Checkpoint Management
RayDP implements checkpointing for fault tolerance:

```scala
class CheckpointManager(appMaster: RayAppMaster, sparkConf: SparkConf) extends Logging {
  
  private val checkpointDir = sparkConf.get("spark.raydp.checkpoint.dir", "/tmp/raydp-checkpoints")
  private val checkpointInterval = sparkConf.getLong("spark.raydp.checkpoint.interval", 60000) // 1 minute
  private val maxRetainedCheckpoints = sparkConf.getInt("spark.raydp.checkpoint.max.retained", 5)
  private val checkpointTracker = mutable.Map[String, CheckpointMetadata]()
  private val checkpointLock = new Object()
  
  def createCheckpoint(applicationId: String, checkpointData: CheckpointData): CheckpointMetadata = {
    checkpointLock.synchronized {
      val checkpointId = s"chkpt_${System.currentTimeMillis()}_${applicationId}"
      val checkpointPath = s"$checkpointDir/$applicationId/$checkpointId"
      
      try {
        // Serialize checkpoint data
        val serializedData = serializeCheckpointData(checkpointData)
        
        // Write to storage (could be local file system, cloud storage, etc.)
        writeCheckpointToFile(checkpointPath, serializedData)
        
        // Create metadata
        val metadata = CheckpointMetadata(
          checkpointId = checkpointId,
          applicationId = applicationId,
          checkpointPath = checkpointPath,
          timestamp = System.currentTimeMillis(),
          dataSize = serializedData.length,
          status = CheckpointStatus.ACTIVE
        )
        
        // Store metadata
        checkpointTracker += (checkpointId -> metadata)
        
        // Clean up old checkpoints
        retainLatestCheckpoints(applicationId)
        
        logInfo(s"Created checkpoint: $checkpointId for application: $applicationId")
        metadata
        
      } catch {
        case e: Exception =>
          logError(s"Failed to create checkpoint for application: $applicationId", e)
          throw e
      }
    }
  }
  
  private def serializeCheckpointData(data: CheckpointData): Array[Byte] = {
    // Use Kryo serializer or Java serialization
    import java.io._
    import java.util.Base64
    
    val baos = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(baos)
    oos.writeObject(data)
    oos.close()
    
    baos.toByteArray
  }
  
  private def writeCheckpointToFile(path: String, data: Array[Byte]): Unit = {
    val file = new File(path)
    file.getParentFile.mkdirs() // Ensure directory exists
    
    val fos = new FileOutputStream(file)
    fos.write(data)
    fos.close()
  }
  
  private def retainLatestCheckpoints(applicationId: String): Unit = {
    val appCheckpoints = checkpointTracker.filter { case (_, metadata) =>
      metadata.applicationId == applicationId && metadata.status == CheckpointStatus.ACTIVE
    }.toSeq.sortBy(_._2.timestamp).reverse
    
    if (appCheckpoints.length > maxRetainedCheckpoints) {
      val checkpointsToDelete = appCheckpoints.drop(maxRetainedCheckpoints)
      
      checkpointsToDelete.foreach { case (checkpointId, metadata) =>
        try {
          deleteCheckpoint(metadata.checkpointPath)
          checkpointTracker(checkpointId) = metadata.copy(status = CheckpointStatus.DELETED)
          logDebug(s"Deleted old checkpoint: $checkpointId")
        } catch {
          case e: Exception =>
            logWarning(s"Failed to delete checkpoint: $checkpointId", e)
        }
      }
    }
  }
  
  def restoreFromCheckpoint(applicationId: String, checkpointId: String): Option[CheckpointData] = {
    checkpointLock.synchronized {
      checkpointTracker.get(checkpointId) match {
        case Some(metadata) if metadata.applicationId == applicationId && 
                                metadata.status == CheckpointStatus.ACTIVE =>
          
          try {
            val data = readCheckpointFromFile(metadata.checkpointPath)
            val deserializedData = deserializeCheckpointData(data)
            logInfo(s"Restored from checkpoint: $checkpointId")
            Some(deserializedData)
          } catch {
            case e: Exception =>
              logError(s"Failed to restore from checkpoint: $checkpointId", e)
              None
          }
          
        case _ =>
          logWarning(s"Checkpoint not found or invalid: $checkpointId")
          None
      }
    }
  }
  
  private def readCheckpointFromFile(path: String): Array[Byte] = {
    val file = new File(path)
    if (!file.exists()) {
      throw new FileNotFoundException(s"Checkpoint file not found: $path")
    }
    
    val fis = new FileInputStream(file)
    val data = Stream.continually(fis.read).takeWhile(_ != -1).map(_.toByte).toArray
    fis.close()
    data
  }
  
  private def deserializeCheckpointData(data: Array[Byte]): CheckpointData = {
    import java.io._
    
    val bais = new ByteArrayInputStream(data)
    val ois = new ObjectInputStream(bais)
    val result = ois.readObject().asInstanceOf[CheckpointData]
    ois.close()
    result
  }
  
  private def deleteCheckpoint(path: String): Unit = {
    val file = new File(path)
    if (file.exists()) {
      file.delete()
    }
  }
  
  def getLatestCheckpoint(applicationId: String): Option[CheckpointMetadata] = {
    checkpointLock.synchronized {
      checkpointTracker.values
        .filter(m => m.applicationId == applicationId && m.status == CheckpointStatus.ACTIVE)
        .maxByOption(_.timestamp)
    }
  }
}

case class CheckpointData(
  applicationState: ApplicationState,
  executorStates: Map[String, ExecutorState],
  taskQueues: Map[String, Seq[TaskInfo]],
  resourceAllocations: Map[String, ResourceAllocation]
)

case class CheckpointMetadata(
  checkpointId: String,
  applicationId: String,
  checkpointPath: String,
  timestamp: Long,
  dataSize: Long,
  status: CheckpointStatus
)

sealed trait CheckpointStatus
object CheckpointStatus {
  case object ACTIVE extends CheckpointStatus
  case object DELETED extends CheckpointStatus
  case object FAILED extends CheckpointStatus
}
```

## Application-Level Recovery

### Application State Management
RayDP maintains application state for recovery purposes:

```scala
class ApplicationStateManager(appMaster: RayAppMaster) extends Logging {
  
  private val applicationStates = mutable.Map[String, ApplicationState]()
  private val stateLock = new Object()
  
  def captureApplicationState(appId: String): ApplicationState = {
    stateLock.synchronized {
      val currentState = ApplicationState(
        appId = appId,
        timestamp = System.currentTimeMillis(),
        executorStates = getExecutorStates(appId),
        taskStates = getTaskStates(appId),
        resourceStates = getResourceStates(appId),
        applicationProgress = getApplicationProgress(appId)
      )
      
      applicationStates += (appId -> currentState)
      currentState
    }
  }
  
  private def getExecutorStates(appId: String): Map[String, ExecutorState] = {
    val activeExecutors = appMaster.getActiveExecutors()
    
    activeExecutors.map { executorId =>
      val executorInfo = appMaster.getExecutorInfo(executorId)
      executorId -> ExecutorState(
        executorId = executorId,
        status = executorInfo.status,
        resources = executorInfo.resources,
        activeTasks = executorInfo.activeTasks,
        pendingTasks = executorInfo.pendingTasks,
        lastHeartbeat = executorInfo.lastHeartbeat
      )
    }.toMap
  }
  
  private def getTaskStates(appId: String): Map[String, TaskState] = {
    val allTasks = appMaster.getAllTasks(appId)
    
    allTasks.map { taskId =>
      val taskInfo = appMaster.getTaskInfo(taskId)
      taskId -> TaskState(
        taskId = taskId,
        status = taskInfo.status,
        assignedExecutor = taskInfo.assignedExecutor,
        startTime = taskInfo.startTime,
        endTime = taskInfo.endTime,
        attemptCount = taskInfo.attemptCount
      )
    }.toMap
  }
  
  private def getResourceStates(appId: String): Map[String, ResourceState] = {
    val resourceAllocations = appMaster.getResourceAllocations(appId)
    
    resourceAllocations.map { case (allocationId, allocation) =>
      allocationId -> ResourceState(
        allocationId = allocationId,
        resources = allocation.resources,
        allocatedAt = allocation.allocatedAt,
        expiresAt = allocation.expiresAt
      )
    }.toMap
  }
  
  private def getApplicationProgress(appId: String): ApplicationProgress = {
    val totalTasks = appMaster.getTotalTaskCount(appId)
    val completedTasks = appMaster.getCompletedTaskCount(appId)
    val failedTasks = appMaster.getFailedTaskCount(appId)
    
    ApplicationProgress(
      totalTasks = totalTasks,
      completedTasks = completedTasks,
      failedTasks = failedTasks,
      progressPercentage = if (totalTasks > 0) (completedTasks.toDouble / totalTasks) * 100 else 0.0
    )
  }
  
  def restoreApplicationState(appId: String, state: ApplicationState): Boolean = {
    stateLock.synchronized {
      try {
        // Restore executor states
        state.executorStates.foreach { case (executorId, executorState) =>
          appMaster.restoreExecutorState(executorId, executorState)
        }
        
        // Restore task states
        state.taskStates.foreach { case (taskId, taskState) =>
          appMaster.restoreTaskState(taskId, taskState)
        }
        
        // Restore resource states
        state.resourceStates.foreach { case (allocationId, resourceState) =>
          appMaster.restoreResourceState(allocationId, resourceState)
        }
        
        logInfo(s"Successfully restored application state for: $appId")
        true
        
      } catch {
        case e: Exception =>
          logError(s"Failed to restore application state for: $appId", e)
          false
      }
    }
  }
  
  def cleanupOldStates(retentionHours: Int): Unit = {
    stateLock.synchronized {
      val cutoffTime = System.currentTimeMillis() - (retentionHours * 60 * 60 * 1000L)
      
      val statesToClean = applicationStates.filter { case (_, state) =>
        state.timestamp < cutoffTime
      }
      
      statesToClean.foreach { case (appId, _) =>
        applicationStates.remove(appId)
        logDebug(s"Cleaned up old application state for: $appId")
      }
    }
  }
}

case class ApplicationState(
  appId: String,
  timestamp: Long,
  executorStates: Map[String, ExecutorState],
  taskStates: Map[String, TaskState],
  resourceStates: Map[String, ResourceState],
  applicationProgress: ApplicationProgress
)

case class ExecutorState(
  executorId: String,
  status: String,
  resources: Map[String, Double],
  activeTasks: Int,
  pendingTasks: Int,
  lastHeartbeat: Long
)

case class TaskState(
  taskId: String,
  status: String,
  assignedExecutor: Option[String],
  startTime: Option[Long],
  endTime: Option[Long],
  attemptCount: Int
)

case class ResourceState(
  allocationId: String,
  resources: Map[String, Double],
  allocatedAt: Long,
  expiresAt: Option[Long]
)

case class ApplicationProgress(
  totalTasks: Int,
  completedTasks: Int,
  failedTasks: Int,
  progressPercentage: Double
)
```

## Network Partition Handling

### Network Resilience Mechanisms
RayDP handles network partitions and connectivity issues:

```scala
class NetworkPartitionHandler(appMaster: RayAppMaster) extends Logging {
  
  private val networkConnectivityMonitors = mutable.Map[String, ConnectivityMonitor]()
  private val partitionDetectedListeners = mutable.Set[PartitionDetectedListener]()
  private val recoveryHandlers = mutable.Map[String, PartitionRecoveryHandler]()
  
  def startMonitoring(executorId: String, endpoint: String): Unit = {
    val monitor = new ConnectivityMonitor(executorId, endpoint)
    networkConnectivityMonitors += (executorId -> monitor)
    monitor.startMonitoring()
  }
  
  def stopMonitoring(executorId: String): Unit = {
    networkConnectivityMonitors.get(executorId) match {
      case Some(monitor) =>
        monitor.stopMonitoring()
        networkConnectivityMonitors.remove(executorId)
      case None =>
        logWarning(s"No monitor found for executor: $executorId")
    }
  }
  
  def onPartitionDetected(executorId: String, endpoint: String, reason: String): Unit = {
    logWarning(s"Network partition detected for executor: $executorId at $endpoint, reason: $reason")
    
    // Notify all registered listeners
    partitionDetectedListeners.foreach(_.onPartitionDetected(executorId, endpoint, reason))
    
    // Trigger recovery process
    triggerPartitionRecovery(executorId, reason)
  }
  
  private def triggerPartitionRecovery(executorId: String, reason: String): Unit = {
    val recoveryHandler = new PartitionRecoveryHandler(appMaster, executorId, reason)
    recoveryHandlers += (executorId -> recoveryHandler)
    
    // Start recovery in background
    Future {
      recoveryHandler.performRecovery()
    }(ExecutionContext.global)
  }
  
  def registerPartitionListener(listener: PartitionDetectedListener): Unit = {
    partitionDetectedListeners += listener
  }
  
  def unregisterPartitionListener(listener: PartitionDetectedListener): Unit = {
    partitionDetectedListeners -= listener
  }
}

class ConnectivityMonitor(executorId: String, endpoint: String) extends Logging {
  
  private val pingInterval = 5000 // 5 seconds
  private var monitoringThread: Option[Thread] = None
  private var isMonitoring = false
  private val handler: NetworkPartitionHandler
  
  def startMonitoring(): Unit = {
    isMonitoring = true
    monitoringThread = Some(new Thread(() => {
      while (isMonitoring) {
        try {
          if (!pingExecutor()) {
            handler.onPartitionDetected(executorId, endpoint, "NETWORK_UNREACHABLE")
          }
          Thread.sleep(pingInterval)
        } catch {
          case e: InterruptedException =>
            logInfo(s"Connectivity monitor for $executorId interrupted")
            isMonitoring = false
          case t: Throwable =>
            logError(s"Error monitoring connectivity for $executorId", t)
            Thread.sleep(pingInterval)
        }
      }
    }, s"ConnectivityMonitor-$executorId"))
    
    monitoringThread.foreach(_.start())
  }
  
  private def pingExecutor(): Boolean = {
    try {
      // Attempt to contact executor via its communication channel
      val success = attemptPing(endpoint)
      if (!success) {
        logDebug(s"Ping failed for executor: $executorId at $endpoint")
      }
      success
    } catch {
      case _: Exception => false
    }
  }
  
  private def attemptPing(endpoint: String): Boolean = {
    // Implementation would depend on the actual communication protocol
    // For example, could be an HTTP health check, TCP connection test, etc.
    true // Simplified for example
  }
  
  def stopMonitoring(): Unit = {
    isMonitoring = false
    monitoringThread.foreach(_.interrupt())
  }
}

trait PartitionDetectedListener {
  def onPartitionDetected(executorId: String, endpoint: String, reason: String): Unit
}

class PartitionRecoveryHandler(
    appMaster: RayAppMaster,
    executorId: String,
    reason: String) extends Logging {
  
  def performRecovery(): Unit = {
    logInfo(s"Starting recovery for partitioned executor: $executorId, reason: $reason")
    
    try {
      // Determine recovery strategy based on reason
      val recoveryStrategy = determineRecoveryStrategy(reason)
      
      recoveryStrategy match {
        case RestartExecutor =>
          restartExecutor()
        case MigrateTasks =>
          migrateTasksToNewExecutor()
        case FailoverToBackup =>
          failoverToBackupExecutor()
        case TerminateAndReschedule =>
          terminateAndRescheduleTasks()
      }
      
      logInfo(s"Recovery completed for executor: $executorId")
      
    } catch {
      case e: Exception =>
        logError(s"Recovery failed for executor: $executorId", e)
        // Fallback to executor termination
        appMaster.terminateExecutor(executorId)
    }
  }
  
  private def determineRecoveryStrategy(reason: String): RecoveryStrategy = {
    reason match {
      case "NETWORK_UNREACHABLE" | "CONNECTION_TIMEOUT" => MigrateTasks
      case "HEARTBEAT_TIMEOUT" => RestartExecutor
      case "RESOURCE_EXHAUSTED" => TerminateAndReschedule
      case _ => MigrateTasks
    }
  }
  
  private def restartExecutor(): Unit = {
    logInfo(s"Restarting executor: $executorId")
    appMaster.restartExecutor(executorId)
  }
  
  private def migrateTasksToNewExecutor(): Unit = {
    logInfo(s"Migrating tasks from failed executor: $executorId")
    
    val affectedTasks = appMaster.getTasksOnExecutor(executorId)
    
    // Create new executor
    val newExecutorId = appMaster.createReplacementExecutor(executorId)
    
    if (newExecutorId.isDefined) {
      // Reassign tasks to new executor
      affectedTasks.foreach { taskId =>
        appMaster.reassignTask(taskId, newExecutorId.get)
      }
      
      // Remove old executor
      appMaster.removeExecutor(executorId)
    } else {
      logWarning(s"Could not create replacement executor for: $executorId")
      // Fall back to task rescheduling
      affectedTasks.foreach { taskId =>
        appMaster.rescheduleTask(taskId, None)
      }
    }
  }
  
  private def failoverToBackupExecutor(): Unit = {
    logInfo(s"Failing over to backup executor for: $executorId")
    // Implementation would involve identifying and activating a backup executor
  }
  
  private def terminateAndRescheduleTasks(): Unit = {
    logInfo(s"Terminating executor and rescheduling tasks: $executorId")
    
    val affectedTasks = appMaster.getTasksOnExecutor(executorId)
    
    // Mark tasks as needing rescheduling
    affectedTasks.foreach { taskId =>
      appMaster.markTaskForRescheduling(taskId)
    }
    
    // Remove the failed executor
    appMaster.removeExecutor(executorId)
  }
}

sealed trait RecoveryStrategy
object RecoveryStrategy {
  case object RestartExecutor extends RecoveryStrategy
  case object MigrateTasks extends RecoveryStrategy
  case object FailoverToBackup extends RecoveryStrategy
  case object TerminateAndReschedule extends RecoveryStrategy
}
```

## Graceful Degradation

### Service Degradation Strategies
RayDP implements graceful degradation when resources are limited:

```scala
class ServiceDegradationManager(appMaster: RayAppMaster) extends Logging {
  
  private val degradationThresholds = Map(
    "cpu_utilization" -> 0.9,  // 90% CPU utilization threshold
    "memory_utilization" -> 0.85,  // 85% memory utilization threshold
    "disk_space" -> 0.1,  // 10% disk space remaining threshold
    "network_bandwidth" -> 0.8  // 80% network bandwidth utilization threshold
  )
  
  private var currentServiceLevel = ServiceLevel.NORMAL
  private val degradationActions = mutable.Map[String, DegradationAction]()
  
  def evaluateServiceLevel(): ServiceLevel = {
    val metrics = appMaster.getClusterMetrics()
    
    // Check various resource utilization thresholds
    val cpuHigh = metrics.cpuUtilization > degradationThresholds("cpu_utilization")
    val memoryHigh = metrics.memoryUtilization > degradationThresholds("memory_utilization")
    val diskLow = metrics.diskSpaceRemaining < degradationThresholds("disk_space")
    
    val currentLevel = (cpuHigh, memoryHigh, diskLow) match {
      case (false, false, false) => ServiceLevel.NORMAL
      case (true, _, _) | (_, true, _) => ServiceLevel.DEGRADED_PERFORMANCE
      case (true, true, true) | (_, _, true) => ServiceLevel.LIMITED_FUNCTIONALITY
    }
    
    if (currentLevel != currentServiceLevel) {
      logWarning(s"Service level changed from $currentServiceLevel to $currentLevel")
      applyDegradationMeasures(currentLevel)
      currentServiceLevel = currentLevel
    }
    
    currentServiceLevel
  }
  
  private def applyDegradationMeasures(level: ServiceLevel): Unit = {
    level match {
      case ServiceLevel.DEGRADED_PERFORMANCE =>
        // Reduce task parallelism, increase batching
        applyPerformanceDegradation()
      case ServiceLevel.LIMITED_FUNCTIONALITY =>
        // Pause non-critical operations, reduce resource usage
        applyLimitedFunctionalityMode()
      case ServiceLevel.NORMAL =>
        // Resume normal operations
        resumeNormalOperations()
      case ServiceLevel.CRITICAL =>
        // Emergency measures
        applyCriticalMode()
    }
  }
  
  private def applyPerformanceDegradation(): Unit = {
    logInfo("Applying performance degradation measures")
    
    // Reduce executor parallelism
    appMaster.reduceExecutorParallelism(factor = 0.7)
    
    // Increase batching for I/O operations
    appMaster.increaseBatchSizes(factor = 1.5)
    
    // Enable more aggressive caching
    appMaster.enableAggressiveCaching()
    
    // Prioritize critical tasks
    appMaster.prioritizeCriticalTasks()
  }
  
  private def applyLimitedFunctionalityMode(): Unit = {
    logInfo("Applying limited functionality mode")
    
    // Pause non-critical background tasks
    appMaster.pauseBackgroundTasks()
    
    // Reduce checkpoint frequency
    appMaster.reduceCheckpointFrequency()
    
    // Limit concurrent task submissions
    appMaster.limitTaskSubmissionRate(to = 10) // 10 tasks per second
    
    // Activate emergency resource conservation
    appMaster.activateResourceConservation()
  }
  
  private def resumeNormalOperations(): Unit = {
    logInfo("Resuming normal operations")
    
    // Restore normal parallelism
    appMaster.restoreNormalParallelism()
    
    // Restore normal batch sizes
    appMaster.restoreNormalBatchSizes()
    
    // Resume paused tasks
    appMaster.resumeBackgroundTasks()
    
    // Restore normal checkpoint frequency
    appMaster.restoreNormalCheckpointFrequency()
  }
  
  private def applyCriticalMode(): Unit = {
    logInfo("Applying critical mode - emergency measures")
    
    // Immediate resource reduction
    appMaster.immediateResourceReduction()
    
    // Pause all non-essential operations
    appMaster.pauseAllNonEssentialOperations()
    
    // Prepare for potential application restart
    appMaster.prepareForRestart()
  }
  
  def getHealthStatus(): HealthStatus = {
    val serviceLevel = evaluateServiceLevel()
    val metrics = appMaster.getClusterMetrics()
    
    HealthStatus(
      serviceLevel = serviceLevel,
      cpuUtilization = metrics.cpuUtilization,
      memoryUtilization = metrics.memoryUtilization,
      diskSpaceRemaining = metrics.diskSpaceRemaining,
      activeExecutors = metrics.activeExecutors,
      pendingTasks = metrics.pendingTasks
    )
  }
}

sealed trait ServiceLevel
object ServiceLevel {
  case object NORMAL extends ServiceLevel
  case object DEGRADED_PERFORMANCE extends ServiceLevel
  case object LIMITED_FUNCTIONALITY extends ServiceLevel
  case object CRITICAL extends ServiceLevel
}

case class HealthStatus(
  serviceLevel: ServiceLevel,
  cpuUtilization: Double,
  memoryUtilization: Double,
  diskSpaceRemaining: Double,
  activeExecutors: Int,
  pendingTasks: Int
)

case class ClusterMetrics(
  cpuUtilization: Double,
  memoryUtilization: Double,
  diskSpaceRemaining: Double,
  networkBandwidthUtilization: Double,
  activeExecutors: Int,
  pendingTasks: Int,
  failedExecutors: Int
)
```

## Summary
The Fault Tolerance and Recovery mechanisms in RayDP include:
1. Robust executor failure detection using heartbeat monitoring
2. Comprehensive checkpoint and recovery systems for state preservation
3. Application-level state management for full recovery capabilities
4. Network partition handling with connectivity monitoring
5. Graceful degradation strategies to maintain service availability under stress

These mechanisms ensure that Spark applications running on RayDP can withstand various types of failures while maintaining data integrity and continuing operation with minimal disruption.