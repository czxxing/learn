# Spark在Ray中的容错与性能优化机制分析

## 概述

本文深入分析RayDP项目中Spark在Ray环境中运行时的容错机制和性能优化策略，包括故障检测、恢复机制、资源优化、调度优化等核心技术原理。

## 1. 容错架构设计

### 1.1 分层容错架构

RayDP采用多层容错架构，结合了Spark和Ray各自的容错机制：

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Spark Driver   │  │ Spark Executor  │  │ Ray Actor   │ │
│  │  (Fault Tolerant│  │ (Fault Tolerant│  │ (Fault Tol. │ │
│  │   Checkpoints)  │  │   Restart)     │  │   Recovery) │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Framework Layer                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │         Spark RDD Lineage & Checkpointing             │ │
│  │  ┌─────────────┐        ┌─────────────┐              │ │
│  │  │ Task DAG    │───────▶│ RDD Lineage │              │ │
│  │  │ (Recovery)  │◀───────│ (Replay)    │              │ │
│  │  └─────────────┘        └─────────────┘              │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Ray Runtime Layer                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │        Ray Actor Fault Tolerance & Recovery           │ │
│  │  ┌─────────────┐        ┌─────────────┐              │ │
│  │  │ Actor State │───────▶│ Snapshot &  │              │ │
│  │  │ (Replication│◀───────│  Restore    │              │ │
│  │  └─────────────┘        └─────────────┘              │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │        Node Failure Detection & Recovery              │ │
│  │  ┌─────────────┐        ┌─────────────┐              │ │
│  │  │ Health      │───────▶│ Node        │              │ │
│  │  │ Monitoring  │◀───────│ Failover    │              │ │
│  │  └─────────────┘        └─────────────┘              │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 容错组件关系

**核心容错组件**：
- **Spark Checkpointing**：RDD检查点机制
- **Ray Actor Recovery**：Ray Actor故障恢复
- **Task Retry Logic**：任务重试逻辑
- **Health Monitor**：健康监控系统

## 2. Spark层容错机制

### 2.1 RDD血统容错

**RDD血统机制**：
```python
class RDDLineageManager:
    def __init__(self):
        self.lineage_graph = nx.DiGraph()  # 使用NetworkX构建DAG
        self.checkpoint_manager = CheckpointManager()
        self.failure_detector = FailureDetector()
    
    def track_rdd_lineage(self, rdd_id, parent_rdds, operation):
        """
        跟踪RDD血统关系
        """
        # 添加当前RDD节点
        self.lineage_graph.add_node(rdd_id, 
                                   operation=operation,
                                   computed=False,
                                   checkpointed=False)
        
        # 添加父RDD边
        for parent_id in parent_rdds:
            self.lineage_graph.add_edge(parent_id, rdd_id)
    
    def recover_from_failure(self, failed_rdd_id):
        """
        从RDD失败中恢复
        """
        # 检查是否可以使用检查点
        if self.checkpoint_manager.is_checkpointed(failed_rdd_id):
            return self._recover_from_checkpoint(failed_rdd_id)
        
        # 否则重新计算整个依赖链
        return self._recompute_from_lineage(failed_rdd_id)
    
    def _recompute_from_lineage(self, failed_rdd_id):
        """
        从血统信息重新计算
        """
        # 找到需要重新计算的所有RDD
        ancestors = nx.ancestors(self.lineage_graph, failed_rdd_id)
        computation_order = list(nx.topological_sort(
            self.lineage_graph.subgraph(ancestors | {failed_rdd_id})
        ))
        
        # 按拓扑顺序重新计算
        for rdd_id in computation_order:
            if not self._is_computed(rdd_id):
                parent_ids = list(self.lineage_graph.predecessors(rdd_id))
                operation = self.lineage_graph.nodes[rdd_id]['operation']
                
                # 执行计算
                self._execute_operation(rdd_id, parent_ids, operation)
                
                # 标记为已计算
                self.lineage_graph.nodes[rdd_id]['computed'] = True
    
    def _recover_from_checkpoint(self, rdd_id):
        """
        从检查点恢复
        """
        checkpoint_path = self.checkpoint_manager.get_checkpoint_path(rdd_id)
        
        # 从检查点加载数据
        rdd_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # 更新状态
        self.lineage_graph.nodes[rdd_id]['computed'] = True
        self.lineage_graph.nodes[rdd_id]['checkpointed'] = True
        
        return rdd_data
    
    def should_checkpoint(self, rdd_id):
        """
        判断是否应该对RDD进行检查点
        """
        # 检查血统链长度
        ancestors = nx.ancestors(self.lineage_graph, rdd_id)
        lineage_length = len(ancestors)
        
        # 检查RDD大小
        estimated_size = self._estimate_rdd_size(rdd_id)
        
        # 检查重用频率
        reuse_count = self._get_rdd_reuse_count(rdd_id)
        
        # 综合判断
        should_checkpoint = (
            lineage_length > self.CHECKPOINT_LINEAGE_THRESHOLD or
            estimated_size > self.CHECKPOINT_SIZE_THRESHOLD or
            reuse_count > self.CHECKPOINT_REUSE_THRESHOLD
        )
        
        return should_checkpoint
```

### 2.2 任务级别容错

**任务重试机制**：
```python
class TaskRetryManager:
    def __init__(self, max_retries=3, retry_delay=1.0, exponential_backoff=True):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.failure_history = {}
        self.metrics_collector = MetricsCollector()
    
    def execute_task_with_retry(self, task, executor_handle):
        """
        带重试的任务执行
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # 记录尝试
                self.metrics_collector.increment_counter('task_attempts')
                
                # 执行任务
                result = executor_handle.execute_task(task)
                
                # 记录成功
                self.metrics_collector.increment_counter('task_successes')
                self.metrics_collector.observe_histogram('task_execution_time', 
                                                       time.time() - task.start_time)
                
                return result
                
            except TaskFailureException as e:
                last_exception = e
                self.metrics_collector.increment_counter('task_failures')
                
                if attempt < self.max_retries:
                    # 检查是否应该继续重试
                    if self._should_retry(task, e, attempt):
                        # 计算重试延迟
                        delay = self._calculate_retry_delay(attempt)
                        
                        # 等待后重试
                        time.sleep(delay)
                        
                        # 可能需要重新调度到不同节点
                        executor_handle = self._get_alternative_executor(task, e)
                    else:
                        # 不应该重试，直接失败
                        break
                else:
                    # 所有重试都失败
                    self.metrics_collector.increment_counter('task_permanent_failures')
                    break
        
        # 任务永久失败
        raise TaskPermanentFailureException(
            f"Task failed after {self.max_retries} retries: {last_exception}"
        )
    
    def _should_retry(self, task, exception, attempt):
        """
        判断是否应该重试任务
        """
        # 检查异常类型
        if isinstance(exception, NonRetryableException):
            return False
        
        # 检查失败原因
        if self._is_data_corruption_error(exception):
            return False
        
        # 检查资源不足
        if self._is_resource_unavailable_error(exception):
            return True
        
        # 检查网络错误
        if self._is_network_error(exception):
            return True
        
        # 默认重试
        return True
    
    def _calculate_retry_delay(self, attempt):
        """
        计算重试延迟
        """
        if self.exponential_backoff:
            base_delay = self.retry_delay * (2 ** attempt)
            # 添加随机抖动避免惊群效应
            jitter = random.uniform(0, base_delay * 0.1)
            return base_delay + jitter
        else:
            return self.retry_delay
    
    def _get_alternative_executor(self, task, failure_exception):
        """
        获取替代执行器
        """
        # 如果失败是由于特定节点问题
        if self._is_node_specific_failure(failure_exception):
            # 获取其他可用的执行器
            alternative_executors = self._find_alternative_executors(task)
            if alternative_executors:
                return random.choice(alternative_executors)
        
        # 否则返回原来的执行器
        return self._get_original_executor(task)
    
    def _is_node_specific_failure(self, exception):
        """
        检查是否是节点特定的失败
        """
        error_msg = str(exception).lower()
        node_specific_errors = [
            'node unavailable', 'connection refused', 'timeout', 'oom'
        ]
        return any(err in error_msg for err in node_specific_errors)
```

### 2.3 检查点机制

**检查点管理**：
```python
class CheckpointManager:
    def __init__(self, storage_backend='ray_object_store', 
                 checkpoint_strategy='adaptive'):
        self.storage_backend = storage_backend
        self.checkpoint_strategy = checkpoint_strategy
        self.checkpoint_locations = {}
        self.checkpoint_metadata = {}
        self.storage_client = self._initialize_storage_client()
    
    def create_checkpoint(self, rdd, checkpoint_path=None):
        """
        创建RDD检查点
        """
        if checkpoint_path is None:
            checkpoint_path = self._generate_checkpoint_path(rdd.id)
        
        try:
            # 物理化RDD数据
            materialized_data = self._materialize_rdd(rdd)
            
            # 保存到存储后端
            if self.storage_backend == 'ray_object_store':
                object_ref = ray.put(materialized_data)
                self.checkpoint_locations[rdd.id] = object_ref
            elif self.storage_backend == 'distributed_filesystem':
                self.storage_client.save(checkpoint_path, materialized_data)
                self.checkpoint_locations[rdd.id] = checkpoint_path
            elif self.storage_backend == 'memory_cache':
                self._save_to_memory_cache(rdd.id, materialized_data)
            
            # 更新元数据
            self.checkpoint_metadata[rdd.id] = {
                'created_at': time.time(),
                'size': self._estimate_size(materialized_data),
                'location': self.checkpoint_locations[rdd.id],
                'strategy': self.checkpoint_strategy
            }
            
            # 清除RDD的血统信息（优化）
            rdd.mark_checkpointed()
            
            return checkpoint_path
            
        except Exception as e:
            raise CheckpointCreationException(f"Failed to create checkpoint: {e}")
    
    def load_checkpoint(self, rdd_id):
        """
        加载检查点数据
        """
        if rdd_id not in self.checkpoint_locations:
            raise ValueError(f"No checkpoint found for RDD {rdd_id}")
        
        location = self.checkpoint_locations[rdd_id]
        
        try:
            if self.storage_backend == 'ray_object_store':
                return ray.get(location)
            elif self.storage_backend == 'distributed_filesystem':
                return self.storage_client.load(location)
            elif self.storage_backend == 'memory_cache':
                return self._load_from_memory_cache(rdd_id)
        except Exception as e:
            raise CheckpointLoadException(f"Failed to load checkpoint: {e}")
    
    def _materialize_rdd(self, rdd):
        """
        物理化RDD数据
        """
        # 触发RDD计算
        computed_partitions = []
        
        for partition_id in range(rdd.num_partitions):
            partition_data = rdd.compute_partition(partition_id)
            computed_partitions.append(partition_data)
        
        return computed_partitions
    
    def should_create_checkpoint(self, rdd):
        """
        判断是否应该创建检查点
        """
        if self.checkpoint_strategy == 'adaptive':
            return self._adaptive_checkpoint_decision(rdd)
        elif self.checkpoint_strategy == 'frequency_based':
            return self._frequency_based_decision(rdd)
        elif self.checkpoint_strategy == 'size_based':
            return self._size_based_decision(rdd)
        else:
            return False
    
    def _adaptive_checkpoint_decision(self, rdd):
        """
        自适应检查点决策
        """
        # 综合考虑多个因素
        lineage_length = self._calculate_lineage_length(rdd)
        estimated_size = self._estimate_rdd_size(rdd)
        reuse_probability = self._estimate_reuse_probability(rdd)
        
        # 权重计算
        score = (
            lineage_length * 0.4 +
            (estimated_size / self.SIZE_THRESHOLD) * 0.3 +
            reuse_probability * 0.3
        )
        
        return score > self.ADAPTIVE_THRESHOLD
    
    def _frequency_based_decision(self, rdd):
        """
        基于频率的检查点决策
        """
        # 检查RDD被使用的频率
        usage_count = self._get_rdd_usage_count(rdd.id)
        return usage_count >= self.FREQUENCY_THRESHOLD
    
    def _size_based_decision(self, rdd):
        """
        基于大小的检查点决策
        """
        size = self._estimate_rdd_size(rdd)
        return size >= self.SIZE_THRESHOLD
```

## 3. Ray层容错机制

### 3.1 Actor故障恢复

**Ray Actor容错**：
```python
import ray
from ray import serve
import logging

class RayActorFaultTolerance:
    def __init__(self, checkpoint_interval=300, replica_count=1):
        self.checkpoint_interval = checkpoint_interval
        self.replica_count = replica_count
        self.actor_snapshots = {}
        self.actor_health_status = {}
        self.logger = logging.getLogger(__name__)
    
    @ray.remote
    class FaultTolerantSparkExecutor:
        def __init__(self, executor_id, checkpoint_enabled=True):
            self.executor_id = executor_id
            self.state = {}
            self.checkpoint_enabled = checkpoint_enabled
            self.last_checkpoint_time = time.time()
            self.task_queue = []
            self.running_tasks = {}
            
            # 初始化Spark执行器
            self.spark_executor = self._initialize_spark_executor()
        
        def execute_task(self, task_description):
            """
            执行Spark任务（带故障恢复）
            """
            try:
                # 更新状态
                self._update_state('executing', task_description.task_id)
                
                # 执行任务
                result = self.spark_executor.execute_task(task_description)
                
                # 任务完成后更新状态
                self._update_state('completed', task_description.task_id)
                
                return result
                
            except Exception as e:
                # 记录错误
                self._update_state('failed', task_description.task_id)
                self.logger.error(f"Task execution failed: {e}")
                
                # 尝试恢复
                self._attempt_recovery(task_description, e)
                
                raise e
        
        def _update_state(self, status, task_id=None):
            """
            更新执行器状态
            """
            self.state.update({
                'status': status,
                'last_update': time.time(),
                'current_task': task_id,
                'spark_executor_state': self.spark_executor.get_state()
            })
            
            # 定期创建检查点
            if (self.checkpoint_enabled and 
                time.time() - self.last_checkpoint_time > self.checkpoint_interval):
                self._create_checkpoint()
        
        def _create_checkpoint(self):
            """
            创建执行器状态检查点
            """
            try:
                # 序列化当前状态
                state_snapshot = {
                    'state': copy.deepcopy(self.state),
                    'task_queue': copy.deepcopy(self.task_queue),
                    'running_tasks': copy.deepcopy(self.running_tasks),
                    'spark_executor_state': self.spark_executor.get_state()
                }
                
                # 保存到Ray对象存储
                self.checkpoint_ref = ray.put(state_snapshot)
                self.last_checkpoint_time = time.time()
                
                self.logger.info(f"Checkpoint created for executor {self.executor_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to create checkpoint: {e}")
        
        def restore_from_checkpoint(self, checkpoint_ref):
            """
            从检查点恢复
            """
            try:
                # 获取检查点数据
                checkpoint_data = ray.get(checkpoint_ref)
                
                # 恢复状态
                self.state = checkpoint_data['state']
                self.task_queue = checkpoint_data['task_queue']
                self.running_tasks = checkpoint_data['running_tasks']
                
                # 恢复Spark执行器状态
                self.spark_executor.restore_state(
                    checkpoint_data['spark_executor_state']
                )
                
                self.logger.info(f"Executor {self.executor_id} restored from checkpoint")
                
            except Exception as e:
                self.logger.error(f"Failed to restore from checkpoint: {e}")
                raise e
        
        def _attempt_recovery(self, task_description, error):
            """
            尝试恢复执行器
            """
            if self._is_recoverable_error(error):
                # 尝试重启Spark执行器
                try:
                    self.spark_executor.shutdown()
                    self.spark_executor = self._initialize_spark_executor()
                    
                    # 重新提交任务
                    self.logger.info(f"Executor recovered, resubmitting task {task_description.task_id}")
                    
                except Exception as recovery_error:
                    self.logger.error(f"Recovery failed: {recovery_error}")
                    # 如果恢复失败，可能需要重建Actor
                    raise recovery_error
    
    def create_fault_tolerant_executor(self, executor_id):
        """
        创建容错执行器
        """
        executor_actor = self.FaultTolerantSparkExecutor.remote(
            executor_id=executor_id
        )
        
        # 监控Actor健康状态
        self._monitor_actor_health(executor_actor)
        
        return executor_actor
    
    def _monitor_actor_health(self, actor_handle):
        """
        监控Actor健康状态
        """
        # 使用Ray的内置健康检查
        def health_check():
            try:
                # 定期ping Actor
                ray.get(actor_handle.ping.remote(), timeout=10.0)
                self.actor_health_status[actor_handle._actor_id.hex()] = 'healthy'
            except ray.exceptions.RayActorError:
                self.actor_health_status[actor_handle._actor_id.hex()] = 'failed'
                # 触发恢复流程
                self._handle_actor_failure(actor_handle)
            except ray.exceptions.GetTimeoutError:
                self.actor_health_status[actor_handle._actor_id.hex()] = 'unresponsive'
        
        # 启动健康检查线程
        health_thread = threading.Thread(target=self._run_health_checks, 
                                       args=(actor_handle,))
        health_thread.daemon = True
        health_thread.start()
    
    def _handle_actor_failure(self, failed_actor):
        """
        处理Actor失败
        """
        actor_id = failed_actor._actor_id.hex()
        
        # 尝试从检查点恢复
        checkpoint_ref = self.actor_snapshots.get(actor_id)
        if checkpoint_ref:
            try:
                # 创建新的Actor实例
                new_actor = self.FaultTolerantSparkExecutor.options(
                    name=f"executor_{actor_id}_recovered"
                ).remote(executor_id=actor_id)
                
                # 恢复状态
                ray.get(new_actor.restore_from_checkpoint.remote(checkpoint_ref))
                
                # 更新引用
                self._update_actor_references(failed_actor, new_actor)
                
                self.logger.info(f"Actor {actor_id} recovered from checkpoint")
                
            except Exception as e:
                self.logger.error(f"Failed to recover actor from checkpoint: {e}")
                # 如果检查点恢复失败，可能需要完全重建
                self._rebuild_actor(failed_actor)
        else:
            # 没有检查点，完全重建
            self._rebuild_actor(failed_actor)
    
    def _rebuild_actor(self, failed_actor):
        """
        重建失败的Actor
        """
        actor_id = failed_actor._actor_id.hex()
        
        # 创建新的Actor
        new_actor = self.FaultTolerantSparkExecutor.remote(
            executor_id=actor_id
        )
        
        # 重新分配任务
        self._redistribute_tasks(failed_actor, new_actor)
        
        self.logger.info(f"Actor {actor_id} rebuilt successfully")
```

### 3.2 Placement Group容错

**Placement Group故障处理**：
```python
class PlacementGroupFaultTolerance:
    def __init__(self, ray_cluster_manager):
        self.ray_cluster_manager = ray_cluster_manager
        self.placement_groups = {}
        self.pg_health_monitors = {}
        self.recovery_strategies = {}
    
    def create_fault_tolerant_placement_group(self, bundles, strategy='STRICT_SPREAD'):
        """
        创建容错的Placement Group
        """
        # 创建Placement Group
        pg = ray.util.placement_group(bundles, strategy=strategy)
        
        # 等待就绪
        ray.get(pg.ready())
        
        pg_id = pg.id.hex()
        self.placement_groups[pg_id] = pg
        
        # 启动健康监控
        self._start_health_monitoring(pg)
        
        # 设置恢复策略
        self.recovery_strategies[pg_id] = self._determine_recovery_strategy(bundles)
        
        return pg
    
    def _start_health_monitoring(self, placement_group):
        """
        启动Placement Group健康监控
        """
        def monitor_pg_health():
            pg_id = placement_group.id.hex()
            
            while True:
                try:
                    # 检查Placement Group状态
                    pg_info = ray.util.placement_group_table(placement_group)
                    
                    if pg_info['state'] != 'CREATED':
                        # Placement Group出现问题
                        self._handle_pg_failure(placement_group, pg_info)
                        break
                    
                    # 检查各个bundle的状态
                    self._check_bundle_health(placement_group)
                    
                    time.sleep(5)  # 每5秒检查一次
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring PG {pg_id}: {e}")
                    time.sleep(10)  # 出错后稍等更长时间
        
        monitor_thread = threading.Thread(target=monitor_pg_health)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        self.pg_health_monitors[placement_group.id.hex()] = monitor_thread
    
    def _check_bundle_health(self, placement_group):
        """
        检查Placement Group中各个bundle的健康状况
        """
        pg_info = ray.util.placement_group_table(placement_group)
        
        if 'bundles' in pg_info:
            for bundle_id, bundle_info in pg_info['bundles'].items():
                if bundle_info['state'] != 'ALIVE':
                    # 某个bundle失败
                    self._handle_bundle_failure(placement_group, bundle_id, bundle_info)
    
    def _handle_pg_failure(self, placement_group, pg_info):
        """
        处理Placement Group失败
        """
        pg_id = placement_group.id.hex()
        failure_reason = pg_info.get('state', 'UNKNOWN')
        
        self.logger.error(f"Placement Group {pg_id} failed: {failure_reason}")
        
        # 根据恢复策略进行处理
        recovery_strategy = self.recovery_strategies.get(pg_id, 'recreate')
        
        if recovery_strategy == 'recreate':
            self._recreate_placement_group(placement_group)
        elif recovery_strategy == 'partial_rebuild':
            self._partial_rebuild_placement_group(placement_group)
        elif recovery_strategy == 'failover':
            self._failover_placement_group(placement_group)
    
    def _recreate_placement_group(self, failed_pg):
        """
        重新创建Placement Group
        """
        pg_id = failed_pg.id.hex()
        
        # 获取原始配置
        original_bundles = self._get_original_bundles(failed_pg)
        original_strategy = self._get_original_strategy(failed_pg)
        
        # 尝试创建新的Placement Group
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            try:
                new_pg = ray.util.placement_group(
                    original_bundles, 
                    strategy=original_strategy
                )
                
                # 等待就绪
                ray.get(new_pg.ready())
                
                # 更新引用
                self.placement_groups[pg_id] = new_pg
                
                # 重新启动监控
                self._start_health_monitoring(new_pg)
                
                self.logger.info(f"Placement Group {pg_id} recreated successfully")
                break
                
            except Exception as e:
                attempts += 1
                self.logger.error(f"Attempt {attempts} to recreate PG failed: {e}")
                
                if attempts >= max_attempts:
                    raise RuntimeError(f"Failed to recreate PG after {max_attempts} attempts")
                
                # 等待一段时间后重试
                time.sleep(10 * attempts)
    
    def _determine_recovery_strategy(self, bundles):
        """
        根据bundle配置确定恢复策略
        """
        total_bundles = len(bundles)
        
        if total_bundles <= 2:
            # 小规模PG，直接重建
            return 'recreate'
        elif total_bundles <= 10:
            # 中等规模PG，部分重建
            return 'partial_rebuild'
        else:
            # 大规模PG，故障转移
            return 'failover'
```

## 4. 性能优化策略

### 4.1 资源调度优化

**智能调度算法**：
```python
class IntelligentSchedulingOptimizer:
    def __init__(self, cluster_resource_manager, task_scheduler):
        self.cluster_rm = cluster_resource_manager
        self.task_scheduler = task_scheduler
        self.performance_predictor = PerformancePredictor()
        self.load_balancer = LoadBalancer()
    
    def schedule_task_optimally(self, task, resource_requirements):
        """
        智能任务调度
        """
        # 获取可用节点
        available_nodes = self.cluster_rm.get_available_nodes()
        
        # 过滤满足资源要求的节点
        qualified_nodes = [
            node for node in available_nodes
            if self._meets_resource_requirements(node, resource_requirements)
        ]
        
        # 计算每个节点的综合评分
        node_scores = {}
        for node in qualified_nodes:
            score = self._calculate_node_score(node, task, resource_requirements)
            node_scores[node.id] = score
        
        # 选择最佳节点
        if node_scores:
            best_node_id = max(node_scores.keys(), key=node_scores.get)
            best_node = next(n for n in qualified_nodes if n.id == best_node_id)
            
            # 调度任务
            return self.task_scheduler.schedule_on_node(task, best_node)
        else:
            # 没有合适的节点，等待或扩容
            return self._handle_no_suitable_nodes(task, resource_requirements)
    
    def _calculate_node_score(self, node, task, resource_reqs):
        """
        计算节点综合评分
        """
        score = 0.0
        
        # 资源充足度评分 (0.0-0.3)
        resource_score = self._calculate_resource_score(node, resource_reqs)
        score += resource_score * 0.3
        
        # 数据本地性评分 (0.0-0.25)
        locality_score = self._calculate_locality_score(node, task)
        score += locality_score * 0.25
        
        # 节点负载评分 (0.0-0.2)
        load_score = self._calculate_load_score(node)
        score += load_score * 0.2
        
        # 网络延迟评分 (0.0-0.15)
        network_score = self._calculate_network_score(node, task)
        score += network_score * 0.15
        
        # 预测性能评分 (0.0-0.1)
        perf_score = self.performance_predictor.predict_performance(node, task)
        score += perf_score * 0.1
        
        return score
    
    def _calculate_resource_score(self, node, resource_reqs):
        """
        计算资源充足度评分
        """
        score = 1.0  # 基础分
        
        for resource_type, required in resource_reqs.items():
            available = node.resources.get(resource_type, 0)
            if available < required:
                return 0.0  # 不满足要求直接返回0
            
            # 计算剩余资源比例
            remaining_ratio = (available - required) / available
            score *= (0.5 + 0.5 * remaining_ratio)  # 保留一定余量
        
        return min(score, 1.0)
    
    def _calculate_locality_score(self, node, task):
        """
        计算数据本地性评分
        """
        if not hasattr(task, 'data_locations') or not task.data_locations:
            return 0.1  # 没有数据位置信息，给较低分数
        
        local_data_size = sum(
            size for loc, size in task.data_locations.items()
            if loc == node.id or loc in node.direct_connected_nodes
        )
        
        total_data_size = sum(task.data_locations.values())
        
        if total_data_size == 0:
            return 0.1
        
        locality_ratio = local_data_size / total_data_size
        return 0.1 + 0.9 * locality_ratio  # 0.1-1.0的范围
    
    def _calculate_load_score(self, node):
        """
        计算节点负载评分
        """
        current_load = self.load_balancer.get_node_load(node.id)
        # 负载越低，分数越高
        return max(0.0, 1.0 - current_load)
    
    def _calculate_network_score(self, node, task):
        """
        计算网络延迟评分
        """
        # 获取与数据源的网络距离
        if hasattr(task, 'data_source_nodes'):
            avg_latency = self._calculate_avg_network_latency(
                node, task.data_source_nodes
            )
            # 延迟越低，分数越高
            normalized_latency = min(avg_latency / 100.0, 1.0)  # 假设100ms为基准
            return max(0.0, 1.0 - normalized_latency)
        
        return 0.5  # 默认中等分数
    
    def _calculate_avg_network_latency(self, target_node, source_nodes):
        """
        计算平均网络延迟
        """
        total_latency = 0
        count = 0
        
        for source_node in source_nodes:
            latency = self.cluster_rm.get_network_latency(target_node.id, source_node)
            total_latency += latency
            count += 1
        
        return total_latency / count if count > 0 else 50.0  # 默认50ms
```

### 4.2 内存优化策略

**内存管理优化**：
```python
class MemoryOptimizationManager:
    def __init__(self):
        self.memory_allocator = MemoryAllocator()
        self.garbage_collector = GarbageCollector()
        self.cache_manager = CacheManager()
        self.mmu = MemoryManagementUnit()  # 内存管理单元
    
    def optimize_memory_usage(self, task_context):
        """
        优化任务内存使用
        """
        # 分析任务内存需求
        mem_profile = self._profile_memory_requirements(task_context)
        
        # 申请最优内存配置
        allocation = self.memory_allocator.allocate_optimal(
            mem_profile.required_memory,
            mem_profile.temp_memory_ratio,
            mem_profile.cache_friendly
        )
        
        # 设置内存限制
        self._configure_memory_limits(allocation, task_context)
        
        # 启动内存监控
        self._start_memory_monitoring(task_context, allocation)
        
        return allocation
    
    def _profile_memory_requirements(self, task_context):
        """
        分析内存需求
        """
        # 获取任务的基本信息
        input_size = self._estimate_input_size(task_context)
        operation_type = task_context.operation_type
        parallelism = task_context.parallelism
        
        # 基于操作类型估算内存需求
        base_memory = self._get_base_memory_requirement(operation_type)
        
        # 考虑输入数据大小
        data_processing_memory = input_size * self.DATA_PROCESSING_FACTOR
        
        # 考虑并行度
        parallel_memory = base_memory * parallelism * self.PARALLEL_FACTOR
        
        # 总内存需求
        total_required = base_memory + data_processing_memory + parallel_memory
        
        # 临时内存比例（用于中间计算）
        temp_ratio = self._calculate_temp_memory_ratio(operation_type)
        
        # 是否适合缓存
        cache_friendly = self._is_cache_friendly_operation(operation_type)
        
        return MemoryProfile(
            required_memory=total_required,
            temp_memory_ratio=temp_ratio,
            cache_friendly=cache_friendly,
            operation_type=operation_type
        )
    
    def _get_base_memory_requirement(self, operation_type):
        """
        获取基本内存需求
        """
        base_requirements = {
            'map': 64 * 1024 * 1024,      # 64MB
            'filter': 32 * 1024 * 1024,   # 32MB
            'reduce': 128 * 1024 * 1024,  # 128MB
            'join': 256 * 1024 * 1024,    # 256MB
            'aggregate': 192 * 1024 * 1024, # 192MB
            'sort': 512 * 1024 * 1024,    # 512MB
            'default': 128 * 1024 * 1024   # 128MB
        }
        
        return base_requirements.get(operation_type, base_requirements['default'])
    
    def _calculate_temp_memory_ratio(self, operation_type):
        """
        计算临时内存比例
        """
        temp_ratios = {
            'map': 0.1,      # 10% 临时内存
            'filter': 0.05,  # 5% 临时内存
            'reduce': 0.3,   # 30% 临时内存
            'join': 0.5,     # 50% 临时内存
            'aggregate': 0.25, # 25% 临时内存
            'sort': 0.8,     # 80% 临时内存
            'default': 0.2   # 20% 临时内存
        }
        
        return temp_ratios.get(operation_type, temp_ratios['default'])
    
    def _configure_memory_limits(self, allocation, task_context):
        """
        配置内存限制
        """
        # 设置JVM堆内存
        heap_size = int(allocation.total_memory * allocation.heap_ratio)
        self._set_jvm_heap_size(heap_size)
        
        # 设置执行内存
        exec_memory = int(allocation.total_memory * allocation.exec_ratio)
        self._set_execution_memory(exec_memory)
        
        # 设置存储内存
        storage_memory = int(allocation.total_memory * allocation.storage_ratio)
        self._set_storage_memory(storage_memory)
    
    def _start_memory_monitoring(self, task_context, allocation):
        """
        启动内存监控
        """
        def monitor_memory_usage():
            last_warning_time = 0
            warning_interval = 30  # 30秒警告间隔
            
            while not task_context.is_completed():
                current_usage = self.mmu.get_current_usage()
                usage_ratio = current_usage / allocation.total_memory
                
                if usage_ratio > 0.9:  # 超过90%使用率
                    current_time = time.time()
                    if current_time - last_warning_time > warning_interval:
                        self._trigger_memory_pressure_response(allocation)
                        last_warning_time = current_time
                
                time.sleep(1)  # 每秒检查一次
        
        monitor_thread = threading.Thread(target=monitor_memory_usage)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def _trigger_memory_pressure_response(self, allocation):
        """
        触发内存压力响应
        """
        # 尝试释放缓存
        freed_memory = self.cache_manager.release_least_used()
        
        # 启动垃圾回收
        self.garbage_collector.force_collection()
        
        # 如果仍然内存紧张，考虑溢出到磁盘
        if self.mmu.get_current_usage() / allocation.total_memory > 0.85:
            self._enable_disk_spilling()

class MemoryProfile:
    def __init__(self, required_memory, temp_memory_ratio, cache_friendly, operation_type):
        self.required_memory = required_memory
        self.temp_memory_ratio = temp_memory_ratio
        self.cache_friendly = cache_friendly
        self.operation_type = operation_type

class MemoryAllocation:
    def __init__(self, total_memory, heap_ratio=0.6, exec_ratio=0.3, storage_ratio=0.1):
        self.total_memory = total_memory
        self.heap_ratio = heap_ratio
        self.exec_ratio = exec_ratio
        self.storage_ratio = storage_ratio
```

### 4.3 缓存优化策略

**智能缓存管理**：
```python
class IntelligentCacheManager:
    def __init__(self, cache_size_limit, eviction_policy='hybrid_lru_mfu'):
        self.cache_size_limit = cache_size_limit
        self.eviction_policy = eviction_policy
        self.cache_store = {}
        self.access_frequency = collections.Counter()
        self.recency_tracker = collections.OrderedDict()
        self.cache_statistics = CacheStatistics()
        
        # 缓存成本效益分析器
        self.cost_benefit_analyzer = CostBenefitAnalyzer()
    
    def get_or_compute(self, key, compute_func, ttl=None):
        """
        获取缓存或计算结果
        """
        # 检查缓存
        if key in self.cache_store:
            cached_item = self.cache_store[key]
            
            # 检查是否过期
            if ttl and time.time() - cached_item.created_at > ttl:
                self._remove_from_cache(key)
            else:
                # 更新访问统计
                self._record_access(key)
                self.cache_statistics.record_hit()
                
                return cached_item.value
        
        # 缓存未命中，执行计算
        self.cache_statistics.record_miss()
        
        # 检查是否值得缓存（成本效益分析）
        if self.cost_benefit_analyzer.should_cache(compute_func, key):
            # 计算结果
            result = compute_func()
            
            # 存储到缓存
            self._put_in_cache(key, result, ttl)
            
            return result
        else:
            # 不值得缓存，直接返回计算结果
            return compute_func()
    
    def _put_in_cache(self, key, value, ttl=None):
        """
        将数据放入缓存
        """
        # 检查缓存大小限制
        item_size = self._estimate_size(value)
        
        if self._get_current_cache_size() + item_size > self.cache_size_limit:
            # 需要驱逐一些项目
            self._evict_items(item_size)
        
        # 创建缓存项
        cache_item = CacheItem(
            value=value,
            size=item_size,
            created_at=time.time(),
            ttl=ttl
        )
        
        # 存储到缓存
        self.cache_store[key] = cache_item
        
        # 更新访问统计
        self._record_access(key)
        
        # 更新统计
        self.cache_statistics.record_put(item_size)
    
    def _evict_items(self, required_size):
        """
        驱逐缓存项以腾出空间
        """
        while self._get_current_cache_size() + required_size > self.cache_size_limit:
            if not self.cache_store:
                break  # 没有更多项目可驱逐
            
            # 根据驱逐策略选择要驱逐的键
            key_to_evict = self._select_key_for_eviction()
            if key_to_evict is None:
                break
            
            # 移除项目
            self._remove_from_cache(key_to_evict)
    
    def _select_key_for_eviction(self):
        """
        根据驱逐策略选择要驱逐的键
        """
        if self.eviction_policy == 'lru':
            # 最近最少使用
            if self.recency_tracker:
                return next(iter(self.recency_tracker))
        
        elif self.eviction_policy == 'mfu':
            # 最频繁使用
            if self.access_frequency:
                return self.access_frequency.most_common()[-1][0]  # 最少使用的
        
        elif self.eviction_policy == 'hybrid_lru_mfu':
            # 混合LRU-MFU策略
            return self._hybrid_eviction_selection()
        
        elif self.eviction_policy == 'cost_based':
            # 基于成本的驱逐
            return self._cost_based_eviction()
        
        # 默认使用LRU
        return next(iter(self.recency_tracker)) if self.recency_tracker else None
    
    def _hybrid_eviction_selection(self):
        """
        混合驱逐选择策略
        """
        # 结合访问频率和最近性
        if not self.cache_store:
            return None
        
        candidates = list(self.cache_store.keys())
        
        # 计算每个候选者的综合评分（频率权重0.6，最近性权重0.4）
        scores = {}
        max_freq = max(self.access_frequency.values()) if self.access_frequency else 1
        max_recency = len(self.recency_tracker) if self.recency_tracker else 1
        
        for key in candidates:
            freq_score = self.access_frequency.get(key, 0) / max_freq if max_freq > 0 else 0
            recency_score = (list(self.recency_tracker.keys()).index(key) if key in self.recency_tracker 
                           else max_recency) / max_recency if max_recency > 0 else 0
            
            # 评分越低越应该被驱逐
            composite_score = 0.6 * freq_score + 0.4 * recency_score
            scores[key] = composite_score
        
        # 返回评分最低的（最应该被驱逐的）
        return min(scores.keys(), key=scores.get) if scores else None
    
    def _cost_based_eviction(self):
        """
        基于成本的驱逐策略
        """
        if not self.cache_store:
            return None
        
        # 计算每个缓存项的成本效益比
        cost_benefit_ratios = {}
        
        for key, item in self.cache_store.items():
            # 成本：存储成本（大小）+ 维护成本（时间）
            storage_cost = item.size
            maintenance_cost = (time.time() - item.created_at) * 0.001  # 时间成本
            total_cost = storage_cost + maintenance_cost
            
            # 效益：访问频率
            benefit = self.access_frequency.get(key, 0)
            
            # 成本效益比：比率越低越应该被驱逐
            ratio = total_cost / (benefit + 1)  # +1 避免除零
            cost_benefit_ratios[key] = ratio
        
        # 返回成本效益比最低的
        return min(cost_benefit_ratios.keys(), key=cost_benefit_ratios.get)
    
    def _record_access(self, key):
        """
        记录访问行为
        """
        # 更新访问频率
        self.access_frequency[key] += 1
        
        # 更新最近性追踪（LRU）
        if key in self.recency_tracker:
            del self.recency_tracker[key]
        self.recency_tracker[key] = time.time()
        
        # 清理过期的LRU条目
        while len(self.recency_tracker) > self.cache_size_limit // 1024:  # 假设平均每个key 1KB
            self.recency_tracker.popitem(last=False)
    
    def _get_current_cache_size(self):
        """
        获取当前缓存大小
        """
        return sum(item.size for item in self.cache_store.values())
    
    def _estimate_size(self, obj):
        """
        估算对象大小
        """
        try:
            return sys.getsizeof(obj)
        except:
            # 如果无法估算，使用默认大小
            return 1024  # 1KB default
    
    def get_cache_efficiency(self):
        """
        获取缓存效率指标
        """
        stats = self.cache_statistics.get_stats()
        hit_rate = stats['hits'] / (stats['hits'] + stats['misses']) if (stats['hits'] + stats['misses']) > 0 else 0
        utilization = self._get_current_cache_size() / self.cache_size_limit if self.cache_size_limit > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'utilization': utilization,
            'efficiency_score': hit_rate * utilization
        }

class CacheItem:
    def __init__(self, value, size, created_at, ttl=None):
        self.value = value
        self.size = size
        self.created_at = created_at
        self.ttl = ttl

class CacheStatistics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.total_size_put = 0
    
    def record_hit(self):
        self.hits += 1
    
    def record_miss(self):
        self.misses += 1
    
    def record_put(self, size):
        self.puts += 1
        self.total_size_put += size
    
    def get_stats(self):
        return {
            'hits': self.hits,
            'misses': self.misses,
            'puts': self.puts,
            'total_size_put': self.total_size_put,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }

class CostBenefitAnalyzer:
    def __init__(self):
        self.computation_costs = {}  # 存储计算成本的历史数据
        self.access_patterns = {}    # 存储访问模式
    
    def should_cache(self, compute_func, key):
        """
        基于成本效益分析决定是否缓存
        """
        # 估算计算成本
        estimated_cost = self._estimate_computation_cost(compute_func)
        
        # 估算缓存收益
        estimated_benefit = self._estimate_caching_benefit(key)
        
        # 计算成本效益比
        cost_benefit_ratio = estimated_cost / estimated_benefit if estimated_benefit > 0 else float('inf')
        
        # 如果成本效益比大于阈值，则缓存
        return cost_benefit_ratio > self.COST_BENEFIT_THRESHOLD
    
    def _estimate_computation_cost(self, compute_func):
        """
        估算计算成本
        """
        # 如果有历史数据，使用历史平均值
        func_name = compute_func.__name__ if hasattr(compute_func, '__name__') else str(compute_func)
        
        if func_name in self.computation_costs:
            return self.computation_costs[func_name]['avg_cost']
        
        # 否则进行采样估算
        sample_times = []
        for _ in range(3):  # 采样3次
            start_time = time.time()
            # 注意：这里不能实际执行函数，只是估算
            # 实际实现中会使用更复杂的估算方法
            sample_times.append(time.time() - start_time)
        
        avg_cost = sum(sample_times) / len(sample_times)
        self.computation_costs[func_name] = {
            'avg_cost': avg_cost,
            'sample_count': 3
        }
        
        return avg_cost
    
    def _estimate_caching_benefit(self, key):
        """
        估算缓存收益
        """
        # 基于历史访问模式估算
        access_count = self.access_patterns.get(key, {}).get('access_count', 0)
        access_frequency = self.access_patterns.get(key, {}).get('frequency', 0)
        
        # 收益与访问频率和次数成正比
        benefit = access_count * access_frequency * self.ACCESS_BENEFIT_FACTOR
        
        # 更新访问模式统计
        if key not in self.access_patterns:
            self.access_patterns[key] = {'access_count': 0, 'frequency': 0}
        
        self.access_patterns[key]['access_count'] += 1
        # 更新频率（简单的滑动窗口）
        current_time = time.time()
        self.access_patterns[key]['last_access'] = current_time
        
        return benefit
```

## 5. 性能监控与调优

### 5.1 性能指标收集

**监控系统架构**：
```python
class PerformanceMonitoringSystem:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.performance_analyzer = PerformanceAnalyzer()
        self.optimizer = PerformanceOptimizer()
        
        # 启动监控循环
        self.monitoring_active = True
        self.start_monitoring()
    
    def start_monitoring(self):
        """
        启动性能监控
        """
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # 收集各种性能指标
                    self._collect_system_metrics()
                    self._collect_spark_metrics()
                    self._collect_ray_metrics()
                    
                    # 分析性能数据
                    insights = self.performance_analyzer.analyze_current_state()
                    
                    # 根据分析结果进行优化
                    self.optimizer.apply_optimizations(insights)
                    
                    # 检查是否需要告警
                    self._check_thresholds_and_alert(insights)
                    
                    time.sleep(self.MONITORING_INTERVAL)
                    
                except Exception as e:
                    self.alert_manager.send_alert('MONITORING_ERROR', str(e))
                    time.sleep(10)  # 出错后稍等
        
        monitor_thread = threading.Thread(target=monitoring_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def _collect_system_metrics(self):
        """
        收集系统级指标
        """
        import psutil
        
        system_metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters(),
            'network_io': psutil.net_io_counters(),
            'process_count': len(psutil.pids()),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        }
        
        self.metrics_collector.record_batch(system_metrics)
    
    def _collect_spark_metrics(self):
        """
        收集Spark指标
        """
        try:
            # 通过Spark的MetricsSystem收集指标
            spark_metrics = {}
            
            # 获取Spark执行器指标
            if SparkContext._active_spark_context:
                # 从Spark的指标系统获取数据
                pass  # 这里需要根据实际的Spark指标系统实现
            
            # 获取任务指标
            task_metrics = self._get_current_task_metrics()
            spark_metrics.update(task_metrics)
            
            self.metrics_collector.record_batch(spark_metrics)
            
        except Exception as e:
            self.metrics_collector.record_error('spark_metrics_collection', str(e))
    
    def _collect_ray_metrics(self):
        """
        收集Ray指标
        """
        try:
            # 获取Ray集群指标
            cluster_metrics = ray.cluster_resources()
            
            # 获取Ray任务指标
            task_metrics = ray.tasks()
            
            # 获取对象存储指标
            object_store_metrics = ray.internal.internal_api.memory_summary()
            
            combined_metrics = {
                **cluster_metrics,
                **task_metrics,
                **object_store_metrics
            }
            
            self.metrics_collector.record_batch(combined_metrics)
            
        except Exception as e:
            self.metrics_collector.record_error('ray_metrics_collection', str(e))
    
    def _get_current_task_metrics(self):
        """
        获取当前任务指标
        """
        # 这里应该连接到Spark的执行上下文
        # 获取当前正在执行的任务指标
        return {}
    
    def _check_thresholds_and_alert(self, insights):
        """
        检查阈值并发送告警
        """
        # CPU使用率过高
        if insights.get('cpu_utilization', 0) > self.HIGH_CPU_THRESHOLD:
            self.alert_manager.send_alert(
                'HIGH_CPU_USAGE',
                f"CPU utilization is high: {insights['cpu_utilization']:.2f}%"
            )
        
        # 内存使用率过高
        if insights.get('memory_utilization', 0) > self.HIGH_MEMORY_THRESHOLD:
            self.alert_manager.send_alert(
                'HIGH_MEMORY_USAGE',
                f"Memory utilization is high: {insights['memory_utilization']:.2f}%"
            )
        
        # 任务执行时间过长
        if insights.get('slow_tasks', 0) > self.SLOW_TASK_THRESHOLD:
            self.alert_manager.send_alert(
                'SLOW_TASKS_DETECTED',
                f"Detected {insights['slow_tasks']} slow tasks"
            )
        
        # 错误率过高
        if insights.get('error_rate', 0) > self.HIGH_ERROR_RATE_THRESHOLD:
            self.alert_manager.send_alert(
                'HIGH_ERROR_RATE',
                f"Error rate is high: {insights['error_rate']:.2f}%"
            )

class MetricsCollector:
    def __init__(self):
        self.metrics_buffer = collections.deque(maxlen=10000)  # 循环缓冲区
        self.aggregated_metrics = {}
        self.metric_lock = threading.Lock()
    
    def record(self, metric_name, value, tags=None):
        """
        记录单个指标
        """
        with self.metric_lock:
            metric_entry = {
                'name': metric_name,
                'value': value,
                'timestamp': time.time(),
                'tags': tags or {}
            }
            self.metrics_buffer.append(metric_entry)
    
    def record_batch(self, metrics_dict):
        """
        批量记录指标
        """
        with self.metric_lock:
            timestamp = time.time()
            for name, value in metrics_dict.items():
                metric_entry = {
                    'name': name,
                    'value': value,
                    'timestamp': timestamp,
                    'tags': {}
                }
                self.metrics_buffer.append(metric_entry)
    
    def get_recent_metrics(self, minutes=5):
        """
        获取最近的指标数据
        """
        cutoff_time = time.time() - (minutes * 60)
        
        with self.metric_lock:
            recent = [m for m in self.metrics_buffer if m['timestamp'] >= cutoff_time]
        
        return recent
    
    def aggregate_metrics(self, window_minutes=1):
        """
        聚合指标数据
        """
        recent_metrics = self.get_recent_metrics(window_minutes)
        
        aggregated = {}
        for metric in recent_metrics:
            name = metric['name']
            if name not in aggregated:
                aggregated[name] = {'values': [], 'timestamps': []}
            
            aggregated[name]['values'].append(metric['value'])
            aggregated[name]['timestamps'].append(metric['timestamp'])
        
        # 计算聚合值
        results = {}
        for name, data in aggregated.items():
            values = data['values']
            results[name] = {
                'mean': sum(values) / len(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'count': len(values),
                'trend': self._calculate_trend(values)
            }
        
        return results
    
    def _calculate_trend(self, values):
        """
        计算指标趋势
        """
        if len(values) < 2:
            return 0
        
        # 简单的线性回归斜率计算
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(a * b for a, b in zip(x, y))
        sum_xx = sum(a * a for a in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0
        return slope
```

### 5.2 自动调优机制

**自适应调优系统**：
```python
class AdaptiveTuningSystem:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.performance_predictor = MLPerformancePredictor()
        self.tuning_history = []
        self.optimization_lock = threading.Lock()
    
    def adaptive_tuning_cycle(self):
        """
        自适应调优周期
        """
        while True:
            try:
                # 评估当前配置性能
                current_performance = self._evaluate_current_configuration()
                
                # 分析性能瓶颈
                bottlenecks = self._identify_bottlenecks(current_performance)
                
                # 生成调优建议
                tuning_recommendations = self._generate_tuning_recommendations(
                    bottlenecks, current_performance
                )
                
                # 应用调优（如果改进明显）
                if self._should_apply_tuning(tuning_recommendations):
                    self._apply_tuning(tuning_recommendations)
                
                # 记录调优历史
                self._record_tuning_event(current_performance, tuning_recommendations)
                
                time.sleep(self.TUNING_CYCLE_INTERVAL)
                
            except Exception as e:
                logging.error(f"Adaptive tuning cycle failed: {e}")
                time.sleep(60)  # 出错后等待1分钟
    
    def _evaluate_current_configuration(self):
        """
        评估当前配置性能
        """
        # 收集当前性能指标
        metrics = self._collect_performance_metrics()
        
        # 计算性能评分
        performance_score = self._calculate_performance_score(metrics)
        
        # 识别配置参数的影响
        parameter_impact = self._analyze_parameter_impact()
        
        return {
            'metrics': metrics,
            'score': performance_score,
            'parameter_impact': parameter_impact
        }
    
    def _identify_bottlenecks(self, performance_data):
        """
        识别性能瓶颈
        """
        bottlenecks = []
        metrics = performance_data['metrics']
        
        # 检查CPU瓶颈
        if metrics.get('cpu_utilization', 0) > self.CPU_BOTTLENECK_THRESHOLD:
            bottlenecks.append({
                'type': 'cpu',
                'severity': metrics['cpu_utilization'] / 100.0,
                'suggested_action': 'increase_parallelism_or_resources'
            })
        
        # 检查内存瓶颈
        if metrics.get('memory_utilization', 0) > self.MEMORY_BOTTLENECK_THRESHOLD:
            bottlenecks.append({
                'type': 'memory',
                'severity': metrics['memory_utilization'] / 100.0,
                'suggested_action': 'adjust_memory_settings_or_add_resources'
            })
        
        # 检查I/O瓶颈
        if metrics.get('io_wait_time', 0) > self.IO_BOTTLENECK_THRESHOLD:
            bottlenecks.append({
                'type': 'io',
                'severity': metrics['io_wait_time'] / self.IO_BOTTLENECK_THRESHOLD,
                'suggested_action': 'optimize_data_locality_or_increase_io_bandwidth'
            })
        
        # 检查网络瓶颈
        if metrics.get('network_latency', 0) > self.NETWORK_BOTTLENECK_THRESHOLD:
            bottlenecks.append({
                'type': 'network',
                'severity': metrics['network_latency'] / self.NETWORK_BOTTLENECK_THRESHOLD,
                'suggested_action': 'optimize_data_partitioning_or_improve_network'
            })
        
        return bottlenecks
    
    def _generate_tuning_recommendations(self, bottlenecks, current_performance):
        """
        生成调优建议
        """
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'cpu':
                recommendation = self._generate_cpu_tuning_recommendation(
                    bottleneck, current_performance
                )
            elif bottleneck['type'] == 'memory':
                recommendation = self._generate_memory_tuning_recommendation(
                    bottleneck, current_performance
                )
            elif bottleneck['type'] == 'io':
                recommendation = self._generate_io_tuning_recommendation(
                    bottleneck, current_performance
                )
            elif bottleneck['type'] == 'network':
                recommendation = self._generate_network_tuning_recommendation(
                    bottleneck, current_performance
                )
            else:
                continue
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_cpu_tuning_recommendation(self, bottleneck, perf_data):
        """
        生成CPU调优建议
        """
        current_parallelism = self.config_manager.get('spark.sql.adaptive.coalescePartitions.enabled')
        current_cores = self.config_manager.get('spark.executor.cores')
        
        recommendation = {
            'parameter': 'parallelism',
            'current_value': current_parallelism,
            'recommended_value': self._suggest_parallelism_increase(current_cores),
            'confidence': bottleneck['severity'],
            'expected_improvement': bottleneck['severity'] * 0.1  # 估计改善程度
        }
        
        return recommendation
    
    def _generate_memory_tuning_recommendation(self, bottleneck, perf_data):
        """
        生成内存调优建议
        """
        current_memory = self.config_manager.get('spark.executor.memory')
        current_fraction = self.config_manager.get('spark.memory.fraction')
        
        # 基于当前使用情况建议调整
        suggested_memory = self._adjust_memory_based_on_usage(
            current_memory, perf_data['metrics']['memory_utilization']
        )
        
        recommendation = {
            'parameter': 'memory',
            'current_value': current_memory,
            'recommended_value': suggested_memory,
            'confidence': bottleneck['severity'],
            'expected_improvement': bottleneck['severity'] * 0.15
        }
        
        return recommendation
    
    def _should_apply_tuning(self, recommendations):
        """
        判断是否应该应用调优
        """
        if not recommendations:
            return False
        
        # 检查改进建议的置信度
        avg_confidence = sum(r['confidence'] for r in recommendations) / len(recommendations)
        
        # 检查预期改进是否显著
        avg_expected_improvement = sum(
            r['expected_improvement'] for r in recommendations
        ) / len(recommendations)
        
        # 应用调优的条件
        return (avg_confidence > self.MIN_CONFIDENCE_THRESHOLD and 
                avg_expected_improvement > self.MIN_IMPROVEMENT_THRESHOLD)
    
    def _apply_tuning(self, recommendations):
        """
        应用调优建议
        """
        with self.optimization_lock:
            for recommendation in recommendations:
                param = recommendation['parameter']
                new_value = recommendation['recommended_value']
                
                # 应用配置更改
                self.config_manager.update(param, new_value)
                
                # 如果是Spark配置，可能需要重启某些组件
                if param.startswith('spark.'):
                    self._apply_spark_config_change(param, new_value)
    
    def _apply_spark_config_change(self, param, new_value):
        """
        应用Spark配置更改
        """
        # 对于某些配置，可能需要重启执行器或应用
        if param in ['spark.executor.cores', 'spark.executor.memory']:
            # 这些配置通常需要重启执行器才能生效
            logging.info(f"Configuration {param} changed, executor restart may be needed")
        else:
            # 其他配置可能可以动态应用
            pass

class MLPerformancePredictor:
    """
    使用机器学习预测性能的类
    """
    def __init__(self):
        self.model = self._initialize_model()
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
    
    def predict_performance(self, config_params, workload_features):
        """
        预测给定配置下的性能
        """
        if not self.is_trained:
            # 如果模型未训练，使用简单的启发式方法
            return self._simple_performance_prediction(config_params, workload_features)
        
        # 提取特征
        features = self.feature_extractor.extract(config_params, workload_features)
        
        # 使用模型预测
        prediction = self.model.predict([features])
        
        return prediction[0]
    
    def _initialize_model(self):
        """
        初始化ML模型（这里使用简单的模型作为示例）
        """
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def train(self, historical_data):
        """
        使用历史数据训练模型
        """
        X = []
        y = []
        
        for record in historical_data:
            features = self.feature_extractor.extract(
                record['config'], record['workload']
            )
            X.append(features)
            y.append(record['performance'])
        
        if X and y:
            self.model.fit(X, y)
            self.is_trained = True

class FeatureExtractor:
    """
    特征提取器
    """
    def extract(self, config_params, workload_features):
        """
        从配置参数和工作负载特征中提取ML特征
        """
        features = []
        
        # 配置参数特征
        features.extend([
            config_params.get('spark.executor.cores', 1),
            config_params.get('spark.executor.memory', 1024),
            config_params.get('spark.sql.adaptive.enabled', 0),
            config_params.get('spark.serializer', 'kyro').count('kyro'),  # 简化的序列化器特征
        ])
        
        # 工作负载特征
        features.extend([
            workload_features.get('data_size_gb', 0),
            workload_features.get('operation_complexity', 1),
            workload_features.get('parallelism_need', 1),
        ])
        
        return features
```

## 总结

RayDP的容错与性能优化机制通过多层次的设计实现了高可靠性和高性能：

1. **分层容错**：结合了Spark RDD血统容错、Ray Actor故障恢复、任务重试等多种机制
2. **智能调度**：基于资源、本地性、负载等多维度的智能调度算法
3. **内存优化**：动态内存分配、垃圾回收、溢出策略等优化措施
4. **缓存优化**：智能缓存策略、成本效益分析、多种驱逐算法
5. **监控调优**：全面的性能监控、自动调优、机器学习预测

这套机制确保了Spark应用在Ray环境中的稳定运行和优异性能，为大规模数据处理提供了可靠的技术保障。