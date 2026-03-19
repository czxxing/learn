# RayDP Source Code Analysis Index

## Table of Contents

### Core Architecture Analysis
1. [01_OVERVIEW_ARCHITECTURE.md](01_OVERVIEW_ARCHITECTURE.md) - Project overview and architecture
2. [02_PYTHON_CONTEXT_MODULE.md](02_PYTHON_CONTEXT_MODULE.md) - Context module analysis
3. [03_PYTHON_SPARK_CLUSTER_MODULE.md](03_PYTHON_SPARK_CLUSTER_MODULE.md) - Spark cluster module analysis
4. [04_PYTHON_RAY_SPARK_MASTER_MODULE.md](04_PYTHON_RAY_SPARK_MASTER_MODULE.md) - Ray Spark master module analysis
5. [05_JAVA_SPARK_ON_RAY_CONFIGS_MODULE.md](05_JAVA_SPARK_ON_RAY_CONFIGS_MODULE.md) - Java Spark-on-Ray configs analysis
6. [06_SPARK_ON_RAY_CREATION_FLOW.md](06_SPARK_ON_RAY_CREATION_FLOW.md) - Spark-on-Ray creation flow analysis

### Spark-on-Ray Principles Analysis
1. [01_ARCHITECTURE_PRINCIPLES.md](spark_on_ray_principles/01_ARCHITECTURE_PRINCIPLES.md) - Architecture principles
2. [02_RESOURCE_MANAGEMENT_PRINCIPLES.md](spark_on_ray_principles/02_RESOURCE_MANAGEMENT_PRINCIPLES.md) - Resource management principles
3. [03_COMMUNICATION_MECHANISM_PRINCIPLES.md](spark_on_ray_principles/03_COMMUNICATION_MECHANISM_PRINCIPLES.md) - Communication mechanism principles
4. [04_DATA_EXCHANGE_MECHANISM_PRINCIPLES.md](spark_on_ray_principles/04_DATA_EXCHANGE_MECHANISM_PRINCIPLES.md) - Data exchange mechanism principles
5. [05_FAULT_TOLERANCE_PERFORMANCE_OPTIMIZATION.md](spark_on_ray_principles/05_FAULT_TOLERANCE_PERFORMANCE_OPTIMIZATION.md) - Fault tolerance and performance optimization

### Spark Master Creation Process Analysis
1. [06_SPARK_MASTER_CREATION_OVERVIEW.md](spark_on_ray_principles/06_SPARK_MASTER_CREATION_OVERVIEW.md) - Spark Master creation overview
2. [07_PYTHON_API_LAYER.md](spark_on_ray_principles/07_PYTHON_API_LAYER.md) - Python API layer analysis
3. [08_PY4J_GATEWAY_COMMUNICATION.md](spark_on_ray_principles/08_PY4J_GATEWAY_COMMUNICATION.md) - Py4J gateway communication analysis
4. [09_JAVA_PROCESS_LAUNCH.md](spark_on_ray_principles/09_JAVA_PROCESS_LAUNCH.md) - Java process launch analysis
5. [10_APP_MASTER_BRIDGE_INTERFACE.md](spark_on_ray_principles/10_APP_MASTER_BRIDGE_INTERFACE.md) - AppMaster bridge interface analysis
6. [11_RAY_APP_MASTER_CORE_IMPLEMENTATION.md](spark_on_ray_principles/11_RAY_APP_MASTER_CORE_IMPLEMENTATION.md) - Ray AppMaster core implementation
7. [12_SPARK_SESSION_INTEGRATION_AND_PYSPARK_INTERACTION.md](spark_on_ray_principles/12_SPARK_SESSION_INTEGRATION_AND_PYSPARK_INTERACTION.md) - SparkSession integration and PySpark interaction
8. [13_SPARK_MASTER_CREATION_SUMMARY.md](spark_on_ray_principles/13_SPARK_MASTER_CREATION_SUMMARY.md) - Spark Master creation summary

### Core Module Analysis
1. [06_CORE_PROJECT_STRUCTURE_AND_CLASS_RELATIONSHIP.md](../06_CORE_PROJECT_STRUCTURE_AND_CLASS_RELATIONSHIP.md) - Core module project structure and class relationship analysis

### RayDPExecutor Creation and Execution Analysis
1. [15_RAYDP_EXECUTOR_CREATION_AND_EXECUTION_OVERVIEW.md](15_00_RAYDP_EXECUTOR_CREATION_AND_EXECUTION_OVERVIEW.md) - RayDPExecutor creation and execution overview
2. [15_01_RAYAPPMASTER_AND_EXECUTOR_MANAGEMENT.md](15_01_RAYAPPMASTER_AND_EXECUTOR_MANAGEMENT.md) - RayAppMaster and Executor management analysis
3. [15_02_RAYDP_EXECUTOR_CREATION_AND_INITIALIZATION.md](15_02_RAYDP_EXECUTOR_CREATION_AND_INITIALIZATION.md) - RayDPExecutor creation and initialization analysis
4. [15_03_EXECUTOR_REGISTRATION_AND_COMMUNICATION.md](15_03_EXECUTOR_REGISTRATION_AND_COMMUNICATION.md) - Executor registration and communication analysis
5. [15_04_EXECUTOR_STARTUP_AND_ENVIRONMENT_PREPARATION.md](15_04_EXECUTOR_STARTUP_AND_ENVIRONMENT_PREPARATION.md) - Executor startup and environment preparation analysis
6. [15_05_SPARK_INTEGRATION_AND_TASK_EXECUTION.md](15_05_SPARK_INTEGRATION_AND_TASK_EXECUTION.md) - Spark integration and task execution analysis
7. [15_06_FAULT_HANDLING_AND_RECOVERY.md](15_06_FAULT_HANDLING_AND_RECOVERY.md) - Fault handling and recovery analysis