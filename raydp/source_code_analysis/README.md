# RayDP 源码分析

本项目包含对RayDP项目的详细源码分析，涵盖架构设计、核心模块、运行原理等多个方面。

## 分析文档目录

### 核心架构分析
- [00 - 项目总体架构分析](./00_OVERVIEW_ARCHITECTURE.md) - RayDP项目整体架构和设计思路
- [01 - Python Context模块分析](./01_PYTHON_CONTEXT_MODULE.md) - 上下文管理模块详细分析
- [02 - Python Spark Cluster模块分析](./02_PYTHON_SPARK_CLUSTER_MODULE.md) - Spark集群管理模块分析
- [03 - Python Ray Spark Master模块分析](./03_PYTHON_RAY_SPARK_MASTER_MODULE.md) - Ray Spark主节点模块分析
- [04 - Java Spark On Ray Configs模块分析](./04_JAVA_SPARK_ON_RAY_CONFIGS_MODULE.md) - Java配置管理模块分析
- [05 - Spark在Ray中创建流程分析](./05_SPARK_ON_RAY_CREATION_FLOW.md) - Spark应用在Ray中创建的详细流程

### Spark-on-Ray 运行原理深度分析
深入了解Spark如何在Ray环境中运行的核心原理：

#### 基础原理系列
- [架构原理分析](./spark_on_ray_principles/01_ARCHITECTURE_PRINCIPLES.md) - RayDP整体架构设计理念和组件关系
- [资源管理原理分析](./spark_on_ray_principles/02_RESOURCE_MANAGEMENT_PRINCIPLES.md) - Spark与Ray资源模型映射及分配策略
- [通信机制原理分析](./spark_on_ray_principles/03_COMMUNICATION_MECHANISM_PRINCIPLES.md) - Python与Java进程间通信机制详解
- [数据交换机制原理分析](./spark_on_ray_principles/04_DATA_EXCHANGE_MECHANISM_PRINCIPLES.md) - Spark与Ray间数据传输机制分析
- [容错与性能优化分析](./spark_on_ray_principles/05_FAULT_TOLERANCE_PERFORMANCE_OPTIMIZATION.md) - 容错机制和性能优化策略

#### Spark Master 创建过程深度分析
- [创建过程概述](./spark_on_ray_principles/06_SPARK_MASTER_CREATION_OVERVIEW.md) - Spark Master创建流程总体概览
- [Python API层分析](./spark_on_ray_principles/07_PYTHON_API_LAYER.md) - Python接口层实现分析
- [Py4J网关通信分析](./spark_on_ray_principles/08_PY4J_GATEWAY_COMMUNICATION.md) - Python与Java跨语言通信机制
- [Java进程启动分析](./spark_on_ray_principles/09_JAVA_PROCESS_LAUNCH.md) - Java AppMaster进程启动过程
- [AppMaster桥接接口分析](./spark_on_ray_principles/10_APP_MASTER_BRIDGE_INTERFACE.md) - Python-Java通信桥梁实现
- [Ray AppMaster核心实现分析](./spark_on_ray_principles/11_RAY_APP_MASTER_CORE_IMPLEMENTATION.md) - AppMaster核心功能实现
- [SparkSession集成与PySpark交互分析](./spark_on_ray_principles/12_SPARK_SESSION_INTEGRATION_AND_PYSPARK_INTERACTION.md) - SparkSession与PySpark集成机制
- [创建过程总结](./spark_on_ray_principles/13_SPARK_MASTER_CREATION_SUMMARY.md) - Spark Master创建流程完整总结

## 项目结构说明

- `./` : 主要架构和模块分析
- `./spark_on_ray_principles/` : Spark-on-Ray运行原理深度分析
- `./spark_run_on_ray/` : RayDP运行流程详细分析

### Core 模块分析
- [Core项目结构与类关系分析](./14_CORE_PROJECT_STRUCTURE_AND_CLASS_RELATIONSHIP.md) - Core模块项目结构和各个类的作用说明及项目关系