# 模块3: Java进程启动分析

## 概述

本模块详细分析RayDP中Java进程的启动机制，包括AppMasterEntryPoint的启动过程、JVM参数配置、进程间通信建立等关键技术。

## 1. Java进程启动架构

### 1.1 进程启动流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Python Process                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │ raydp.init_spark│  │ Subprocess    │  │ Connection Setup      │ │
│  │ ()              │  │ Management    │  │ (File-based)          │ │
│  │                 │  │                 │  │                         │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌───────────────────┐ │ │
│  │ │Prepare      │ │  │ │Launch Java  │ │  │ │Wait for Port    │ │ │
│  │ │Config       │ │  │ │Process      │ │  │ │Info from File   │ │ │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └───────────────────┘ │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Java Process (AppMasterEntryPoint)             │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    JVM Startup                                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │ │
│  │  │JVM Params   │  │Class Loading│  │GatewayServer Start    │ │ │
│  │  │Setup        │  │& Init       │  │(Py4J Bridge)          │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Process Communication Channel                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │Temp File      │  │TCP Socket     │  │Shared Memory         │ │
│  │(Port Info)    │  │(Commands)     │  │(Object Transfer)     │ │
│  │                 │  │                 │  │                         │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌───────────────────┐ │ │
│  │ │Write Port  │ │  │ │Establish    │ │  │ │Share Objects     │ │ │
│  │ │to File     │ │  │ │Connection   │ │  │ │via Plasma        │ │ │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └───────────────────┘ │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Java进程启动流程

### 2.1 Python端启动流程

```python
# raydp/spark/app_master_launcher.py
import subprocess
import os
import tempfile
import time
import struct

class AppMasterLauncher:
    def __init__(self):
        self.java_proc = None
        self.conn_info_path = None
    
    def start_java_process(self):
        """
        启动Java进程
        """
        # 1. 创建连接信息文件
        self.conn_info_path = self._create_connection_info_file()
        
        # 2. 设置环境变量
        env = self._prepare_environment()
        
        # 3. 构建Java命令
        java_cmd = self._build_java_command()
        
        # 4. 启动Java进程
        self.java_proc = self._launch_java_subprocess(java_cmd, env)
        
        # 5. 等待端口信息
        port = self._wait_for_port_info()
        
        return port
    
    def _create_connection_info_file(self):
        """
        创建连接信息文件
        """
        # 使用临时目录创建唯一的连接信息文件
        temp_dir = tempfile.gettempdir()
        file_name = f"raydp_appmaster_conn_{os.getpid()}_{int(time.time())}.tmp"
        conn_info_path = os.path.join(temp_dir, file_name)
        
        print(f"Creating connection info file: {conn_info_path}")
        return conn_info_path
    
    def _prepare_environment(self):
        """
        准备Java进程的环境变量
        """
        env = os.environ.copy()
        
        # 设置连接信息文件路径
        env["_RAYDP_APPMASTER_CONN_INFO_PATH"] = self.conn_info_path
        
        # 可能还需要设置其他环境变量
        # 例如Java堆大小、GC参数等
        if "JAVA_OPTS" not in env:
            env["JAVA_OPTS"] = "-Xmx2g -XX:+UseG1GC"
        
        return env
    
    def _build_java_command(self):
        """
        构建Java启动命令
        """
        # 获取classpath
        classpath = self._get_java_classpath()
        
        # 获取Java代理路径
        javaagent_path = self._get_java_agent_path()
        
        # 构建命令参数
        cmd = [
            "java",
            "-cp", classpath,
            f"-javaagent:{javaagent_path}",
            # JVM参数
            "-Xmx2g",  # 堆内存
            "-Xms512m",  # 初始堆内存
            "-XX:+UseG1GC",  # 使用G1垃圾收集器
            "-XX:MaxGCPauseMillis=200",  # 最大GC暂停时间
            # 系统属性
            "-Dspark.master=ray",  # Spark主节点设置
            "-Dspark.driver.host=localhost",  # 驱动程序主机
            # 主类
            "org.apache.spark.deploy.raydp.AppMasterEntryPoint"
        ]
        
        return cmd
    
    def _get_java_classpath(self):
        """
        获取Java classpath
        """
        # 从RayDP工具获取Spark相关的classpath
        from raydp import utils
        return utils.get_spark_classpath()
    
    def _get_java_agent_path(self):
        """
        获取Java agent路径
        """
        from raydp import utils
        return utils.get_spark_javaagent_path()
    
    def _launch_java_subprocess(self, cmd, env):
        """
        启动Java子进程
        """
        print(f"Launching Java process with command: {' '.join(cmd)}")
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid  # 创建新的进程组
            )
            
            print(f"Java process started with PID: {proc.pid}")
            return proc
            
        except FileNotFoundError:
            raise RuntimeError("Java executable not found. Please ensure Java is installed and in PATH.")
        except Exception as e:
            raise RuntimeError(f"Failed to launch Java process: {e}")
    
    def _wait_for_port_info(self):
        """
        等待Java进程写入端口信息
        """
        max_wait_time = 60  # 最大等待60秒
        poll_interval = 0.1  # 每0.1秒轮询一次
        elapsed = 0
        
        print(f"Waiting for port info in file: {self.conn_info_path}")
        
        while elapsed < max_wait_time:
            if os.path.exists(self.conn_info_path):
                try:
                    with open(self.conn_info_path, 'rb') as f:
                        # 读取4字节的端口号（整数）
                        port_bytes = f.read(4)
                        if len(port_bytes) == 4:
                            # 使用struct解析4字节整数（大端序）
                            port = struct.unpack('>I', port_bytes)[0]
                            
                            print(f"Successfully read port {port} from {self.conn_info_path}")
                            return port
                        else:
                            print(f"Insufficient bytes read from file, got {len(port_bytes)} bytes")
                except Exception as e:
                    print(f"Error reading port from {self.conn_info_path}: {e}")
            
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        # 等待超时，检查Java进程状态
        if self.java_proc and self.java_proc.poll() is not None:
            # Java进程已退出，获取错误输出
            stdout, stderr = self.java_proc.communicate()
            raise RuntimeError(
                f"Java process exited prematurely. Exit code: {self.java_proc.returncode}\n"
                f"Stdout: {stdout.decode()}\n"
                f"Stderr: {stderr.decode()}"
            )
        
        raise RuntimeError(
            f"Timed out waiting for gateway port after {max_wait_time}s. "
            f"Java process may still be running with PID: {self.java_proc.pid if self.java_proc else 'unknown'}"
        )
```

### 2.2 Java端启动流程

```java
// org.apache.spark.deploy.raydp.AppMasterEntryPoint
package org.apache.spark.deploy.raydp;

import py4j.GatewayServer;
import java.io.RandomAccessFile;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Properties;

public class AppMasterEntryPoint {
    private static GatewayServer gatewayServer;
    private static AppMasterBridge bridge;

    public static void main(String[] args) {
        try {
            System.out.println("AppMasterEntryPoint starting...");
            
            // 1. 读取连接信息文件路径
            String connInfoPath = System.getenv("_RAYDP_APPMASTER_CONN_INFO_PATH");
            if (connInfoPath == null || connInfoPath.isEmpty()) {
                throw new RuntimeException(
                    "Connection info path not set. Expected environment variable: _RAYDP_APPMASTER_CONN_INFO_PATH"
                );
            }
            
            System.out.println("Connection info path: " + connInfoPath);
            
            // 2. 初始化桥接对象
            bridge = new AppMasterBridge();
            
            // 3. 创建并启动网关服务器
            gatewayServer = createAndStartGatewayServer(bridge);
            
            // 4. 获取并写入端口信息
            int port = gatewayServer.getListeningPort();
            writePortToFile(connInfoPath, port);
            
            System.out.println("Gateway server started on port: " + port);
            
            // 5. 初始化AppMaster
            bridge.initialize();
            
            System.out.println("AppMaster initialized successfully.");
            
            // 6. 保持进程运行
            keepProcessRunning();
            
        } catch (Exception e) {
            System.err.println("Failed to start AppMasterEntryPoint: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    /**
     * 创建并启动网关服务器
     */
    private static GatewayServer createAndStartGatewayServer(AppMasterBridge bridge) throws Exception {
        System.out.println("Creating GatewayServer...");
        
        // 创建网关服务器
        GatewayServer server = new GatewayServer(
            bridge,                                    // 入口点对象
            0,                                         // 自动分配端口
            new InetSocketAddress("localhost", 0),    // 绑定地址
            GatewayServer.DEFAULT_CONNECT_TIMEOUT,    // 连接超时
            GatewayServer.DEFAULT_READ_TIMEOUT,       // 读取超时
            null                                       // 监听器
        );
        
        System.out.println("Starting GatewayServer...");
        
        // 启动服务器
        server.startup();
        
        System.out.println("GatewayServer startup completed.");
        
        return server;
    }
    
    /**
     * 将端口写入文件
     */
    private static void writePortToFile(String filePath, int port) throws IOException {
        System.out.println("Writing port " + port + " to file: " + filePath);
        
        // 确保父目录存在
        java.io.File file = new java.io.File(filePath);
        file.getParentFile().mkdirs();
        
        try (RandomAccessFile randomAccessFile = new RandomAccessFile(file, "rw");
             FileChannel channel = randomAccessFile.getChannel()) {
            
            // 使用ByteBuffer写入4字节整数（大端序）
            ByteBuffer buffer = ByteBuffer.allocate(4);
            buffer.putInt(port);  // 写入端口号
            buffer.flip();        // 翻转buffer，准备写入
            
            // 写入文件
            int bytesWritten = channel.write(buffer);
            
            if (bytesWritten != 4) {
                throw new IOException("Failed to write complete port number to file. Wrote " + bytesWritten + " bytes.");
            }
            
            System.out.println("Successfully wrote port " + port + " to file.");
        }
    }
    
    /**
     * 保持进程运行
     */
    private static void keepProcessRunning() {
        System.out.println("Keeping process running...");
        
        // 使用同步块和wait()来保持进程运行
        synchronized (AppMasterEntryPoint.class) {
            while (true) {
                try {
                    // 永久等待，直到被中断
                    AppMasterEntryPoint.class.wait();
                } catch (InterruptedException e) {
                    System.out.println("AppMasterEntryPoint interrupted, shutting down...");
                    Thread.currentThread().interrupt();  // 重新设置中断状态
                    break;
                }
            }
        }
        
        // 关闭网关服务器
        shutdownGatewayServer();
    }
    
    /**
     * 关闭网关服务器
     */
    private static void shutdownGatewayServer() {
        if (gatewayServer != null) {
            try {
                System.out.println("Shutting down GatewayServer...");
                gatewayServer.shutdown();
                System.out.println("GatewayServer shut down successfully.");
            } catch (Exception e) {
                System.err.println("Error shutting down GatewayServer: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
}
```

## 3. JVM参数配置分析

### 3.1 JVM参数设置策略

```python
class JVMParameterBuilder:
    def __init__(self):
        self.parameters = []
        self.system_properties = {}
    
    def set_memory_options(self, heap_size="2g", initial_heap="512m", 
                          metaspace_size="256m"):
        """
        设置内存相关参数
        """
        self.parameters.extend([
            f"-Xmx{heap_size}",      # 最大堆内存
            f"-Xms{initial_heap}",   # 初始堆内存
            f"-XX:MetaspaceSize={metaspace_size}",  # 元空间初始大小
            "-XX:+UseG1GC",         # 使用G1垃圾收集器
            "-XX:MaxGCPauseMillis=200",  # 最大GC暂停时间
            "-XX:G1HeapRegionSize=16m",  # G1区域大小
        ])
        return self
    
    def set_gc_options(self, gc_log_file=None, gc_verbose=False):
        """
        设置GC相关参数
        """
        gc_options = [
            "-XX:+UseG1GC",
            "-XX:MaxGCPauseMillis=200",
            "-XX:G1HeapRegionSize=16m",
            "-XX:G1MixedGCCountTarget=8",
            "-XX:InitiatingHeapOccupancyPercent=35",
        ]
        
        if gc_verbose:
            gc_options.extend([
                "-verbose:gc",
                "-XX:+PrintGCDetails",
                "-XX:+PrintGCTimeStamps",
            ])
        
        if gc_log_file:
            gc_options.extend([
                f"-Xloggc:{gc_log_file}",
                "-XX:+UseGCLogFileRotation",
                "-XX:NumberOfGCLogFiles=5",
                "-XX:GCLogFileSize=10M",
            ])
        
        self.parameters.extend(gc_options)
        return self
    
    def set_system_properties(self, properties):
        """
        设置系统属性
        """
        for key, value in properties.items():
            self.system_properties[key] = value
        return self
    
    def set_spark_specific_options(self):
        """
        设置Spark特定的JVM选项
        """
        spark_props = {
            "spark.master": "ray",
            "spark.submit.deployMode": "client",
            "spark.sql.adaptive.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        }
        
        for key, value in spark_props.items():
            self.parameters.append(f"-D{key}={value}")
        
        return self
    
    def build(self):
        """
        构建最终的JVM参数列表
        """
        result = self.parameters.copy()
        
        # 添加系统属性
        for key, value in self.system_properties.items():
            result.append(f"-D{key}={value}")
        
        return result

# 使用示例
def build_java_command_with_optimized_jvm():
    jvm_builder = JVMParameterBuilder()
    
    # 设置内存选项
    jvm_builder.set_memory_options(
        heap_size="4g",
        initial_heap="1g",
        metaspace_size="512m"
    )
    
    # 设置GC选项
    jvm_builder.set_gc_options(gc_verbose=True)
    
    # 设置Spark特定选项
    jvm_builder.set_spark_specific_options()
    
    # 构建JVM参数
    jvm_args = jvm_builder.build()
    
    # 构建完整命令
    cmd = ["java"] + jvm_args + [
        "-cp", get_classpath(),
        "-javaagent:" + get_javaagent_path(),
        "org.apache.spark.deploy.raydp.AppMasterEntryPoint"
    ]
    
    return cmd
```

### 3.2 JVM参数验证

```python
class JVMParameterValidator:
    @staticmethod
    def validate_heap_size(heap_size_str):
        """
        验证堆大小参数
        """
        import re
        
        # 支持的单位：g/G, m/M, k/K
        pattern = r'^(\d+)([gGmMkK]?)$'
        match = re.match(pattern, heap_size_str)
        
        if not match:
            raise ValueError(f"Invalid heap size format: {heap_size_str}")
        
        size_val = int(match.group(1))
        unit = match.group(2).upper() if match.group(2) else 'M'  # 默认单位为M
        
        # 验证合理性
        if unit == 'K':
            actual_size = size_val
        elif unit == 'M':
            actual_size = size_val * 1024 * 1024
        elif unit == 'G':
            actual_size = size_val * 1024 * 1024 * 1024
        else:
            raise ValueError(f"Unsupported unit: {unit}")
        
        # 检查最小值（至少1MB）
        if actual_size < 1024 * 1024:
            raise ValueError(f"Heap size too small: {heap_size_str}. Minimum is 1M.")
        
        # 检查最大值（不超过系统内存的80%）
        import psutil
        total_memory = psutil.virtual_memory().total
        max_recommended = int(total_memory * 0.8)
        
        if actual_size > max_recommended:
            raise ValueError(
                f"Heap size {heap_size_str} exceeds 80% of system memory. "
                f"Recommended maximum: {max_recommended / (1024**3):.1f}G"
            )
        
        return True
    
    @staticmethod
    def validate_gc_parameters(params):
        """
        验证GC相关参数
        """
        gc_params = [p for p in params if 'gc' in p.lower() or 'g1' in p.lower()]
        
        # 检查GC参数的合理性
        for param in gc_params:
            if '-XX:+UseG1GC' in param:
                # 检查G1相关的参数是否合理
                if any('-XX:+UseParallelGC' in p or '-XX:+UseConcMarkSweepGC' in p for p in params):
                    raise ValueError("Multiple GC algorithms specified. Choose one: G1, Parallel, or CMS.")
        
        return True
```

## 4. 进程间通信机制

### 4.1 文件通信机制

```python
import os
import time
import struct
import threading
from pathlib import Path

class FileBasedCommunication:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or Path(tempfile.gettempdir())
        self.comm_files = {}
    
    def create_communication_file(self, prefix="raydp_comm", suffix=".tmp"):
        """
        创建通信文件
        """
        file_path = self.base_dir / f"{prefix}_{os.getpid()}_{int(time.time() * 1000)}{suffix}"
        self.comm_files[file_path.name] = file_path
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Created communication file: {file_path}")
        return file_path
    
    def write_port_to_file(self, file_path, port):
        """
        将端口写入文件
        """
        try:
            # 使用二进制模式写入4字节端口号
            with open(file_path, 'wb') as f:
                port_bytes = struct.pack('>I', port)  # 大端序4字节整数
                f.write(port_bytes)
            
            print(f"Successfully wrote port {port} to {file_path}")
            return True
        except Exception as e:
            print(f"Failed to write port to file {file_path}: {e}")
            return False
    
    def read_port_from_file(self, file_path, timeout=60):
        """
        从文件读取端口
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        port_bytes = f.read(4)
                        if len(port_bytes) == 4:
                            port = struct.unpack('>I', port_bytes)[0]
                            print(f"Successfully read port {port} from {file_path}")
                            return port
                        else:
                            print(f"Insufficient bytes in {file_path}, retrying...")
                else:
                    print(f"File {file_path} does not exist yet, retrying...")
            
            except Exception as e:
                print(f"Error reading from {file_path}: {e}")
            
            time.sleep(0.1)  # 等待0.1秒后重试
        
        raise TimeoutError(f"Timeout waiting for port info in {file_path}")
    
    def write_data_to_file(self, file_path, data, format_type='json'):
        """
        将数据写入通信文件
        """
        import json
        import pickle
        
        try:
            with open(file_path, 'wb') as f:
                if format_type == 'json':
                    json_data = json.dumps(data).encode('utf-8')
                    f.write(json_data)
                elif format_type == 'pickle':
                    pickle.dump(data, f)
                elif format_type == 'binary':
                    f.write(data)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
            
            print(f"Successfully wrote data to {file_path}")
            return True
        except Exception as e:
            print(f"Failed to write data to {file_path}: {e}")
            return False
    
    def read_data_from_file(self, file_path, format_type='json'):
        """
        从通信文件读取数据
        """
        import json
        import pickle
        
        try:
            with open(file_path, 'rb') as f:
                if format_type == 'json':
                    content = f.read().decode('utf-8')
                    return json.loads(content)
                elif format_type == 'pickle':
                    return pickle.load(f)
                elif format_type == 'binary':
                    return f.read()
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:
            print(f"Failed to read data from {file_path}: {e}")
            return None
    
    def cleanup_communication_files(self):
        """
        清理通信文件
        """
        for file_path in self.comm_files.values():
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed communication file: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

# Java端对应的实现
"""
// Java端文件通信辅助类
public class FileCommunicationHelper {
    public static boolean writePortToFile(String filePath, int port) {
        try {
            Path path = Paths.get(filePath);
            Files.createDirectories(path.getParent()); // 确保目录存在
            
            try (RandomAccessFile file = new RandomAccessFile(filePath, "rw");
                 FileChannel channel = file.getChannel()) {
                
                ByteBuffer buffer = ByteBuffer.allocate(4);
                buffer.putInt(port);
                buffer.flip();
                
                channel.write(buffer);
            }
            
            System.out.println("Wrote port " + port + " to " + filePath);
            return true;
        } catch (Exception e) {
            System.err.println("Failed to write port to file: " + e.getMessage());
            return false;
        }
    }
    
    public static Integer readPortFromFile(String filePath, int timeoutSeconds) {
        long startTime = System.currentTimeMillis();
        
        while ((System.currentTimeMillis() - startTime) < timeoutSeconds * 1000) {
            try {
                Path path = Paths.get(filePath);
                if (Files.exists(path)) {
                    try (RandomAccessFile file = new RandomAccessFile(filePath, "r");
                         FileChannel channel = file.getChannel()) {
                        
                        ByteBuffer buffer = ByteBuffer.allocate(4);
                        int bytesRead = channel.read(buffer);
                        
                        if (bytesRead == 4) {
                            buffer.flip();
                            int port = buffer.getInt();
                            System.out.println("Read port " + port + " from " + filePath);
                            return port;
                        }
                    }
                }
            } catch (Exception e) {
                System.out.println("Error reading from " + filePath + ": " + e.getMessage());
            }
            
            try {
                Thread.sleep(100); // 等待100毫秒
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        
        return null;
    }
}
"""
```

### 4.2 信号处理机制

```python
import signal
import atexit
import os
import subprocess
import time

class ProcessSignalHandler:
    def __init__(self, java_processes=None):
        self.java_processes = java_processes or []
        self.cleanup_functions = []
        self.setup_signal_handlers()
        self.setup_cleanup_handlers()
    
    def setup_signal_handlers(self):
        """
        设置信号处理器
        """
        signal.signal(signal.SIGTERM, self._handle_termination_signal)
        signal.signal(signal.SIGINT, self._handle_interrupt_signal)
        signal.signal(signal.SIGHUP, self._handle_hangup_signal)
    
    def setup_cleanup_handlers(self):
        """
        设置退出清理处理器
        """
        atexit.register(self._cleanup_on_exit)
    
    def add_java_process(self, proc):
        """
        添加Java进程到管理列表
        """
        self.java_processes.append(proc)
    
    def _handle_termination_signal(self, signum, frame):
        """
        处理SIGTERM信号
        """
        print(f"Received SIGTERM ({signum}), initiating graceful shutdown...")
        self._graceful_shutdown()
        os._exit(0)  # 强制退出
    
    def _handle_interrupt_signal(self, signum, frame):
        """
        处理SIGINT信号 (Ctrl+C)
        """
        print(f"Received SIGINT ({signum}), initiating quick shutdown...")
        self._quick_shutdown()
        os._exit(1)  # 强制退出
    
    def _handle_hangup_signal(self, signum, frame):
        """
        处理SIGHUP信号
        """
        print(f"Received SIGHUP ({signum}), handling terminal disconnect...")
        self._graceful_shutdown()
    
    def _graceful_shutdown(self):
        """
        优雅关闭所有进程
        """
        print("Starting graceful shutdown sequence...")
        
        # 首先停止所有Java进程
        for proc in self.java_processes:
            if proc and proc.poll() is None:  # 进程仍在运行
                print(f"Terminating Java process {proc.pid}...")
                proc.terminate()  # 发送SIGTERM
        
        # 等待进程结束
        for proc in self.java_processes:
            if proc:
                try:
                    proc.wait(timeout=10)  # 等待最多10秒
                    print(f"Java process {proc.pid} terminated gracefully.")
                except subprocess.TimeoutExpired:
                    print(f"Java process {proc.pid} did not terminate in time, killing...")
                    proc.kill()  # 强制杀死进程
        
        # 执行清理函数
        for cleanup_func in self.cleanup_functions:
            try:
                cleanup_func()
            except Exception as e:
                print(f"Error in cleanup function: {e}")
    
    def _quick_shutdown(self):
        """
        快速关闭所有进程
        """
        print("Starting quick shutdown sequence...")
        
        # 直接杀死所有Java进程
        for proc in self.java_processes:
            if proc and proc.poll() is None:
                print(f"Killing Java process {proc.pid}...")
                proc.kill()
        
        # 立即退出
        os._exit(1)
    
    def _cleanup_on_exit(self):
        """
        程序正常退出时的清理
        """
        print("Performing cleanup on exit...")
        self._graceful_shutdown()
    
    def add_cleanup_function(self, func):
        """
        添加清理函数
        """
        self.cleanup_functions.append(func)

# 使用示例
def manage_java_process_with_signals():
    signal_handler = ProcessSignalHandler()
    
    # 启动Java进程
    java_proc = subprocess.Popen([
        "java", "-cp", get_classpath(), 
        "org.apache.spark.deploy.raydp.AppMasterEntryPoint"
    ])
    
    # 添加到信号处理器
    signal_handler.add_java_process(java_proc)
    
    # 添加自定义清理函数
    def custom_cleanup():
        print("Performing custom cleanup...")
        # 清理临时文件等
    
    signal_handler.add_cleanup_function(custom_cleanup)
    
    return java_proc
```

## 5. 错误处理与诊断

### 5.1 Java进程错误处理

```python
class JavaProcessErrorHandler:
    @staticmethod
    def handle_java_process_startup_error(proc, stdout, stderr):
        """
        处理Java进程启动错误
        """
        error_info = {
            'return_code': proc.returncode,
            'stdout': stdout.decode('utf-8', errors='replace'),
            'stderr': stderr.decode('utf-8', errors='replace'),
            'timestamp': time.time()
        }
        
        # 分析错误类型
        error_type = JavaProcessErrorHandler._analyze_error_type(error_info)
        
        # 根据错误类型提供相应处理
        if error_type == 'CLASS_NOT_FOUND':
            return JavaProcessErrorHandler._handle_class_not_found(error_info)
        elif error_type == 'NO_SUCH_FILE':
            return JavaProcessErrorHandler._handle_no_such_file(error_info)
        elif error_type == 'OUT_OF_MEMORY':
            return JavaProcessErrorHandler._handle_out_of_memory(error_info)
        elif error_type == 'ACCESS_DENIED':
            return JavaProcessErrorHandler._handle_access_denied(error_info)
        else:
            return JavaProcessErrorHandler._handle_generic_error(error_info)
    
    @staticmethod
    def _analyze_error_type(error_info):
        """
        分析错误类型
        """
        stderr_content = error_info['stderr'].lower()
        
        if 'could not find or load main class' in stderr_content:
            return 'CLASS_NOT_FOUND'
        elif 'no such file or directory' in stderr_content:
            return 'NO_SUCH_FILE'
        elif 'outofmemoryerror' in stderr_content or 'java heap space' in stderr_content:
            return 'OUT_OF_MEMORY'
        elif 'permission denied' in stderr_content or 'access denied' in stderr_content:
            return 'ACCESS_DENIED'
        elif 'exception in thread' in stderr_content:
            return 'EXCEPTION'
        else:
            return 'UNKNOWN'
    
    @staticmethod
    def _handle_class_not_found(error_info):
        """
        处理类找不到错误
        """
        suggestion = (
            "Please check if the Java classpath is correctly set.\n"
            "Make sure all required JAR files are included in the classpath.\n"
            "Verify that the main class 'org.apache.spark.deploy.raydp.AppMasterEntryPoint' exists."
        )
        
        return {
            'type': 'CLASS_NOT_FOUND',
            'message': error_info['stderr'],
            'suggestion': suggestion,
            'details': error_info
        }
    
    @staticmethod
    def _handle_no_such_file(error_info):
        """
        处理文件不存在错误
        """
        suggestion = (
            "Please check if all required files and directories exist.\n"
            "Verify the paths used in the Java command."
        )
        
        return {
            'type': 'NO_SUCH_FILE',
            'message': error_info['stderr'],
            'suggestion': suggestion,
            'details': error_info
        }
    
    @staticmethod
    def _handle_out_of_memory(error_info):
        """
        处理内存不足错误
        """
        suggestion = (
            "Increase the Java heap size using -Xmx parameter.\n"
            "Current configuration may be insufficient for the workload.\n"
            "Consider optimizing memory usage in the application."
        )
        
        return {
            'type': 'OUT_OF_MEMORY',
            'message': error_info['stderr'],
            'suggestion': suggestion,
            'details': error_info
        }
    
    @staticmethod
    def _handle_access_denied(error_info):
        """
        处理权限不足错误
        """
        suggestion = (
            "Check file permissions for the Java executable and related files.\n"
            "Ensure the user has sufficient privileges to execute Java processes."
        )
        
        return {
            'type': 'ACCESS_DENIED',
            'message': error_info['stderr'],
            'suggestion': suggestion,
            'details': error_info
        }
    
    @staticmethod
    def _handle_generic_error(error_info):
        """
        处理通用错误
        """
        return {
            'type': 'GENERIC',
            'message': error_info['stderr'],
            'suggestion': 'Please check the Java process logs for more details.',
            'details': error_info
        }
```

### 5.2 进程监控与诊断

```python
import psutil
import threading
import time

class JavaProcessMonitor:
    def __init__(self, proc):
        self.proc = proc
        self.monitoring = False
        self.metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_info': [],
            'status': []
        }
        self.monitoring_thread = None
    
    def start_monitoring(self, interval=1.0):
        """
        开始监控Java进程
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        print(f"Started monitoring Java process {self.proc.pid}")
    
    def stop_monitoring(self):
        """
        停止监控
        """
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        print(f"Stopped monitoring Java process {self.proc.pid}")
    
    def _monitor_loop(self, interval):
        """
        监控循环
        """
        try:
            # 获取进程对象
            psutil_proc = psutil.Process(self.proc.pid)
            
            while self.monitoring:
                try:
                    # 获取CPU使用率
                    cpu_percent = psutil_proc.cpu_percent()
                    
                    # 获取内存信息
                    memory_info = psutil_proc.memory_info()
                    memory_percent = psutil_proc.memory_percent()
                    
                    # 获取进程状态
                    status = psutil_proc.status()
                    
                    # 存储指标
                    self.metrics['cpu_percent'].append(cpu_percent)
                    self.metrics['memory_percent'].append(memory_percent)
                    self.metrics['memory_info'].append(memory_info)
                    self.metrics['status'].append(status)
                    
                    # 保持历史记录长度
                    max_len = 1000  # 最多保留1000个数据点
                    for key in self.metrics:
                        if len(self.metrics[key]) > max_len:
                            self.metrics[key] = self.metrics[key][-max_len:]
                    
                except psutil.NoSuchProcess:
                    # 进程已退出
                    print(f"Java process {self.proc.pid} has exited")
                    break
                except Exception as e:
                    print(f"Error monitoring process {self.proc.pid}: {e}")
                
                time.sleep(interval)
                
        except Exception as e:
            print(f"Monitor loop error: {e}")
    
    def get_current_metrics(self):
        """
        获取当前指标
        """
        if not self.metrics['cpu_percent']:
            return None
        
        return {
            'cpu_percent': self.metrics['cpu_percent'][-1],
            'memory_percent': self.metrics['memory_percent'][-1],
            'memory_info': self.metrics['memory_info'][-1]._asdict() if self.metrics['memory_info'] else {},
            'status': self.metrics['status'][-1],
            'pid': self.proc.pid
        }
    
    def get_historical_metrics(self):
        """
        获取历史指标
        """
        return self.metrics.copy()
    
    def is_process_alive(self):
        """
        检查进程是否存活
        """
        return self.proc.poll() is None

# 使用示例
def monitor_java_process():
    # 启动Java进程
    java_proc = subprocess.Popen([
        "java", "-cp", get_classpath(),
        "org.apache.spark.deploy.raydp.AppMasterEntryPoint"
    ])
    
    # 创建监控器
    monitor = JavaProcessMonitor(java_proc)
    
    # 开始监控
    monitor.start_monitoring(interval=0.5)
    
    try:
        # 监控一段时间
        for i in range(10):
            metrics = monitor.get_current_metrics()
            if metrics:
                print(f"Process {metrics['pid']}: "
                      f"CPU={metrics['cpu_percent']:.1f}%, "
                      f"Memory={metrics['memory_percent']:.1f}%")
            time.sleep(1)
    finally:
        # 停止监控
        monitor.stop_monitoring()
```

## 6. 性能优化考虑

### 6.1 启动时间优化

```python
import concurrent.futures
import threading

class OptimizedJavaLauncher:
    def __init__(self):
        self.process_cache = {}
        self.cache_lock = threading.Lock()
    
    def launch_with_cache(self, classpath, main_class, args=None, use_cache=True):
        """
        使用缓存优化的Java进程启动
        """
        cache_key = self._generate_cache_key(classpath, main_class, args)
        
        if use_cache:
            with self.cache_lock:
                if cache_key in self.process_cache:
                    cached_proc = self.process_cache[cache_key]
                    if cached_proc.poll() is None:  # 进程仍在运行
                        print(f"Using cached process for key: {cache_key}")
                        return cached_proc
                    else:
                        # 进程已退出，从缓存中移除
                        del self.process_cache[cache_key]
        
        # 启动新进程
        proc = self._launch_java_process(classpath, main_class, args)
        
        if use_cache:
            with self.cache_lock:
                self.process_cache[cache_key] = proc
        
        return proc
    
    def _generate_cache_key(self, classpath, main_class, args):
        """
        生成缓存键
        """
        import hashlib
        key_str = f"{classpath}:{main_class}:{args}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _launch_java_process(self, classpath, main_class, args):
        """
        启动Java进程
        """
        cmd = ["java", "-cp", classpath]
        
        # 添加优化参数
        cmd.extend([
            "-XX:TieredStopAtLevel=1",  # 禁用C2编译器以加快启动
            "-noverify",               # 跳过字节码验证
            "-XX:+UseSerialGC",        # 使用串行GC加快启动
        ])
        
        cmd.append(main_class)
        
        if args:
            cmd.extend(args)
        
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    def prewarm_jvm(self, classpath, main_class, num_instances=1):
        """
        预热JVM实例
        """
        futures = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_instances) as executor:
            for i in range(num_instances):
                future = executor.submit(
                    self._launch_java_process, classpath, main_class, None
                )
                futures.append(future)
        
        # 等待所有实例启动
        processes = []
        for future in concurrent.futures.as_completed(futures):
            proc = future.result()
            processes.append(proc)
        
        return processes
```

### 6.2 资源管理优化

```python
class ResourceManager:
    def __init__(self):
        self.java_processes = []
        self.comm_files = []
        self.resources_lock = threading.Lock()
    
    def register_java_process(self, proc):
        """
        注册Java进程用于资源管理
        """
        with self.resources_lock:
            self.java_processes.append(proc)
    
    def register_communication_file(self, file_path):
        """
        注册通信文件用于资源管理
        """
        with self.resources_lock:
            self.comm_files.append(file_path)
    
    def cleanup_resources(self):
        """
        清理所有资源
        """
        with self.resources_lock:
            # 终止所有Java进程
            for proc in self.java_processes:
                try:
                    if proc and proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                except Exception as e:
                    print(f"Error terminating process: {e}")
            
            # 删除所有通信文件
            for file_path in self.comm_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
            
            # 清空列表
            self.java_processes.clear()
            self.comm_files.clear()
    
    def get_resource_usage(self):
        """
        获取资源使用情况
        """
        with self.resources_lock:
            processes_running = sum(
                1 for proc in self.java_processes 
                if proc and proc.poll() is None
            )
            
            files_exist = sum(
                1 for file_path in self.comm_files 
                if os.path.exists(file_path)
            )
            
            return {
                'java_processes_running': processes_running,
                'communication_files': len(self.comm_files),
                'existing_files': files_exist
            }
```

## 总结

Java进程启动是RayDP中一个关键的技术环节，涉及多个重要方面：

1. **进程管理**：通过subprocess启动和管理Java进程
2. **JVM配置**：优化的JVM参数设置以提升性能
3. **通信机制**：文件、Socket等多种通信方式
4. **错误处理**：完善的错误检测和恢复机制
5. **资源管理**：进程和文件资源的生命周期管理
6. **性能优化**：启动时间、内存使用等方面的优化

这种设计确保了Java进程能够可靠启动并与Python进程有效通信，为Spark on Ray的运行奠定了基础。