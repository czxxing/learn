# Step 8: 错误处理与故障排除

## 概述

本步骤详细介绍 MemBrain 的常见错误处理方法、故障排除技巧，以及最佳实践，帮助用户快速定位和解决问题。

## 8.1 常见错误

### API 错误

#### 错误 1: 参数错误 (400)

```json
{
    "detail": "messages required when store=True"
}
```

**原因**: 当 `store=True` 时，必须提供 `messages` 参数

**解决方案**:

```python
# 错误示例
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": []  # 空消息列表
}

# 正确示例
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": True,
    "messages": [
        {"speaker": "user", "content": "Hello"}
    ]
}
```

#### 错误 2: 缺少必要参数 (400)

```json
{
    "detail": "at least one of store or digest must be True"
}
```

**原因**: `store` 和 `digest` 不能同时为 `False`

**解决方案**:

```python
# 错误示例
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": False,
    "digest": False,
    "messages": [...]
}

# 正确示例 - 至少一个为 True
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "store": True,
    "digest": False,  # 或 True
    "messages": [...]
}
```

#### 错误 3: 数据集/任务不存在 (404)

```json
{
    "detail": "Task 'user_999' not found in dataset 'personamem_v2'"
}
```

**原因**: 搜索时指定的 task 不存在

**解决方案**:

```python
# 确保先写入数据，再搜索
# 或者使用 store=True 创建新数据

payload = {
    "dataset": "personamem_v2",
    "task": "user_001",
    "store": True,
    "digest": False,
    "messages": [{"speaker": "user", "content": "init"}]
}

# 先创建数据
result = requests.post(url, json=payload)

# 然后再搜索
search_result = client.search(
    dataset="personamem_v2",
    task="user_001",
    question="What did I do?"
)
```

### 网络错误

#### 错误 4: 连接失败

```python
# 错误
requests.exceptions.ConnectionError: [Errno 111] Connection refused

# 原因: MemBrain 服务器未启动
# 解决方案:
# 1. 启动服务器
# python -m uvicorn membrain.api.server:app --host 0.0.0.0 --port 9574

# 2. 检查端口是否被占用
# netstat -tulpn | grep 9574
```

#### 错误 5: 请求超时

```python
# 错误
requests.exceptions.Timeout: Request timed out

# 原因: 请求处理时间过长
# 解决方案:
# 1. 增加超时时间
response = requests.post(url, json=payload, timeout=60)

# 2. 使用更快的搜索模式
payload = {
    ...
    "mode": "direct",  # 跳过 LLM 扩展
    "strategy": "rrf"  # 快速融合
}
```

### LLM 错误

#### 错误 6: API 密钥无效

```python
# 错误
openai.error.AuthenticationError: Invalid API Key

# 解决方案:
# 1. 检查环境变量
import os
print(os.getenv("OPENAI_API_KEY"))

# 2. 设置正确的 API 密钥
os.environ["OPENAI_API_KEY"] = "sk-..."
```

#### 错误 7: API 配额耗尽

```python
# 错误
openai.error.RateLimitError: You exceeded your current quota

# 解决方案:
# 1. 检查 API 使用量
# 2. 升级订阅计划
# 3. 等待配额重置
```

## 8.2 错误处理示例

### 基础错误处理

```python
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout

def safe_request(url, payload, max_retries=3):
    """带重试的请求"""
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            
            # 检查 HTTP 错误
            if response.status_code == 400:
                error = response.json()
                print(f"参数错误: {error.get('detail')}")
                return None
                
            elif response.status_code == 404:
                error = response.json()
                print(f"资源不存在: {error.get('detail')}")
                return None
                
            elif response.status_code >= 500:
                print(f"服务器错误: {response.status_code}")
                continue
                
            response.raise_for_status()
            return response.json()
            
        except ConnectionError:
            print(f"连接失败 (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
                
        except Timeout:
            print(f"请求超时 (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
                
        except HTTPError as e:
            print(f"HTTP 错误: {e}")
            break
            
    return None
```

### 高级错误处理类

```python
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout
from typing import Optional, Dict, Any
import time

class MemBrainError(Exception):
    """MemBrain 基础异常"""
    pass

class MemBrainAPIError(MemBrainError):
    """API 错误"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")

class MemBrainConnectionError(MemBrainError):
    """连接错误"""
    pass

class MemBrainTimeoutError(MemBrainError):
    """超时错误"""
    pass


class MemBrainClient:
    """带错误处理的 MemBrain 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:9574/api"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        payload: Optional[Dict] = None,
        max_retries: int = 3,
        timeout: int = 30
    ) -> Optional[Dict]:
        """发送请求，带错误处理"""
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                if method == "POST":
                    response = self.session.post(
                        url, 
                        json=payload, 
                        timeout=timeout
                    )
                else:
                    raise ValueError(f"不支持的方法: {method}")
                
                # 检查状态码
                if response.status_code == 400:
                    error = response.json()
                    raise MemBrainAPIError(400, error.get('detail', '参数错误'))
                    
                elif response.status_code == 404:
                    error = response.json()
                    raise MemBrainAPIError(404, error.get('detail', '资源不存在'))
                    
                elif response.status_code >= 500:
                    print(f"服务器错误 {response.status_code}，重试 {attempt + 1}/{max_retries}")
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.ConnectionError as e:
                if attempt == max_retries - 1:
                    raise MemBrainConnectionError(f"连接失败: {e}")
                time.sleep(2)
                
            except requests.exceptions.Timeout as e:
                if attempt == max_retries - 1:
                    raise MemBrainTimeoutError(f"请求超时: {e}")
                time.sleep(2)
                
            except HTTPError as e:
                raise MemBrainAPIError(e.response.status_code, str(e))
                
        return None
    
    def store_and_digest(self, dataset: str, task: str, messages: list) -> Optional[Dict]:
        """存储并消化"""
        return self._request(
            "POST", 
            "memory",
            {
                "dataset": dataset,
                "task": task,
                "store": True,
                "digest": True,
                "messages": messages
            }
        )
    
    def search(
        self, 
        dataset: str, 
        task: str, 
        question: str,
        **kwargs
    ) -> Optional[Dict]:
        """搜索"""
        return self._request(
            "POST",
            "memory/search",
            {
                "dataset": dataset,
                "task": task,
                "question": question,
                **kwargs
            }
        )


# 使用示例
try:
    client = MemBrainClient()
    
    # 存储
    result = client.store_and_digest(
        dataset="my_app",
        task="user_001",
        messages=[{"speaker": "user", "content": "Hello"}]
    )
    print(f"存储成功: {result}")
    
    # 搜索
    import time
    time.sleep(5)  # 等待消化
    
    result = client.search(
        dataset="my_app",
        task="user_001",
        question="What did I say?"
    )
    print(f"搜索成功: {result}")
    
except MemBrainAPIError as e:
    print(f"API 错误: {e.status_code} - {e.message}")
    
except MemBrainConnectionError as e:
    print(f"连接错误: {e}")
    
except MemBrainTimeoutError as e:
    print(f"超时错误: {e}")
    
except MemBrainError as e:
    print(f"MemBrain 错误: {e}")
```

## 8.3 故障排除

### 问题 1: 搜索结果为空

**症状**: 搜索返回空结果

**可能原因**:
1. 数据未被消化
2. 数据集/任务不匹配
3. 搜索问题与存储内容不相关

**排查步骤**:

```python
# 步骤 1: 检查数据是否存储
result = client.store_and_digest(
    dataset="my_app",
    task="user_001",
    messages=[{"speaker": "user", "content": "I love playing guitar"}]
)
print(f"存储状态: {result['status']}")

# 步骤 2: 等待消化完成
import time
time.sleep(5)

# 步骤 3: 使用更广泛的搜索词
result = client.search(
    dataset="my_app",
    task="user_001",
    question="music",  # 尝试更广泛的关键词
    mode="expand",
    top_k=20
)
print(f"结果数量: {len(result['facts'])}")
```

### 问题 2: 消化失败

**症状**: 写入成功但搜索无结果

**可能原因**:
1. LLM API 问题
2. 提取超时
3. 实体/事实生成失败

**排查步骤**:

```python
import logging

# 开启详细日志
logging.basicConfig(level=logging.DEBUG)

# 检查服务器日志
# 查找 "extract" 或 "digest" 相关的日志

# 手动触发消化
result = client.store_and_digest(
    dataset="my_app",
    task="user_001",
    messages=[{"speaker": "user", "content": "Test"}]
)

# 检查是否有 digest_queued
print(f"状态: {result.get('status')}")
```

### 问题 3: 搜索结果不相关

**症状**: 返回的事实与问题不相关

**可能原因**:
1. 搜索模式选择不当
2. top_k 太小
3. 融合策略不适当

**解决方案**:

```python
# 使用更精确的搜索模式
result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do yesterday?",
    mode="reflect",     # 使用反思模式
    strategy="rerank",  # 使用重排序
    top_k=20            # 增加返回数量
)
```

### 问题 4: 响应速度慢

**症状**: 搜索响应时间过长

**可能原因**:
1. LLM 调用延迟
2. 数据库查询慢
3. 网络延迟

**解决方案**:

```python
# 使用更快的搜索模式
result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do?",
    mode="direct",   # 跳过 LLM 扩展
    strategy="rrf"   # 快速融合
)
```

### 问题 5: Token 预算超限

**症状**: 返回结果被截断

**可能原因**:
1. top_k 太大
2. 事实文本太长

**解决方案**:

```python
# 减少 top_k
result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do?",
    top_k=5  # 减少返回数量
)

# 使用 direct 模式
result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do?",
    mode="direct"
)
```

## 8.4 日志分析

### 开启调试日志

```python
import logging

# 开启所有模块的调试日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 开启特定模块的日志
logging.getLogger("membrain.search").setLevel(logging.DEBUG)
logging.getLogger("membrain.ingest").setLevel(logging.DEBUG)
logging.getLogger("membrain.entity").setLevel(logging.DEBUG)
```

### 常见日志关键词

| 关键词 | 说明 |
|--------|------|
| `entity-extractor` | 实体提取日志 |
| `fact-generator` | 事实生成日志 |
| `entity-resolver` | 实体消重日志 |
| `search-query` | 搜索查询日志 |
| `retrieval` | 检索日志 |

### 分析日志示例

```bash
# 搜索实体提取相关日志
grep "entity-extractor" membrain.log

# 搜索错误
grep "ERROR" membrain.log

# 搜索特定时间段的日志
grep "2024-01-15 14:3" membrain.log
```

## 8.5 健康检查

### API 健康检查

```python
import requests

def health_check(base_url="http://localhost:9574"):
    """检查 MemBrain 服务健康状态"""
    
    try:
        # 检查根路径
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"根路径状态: {response.status_code}")
        
        # 检查 docs
        response = requests.get(f"{base_url}/docs", timeout=5)
        print(f"文档路径状态: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False


# 执行健康检查
if health_check():
    print("MemBrain 服务正常")
else:
    print("MemBrain 服务异常")
```

### 数据库健康检查

```python
# 需要直接访问数据库
# 检查表是否存在

def check_database():
    """检查数据库健康状态"""
    
    # 检查表是否存在
    tables = ["datasets", "tasks", "chat_sessions", "chat_messages", 
              "entities", "facts", "fact_refs", "entity_trees"]
    
    for table in tables:
        # 查询表是否存在
        # SELECT * FROM information_schema.tables WHERE table_name = ?
        pass
```

## 8.6 性能监控

### 监控指标

```python
import time
from functools import wraps

def monitor_performance(func):
    """性能监控装饰器"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"{func.__name__} 执行时间: {duration:.2f}秒")
        
        return result
    
    return wrapper


# 使用示例
@monitor_performance
def slow_search():
    result = client.search(
        dataset="my_app",
        task="user_001",
        question="What did I do?",
        mode="expand"
    )
    return result


result = slow_search()
```

## 8.7 常见问题 FAQ

### Q1: 为什么搜索不到刚写入的数据?

**A**: 写入是异步消化的，需要等待几秒后再搜索。建议:
```python
result = client.store_and_digest(...)
time.sleep(5)  # 等待消化
result = client.search(...)
```

### Q2: 如何提高搜索精度?

**A**: 
1. 使用 `mode="reflect"` 模式
2. 使用 `strategy="rerank"` 策略
3. 调整 `top_k` 参数

### Q3: 如何加快搜索速度?

**A**:
1. 使用 `mode="direct"` 模式
2. 使用 `strategy="rrf"` 策略
3. 减少 `top_k` 参数

### Q4: 实体消重不起作用?

**A**: 检查配置:
```python
RESOLVER_LLM_ENABLED = True  # 确保启用 LLM
RESOLVER_JACCARD_THRESHOLD = 0.9  # 确保阈值合理
```

### Q5: 内存占用过高?

**A**:
1. 减少 `top_k` 参数
2. 减少 token 预算
3. 使用更小的嵌入模型

## 总结

本步骤介绍了：
- API 常见错误及解决方案
- 网络错误处理
- LLM 错误处理
- 完整的错误处理示例
- 常见问题的故障排除步骤
- 日志分析方法
- 健康检查方法
- 性能监控技巧
- 常见问题 FAQ

通过这些错误处理和故障排除方法，可以快速定位和解决 MemBrain 使用中的问题。