# Step 6: 高级功能

## 概述

本步骤介绍 MemBrain 的高级功能，包括多数据集管理、批量处理、自定义搜索策略、异步处理等。

## 6.1 多数据集管理

### 为不同用户创建独立数据集

```python
# 为不同用户创建不同的数据集
client = MemBrainClient()

# 用户列表
users = ["user_001", "user_002", "user_003"]

# 为每个用户存储数据
for user_id in users:
    messages = [
        {"speaker": "user", "content": f"This is {user_id}'s personal data."},
    ]
    
    result = client.store_and_digest(
        dataset="personamem_v2",
        task=user_id,
        messages=messages
    )
    print(f"Stored for {user_id}: {result['status']}")
```

### 使用场景

| 场景 | dataset | task | 说明 |
|------|---------|------|------|
| 多用户应用 | "app_data" | user_id | 每个用户独立 task |
| 多设备应用 | "app_data" | device_id | 每个设备独立 task |
| 多角色应用 | "app_data" | role_name | 每个角色独立 task |

### 数据隔离

```python
# 搜索时指定用户，确保数据隔离
result_user1 = client.search(
    dataset="personamem_v2",
    task="user_001",  # 只搜索 user_001 的数据
    question="What did I do?"
)

result_user2 = client.search(
    dataset="personamem_v2",
    task="user_002",  # 只搜索 user_002 的数据
    question="What did I do?"
)
```

## 6.2 批量写入

### 基本批量写入

```python
# 批量处理多个会话
def batch_ingest(client, sessions, dataset="my_app", task="user_001"):
    """批量写入多个会话
    
    Args:
        client: MemBrainClient 实例
        sessions: 会话列表，每个会话是消息列表
        dataset: 数据集名称
        task: 任务标识
    """
    results = []
    
    for i, messages in enumerate(sessions):
        print(f"处理会话 {i+1}/{len(sessions)}")
        
        result = client.store_and_digest(
            dataset=dataset,
            task=task,
            messages=messages
        )
        
        results.append(result)
        
        # 避免过快请求
        time.sleep(1)
    
    return results


# 使用
sessions = [
    [  # 会话 1
        {"speaker": "user", "content": "Hello"},
        {"speaker": "assistant", "content": "Hi!"}
    ],
    [  # 会话 2
        {"speaker": "user", "content": "How are you?"},
        {"speaker": "assistant", "content": "I'm good!"}
    ],
    [  # 会话 3
        {"speaker": "user", "content": "What's the weather?"},
        {"speaker": "assistant", "content": "It's sunny."}
    ],
]

results = batch_ingest(client, sessions)
```

### 并发批量写入

```python
import asyncio
import aiohttp

async def async_batch_ingest(sessions, dataset="my_app", task="user_001"):
    """异步批量写入"""
    
    async def send_session(session_data, session_id):
        async with aiohttp.ClientSession() as session:
            url = "http://localhost:9574/api/memory"
            payload = {
                "dataset": dataset,
                "task": task,
                "store": True,
                "digest": True,
                "messages": session_data
            }
            async with session.post(url, json=payload) as response:
                return await response.json()
    
    tasks = [
        send_session(messages, i) 
        for i, messages in enumerate(sessions)
    ]
    
    results = await asyncio.gather(*tasks)
    return results


# 使用
sessions = [
    [{"speaker": "user", "content": "Message 1"}],
    [{"speaker": "user", "content": "Message 2"}],
    [{"speaker": "user", "content": "Message 3"}],
]

results = asyncio.run(async_batch_ingest(sessions))
```

## 6.3 自定义搜索策略

### 快速搜索 (低延迟)

```python
# 适用于实时对话场景，需要快速响应

result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do today?",
    mode="direct",      # 跳过 LLM 扩展，3 条检索路径
    strategy="rrf"     # 快速融合
)

# 特点:
# - 延迟最低
# - 跳过 LLM 扩展步骤
# - 只使用 3 条检索路径 (A+B+C)
# - 适用于实时对话、语音助手等场景
```

### 高精度搜索

```python
# 适用于需要高质量答案的场景

result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do today?",
    mode="reflect",     # 使用反思模式
    strategy="rerank"  # 使用交叉编码器重排
)

# 特点:
# - 延迟较高
# - 使用 LLM 反思增强
# - 使用交叉编码器重排
# - 适用于分析报告、总结等场景
```

### 平衡搜索

```python
# 适用于大多数场景

result = client.search(
    dataset="my_app",
    task="user_001",
    question="What did I do today?",
    mode="expand",      # 默认扩展模式 (默认)
    strategy="rrf"     # 默认融合策略 (默认)
)

# 特点:
# - 延迟适中
# - 使用 LLM 扩展
# - 6 条检索路径
# - 适用于大多数场景
```

### 搜索策略对比

| 场景 | mode | strategy | 延迟 | 质量 |
|------|------|----------|------|------|
| 实时对话 | direct | rrf | 低 | 中 |
| 大多数场景 | expand | rrf | 中 | 高 |
| 分析报告 | reflect | rerank | 高 | 最高 |

## 6.4 异步处理

### 异步写入

```python
import asyncio

async def async_ingest(client, messages):
    """异步写入"""
    
    # 使用 aiohttp 进行异步请求
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        url = f"{client.base_url}/memory"
        payload = {
            "dataset": "my_app",
            "task": "user_001",
            "store": True,
            "digest": True,
            "messages": messages
        }
        
        async with session.post(url, json=payload) as response:
            result = await response.json()
            return result


async def main():
    # 异步执行多个写入
    messages_list = [
        [{"speaker": "user", "content": "Message 1"}],
        [{"speaker": "user", "content": "Message 2"}],
        [{"speaker": "user", "content": "Message 3"}],
    ]
    
    tasks = [async_ingest(client, msgs) for msgs in messages_list]
    results = await asyncio.gather(*tasks)
    
    for r in results:
        print(r)


asyncio.run(main())
```

### 异步搜索

```python
import asyncio
import aiohttp

async def async_search(client, question):
    """异步搜索"""
    
    async with aiohttp.ClientSession() as session:
        url = f"{client.base_url}/memory/search"
        payload = {
            "dataset": "my_app",
            "task": "user_001",
            "question": question,
            "mode": "expand",
            "strategy": "rrf"
        }
        
        async with session.post(url, json=payload) as response:
            result = await response.json()
            return result


async def main():
    # 同时搜索多个问题
    questions = [
        "What did I do yesterday?",
        "What is my job?",
        "What are my hobbies?"
    ]
    
    tasks = [async_search(client, q) for q in questions]
    results = await asyncio.gather(*tasks)
    
    for q, r in zip(questions, results):
        print(f"Question: {q}")
        print(f"Context: {r['packed_context'][:100]}...")
        print()


asyncio.run(main())
```

## 6.5 自定义 Agent Profile

### 使用不同的 Agent Profile

```python
# Agent Profile 可以影响记忆提取的行为
# 不同的 profile 适用于不同的场景

# Profile 1: 详细模式
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "agent_profile": "detailed",  # 详细提取
    "store": True,
    "digest": True,
    "messages": [...]
}

# Profile 2: 简洁模式
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "agent_profile": "concise",  # 简洁提取
    "store": True,
    "digest": True,
    "messages": [...]
}

# Profile 3: 隐私模式
payload = {
    "dataset": "my_app",
    "task": "user_001",
    "agent_profile": "privacy",  # 隐私保护
    "store": True,
    "digest": True,
    "messages": [...]
}
```

### 创建自定义 Profile

```python
# 在配置文件中定义自定义 profile
# 配置文件: config/agent_profiles.yaml

profiles:
  my_custom_profile:
    extraction:
      entity_types: ["person", "place", "event", "activity"]
      max_entities: 50
      min_confidence: 0.8
    
    fact_generation:
      max_facts: 100
      include_context: true
    
    entity_resolution:
      use_llm: true
      threshold: 0.9
```

## 6.6 记忆更新与删除

### 更新记忆

```python
# MemBrain 不支持直接更新
# 需要删除后重新写入

# 方法 1: 删除会话
def delete_session(session_id):
    """删除特定会话"""
    # 直接删除数据库记录 (需要管理员权限)
    pass

# 方法 2: 归档事实
def archive_fact(fact_id):
    """归档特定事实"""
    # 将 fact 状态改为 archived
    # 需要直接操作数据库
    pass

# 方法 3: 重新消化会话
def redigest_session(session_id):
    """重新消化会话"""
    # 设置会话为未消化状态，然后重新消化
    pass
```

## 6.7 监控与调试

### 监控消化状态

```python
def check_digest_status(dataset, task):
    """检查消化状态"""
    
    # 查询数据库获取状态
    # 需要直接访问数据库
    
    pass


def get_statistics(dataset, task):
    """获取统计信息"""
    
    # 获取实体数量、事实数量、会话数量等
    # 需要直接访问数据库
    
    pass
```

### 调试搜索结果

```python
def debug_search(question, dataset="my_app", task="user_001"):
    """调试搜索结果"""
    
    result = client.search(
        dataset=dataset,
        task=task,
        question=question,
        mode="expand",
        strategy="rerank"  # 使用 rerank 获取更详细的信息
    )
    
    print("=== 搜索结果调试 ===")
    print(f"问题: {question}")
    print(f"Token 数量: {result['packed_token_count']}")
    print(f"事实数量: {len(result['facts'])}")
    print(f"会话数量: {len(result['sessions'])}")
    
    print("\n--- 事实详情 ---")
    for fact in result["facts"]:
        print(f"  ID: {fact['fact_id']}")
        print(f"  文本: {fact['text']}")
        print(f"  来源: {fact['source']}")
        print(f"  分数: {fact['rerank_score']}")
        print(f"  时间: {fact.get('time_info')}")
        print()
    
    print("\n--- 会话详情 ---")
    for session in result["sessions"]:
        print(f"  ID: {session['session_id']}")
        print(f"  主题: {session['subject']}")
        print(f"  分数: {session['score']}")
        print()
```

## 6.8 集成示例

### 与 LangChain 集成

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

def langchain_example():
    """与 LangChain 集成的示例"""
    
    # 1. 搜索记忆
    search_result = client.search(
        dataset="my_app",
        task="user_001",
        question="What is my background?",
        mode="expand",
        strategy="rrf"
    )
    
    context = search_result["packed_context"]
    
    # 2. 创建 LangChain 提示
    prompt = PromptTemplate(
        template="Based on the following context, answer the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["context", "question"]
    )
    
    # 3. 创建链
    llm = ChatOpenAI(model_name="gpt-4")
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 4. 执行
    result = chain.run(context=context, question="What is my background?")
    
    return result
```

### 与 LlamaIndex 集成

```python
from llama_index import GPTSimpleVectorIndex, Document

def llamaindex_example():
    """与 LlamaIndex 集成的示例"""
    
    # 1. 搜索记忆
    search_result = client.search(
        dataset="my_app",
        task="user_001",
        question="What did I do?",
        mode="expand",
        strategy="rrf"
    )
    
    # 2. 转换为 Document
    facts_text = "\n".join([f["text"] for f in search_result["facts"]])
    documents = [Document(facts_text)]
    
    # 3. 创建索引
    index = GPTSimpleVectorIndex(documents)
    
    # 4. 查询
    response = index.query("What did I do?")
    
    return response
```

## 6.9 性能优化建议

### 1. 批量写入

```python
# 批量写入比单条写入更高效
batch_size = 10  # 每批 10 条会话
```

### 2. 选择合适的搜索模式

```python
# 实时对话使用 direct 模式
# 报告生成使用 expand 或 reflect 模式
```

### 3. 调整 top_k

```python
# 根据实际需要调整
# top_k 越大，token 越多，延迟越高
```

### 4. 使用连接池

```python
# 使用 requests.Session() 复用连接
session = requests.Session()
session.post(url, json=payload)
```

## 总结

本步骤介绍了：
- 多数据集管理方法
- 批量写入和并发处理
- 自定义搜索策略的选择
- 异步处理方法
- Agent Profile 的使用
- 记忆更新与删除
- 监控与调试方法
- 与 LangChain、LlamaIndex 的集成
- 性能优化建议

这些高级功能可以帮助构建更复杂、更高效的 MemBrain 应用。