# mem0 add() 操作完整流程图

## 总体架构流程

```mermaid
flowchart TD
    A[mem0.add 入口] --> B{输入预处理}
    
    B --> B1[验证 memory_type]
    B --> B2[转换消息格式 str/dict → list[dict]]
    B --> B3[构建 metadata 和 filters]
    B --> B4[解析视觉消息 enable_vision]
    
    B --> C{memory_type ==<br/>'procedural_memory'?}
    
    C -->|YES| D[_create_procedural_memory]
    C -->|NO| E[并行执行]
    
    D --> D1[LLM 生成程序性记忆]
    D1 --> D2[插入向量存储]
    D2 --> D3[返回结果]
    
    E --> E1[_add_to_vector_store]
    E --> E2[_add_to_graph]
    
    E1 --> F[返回结果]
    E2 --> F
    
    F --> G[{results, relations}]
    
    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

## 详细流程：_add_to_vector_store (infer=True)

```mermaid
flowchart TD
    A[_add_to_vector_store<br/>infer=True] --> B[消息解析]
    
    B --> B1[parse_messages]
    B --> B2[判断 is_agent_memory]
    
    B2 --> B3{agent_id 存在<br/>+ assistant 消息?}
    B3 -->|YES| B4[使用 AGENT_MEMORY<br/>EXTRACTION_PROMPT]
    B3 -->|NO| B5[使用 USER_MEMORY<br/>EXTRACTION_PROMPT]
    
    B4 --> C[LLM 事实提取 - 步骤1]
    B5 --> C
    
    C --> C1[调用 LLM.generate_response]
    C1 --> C2[解析 JSON response]
    C2 --> C3[提取 facts 数组]
    C3 --> D[搜索现有记忆]
    
    D --> D1{对每个 fact}
    D1 -->|循环| D2[生成 embedding]
    D2 --> D3[向量搜索 limit=5]
    D3 --> D4[收集相关记忆]
    D1 -->|结束| E[LLM 决策操作 - 步骤2]
    
    E --> E1[get_update_memory_messages]
    E1 --> E2[调用 LLM.generate_response]
    E2 --> E3[解析 ADD/UPDATE/DELETE/NONE 操作]
    E3 --> F{执行内存操作}
    
    F -->|ADD| F1[_create_memory]
    F1 --> F1a[检查 embedding 缓存]
    F1 --> F1b[生成/获取 embedding]
    F1 --> F1c[创建 memory_id]
    F1 --> F1d[构建 metadata]
    F1 --> F1e[插入向量存储]
    F1 --> F1f[记录历史到 SQLite]
    F1 --> F1g[返回结果]
    
    F -->|UPDATE| F2[_update_memory]
    F2 --> F2a[获取现有记忆]
    F2 --> F2b[更新内容 + hash]
    F2 --> F2c[更新向量存储]
    F2 --> F2d[记录历史]
    F2 --> F2e[返回结果]
    
    F -->|DELETE| F3[_delete_memory]
    F3 --> F3a[软删除记忆]
    F3 --> F3b[记录历史]
    F3 --> F3c[返回结果]
    
    F -->|NONE| F4[更新 session IDs]
    F4 --> F4a[获取记忆]
    F4 --> F4b[更新 agent_id/run_id]
    F4 --> F4c[返回结果]
    
    F1g --> G[返回结果列表]
    F2e --> G
    F3c --> G
    F4c --> G
    
    style A fill:#ff9,stroke:#333
    style G fill:#9f9,stroke:#333
```

## 详细流程：_add_to_graph

```mermaid
flowchart TD
    A[_add_to_graph] --> B{enable_graph?}
    
    B -->|NO| C[返回空列表]
    B -->|YES| D[设置默认 user_id]
    
    D --> E[提取消息内容]
    E --> E1[拼接非 system 消息内容]
    E1 --> F[graph.add]
    
    F --> F1[提取实体]
    F1 --> F2[提取关系]
    F2 --> F3[添加到图数据库]
    F3 --> G[返回添加的实体]
    
    style A fill:#9ff,stroke:#333
    style G fill:#9f9,stroke:#333
```

## 核心操作：_create_memory

```mermaid
flowchart TD
    A[_create_memory] --> B[获取/生成 embedding]
    
    B --> B1{existing_embeddings<br/>是 dict 且包含 data?}
    B1 -->|YES| B2[直接使用缓存]
    B1 -->|NO| B3[调用 embedding_model]
    
    B2 --> C[生成 memory_id]
    B3 --> C
    
    C --> D[UUID]
    
    D --> E[构建 metadata]
    E --> E1[data: 记忆内容]
    E --> E2[hash: MD5]
    E --> E3[created_at: 时间戳]
    E --> E4[updated_at: 时间戳]
    E --> E5[...其他字段]
    
    E5 --> F[插入向量存储]
    F --> F1[vector_store.insert]
    F1 --> F2[vectors, ids, payloads]
    
    F2 --> G[记录历史到 SQLite]
    G --> G1[db.add_history]
    G1 --> G2[memory_id, action, actor_id, role]
    
    G2 --> H[返回 memory_id]
    
    style A fill:#f9f,stroke:#333
    style H fill:#9f9,stroke:#333
```

## 简化流程：infer=False

```mermaid
flowchart TD
    A[_add_to_vector_store<br/>infer=False] --> B[遍历消息]
    
    B --> C{每条消息}
    C -->|循环| D[验证消息格式]
    
    D --> D1{是 dict?<br/>有 role?<br/>有 content?}
    D1 -->|NO| D2[跳过/警告]
    D1 -->|YES| E{role == system?}
    
    E -->|YES| E1[跳过]
    E -->|NO| F[构建 per_msg_meta]
    
    E1 --> C
    F --> F1[复制 metadata]
    F1 --> F2[添加 role]
    F1 --> F3[添加 actor_id]
    
    F3 --> G[生成 embedding]
    G --> H[_create_memory]
    
    H --> I[添加结果到列表]
    I --> C
    
    C -->|结束| J[返回结果列表]
    
    style A fill:#ff9,stroke:#333
    style J fill:#9f9,stroke:#333
```

## 完整时序图

```mermaid
sequenceDiagram
    participant U as 用户
    participant A as mem0.add()
    participant M as Memory 主类
    participant V as _add_to_vector_store
    participant G as _add_to_graph
    participant LLM as LLM
    participant EMB as Embedding Model
    participant VS as Vector Store
    participant DB as SQLite
    participant GR as Graph Store

    U->>A: add(messages, user_id, ...)

    A->>A: 输入预处理
    A->>M: 判断 memory_type

    alt procedural_memory
        M->>LLM: 生成程序性记忆
        LLM-->>M: 返回记忆内容
        M->>V: 插入向量存储
    else 普通记忆 (并行)
        M->>V: _add_to_vector_store
        M->>G: _add_to_graph
        
        par 向量存储流程
            V->>LLM: 事实提取 (步骤1)
            LLM-->>V: facts 列表
            
            V->>EMB: 生成 embedding
            EMB-->>V: vectors
            
            V->>VS: 搜索相似记忆
            VS-->>V: existing memories
            
            V->>LLM: 决策操作 (步骤2)
            LLM-->>V: ADD/UPDATE/DELETE/NONE
            
            alt ADD
                V->>V: _create_memory
                V->>EMB: 生成 embedding
                EMB-->>V: vectors
                V->>VS: 插入向量
                V->>DB: 记录历史
            end
        and 图存储流程
            G->>GR: 添加实体和关系
            GR-->>G: entities
        end
    end

    A-->>U: {results, relations}
```

## 流程决策图

```mermaid
flowchart TD
    START([开始 add]) --> A{检查 memory_type}
    
    A -->|procedural_memory| P[程序性记忆流程]
    A -->|其他| B{infer 参数}
    
    P --> P1[LLM 生成记忆]
    P1 --> P2[插入向量存储]
    P2 --> END([返回结果])
    
    B -->|infer=True| C[智能模式]
    B -->|infer=False| D[直接模式]
    
    C --> C1[LLM 事实提取]
    C1 --> C2[搜索现有记忆]
    C2 --> C3[LLM 决策操作]
    C3 --> C4{根据 event 执行}
    
    C4 -->|ADD| C5[_create_memory]
    C4 -->|UPDATE| C6[_update_memory]
    C4 -->|DELETE| C7[_delete_memory]
    C4 -->|NONE| C8[更新 session IDs]
    
    C5 --> END
    C6 --> END
    C7 --> END
    C8 --> END
    
    D --> D1[遍历消息]
    D1 --> D2[直接 _create_memory]
    D2 --> END
    
    style START fill:#f9f,stroke:#333
    style END fill:#9f9,stroke:#333
```

## 存储层级关系

```mermaid
graph TB
    subgraph 用户层
        U[用户调用 add]
    end
    
    subgraph mem0 核心层
        A[add 方法]
        V[_add_to_vector_store]
        G[_add_to_graph]
    end
    
    subgraph 存储层
        subgraph 向量存储
            VS[Vector Store]
            EMB[Embedding Model]
        end
        
        subgraph 关系存储
            GR[Graph Store]
        end
        
        subgraph 历史记录
            DB[SQLite History]
        end
    end
    
    U --> A
    
    A --> V
    A --> G
    
    V --> EMB
    V --> VS
    V --> DB
    
    G --> GR
    
    style U fill:#f9f,stroke:#333
    style VS fill:#9ff,stroke:#333
    style GR fill:#9ff,stroke:#333
    style DB fill:#9ff,stroke:#333
```