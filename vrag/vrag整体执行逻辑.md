


          
让我深入分析VRAG的执行流程和代码调用链：


        
            
toolName: view_files
            
status: success
          
            
filePath: /home/project/VRAG/search_engine/search_engine_api.py
          
## VRAG 执行流程与代码调用深度分析

### 一、整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        三层架构                                     │
├─────────────────────────────────────────────────────────────────────┤
│  1. Streamlit 前端        2. VRAG Agent        3. 后端服务         │
│     (demo/app.py)         (demo/vrag_agent.py)  (vLLM + Search)   │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 二、启动流程

#### 阶段1：服务启动（run_demo.sh）

```bash
# 1. 启动搜索引擎
python search_engine/search_engine_api.py
# → 加载 Qwen3-VL-Embedding-2B 模型
# → 构建图像索引 (FAISS)

# 2. 启动 vLLM 服务器
vllm serve autumncc/Qwen2.5-VL-7B-VRAG --port 8002
# → 加载 VRAG 微调后的多模态模型
# → 提供 OpenAI 兼容 API

# 3. 启动 Streamlit
streamlit run demo/app.py --port 8501
# → 启动 Web UI
# → 初始化 VRAG Agent
```

#### 阶段2：组件初始化

**搜索引擎初始化**（search_engine_api.py:11-12）：
```python
engine = SearchEngine(model_path="Qwen3-VL-Embedding-2B")
engine.load_multi_index_corpus_together(["search_engine/corpus/image_index"])
```

**VRAG Agent 初始化**（vrag_agent.py:15-21）：
```python
class VRAG:
    def __init__(self, 
                base_url='http://0.0.0.0:8002/v1',  # vLLM API
                search_url='http://0.0.0.0:8001/search'):  # 搜索引擎
        
        self.client = OpenAI(base_url=base_url, api_key='EMPTY')
        self.search_url = search_url
```

---

### 三、核心执行流程（问答流程）

#### 完整调用链

```
用户提问 → Streamlit → VRAG.run() → vLLM推理 → 解析动作 → 搜索/裁剪 → 返回答案
```

#### 阶段1：用户输入

**Streamlit 前端处理**（app.py:116-125）：
```python
question = st.text_input("Question Input:", value=selected_example)
if submit_button and question:
    generator = agent.run(question)  # 调用 VRAG Agent
```

#### 阶段2：VRAG Agent 推理循环

**核心推理逻辑**（vrag_agent.py:67-180）：

```python
def run(self, question):
    # 1. 构建提示词
    prompt = prompt_ins.format(question=question)
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }]
    
    # 2. 循环推理（最多 max_steps 步）
    while max_steps > 0:
        # 3. 调用 vLLM 生成响应
        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=messages,
            stream=False
        )
        response_content = response.choices[0].message.content
        
        # 4. 解析响应中的思考和动作
        # <think>...</think> - 思考过程
        # <search>...</search> - 搜索查询
        # <bbox>...</bbox> - 图像裁剪区域
        # <answer>...</answer> - 最终答案
        
        # 5. 根据动作类型执行相应操作
        if action == 'search':
            search_results = self.search(content)  # 调用搜索引擎
            # 处理图像...
        
        elif action == 'bbox':
            crop_region = self.image_raw[-1].crop(crop_region_bbox)  # 裁剪图像
        
        elif action == 'answer':
            return 'answer', content, raw_content  # 返回答案
```

#### 阶段3：提示词模板

**关键提示词设计**（vrag_agent.py:11）：
```python
prompt_ins = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>. Question: {question}'''
```

#### 阶段4：搜索引擎调用

**搜索接口**（vrag_agent.py:59-65）：
```python
def search(self, query):
    search_response = requests.post(
        self.search_url, 
        json={"queries": [query], "top_k": 3}
    )
    search_results = search_response.json()
    image_path_list = [result['image_file'] for result in search_results[0]]
    return image_path_list
```

**搜索引擎处理**（search_engine_api.py:14-42）：
```python
@app.post("/search")
async def search(request: Request):
    body = await request.json()
    queries = body.get("queries", [])
    top_k = body.get("top_k", 3)
    
    search_results = engine.search(queries, top_k)  # 调用 SearchEngine
    return {"results": search_results}
```

#### 阶段5：图像处理

**图像预处理**（vrag_agent.py:31-57）：
```python
def process_image(self, image):
    # 1. 打开图像
    if isinstance(image, str):
        image = Image.open(image)
    
    # 2. 调整尺寸（限制像素范围）
    if (image.width * image.height) > self.max_pixels:
        resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
    
    # 3. 转换格式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 4. Base64 编码
    byte_stream = BytesIO()
    image.save(byte_stream, format="JPEG")
    base64_string = base64.b64encode(byte_stream.getvalue()).decode("utf-8")
    return f"data:image;base64,{base64_string}"
```

#### 阶段6：动作解析

**正则表达式解析**（vrag_agent.py:99-113）：
```python
# 解析思考内容
pattern = r'<think>(.*?)</think>'
match = re.search(pattern, response_content, re.DOTALL)
thought = match.group(1)

# 解析动作类型
pattern = r'<(search|answer|bbox)>(.*?)</\1>'
match = re.search(pattern, response_content, re.DOTALL)
if match:
    action = match.group(1)    # search/answer/bbox
    content = match.group(2)   # 动作内容
```

---

### 四、数据流与状态管理

#### 数据流转图

```
用户提问 (文本)
       ↓
Streamlit (demo/app.py)
       ↓
VRAG.run() (demo/vrag_agent.py)
       ↓
┌─────────────────────────────────────┐
│  vLLM API (端口8002)               │
│  POST /v1/chat/completions         │
│  {                                 │
│    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
│    "messages": [...]               │
│  }                                 │
└─────────────────────────────────────┘
       ↓
LLM 响应 (<think>...<search>...</search>...)
       ↓
动作解析
       ↓
┌─────────────────────────────────────┐
│  搜索引擎 API (端口8001)            │
│  POST /search                      │
│  {"queries": [...], "top_k": 3}    │
└─────────────────────────────────────┘
       ↓
图像路径列表
       ↓
图像加载 + Base64编码
       ↓
再次调用 vLLM（含图像）
       ↓
最终答案 (<answer>...</answer>)
       ↓
Streamlit 显示
```

#### 状态管理

**VRAG Agent 状态**（vrag_agent.py:68-70）：
```python
def run(self, question):
    self.image_raw = []      # 原始图像列表
    self.image_input = []    # 处理后的图像列表
    self.image_path = []     # 图像路径列表
```

---

### 五、前端展示流程

**Streamlit 响应处理**（app.py:133-174）：

```python
if submit_button and question:
    generator = agent.run(question)
    try:
        while True:
            action, content, raw_content = next(generator)
            
            if action == 'think':
                think = f"💭 Thinking: {content}"
            
            elif action == 'search':
                # 打字机效果显示搜索动作
                typewriter_effect(st.empty(), f'{think} <br> 🔍 Searching: {content}')
            
            elif action == 'search_image':
                # 显示检索到的图像
                st.image(content, width=350)
            
            elif action == 'bbox':
                # 显示图像裁剪区域
                typewriter_effect(st.empty(), f'{think} <br> 📷 ROI: {content}')
            
            elif action == 'crop_image':
                # 显示原图和裁剪后的图像
                with col1:
                    st.image(raw_content, width=350)  # 原图（带bbox标记）
                with col2:
                    st.image(content, width=350)      # 裁剪区域
    
    except StopIteration as e:
        action, content, raw_response = e.value
        if action == 'answer':
            st.success(f"✅ Answer: {content}")
```

---

### 六、关键技术点

#### 1. 多模态输入处理

```python
# 将图像转换为模型可接受的格式
user_content = [{
    'type': 'image_url',
    'image_url': {
        'url': f"data:image;base64,{base64_string}"
    }
}]
```

#### 2. 迭代推理机制

```python
max_steps = self.max_steps  # 默认10步
while max_steps > 0:
    # 推理...
    max_steps -= 1
    
    if max_steps == 0:
        # 强制要求回答
        user_content.append({
            'type': 'text',
            'text': 'please answer the question now with answer in <answer> ... </answer>'
        })
```

#### 3. OpenAI API 兼容性

```python
# vLLM 提供 OpenAI 兼容接口
self.client = OpenAI(base_url='http://0.0.0.0:8002/v1', api_key='EMPTY')

# 调用方式与 OpenAI API 完全一致
response = self.client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=messages
)
```

---

### 七、完整执行时序图

```
用户                    Streamlit               VRAG Agent             vLLM Server           Search Engine
  │                         │                       │                       │                       │
  │   [输入问题]            │                       │                       │                       │
  │ ──────────────────────→│                       │                       │                       │
  │                         │   [agent.run(q)]     │                       │                       │
  │                         │ ───────────────────→│                       │                       │
  │                         │                       │   [POST /v1/chat/completions]
  │                         │                       │ ───────────────────→│                       │
  │                         │                       │                       │   [生成响应]         │
  │                         │                       │                       │ ←───────────────────│
  │                         │                       │   [解析<think><search>]</parameter></function>
  │                         │                       │ ←───────────────────│                       │
  │                         │                       │                       │                       │
  │                         │                       │   [POST /search]     │                       │
  │                         │                       │ ─────────────────────────────────────────────→│
  │                         │                       │                       │                       │   [图像检索]
  │                         │                       │                       │                       │ ←───────────
  │                         │                       │   [返回图像路径]     │                       │
  │                         │                       │ ←────────────────────────────────────────────│
  │                         │                       │                       │                       │
  │                         │                       │   [加载图像+编码]    │                       │
  │                         │                       │                       │                       │
  │                         │                       │   [再次调用vLLM]     │                       │
  │                         │                       │ ───────────────────→│                       │
  │                         │                       │                       │   [分析图像]         │
  │                         │                       │                       │ ←───────────────────│
  │                         │                       │   [解析<answer>]     │                       │
  │                         │                       │ ←───────────────────│                       │
  │                         │   [返回答案]          │                       │                       │
  │                         │ ←───────────────────│                       │                       │
  │   [显示答案]            │                       │                       │                       │
  │ ←─────────────────────│                       │                       │                       │
```

---

### 八、总结

| 阶段 | 组件 | 文件 | 关键操作 |
|------|------|------|----------|
| **启动** | 搜索引擎 | search_engine_api.py | 加载图像索引 |
| **启动** | vLLM | vllm serve | 加载模型权重 |
| **启动** | Streamlit | app.py | 初始化界面 |
| **问答** | VRAG Agent | vrag_agent.py | 构建提示词 |
| **问答** | vLLM | API调用 | 生成响应 |
| **问答** | VRAG Agent | vrag_agent.py | 解析动作 |
| **问答** | 搜索引擎 | API调用 | 图像检索 |
| **问答** | VRAG Agent | vrag_agent.py | 图像处理 |
| **问答** | vLLM | API调用 | 分析图像 |
| **结束** | Streamlit | app.py | 显示结果 |

VRAG的核心设计在于**将LLM的推理能力与外部检索系统相结合**，通过结构化的提示词模板引导模型自主决定是否需要搜索，形成一个闭环的检索增强生成系统。
        