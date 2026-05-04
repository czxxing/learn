


          
我需要先探索项目结构，找到vrag-agent相关的代码文件。让我先查看项目目录结构。
        
            
toolName: view_folder
            
status: success
          
/home/project/VRAG
          
让我搜索项目中与vrag-agent相关的文件：

        
            
toolName: file_search
            
status: success
          
          

        
            
toolName: search_by_regex
            
status: success
          
query: vrag
          
让我先查看主要的vrag-agent文件内容：


        
            
toolName: view_files
            
status: success
          
            
filePath: /home/project/VRAG/demo/vrag_agent.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/project/VRAG/demo/app.py
          
让我继续查看相关的辅助文件以获得更完整的理解：

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/project/VRAG/demo/vimrag_utils.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/project/VRAG/demo/vimrag_prompt.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/project/VRAG/search_engine/search_engine_api.py
          
# VRAG Agent 执行过程逐行分析

## 一、整体架构概述

`vrag_agent.py` 实现了一个**视觉检索增强生成 (Visual Retrieval-Augmented Generation, VRAG)** 智能代理。该代理能够：
1. 理解用户问题并进行推理思考
2. 调用外部搜索引擎检索相关图片
3. 对图片进行区域裁剪以获取更清晰的信息
4. 基于收集到的信息给出最终答案

---

## 二、逐行代码分析

### 1. 导入依赖模块

```python
import base64
import json
import re
import requests
import math
from io import BytesIO

from openai import OpenAI
from PIL import Image, ImageDraw
```

| 模块 | 用途 |
|------|------|
| `base64` | 将图片转换为Base64编码格式 |
| `json` | 处理JSON数据（用于bbox坐标） |
| `re` | 正则表达式解析LLM返回的XML标签 |
| `requests` | 调用搜索引擎API |
| `math` | 计算图片缩放因子 |
| `BytesIO` | 在内存中处理图片字节流 |
| `OpenAI` | 调用大语言模型（兼容Qwen API） |
| `PIL.Image` | 图片加载、裁剪、缩放等操作 |
| `PIL.ImageDraw` | 在图片上绘制矩形框（标记ROI区域） |

---

### 2. 系统提示词定义

```python
prompt_ins = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
'''
```

**设计意图**：
- 定义了代理与LLM交互的**协议格式**
- `<think>` 标签：要求LLM输出推理过程
- `<search>` 标签：触发搜索引擎调用
- `<bbox>` 标签：触发图片裁剪操作
- `<answer>` 标签：表示回答完成

---

### 3. VRAG类初始化

```python
class VRAG:
    def __init__(self, 
                base_url='http://0.0.0.0:8002/v1', 
                search_url='http://0.0.0.0:8001/search',
                generator=True,
                api_key='EMPTY'):
    
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.search_url = search_url

        self.max_pixels = 512 * 28 * 28
        self.min_pixels = 256 * 28 * 28
        self.repeated_nums = 1
        self.max_steps = 10

        self.generator = generator
```

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `base_url` | `http://0.0.0.0:8002/v1` | 大语言模型API地址 |
| `search_url` | `http://0.0.0.0:8001/search` | 搜索引擎API地址 |
| `generator` | `True` | 是否使用生成器模式（流式输出） |
| `api_key` | `EMPTY` | API密钥（Qwen本地部署时可设为任意值） |

**关键配置**：
- `max_pixels` / `min_pixels`：图片像素数的上下限（确保模型能正常处理）
- `repeated_nums`：防止重复使用同一张图片
- `max_steps`：最大推理步数（防止无限循环）

---

### 4. 图片处理方法 `process_image`

```python
def process_image(self, image):
    # 1. 图片格式统一
    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, str):
        image = Image.open(image)

    # 2. 根据像素限制缩放图片
    if (image.width * image.height) > self.max_pixels:
        resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < self.min_pixels:
        resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    # 3. 转换为RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 4. 编码为Base64格式
    byte_stream = BytesIO()
    image.save(byte_stream, format="JPEG")
    byte_array = byte_stream.getvalue()
    base64_encoded_image = base64.b64encode(byte_array)
    base64_string = base64_encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{base64_string}"

    return image, base64_qwen
```

**执行流程**：
1. **格式统一**：支持字典（含bytes）和文件路径两种输入格式
2. **像素归一化**：确保图片尺寸在模型可接受范围内
3. **颜色模式转换**：统一转换为RGB（PNG可能带alpha通道）
4. **Base64编码**：转换为Data URI格式，便于通过API传输

---

### 5. 搜索方法 `search`

```python
def search(self, query):
    if isinstance(query, str):
        query = [query]
    search_response = requests.post(self.search_url, json={"queries": query, "top_k": 3})
    search_results = search_response.json()
    image_path_list = [result['image_file'] for result in search_results[0]]
    return image_path_list
```

**执行流程**：
1. 将字符串查询包装为列表格式
2. 向搜索引擎发送POST请求，获取top-3相关图片
3. 提取图片文件路径列表返回

---

### 6. 核心运行方法 `run`

这是整个代理的核心逻辑，采用**生成器模式**实现流式输出。

#### 6.1 初始化阶段

```python
def run(self, question):
    self.image_raw = []      # 存储原始尺寸图片
    self.image_input = []    # 存储处理后输入模型的图片
    self.image_path = []     # 存储图片路径（去重用）
    
    prompt = prompt_ins.format(question=question)
    messages = [dict(
        role="user",
        content=[
            {
                "type": "text",
                "text": prompt,
            }
        ]
    )]
```

**数据结构**：
- `image_raw`：保存原始高分辨率图片（用于精确裁剪）
- `image_input`：保存缩放后的图片（用于输入模型）
- `image_path`：记录已使用的图片路径（防止重复）

#### 6.2 主循环：推理-动作-反馈

```python
max_steps = self.max_steps
while True:
    ## 1. 调用LLM获取响应
    response = self.client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=messages,
        stream=False
    )
    response_content = response.choices[0].message.content
    
    # 将响应加入对话历史
    messages.append(dict(
        role="assistant",
        content=[{
            "type": "text",
            "text": response_content
        }]
    ))
```

**LLM调用细节**：
- 使用Qwen2.5-VL-7B多模态模型
- `stream=False`：非流式响应（等待完整结果）

---

#### 6.3 解析LLM响应

```python
## 2. 解析思考内容
pattern = r'<think>(.*?)</think>'
match = re.search(pattern, response_content, re.DOTALL)
thought = match.group(1)
if self.generator:
    yield 'think', thought, match.group(0)

## 3. 解析动作指令
pattern = r'<(search|answer|bbox)>(.*?)</\1>'
match = re.search(pattern, response_content, re.DOTALL)
if match:
    raw_content = match.group(0)
    content = match.group(2).strip()
    action = match.group(1)
else:
    content = ''
    action = None
```

**正则解析逻辑**：
- `<think>...</think>`：提取推理思考内容
- `<search|answer|bbox>...</search|answer|bbox>`：提取动作指令

**动作类型**：
| 动作 | 含义 |
|------|------|
| `search` | 需要搜索相关图片 |
| `answer` | 可以直接回答问题 |
| `bbox` | 需要裁剪图片的特定区域 |

---

#### 6.4 终止条件判断

```python
## 4. 终止条件判断
if action == 'answer':
    return 'answer', content, raw_content
elif max_steps == 0:
    return 'answer', 'Sorry, I can not retrieval something about the question.', ''
elif self.generator:
    yield action, content, raw_content
```

**终止逻辑**：
1. 如果收到`answer`动作，直接返回最终答案
2. 如果达到最大步数仍未回答，返回失败提示
3. 否则通过生成器yield当前动作状态

---

#### 6.5 执行具体动作

##### 搜索动作 (`search`)

```python
if action == 'search':
    search_results = self.search(content)
    # 查找未使用过的图片
    while len(search_results) > 0:
        image_path = search_results.pop(0)
        if self.image_path.count(image_path) >= self.repeated_nums:
            continue
        else:
            self.image_path.append(image_path)
            break

    # 加载并处理图片
    image_raw = Image.open(image_path)
    image_input, img_base64 = self.process_image(image_raw)
    
    # 构建用户消息（图片）
    user_content=[{
        'type': 'image_url',
        'image_url': {
            'url': img_base64
        }
    }]
    
    # 保存图片引用
    self.image_raw.append(image_raw)
    self.image_input.append(image_input)
    
    if self.generator:
        yield 'search_image', self.image_input[-1], raw_content
```

**搜索执行流程**：
1. 调用搜索API获取图片列表
2. 过滤已使用的图片（去重）
3. 处理图片（缩放、编码）
4. 将图片加入对话历史，供LLM下一步推理使用

---

##### 裁剪动作 (`bbox`)

```python
elif action == 'bbox':
    bbox = json.loads(content)  # 解析裁剪区域坐标
    
    # 计算实际裁剪区域（考虑原始图片与输入图片的尺寸差异）
    input_w, input_h = self.image_input[-1].size
    raw_w, raw_h = self.image_raw[-1].size
    crop_region_bbox = (
        bbox[0] * raw_w / input_w, 
        bbox[1] * raw_h / input_h, 
        bbox[2] * raw_w / input_w, 
        bbox[3] * raw_h / input_h
    )
    
    # 添加padding，扩大裁剪区域
    pad_size = 56
    crop_region_bbox = [
        max(crop_region_bbox[0]-pad_size, 0), 
        max(crop_region_bbox[1]-pad_size, 0), 
        min(crop_region_bbox[2]+pad_size, raw_w), 
        min(crop_region_bbox[3]+pad_size, raw_h)
    ]
    
    # 执行裁剪
    crop_region = self.image_raw[-1].crop(crop_region_bbox)
    image_input, img_base64 = self.process_image(crop_region)
    
    # 构建用户消息
    user_content=[{
        'type': 'image_url',
        'image_url': {
            'url': img_base64
        }
    }]
    
    # 保存裁剪结果
    self.image_raw.append(crop_region)
    self.image_input.append(image_input)

    if self.generator:
        # 在原图上绘制裁剪框
        image_to_draw = self.image_input[-2].copy()
        draw = ImageDraw.Draw(image_to_draw)
        draw.rectangle(bbox, outline=(160, 32, 240), width=7)
        yield 'crop_image', self.image_input[-1], image_to_draw
```

**裁剪执行流程**：
1. 解析JSON格式的bbox坐标 `[x1, y1, x2, y2]`
2. **坐标映射**：将输入图片坐标转换为原始图片坐标（因为输入图片可能被缩放）
3. **添加padding**：向外扩展56像素，确保裁剪区域包含足够上下文
4. 执行裁剪并处理
5. 绘制裁剪框可视化，便于用户理解

---

#### 6.6 循环控制

```python
max_steps -= 1
if max_steps == 0:
    user_content.append({
        'type': 'text',
        'text': 'please answer the question now with answer in <answer> ... </answer>' 
    })

messages.append(dict(
    role='user',
    content=user_content
))
```

**循环逻辑**：
- 每执行一次动作，步数减1
- 当步数为0时，强制LLM给出答案
- 将图片/文本加入对话历史，进入下一轮循环

---

### 7. 主程序入口

```python
if __name__ == '__main__':
    agent = VRAG()
    generator = agent.run('How are u?')
    while True:
        print(next(generator))
```

简单的测试代码，展示如何使用VRAG代理。

---

## 三、执行流程图

```
用户提问 → 构建初始prompt → 调用LLM
                            ↓
                    解析响应内容
                    ├─ <think> 推理内容
                    └─ <action> 动作指令
                            ↓
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
         <search>      <bbox>       <answer>
              ↓             ↓             ↓
        调用搜索API    裁剪图片区域    返回最终答案
              ↓             ↓
        获取图片路径    获取裁剪区域
              ↓             ↓
        处理图片(Base64) 处理裁剪图片
              ↓             ↓
        加入对话历史    加入对话历史
              ↓             ↓
         回到LLM循环   回到LLM循环
```

---

## 四、关键设计亮点

| 设计点 | 实现方式 | 优势 |
|--------|----------|------|
| **生成器模式** | `yield` 流式输出 | 支持实时展示推理过程 |
| **图片去重** | `image_path` 记录已使用路径 | 避免重复检索相同图片 |
| **坐标映射** | 原始图与输入图比例换算 | 保证裁剪精度 |
| **Padding扩展** | 裁剪区域向外扩展56像素 | 保留上下文信息 |
| **最大步数限制** | `max_steps=10` | 防止无限循环 |
| **格式协议** | XML标签格式 | LLM易于理解和生成 |

---

## 五、输入输出示例

**输入**：
```python
agent = VRAG()
generator = agent.run("What is the most commonly used travel app?")
```

**输出流程**（通过生成器逐步返回）：
```python
('think', 'I need to find information about the most commonly used travel app.', '<think>...</think>')
('search', 'most commonly used travel app', '<search>...</search>')
('search_image', <PIL.Image>, '<search>...</search>')
('think', 'Looking at the search results, I can see...', '<think>...</think>')
('answer', 'According to the search results, the most commonly used travel app is...', '<answer>...</answer>')
```

---

## 六、与前端app.py的协作

`app.py` 通过Streamlit构建UI，调用`vrag_agent.py`的流程：

1. 用户输入问题 → 点击提交
2. 创建`VRAG`实例，调用`run()`方法获取生成器
3. 循环迭代生成器，根据动作类型展示：
   - `think`：显示推理过程（打字机效果）
   - `search`：显示搜索关键词
   - `search_image`：显示检索到的图片
   - `bbox`：显示裁剪区域坐标
   - `crop_image`：显示原图+裁剪框和裁剪结果
   - `answer`：显示最终答案

整个交互过程实现了**可视化的推理链条**，用户可以清晰看到代理如何一步步获取信息并得出结论。
        