# 第十八章：向量索引 - HNSW 实现

## 🎯 核心概览

HNSW（Hierarchical Navigable Small World）是一种分层图结构，支持高效的向量搜索和增量索引。

---

## 📊 HNSW 原理

### 分层结构

```
Layer 2:  A --- C
          |     |
Layer 1:  A - B - C - D
          |   X   |   |
Layer 0:  A - B - C - D - E - F
                    X
```

- **底层（Layer 0）**：包含所有向量，完整连接
- **高层（Layer 1+）**：跳过某些节点，快速导航

### 搜索过程

```
1. 从顶层随机节点开始
2. 贪心搜索找最近邻
3. 下降一层
4. 重复步骤 2-3 直到底层
5. 在底层进行详细搜索
```

---

## 🔧 核心特性

### 增量支持

```rust
let mut hnsw = HnswIndex::new(768, max_connections);

// 添加新向量（无需重建）
hnsw.insert(new_vector, new_id)?;

// Lance 中：
// - IVF_PQ 需要重新训练
// - HNSW 可以直接插入
```

### 性能特征

```
100 万向量：

IVF_PQ：
- 构建时间：10 分钟
- 搜索时间：5ms
- 插入：不支持增量（需重建）

HNSW：
- 构建时间：30 分钟
- 搜索时间：20ms
- 插入：O(log N) ≈ 1ms/向量
```

---

## 💡 选择指南

| 特性 | IVF_PQ | HNSW |
|------|--------|------|
| 搜索速度 | 快 | 较快 |
| 增量支持 | 否 | 是 |
| 构建时间 | 快 | 慢 |
| 内存占用 | 低 | 中等 |
| 适用场景 | 静态数据 | 动态数据 |

---

## 🎯 HNSW 的设计思想

### HNSW 解决的问题

IVF 的局限：
1. **静态数据结构**：无法高效处理增量数据
2. **固定质心**：新数据可能不适应现有簇
3. **重建成本高**：添加 100 万新向量需要重新训练 KMeans

**HNSW 的新思路**：用图连接相邻向量，通过贪心遍历找最近邻

```
传统思维 (IVF)：
空间分割
└─ 已知全局统计 (质心)
└─ 搜索时逐个扫描分区

图论思维 (HNSW)：
图连接
└─ 构建时连接相邻节点
└─ 搜索时沿着边贪心走
```

### HNSW 的分层导航

#### 为什么需要分层？

```
Flat 搜索：遍历所有 N 个向量
耗时：O(N)

HNSW 分层搜索：分层导航
Layer 3:  A ─── C              (顶层：快速定位)
Layer 2:  A ─ X ─ C            (中层：逐步精化)
Layer 1:  A ─ B ─ X ─ C ─ D    (底层：完整搜索)
Layer 0:  A ─ B ─ C ─ D ─ E─...

搜索过程：
1. 从顶层随机节点开始
2. 在该层贪心走到最近节点
3. 下降一层，重复贪心
4. 直到底层，返回 Top-K

时间复杂度：O(log N)  同时对高维友好
```

#### 分层的数学原理

```
Small-World 网络性质：
- 高聚集系数：邻居的邻居可能是你的邻居
- 短路径长度：任意两点的距离短

分层的作用：
- Layer 0 (底层)：完整连接，所有向量
- Layer i (上层)：跳过某些节点，快速导航
- 层数：log(N) 左右，避免线性增长

参数 M=20 的选择：
- 太少 (M=2)：图不连通、搜索困难
- 太多 (M=100)：内存爆炸、构建缓慢
- 合适 (M=20)：权衡连接质量与成本
```

### HNSW vs IVF：设计对比

```
IVF 的思想：
┌─────────────────────────────┐
│ 预计算全局统计 (KMeans)      │
│ 分割空间成 256 个分区        │
│ 搜索时选择最相关的分区       │
│ 限制：固定结构，难以增量     │
└─────────────────────────────┘

HNSW 的思想：
┌─────────────────────────────┐
│ 动态构建局部图              │
│ 每个向量连接 M 个邻居       │
│ 搜索时沿着图遍历            │
│ 优点：灵活，支持增量 ✓      │
└─────────────────────────────┘

概念      │ IVF           │ HNSW
━━━━━━━━┿───────────────┼──────────────
数据分割  │ 全局 KMeans   │ 局部图连接
搜索方式  │ 按簇扫描      │ 图上遍历
增量更新  │ 困难          │ 简单 ✓
精确度    │ 99%           │ 99.8% ✓
搜索速度  │ 5ms ✓         │ 20ms
内存占用  │ 低 ✓          │ 中等
```

---

## 🔑 HNSW 参数详解

### max_level（最大层数）

```
含义：图的高度

选择原则：
max_level = floor(ln(N))  (N = 向量数)

例子：
10 万向量   → max_level ≈ 11
1000 万向量 → max_level ≈ 16
1 亿向量    → max_level ≈ 17

权衡：
- max_level ↑ → 搜索快 (更多导航层)
               → 内存增加
- max_level ↓ → 内存省
               → 搜索变慢
```

### m（连接数）

```
含义：每个节点的邻接数量
默认值：20

理论依据：
m = 5-50 时性能稳定
m < 5：图不连通
m > 50：收益递减，内存爆炸

内存计算：
内存 = N × m × avg_level × 指针大小
     = 1M × 20 × 10 × 8字节
     ≈ 1.6GB (仅图结构)

对比 IVF_PQ：
IVF_PQ 内存 = 1M × 8字节 = 8MB
HNSW 内存   = 1M × 20 × 10 × 8字节 = 1.6GB
HNSW 的开销是 IVF_PQ 的 200 倍！
```

### ef_construction（构建候选集）

```
含义：构建图时的搜索范围
默认值：150

过程：
1. 插入第 i 个节点
2. 搜索 ef_construction 个候选
3. 选择其中最近的 m 个连接
4. 更新现有节点的邻接表

权衡：
ef_construction ↑ → 图质量好 → 搜索快 ✓
                 → 构建慢
ef_construction ↓ → 构建快
                 → 图质量差 → 搜索慢

经验公式：
ef_construction = 200 + sqrt(N)  (好的起点)
```

---

## 💻 代码实现示例

### Python：HNSW 索引构建与搜索

```python
import lance
import numpy as np
import time

# 生成测试数据
data = {
    'id': list(range(500_000)),
    'vector': [np.random.rand(768).astype(np.float32) for _ in range(500_000)],
}

dataset = lance.write_dataset(data, 'hnsw_test.lance')

# 步骤1：创建 IVF_HNSW_PQ 索引（混合方案）
print("构建 IVF_HNSW_PQ 索引...")
start_time = time.time()

dataset.create_index(
    'vector',
    index_type='IVF_HNSW_PQ',
    name='hnsw_idx',
    metric='L2',
    # IVF 参数
    num_partitions=64,
    # HNSW 参数
    max_level=7,           # 分层数
    m=20,                  # 连接数
    ef_construction=100,   # 构建候选集
    # PQ 参数
    num_bits=8,
    num_sub_vectors=8,
    replace=True,
)

build_time = time.time() - start_time
print(f"索引构建完成: {build_time:.2f}s")

# 步骤2：执行搜索
query_vector = np.random.rand(768).astype(np.float32)

print("\nHNSW 搜索性能：")
for ef in [10, 50, 100]:
    start_time = time.time()
    results = dataset.search(
        query_vector,
        k=100,
        ef=ef,  # HNSW 搜索参数
        nprobes=1,  # IVF 分区
    ).to_list()
    search_time = (time.time() - start_time) * 1000
    print(f"  ef={ef:3d}, 耗时={search_time:.2f}ms")

# 步骤3：增量数据追加（HNSW 的优势）
print("\n测试增量数据追加...")
new_data = {
    'id': list(range(500_000, 600_000)),
    'vector': [np.random.rand(768).astype(np.float32) for _ in range(100_000)],
}

append_start = time.time()
dataset.add(new_data)  # 快速追加
append_time = time.time() - append_start

print(f"追加 100k 向量耗时: {append_time:.2f}s")
```

### Rust：HNSW 索引构建

```rust
use lance_index::vector::{
    hnsw::builder::{HnswBuildParams, HNSW},
    flat::storage::FlatFloatStorage,
};
use lance_linalg::distance::DistanceType;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 步骤1：配置 HNSW 参数
    let params = HnswBuildParams::default()
        .max_level(7)              // 图的层数
        .num_edges(20)             // 每个节点的连接数
        .ef_construction(150);     // 构建时的候选集
    
    // 步骤2：构建 HNSW 索引
    println!("开始构建 HNSW 索引...");
    let start = std::time::Instant::now();
    
    let hnsw = HNSW::index_vectors(vector_store.as_ref(), params)?;
    
    let build_time = start.elapsed();
    println!("HNSW 索引构建完成: {:?}", build_time);
    println!("  节点数: {}", hnsw.len());
    println!("  最大层数: {}", hnsw.max_level());
    
    // 步骤3：执行搜索
    println!("\nHNSW 搜索性能测试:");
    
    for ef in [10, 50, 100] {
        let start = std::time::Instant::now();
        let results = hnsw.search_basic(
            query_vector.clone(),
            100,           // k
            ef,            // ef 参数
            None,          // 预过滤器
            vector_store.as_ref(),
        )?;
        
        let search_time = start.elapsed().as_micros();
        println!("  ef={:3}, 耗时={:.0}us", ef, search_time);
    }
    
    // 步骤4：分析图的结构
    println!("\nHNSW 图结构分析:");
    for level in 0..hnsw.max_level() as usize {
        let num_nodes = hnsw.num_nodes(level);
        println!("  Layer {}: {} 个节点", level, num_nodes);
    }
    
    Ok(())
}
```

---

## 📊 性能对比

### HNSW vs IVF_PQ（1000 万向量）

```
指标           IVF_PQ      HNSW
─────────────────────────────────
搜索延迟       5ms         25ms
Recall        99%         99.8% ✓
QPS/机器      1600        400
构建时间       7m          30m
内存占用       1GB         31GB
增量支持       否          是 ✓

使用建议：
- 追求速度 → IVF_PQ
- 追求精度 → HNSW
- 追求增量 → HNSW ✓
```

### 不同场景的选择

```
场景               推荐      原因
─────────────────────────────────────────
1M-10M 静态数据    IVF_PQ   速度优先
1M-10M 动态数据    HNSW     增量优先
10M-100M 数据      IVF_PQ   大规模优化
实时更新系统       HNSW     O(log N) 插入
精度优先场景       HNSW     99.8% recall
```

---

## 🎯 应用场景：内容审核系统

### 背景
- 不安全图片库：5000 万
- 新图片每月增加：100 万
- 实时审核：1000 QPS
- 精度要求：99.5% 以上

### 为什么选择 IVF_HNSW_PQ？

1. **数据持续增长** → 需要增量能力
2. **精度要求高** → HNSW 有更好 recall
3. **吞吐量中等** → 10-20ms 搜索可接受
4. **成本约束** → 128GB 内存可接受

### 实现架构

```python
class ContentModerator:
    def __init__(self):
        self.dataset = lance.open('unsafe_images.lance')
    
    def moderate(self, image_vector):
        """实时审核"""
        results = self.dataset.search(
            image_vector,
            k=10,
            nprobes=8,
            ef=200,  # HNSW 参数
        ).to_list()
        
        if results and results[0]['_distance'] > 0.9:
            return {'is_unsafe': True, 'confidence': results[0]['_distance']}
        return {'is_unsafe': False}
    
    def add_unsafe_images(self, new_vectors):
        """增量添加（HNSW 支持）"""
        for vec in new_vectors:
            self.dataset.add(vec)
```

---

## 总结

- **HNSW**：动态向量索引的最佳选择
- **核心优势**：增量支持、精度高、延迟稳定
- **核心劣势**：内存开销大、搜索较慢
- **参数调优**：max_level、m、ef_construction 三大参数
- **应用指南**：数据实时更新或精度优先时选择 HNSW

与 IVF_PQ 对比，HNSW 在以下场景优选：
- ✓ 需要实时增量更新
- ✓ 精度要求 >99%
- ✓ 数据规模 <10 亿
- ✓ 内存充足

IVF_PQ 在以下场景优选：
- ✓ 追求极致搜索速度
- ✓ 数据规模 >10 亿
- ✓ 内存严格限制
- ✓ 静态或日更数据
