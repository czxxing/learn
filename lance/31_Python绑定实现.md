# 第三十一章：Python 绑定实现

## 🎯 核心概览

Lance 通过 PyO3 将高性能的 Rust 实现暴露给 Python，同时管理异步运行时、处理错误转换等。Python API 让用户能够像使用 Pandas 一样使用 Lance。

---

## 🐍 PyO3 基础

### Rust 函数导出为 Python

```rust
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pymodule]
fn lance(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDataset>()?;
    m.add_class::<PySearchResults>()?;
    m.add_function(wrap_pyfunction!(open_dataset, m)?)?;
    Ok(())
}

#[pyfunction]
fn open_dataset(path: String) -> PyResult<PyDataset> {
    let dataset = Dataset::open(&path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataset { inner: dataset })
}

#[pyclass]
pub struct PyDataset {
    #[pyo3(get)]
    inner: Arc<Dataset>,
}

#[pymethods]
impl PyDataset {
    #[pyo3(name = "count_rows")]
    fn count_rows(&self) -> PyResult<u64> {
        Ok(self.inner.count_rows())
    }
    
    #[pyo3(name = "schema")]
    fn schema_py(&self) -> PyResult<PyObject> {
        let schema = self.inner.schema();
        // 转换为 Python dict
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for field in schema.fields {
                dict.set_item(&field.name, field.data_type.to_string())?;
            }
            Ok(dict.into())
        })
    }
    
    #[pyo3(name = "search")]
    fn search_py(&self, query: Vec<f32>, k: usize) -> PyResult<PySearchResults> {
        let results = self.inner.search(&query, k)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string()
            ))?;
        Ok(PySearchResults { inner: results })
    }
}

#[pyclass]
pub struct PySearchResults {
    inner: Vec<SearchResult>,
}

#[pymethods]
impl PySearchResults {
    fn __iter__(slf: PyRefMut<Self>) -> PyResult<PySearchResultsIterator> {
        Ok(PySearchResultsIterator {
            iter: slf.inner.clone().into_iter(),
        })
    }
}

#[pyclass]
pub struct PySearchResultsIterator {
    iter: std::vec::IntoIter<SearchResult>,
}

#[pymethods]
impl PySearchResultsIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
    
    fn __next__(&mut self) -> Option<PyObject> {
        self.iter.next().map(|result| {
            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                dict.set_item("id", result.id).ok();
                dict.set_item("distance", result.distance).ok();
                dict.into()
            })
        })
    }
}
```

---

## ⚙️ 异步处理

### Tokio 运行时管理

```rust
use pyo3::types::PyCoroutine;
use tokio::runtime::Runtime;

thread_local! {
    static RUNTIME: Runtime = Runtime::new().unwrap();
}

#[pyfunction]
fn search_async(path: String, query: Vec<f32>) -> PyResult<PyObject> {
    let future = async {
        let dataset = Dataset::open(&path)?;
        dataset.search_async(&query, 10).await
    };
    
    // 在 Tokio 运行时中执行异步代码
    let results = RUNTIME.with(|rt| {
        rt.block_on(future)
    }).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
    })?;
    
    Python::with_gil(|py| {
        Ok(results_to_pyobject(&results, py))
    })
}
```

### Python 异步 API

```rust
use pyo3::types::PyCoroutine;

#[pyfunction]
fn search_coroutine(path: String, query: Vec<f32>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // 创建 Python 协程
        let coroutine = PyCoroutine::from_async_fn(py, async move {
            let dataset = Dataset::open(&path)?;
            dataset.search_async(&query, 10).await
        })?;
        
        Ok(coroutine.into())
    })
}
```

---

## 🔄 错误处理

### 错误转换

```rust
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};

impl From<LanceError> for PyErr {
    fn from(err: LanceError) -> PyErr {
        match err {
            LanceError::IoError(msg) => {
                PyIOError::new_err(msg)
            }
            LanceError::InvalidInput(msg) => {
                PyValueError::new_err(msg)
            }
            LanceError::NotFound(msg) => {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(msg)
            }
            LanceError::Other(msg) => {
                PyRuntimeError::new_err(msg)
            }
        }
    }
}

#[pyfunction]
fn may_fail() -> PyResult<String> {
    do_something()
        .map_err(|e| PyErr::from(e))?;
    Ok("success".to_string())
}
```

---

## 📦 类型转换

### NumPy 集成

```rust
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};

#[pyfunction]
fn search_numpy(
    vectors: PyReadonlyArray2<f32>,  // 输入：[N, D] 浮点数组
) -> PyResult<PyArray1<u64>> {
    let vectors = vectors.as_array();
    
    let mut results = Vec::new();
    for row in vectors.rows() {
        let row_vec: Vec<f32> = row.to_vec();
        // 处理...
        results.push(0u64);  // 示例
    }
    
    Python::with_gil(|py| {
        let result_array = PyArray1::from_slice(py, &results);
        Ok(result_array.to_owned())
    })
}
```

### Arrow 集成

```rust
use arrow::pyarrow::{PyArrowType, ArrowException};

#[pyfunction]
fn to_arrow(
    dataset: &PyDataset,
) -> PyResult<PyObject> {
    let batch = dataset.inner.to_record_batch()
        .map_err(|e| PyErr::new::<ArrowException, _>(e.to_string()))?;
    
    Python::with_gil(|py| {
        // 转换为 PyArrow Table
        Ok(batch.to_pyarrow(py)?)
    })
}
```

---

## 📚 总结

Python 绑定通过 PyO3 实现：
1. **高效互操作**：Rust 性能 + Python 易用性
2. **异步支持**：Tokio + Python asyncio
3. **错误转换**：自动映射到 Python 异常
4. **数据集成**：NumPy、Arrow 互操作
