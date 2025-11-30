
          
# 音频指标收集代码分析

下面是data-juicer中三个音频指标（音频时长、音频信噪比、音频大小）的收集实现分析：

## 1. 音频时长(audio_duration)收集分析

`audio_duration_filter.py`实现了音频时长的计算和过滤：

```python
# 导入必要的库
import librosa
import numpy as np
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_audio, load_data_with_context

# 核心计算函数
class AudioDurationFilter(Filter):
    # ...
    def compute_stats_single(self, sample, context=False):
        # 检查是否已经计算过
        if StatsKeys.audio_duration in sample[Fields.stats]:
            return sample

        # 样本中没有音频
        if self.audio_key not in sample or not sample[self.audio_key]:
            sample[Fields.stats][StatsKeys.audio_duration] = np.array([], dtype=np.float64)
            return sample

        # 加载音频文件
        loaded_audio_keys = sample[self.audio_key]
        sample, audios = load_data_with_context(sample, context, loaded_audio_keys, load_audio)

        # 计算每个音频的时长 - 核心计算逻辑
        audio_durations = {
            audio_key: librosa.get_duration(y=audio[0], sr=audio[1]) 
            for audio_key, audio in audios.items()
        }

        # 存储结果到样本的stats字段
        sample[Fields.stats][StatsKeys.audio_duration] = [
            audio_durations[audio_key] for audio_key in sample[self.audio_key]
        ]

        return sample
```

`mm_utils.py`中的音频加载函数：

```python
def load_audio(path, sampling_rate=None):
    # 使用datasets库的Audio特性加载音频
    aud_feature = Audio(sampling_rate)
    aud = aud_feature.decode_example(aud_feature.encode_example(path))
    # 返回音频数据数组和采样率
    return aud["array"], aud["sampling_rate"]
```

## 2. 音频信噪比(audio_nmf_snr)收集分析

`audio_nmf_snr_filter.py`实现了使用NMF(非负矩阵分解)算法计算音频信噪比：

```python
# NMF信号分离函数
# 将音频分离为信号和噪声部分
def separate_signal_noise(audio, n_components=2, nmf_iter=500):
    # 使用短时傅里叶变换转换到频谱域
    S = np.abs(librosa.stft(audio))

    # 运行NMF分解音频
    W, H = decompose(S, n_components=n_components, init="random", 
                     random_state=0, max_iter=nmf_iter)

    # 获取信号和噪声
    signal = np.dot(W[:, 0:1], H[0:1, :])
    noise = np.dot(W[:, 1:2], H[1:2, :])

    # 转换回时域
    signal_audio = librosa.istft(signal * np.exp(1j * np.angle(S)))
    noise_audio = librosa.istft(noise * np.exp(1j * np.angle(S)))

    return signal_audio, noise_audio

# SNR计算函数
def compute_nmf_snr(audio_data, nmf_iter=500):
    # 分离信号和噪声
    signal, noise = separate_signal_noise(audio_data, n_components=2, nmf_iter=nmf_iter)

    # 计算信号和噪声的功率
    power_signal = np.mean(signal**2)
    power_noise = np.mean(noise**2)

    # 计算dB单位的SNR
    if power_noise == 0:
        snr = np.finfo(np.float64).max
    else:
        snr = 10 * np.log10(power_signal / power_noise)

    return snr

# 在过滤器类中收集SNR指标
class AudioNMFSNRFilter(Filter):
    # ...
    def compute_stats_single(self, sample, context=False):
        # 检查是否已经计算过
        if StatsKeys.audio_nmf_snr in sample[Fields.stats]:
            return sample

        # 样本中没有音频
        if self.audio_key not in sample or not sample[self.audio_key]:
            sample[Fields.stats][StatsKeys.audio_nmf_snr] = np.array([], dtype=np.float64)
            return sample

        # 加载音频
        loaded_audio_keys = sample[self.audio_key]
        sample, audios = load_data_with_context(sample, context, loaded_audio_keys, load_audio)

        # 计算每个音频的SNR - 核心计算逻辑
        audio_snrs = {audio_key: compute_nmf_snr(audio[0], self.nmf_iter_num) 
                     for audio_key, audio in audios.items()}

        # 存储结果
        sample[Fields.stats][StatsKeys.audio_nmf_snr] = [
            audio_snrs[audio_key] for audio_key in sample[self.audio_key]
        ]

        return sample
```

## 3. 音频大小(audio_sizes)收集分析

`audio_size_filter.py`实现了音频文件大小的计算：

```python
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import get_file_size, size_to_bytes

class AudioSizeFilter(Filter):
    # ...
    def compute_stats_single(self, sample, context=False):
        # 检查是否已经计算过
        if StatsKeys.audio_sizes in sample[Fields.stats]:
            return sample

        # 样本中没有音频
        if self.audio_key not in sample or not sample[self.audio_key]:
            sample[Fields.stats][StatsKeys.audio_sizes] = np.array([], dtype=np.float64)
            return sample

        # 对于大小计算，不需要将音频加载到内存 - 核心计算逻辑
        sample[Fields.stats][StatsKeys.audio_sizes] = [
            get_file_size(aud_path) for aud_path in sample[self.audio_key]
        ]

        return sample
```

`mm_utils.py`中的文件大小获取函数：

```python
def get_file_size(path):
    import os
    # 直接使用os.path.getsize获取文件字节大小
    return os.path.getsize(path)
```

## 指标收集流程总结

1. **统一的指标存储位置**：所有指标都存储在样本的`Fields.stats`字段下，使用`StatsKeys`中定义的键名

2. **计算流程**：
   - 检查指标是否已存在（避免重复计算）
   - 检查样本是否包含音频
   - 加载音频文件（对于时长和SNR）或直接获取文件信息（对于大小）
   - 计算指标值
   - 将结果存储到样本的stats字段

3. **差异点**：
   - **音频时长**：使用librosa库计算音频文件的实际播放时长
   - **音频信噪比**：使用NMF算法分解音频，计算信号和噪声的功率比
   - **音频大小**：直接通过操作系统API获取文件的字节大小，无需解码音频

4. **性能考虑**：
   - 使用上下文缓存避免重复加载音频
   - 文件大小计算不需要解码音频，性能最优
   - SNR计算需要执行NMF分解，计算成本最高

这些指标为音频数据的质量评估提供了重要依据，在data-juicer的数据处理管道中用于过滤低质量的音频数据。
        