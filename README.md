# audio-diarization

## 简介

基于 WhisperX 的音频转录、对齐和说话人分离工具。支持自动检测 GPU、分段文本合并、按说话人拆分音频等功能。

## 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [安装说明](#安装说明)
- [使用方法](#使用方法)
- [命令行参数](#命令行参数)
- [输出文件](#输出文件)
- [高级功能](#高级功能)

## 功能特性

- 自动检测 CUDA GPU 并优先使用
- 支持 WhisperX 多语言转录（默认 large-v3 模型）
- 自动对齐转录文本与音频时间戳
- 说话人分离（Diarization）
- 支持韩语、英语等多种语言的句子结束识别
- 自动合并短分段为完整句子
- 按说话人拆分并导出独立音频文件
- 支持断点续传（缓存中间结果 JSON）
- 支持 HuggingFace 镜像加速下载

## 环境要求

- Python 3.10+
- CUDA 支持（可选，推荐用于加速）
- 依赖库：
  - whisperx
  - typer
  - librosa
  - soundfile
  - numpy
  - torch
  - nltk
  - tqdm

## 安装说明

### 1. 克隆仓库

```bash
git clone https://github.com/as436845345/audio-diarization.git
cd audio-diarization
```

### 2. 安装依赖

```bash
pip install whisperx typer librosa soundfile numpy torch nltk tqdm
```

### 3. 配置 NLTK 数据（可选）

如需使用特定语言的句子分割功能，可下载 NLTK 数据：

```python
import nltk
nltk.download('punkt')
```

## 使用方法

### 基本示例

```bash
# 基本使用（自动检测输出目录）
python audio_diarization.py -i "path/to/audio.wav"

# 指定输出目录
python audio_diarization.py -i "path/to/audio.wav" -o "path/to/output"

# 使用 HuggingFace token
python audio_diarization.py -i "path/to/audio.wav" --token "hf_xxx"

# 指定 Whisper 模型
python audio_diarization.py -i "path/to/audio.wav" --model-name "medium"

# 指定说话人数量范围
python audio_diarization.py -i "path/to/audio.wav" --min-speakers 2 --max-speakers 4

# 使用镜像下载模型
python audio_diarization.py -i "path/to/audio.wav" --download-type token

# 指定模型缓存目录
python audio_diarization.py -i "path/to/audio.wav" --model-dir "path/to/models"

# 指定 NLTK 数据路径
python audio_diarization.py -i "path/to/audio.wav" --nltk-path "path/to/nltk_data"
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `--input`, `-i` | 选项 | 必填 | 输入音频文件路径 |
| `--output`, `-o` | 选项 | 音频同目录 | 输出目录 |
| `--download-type`, `-dt` | 选项 | `token` | 模型下载方式：`token` 或 `proxy` |
| `--token` | 选项 | 无 | HuggingFace token（用于下载模型） |
| `--model-dir`, `-md` | 选项 | 无 | 模型缓存目录 |
| `--model-name`, `-mn` | 选项 | `large-v3` | Whisper 模型名称 |
| `--min-speakers` | 选项 | 无 | 最小说话人数量 |
| `--max-speakers` | 选项 | 无 | 最大说话人数量 |
| `--nltk-path` | 选项 | 无 | NLTK 数据目录路径 |

## 输出文件

执行完成后，输出目录将生成以下文件：

| 文件名 | 说明 |
| ------ | ---- |
| `transcribe.json` | 初始转录结果（Whisper 输出） |
| `align.json` | 对齐后的时间戳结果 |
| `speakers.json` | 带说话人标签的分段结果 |
| `transcript.json` | 合并后的完整转录文本 |
| `SPEAKER/` | 按说话人拆分的音频文件目录 |

### 输出目录结构

```
output/
├── transcribe.json
├── align.json
├── speakers.json
├── transcript.json
└── SPEAKER/
    ├── SPEAKER_00.wav
    ├── SPEAKER_01.wav
    └── ...
```

## 高级功能

### 断点续传

工具会自动检测已生成的中间结果文件。如果某个步骤的 JSON 文件已存在，则跳过该步骤直接加载缓存结果。适用于：

- 网络中断后重新执行
- 调整参数后仅重新处理部分步骤
- 批量处理时避免重复计算

### 语言自适应句子合并

`merge_segments` 函数根据语言特性自动合并短分段：

- **韩语**：识别语尾（죠, 요, 다, 네요, 구나, 어요, 습니다, 까, 나요, 지）
- **通用**：识别句子结束标点（!\"').:;?]}~…）

### GPU 显存管理

`release_gpu_resources` 函数在每步处理后自动释放 GPU 显存，防止多模型加载时显存不足。

### 说话人音频导出

`generate_speaker_audio` 函数将每个说话人的所有片段拼接为独立 WAV 文件：

- 默认在每个片段前后增加 0.05 秒缓冲
- 输出至 `SPEAKER` 子目录
- 支持自定义采样率（默认 24000Hz）

### 支持的语言

WhisperX 支持 90+ 种语言，包括：

- 中文（zh）
- 英语（en）
- 韩语（kr）
- 日语（ja）
- 法语（fr）
- 德语（de）
- 西班牙语（es）
- 等...

语言由 Whisper 模型自动检测，无需手动指定。
