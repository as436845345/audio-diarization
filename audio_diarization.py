import json  # 用于读写 JSON 文件
import os  # 文件路径处理

import typer  # CLI 框架
from typing_extensions import Annotated  # 用于类型标注

import file_operation as fo


def get_device():
    """
    检测当前环境是否支持 CUDA GPU。

    Returns
    -------
    str
        如果 CUDA 可用返回 'cuda'，否则返回 'cpu'
    """
    from torch import cuda  # 导入 torch.cuda

    return "cuda" if cuda.is_available() else "cpu"  # 根据 CUDA 是否可用返回设备类型


def release_gpu_resources():
    """
    释放 GPU 显存。

    用于删除模型后清理显存，
    防止后续模型加载失败。
    """
    import gc  # Python 垃圾回收
    from torch import cuda  # CUDA 控制

    gc.collect()  # 手动触发垃圾回收
    cuda.empty_cache()  # 清空 CUDA cache


def merge_segments(
    transcript,
    ending_punct="!\"').:;?]}~…",
    sentence_endings: list[str] | None = None,
):
    """
    合并 WhisperX 分段文本。

    WhisperX 的分段通常较短，
    此函数根据句子结束标点或韩语语尾合并为完整句子。

    Parameters
    ----------
    transcript : list
        WhisperX 生成的分段
    ending_punct : str
        句子结束标点
    sentence_endings : list[str] | None
        句尾列表

    Returns
    -------
    list
        合并后的 transcript
    """

    merged = []  # 存储合并后的结果
    buffer = None  # 当前正在拼接的句子

    # 遍历所有分段
    for seg in transcript:
        text = seg["text"].strip()  # 去掉首尾空格

        # 空文本直接跳过
        if not text:
            continue

        # 如果 buffer 为空说明是新的句子
        if buffer is None:
            buffer = seg.copy()  # 复制 segment
            buffer["text"] = text
        else:
            # 获取 buffer 的文本
            b_text = buffer["text"].rstrip()

            # 取最后一个字符
            last_char = b_text[-1]

            # 判断是否句子结束
            is_ending = (last_char in ending_punct) or b_text.endswith(
                tuple(sentence_endings or ())
            )

            # 如果是句子结束
            if is_ending:
                merged.append(buffer)  # 保存当前句子
                buffer = seg.copy()  # 新句子
                buffer["text"] = text
            else:
                # 否则继续拼接
                buffer["text"] = b_text + " " + text.lstrip()
                buffer["end"] = seg["end"]

    # 最后一个 buffer 也加入结果
    if buffer:
        merged.append(buffer)

    return merged


def generate_speaker_audio(
    wav_path: str,
    transcript: list[dict],
    output_dir: str,
    delay: float = 0.05,
    sample_rate: int = 24000,
    output_folder_name: str = "SPEAKER",
) -> dict[str, str]:
    """
    根据说话人拆分音频。

    将 transcript 中每个 speaker 的音频
    拼接为一个单独 WAV 文件。

    Parameters
    ----------
    wav_path : str
        输入音频路径
    transcript : list[dict]
        带 speaker 的分段
    output_dir : str
        输出目录
    delay : float
        每个分段前后增加的时间
    sample_rate : int
        音频采样率
    output_folder_name : str
        输出文件夹名称

    Returns
    -------
    dict[str, str]
        speaker -> wav 文件路径
    """

    import librosa
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm

    # 加载音频
    audio_data, _ = librosa.load(wav_path, sr=sample_rate)

    # 音频长度
    length = len(audio_data)

    # 每个 speaker 的音频块
    speaker_segments: dict[str, list[np.ndarray]] = {}

    # 遍历所有分段
    for segment in tqdm(transcript, desc="Processing segments"):
        speaker = segment.get("speaker", "UNKNOWN")
        start_time = segment["start"]
        end_time = segment["end"]

        # 检查时间是否合法
        if start_time >= end_time or start_time < 0:
            print(f"Skip invalid segment: {speaker} [{start_time}, {end_time}]")
            continue

        # 计算样本索引
        start_idx = max(0, int((start_time - delay) * sample_rate))
        end_idx = min(int((end_time + delay) * sample_rate), length)

        if start_idx >= end_idx:
            continue

        # 截取音频
        chunk = audio_data[start_idx:end_idx].copy()

        # 添加到 speaker
        speaker_segments.setdefault(speaker, []).append(chunk)

    # speaker 输出目录
    speaker_folder = os.path.join(output_dir, output_folder_name)
    os.makedirs(speaker_folder, exist_ok=True)

    output_files = {}

    # 保存每个 speaker
    for speaker, chunks in tqdm(speaker_segments.items(), desc="Saving audio"):
        if not chunks:
            continue

        # 拼接音频
        speaker_audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        output_path = os.path.join(speaker_folder, f"{speaker}.wav")

        try:
            sf.write(str(output_path), speaker_audio, sample_rate)

            output_files[speaker] = str(output_path)

            duration = len(speaker_audio) / sample_rate
            print(f"✓ {speaker}: {os.path.basename(output_path)} ({duration:.1f}s)")
        except Exception as e:
            print(f"✗ Save failed {output_path}: {e}")
            raise typer.Exit(1)

    print(f"Finished: generated {len(output_files)} speaker files")

    return output_files


LANGUAGE_SENTENCE_ENDINGS = {
    "kr": [
        "죠",
        "요",
        "다",
        "네요",
        "구나",
        "어요",
        "습니다",
        "까",
        "나요",
        "지",
    ]
}


def main(
    wav_path: Annotated[str, typer.Option("--input", "-i", help="Input audio file")],
    output_dir: Annotated[
        str | None,
        typer.Option(
            "--output", "-o", help="Output directory (default: same as audio)"
        ),
    ] = None,
    download_mode: Annotated[
        str,
        typer.Option(
            "--download-mode",
            "-dm",
            help="Model download mode: token or proxy",
        ),
    ] = "token",
    token: Annotated[
        str | None,
        typer.Option("--token", help="HuggingFace token"),
    ] = None,
    model_dir: Annotated[
        str | None,
        typer.Option("--model-dir", "-md", help="Model cache directory"),
    ] = None,
    model_name: Annotated[
        str,
        typer.Option("--model-name", "-mn", help="Whisper model name"),
    ] = "large-v3",
    min_speakers: Annotated[
        int | None,
        typer.Option("--min-speakers", help="Minimum number of speakers"),
    ] = None,
    max_speakers: Annotated[
        int | None,
        typer.Option("--max-speakers", help="Maximum number of speakers"),
    ] = None,
    nltk_data_path: Annotated[
        str | None,
        typer.Option("--nltk-path", help="NLTK data directory"),
    ] = None,
):
    """
    WhisperX 音频转录 + 对齐 + 说话人分离 CLI 工具。
    """

    # 使用 HuggingFace 镜像
    if download_mode == "token":
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        if not token:
            print("下载模式为 `token` 时必须通过 token！")
            raise typer.Exit(1)

    # 设置 NLTK 路径
    if nltk_data_path is not None:
        import nltk

        nltk.data.path.append(nltk_data_path)

    import whisperx
    from whisperx.diarize import DiarizationPipeline

    device = get_device()

    if output_dir is None:
        output_dir = os.path.dirname(wav_path)

    transcribe_json_path = os.path.join(output_dir, "transcribe.json")
    align_json_path = os.path.join(output_dir, "align.json")
    speakers_json_path = os.path.join(output_dir, "speakers.json")
    transcript_json_path = os.path.join(output_dir, "transcript.json")

    print(f"[INFO] device: {device}")

    print("[INFO] transcribing...")

    transcribe_result = fo.load_from_json(transcribe_json_path)

    if transcribe_result is None:
        model = whisperx.load_model(
            model_name,
            device,
            download_root=model_dir,
            use_auth_token=token,
        )

        audio = whisperx.load_audio(wav_path)

        transcribe_result = model.transcribe(audio, batch_size=8)

        fo.save_to_json(transcribe_json_path, transcribe_result)

    language = transcribe_result["language"]

    print("[INFO] aligning...")

    align_result = fo.load_from_json(align_json_path)

    if align_result is None:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_dir=model_dir,
        )

        align_result = whisperx.align(
            transcribe_result["segments"],
            align_model,
            align_metadata,
            wav_path,
            device,
            return_char_alignments=False,
        )

        del align_model
        release_gpu_resources()

        fo.save_to_json(align_json_path, align_result)

    print("[INFO] diarization...")

    speakers_result = fo.load_from_json(speakers_json_path)

    if speakers_result is None:
        diarize_model = DiarizationPipeline(
            token=token,
            device=device,
            cache_dir=model_dir,
        )

        diarize_segments = diarize_model(
            wav_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        speakers_result = whisperx.assign_word_speakers(
            diarize_segments,
            align_result,
        )

        fo.save_to_json(speakers_json_path, speakers_result)

    print("[INFO] transcript...")

    transcript = fo.load_from_json(transcript_json_path)

    if transcript is None:
        transcript = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "speaker": seg.get("speaker", "SPEAKER_00"),
            }
            for seg in speakers_result["segments"]
        ]

        sentence_endings = LANGUAGE_SENTENCE_ENDINGS.get(language)

        transcript = merge_segments(
            transcript,
            sentence_endings=sentence_endings,
        )

        fo.save_to_json(transcript_json_path, transcript)

    print("[INFO] generating speaker audio...")

    generate_speaker_audio(wav_path, transcript, output_dir)  # type: ignore


if __name__ == "__main__":
    typer.run(main)
