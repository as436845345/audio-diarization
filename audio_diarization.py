import json
import os

import typer
from typing_extensions import Annotated


def get_device():
    from torch import cuda

    return "cuda" if cuda.is_available() else "cpu"


def release_gpu_resources():
    import gc
    from torch import cuda

    gc.collect()
    cuda.empty_cache()


def load_from_json(path: str) -> dict | None:
    if os.path.isfile(path):
        print("[INFO] load from file...")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def save_to_json(path: str, data):
    # TODO: 不确定是写在这里还是一开始创建好
    os.makedirs(path, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_segments(
    transcript,
    ending_punct="!\"').:;?]}~…",
    ending_eojil: list[str] | None = None,
):

    merged = []
    buffer = None

    for seg in transcript:
        text = seg["text"].strip()
        if not text:
            continue

        if buffer is None:
            buffer = seg.copy()
            buffer["text"] = text
        else:
            # 判断：标点结尾 或 韩语语尾结尾
            b_text = buffer["text"].rstrip()
            last_char = b_text[-1]

            is_ending = (last_char in ending_punct) or b_text.endswith(
                tuple(ending_eojil or ())
            )

            if is_ending:
                merged.append(buffer)
                buffer = seg.copy()
                buffer["text"] = text
            else:
                # 合并时注意韩语空格规则
                buffer["text"] = b_text + " " + text.lstrip()
                buffer["end"] = seg["end"]

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
    import librosa
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm

    # 加载音频
    audio_data, _ = librosa.load(wav_path, sr=sample_rate)
    length = len(audio_data)

    # 按说话人收集片段（避免重复拼接）
    speaker_segments: dict[str, list[np.ndarray]] = {}

    for segment in tqdm(transcript, desc="处理分段"):
        speaker = segment.get("speaker", "UNKNOWN")
        start_time = segment["start"]
        end_time = segment["end"]

        # 验证时间戳
        if start_time >= end_time or start_time < 0:
            print(f"跳过无效时间段: {speaker} [{start_time}, {end_time}]")
            continue

        # 计算带 delay 的索引
        start_idx = max(0, int((start_time - delay) * sample_rate))
        end_idx = min(int((end_time + delay) * sample_rate), length)

        if start_idx >= end_idx:
            continue

        # 截取并收集
        chunk = audio_data[start_idx:end_idx].copy()
        speaker_segments.setdefault(speaker, []).append(chunk)

    speaker_folder = os.path.join(output_dir, output_folder_name)
    os.makedirs(speaker_folder, exist_ok=True)

    # 拼接并保存
    output_files = {}
    for speaker, chunks in tqdm(speaker_segments.items(), desc="保存音频"):
        if not chunks:
            continue

        # 一次性拼接
        speaker_audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        output_path = os.path.join(speaker_folder, f"{speaker}.wav")
        try:
            sf.write(str(output_path), speaker_audio, sample_rate)
            output_files[speaker] = str(output_path)
            duration = len(speaker_audio) / sample_rate
            print(f"✓ {speaker}: {os.path.basename(output_path)} ({duration:.1f}秒)")
        except Exception as e:
            print(f"✗ 保存失败 {output_path}: {e}")
            exit(1)

    print(f"完成: 生成 {len(output_files)} 个说话人音频")
    return output_files


__transcript_eojils = {
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
    wav_path: Annotated[str, typer.Option("--input", "-i", help="")],
    output_dir: Annotated[str | None, typer.Option("--outpub", "-o", help="")] = None,
    download_type: Annotated[
        str, typer.Option("--download-type", "-dt", help="token/proxy")
    ] = "token",
    token: Annotated[str | None, typer.Option("--token", help="")] = None,
    model_dir: Annotated[
        str | None, typer.Option("--model-dir", "-md", help="")
    ] = None,
    model_name: Annotated[
        str | None, typer.Option("--model-name", "-mn", help="")
    ] = "large-v3",
    min_speakers: Annotated[
        int | None, typer.Option("--min-speakers", "-mis", help="")
    ] = None,
    max_speakers: Annotated[
        int | None, typer.Option("--max-speakers", "-mas", help="")
    ] = None,
    nltk_data_path: Annotated[str | None, typer.Option("--nltk-path", help="")] = None,
):
    if download_type == "token":
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # token = "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"

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

    print(f"[INFO] use device: {device}")

    print("[INFO] transcribe...")

    # 1. Transcribe with original whisper (batched)
    transcribe_result = load_from_json(transcribe_json_path)

    if transcribe_result is None:
        print("[INFO] load model...")

        # use_auth_token: https://github.com/m-bain/whisperX/blob/main/whisperx/asr.py#L361
        model = whisperx.load_model(
            model_name, device, download_root=model_dir, use_auth_token=token
        )

        audio = whisperx.load_audio(wav_path)
        transcribe_result = model.transcribe(audio, batch_size=8)

        print("[INFO] save transcribe to json file...")

        save_to_json(transcribe_json_path, transcribe_result)

    language = transcribe_result["language"]
    if language == "nn":
        print(f"No language detected in {wav_path}")
        exit(1)

    print(f"[SUCCESS] transcribe successful!")

    print("[INFO] align...")

    # 2. Align whisper output
    align_result = load_from_json(align_json_path)

    if align_result is None:
        print("[INFO] load align model...")

        align_model, align_metadata = whisperx.load_align_model(
            language_code=language, device=device, model_dir=model_dir
        )

        align_result = whisperx.align(
            transcribe_result["segments"],
            align_model,
            align_metadata,
            wav_path,
            device,
            return_char_alignments=False,
        )

        # delete model if low on GPU resources
        del align_model
        release_gpu_resources()

        print("[INFO] save align to json file...")

        save_to_json(align_json_path, align_result)

    # 3. Assign speaker labels
    speakers_result = load_from_json(speakers_json_path)

    if speakers_result is None:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L832
        # 类似的函数都是用 `cache_dir`
        diarize_model = DiarizationPipeline(
            token=token, device=device, cache_dir=model_dir
        )

        # add min/max number of speakers if known
        diarize_segments = diarize_model(
            wav_path, min_speakers=min_speakers, max_speakers=max_speakers
        )

        # 为每个词分配说话人
        speakers_result = whisperx.assign_word_speakers(diarize_segments, align_result)

        print("[INFO] save speakers to json file...")

        save_to_json(speakers_json_path, speakers_result)

    print("[INFO] transcript...")

    transcript = load_from_json(transcript_json_path)

    if transcript is None:
        # 转换为简化结构
        transcript = [
            {
                "start": segement["start"],
                "end": segement["end"],
                "text": segement["text"].strip(),
                # "speaker": segement.get("speaker", "SPEAKER_00"),
                "speaker": "SPEAKER_00",
            }
            for segement in speakers_result["segments"]
        ]

        merge_eojil = __transcript_eojils.get(language, None)

        transcript = merge_segments(transcript, ending_eojil=merge_eojil)
        save_to_json(transcript_json_path, transcript)

    print("[INFO] generate speaker audio...")

    generate_speaker_audio(wav_audio_path, transcript, output_dir)  # type: ignore


if __name__ == "__main__":
    typer.run(main)
