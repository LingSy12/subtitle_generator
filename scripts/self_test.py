from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ytsubtitle.gui import (
    build_transcribe_command,
    child_process_env,
    ensure_srt_suffix,
    estimate_time_left,
    format_time_left,
    normalize_output_formats,
    split_output_target,
)
from ytsubtitle.cli import configure_standard_streams
from ytsubtitle.cli import (
    DEFAULT_CUDA_MEDIA_CHUNK_SECONDS,
    LOW_VRAM_CHUNK_LENGTH,
    LOW_VRAM_COMPUTE_TYPE,
    TranscriptionAttempt,
    build_transcription_attempt,
    cpu_fallback_compute_type,
    effective_vad,
    effective_word_timestamps,
    enforce_monotonic_cue,
    finalize_output_files,
    is_cuda_out_of_memory_error,
    is_cuda_runtime_error,
    offset_segment_time,
    partial_output_base,
    should_retry_on_cpu,
    should_retry_on_low_vram_gpu,
)
from ytsubtitle.transcriber import configure_cuda_runtime_path, cuda_runtime_directories
from ytsubtitle.subtitles import (
    Cue,
    Segment,
    SubtitleOutputs,
    format_ass_time,
    format_lrc_time,
    format_sbv_time,
    format_srt_time,
    format_vtt_time,
    normalize_text,
    segment_to_cues,
    wrap_text,
)


def test_time_formatting() -> None:
    assert format_srt_time(3723.456) == "01:02:03,456"
    assert format_vtt_time(3723.456) == "01:02:03.456"
    assert format_sbv_time(3723.456) == "1:02:03.456"
    assert format_ass_time(3723.456) == "1:02:03.46"
    assert format_lrc_time(3723.456) == "62:03.46"


def test_chinese_normalization() -> None:
    assert (
        normalize_text("\u4f60 \u597d \uff0c \u4e16\u754c \uff01")
        == "\u4f60\u597d\uff0c\u4e16\u754c\uff01"
    )


def test_wrapping_and_cue_split() -> None:
    text = (
        "\u8fd9\u662f\u4e00\u4e2a\u5f88\u957f\u7684\u4e2d\u6587"
        "\u5b57\u5e55\u53e5\u5b50\uff0c\u9700\u8981\u81ea\u52a8"
        "\u6362\u884c\u5e76\u4e14\u5728\u5fc5\u8981\u7684\u65f6"
        "\u5019\u62c6\u6210\u591a\u4e2a\u5b57\u5e55\u5757\u3002"
    )
    lines = wrap_text(text, max_line_chars=10)
    assert all(len(line) <= 11 for line in lines)

    cues = segment_to_cues(
        Segment(start=1.0, end=9.0, text=text),
        max_line_chars=10,
        max_lines=2,
        min_cue_duration=0.8,
    )
    assert len(cues) >= 2
    assert cues[0].start == 1.0
    assert cues[-1].end == 9.0


def test_output_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "demo"
        formats = ["srt", "vtt", "sbv", "ass", "ttml", "lrc", "csv", "txt", "jsonl"]
        with SubtitleOutputs(base, formats) as outputs:
            outputs.write_cue(
                segment_to_cues(
                    Segment(start=0, end=2, text="\u4f60\u597d\uff0c\u4e16\u754c\uff01"),
                    max_line_chars=18,
                    max_lines=2,
                    min_cue_duration=0.8,
                )[0]
            )

        assert (base.with_suffix(".srt")).read_text(encoding="utf-8-sig").startswith("1")
        assert (base.with_suffix(".vtt")).read_text(encoding="utf-8").startswith("WEBVTT")
        assert "0:00:00.000,0:00:02.000" in (
            base.with_suffix(".sbv")
        ).read_text(encoding="utf-8")
        assert "[Events]" in (base.with_suffix(".ass")).read_text(encoding="utf-8-sig")
        assert "</tt>" in (base.with_suffix(".ttml")).read_text(encoding="utf-8")
        assert "[00:00.00]" in (base.with_suffix(".lrc")).read_text(encoding="utf-8-sig")
        assert "start,end,text" in (base.with_suffix(".csv")).read_text(encoding="utf-8-sig")
        assert "\u4f60\u597d" in (base.with_suffix(".txt")).read_text(encoding="utf-8")
        assert (base.with_suffix(".segments.jsonl")).exists()


def test_gui_command_builder() -> None:
    command = build_transcribe_command(
        python_exe="python",
        input_path="D:\\Videos\\demo.mp4",
        output_file="D:\\Subs\\custom_name.srt",
        model="medium",
        language="zh",
        script="simplified",
        device="cpu",
        compute_type="int8",
        vad=True,
        output_formats=["srt", "ass", "csv", "jsonl"],
        extract_audio=False,
        initial_prompt="Keep names.",
        low_vram=True,
        accurate_timing=True,
    )
    assert command[:3] == ["python", "-m", "ytsubtitle"]
    assert "--basename" in command
    assert "custom_name" in command
    assert "--progress-json" in command
    assert "--progress-every" in command
    assert "--vad" in command
    assert "--low-vram" in command
    assert "--accurate-timing" in command
    assert "--extract-audio" not in command
    assert "srt,ass,csv,jsonl" in command
    assert command[-2:] == ["--initial-prompt", "Keep names."]


def test_output_target_helpers() -> None:
    assert ensure_srt_suffix("D:\\Subs\\demo") == "D:\\Subs\\demo.srt"
    output_dir, basename, srt_path = split_output_target("D:\\Subs\\demo.txt")
    assert output_dir.endswith("\\Subs")
    assert basename == "demo"
    assert srt_path.name == "demo.srt"


def test_eta_helpers() -> None:
    assert estimate_time_left(25, 100, 10) == 30
    assert estimate_time_left(0, 100, 10) is None
    assert format_time_left(59.4) == "59s"
    assert format_time_left(65) == "1m 05s"
    assert format_time_left(3661) == "1h 01m"


def test_format_selection_helpers() -> None:
    assert normalize_output_formats(["srt", "SRT", "bad", "csv"]) == ["srt", "csv"]


def test_timing_helpers() -> None:
    args = type(
        "Args",
        (),
        {"vad": True, "word_timestamps": False, "accurate_timing": True},
    )()
    assert not effective_vad(args)
    assert not effective_word_timestamps(args)
    args.word_timestamps = True
    assert effective_word_timestamps(args)

    shifted = offset_segment_time(Segment(start=2, end=4, text="old"), -1.5, "new")
    assert shifted == Segment(start=0.5, end=2.5, text="new")
    assert offset_segment_time(Segment(start=0.2, end=0.4, text="old"), -1, "new") is None

    cue = enforce_monotonic_cue(Cue(start=1, end=2, text="hello"), 1.5, 0.8)
    assert cue == Cue(start=1.5, end=2.3, text="hello")

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "demo"
        partial_base = partial_output_base(base)
        partial_srt = partial_base.with_suffix(".srt")
        partial_jsonl = partial_base.with_suffix(".segments.jsonl")
        partial_srt.write_text("1\n", encoding="utf-8")
        partial_jsonl.write_text("{}\n", encoding="utf-8")
        written = finalize_output_files([partial_srt, partial_jsonl], partial_base, base)
        assert base.with_suffix(".srt") in written
        assert base.with_suffix(".segments.jsonl") in written
        assert not partial_srt.exists()


def test_encoding_helpers() -> None:
    configure_standard_streams()
    env = child_process_env()
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"


def test_cuda_fallback_helpers() -> None:
    error = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
    assert is_cuda_runtime_error(error)
    assert should_retry_on_cpu("cuda", error)
    assert not should_retry_on_cpu("cpu", error)
    assert cpu_fallback_compute_type("auto") == "int8"
    assert cpu_fallback_compute_type("float16") == "int8"
    assert cpu_fallback_compute_type("float32") == "float32"

    oom_error = RuntimeError("CUDA failed with error out of memory")
    current = TranscriptionAttempt(
        device="cuda",
        compute_type="auto",
        beam_size=5,
        chunk_length=None,
    )
    retry = TranscriptionAttempt(
        device="cuda",
        compute_type=LOW_VRAM_COMPUTE_TYPE,
        beam_size=1,
        chunk_length=LOW_VRAM_CHUNK_LENGTH,
        media_chunk_seconds=None,
        low_vram=True,
    )
    args = type(
        "Args",
        (),
        {
            "device": "cuda",
            "compute_type": "auto",
            "beam_size": 5,
            "chunk_length": None,
            "accurate_timing": False,
            "media_chunk_seconds": None,
        },
    )()
    built_retry = build_transcription_attempt(args, low_vram=True)
    assert built_retry == retry
    args.accurate_timing = True
    assert (
        build_transcription_attempt(args, low_vram=True).media_chunk_seconds
        == DEFAULT_CUDA_MEDIA_CHUNK_SECONDS
    )
    args.device = "cpu"
    assert build_transcription_attempt(args, low_vram=True) == TranscriptionAttempt(
        device="cpu",
        compute_type="auto",
        beam_size=5,
        chunk_length=None,
    )
    assert is_cuda_out_of_memory_error(oom_error)
    assert should_retry_on_low_vram_gpu("cuda", oom_error, current, retry)
    assert not should_retry_on_low_vram_gpu("cpu", oom_error, current, retry)


def test_cuda_runtime_path_helpers() -> None:
    directories = cuda_runtime_directories()
    if (PROJECT_ROOT / "vendor" / "cuda12" / "cublas64_12.dll").exists():
        assert PROJECT_ROOT / "vendor" / "cuda12" in directories
        added = configure_cuda_runtime_path()
        assert PROJECT_ROOT / "vendor" / "cuda12" in added


if __name__ == "__main__":
    test_time_formatting()
    test_chinese_normalization()
    test_wrapping_and_cue_split()
    test_output_files()
    test_gui_command_builder()
    test_output_target_helpers()
    test_eta_helpers()
    test_format_selection_helpers()
    test_timing_helpers()
    test_encoding_helpers()
    test_cuda_fallback_helpers()
    test_cuda_runtime_path_helpers()
    print("self-test passed")
