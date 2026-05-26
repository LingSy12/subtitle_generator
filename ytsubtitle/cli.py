from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .audio import extract_audio, ffmpeg_available
from .subtitles import (
    SubtitleOutputs,
    convert_chinese_script,
    normalize_text,
    segment_to_cues,
)
from .transcriber import (
    DEFAULT_INITIAL_PROMPT,
    supported_compute_types,
    transcribe_segments,
)


SUPPORTED_FORMATS = {"srt", "vtt", "sbv", "ass", "ttml", "lrc", "csv", "txt", "jsonl"}
LOW_VRAM_COMPUTE_TYPE = "int8"
LOW_VRAM_COMPUTE_PREFERENCES = (
    "int8",
    "int8_float16",
    "int8_float32",
    "float16",
    "float32",
)
LOW_VRAM_CHUNK_LENGTH = 15
LOW_VRAM_BEAM_SIZE = 1
DEFAULT_CUDA_MEDIA_CHUNK_SECONDS = 120
DEFAULT_MEDIA_CHUNK_OVERLAP_SECONDS = 2
# Drop segments Whisper itself is fairly sure are silence/non-speech. faster-whisper
# only filters when no_speech_prob > 0.6 AND avg_logprob < -1.0 — so confident
# hallucinations in silent regions slip through. We filter on no_speech_prob alone.
HALLUCINATION_NO_SPEECH_THRESHOLD = 0.6
# Beam search over a silent/music region can lock onto a high-frequency training
# artifact (a subtitle-credit watermark) and emit it across many segments in a
# row. faster-whisper's repeat suppression doesn't span segments, so we drop any
# run of this many or more consecutive segments with identical text. Real speech
# rarely repeats one phrase 5+ times back-to-back.
HALLUCINATION_REPEAT_LIMIT = 5


@dataclass(frozen=True)
class TranscriptionAttempt:
    device: str
    compute_type: str
    beam_size: int
    chunk_length: int | None
    media_chunk_seconds: int | None = None
    low_vram: bool = False


def main(argv: list[str] | None = None) -> int:
    configure_standard_streams()

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.beam_size < 1:
        parser.error("--beam-size must be 1 or greater.")
    if args.chunk_length is not None and args.chunk_length < 1:
        parser.error("--chunk-length must be 1 or greater.")
    if args.media_chunk_seconds is not None and args.media_chunk_seconds < 1:
        parser.error("--media-chunk-seconds must be 1 or greater.")
    if args.media_chunk_overlap_seconds < 0:
        parser.error("--media-chunk-overlap-seconds must be 0 or greater.")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")

    formats = parse_formats(args.formats)
    output_dir = Path(args.output_dir).expanduser().resolve()
    basename = args.basename or input_path.stem
    output_base = output_dir / basename

    media_path = input_path
    if args.extract_audio:
        audio_path = output_dir / "audio-cache" / f"{input_path.stem}.wav"
        print(f"Extracting audio to {audio_path} ...", flush=True)
        media_path = extract_audio(
            input_path=input_path,
            output_path=audio_path,
            ffmpeg_bin=args.ffmpeg,
            overwrite=args.overwrite_audio,
        )
    elif not ffmpeg_available(args.ffmpeg):
        print(
            "FFmpeg is not on PATH; using PyAV direct media decoding from faster-whisper.",
            flush=True,
        )

    initial_prompt = args.initial_prompt
    if initial_prompt is None and args.language.startswith("zh"):
        initial_prompt = DEFAULT_INITIAL_PROMPT

    attempt = build_transcription_attempt(args, low_vram=args.low_vram)
    try:
        written_paths, cue_count, segment_count = run_transcription_attempt(
            args=args,
            media_path=media_path,
            output_base=output_base,
            formats=formats,
            initial_prompt=initial_prompt,
            attempt=attempt,
        )
    except (RuntimeError, ValueError) as exc:
        low_vram_attempt = build_transcription_attempt(args, low_vram=True)
        exc_text = str(exc)
        if should_retry_on_low_vram_gpu(args.device, exc, attempt, low_vram_attempt):
            release_failed_attempt_resources(exc)
            print(
                "CUDA ran out of GPU memory "
                f"({exc_text}). Retrying on GPU in low-VRAM mode "
                f"(compute type {low_vram_attempt.compute_type}, "
                f"beam size {low_vram_attempt.beam_size}, "
                f"chunk length {low_vram_attempt.chunk_length}s)...",
                flush=True,
            )
            try:
                written_paths, cue_count, segment_count = run_transcription_attempt(
                    args=args,
                    media_path=media_path,
                    output_base=output_base,
                    formats=formats,
                    initial_prompt=initial_prompt,
                    attempt=low_vram_attempt,
                )
            except (RuntimeError, ValueError) as low_vram_exc:
                low_vram_exc_text = str(low_vram_exc)
                if not args.no_cpu_fallback and should_retry_on_cpu(args.device, low_vram_exc):
                    release_failed_attempt_resources(low_vram_exc)
                    fallback_compute_type = cpu_fallback_compute_type(args.compute_type)
                    print(
                        "Low-VRAM GPU retry still failed "
                        f"({low_vram_exc_text}). Retrying locally on CPU with compute type "
                        f"{fallback_compute_type}...",
                        flush=True,
                    )
                    try:
                        written_paths, cue_count, segment_count = run_transcription_attempt(
                            args=args,
                            media_path=media_path,
                            output_base=output_base,
                            formats=formats,
                            initial_prompt=initial_prompt,
                            attempt=TranscriptionAttempt(
                                device="cpu",
                                compute_type=fallback_compute_type,
                                beam_size=args.beam_size,
                                chunk_length=args.chunk_length,
                            ),
                        )
                    except (RuntimeError, ValueError) as cpu_exc:
                        report_failure(str(cpu_exc), output_base, args)
                        return 2
                else:
                    report_failure(low_vram_exc_text, output_base, args)
                    return 2
        elif not args.no_cpu_fallback and should_retry_on_cpu(args.device, exc):
            release_failed_attempt_resources(exc)
            fallback_compute_type = cpu_fallback_compute_type(args.compute_type)
            print(
                "GPU/CUDA is unavailable "
                f"({exc_text}). Retrying locally on CPU with compute type {fallback_compute_type}...",
                flush=True,
            )
            try:
                written_paths, cue_count, segment_count = run_transcription_attempt(
                    args=args,
                    media_path=media_path,
                    output_base=output_base,
                    formats=formats,
                    initial_prompt=initial_prompt,
                    attempt=TranscriptionAttempt(
                        device="cpu",
                        compute_type=fallback_compute_type,
                        beam_size=args.beam_size,
                        chunk_length=args.chunk_length,
                    ),
                )
            except (RuntimeError, ValueError) as cpu_exc:
                report_failure(str(cpu_exc), output_base, args)
                return 2
        else:
            report_failure(exc_text, output_base, args)
            return 2

    print("")
    print(f"Done. Generated {cue_count} subtitle cues from {segment_count} segments.")
    if args.progress_json:
        print_progress_event(
            "done",
            segment_count=segment_count,
            cue_count=cue_count,
        )
    for path in written_paths:
        print(f"- {path}")
    print("")
    print("Upload the .srt file to YouTube Studio as Chinese subtitles with timing.")
    return 0


def run_transcription_attempt(
    args: argparse.Namespace,
    media_path: Path,
    output_base: Path,
    formats: list[str],
    initial_prompt: str | None,
    attempt: TranscriptionAttempt,
) -> tuple[list[Path], int, int]:
    segments, info = transcribe_segments(
        media_path=media_path,
        model_name_or_path=args.model,
        language=args.language,
        task=args.task,
        device=attempt.device,
        compute_type=attempt.compute_type,
        beam_size=attempt.beam_size,
        vad=effective_vad(args),
        initial_prompt=initial_prompt,
        chunk_length=attempt.chunk_length,
        word_timestamps=effective_word_timestamps(args),
        media_chunk_seconds=attempt.media_chunk_seconds,
        media_chunk_overlap_seconds=args.media_chunk_overlap_seconds,
        allow_chunk_cpu_fallback=not args.no_cpu_fallback,
        start_seconds=args.start_seconds,
    )
    segments = drop_consecutive_repeats(segments, HALLUCINATION_REPEAT_LIMIT)

    print(f"Using device: {attempt.device} ({attempt.compute_type})", flush=True)
    if attempt.low_vram:
        print(
            "Low-VRAM GPU mode: "
            f"beam size {attempt.beam_size}, chunk length {attempt.chunk_length}s",
            flush=True,
        )
    elif attempt.chunk_length is not None:
        print(f"Chunk length: {attempt.chunk_length}s", flush=True)
    if attempt.media_chunk_seconds is not None:
        print(
            "Media chunking: "
            f"{attempt.media_chunk_seconds}s chunks, "
            f"{args.media_chunk_overlap_seconds}s overlap",
            flush=True,
        )
    if args.accurate_timing:
        print("Accurate timing mode: VAD disabled", flush=True)
    elif args.word_timestamps:
        print("Word timestamps enabled", flush=True)
    if args.time_offset:
        print(f"Subtitle time offset: {args.time_offset:+.3f}s", flush=True)
    print_model_info(info)
    duration = getattr(info, "duration", None)
    if args.progress_json and duration is not None:
        print_progress_event("duration", duration=float(duration))
    print(f"Writing subtitles to {output_base.parent} ...", flush=True)

    cue_count = 0
    segment_count = 0
    last_progress_at = -1.0
    progress_started_at = time.monotonic()
    temp_output_base = partial_output_base(output_base)
    archive_existing_partials(temp_output_base, formats)

    with SubtitleOutputs(output_base=temp_output_base, formats=formats) as outputs:
        last_cue_end = 0.0
        for segment in segments:
            segment_count += 1
            if segment.no_speech_prob >= HALLUCINATION_NO_SPEECH_THRESHOLD:
                continue
            text = normalize_text(segment.text)
            if args.script != "none":
                text = convert_chinese_script(text, args.script)

            word_converter = (
                (lambda t: convert_chinese_script(t, args.script))
                if args.script != "none"
                else None
            )
            segment = offset_segment_time(
                segment, args.time_offset, text, word_text_fn=word_converter
            )
            if segment is None:
                continue

            cues = segment_to_cues(
                segment,
                max_line_chars=args.max_line_chars,
                max_lines=args.max_lines,
                min_cue_duration=args.min_cue_duration,
                gap_split_seconds=args.cue_gap_seconds,
            )
            for cue in cues:
                cue = enforce_monotonic_cue(
                    cue,
                    previous_end=last_cue_end,
                    min_duration=args.min_cue_duration,
                )
                outputs.write_cue(cue)
                cue_count += 1
                last_cue_end = cue.end

            if segment.end - args.time_offset - last_progress_at >= args.progress_every:
                progress_seconds = segment.end - args.time_offset
                eta_seconds = estimate_time_left(
                    processed_seconds=max(0.0, float(progress_seconds) - args.start_seconds),
                    duration_seconds=max(
                        0.0,
                        (float(duration) if duration is not None else 0.0) - args.start_seconds,
                    ),
                    elapsed_seconds=time.monotonic() - progress_started_at,
                )
                eta_text = (
                    f", ETA {format_time_left(eta_seconds)} left"
                    if eta_seconds is not None
                    else ""
                )
                print(
                    f"Processed {format_duration(progress_seconds)} "
                    f"({segment_count} segments, {cue_count} cues{eta_text})",
                    flush=True,
                )
                if args.progress_json:
                    print_progress_event(
                        "progress",
                        seconds=float(progress_seconds),
                        segment_count=segment_count,
                        cue_count=cue_count,
                        eta_seconds=eta_seconds,
                    )
                last_progress_at = progress_seconds

        temp_paths = outputs.paths

    written_paths = finalize_output_files(temp_paths, temp_output_base, output_base)
    return written_paths, cue_count, segment_count


def drop_consecutive_repeats(segments, limit: int):
    """Yield segments, dropping runs of ``limit``+ consecutive identical texts.

    Beam search in silent/music regions can lock onto a training-data artifact
    (a subtitle-credit watermark) and repeat it across many segments. That
    cross-segment loop is a hallucination, not speech, so the whole run is
    dropped. Shorter repeats — a speaker actually saying a word twice — pass
    through untouched.
    """
    run: list = []
    for segment in segments:
        if run and segment.text.strip() == run[0].text.strip():
            run.append(segment)
            continue
        yield from _emit_repeat_run(run, limit)
        run = [segment]
    yield from _emit_repeat_run(run, limit)


def _emit_repeat_run(run: list, limit: int):
    if not run:
        return
    if len(run) >= limit:
        print(
            f"[hallucination] dropped {len(run)}x repeated "
            f"'{run[0].text.strip()}' "
            f"({format_duration(run[0].start)}-{format_duration(run[-1].end)})",
            flush=True,
        )
        return
    yield from run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ytsubtitle",
        description="Recognize speech from long videos and generate Chinese YouTube subtitles.",
    )
    parser.add_argument("input", help="Video or audio file to transcribe.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="subtitles",
        help="Directory for generated subtitle files. Default: subtitles",
    )
    parser.add_argument(
        "--basename",
        help="Output file basename. Default: input file name without extension.",
    )
    parser.add_argument(
        "--formats",
        default="srt,vtt,txt",
        help=(
            "Comma-separated formats: srt,vtt,sbv,ass,ttml,lrc,csv,txt,jsonl. "
            "Default: srt,vtt,txt"
        ),
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="faster-whisper model size/name or local model path. Default: medium",
    )
    parser.add_argument("--language", default="zh", help="Speech language. Default: zh")
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Use transcribe for Chinese subtitles, translate for English output.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for faster-whisper: auto, cpu, cuda. Default: auto",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help=(
            "Compute type: auto, int8, int8_float32, int8_float16, "
            "float16, float32. Default: auto"
        ),
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Decoding beam size. Larger can improve accuracy but slows down.",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        help=(
            "Whisper audio window length in seconds. Lower values use less GPU VRAM. "
            "Default: faster-whisper default"
        ),
    )
    parser.add_argument(
        "--media-chunk-seconds",
        type=int,
        help=(
            "Process long media in fixed-size chunks while preserving original "
            "timestamps. CUDA accurate-timing runs default to 120."
        ),
    )
    parser.add_argument(
        "--media-chunk-overlap-seconds",
        type=int,
        default=DEFAULT_MEDIA_CHUNK_OVERLAP_SECONDS,
        help="Overlap between media chunks in seconds. Default: 2",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help=(
            "Use safer CUDA settings for smaller GPUs: supported int8/quantized "
            "compute, beam size 1, and 15-second chunks."
        ),
    )
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Fail instead of retrying on CPU after CUDA errors.",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable voice activity detection to skip silence.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Use word-level timestamp alignment for tighter subtitle timings.",
    )
    parser.add_argument(
        "--accurate-timing",
        action="store_true",
        help=(
            "Prefer subtitle timing accuracy over speed by disabling VAD silence "
            "skipping."
        ),
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help=(
            "Shift all subtitle times by seconds. Positive delays subtitles; "
            "negative shows them earlier. Default: 0"
        ),
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help=(
            "Resume by skipping the first N seconds of media. Requires "
            "--media-chunk-seconds. Output timestamps stay absolute. Default: 0"
        ),
    )
    parser.add_argument(
        "--cue-gap-seconds",
        type=float,
        default=0.7,
        help=(
            "Split into a new cue whenever the gap between consecutive words "
            "exceeds this many seconds. Larger keeps long phrases together; "
            "smaller produces tighter, denser cues. Default: 0.7"
        ),
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional prompt to guide names, terminology, and Simplified Chinese style.",
    )
    parser.add_argument(
        "--script",
        choices=["none", "simplified", "traditional"],
        default="none",
        help="Convert Chinese subtitle script after recognition. Default: none",
    )
    parser.add_argument(
        "--max-line-chars",
        type=int,
        default=18,
        help="Maximum characters per subtitle line. Default: 18",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=2,
        help="Maximum lines per subtitle cue. Default: 2",
    )
    parser.add_argument(
        "--min-cue-duration",
        type=float,
        default=0.8,
        help="Minimum cue duration in seconds. Default: 0.8",
    )
    parser.add_argument(
        "--progress-every",
        type=float,
        default=60.0,
        help="Print progress every N seconds of media. Default: 60",
    )
    parser.add_argument(
        "--extract-audio",
        action="store_true",
        help="Use system FFmpeg to extract 16 kHz mono WAV before transcribing.",
    )
    parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Overwrite cached WAV when --extract-audio is used.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg executable path for --extract-audio. Default: ffmpeg",
    )
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def parse_formats(raw_formats: str) -> list[str]:
    formats = [part.strip().lower() for part in raw_formats.split(",") if part.strip()]
    unknown = sorted(set(formats) - SUPPORTED_FORMATS)
    if unknown:
        raise SystemExit(f"Unsupported format(s): {', '.join(unknown)}")
    if not formats:
        raise SystemExit("At least one output format is required.")
    return formats


def effective_vad(args: argparse.Namespace) -> bool:
    return bool(args.vad and not args.accurate_timing)


def effective_word_timestamps(args: argparse.Namespace) -> bool:
    return bool(args.word_timestamps)


def partial_output_base(output_base: Path) -> Path:
    return output_base.with_name(f"{output_base.name}__partial")


def archive_existing_partials(temp_output_base: Path, formats: list[str]) -> list[Path]:
    """Rename existing __partial.<ext> files so a new run cannot overwrite them.

    Returns the list of archived paths (empty if none existed).
    """
    parent = temp_output_base.parent
    if not parent.exists():
        return []
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    archived: list[Path] = []
    for old in parent.glob(f"{temp_output_base.name}.*"):
        if not old.is_file():
            continue
        renamed = old.with_name(
            old.name.replace(
                temp_output_base.name,
                f"{temp_output_base.name}-{timestamp}",
                1,
            )
        )
        try:
            old.rename(renamed)
            archived.append(renamed)
        except OSError:
            continue
    if archived:
        print(
            f"Archived previous partial files: {', '.join(p.name for p in archived)}",
            flush=True,
        )
    return archived


def existing_partial_paths(temp_output_base: Path) -> list[Path]:
    parent = temp_output_base.parent
    if not parent.exists():
        return []
    return [p for p in parent.glob(f"{temp_output_base.name}.*") if p.is_file()]


def report_failure(
    message: str,
    output_base: Path,
    args: argparse.Namespace | None = None,
) -> None:
    print(f"Error: {message}", file=sys.stderr)

    if args is not None:
        suggestions = diagnose_failure_flags(message, args)
        if suggestions:
            print("", file=sys.stderr)
            print("Likely fix:", file=sys.stderr)
            for line in suggestions:
                print(f"  - {line}", file=sys.stderr)

    partial_base = partial_output_base(output_base)
    partials = existing_partial_paths(partial_base)
    if not partials:
        return
    last_end = read_last_cue_end_seconds(partial_base)
    print("", file=sys.stderr)
    print("Partial subtitles saved (kept on disk):", file=sys.stderr)
    for path in partials:
        print(f"  {path}", file=sys.stderr)
    if last_end is not None:
        print(
            f"\nTo resume, run again with --start-seconds {last_end:.2f} "
            "and the same --media-chunk-seconds value, then merge with the partial "
            "files above.",
            file=sys.stderr,
        )


def diagnose_failure_flags(message: str, args: argparse.Namespace) -> list[str]:
    """Return human-readable suggestions for which CLI flags likely caused the failure."""
    suggestions: list[str] = []
    is_gpu_crash = "killed by the OS" in message or "GPU process crash" in message
    if is_gpu_crash and getattr(args, "no_cpu_fallback", False):
        suggestions.append(
            "Remove --no-cpu-fallback (or uncheck 'GPU only' in the GUI). "
            "Your GPU driver crashed; CPU rescue can finish the stuck chunks."
        )
    if getattr(args, "accurate_timing", False):
        suggestions.append(
            "Remove --accurate-timing. It disables VAD, which lets Whisper "
            "hallucinate text in silent regions with bad timestamps. "
            "Use --vad --word-timestamps instead for accurate alignment."
        )
    return suggestions


def read_last_cue_end_seconds(temp_output_base: Path) -> float | None:
    """Read the end time of the last cue from the partial SRT, if available."""
    srt_path = temp_output_base.with_suffix(".srt")
    if not srt_path.exists():
        return None
    last_end: float | None = None
    try:
        with srt_path.open("r", encoding="utf-8-sig", errors="replace") as fh:
            for line in fh:
                if "-->" not in line:
                    continue
                _, _, right = line.partition("-->")
                end_text = right.strip().split(" ", 1)[0]
                parsed = parse_srt_timestamp(end_text)
                if parsed is not None:
                    last_end = parsed
    except OSError:
        return None
    return last_end


def parse_srt_timestamp(text: str) -> float | None:
    text = text.strip().replace(",", ".")
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def finalize_output_files(
    temp_paths: list[Path],
    temp_output_base: Path,
    output_base: Path,
) -> list[Path]:
    written_paths: list[Path] = []
    temp_name = temp_output_base.name
    final_name = output_base.name
    for temp_path in temp_paths:
        if not temp_path.exists():
            continue
        final_path = temp_path.with_name(
            temp_path.name.replace(temp_name, final_name, 1)
        )
        temp_path.replace(final_path)
        written_paths.append(final_path)
    return written_paths


def offset_segment_time(segment, offset: float, text: str, word_text_fn=None):
    start = float(getattr(segment, "start")) + offset
    end = float(getattr(segment, "end")) + offset
    if end <= 0:
        return None
    words = getattr(segment, "words", ()) or ()
    shifted_words = tuple(
        type(word)(
            start=max(0.0, float(word.start) + offset),
            end=max(0.0, float(word.end) + offset),
            text=word_text_fn(word.text) if word_text_fn else word.text,
        )
        for word in words
    )
    return type(segment)(
        start=max(0.0, start),
        end=max(0.0, end),
        text=text,
        words=shifted_words,
    )


def enforce_monotonic_cue(cue: object, previous_end: float, min_duration: float):
    start = max(float(getattr(cue, "start")), previous_end)
    end = max(float(getattr(cue, "end")), start + min_duration)
    return type(cue)(start=start, end=end, text=str(getattr(cue, "text")))


def build_transcription_attempt(
    args: argparse.Namespace,
    low_vram: bool,
) -> TranscriptionAttempt:
    if not low_vram or args.device.lower() == "cpu":
        return TranscriptionAttempt(
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            chunk_length=args.chunk_length,
            media_chunk_seconds=effective_media_chunk_seconds(args),
        )

    return TranscriptionAttempt(
        device=args.device,
        compute_type=low_vram_compute_type(args.compute_type, args.device),
        beam_size=min(args.beam_size, LOW_VRAM_BEAM_SIZE),
        chunk_length=low_vram_chunk_length(args.chunk_length),
        media_chunk_seconds=effective_media_chunk_seconds(args),
        low_vram=True,
    )


def low_vram_compute_type(compute_type: str, device: str) -> str:
    compute_type = compute_type.lower()
    supported = supported_gpu_compute_types(device)
    if compute_type in LOW_VRAM_COMPUTE_PREFERENCES and (
        not supported or compute_type in supported
    ):
        return compute_type

    for candidate in LOW_VRAM_COMPUTE_PREFERENCES:
        if not supported or candidate in supported:
            return candidate

    return LOW_VRAM_COMPUTE_TYPE


def supported_gpu_compute_types(device: str) -> set[str]:
    if device.lower() == "cpu":
        return set()
    return supported_compute_types("cuda")


def low_vram_chunk_length(chunk_length: int | None) -> int:
    if chunk_length is None:
        return LOW_VRAM_CHUNK_LENGTH
    return min(chunk_length, LOW_VRAM_CHUNK_LENGTH)


def effective_media_chunk_seconds(args: argparse.Namespace) -> int | None:
    if args.media_chunk_seconds is not None:
        return args.media_chunk_seconds
    if args.accurate_timing and args.device.lower() != "cpu":
        return DEFAULT_CUDA_MEDIA_CHUNK_SECONDS
    return None


def should_retry_on_low_vram_gpu(
    device: str,
    exc: RuntimeError,
    current: TranscriptionAttempt,
    retry: TranscriptionAttempt,
) -> bool:
    if device.lower() == "cpu" or current.low_vram:
        return False
    if not is_cuda_out_of_memory_error(exc):
        return False
    return (
        retry.compute_type != current.compute_type
        or retry.beam_size < current.beam_size
        or (
            retry.chunk_length is not None
            and (
                current.chunk_length is None
                or retry.chunk_length < current.chunk_length
            )
        )
    )


def release_failed_attempt_resources(exc: BaseException) -> None:
    try:
        exc.__traceback__ = None
    except (AttributeError, TypeError):
        pass
    gc.collect()


def should_retry_on_cpu(device: str, exc: RuntimeError) -> bool:
    if device.lower() == "cpu":
        return False
    return is_cuda_runtime_error(exc)


def is_cuda_out_of_memory_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    oom_markers = [
        "out of memory",
        "cuda_error_out_of_memory",
        "cublas_status_alloc_failed",
        "failed to allocate",
        "memory allocation",
    ]
    return is_cuda_runtime_error(exc) and any(marker in message for marker in oom_markers)


def is_cuda_runtime_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    cuda_markers = [
        "cublas",
        "cudnn",
        "cuda",
        "cufft",
        "curand",
        "cublas64",
        "cudnn64",
        "cudart64",
        "is not found or cannot be loaded",
        "no cuda",
        "cuda driver",
    ]
    return any(marker in message for marker in cuda_markers)


def cpu_fallback_compute_type(compute_type: str) -> str:
    if compute_type in {"int8", "float32"}:
        return compute_type
    return "int8"


def print_model_info(info: object) -> None:
    language = getattr(info, "language", None)
    language_probability = getattr(info, "language_probability", None)
    duration = getattr(info, "duration", None)

    if duration is not None:
        print(f"Detected media duration: {format_duration(float(duration))}", flush=True)
    if language is not None:
        if language_probability is not None:
            print(
                f"Detected language: {language} ({float(language_probability):.1%})",
                flush=True,
            )
        else:
            print(f"Detected language: {language}", flush=True)


def configure_standard_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours}h {minutes:02}m {secs:02}s"
    return f"{minutes}m {secs:02}s"


def estimate_time_left(
    processed_seconds: float,
    duration_seconds: float,
    elapsed_seconds: float,
) -> float | None:
    if processed_seconds <= 0 or duration_seconds <= 0 or elapsed_seconds <= 0:
        return None
    remaining_media_seconds = max(0.0, duration_seconds - processed_seconds)
    seconds_per_media_second = elapsed_seconds / processed_seconds
    return remaining_media_seconds * seconds_per_media_second


def format_time_left(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours}h {minutes:02}m"
    if minutes:
        return f"{minutes}m {secs:02}s"
    return f"{secs}s"


def print_progress_event(event: str, **payload: object) -> None:
    payload = {"event": event, **payload}
    print(
        "__YTSUBTITLE_PROGRESS__" + json.dumps(payload, ensure_ascii=True),
        flush=True,
    )
