from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .transcriber import configure_cuda_runtime_path, decode_audio_slice


def main(argv: list[str] | None = None) -> int:
    configure_standard_streams()
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_cuda_runtime_path()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        print(f"Missing dependency: faster-whisper ({exc})", file=sys.stderr)
        return 2

    if args.serve:
        return run_serve(args, WhisperModel)
    return run_one_shot(args, WhisperModel)


def run_one_shot(args: argparse.Namespace, WhisperModel) -> int:
    audio = decode_audio_slice(
        input_file=str(Path(args.input).expanduser().resolve()),
        start_seconds=args.start,
        duration_seconds=args.duration,
        sampling_rate=16000,
    )
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    segments, _info = model.transcribe(
        audio,
        language=args.language or None,
        task=args.task,
        beam_size=args.beam_size,
        vad_filter=args.vad,
        initial_prompt=args.initial_prompt,
        condition_on_previous_text=True,
        chunk_length=args.chunk_length,
        word_timestamps=args.word_timestamps,
    )
    for segment in segments:
        print(
            json.dumps(
                _segment_payload(segment),
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


def run_serve(args: argparse.Namespace, WhisperModel) -> int:
    """Persistent worker mode: load model once, process requests from stdin.

    Protocol (line-delimited JSON, one message per line):
        ready:      {"type": "ready"}
        request:    {"action": "transcribe", "input": "...", "start": 0.0, "duration": 60.0,
                     "vad": false}
        request:    {"action": "shutdown"}
        segment:    {"type": "segment", "start": 0.5, "end": 2.3, "text": "..."}
        done:       {"type": "done"}
        error:      {"type": "error", "message": "..."}
    """
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    emit({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"type": "error", "message": f"bad JSON: {exc}"})
            continue

        action = request.get("action")
        if action == "shutdown":
            break
        if action != "transcribe":
            emit({"type": "error", "message": f"unknown action: {action!r}"})
            continue

        try:
            audio = decode_audio_slice(
                input_file=str(Path(request["input"]).expanduser().resolve()),
                start_seconds=float(request["start"]),
                duration_seconds=float(request["duration"]),
                sampling_rate=16000,
            )
            segments, _info = model.transcribe(
                audio,
                language=args.language or None,
                task=args.task,
                beam_size=args.beam_size,
                vad_filter=bool(request.get("vad", args.vad)),
                initial_prompt=args.initial_prompt,
                condition_on_previous_text=True,
                chunk_length=args.chunk_length,
                word_timestamps=args.word_timestamps,
            )
            for segment in segments:
                emit({"type": "segment", **_segment_payload(segment)})
            emit({"type": "done"})
        except Exception as exc:
            emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return 0


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _segment_payload(segment) -> dict:
    payload: dict = {
        "start": float(segment.start),
        "end": float(segment.end),
        "text": str(segment.text),
        "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
    }
    raw_words = getattr(segment, "words", None) or ()
    words: list[dict] = []
    for w in raw_words:
        try:
            wtext = str(getattr(w, "word", getattr(w, "text", ""))).strip()
            if not wtext:
                continue
            words.append(
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "text": wtext,
                }
            )
        except (AttributeError, TypeError, ValueError):
            continue
    if words:
        payload["words"] = words
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ytsubtitle.chunk_worker",
        description="Internal worker for transcribing one media chunk.",
    )
    parser.add_argument("input", nargs="?", default="")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--initial-prompt")
    parser.add_argument("--chunk-length", type=int)
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Persistent mode: load model once and read transcribe requests from stdin.",
    )
    return parser


def configure_standard_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    if exit_code == 0:
        os._exit(0)
    raise SystemExit(exit_code)
