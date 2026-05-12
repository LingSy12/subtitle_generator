from __future__ import annotations

import os
import gc
import io
import json
import subprocess
import sys
import tempfile
import wave
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .subtitles import Segment, Word


DEFAULT_INITIAL_PROMPT = (
    "\u4ee5\u4e0b\u662f\u4e2d\u6587\u89c6\u9891\u5185\u5bb9\u3002"
    "\u8bf7\u751f\u6210\u7b80\u4f53\u4e2d\u6587\u5b57\u5e55\uff0c"
    "\u6807\u70b9\u81ea\u7136\uff0c\u4fdd\u7559\u4eba\u540d\u3001"
    "\u5730\u540d\u3001\u54c1\u724c\u540d\u548c\u4e13\u4e1a\u672f\u8bed\u3002"
)

_DLL_DIRECTORY_HANDLES: list[object] = []


def transcribe_segments(
    media_path: Path,
    model_name_or_path: str,
    language: str,
    task: str,
    device: str,
    compute_type: str,
    beam_size: int,
    vad: bool,
    initial_prompt: str | None,
    chunk_length: int | None = None,
    word_timestamps: bool = False,
    media_chunk_seconds: int | None = None,
    media_chunk_overlap_seconds: int = 2,
    allow_chunk_cpu_fallback: bool = True,
    start_seconds: float = 0.0,
) -> tuple[Iterator[Segment], Any]:
    configure_cuda_runtime_path()

    if media_chunk_seconds is not None and media_chunk_seconds > 0:
        return transcribe_media_chunks(
            media_path=media_path,
            model_name_or_path=model_name_or_path,
            language=language,
            task=task,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad=vad,
            initial_prompt=initial_prompt,
            chunk_length=chunk_length,
            word_timestamps=word_timestamps,
            media_chunk_seconds=media_chunk_seconds,
            media_chunk_overlap_seconds=media_chunk_overlap_seconds,
            allow_chunk_cpu_fallback=allow_chunk_cpu_fallback,
            start_seconds=start_seconds,
        )
    if start_seconds and start_seconds > 0:
        raise ValueError(
            "--start-seconds requires media chunking; pass --media-chunk-seconds N too."
        )

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: faster-whisper. Run setup.ps1 before transcribing."
        ) from exc

    model = WhisperModel(
        model_name_or_path,
        device=device,
        compute_type=compute_type,
    )
    segments, info = model.transcribe(
        str(media_path),
        language=language or None,
        task=task,
        beam_size=beam_size,
        vad_filter=vad,
        initial_prompt=initial_prompt,
        condition_on_previous_text=True,
        chunk_length=chunk_length,
        word_timestamps=word_timestamps,
    )

    def iterator() -> Iterator[Segment]:
        for segment in segments:
            yield Segment(
                start=float(segment.start),
                end=float(segment.end),
                text=str(segment.text),
                words=_words_from_faster_whisper(segment),
                no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
            )

    return iterator(), info


def _words_from_faster_whisper(segment) -> tuple:
    raw = getattr(segment, "words", None) or ()
    out: list[Word] = []
    for w in raw:
        try:
            text = str(getattr(w, "word", getattr(w, "text", ""))).strip()
            if not text:
                continue
            out.append(Word(start=float(w.start), end=float(w.end), text=text))
        except (AttributeError, TypeError, ValueError):
            continue
    return tuple(out)


def _words_from_payload(raw_words) -> tuple:
    if not raw_words:
        return ()
    out: list[Word] = []
    for w in raw_words:
        try:
            text = str(w.get("text") or "").strip()
            if not text:
                continue
            out.append(Word(start=float(w["start"]), end=float(w["end"]), text=text))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(out)


def transcribe_media_chunks(
    media_path: Path,
    model_name_or_path: str,
    language: str,
    task: str,
    device: str,
    compute_type: str,
    beam_size: int,
    vad: bool,
    initial_prompt: str | None,
    chunk_length: int | None,
    word_timestamps: bool,
    media_chunk_seconds: int,
    media_chunk_overlap_seconds: int,
    allow_chunk_cpu_fallback: bool,
    start_seconds: float = 0.0,
) -> tuple[Iterator[Segment], Any]:
    audio_cache_path, duration = create_audio_cache_wav(media_path)
    chunk_seconds = max(1, int(media_chunk_seconds))
    overlap_seconds = max(0, int(media_chunk_overlap_seconds))
    initial_assigned_start = max(0.0, float(start_seconds))

    info = SimpleNamespace(
        language=language or None,
        language_probability=1.0 if language else None,
        duration=duration,
    )

    def iterator() -> Iterator[Segment]:
        worker = PersistentChunkWorker(
            model_name_or_path=model_name_or_path,
            language=language,
            task=task,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad=vad,
            initial_prompt=initial_prompt,
            chunk_length=chunk_length,
            word_timestamps=word_timestamps,
            allow_cpu_fallback=allow_chunk_cpu_fallback,
        )
        try:
            assigned_start = initial_assigned_start
            while assigned_start < duration:
                assigned_end = min(duration, assigned_start + chunk_seconds)
                slice_start = max(0.0, assigned_start - overlap_seconds)
                slice_end = assigned_end

                segments_for_chunk = worker.transcribe_slice(
                    audio_path=audio_cache_path,
                    start_seconds=slice_start,
                    duration_seconds=slice_end - slice_start,
                )
                for segment in segments_for_chunk:
                    start = segment.start + slice_start
                    end = segment.end + slice_start
                    if end <= assigned_start:
                        continue
                    shifted_words = tuple(
                        Word(
                            start=word.start + slice_start,
                            end=word.end + slice_start,
                            text=word.text,
                        )
                        for word in (segment.words or ())
                    )
                    yield Segment(
                        start=start,
                        end=end,
                        text=segment.text,
                        words=shifted_words,
                        no_speech_prob=segment.no_speech_prob,
                    )

                assigned_start = assigned_end
        finally:
            worker.shutdown()
            try:
                audio_cache_path.unlink()
            except OSError:
                pass

    return iterator(), info


class PersistentChunkWorker:
    """Long-lived chunk_worker subprocess. Loads the model once and serves
    transcribe requests over stdin/stdout.

    Key behaviors:
    - First successful start emits a ``ready`` message; we block until then.
    - If the worker process crashes mid-job (driver/cuDNN), we automatically
      restart on the same device unless that device is GPU and we've already
      crashed once -- then we switch to CPU (subject to allow_cpu_fallback).
    """

    def __init__(
        self,
        model_name_or_path: str,
        language: str,
        task: str,
        device: str,
        compute_type: str,
        beam_size: int,
        vad: bool,
        initial_prompt: str | None,
        chunk_length: int | None,
        word_timestamps: bool,
        allow_cpu_fallback: bool,
    ) -> None:
        self._model = model_name_or_path
        self._language = language
        self._task = task
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._vad = vad
        self._initial_prompt = initial_prompt
        self._chunk_length = chunk_length
        self._word_timestamps = word_timestamps
        self._allow_cpu_fallback = allow_cpu_fallback
        self._gpu_dead = False
        self._proc: subprocess.Popen | None = None

    def transcribe_slice(
        self,
        audio_path: Path,
        start_seconds: float,
        duration_seconds: float,
    ) -> list[Segment]:
        request = {
            "action": "transcribe",
            "input": str(audio_path),
            "start": float(start_seconds),
            "duration": float(duration_seconds),
        }
        try:
            return self._send_and_collect(request)
        except WorkerCrashedError as exc:
            self._handle_worker_crash(exc, start_seconds, duration_seconds)
            return self._send_and_collect(request)

    def _send_and_collect(self, request: dict) -> list[Segment]:
        self._ensure_started()
        proc = self._proc
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except OSError as exc:
            raise WorkerCrashedError(self._read_stderr_tail(), exit_code=proc.poll()) from exc

        segments: list[Segment] = []
        while True:
            line = proc.stdout.readline()
            if not line:
                proc.wait(timeout=10)
                raise WorkerCrashedError(
                    self._read_stderr_tail(),
                    exit_code=proc.returncode if proc.returncode is not None else -1,
                )
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type")
            if kind == "segment":
                segments.append(
                    Segment(
                        start=float(msg["start"]),
                        end=float(msg["end"]),
                        text=str(msg["text"]),
                        words=_words_from_payload(msg.get("words")),
                        no_speech_prob=float(msg.get("no_speech_prob", 0.0) or 0.0),
                    )
                )
            elif kind == "done":
                return segments
            elif kind == "error":
                raise RuntimeError(
                    f"chunk worker reported error: {msg.get('message', '<unknown>')}"
                )
            # else: ignore (e.g., late "ready")

    def _handle_worker_crash(
        self,
        exc: "WorkerCrashedError",
        start_seconds: float,
        duration_seconds: float,
    ) -> None:
        self._teardown_proc()
        on_gpu = self._device.lower() != "cpu" and not self._gpu_dead
        if on_gpu and exc.is_process_crash():
            if not self._allow_cpu_fallback:
                raise RuntimeError(
                    f"GPU persistent-worker crash at "
                    f"{start_seconds:.2f}s+{duration_seconds:.2f}s "
                    f"({exc.summary()}). --no-cpu-fallback is set so we won't "
                    "switch to CPU. Remove --no-cpu-fallback to let CPU rescue "
                    "stuck chunks, or use a different --compute-type."
                ) from exc
            print(
                f"[persistent-worker] GPU process crashed "
                f"({exc.summary()}); restarting on CPU for the rest of the run.",
                file=sys.stderr,
                flush=True,
            )
            self._gpu_dead = True
            return
        # Same-device transient crash — restart and let the caller retry.
        print(
            f"[persistent-worker] worker exited "
            f"({exc.summary()}); restarting on {self._effective_device()}.",
            file=sys.stderr,
            flush=True,
        )

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        device = self._effective_device()
        compute_type = self._effective_compute_type(device)
        command = [
            sys.executable,
            "-m",
            "ytsubtitle.chunk_worker",
            "--serve",
            "--model",
            self._model,
            "--language",
            self._language,
            "--task",
            self._task,
            "--device",
            device,
            "--compute-type",
            compute_type,
            "--beam-size",
            str(self._beam_size),
        ]
        if self._vad:
            command.append("--vad")
        if self._initial_prompt is not None:
            command.extend(["--initial-prompt", self._initial_prompt])
        if self._chunk_length is not None:
            command.extend(["--chunk-length", str(self._chunk_length)])
        if self._word_timestamps:
            command.append("--word-timestamps")

        self._proc = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        # Drain "ready" so our first transcribe doesn't see it interleaved
        # with segment output.
        self._wait_for_ready()

    def _wait_for_ready(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line:
                proc.wait(timeout=10)
                raise WorkerCrashedError(
                    self._read_stderr_tail(),
                    exit_code=proc.returncode if proc.returncode is not None else -1,
                )
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "ready":
                return
            if msg.get("type") == "error":
                raise RuntimeError(
                    f"chunk worker startup error: {msg.get('message', '<unknown>')}"
                )

    def _effective_device(self) -> str:
        if self._gpu_dead and self._device.lower() != "cpu":
            return "cpu"
        return self._device

    def _effective_compute_type(self, device: str) -> str:
        if self._gpu_dead and device.lower() == "cpu":
            return "int8"
        return self._compute_type

    def _read_stderr_tail(self, max_chars: int = 1200) -> str:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return ""
        try:
            data = proc.stderr.read() or ""
        except (OSError, ValueError):
            return ""
        text = data.strip()
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    def _teardown_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        gc.collect()

    def shutdown(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                proc.stdin.flush()
        except OSError:
            pass
        self._teardown_proc()


class WorkerCrashedError(Exception):
    def __init__(self, stderr_tail: str, exit_code: int) -> None:
        super().__init__(stderr_tail)
        self.stderr_tail = stderr_tail
        self.exit_code = exit_code

    def is_process_crash(self) -> bool:
        if not self.stderr_tail:
            return True  # no stderr at all = OS-killed
        code = (self.exit_code or 0) & 0xFFFFFFFF
        return code in _WINDOWS_FATAL_EXIT_CODES

    def summary(self) -> str:
        code = (self.exit_code or 0) & 0xFFFFFFFF
        label = _WINDOWS_FATAL_EXIT_CODES.get(code, "")
        head = f"exit 0x{code:08X}"
        if label:
            head = f"{head} {label}"
        if self.stderr_tail:
            tail = self.stderr_tail.replace("\n", " ")
            if len(tail) > 240:
                tail = tail[-240:]
            return f"{head}: {tail}"
        return head


def run_media_chunk_worker_resilient(
    media_path: Path,
    model_name_or_path: str,
    language: str,
    task: str,
    device: str,
    compute_type: str,
    beam_size: int,
    vad: bool,
    initial_prompt: str | None,
    chunk_length: int | None,
    word_timestamps: bool,
    start_seconds: float,
    duration_seconds: float,
    min_duration_seconds: float = 30.0,
    allow_cpu_fallback: bool = True,
    on_gpu_process_crash=None,
) -> list[Segment]:
    try:
        return run_media_chunk_worker(
            media_path=media_path,
            model_name_or_path=model_name_or_path,
            language=language,
            task=task,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad=vad,
            initial_prompt=initial_prompt,
            chunk_length=chunk_length,
            word_timestamps=word_timestamps,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
    except RuntimeError as exc:
        summary = summarize_chunk_error(exc)
        if is_process_crash_error(exc) and device.lower() != "cpu":
            if allow_cpu_fallback:
                print(
                    f"[chunk {start_seconds:.1f}s+{duration_seconds:.1f}s] "
                    "GPU worker was killed by the OS (driver/cuDNN crash); "
                    "skipping split-retry and running this chunk on CPU. "
                    "Subsequent chunks will also use CPU.",
                    file=sys.stderr,
                    flush=True,
                )
                if on_gpu_process_crash is not None:
                    try:
                        on_gpu_process_crash()
                    except Exception:
                        pass
                return run_media_chunk_worker(
                    media_path=media_path,
                    model_name_or_path=model_name_or_path,
                    language=language,
                    task=task,
                    device="cpu",
                    compute_type="int8",
                    beam_size=beam_size,
                    vad=vad,
                    initial_prompt=initial_prompt,
                    chunk_length=chunk_length,
                    word_timestamps=word_timestamps,
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                )
            # CPU fallback explicitly disabled — fail fast instead of pointless
            # split-retry on a GPU whose driver state is already broken.
            raise RuntimeError(
                f"GPU process crash at {start_seconds:.2f}s+{duration_seconds:.2f}s "
                f"({summary}). --no-cpu-fallback is set so split-retry was skipped — "
                "remove --no-cpu-fallback (or uncheck 'GPU only' in the GUI) "
                "to let CPU finish stuck chunks."
            ) from exc
        if duration_seconds <= min_duration_seconds * 2:
            if allow_cpu_fallback and device.lower() != "cpu":
                print(
                    f"[chunk {start_seconds:.1f}s+{duration_seconds:.1f}s] "
                    f"GPU failed ({summary}); falling back to CPU int8.",
                    file=sys.stderr,
                    flush=True,
                )
                return run_media_chunk_worker(
                    media_path=media_path,
                    model_name_or_path=model_name_or_path,
                    language=language,
                    task=task,
                    device="cpu",
                    compute_type="int8",
                    beam_size=beam_size,
                    vad=vad,
                    initial_prompt=initial_prompt,
                    chunk_length=chunk_length,
                    word_timestamps=word_timestamps,
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                )
            raise

        first_duration = duration_seconds / 2
        second_start = start_seconds + first_duration
        second_duration = duration_seconds - first_duration
        print(
            f"[chunk {start_seconds:.1f}s+{duration_seconds:.1f}s] "
            f"failed ({summary}); splitting into 2 halves and retrying on {device}.",
            file=sys.stderr,
            flush=True,
        )
        first_segments = run_media_chunk_worker_resilient(
            media_path=media_path,
            model_name_or_path=model_name_or_path,
            language=language,
            task=task,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad=vad,
            initial_prompt=initial_prompt,
            chunk_length=chunk_length,
            word_timestamps=word_timestamps,
            start_seconds=start_seconds,
            duration_seconds=first_duration,
            min_duration_seconds=min_duration_seconds,
            allow_cpu_fallback=allow_cpu_fallback,
            on_gpu_process_crash=on_gpu_process_crash,
        )
        second_segments = run_media_chunk_worker_resilient(
            media_path=media_path,
            model_name_or_path=model_name_or_path,
            language=language,
            task=task,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad=vad,
            initial_prompt=initial_prompt,
            chunk_length=chunk_length,
            word_timestamps=word_timestamps,
            start_seconds=second_start,
            duration_seconds=second_duration,
            min_duration_seconds=min_duration_seconds,
            allow_cpu_fallback=allow_cpu_fallback,
            on_gpu_process_crash=on_gpu_process_crash,
        )
        second_offset = second_start - start_seconds
        return first_segments + [
            Segment(
                start=segment.start + second_offset,
                end=segment.end + second_offset,
                text=segment.text,
                no_speech_prob=segment.no_speech_prob,
            )
            for segment in second_segments
        ]


def summarize_chunk_error(exc: BaseException, max_chars: int = 240) -> str:
    text = str(exc).strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text or exc.__class__.__name__


_WINDOWS_FATAL_EXIT_CODES = {
    0xC0000005: "STATUS_ACCESS_VIOLATION (segfault)",
    0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN (commonly a CUDA/cuDNN/DLL crash)",
    0xC000013A: "STATUS_CONTROL_C_EXIT",
    0xC0000142: "STATUS_DLL_INIT_FAILED",
}


def describe_subprocess_crash(returncode: int) -> str:
    """Build a useful detail string when a worker process dies without writing stderr."""
    code = returncode & 0xFFFFFFFF
    label = _WINDOWS_FATAL_EXIT_CODES.get(code)
    suffix = (
        " (worker process was killed by the OS — almost always a GPU driver/cuDNN crash, "
        "not a Python exception)"
    )
    if label:
        return f"{label} 0x{code:08X}{suffix}"
    return f"no stderr; exit 0x{code:08X}{suffix}"


def is_process_crash_error(exc: BaseException) -> bool:
    """True when a chunk failed because the worker process was killed by the OS.

    Heuristic: the error string carries the text inserted by describe_subprocess_crash.
    """
    text = str(exc)
    if "killed by the OS" in text:
        return True
    for code in _WINDOWS_FATAL_EXIT_CODES:
        if f"0x{code:08X}" in text or f"exit code {code}" in text or f"exit code {code & 0xFFFFFFFF}" in text:
            return True
    return False


def run_media_chunk_worker(
    media_path: Path,
    model_name_or_path: str,
    language: str,
    task: str,
    device: str,
    compute_type: str,
    beam_size: int,
    vad: bool,
    initial_prompt: str | None,
    chunk_length: int | None,
    word_timestamps: bool,
    start_seconds: float,
    duration_seconds: float,
) -> list[Segment]:
    command = [
        sys.executable,
        "-m",
        "ytsubtitle.chunk_worker",
        str(media_path),
        "--start",
        format_seconds_arg(start_seconds),
        "--duration",
        format_seconds_arg(duration_seconds),
        "--model",
        model_name_or_path,
        "--language",
        language,
        "--task",
        task,
        "--device",
        device,
        "--compute-type",
        compute_type,
        "--beam-size",
        str(beam_size),
    ]
    if vad:
        command.append("--vad")
    if initial_prompt is not None:
        command.extend(["--initial-prompt", initial_prompt])
    if chunk_length is not None:
        command.extend(["--chunk-length", str(chunk_length)])
    if word_timestamps:
        command.append("--word-timestamps")

    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stderr_text = (completed.stderr or "").strip()
        if stderr_text:
            detail = stderr_text[-1200:] if len(stderr_text) > 1200 else stderr_text
        else:
            detail = describe_subprocess_crash(completed.returncode)
        raise RuntimeError(
            "CUDA/media chunk worker failed "
            f"at {start_seconds:.2f}s for {duration_seconds:.2f}s "
            f"with exit code {completed.returncode}: {detail}"
        )

    segments: list[Segment] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        segments.append(
            Segment(
                start=float(payload["start"]),
                end=float(payload["end"]),
                text=str(payload["text"]),
                no_speech_prob=float(payload.get("no_speech_prob", 0.0) or 0.0),
            )
        )
    return segments


def format_seconds_arg(seconds: float) -> str:
    return f"{seconds:.3f}".rstrip("0").rstrip(".")


def create_audio_cache_wav(media_path: Path, sampling_rate: int = 16000) -> tuple[Path, float]:
    """Decode the input media to a 16 kHz mono PCM WAV.

    Tracks each decoded audio frame's `pts` and pads any mid-stream gaps
    (dropped/invalid frames, or audio stream timing jumps) with silence so the
    WAV's sample-position-to-time mapping matches the original media timeline.
    Also tail-pads the WAV up to `container.duration` when the audio stream
    ends earlier than the video — common with phone screen recordings.
    Without these pads, segments after a gap end up shifted earlier in the
    output, which breaks subtitle timing.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Missing dependency: av/PyAV from faster-whisper.") from exc

    fd, temp_name = tempfile.mkstemp(
        prefix=f"{media_path.stem}.",
        suffix=".ytsubtitle.wav",
    )
    os.close(fd)
    cache_path = Path(temp_name)
    sample_count = 0
    silence_bytes = b"\x00\x00"  # one 16-bit mono silent sample
    gap_threshold_seconds = 0.05  # ignore typical sub-frame jitter
    total_padded_seconds = 0.0

    def make_resampler():
        return av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=sampling_rate,
        )

    resampler = make_resampler()

    try:
        with (
            av.open(str(media_path), mode="r", metadata_errors="ignore") as container,
            wave.open(str(cache_path), "wb") as wav_file,
        ):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sampling_rate)

            container_duration = (
                float(container.duration) / float(av.time_base)
                if container.duration is not None
                else 0.0
            )

            def write_resampled(resampled):
                nonlocal sample_count
                array = resampled.to_ndarray()
                wav_file.writeframes(array.tobytes())
                sample_count += int(array.size)

            def flush_resampler():
                nonlocal resampler
                for resampled in resampler.resample(None):
                    write_resampled(resampled)

            def emit_silence(seconds: float) -> int:
                nonlocal sample_count
                pad = int(round(seconds * sampling_rate))
                if pad <= 0:
                    return 0
                wav_file.writeframes(silence_bytes * pad)
                sample_count += pad
                return pad

            last_input_end = 0.0
            iterator = iter(container.decode(audio=0))
            while True:
                try:
                    frame = next(iterator)
                except StopIteration:
                    break
                except av.error.InvalidDataError:
                    continue

                if frame.pts is not None:
                    frame_start = float(frame.pts * frame.time_base)
                    gap = frame_start - last_input_end
                    if gap > gap_threshold_seconds:
                        flush_resampler()
                        emit_silence(gap)
                        total_padded_seconds += gap
                        resampler = make_resampler()
                    last_input_end = frame_start + (
                        float(frame.samples) / float(frame.sample_rate)
                    )

                # Strip pts so the resampler doesn't try to remap timing across
                # our manual gap pads.
                frame.pts = None
                try:
                    resampled_list = resampler.resample(frame)
                except av.error.InvalidDataError:
                    continue
                for resampled in resampled_list:
                    write_resampled(resampled)

            flush_resampler()

            current_seconds = sample_count / sampling_rate
            tail_gap = container_duration - current_seconds
            if tail_gap > gap_threshold_seconds:
                emit_silence(tail_gap)
                total_padded_seconds += tail_gap

            if total_padded_seconds > 1.0:
                print(
                    f"[audio cache] padded {total_padded_seconds:.1f}s of silence "
                    f"into gaps/tail to align with container duration "
                    f"({container_duration:.1f}s).",
                    flush=True,
                )
    except Exception:
        try:
            cache_path.unlink()
        except OSError:
            pass
        raise
    finally:
        del resampler
        gc.collect()

    return cache_path, sample_count / sampling_rate


def media_duration_seconds(media_path: Path) -> float:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Missing dependency: av/PyAV from faster-whisper.") from exc

    with av.open(str(media_path), mode="r", metadata_errors="ignore") as container:
        if container.duration is not None:
            return float(container.duration) / float(av.time_base)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if audio_streams and audio_streams[0].duration is not None:
            return float(audio_streams[0].duration * audio_streams[0].time_base)

    raise RuntimeError(f"Could not determine media duration: {media_path}")


def decode_audio_slice(
    input_file: str,
    start_seconds: float,
    duration_seconds: float,
    sampling_rate: int = 16000,
):
    if input_file.lower().endswith(".ytsubtitle.wav"):
        return decode_wav_slice(
            input_file=input_file,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            sampling_rate=sampling_rate,
        )

    try:
        import av
        import numpy as np
        from faster_whisper.audio import (
            _group_frames,
            _ignore_invalid_frames,
            _resample_frames,
        )
    except ImportError as exc:
        raise RuntimeError("Missing dependency: av/PyAV from faster-whisper.") from exc

    end_seconds = max(start_seconds, start_seconds + duration_seconds)
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sampling_rate,
    )
    raw_buffer = io.BytesIO()
    dtype = None
    first_frame_time: float | None = None

    with av.open(input_file, mode="r", metadata_errors="ignore") as container:
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            raise RuntimeError(f"No audio stream found: {input_file}")
        stream = audio_streams[0]
        seek_target = max(0, int(start_seconds * av.time_base))
        container.seek(seek_target, any_frame=False, backward=True)

        def selected_frames():
            nonlocal first_frame_time
            for frame in container.decode(stream):
                if frame.pts is not None:
                    frame_time = float(frame.pts * frame.time_base)
                    frame_duration = float(frame.samples / frame.sample_rate)
                    if frame_time > end_seconds + 1:
                        break
                    if frame_time + frame_duration < start_seconds:
                        continue
                    if first_frame_time is None:
                        first_frame_time = frame_time
                yield frame

        frames = _ignore_invalid_frames(selected_frames())
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)

        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    del resampler
    gc.collect()

    if dtype is None:
        return np.array([], dtype=np.float32)

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
    audio = audio.astype(np.float32) / 32768.0

    if first_frame_time is not None and first_frame_time < start_seconds:
        trim_start = int(round((start_seconds - first_frame_time) * sampling_rate))
        audio = audio[min(trim_start, len(audio)) :]
    target_samples = int(round(duration_seconds * sampling_rate))
    if target_samples >= 0:
        audio = audio[:target_samples]
    return audio


def decode_wav_slice(
    input_file: str,
    start_seconds: float,
    duration_seconds: float,
    sampling_rate: int = 16000,
):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Missing dependency: numpy from faster-whisper.") from exc

    with wave.open(input_file, "rb") as wav_file:
        source_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        if source_rate != sampling_rate or sample_width != 2:
            raise RuntimeError(
                "Unexpected audio cache format: "
                f"{source_rate} Hz, {channels} channel(s), {sample_width} bytes/sample"
            )
        start_frame = max(0, int(round(start_seconds * source_rate)))
        frame_count = max(0, int(round(duration_seconds * source_rate)))
        start_frame = min(start_frame, wav_file.getnframes())
        wav_file.setpos(start_frame)
        raw = wav_file.readframes(frame_count)

    audio = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return audio.astype(np.float32) / 32768.0


def configure_cuda_runtime_path() -> list[Path]:
    added: list[Path] = []
    for directory in cuda_runtime_directories():
        if not directory.exists():
            continue

        directory_text = str(directory)
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if directory_text not in path_entries:
            os.environ["PATH"] = directory_text + os.pathsep + os.environ.get("PATH", "")

        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None and not any(
            str(existing) == directory_text for existing in added
        ):
            try:
                _DLL_DIRECTORY_HANDLES.append(add_dll_directory(directory_text))
            except OSError:
                pass
        added.append(directory)

    return added


def supported_compute_types(device: str) -> set[str]:
    configure_cuda_runtime_path()
    try:
        import ctranslate2
    except ImportError:
        return set()

    try:
        return {str(value) for value in ctranslate2.get_supported_compute_types(device)}
    except Exception:
        return set()


def cuda_runtime_directories() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "vendor" / "cuda12",
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"),
    ]
    return [path for path in candidates if (path / "cublas64_12.dll").exists()]
