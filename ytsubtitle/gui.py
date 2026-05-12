from __future__ import annotations

import os
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


VIDEO_FILE_TYPES = [
    ("Media files", "*.mp4 *.mkv *.mov *.avi *.m4v *.webm *.mp3 *.wav *.m4a *.flac"),
    ("Video files", "*.mp4 *.mkv *.mov *.avi *.m4v *.webm"),
    ("Audio files", "*.mp3 *.wav *.m4a *.flac"),
    ("All files", "*.*"),
]

OUTPUT_FILE_TYPES = [
    ("SubRip subtitle", "*.srt"),
    ("All files", "*.*"),
]

PROGRESS_PREFIX = "__YTSUBTITLE_PROGRESS__"
OUTPUT_FORMAT_OPTIONS = [
    ("srt", "SRT"),
    ("vtt", "VTT"),
    ("sbv", "SBV"),
    ("ass", "ASS"),
    ("ttml", "TTML"),
    ("lrc", "LRC"),
    ("csv", "CSV"),
    ("txt", "TXT"),
    ("jsonl", "JSONL"),
]
DEFAULT_OUTPUT_FORMATS = {"srt", "vtt", "txt"}
MODEL_OPTIONS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
]
# Languages supported by Whisper (multilingual checkpoints recognise all of them;
# `.en` checkpoints only support English). Codes match faster-whisper's
# tokenizer.LANGUAGES keys.
LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("", "Auto-detect"),
    ("af", "Afrikaans"),
    ("sq", "Albanian"),
    ("am", "Amharic"),
    ("ar", "Arabic"),
    ("hy", "Armenian"),
    ("as", "Assamese"),
    ("az", "Azerbaijani"),
    ("ba", "Bashkir"),
    ("eu", "Basque"),
    ("be", "Belarusian"),
    ("bn", "Bengali"),
    ("bs", "Bosnian"),
    ("br", "Breton"),
    ("bg", "Bulgarian"),
    ("my", "Burmese"),
    ("yue", "Cantonese"),
    ("ca", "Catalan"),
    ("zh", "Chinese"),
    ("hr", "Croatian"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("et", "Estonian"),
    ("fo", "Faroese"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("gl", "Galician"),
    ("ka", "Georgian"),
    ("de", "German"),
    ("el", "Greek"),
    ("gu", "Gujarati"),
    ("ht", "Haitian Creole"),
    ("ha", "Hausa"),
    ("haw", "Hawaiian"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("is", "Icelandic"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("jw", "Javanese"),
    ("kn", "Kannada"),
    ("kk", "Kazakh"),
    ("km", "Khmer"),
    ("ko", "Korean"),
    ("lo", "Lao"),
    ("la", "Latin"),
    ("lv", "Latvian"),
    ("ln", "Lingala"),
    ("lt", "Lithuanian"),
    ("lb", "Luxembourgish"),
    ("mk", "Macedonian"),
    ("mg", "Malagasy"),
    ("ms", "Malay"),
    ("ml", "Malayalam"),
    ("mt", "Maltese"),
    ("mi", "Maori"),
    ("mr", "Marathi"),
    ("mn", "Mongolian"),
    ("ne", "Nepali"),
    ("no", "Norwegian"),
    ("nn", "Nynorsk"),
    ("oc", "Occitan"),
    ("ps", "Pashto"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("pa", "Punjabi"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sa", "Sanskrit"),
    ("sr", "Serbian"),
    ("sn", "Shona"),
    ("sd", "Sindhi"),
    ("si", "Sinhala"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("so", "Somali"),
    ("es", "Spanish"),
    ("su", "Sundanese"),
    ("sw", "Swahili"),
    ("sv", "Swedish"),
    ("tl", "Tagalog"),
    ("tg", "Tajik"),
    ("ta", "Tamil"),
    ("tt", "Tatar"),
    ("te", "Telugu"),
    ("th", "Thai"),
    ("bo", "Tibetan"),
    ("tr", "Turkish"),
    ("tk", "Turkmen"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("uz", "Uzbek"),
    ("vi", "Vietnamese"),
    ("cy", "Welsh"),
    ("yi", "Yiddish"),
    ("yo", "Yoruba"),
]
LANGUAGE_DISPLAY_TO_CODE: dict[str, str] = {
    (f"{label} ({code})" if code else label): code for code, label in LANGUAGE_OPTIONS
}
LANGUAGE_DISPLAY_VALUES: list[str] = list(LANGUAGE_DISPLAY_TO_CODE.keys())
LANGUAGE_CODE_TO_DISPLAY: dict[str, str] = {
    code: display for display, code in LANGUAGE_DISPLAY_TO_CODE.items()
}
TASK_OPTIONS: list[tuple[str, str]] = [
    ("transcribe", "Transcribe (same language)"),
    ("translate", "Translate to English"),
]
TASK_DISPLAY_VALUES: list[str] = [label for _, label in TASK_OPTIONS]
TASK_DISPLAY_TO_VALUE: dict[str, str] = {label: value for value, label in TASK_OPTIONS}
TASK_VALUE_TO_DISPLAY: dict[str, str] = {value: label for value, label in TASK_OPTIONS}
DEFAULT_TRANSCRIBE_PROMPT = (
    "Chinese video subtitles. Output clear Chinese punctuation. "
    "Keep names, brands, and technical terms."
)
DEFAULT_TRANSLATE_PROMPT = (
    "Translate the audio into clear English. "
    "Keep names, brands, and technical terms."
)
DEFAULT_PROMPTS_BY_TASK: dict[str, str] = {
    "transcribe": DEFAULT_TRANSCRIBE_PROMPT,
    "translate": DEFAULT_TRANSLATE_PROMPT,
}
KNOWN_DEFAULT_PROMPTS: set[str] = set(DEFAULT_PROMPTS_BY_TASK.values())
SCRIPT_OPTIONS = ["none", "simplified", "traditional"]
RUN_ON_OPTIONS = ["Auto", "CPU", "GPU (NVIDIA CUDA)"]
RUN_ON_TO_DEVICE = {
    "Auto": "auto",
    "CPU": "cpu",
    "GPU (NVIDIA CUDA)": "cuda",
}
COMPUTE_OPTIONS = ["auto", "int8", "int8_float32", "int8_float16", "float16", "float32"]
BEAM_SIZE_OPTIONS = ["1", "3", "5"]
MEDIA_CHUNK_OPTIONS = ["Off", "300", "600", "900", "1200"]


def build_transcribe_command(
    python_exe: str,
    input_path: str,
    output_file: str,
    model: str,
    language: str,
    script: str,
    device: str,
    compute_type: str,
    vad: bool,
    output_formats: Iterable[str],
    extract_audio: bool,
    initial_prompt: str,
    low_vram: bool = False,
    chunk_length: int | None = None,
    accurate_timing: bool = False,
    beam_size: int = 1,
    word_timestamps: bool = True,
    media_chunk_seconds: int | None = 300,
    media_chunk_overlap_seconds: int = 3,
    no_cpu_fallback: bool = False,
    task: str = "transcribe",
) -> list[str]:
    output_dir, basename, _srt_path = split_output_target(output_file)
    formats = ",".join(normalize_output_formats(output_formats))
    command = [
        python_exe,
        "-m",
        "ytsubtitle",
        input_path,
        "--output-dir",
        output_dir,
        "--basename",
        basename,
        "--model",
        model,
        "--language",
        language,
        "--task",
        task,
        "--script",
        script,
        "--device",
        device,
        "--compute-type",
        compute_type,
        "--beam-size",
        str(max(1, int(beam_size))),
        "--media-chunk-overlap-seconds",
        str(max(0, int(media_chunk_overlap_seconds))),
        "--formats",
        formats,
        "--progress-every",
        "10",
        "--progress-json",
    ]
    if media_chunk_seconds is not None:
        command.extend(["--media-chunk-seconds", str(int(media_chunk_seconds))])
    if word_timestamps:
        command.append("--word-timestamps")
    if vad:
        command.append("--vad")
    if extract_audio:
        command.append("--extract-audio")
    if low_vram:
        command.append("--low-vram")
    if chunk_length is not None:
        command.extend(["--chunk-length", str(chunk_length)])
    if accurate_timing:
        command.append("--accurate-timing")
    if no_cpu_fallback:
        command.append("--no-cpu-fallback")
    if initial_prompt.strip():
        command.extend(["--initial-prompt", initial_prompt.strip()])
    return command


class SubtitleGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Subtitle Generator")
        self.minsize(920, 680)

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[tuple[str, str | int | None]] = queue.Queue()

        self.input_path = tk.StringVar()
        self.output_file = tk.StringVar(value=str(project_root() / "subtitles" / "subtitle.srt"))
        self.output_file_is_auto = True
        self.model = tk.StringVar(value="medium")
        self.language = tk.StringVar(
            value=LANGUAGE_CODE_TO_DISPLAY.get("zh", "Auto-detect")
        )
        self.task = tk.StringVar(value=TASK_VALUE_TO_DISPLAY["transcribe"])
        self.script = tk.StringVar(value="simplified")
        self.run_on = tk.StringVar(value="Auto")
        self.compute_type = tk.StringVar(value="int8_float32")
        self.beam_size = tk.StringVar(value="1")
        self.media_chunk = tk.StringVar(value="300")
        self.vad = tk.BooleanVar(value=False)
        self.low_vram = tk.BooleanVar(value=False)
        self.accurate_timing = tk.BooleanVar(value=False)
        self.word_timestamps = tk.BooleanVar(value=True)
        self.gpu_only = tk.BooleanVar(value=True)
        self.output_format_vars = {
            value: tk.BooleanVar(value=value in DEFAULT_OUTPUT_FORMATS)
            for value, _label in OUTPUT_FORMAT_OPTIONS
        }
        self.extract_audio = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready")
        self.progress_text = tk.StringVar(value="Ready")
        self.duration_seconds = 0.0
        self.last_progress_seconds = 0.0
        self.progress_started_at = 0.0

        self._build_ui()
        self.task.trace_add("write", self._on_task_change)
        self.after(100, self._drain_log_queue)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(16, 14, 16, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = ttk.Label(
            header,
            text="Subtitle Generator",
            font=("Segoe UI", 16, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")
        subtitle = ttk.Label(
            header,
            text="Pick a video or audio file, choose a language, and generate timed subtitle files.",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        body = ttk.PanedWindow(self, orient=tk.VERTICAL)
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))

        controls = ttk.Frame(body, padding=12)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)
        body.add(controls, weight=0)

        row = 0
        ttk.Label(controls, text="Video/audio").grid(row=row, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.input_path).grid(
            row=row, column=1, columnspan=3, sticky="ew", padx=(10, 8)
        )
        ttk.Button(controls, text="Browse", command=self._browse_input).grid(
            row=row, column=4, sticky="ew"
        )

        row += 1
        ttk.Label(controls, text="Output base file").grid(
            row=row, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(controls, textvariable=self.output_file).grid(
            row=row, column=1, columnspan=3, sticky="ew", padx=(10, 8), pady=(10, 0)
        )
        ttk.Button(controls, text="Save As", command=self._browse_output).grid(
            row=row, column=4, sticky="ew", pady=(10, 0)
        )

        row += 1
        self._add_combo_row(
            controls,
            row,
            [
                ("Model", self.model, MODEL_OPTIONS),
                ("Script", self.script, SCRIPT_OPTIONS),
                ("Run on", self.run_on, RUN_ON_OPTIONS),
                ("Compute", self.compute_type, COMPUTE_OPTIONS),
            ],
        )

        row += 1
        self._add_combo_row(
            controls,
            row,
            [
                ("Beam", self.beam_size, BEAM_SIZE_OPTIONS),
                ("Long media chunks (s)", self.media_chunk, MEDIA_CHUNK_OPTIONS),
            ],
        )

        row += 1
        ttk.Label(controls, text="Language").grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(
            controls,
            textvariable=self.language,
            values=LANGUAGE_DISPLAY_VALUES,
            state="normal",
            width=22,
        ).grid(row=row, column=1, sticky="w", padx=(10, 8), pady=(10, 0))
        ttk.Checkbutton(controls, text="Skip silence (VAD)", variable=self.vad).grid(
            row=row, column=2, sticky="w", padx=(8, 8), pady=(10, 0)
        )
        ttk.Checkbutton(
            controls, text="Word timestamps", variable=self.word_timestamps
        ).grid(row=row, column=3, sticky="w", padx=(8, 8), pady=(10, 0))

        row += 1
        ttk.Label(controls, text="Task").grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(
            controls,
            textvariable=self.task,
            values=TASK_DISPLAY_VALUES,
            state="readonly",
            width=28,
        ).grid(row=row, column=1, sticky="w", padx=(10, 8), pady=(10, 0))

        row += 1
        ttk.Label(controls, text="GPU").grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Checkbutton(controls, text="Limit GPU VRAM", variable=self.low_vram).grid(
            row=row, column=1, sticky="w", padx=(10, 8), pady=(10, 0)
        )
        ttk.Checkbutton(
            controls, text="Accurate timing", variable=self.accurate_timing
        ).grid(row=row, column=2, sticky="w", padx=(8, 8), pady=(10, 0))
        ttk.Checkbutton(
            controls, text="GPU only (no CPU fallback)", variable=self.gpu_only
        ).grid(row=row, column=3, sticky="w", padx=(8, 8), pady=(10, 0))
        ttk.Checkbutton(
            controls, text="Extract audio with FFmpeg", variable=self.extract_audio
        ).grid(
            row=row, column=4, sticky="w", padx=(8, 0), pady=(10, 0)
        )

        row += 1
        ttk.Label(controls, text="Formats").grid(row=row, column=0, sticky="nw", pady=(10, 0))
        formats_frame = ttk.Frame(controls)
        formats_frame.grid(
            row=row, column=1, columnspan=4, sticky="ew", padx=(10, 0), pady=(10, 0)
        )
        for index, (value, label) in enumerate(OUTPUT_FORMAT_OPTIONS):
            ttk.Checkbutton(
                formats_frame,
                text=label,
                variable=self.output_format_vars[value],
            ).grid(row=index // 5, column=index % 5, sticky="w", padx=(0, 18), pady=(0, 4))

        row += 1
        ttk.Label(controls, text="Prompt").grid(row=row, column=0, sticky="nw", pady=(10, 0))
        self.prompt_text = tk.Text(controls, height=3, wrap="word", undo=True)
        self.prompt_text.grid(
            row=row, column=1, columnspan=4, sticky="ew", padx=(10, 0), pady=(10, 0)
        )
        self.prompt_text.insert("1.0", DEFAULT_TRANSCRIBE_PROMPT)

        row += 1
        actions = ttk.Frame(controls)
        actions.grid(row=row, column=0, columnspan=5, sticky="ew", pady=(14, 0))
        actions.columnconfigure(4, weight=1)

        self.start_button = ttk.Button(actions, text="Generate Subtitles", command=self._start)
        self.start_button.grid(row=0, column=0, sticky="w")
        self.cancel_button = ttk.Button(
            actions, text="Cancel", command=self._cancel, state=tk.DISABLED
        )
        self.cancel_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Button(actions, text="Open Output Folder", command=self._open_output_dir).grid(
            row=0, column=2, sticky="w", padx=(8, 0)
        )
        ttk.Button(actions, text="Clear Log", command=self._clear_log).grid(
            row=0, column=3, sticky="w", padx=(8, 0)
        )
        ttk.Label(actions, textvariable=self.status).grid(row=0, column=4, sticky="e")

        log_frame = ttk.Frame(body, padding=(0, 10, 0, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        body.add(log_frame, weight=1)

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            state=tk.DISABLED,
            font=("Consolas", 10),
            height=18,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        progress_frame = ttk.Frame(self)
        progress_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        progress_frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(
            progress_frame, mode="determinate", maximum=100, value=0
        )
        self.progress.grid(row=0, column=0, sticky="ew")
        ttk.Label(progress_frame, textvariable=self.progress_text, width=46).grid(
            row=0, column=1, sticky="e", padx=(10, 0)
        )

    def _add_combo_row(
        self,
        parent: ttk.Frame,
        row: int,
        items: Iterable[tuple[str, tk.StringVar, list[str]]],
    ) -> None:
        combo_frame = ttk.Frame(parent)
        combo_frame.grid(row=row, column=0, columnspan=5, sticky="ew", pady=(10, 0))

        for index, (label, variable, values) in enumerate(items):
            combo_frame.columnconfigure(index, weight=1)
            group = ttk.Frame(combo_frame)
            group.grid(row=0, column=index, sticky="ew", padx=(0, 12))
            group.columnconfigure(1, weight=1)
            ttk.Label(group, text=label).grid(row=0, column=0, sticky="w")
            combo = ttk.Combobox(
                group,
                textvariable=variable,
                values=values,
                state="normal" if label == "Model" else "readonly",
                width=14,
            )
            combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(title="Choose video or audio", filetypes=VIDEO_FILE_TYPES)
        if path:
            self.input_path.set(path)
            if self.output_file_is_auto:
                self.output_file.set(str(default_output_file_for(path)))

    def _browse_output(self) -> None:
        current = ensure_srt_suffix(self.output_file.get().strip())
        current_path = Path(current).resolve()
        path = filedialog.asksaveasfilename(
            title="Choose output subtitle file",
            defaultextension=".srt",
            filetypes=OUTPUT_FILE_TYPES,
            initialdir=str(current_path.parent),
            initialfile=current_path.name,
        )
        if path:
            self.output_file.set(str(ensure_srt_suffix(path)))
            self.output_file_is_auto = False

    def _start(self) -> None:
        input_path = self.input_path.get().strip()
        output_file = ensure_srt_suffix(self.output_file.get().strip())

        if not input_path:
            messagebox.showerror("Missing video", "Choose a video or audio file first.")
            return
        if not Path(input_path).exists():
            messagebox.showerror("File not found", f"Input file does not exist:\n{input_path}")
            return
        if not output_file:
            messagebox.showerror("Missing output file", "Choose an output .srt file.")
            return
        output_formats = self._selected_output_formats()
        if not output_formats:
            messagebox.showerror("Missing output format", "Choose at least one output format.")
            return

        self.output_file.set(output_file)
        output_dir, _basename, _srt_path = split_output_target(output_file)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        python_exe = find_python_executable()
        command = build_transcribe_command(
            python_exe=python_exe,
            input_path=input_path,
            output_file=output_file,
            model=self.model.get(),
            language=parse_language_value(self.language.get()),
            script=self.script.get(),
            device=RUN_ON_TO_DEVICE.get(self.run_on.get(), "auto"),
            compute_type=self.compute_type.get(),
            vad=self.vad.get(),
            output_formats=output_formats,
            extract_audio=self.extract_audio.get(),
            initial_prompt=self.prompt_text.get("1.0", tk.END),
            low_vram=self.low_vram.get(),
            accurate_timing=self.accurate_timing.get(),
            beam_size=parse_int_option(self.beam_size.get(), 1),
            word_timestamps=self.word_timestamps.get(),
            media_chunk_seconds=parse_chunk_option(self.media_chunk.get()),
            media_chunk_overlap_seconds=3,
            no_cpu_fallback=self.gpu_only.get()
            and RUN_ON_TO_DEVICE.get(self.run_on.get(), "auto") != "cpu",
            task=parse_task_value(self.task.get()),
        )

        self._clear_log()
        self._append_log("Command:\n" + printable_command(command) + "\n\n")
        self._set_running(True)

        thread = threading.Thread(target=self._run_process, args=(command,), daemon=True)
        thread.start()

    def _run_process(self, command: list[str]) -> None:
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(project_root()),
                env=child_process_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self.log_queue.put(("line", f"Failed to start transcription: {exc}\n"))
            self.log_queue.put(("done", 1))
            return

        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.log_queue.put(("line", line))

        return_code = self.process.wait()
        self.log_queue.put(("done", return_code))

    def _cancel(self) -> None:
        if self.process and self.process.poll() is None:
            self._append_log("\nCancelling transcription...\n")
            self.process.terminate()
            self.cancel_button.configure(state=tk.DISABLED)

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    line = str(payload)
                    if not self._handle_progress_line(line):
                        self._append_log(line)
                elif kind == "done":
                    code = int(payload or 0)
                    self._set_running(False)
                    if code == 0:
                        self.status.set("Done")
                        self.progress.configure(value=100)
                        self.progress_text.set("100% complete")
                        self._append_log("\nFinished successfully.\n")
                    else:
                        self.status.set(f"Stopped with exit code {code}")
                        self._append_log(f"\nStopped with exit code {code}.\n")
                    self.process = None
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def _set_running(self, running: bool) -> None:
        if running:
            self.status.set("Running")
            self.duration_seconds = 0.0
            self.last_progress_seconds = 0.0
            self.progress_started_at = 0.0
            self.progress.configure(value=0)
            self.progress_text.set("Starting...")
            self.start_button.configure(state=tk.DISABLED)
            self.cancel_button.configure(state=tk.NORMAL)
        else:
            self.start_button.configure(state=tk.NORMAL)
            self.cancel_button.configure(state=tk.DISABLED)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _open_output_dir(self) -> None:
        output_file = ensure_srt_suffix(self.output_file.get().strip())
        path = split_output_target(output_file)[2].parent if output_file else project_root() / "subtitles"
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(path)
        except OSError as exc:
            messagebox.showerror("Cannot open folder", str(exc))

    def _selected_output_formats(self) -> list[str]:
        return [
            value
            for value, _label in OUTPUT_FORMAT_OPTIONS
            if self.output_format_vars[value].get()
        ]

    def _handle_progress_line(self, line: str) -> bool:
        if not line.startswith(PROGRESS_PREFIX):
            return False

        try:
            event = json.loads(line[len(PROGRESS_PREFIX) :])
        except json.JSONDecodeError:
            return True

        event_name = event.get("event")
        if event_name == "duration":
            self.duration_seconds = max(0.0, float(event.get("duration") or 0.0))
            self.progress_started_at = time.monotonic()
            self.status.set("Transcribing")
            self.progress.configure(value=0)
            if self.duration_seconds:
                self.progress_text.set(
                    f"0% of {format_seconds(self.duration_seconds)} - ETA calculating"
                )
            else:
                self.progress_text.set("Transcribing")
            return True

        if event_name == "progress":
            seconds = max(0.0, float(event.get("seconds") or 0.0))
            self.last_progress_seconds = seconds
            if self.duration_seconds > 0:
                percent = min(100.0, seconds / self.duration_seconds * 100.0)
                eta_seconds = event.get("eta_seconds")
                if eta_seconds is None:
                    eta_seconds = estimate_time_left(
                        processed_seconds=seconds,
                        duration_seconds=self.duration_seconds,
                        elapsed_seconds=time.monotonic() - self.progress_started_at,
                    )
                eta_text = (
                    f" - ETA {format_time_left(float(eta_seconds))} left"
                    if eta_seconds is not None
                    else " - ETA calculating"
                )
                self.progress.configure(value=percent)
                self.progress_text.set(
                    f"{percent:5.1f}%  {format_seconds(seconds)} / "
                    f"{format_seconds(self.duration_seconds)}{eta_text}"
                )
                status_eta = (
                    f", {format_time_left(float(eta_seconds))} left"
                    if eta_seconds is not None
                    else ""
                )
                self.status.set(f"Transcribing {percent:.1f}%{status_eta}")
            else:
                self.progress_text.set(f"Processed {format_seconds(seconds)}")
            return True

        if event_name == "done":
            self.progress.configure(value=100)
            self.progress_text.set("100% complete")
            return True

        return True

    def _on_task_change(self, *_args: object) -> None:
        current = self.prompt_text.get("1.0", "end-1c").strip()
        if current and current not in KNOWN_DEFAULT_PROMPTS:
            return
        task_value = parse_task_value(self.task.get())
        new_prompt = DEFAULT_PROMPTS_BY_TASK.get(task_value, DEFAULT_TRANSCRIBE_PROMPT)
        if new_prompt == current:
            return
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", new_prompt)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_srt_suffix(path: str) -> str:
    path = path.strip()
    if not path:
        return ""
    output_path = Path(path).expanduser()
    if output_path.suffix.lower() != ".srt":
        output_path = output_path.with_suffix(".srt")
    return str(output_path)


def split_output_target(output_file: str) -> tuple[str, str, Path]:
    output_file = ensure_srt_suffix(output_file)
    if not output_file:
        raise ValueError("Output file is required.")
    srt_path = Path(output_file).expanduser().resolve()
    return str(srt_path.parent), srt_path.stem, srt_path


def default_output_file_for(input_path: str) -> Path:
    input_file = Path(input_path).expanduser().resolve()
    return input_file.parent / "subtitles" / f"{input_file.stem}.srt"


def normalize_output_formats(output_formats: Iterable[str]) -> list[str]:
    supported = {value for value, _label in OUTPUT_FORMAT_OPTIONS}
    selected: list[str] = []
    for value in output_formats:
        value = value.strip().lower()
        if value in supported and value not in selected:
            selected.append(value)
    if not selected:
        raise ValueError("At least one output format is required.")
    return selected


def parse_task_value(value: str) -> str:
    text = (value or "").strip()
    if text in TASK_DISPLAY_TO_VALUE:
        return TASK_DISPLAY_TO_VALUE[text]
    lowered = text.lower()
    if lowered in {value for value, _ in TASK_OPTIONS}:
        return lowered
    return "transcribe"


def parse_language_value(value: str) -> str:
    text = (value or "").strip()
    if not text or text.lower() in {"auto", "auto-detect"}:
        return ""
    if text in LANGUAGE_DISPLAY_TO_CODE:
        return LANGUAGE_DISPLAY_TO_CODE[text]
    if "(" in text and text.endswith(")"):
        inner = text.rsplit("(", 1)[1].rstrip(")").strip()
        if inner:
            return inner.lower()
    return text.lower()


def parse_int_option(value: str, default: int) -> int:
    try:
        return max(1, int(str(value).strip()))
    except (TypeError, ValueError):
        return default


def parse_chunk_option(value: str) -> int | None:
    text = str(value).strip().lower()
    if not text or text in {"off", "none", "0"}:
        return None
    try:
        seconds = int(text)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def find_python_executable() -> str:
    root = project_root()
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def child_process_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    cuda_runtime = project_root() / "vendor" / "cuda12"
    if cuda_runtime.exists():
        env["PATH"] = str(cuda_runtime) + os.pathsep + env.get("PATH", "")
    return env


def printable_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours}:{minutes:02}:{secs:02}"
    return f"{minutes}:{secs:02}"


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


def main() -> int:
    app = SubtitleGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
