from __future__ import annotations

import os
import json
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


VIDEO_FILE_TYPES = [
    ("Media files", "*.mp4 *.mkv *.mov *.avi *.m4v *.webm *.mp3 *.wav *.m4a *.flac"),
    ("Video files", "*.mp4 *.mkv *.mov *.avi *.m4v *.webm"),
    ("Audio files", "*.mp3 *.wav *.m4a *.flac"),
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
        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.queue_items: list[str] = []
        self.queue_running = False
        self.cancel_requested = False
        self.output_dir = tk.StringVar(value="")
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
        self.delete_downloads_after = tk.BooleanVar(value=False)
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
            text="Add one or more files to the queue, choose options, and generate subtitles.",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        body = ttk.PanedWindow(self, orient=tk.VERTICAL)
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))

        controls = ttk.Frame(body, padding=12)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)
        body.add(controls, weight=0)

        row = 0
        ttk.Label(controls, text="Files").grid(row=row, column=0, sticky="nw")

        queue_frame = ttk.Frame(controls)
        queue_frame.grid(row=row, column=1, columnspan=3, sticky="nsew", padx=(10, 8))
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(0, weight=1)

        self.queue_tree = ttk.Treeview(
            queue_frame,
            columns=("file", "status"),
            show="headings",
            height=5,
            selectmode="extended",
        )
        self.queue_tree.heading("file", text="File")
        self.queue_tree.heading("status", text="Status")
        self.queue_tree.column("file", width=460, anchor="w")
        self.queue_tree.column("status", width=110, anchor="w")
        self.queue_tree.grid(row=0, column=0, sticky="nsew")
        queue_scroll = ttk.Scrollbar(
            queue_frame, orient="vertical", command=self.queue_tree.yview
        )
        queue_scroll.grid(row=0, column=1, sticky="ns")
        self.queue_tree.configure(yscrollcommand=queue_scroll.set)

        queue_buttons = ttk.Frame(controls)
        queue_buttons.grid(row=row, column=4, sticky="nsew")
        self.add_files_button = ttk.Button(
            queue_buttons, text="Add Files", command=self._add_files
        )
        self.add_files_button.grid(row=0, column=0, sticky="ew")
        self.add_url_button = ttk.Button(
            queue_buttons, text="Add URL", command=self._add_url
        )
        self.add_url_button.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.remove_button = ttk.Button(
            queue_buttons, text="Remove", command=self._remove_selected
        )
        self.remove_button.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.clear_queue_button = ttk.Button(
            queue_buttons, text="Clear", command=self._clear_queue
        )
        self.clear_queue_button.grid(row=3, column=0, sticky="ew", pady=(4, 0))

        row += 1
        ttk.Label(controls, text="Output folder").grid(
            row=row, column=0, sticky="w", pady=(10, 0)
        )
        self.output_dir_entry = ttk.Entry(controls, textvariable=self.output_dir)
        self.output_dir_entry.grid(
            row=row, column=1, columnspan=3, sticky="ew", padx=(10, 8), pady=(10, 0)
        )
        self.output_dir_browse_button = ttk.Button(
            controls, text="Browse", command=self._browse_output_dir
        )
        self.output_dir_browse_button.grid(row=row, column=4, sticky="ew", pady=(10, 0))

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
        ttk.Checkbutton(
            controls,
            text="Delete URL audio after",
            variable=self.delete_downloads_after,
        ).grid(row=row, column=4, sticky="w", padx=(8, 0), pady=(10, 0))

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

        self.start_button = ttk.Button(actions, text="Start Queue", command=self._start)
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

    def _add_files(self) -> None:
        if self.queue_running:
            return
        paths = filedialog.askopenfilenames(
            title="Choose video or audio files", filetypes=VIDEO_FILE_TYPES
        )
        for path in paths:
            self.queue_items.append(path)
        self._rebuild_queue_tree()

    def _add_url(self) -> None:
        if self.queue_running:
            return
        url = simpledialog.askstring(
            "Add URL",
            "Paste a YouTube (or other video) URL:",
            parent=self,
        )
        if not url:
            return
        url = url.strip()
        if not is_url(url):
            messagebox.showerror(
                "Invalid URL",
                "URL must start with http:// or https://",
            )
            return
        self.queue_items.append(url)
        self._rebuild_queue_tree()

    def _remove_selected(self) -> None:
        if self.queue_running:
            return
        selection = self.queue_tree.selection()
        if not selection:
            return
        indices = sorted({int(iid) for iid in selection}, reverse=True)
        for idx in indices:
            if 0 <= idx < len(self.queue_items):
                del self.queue_items[idx]
        self._rebuild_queue_tree()

    def _clear_queue(self) -> None:
        if self.queue_running:
            return
        self.queue_items.clear()
        self.queue_tree.delete(*self.queue_tree.get_children())

    def _browse_output_dir(self) -> None:
        if self.queue_running:
            return
        current = self.output_dir.get().strip()
        if current and Path(current).exists():
            initial = current
        elif self.queue_items:
            initial = str(Path(self.queue_items[0]).parent)
        else:
            initial = str(project_root())
        path = filedialog.askdirectory(
            title="Choose output folder (leave empty for per-file default)",
            initialdir=initial,
            mustexist=False,
        )
        if path:
            self.output_dir.set(path)

    def _rebuild_queue_tree(self) -> None:
        self.queue_tree.delete(*self.queue_tree.get_children())
        for index, item in enumerate(self.queue_items):
            self.queue_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(display_name_for_queue_item(item), "Pending"),
            )

    def _set_item_status(self, index: int, status: str) -> None:
        iid = str(index)
        if not self.queue_tree.exists(iid):
            return
        if 0 <= index < len(self.queue_items):
            name = display_name_for_queue_item(self.queue_items[index])
            self.queue_tree.item(iid, values=(name, status))

    def _start(self) -> None:
        if self.queue_running:
            return
        if not self.queue_items:
            messagebox.showerror("Empty queue", "Add at least one file to the queue.")
            return
        missing = [
            p for p in self.queue_items if not is_url(p) and not Path(p).exists()
        ]
        if missing:
            messagebox.showerror(
                "File not found", "These files do not exist:\n" + "\n".join(missing)
            )
            return
        output_formats = self._selected_output_formats()
        if not output_formats:
            messagebox.showerror("Missing output format", "Choose at least one output format.")
            return

        output_dir_text = self.output_dir.get().strip()
        output_dir_override: str | None = output_dir_text or None
        if output_dir_override:
            try:
                Path(output_dir_override).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("Cannot create output folder", str(exc))
                return

        for index in range(len(self.queue_items)):
            self._set_item_status(index, "Pending")

        device = RUN_ON_TO_DEVICE.get(self.run_on.get(), "auto")
        config = {
            "model": self.model.get(),
            "language": parse_language_value(self.language.get()),
            "script": self.script.get(),
            "device": device,
            "compute_type": self.compute_type.get(),
            "vad": self.vad.get(),
            "output_formats": output_formats,
            "extract_audio": self.extract_audio.get(),
            "initial_prompt": self.prompt_text.get("1.0", tk.END),
            "low_vram": self.low_vram.get(),
            "accurate_timing": self.accurate_timing.get(),
            "beam_size": parse_int_option(self.beam_size.get(), 1),
            "word_timestamps": self.word_timestamps.get(),
            "media_chunk_seconds": parse_chunk_option(self.media_chunk.get()),
            "media_chunk_overlap_seconds": 3,
            "no_cpu_fallback": self.gpu_only.get() and device != "cpu",
            "task": parse_task_value(self.task.get()),
            "delete_downloads_after": self.delete_downloads_after.get(),
        }

        self._clear_log()
        self.cancel_requested = False
        self._set_running(True)

        thread = threading.Thread(
            target=self._run_queue,
            args=(list(self.queue_items), config, output_dir_override),
            daemon=True,
        )
        thread.start()

    def _download_url(self, url: str, dest_dir: Path) -> Path | None:
        try:
            from yt_dlp import YoutubeDL
            from yt_dlp.utils import DownloadError
        except ImportError:
            self.log_queue.put((
                "line",
                "yt-dlp is not installed. Run:\n"
                "    .venv\\Scripts\\pip install yt-dlp\n",
            ))
            return None

        dest_dir.mkdir(parents=True, exist_ok=True)
        log_q = self.log_queue
        cancel_flag = lambda: self.cancel_requested  # noqa: E731
        ansi_re = re.compile(r"\x1b\[[0-9;]*[mGKHF]")

        def clean(text: str) -> str:
            return ansi_re.sub("", text or "")

        class _GuiLogger:
            def debug(self, msg: str) -> None: return
            def info(self, msg: str) -> None:
                if msg:
                    log_q.put(("line", clean(msg) + "\n"))
            def warning(self, msg: str) -> None:
                log_q.put(("line", f"[yt-dlp] {clean(msg)}\n"))
            def error(self, msg: str) -> None:
                log_q.put(("line", f"[yt-dlp error] {clean(msg)}\n"))

        last_bucket: list[int] = [-1]

        def _hook(d: dict) -> None:
            if cancel_flag():
                raise DownloadCancelled()
            if d.get("status") == "downloading":
                downloaded = d.get("downloaded_bytes") or 0
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                if total <= 0:
                    return
                bucket = int(downloaded * 100 / total / 5)
                if bucket == last_bucket[0]:
                    return
                last_bucket[0] = bucket
                pct = bucket * 5
                speed = clean(d.get("_speed_str") or "").strip()
                eta = clean(d.get("_eta_str") or "").strip()
                log_q.put((
                    "line",
                    f"Downloading audio: {pct:3d}%  speed {speed}  ETA {eta}\n",
                ))
            elif d.get("status") == "finished":
                log_q.put(("line", "Download finished, finalising file...\n"))

        opts = {
            "format": "bestaudio",
            "outtmpl": str(dest_dir / "%(title)s-%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": False,
            "no_color": True,
            "logger": _GuiLogger(),
            "progress_hooks": [_hook],
        }

        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filepath = ydl.prepare_filename(info)
        except DownloadCancelled:
            log_q.put(("line", "Download cancelled.\n"))
            return None
        except DownloadError as exc:
            log_q.put(("line", f"Download failed: {exc}\n"))
            return None
        except Exception as exc:
            log_q.put(("line", f"Download error: {exc}\n"))
            return None

        downloaded = Path(filepath)
        if not downloaded.exists():
            log_q.put(("line", f"Expected file not found: {downloaded}\n"))
            return None
        return downloaded

    def _run_queue(
        self,
        paths: list[str],
        config: dict,
        output_dir_override: str | None,
    ) -> None:
        python_exe = find_python_executable()
        total = len(paths)
        delete_downloads_after = config.pop("delete_downloads_after", False)
        if output_dir_override:
            download_dir = Path(output_dir_override) / "downloads"
        else:
            download_dir = project_root() / "downloads"

        for index, input_entry in enumerate(paths):
            if self.cancel_requested:
                self.log_queue.put(("item_status", (index, "Cancelled")))
                continue

            downloaded_path: Path | None = None
            if is_url(input_entry):
                self.log_queue.put(("item_status", (index, "Downloading")))
                self.log_queue.put(("reset_progress", None))
                self.log_queue.put((
                    "line",
                    f"\n=== [{index + 1}/{total}] Downloading {input_entry} ===\n",
                ))
                downloaded = self._download_url(input_entry, download_dir)
                if downloaded is None:
                    status = "Cancelled" if self.cancel_requested else "Download failed"
                    self.log_queue.put(("item_status", (index, status)))
                    continue
                downloaded_path = downloaded
                input_path = str(downloaded)
                self.log_queue.put(("line", f"Saved to: {input_path}\n"))
            else:
                input_path = input_entry

            if output_dir_override:
                output_file = str(
                    Path(output_dir_override) / f"{Path(input_path).stem}.srt"
                )
            else:
                output_file = str(default_output_file_for(input_path))
            output_dir, _basename, _srt_path = split_output_target(output_file)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            command = build_transcribe_command(
                python_exe=python_exe,
                input_path=input_path,
                output_file=output_file,
                **config,
            )

            self.log_queue.put(("item_status", (index, "Running")))
            self.log_queue.put(("reset_progress", None))
            self.log_queue.put(
                ("line", f"\n=== [{index + 1}/{total}] {Path(input_path).name} ===\n")
            )
            self.log_queue.put(("line", "Command:\n" + printable_command(command) + "\n\n"))

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
                self.log_queue.put(("item_status", (index, "Error")))
                continue

            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.log_queue.put(("line", line))

            return_code = self.process.wait()
            self.process = None

            if return_code == 0:
                status = "Done"
            elif self.cancel_requested:
                status = "Cancelled"
            else:
                status = f"Error ({return_code})"
            self.log_queue.put(("item_status", (index, status)))
            self.log_queue.put(
                ("line", f"\n--- {Path(input_path).name}: {status} ---\n")
            )

            if (
                return_code == 0
                and delete_downloads_after
                and downloaded_path is not None
            ):
                try:
                    downloaded_path.unlink()
                    self.log_queue.put(
                        ("line", f"Deleted downloaded audio: {downloaded_path}\n")
                    )
                except OSError as exc:
                    self.log_queue.put(
                        ("line", f"Could not delete {downloaded_path}: {exc}\n")
                    )

        self.log_queue.put(("queue_done", None))

    def _cancel(self) -> None:
        self.cancel_requested = True
        proc = self.process
        if proc is not None and proc.poll() is None:
            self._append_log("\nCancelling current transcription...\n")
            try:
                proc.terminate()
            except OSError:
                pass
        self.cancel_button.configure(state=tk.DISABLED)

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    line = str(payload)
                    if not self._handle_progress_line(line):
                        self._append_log(line)
                elif kind == "item_status":
                    assert isinstance(payload, tuple)
                    index, status = payload
                    self._set_item_status(int(index), str(status))
                elif kind == "reset_progress":
                    self._reset_progress()
                elif kind == "queue_done":
                    self._set_running(False)
                    if self.cancel_requested:
                        self.status.set("Cancelled")
                        self._append_log("\nQueue cancelled.\n")
                    else:
                        self.status.set("Done")
                        self.progress.configure(value=100)
                        self.progress_text.set("Queue complete")
                        self._append_log("\nQueue finished.\n")
                    self.process = None
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def _reset_progress(self) -> None:
        self.duration_seconds = 0.0
        self.last_progress_seconds = 0.0
        self.progress_started_at = 0.0
        self.progress.configure(value=0)
        self.progress_text.set("Starting...")

    def _set_running(self, running: bool) -> None:
        self.queue_running = running
        if running:
            self.status.set("Running")
            self._reset_progress()
            self.start_button.configure(state=tk.DISABLED)
            self.cancel_button.configure(state=tk.NORMAL)
            self.add_files_button.configure(state=tk.DISABLED)
            self.remove_button.configure(state=tk.DISABLED)
            self.clear_queue_button.configure(state=tk.DISABLED)
            self.output_dir_entry.configure(state=tk.DISABLED)
            self.output_dir_browse_button.configure(state=tk.DISABLED)
        else:
            self.start_button.configure(state=tk.NORMAL)
            self.cancel_button.configure(state=tk.DISABLED)
            self.add_files_button.configure(state=tk.NORMAL)
            self.remove_button.configure(state=tk.NORMAL)
            self.clear_queue_button.configure(state=tk.NORMAL)
            self.output_dir_entry.configure(state=tk.NORMAL)
            self.output_dir_browse_button.configure(state=tk.NORMAL)

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
        override = self.output_dir.get().strip()
        if override:
            path = Path(override)
        elif self.queue_items:
            path = default_output_file_for(self.queue_items[0]).parent
        else:
            path = project_root() / "subtitles"
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


def is_url(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def display_name_for_queue_item(item: str) -> str:
    if is_url(item):
        return item
    return Path(item).name


class DownloadCancelled(Exception):
    """Raised inside a yt-dlp progress hook to abort a download."""


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
