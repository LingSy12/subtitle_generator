from __future__ import annotations

import json
import re
import csv
from html import escape as escape_xml
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_CJK = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_BREAK_AFTER = set("，。！？；：、,.!?;:")
_BREAK_BEFORE = set("([{（【《“\"'")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: tuple = ()  # tuple[Word, ...]; empty if word_timestamps were off
    no_speech_prob: float = 0.0


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text)
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    text = re.sub(rf"([，。！？；：、])\s+(?=[{_CJK}])", r"\1", text)
    text = re.sub(r"([（【《“])\s+", r"\1", text)
    text = re.sub(r"\s+([）】》”])", r"\1", text)
    return text.strip()


def convert_chinese_script(text: str, script: str) -> str:
    if script == "none":
        return text

    converter = _get_opencc_converter(script)
    return converter.convert(text)


@lru_cache(maxsize=2)
def _get_opencc_converter(script: str):
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise RuntimeError(
            "Chinese script conversion requires opencc-python-reimplemented. "
            "Run setup.ps1 or use --script none."
        ) from exc

    return OpenCC("t2s" if script == "simplified" else "s2t")


def format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02}:{minutes:02}:{seconds_part:02},{ms:03}"


def format_vtt_time(seconds: float) -> str:
    return format_srt_time(seconds).replace(",", ".")


def format_sbv_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours}:{minutes:02}:{seconds_part:02}.{ms:03}"


def format_ass_time(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    cs = total_cs % 100
    total_seconds = total_cs // 100
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours}:{minutes:02}:{seconds_part:02}.{cs:02}"


def format_lrc_time(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    cs = total_cs % 100
    total_seconds = total_cs // 100
    seconds_part = total_seconds % 60
    minutes = total_seconds // 60
    return f"{minutes:02}:{seconds_part:02}.{cs:02}"


def wrap_text(text: str, max_line_chars: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    lines: list[str] = []
    rest = text
    while len(rest) > max_line_chars:
        split_at = _find_split(rest, max_line_chars)
        line = rest[:split_at].strip()
        if line:
            lines.append(line)
        rest = rest[split_at:].strip()

    if rest:
        lines.append(rest)
    return lines


def _find_split(text: str, max_line_chars: int) -> int:
    search_end = min(len(text), max_line_chars + 1)
    min_good = max(1, int(max_line_chars * 0.55))

    for i in range(search_end - 1, min_good - 1, -1):
        char = text[i - 1]
        if char in _BREAK_AFTER:
            return i
        if char.isspace():
            return i
        if text[i] in _BREAK_BEFORE:
            return i

    return max_line_chars


_SENTENCE_PUNCT = set("。！？.!?")


def segment_to_cues(
    segment: Segment,
    max_line_chars: int,
    max_lines: int,
    min_cue_duration: float,
    gap_split_seconds: float = 0.7,
) -> list[Cue]:
    """Build cues from a transcribed segment.

    Prefers word-level timestamps (set via faster-whisper `word_timestamps=True`)
    and groups them into cues using natural pauses, sentence punctuation, and
    line-length limits — like CapCut's auto-subtitle feature. Falls back to the
    older proportional-split logic when no per-word data is available.
    """
    if segment.words:
        cues = _cues_from_words(
            segment.words,
            max_line_chars=max_line_chars,
            max_lines=max_lines,
            min_cue_duration=min_cue_duration,
            gap_split_seconds=gap_split_seconds,
        )
        if cues:
            return cues
        # If word data was empty for some reason fall through to legacy path.

    return _cues_from_segment_text(
        segment,
        max_line_chars=max_line_chars,
        max_lines=max_lines,
        min_cue_duration=min_cue_duration,
    )


def _cues_from_words(
    words: tuple,
    max_line_chars: int,
    max_lines: int,
    min_cue_duration: float,
    gap_split_seconds: float,
) -> list[Cue]:
    cleaned: list[Word] = []
    for w in words:
        text = (w.text or "").strip()
        if not text:
            continue
        cleaned.append(w)
    if not cleaned:
        return []

    max_chars = max(1, max_line_chars * max(1, max_lines))
    cues: list[Cue] = []
    current: list[Word] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current_chars
        if not current:
            return
        raw_text = "".join(w.text for w in current)
        text = normalize_text(raw_text)
        if not text:
            current.clear()
            current_chars = 0
            return
        lines = wrap_text(text, max_line_chars)
        if not lines:
            current.clear()
            current_chars = 0
            return
        cue_text = "\n".join(lines[:max_lines])
        start = max(0.0, float(current[0].start))
        # last_end is taken straight from the final word's end-time — this is
        # the whole point of the word-based path: end stays tight to actual
        # speech instead of running past it into silence.
        last_end = max(start, float(current[-1].end))
        if last_end - start < min_cue_duration:
            last_end = start + min_cue_duration
        cues.append(Cue(start=start, end=last_end, text=cue_text))
        current.clear()
        current_chars = 0

    for index, word in enumerate(cleaned):
        word_chars = len(word.text.strip())
        # If adding this word would exceed the per-cue char budget, flush first
        # so the new word starts a fresh cue.
        if current and current_chars + word_chars > max_chars:
            flush()
        current.append(word)
        current_chars += word_chars

        if index + 1 < len(cleaned):
            next_word = cleaned[index + 1]
            gap = float(next_word.start) - float(word.end)
        else:
            gap = float("inf")

        ends_with_sentence_punct = bool(word.text) and word.text.strip()[-1:] in _SENTENCE_PUNCT
        if gap >= gap_split_seconds or ends_with_sentence_punct:
            flush()

    flush()
    return cues


def _cues_from_segment_text(
    segment: Segment,
    max_line_chars: int,
    max_lines: int,
    min_cue_duration: float,
) -> list[Cue]:
    text = normalize_text(segment.text)
    if not text:
        return []

    lines = wrap_text(text, max_line_chars)
    cue_texts = [
        "\n".join(lines[index : index + max_lines])
        for index in range(0, len(lines), max_lines)
    ]

    start = max(0.0, segment.start)
    end = max(start + min_cue_duration, segment.end)
    duration = end - start

    if len(cue_texts) == 1:
        return [Cue(start=start, end=end, text=cue_texts[0])]

    weights = [max(1, len(text.replace("\n", ""))) for text in cue_texts]
    total_weight = sum(weights)
    cues: list[Cue] = []
    current_start = start
    elapsed_weight = 0

    for index, cue_text in enumerate(cue_texts):
        elapsed_weight += weights[index]
        if index == len(cue_texts) - 1:
            current_end = end
        else:
            current_end = start + duration * (elapsed_weight / total_weight)
            current_end = min(current_end, end)

        if current_end <= current_start:
            current_end = min(end, current_start + min_cue_duration)

        cues.append(Cue(start=current_start, end=current_end, text=cue_text))
        current_start = current_end

    return cues


class SubtitleOutputs:
    def __init__(self, output_base: Path, formats: Iterable[str]) -> None:
        self.output_base = output_base
        self.formats = set(formats)
        self._files = {}
        self._csv_writer = None
        self._cue_index = 1

        output_base.parent.mkdir(parents=True, exist_ok=True)
        if "srt" in self.formats:
            self._files["srt"] = (output_base.with_suffix(".srt")).open(
                "w", encoding="utf-8-sig", newline="\n"
            )
        if "vtt" in self.formats:
            file = (output_base.with_suffix(".vtt")).open(
                "w", encoding="utf-8", newline="\n"
            )
            file.write("WEBVTT\n\n")
            self._files["vtt"] = file
        if "sbv" in self.formats:
            self._files["sbv"] = (output_base.with_suffix(".sbv")).open(
                "w", encoding="utf-8", newline="\n"
            )
        if "ass" in self.formats:
            file = (output_base.with_suffix(".ass")).open(
                "w", encoding="utf-8-sig", newline="\n"
            )
            file.write(ass_header())
            self._files["ass"] = file
        if "ttml" in self.formats:
            file = (output_base.with_suffix(".ttml")).open(
                "w", encoding="utf-8", newline="\n"
            )
            file.write(ttml_header())
            self._files["ttml"] = file
        if "lrc" in self.formats:
            self._files["lrc"] = (output_base.with_suffix(".lrc")).open(
                "w", encoding="utf-8-sig", newline="\n"
            )
        if "csv" in self.formats:
            file = (output_base.with_suffix(".csv")).open(
                "w", encoding="utf-8-sig", newline=""
            )
            self._files["csv"] = file
            self._csv_writer = csv.writer(file)
            self._csv_writer.writerow(["index", "start", "end", "text"])
        if "txt" in self.formats:
            self._files["txt"] = (output_base.with_suffix(".txt")).open(
                "w", encoding="utf-8", newline="\n"
            )
        if "jsonl" in self.formats:
            self._files["jsonl"] = (output_base.with_suffix(".segments.jsonl")).open(
                "w", encoding="utf-8", newline="\n"
            )

    @property
    def paths(self) -> list[Path]:
        paths: list[Path] = []
        if "srt" in self.formats:
            paths.append(self.output_base.with_suffix(".srt"))
        if "vtt" in self.formats:
            paths.append(self.output_base.with_suffix(".vtt"))
        if "sbv" in self.formats:
            paths.append(self.output_base.with_suffix(".sbv"))
        if "ass" in self.formats:
            paths.append(self.output_base.with_suffix(".ass"))
        if "ttml" in self.formats:
            paths.append(self.output_base.with_suffix(".ttml"))
        if "lrc" in self.formats:
            paths.append(self.output_base.with_suffix(".lrc"))
        if "csv" in self.formats:
            paths.append(self.output_base.with_suffix(".csv"))
        if "txt" in self.formats:
            paths.append(self.output_base.with_suffix(".txt"))
        if "jsonl" in self.formats:
            paths.append(self.output_base.with_suffix(".segments.jsonl"))
        return paths

    def write_cue(self, cue: Cue) -> None:
        if "srt" in self._files:
            self._files["srt"].write(
                f"{self._cue_index}\n"
                f"{format_srt_time(cue.start)} --> {format_srt_time(cue.end)}\n"
                f"{cue.text}\n\n"
            )
        if "vtt" in self._files:
            self._files["vtt"].write(
                f"{format_vtt_time(cue.start)} --> {format_vtt_time(cue.end)}\n"
                f"{cue.text}\n\n"
            )
        if "sbv" in self._files:
            self._files["sbv"].write(
                f"{format_sbv_time(cue.start)},{format_sbv_time(cue.end)}\n"
                f"{cue.text}\n\n"
            )
        if "ass" in self._files:
            ass_text = cue.text.replace("\n", r"\N")
            self._files["ass"].write(
                "Dialogue: 0,"
                f"{format_ass_time(cue.start)},{format_ass_time(cue.end)},"
                f"Default,,0,0,0,,{ass_text}\n"
            )
        if "ttml" in self._files:
            ttml_text = "<br/>".join(escape_xml(line) for line in cue.text.splitlines())
            self._files["ttml"].write(
                f'      <p begin="{format_vtt_time(cue.start)}" '
                f'end="{format_vtt_time(cue.end)}">{ttml_text}</p>\n'
            )
        if "lrc" in self._files:
            one_line = cue.text.replace("\n", "")
            self._files["lrc"].write(f"[{format_lrc_time(cue.start)}]{one_line}\n")
        if self._csv_writer is not None:
            self._csv_writer.writerow(
                [
                    self._cue_index,
                    format_vtt_time(cue.start),
                    format_vtt_time(cue.end),
                    cue.text.replace("\n", "\\n"),
                ]
            )
        if "txt" in self._files:
            one_line = cue.text.replace("\n", "")
            self._files["txt"].write(f"{one_line}\n")
        if "jsonl" in self._files:
            self._files["jsonl"].write(
                json.dumps(asdict(cue), ensure_ascii=False) + "\n"
            )
        self._cue_index += 1

    def close(self) -> None:
        if "ttml" in self._files:
            self._files["ttml"].write(ttml_footer())
        for file in self._files.values():
            file.close()

    def __enter__(self) -> "SubtitleOutputs":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def ass_header() -> str:
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.601\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Microsoft YaHei,42,&H00FFFFFF,&H000000FF,&H00000000,"
        "&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,48,48,36,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )


def ttml_header() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<tt xmlns="http://www.w3.org/ns/ttml">\n'
        "  <body>\n"
        "    <div>\n"
    )


def ttml_footer() -> str:
    return "    </div>\n  </body>\n</tt>\n"
