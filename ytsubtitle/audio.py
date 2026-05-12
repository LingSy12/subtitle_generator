from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_available(ffmpeg_bin: str = "ffmpeg") -> bool:
    return shutil.which(ffmpeg_bin) is not None or Path(ffmpeg_bin).exists()


def extract_audio(
    input_path: Path,
    output_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path

    if not ffmpeg_available(ffmpeg_bin):
        raise RuntimeError(
            f"FFmpeg was not found: {ffmpeg_bin}. Install FFmpeg or omit --extract-audio."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"FFmpeg audio extraction failed.\n{stderr}")
    return output_path
