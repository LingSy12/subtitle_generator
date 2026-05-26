# Chinese Subtitle Generator

Local command-line software for recognizing speech in long videos and generating
Chinese subtitle timing files.

It creates:

- `.srt` subtitle file, widely supported by players and platforms
- `.vtt` subtitle file, used by web players
- `.sbv`, `.ttml`, `.ass`, `.lrc`, and `.csv` timed exports
- `.txt` plain transcript
- optional `.jsonl` segment log for checking timings

The transcriber uses [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
locally, so multi-hour videos do not need to be uploaded to another service.

## 1. Install

Open PowerShell in this folder:

```powershell
cd D:\youtubesubtitle
.\setup.ps1
```

If Windows blocks PowerShell scripts, use:

```powershell
.\setup.bat
```

The first real transcription will download the selected Whisper model. For long
Chinese videos, start with `medium`; use `large-v3` if you want better accuracy
and have enough disk/RAM/GPU.

## 2. Use The GUI

```powershell
.\gui.bat
```

Then choose your video, choose the output base file name, tick the subtitle
formats you want, pick a model, choose **CPU** or **GPU (NVIDIA CUDA)**, and
click **Generate Subtitles**.
For smaller NVIDIA GPUs, tick **Limit GPU VRAM** before starting.
If the generated subtitles drift or do not line up with the video, tick
**Accurate timing** and regenerate.

You can also paste a YouTube (or other supported site) URL: click **Add URL**,
paste the link, and the GUI downloads the audio with
[`yt-dlp`](https://github.com/yt-dlp/yt-dlp) before transcribing. The audio
file is saved under a `downloads/` folder next to your output (or under
`D:\youtubesubtitle\downloads\` if no output folder is set). Tick
**Delete URL audio after** if you only want the subtitle file kept. Public and
unlisted videos work without sign-in; private videos are not supported.

The GUI runs the local Python transcriber on this computer. It shows a progress
bar with estimated time left during transcription and writes the selected output
formats next to the chosen base path. The first model use may need internet to
download the model; after that, transcription can run locally from the cached
model. You can also type a local faster-whisper model folder into the Model box.
If GPU/CUDA is selected but CUDA libraries are missing, the app retries locally
on CPU automatically.
Final subtitle files are replaced only after a successful run; interrupted runs
leave `__partial` files behind.

To check whether GPU support is visible to the app:

```powershell
.\check_gpu.bat
```

## 3. Generate Chinese subtitles from command line

```powershell
.\run.ps1 "D:\Videos\my-long-video.mp4"
```

Or:

```powershell
.\run.bat "D:\Videos\my-long-video.mp4"
```

Output files are written to `D:\youtubesubtitle\subtitles`.

Example with a stronger model:

```powershell
.\run.ps1 "D:\Videos\lecture.mp4" -Model large-v3
```

Example with more subtitle formats:

```powershell
.\run.ps1 "D:\Videos\lecture.mp4" -Formats "srt,vtt,sbv,ass,ttml,lrc,csv,txt,jsonl"
```

Traditional Chinese output:

```powershell
.\run.ps1 "D:\Videos\lecture.mp4" -Script traditional
```

Example using GPU:

```powershell
.\run.ps1 "D:\Videos\lecture.mp4" -Model large-v3 -Device cuda -ComputeType float16
```

If CUDA runs out of memory, use low-VRAM GPU mode. This keeps the job on the
GPU but uses quantized compute, beam size 1, and shorter 15-second chunks:

```powershell
.\run.ps1 "D:\Videos\lecture.mp4" -Model medium -Device cuda -LowVram
```

If subtitle timing does not match the video, regenerate in accurate timing mode.
This disables VAD silence skipping so subtitle times stay on the original video
timeline:

```powershell
.\run.ps1 "D:\Videos\lecture.mp4" -Model medium -Device cuda -LowVram -AccurateTiming
```

CUDA accurate-timing runs first decode the media into a temporary continuous
16 kHz audio cache, then process it in isolated 120-second worker processes
with 2 seconds of overlap. This keeps subtitle timestamps on Whisper's actual
audio timeline and releases GPU memory after every chunk. If your GPU is still
tight, lower the chunk size:

```powershell
.\run.ps1 "D:\Videos\lecture.mp4" -Model medium -Device cuda -LowVram -AccurateTiming -MediaChunkSeconds 60
```

For a fixed offset, shift all subtitles. Positive values delay subtitles;
negative values show them earlier:

```powershell
.\run.ps1 "D:\Videos\lecture.mp4" -TimeOffset 1.25
```

Example direct Python command:

```powershell
python -m ytsubtitle "D:\Videos\lecture.mp4" --language zh --model medium
```

## Notes For Long Videos

- `medium` is usually a good first pass on CPU.
- `large-v3` is more accurate but much slower and larger.
- Use `--vad` to skip silence and speed up some recordings.
- If subtitle timings do not line up, use `--accurate-timing`. This is slower
  because it keeps the original audio timeline instead of relying on VAD silence
  removal.
- `--word-timestamps` can make cue boundaries tighter, but it is heavier on GPU
  memory. Leave it off for long videos unless you specifically need it.
- Use `--low-vram` with `--device cuda` when GPU memory is tight. You can also
  set `--beam-size 1` or `--chunk-length 15` directly.
- Use `--time-offset 1.25` or `--time-offset -1.25` when all captions are off
  by the same amount.
- After a CUDA out-of-memory error, the CLI automatically retries once in
  low-VRAM GPU mode before falling back to CPU. Add `--no-cpu-fallback` if you
  want the run to fail instead of switching to CPU.
- If the video is noisy, add a custom prompt:

```powershell
python -m ytsubtitle "D:\Videos\talk.mp4" --initial-prompt "Chinese interview subtitles. Output Simplified Chinese. Keep names and brand names."
```

## Optional FFmpeg

The tool can work without system FFmpeg because `faster-whisper` uses PyAV for
media decoding. If you install FFmpeg and add it to PATH, you can pre-extract
16 kHz mono audio:

```powershell
python -m ytsubtitle "D:\Videos\lecture.mp4" --extract-audio
```

This can make repeated transcription runs more predictable for very large files.
