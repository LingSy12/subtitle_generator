$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$python = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $python = ".\.venv\Scripts\python.exe"
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

& $python -c "from ytsubtitle.transcriber import configure_cuda_runtime_path; paths = configure_cuda_runtime_path(); import ctranslate2; print('CUDA runtime paths:', [str(p) for p in paths]); print('CUDA devices:', ctranslate2.get_cuda_device_count()); print('CUDA compute types:', sorted(ctranslate2.get_supported_compute_types('cuda')))"
