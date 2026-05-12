param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputPath,

    [string]$OutputDir = "subtitles",
    [string]$Model = "medium",
    [string]$Language = "zh",
    [ValidateSet("none", "simplified", "traditional")]
    [string]$Script = "none",
    [string]$Device = "auto",
    [string]$ComputeType = "auto",
    [string]$Formats = "srt,vtt,txt",
    [switch]$Vad,
    [switch]$Jsonl,
    [switch]$ExtractAudio,
    [switch]$LowVram,
    [int]$ChunkLength = 0,
    [switch]$NoCpuFallback,
    [switch]$WordTimestamps,
    [switch]$AccurateTiming,
    [double]$TimeOffset = 0,
    [int]$MediaChunkSeconds = 0,
    [int]$MediaChunkOverlapSeconds = 2
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$python = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $python = ".\.venv\Scripts\python.exe"
}

$formatList = @(
    $Formats.Split(",") |
        ForEach-Object { $_.Trim().ToLowerInvariant() } |
        Where-Object { $_ }
)
if ($Jsonl -and -not ($formatList -contains "jsonl")) {
    $formatList += "jsonl"
}
$formats = $formatList -join ","

$argsList = @(
    "-m", "ytsubtitle",
    $InputPath,
    "--output-dir", $OutputDir,
    "--model", $Model,
    "--language", $Language,
    "--script", $Script,
    "--device", $Device,
    "--compute-type", $ComputeType,
    "--formats", $formats
)

if ($Vad) {
    $argsList += "--vad"
}

if ($ExtractAudio) {
    $argsList += "--extract-audio"
}

if ($LowVram) {
    $argsList += "--low-vram"
}

if ($ChunkLength -gt 0) {
    $argsList += @("--chunk-length", $ChunkLength.ToString())
}

if ($NoCpuFallback) {
    $argsList += "--no-cpu-fallback"
}

if ($WordTimestamps) {
    $argsList += "--word-timestamps"
}

if ($AccurateTiming) {
    $argsList += "--accurate-timing"
}

if ($TimeOffset -ne 0) {
    $argsList += @("--time-offset", $TimeOffset.ToString([Globalization.CultureInfo]::InvariantCulture))
}

if ($MediaChunkSeconds -gt 0) {
    $argsList += @("--media-chunk-seconds", $MediaChunkSeconds.ToString())
}

if ($MediaChunkOverlapSeconds -ne 2) {
    $argsList += @("--media-chunk-overlap-seconds", $MediaChunkOverlapSeconds.ToString())
}

& $python @argsList
