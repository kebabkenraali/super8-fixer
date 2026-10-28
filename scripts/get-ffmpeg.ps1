# Downloads an ffmpeg build (BtbN FFmpeg-Builds, GPL, win64) into ffmpeg\bin next to Super8Fixer.exe.
# Usage: powershell -ExecutionPolicy Bypass -File get-ffmpeg.ps1 [target folder]
param([string]$Target = $PSScriptRoot)

$ErrorActionPreference = "Stop"
$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$bin = Join-Path $Target "ffmpeg\bin"
if ((Test-Path (Join-Path $bin "ffmpeg.exe")) -and (Test-Path (Join-Path $bin "ffprobe.exe"))) {
    Write-Host "ffmpeg is already in $bin"
    exit 0
}
Write-Host "Super8 Fixer needs ffmpeg to read and write video."
Write-Host "Downloading $url"
Write-Host "(about 200 MB; FFmpeg is licensed under the GPL, see https://ffmpeg.org/legal.html)"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("super8fixer-ffmpeg-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force $tmp | Out-Null
try {
    $zip = Join-Path $tmp "ffmpeg.zip"
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $src = Get-ChildItem $tmp -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    if (-not $src) { throw "ffmpeg.exe was not found in the download" }
    New-Item -ItemType Directory -Force $bin | Out-Null
    foreach ($f in "ffmpeg.exe", "ffprobe.exe") { Copy-Item (Join-Path $src.DirectoryName $f) $bin -Force }
    $lic = Get-ChildItem $tmp -Recurse -Filter LICENSE.txt | Select-Object -First 1
    if ($lic) { Copy-Item $lic.FullName (Join-Path $Target "ffmpeg\LICENSE.txt") -Force }
    Write-Host "Done: $bin"
}
finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
