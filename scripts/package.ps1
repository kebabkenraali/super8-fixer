# Builds the release folder Super8Fixer\ and the zip dist\Super8Fixer-<version>-win64.zip.
# Usage (from the repository root): powershell -ExecutionPolicy Bypass -File scripts\package.ps1 [-NoBuild] [-Out folder]
param([switch]$NoBuild, [string]$Out = "")

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$version = (Select-String -Path Cargo.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $root "target" }

if (-not $NoBuild) { cargo build --release; if ($LASTEXITCODE) { throw "cargo build failed" } }

$out = if ($Out) { $Out } else { Join-Path $root "Super8Fixer" }
New-Item -ItemType Directory -Force $out, "$out\models" | Out-Null
Copy-Item "$target\release\Super8Fixer.exe", "$target\release\DirectML.dll" $out -Force
Copy-Item "models\slip_classifier.onnx", "models\slip_classifier.json" "$out\models\" -Force

# Visual C++ runtime from the Visual Studio redistributable folder (app-local deployment)
$crt = Get-ChildItem "C:\Program Files\Microsoft Visual Studio", "C:\Program Files (x86)\Microsoft Visual Studio" -Recurse -Directory `
    -Filter "Microsoft.VC*.CRT" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match "\\Redist\\MSVC\\[^\\]+\\x64\\" } |
    Sort-Object FullName -Descending | Select-Object -First 1
if ($crt) {
    foreach ($f in "msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll") { Copy-Item (Join-Path $crt.FullName $f) $out -Force }
} else {
    Write-Warning "Visual C++ redistributable folder not found: users need the Visual C++ Redistributable (x64) installed."
}

Copy-Item "scripts\get-ffmpeg.ps1", "scripts\get-ffmpeg.cmd" $out -Force
Copy-Item "README.md", "LICENSE.md", "MODEL_CARD.md", "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_LICENSES.txt" $out -Force
New-Item -ItemType Directory -Force "$out\licenses" | Out-Null
Copy-Item "licenses\*" "$out\licenses\" -Force

# the zip never contains ffmpeg, settings or anything else that was put into the folder while testing
New-Item -ItemType Directory -Force "dist" | Out-Null
$zip = Join-Path $root "dist\Super8Fixer-$version-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
$stage = Join-Path ([System.IO.Path]::GetTempPath()) ("s8pkg-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force "$stage\Super8Fixer" | Out-Null
Get-ChildItem $out -Exclude "ffmpeg", "super8fixer_settings.json", "*_personal.onnx" | Copy-Item -Destination "$stage\Super8Fixer" -Recurse -Force
Get-ChildItem "$stage\Super8Fixer\models" -Filter "*_personal.onnx" -ErrorAction SilentlyContinue | Remove-Item
Compress-Archive -Path "$stage\Super8Fixer" -DestinationPath $zip
Remove-Item $stage -Recurse -Force
Write-Host "Folder: $out"
Write-Host "Zip:    $zip"
