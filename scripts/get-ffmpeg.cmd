@echo off
rem Downloads ffmpeg for Super8 Fixer into ffmpeg\bin next to this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0get-ffmpeg.ps1" "%~dp0."
pause
