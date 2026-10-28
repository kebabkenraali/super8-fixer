"""Shared helpers for the training tools: finding ffmpeg and reading basic video information."""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def find_ffmpeg(given=None):
    """ffmpeg from --ffmpeg, PATH, imageio-ffmpeg, or the ffmpeg folder of the application."""
    if given:
        return given
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    for p in (os.path.join(HERE, "..", "Super8Fixer", "ffmpeg", "bin", "ffmpeg.exe"), r"C:\ffmpeg\bin\ffmpeg.exe"):
        if os.path.exists(p):
            return os.path.abspath(p)
    sys.exit("ffmpeg not found; give the path: --ffmpeg C:\\path\\ffmpeg.exe")


def video_info(ff, src):
    """(width, height, fps, seconds, has_audio)"""
    r = subprocess.run([ff, "-hide_banner", "-i", src], capture_output=True, text=True, errors="replace")
    txt = r.stderr
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", txt)
    fps = re.search(r"(\d+(?:\.\d+)?) fps", txt)
    dur = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", txt)
    if not m:
        sys.exit("No video stream found: " + src)
    w, h = int(m.group(1)), int(m.group(2))
    f = float(fps.group(1)) if fps else 25.0
    secs = int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)) if dur else 0
    return w, h, f, secs, ("Audio:" in txt)


def seconds(s):
    """'mm:ss.ss' -> seconds"""
    return sum(float(v) * 60 ** k for k, v in enumerate(reversed(s.split(":"))))
