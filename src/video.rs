//! Video input and output through ffmpeg (raw frames over pipes; audio is copied from the source).

use std::io::{Read, Write};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, anyhow, bail};

const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Application folder: the folder of the exe, holding models\ and ffmpeg\bin\. In a development
/// build (inside a target folder) the parent folders are searched for a Super8Fixer\ or models\ folder.
pub fn app_dir() -> PathBuf {
    static D: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        let exe = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())).unwrap_or_else(|| PathBuf::from("."));
        if !exe.components().any(|c| c.as_os_str().eq_ignore_ascii_case("target")) {
            return exe;
        }
        let mut d = exe.clone();
        for _ in 0..6 {
            if d.join("Super8Fixer").join("models").is_dir() {
                return d.join("Super8Fixer");
            }
            if d.join("models").is_dir() {
                return d;
            }
            match d.parent() {
                Some(p) => d = p.to_path_buf(),
                None => break,
            }
        }
        exe
    })
    .clone()
}

/// Finds a program: <app>\ffmpeg\bin, <app>\ffmpeg, <app>, then PATH.
pub fn program(name: &str) -> Result<PathBuf> {
    let d = app_dir();
    let file = format!("{name}.exe");
    for p in [d.join("ffmpeg").join("bin"), d.join("ffmpeg"), d.clone()] {
        let c = p.join(&file);
        if c.is_file() {
            return Ok(c);
        }
    }
    if let Ok(path) = std::env::var("PATH") {
        for p in std::env::split_paths(&path) {
            let c = p.join(&file);
            if c.is_file() {
                return Ok(c);
            }
        }
    }
    bail!(
        "{file} not found. Run get-ffmpeg.cmd in {} to download ffmpeg, or copy ffmpeg.exe and ffprobe.exe to {}",
        d.display(),
        d.join("ffmpeg").join("bin").display()
    )
}

fn command(name: &str) -> Result<Command> {
    let mut c = Command::new(program(name)?);
    c.creation_flags(CREATE_NO_WINDOW);
    Ok(c)
}

#[derive(Clone, Debug)]
pub struct VideoInfo {
    pub path: String,
    pub w: usize,
    pub h: usize,
    pub fps: f64,
    /// frame rate as a fraction for ffmpeg (-r), e.g. "25/1"
    pub fps_text: String,
    pub duration: f64,
    /// estimated number of frames
    pub n: usize,
    pub audio: bool,
    /// the picture uses the full 0-255 range (pc / yuvj); otherwise TV range 16-235
    pub full_range: bool,
}

impl VideoInfo {
    pub fn read(path: &str) -> Result<VideoInfo> {
        if !Path::new(path).is_file() {
            bail!("File not found: {path}");
        }
        let o = command("ffprobe")?
            .args(["-v", "error", "-print_format", "json", "-show_streams", "-show_format"])
            .arg(path)
            .output()
            .context("ffprobe did not start")?;
        if !o.status.success() {
            bail!("Cannot read the video: {path}\n{}", String::from_utf8_lossy(&o.stderr));
        }
        let j: serde_json::Value = serde_json::from_slice(&o.stdout)?;
        let streams = j["streams"].as_array().cloned().unwrap_or_default();
        let v = streams.iter().find(|s| s["codec_type"] == "video").ok_or_else(|| anyhow!("The file has no video stream: {path}"))?;
        let w = v["width"].as_u64().unwrap_or(0) as usize;
        let h = v["height"].as_u64().unwrap_or(0) as usize;
        let fraction = |s: &str| -> (u64, u64) {
            let mut o = s.split('/');
            (o.next().and_then(|x| x.parse().ok()).unwrap_or(0), o.next().and_then(|x| x.parse().ok()).unwrap_or(1).max(1))
        };
        let (mut a, mut b) = fraction(v["avg_frame_rate"].as_str().unwrap_or("0/1"));
        if a == 0 {
            (a, b) = fraction(v["r_frame_rate"].as_str().unwrap_or("25/1"));
        }
        if a == 0 {
            (a, b) = (25, 1);
        }
        let fps = a as f64 / b as f64;
        let mut duration: f64 = v["duration"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        if duration == 0.0 {
            duration = j["format"]["duration"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        }
        let n = v["nb_frames"].as_str().and_then(|s| s.parse().ok()).unwrap_or((duration * fps).round() as usize);
        let audio = streams.iter().any(|s| s["codec_type"] == "audio");
        let full_range = v["color_range"] == "pc" || v["pix_fmt"].as_str().is_some_and(|p| p.starts_with("yuvj"));
        if w == 0 || h == 0 {
            bail!("Could not determine the video size: {path}");
        }
        Ok(VideoInfo { path: path.to_string(), w, h, fps, fps_text: format!("{a}/{b}"), duration, n, audio, full_range })
    }
}

/// Raw frame reader: every frame is `size` bytes.
pub struct Reader {
    child: Child,
    out: ChildStdout,
    pub size: usize,
}

impl Reader {
    /// `filter`: ffmpeg -vf (e.g. "scale=64:540:flags=area"), `pix_fmt`: "rgb24", "gray" or "yuv420p".
    pub fn new(path: &str, filter: Option<&str>, pix_fmt: &str, size: usize, start_s: Option<f64>, max_frames: Option<usize>) -> Result<Reader> {
        let mut c = command("ffmpeg")?;
        c.args(["-v", "error"]);
        if let Some(a) = start_s {
            c.args(["-ss", &format!("{a:.4}")]);
        }
        c.arg("-i").arg(path);
        if let Some(n) = max_frames {
            c.args(["-frames:v", &n.to_string()]);
        }
        if let Some(s) = filter {
            c.args(["-vf", s]);
        }
        c.args(["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]);
        let mut child = c.stdout(Stdio::piped()).stderr(Stdio::null()).stdin(Stdio::null()).spawn().context("ffmpeg did not start")?;
        let out = child.stdout.take().unwrap();
        Ok(Reader { child, out, size })
    }

    /// Reads the next frame into `buf`; false when the video has ended.
    pub fn read(&mut self, buf: &mut [u8]) -> bool {
        self.out.read_exact(&mut buf[..self.size]).is_ok()
    }

    pub fn next_frame(&mut self) -> Option<Vec<u8>> {
        let mut b = vec![0u8; self.size];
        if self.read(&mut b) { Some(b) } else { None }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// x264 writer; audio is copied from the source when one is given.
pub struct Writer {
    child: Child,
    stdin: Option<ChildStdin>,
    path: String,
}

impl Writer {
    pub fn new(path: &str, v: &VideoInfo, crf: u32, preset: &str, audio_source: Option<&str>) -> Result<Writer> {
        let mut c = command("ffmpeg")?;
        c.args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{}x{}", v.w, v.h), "-r", &v.fps_text, "-i", "-"]);
        if let Some(a) = audio_source {
            c.arg("-i").arg(a).args(["-map", "0:v", "-map", "1:a", "-c:a", "copy"]);
        }
        c.args(["-c:v", "libx264", "-crf", &crf.to_string(), "-preset", preset, "-pix_fmt", "yuv420p", "-movflags", "+faststart"]);
        c.arg(path);
        let mut child = c.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().context("ffmpeg did not start")?;
        let stdin = child.stdin.take();
        Ok(Writer { child, stdin, path: path.to_string() })
    }

    pub fn write(&mut self, frame: &[u8]) -> Result<()> {
        self.stdin.as_mut().unwrap().write_all(frame).map_err(|e| anyhow!("Writing the video was interrupted ({}): {e}", self.path))
    }

    pub fn finish(mut self) -> Result<()> {
        drop(self.stdin.take());
        let mut err = String::new();
        if let Some(mut e) = self.child.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        let s = self.child.wait()?;
        if !s.success() {
            bail!("ffmpeg failed ({}): {}", self.path, err.trim());
        }
        Ok(())
    }
}

/// "mm:ss.ss"
pub fn timestamp(i: usize, fps: f64) -> String {
    let s = i as f64 / fps;
    format!("{:02}:{:05.2}", (s / 60.0).floor() as u64, s % 60.0)
}

/// "03:37-03:44" or "217.5-224" -> (first frame, end frame)
pub fn parse_range(v: &str, fps: f64) -> Result<(usize, usize)> {
    let secs = |s: &str| -> Result<f64> {
        let mut t = 0.0;
        for (k, o) in s.trim().split(':').rev().enumerate() {
            t += o.trim().parse::<f64>().map_err(|_| anyhow!("invalid time: {s}"))? * 60f64.powi(k as i32);
        }
        Ok(t)
    };
    let (a, b) = v.split_once('-').ok_or_else(|| anyhow!("range must be START-END, e.g. 03:37-03:44: {v}"))?;
    Ok(((secs(a)? * fps) as usize, (secs(b)? * fps) as usize + 1))
}
