//! Super8 Fixer: repairs frame-line slips in digitized Super 8 / 8 mm film, removes dust and dirt
//! and restores faded colours, as a native Windows application. Without arguments (or with a video
//! file as the only argument) the window opens; command line:
//!   Super8Fixer.exe {repair|compare} VIDEO [options]

#![windows_subsystem = "windows"]

mod analysis;
mod assemble;
mod enhance;
mod font;
mod gui;
mod model;
mod refine;
mod repair;
mod video;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, anyhow, bail};

pub enum Message {
    Log(String),
    Status(String, usize, usize),
}

type Receiver = Box<dyn Fn(Message) + Send + Sync>;

fn receiver() -> &'static Mutex<Option<Receiver>> {
    static R: OnceLock<Mutex<Option<Receiver>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(None))
}

pub fn set_receiver(f: Receiver) {
    *receiver().lock().unwrap() = Some(f);
}

pub fn log(t: &str) {
    match &*receiver().lock().unwrap() {
        Some(f) => f(Message::Log(t.to_string())),
        None => println!("{t}"),
    }
}

pub fn progress(t: &str, i: usize, n: usize) {
    match &*receiver().lock().unwrap() {
        Some(f) => f(Message::Status(t.to_string(), i, n)),
        None => {
            use std::io::Write;
            print!("\r{t}   ");
            let _ = std::io::stdout().flush();
            if i >= n {
                println!();
            }
        }
    }
}

/// Stops the running job (the Stop button).
pub static STOP: AtomicBool = AtomicBool::new(false);

pub fn check_stop() -> Result<()> {
    if STOP.load(Ordering::Relaxed) {
        bail!("Stopped.");
    }
    Ok(())
}

// --------------------------------------------------------------------------
// Command line
// --------------------------------------------------------------------------
struct Args(Vec<String>);

impl Args {
    fn value(&mut self, name: &str) -> Option<String> {
        let i = self.0.iter().position(|a| a == name)?;
        if i + 1 >= self.0.len() {
            return None;
        }
        let v = self.0.remove(i + 1);
        self.0.remove(i);
        Some(v)
    }
    fn values(&mut self, name: &str) -> Vec<String> {
        let mut v = Vec::new();
        while let Some(x) = self.value(name) {
            v.push(x);
        }
        v
    }
    fn flag(&mut self, name: &str) -> bool {
        match self.0.iter().position(|a| a == name) {
            Some(i) => {
                self.0.remove(i);
                true
            }
            None => false,
        }
    }
    fn number<T: std::str::FromStr>(&mut self, name: &str, default: T) -> Result<T> {
        match self.value(name) {
            Some(v) => v.parse().map_err(|_| anyhow!("invalid value {name} {v}")),
            None => Ok(default),
        }
    }
}

const HELP: &str = "Usage:
  Super8Fixer.exe [VIDEO]                    open the window
  Super8Fixer.exe repair VIDEO [options]     repair slips, remove dust, restore colours -> VIDEO_repaired.mp4
  Super8Fixer.exe compare VIDEO --at 05:10   10 s side by side: original | result -> VIDEO_compare_05m10s.mp4

Repair:
  --analyze-only         only the report and the before/after image
  --threshold 0.45       confidence needed to repair a run (when the classifier does not decide)
  --skip 03:37-03:44     never repair runs in this range (repeatable)
  --force 03:37-03:44    always repair runs in this range (repeatable)
  --no-model             do not use the slip classifier
  --model-threshold 0.5  classifier probability needed to repair a run
  --roll-only            only roll the frames, do not rebuild them from neighbours
  --refine               experimental: automatic seam refinement against neighbouring frames
  Edits saved in the window (VIDEO.s8edits.json) are applied automatically.
Picture:
  --no-dust              do not remove dust and dirt
  --dust-strength 1.0    dust sensitivity, 0.5 (clear spots only) .. 2.0 (also faint ones)
  --colour natural       colour restoration: off, natural or vivid
Compare:
  --at 05:10             centre of the clip (default: the middle of the first repaired run)
  --seconds 10           length of the clip
Common:
  -o FILE                output file (repair)
  --crf 16  --preset medium   x264 quality
  --cpu                  do not use the GPU";

fn command_line(args: Vec<String>) -> Result<()> {
    let mut a = Args(args);
    let cmd = a.0.remove(0);
    if matches!(cmd.as_str(), "-h" | "--help" | "help") {
        println!("{HELP}");
        return Ok(());
    }
    if !matches!(cmd.as_str(), "repair" | "compare") {
        bail!("Unknown command {cmd}\n{HELP}");
    }
    let out = a.value("-o").or_else(|| a.value("--output"));
    let colour = match a.value("--colour").or_else(|| a.value("--color")) {
        Some(c) => enhance::Colour::parse(&c).ok_or_else(|| anyhow!("--colour must be off, natural or vivid"))?,
        None => enhance::Colour::Natural,
    };
    let rs = repair::RepairSettings {
        threshold: a.number("--threshold", 0.45f32)?,
        use_model: !a.flag("--no-model"),
        model_threshold: a.number("--model-threshold", 0.5f32)?,
        skip: a.values("--skip"),
        force: a.values("--force"),
        assemble: !a.flag("--roll-only"),
        refine: a.flag("--refine"),
        enhance: enhance::EnhanceSettings { dust: !a.flag("--no-dust"), dust_strength: a.number("--dust-strength", 1.0f32)?, colour },
        crf: a.number("--crf", 16u32)?,
        preset: a.value("--preset").unwrap_or_else(|| "medium".into()),
        gpu: !a.flag("--cpu"),
    };
    let analyze_only = a.flag("--analyze-only");
    let at = a.value("--at");
    let seconds = a.number("--seconds", 10.0f64)?;
    if let Some(x) = a.0.iter().find(|x| x.starts_with("--")) {
        bail!("Unknown option {x}\n{HELP}");
    }
    let video = a.0.first().cloned().ok_or_else(|| anyhow!("the video is missing\n{HELP}"))?;
    let stem = repair::stem(&video);
    let an = repair::analyze(&video, &rs)?;
    if cmd == "compare" {
        let centre = match at {
            Some(t) => video::parse_range(&format!("{t}-{t}"), an.info.fps)?.0,
            None => an.runs.iter().find(|r| r.repair).map(|r| (r.start + r.end) / 2).ok_or_else(|| anyhow!("no repaired runs; give --at"))?,
        };
        repair::compare_clip(&an, &rs, centre, seconds)?;
        return Ok(());
    }
    let img = repair::with_suffix(&stem, "_before_after.png");
    repair::before_after(&an, &img, &rs)?;
    if an.runs.iter().any(|r| r.repair) {
        log(&format!("Before/after images: {img}"));
    }
    if analyze_only {
        return Ok(());
    }
    let o = out.unwrap_or_else(|| repair::with_suffix(&stem, "_repaired.mp4"));
    repair::render(&an, &rs, &o)?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let gui = args.is_empty() || (args.len() == 1 && std::path::Path::new(&args[0]).is_file());
    if gui {
        if let Err(e) = gui::run(args.first().cloned()) {
            rfd::MessageDialog::new().set_title("Super8 Fixer").set_description(format!("Could not start:\n{e}")).show();
        }
        return;
    }
    unsafe {
        windows_sys::Win32::System::Console::AttachConsole(windows_sys::Win32::System::Console::ATTACH_PARENT_PROCESS);
    }
    if let Err(e) = command_line(args) {
        eprintln!("\nERROR: {e:#}");
        std::process::exit(1);
    }
}
