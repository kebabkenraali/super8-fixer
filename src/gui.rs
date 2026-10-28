//! The window (egui): video, settings, run table with repair decisions, run editor for manual
//! corrections, before/after preview, repair and stabilization in a background thread, progress
//! and log.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use eframe::egui::{self, Color32, RichText, TextureHandle};
use serde::{Deserialize, Serialize};

use crate::assemble::Fill;
use crate::repair::{self, Analysis, Preview, RepairSettings};
use crate::stabilize::StabilizeSettings;
use crate::video::timestamp;

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
struct Settings {
    video: String,
    threshold: f32,
    use_model: bool,
    model_threshold: f32,
    assemble: bool,
    refine: bool,
    stabilize: bool,
    smoothing: f64,
    max_shift: f64,
    scene_threshold: f64,
    crf: u32,
    preset: String,
    gpu: bool,
    show_seam: bool,
}

impl Default for Settings {
    fn default() -> Self {
        let r = RepairSettings::default();
        let s = StabilizeSettings::default();
        Settings {
            video: String::new(),
            threshold: r.threshold,
            use_model: r.use_model,
            model_threshold: r.model_threshold,
            assemble: r.assemble,
            refine: r.refine,
            stabilize: true,
            smoothing: s.smoothing,
            max_shift: s.max_shift,
            scene_threshold: s.scene_threshold,
            crf: r.crf,
            preset: r.preset,
            gpu: true,
            show_seam: true,
        }
    }
}

fn settings_file() -> PathBuf {
    crate::video::app_dir().join("super8fixer_settings.json")
}

impl Settings {
    fn load() -> Settings {
        std::fs::read_to_string(settings_file()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }
    fn save(&self) {
        if let Ok(t) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(settings_file(), t);
        }
    }
    fn repair(&self) -> RepairSettings {
        RepairSettings {
            threshold: self.threshold,
            use_model: self.use_model,
            model_threshold: self.model_threshold,
            skip: vec![],
            force: vec![],
            assemble: self.assemble,
            refine: self.refine,
            crf: self.crf,
            preset: self.preset.clone(),
            gpu: self.gpu,
        }
    }
    fn stabilize(&self) -> StabilizeSettings {
        StabilizeSettings {
            smoothing: self.smoothing,
            max_shift: self.max_shift,
            scene_threshold: self.scene_threshold,
            crf: self.crf,
            preset: self.preset.clone(),
        }
    }
}

enum JobResult {
    Analysis(Box<Analysis>),
    Done(String),
    Error(String),
}

struct PreviewTextures {
    frame: usize,
    before: TextureHandle,
    after: TextureHandle,
}

#[derive(Clone, Copy, PartialEq)]
enum Show {
    All,
    Repaired,
    Skipped,
    Edited,
}

struct App {
    s: Settings,
    log: Arc<Mutex<Vec<String>>>,
    status: Arc<Mutex<(String, f32)>>,
    job: Option<Receiver<JobResult>>,
    analysis: Option<Analysis>,
    selected: Option<usize>,
    frame: usize,
    preview: Option<PreviewTextures>,
    preview_job: Option<(usize, Receiver<Result<Preview, String>>)>,
    /// another preview was requested while one was running
    preview_again: bool,
    last_output: Option<String>,
    show: Show,
    /// seam tools also change the following frames of the run
    to_run_end: bool,
    /// test run (SUPER8FIXER_TEST=1): the analysis starts by itself
    test_start: bool,
}

pub fn run(video: Option<String>) -> Result<()> {
    let viewport = egui::ViewportBuilder::default()
        .with_title("Super8 Fixer")
        .with_inner_size([1400.0, 900.0])
        .with_min_inner_size([1000.0, 640.0])
        .with_drag_and_drop(true);
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    eframe::run_native("super8fixer", options, Box::new(move |cc| Ok(Box::new(App::new(cc, video)))))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

const VIDEO_TYPES: &[&str] = &["mp4", "mkv", "avi", "mov", "m4v", "webm", "mpg", "ts"];

/// Downscales an RGB frame for display and marks the seam row(s) when given.
fn display_image(rgb: &[u8], w: usize, h: usize, width: usize, seam: Option<usize>) -> egui::ColorImage {
    let im = image::RgbImage::from_raw(w as u32, h as u32, rgb.to_vec()).unwrap();
    let k = (width as f64 / w as f64).min(1.0);
    let mut p = image::imageops::resize(&im, (w as f64 * k) as u32, (h as f64 * k) as u32, image::imageops::FilterType::Triangle);
    if let Some(row) = seam {
        let y = ((row as f64 * k) as u32).min(p.height().saturating_sub(1));
        for x in 0..p.width() {
            // dashed so that the picture under it stays visible
            if (x / 6) % 2 == 0 {
                p.put_pixel(x, y, image::Rgb([255, 40, 40]));
            }
        }
    }
    egui::ColorImage::from_rgb([p.width() as usize, p.height() as usize], p.as_raw())
}

impl App {
    fn new(cc: &eframe::CreationContext, video: Option<String>) -> App {
        let log = Arc::new(Mutex::new(Vec::new()));
        let status = Arc::new(Mutex::new((String::from("Ready."), 0.0f32)));
        let (l, t, ctx) = (log.clone(), status.clone(), cc.egui_ctx.clone());
        crate::set_receiver(Box::new(move |m| {
            match m {
                crate::Message::Log(s) => {
                    let mut l = l.lock().unwrap();
                    l.push(s);
                    let n = l.len();
                    if n > 2000 {
                        l.drain(..n - 2000);
                    }
                }
                crate::Message::Status(s, i, n) => *t.lock().unwrap() = (s, i as f32 / n.max(1) as f32),
            }
            ctx.request_repaint();
        }));
        let mut s = Settings::load();
        if let Some(v) = video {
            s.video = v;
        }
        crate::log(&format!("Super8 Fixer {} - application folder {}", env!("CARGO_PKG_VERSION"), crate::video::app_dir().display()));
        App {
            s,
            log,
            status,
            job: None,
            analysis: None,
            selected: None,
            frame: 0,
            preview: None,
            preview_job: None,
            preview_again: false,
            last_output: None,
            show: Show::All,
            to_run_end: false,
            test_start: std::env::var_os("SUPER8FIXER_TEST").is_some(),
        }
    }

    fn start_job(&mut self, f: impl FnOnce() -> Result<JobResult> + Send + 'static) {
        crate::STOP.store(false, Ordering::Relaxed);
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let r = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => JobResult::Error(format!("{e:#}")),
                Err(_) => JobResult::Error("Unexpected error (panic).".into()),
            };
            let _ = tx.send(r);
        });
        self.job = Some(rx);
        self.s.save();
    }

    fn analyze(&mut self) {
        let video = self.s.video.clone();
        let a = self.s.repair();
        self.analysis = None;
        self.selected = None;
        self.preview = None;
        crate::log(&format!("- Analysis: {video}"));
        self.start_job(move || Ok(JobResult::Analysis(Box::new(repair::analyze(&video, &a)?))));
    }

    fn repair(&mut self) {
        let Some(an) = self.analysis.clone() else { return };
        let a = self.s.repair();
        let sa = self.s.stabilize();
        let stabilize = self.s.stabilize;
        let stem = repair::stem(&an.info.path);
        let video = an.info.path.clone();
        let _ = crate::analysis::report(Path::new(&repair::with_suffix(&stem, "_repair_report.txt")), &an.runs, an.info.fps);
        let _ = an.save_edits();
        crate::log(&format!("- Repair: {video}"));
        self.start_job(move || {
            let o = repair::with_suffix(&stem, "_repaired.mp4");
            repair::render(&an, &a, &o)?;
            if !stabilize {
                return Ok(JobResult::Done(o));
            }
            let v = crate::video::VideoInfo::read(&o)?;
            let st = repair::with_suffix(&repair::stem(&o), "_stable.mp4");
            crate::stabilize::stabilize(&v, &sa, Some(&st))?;
            Ok(JobResult::Done(st))
        });
    }

    fn stabilize_only(&mut self) {
        let video = self.s.video.clone();
        let sa = self.s.stabilize();
        crate::log(&format!("- Stabilization: {video}"));
        self.start_job(move || {
            let v = crate::video::VideoInfo::read(&video)?;
            let st = repair::with_suffix(&repair::stem(&video), "_stable.mp4");
            crate::stabilize::stabilize(&v, &sa, Some(&st))?;
            Ok(JobResult::Done(st))
        });
    }

    fn request_preview(&mut self, frame: usize) {
        let Some(an) = &self.analysis else { return };
        if self.preview_job.is_some() {
            self.preview_again = true;
            return;
        }
        let an = an.clone();
        let assemble = self.s.assemble;
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(repair::preview(&an, frame, assemble).map_err(|e| format!("{e:#}")));
        });
        self.preview_job = Some((frame, rx));
    }

    /// The user changed something in the analysis: save the edits and refresh the preview.
    fn edited(&mut self) {
        if let Some(an) = &self.analysis {
            if let Err(e) = an.save_edits() {
                crate::log(&format!("Saving the edits failed: {e}"));
            }
        }
        self.request_preview(self.frame);
    }

    fn handle_results(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.job {
            if let Ok(t) = rx.try_recv() {
                self.job = None;
                match t {
                    JobResult::Analysis(a) => {
                        self.analysis = Some(*a);
                        *self.status.lock().unwrap() = ("Analysis done. Check the runs, then repair.".into(), 1.0);
                        let first = self.analysis.as_ref().and_then(|a| a.runs.iter().position(|r| r.repair));
                        if let Some(i) = first {
                            self.select(i);
                        }
                    }
                    JobResult::Done(p) => {
                        *self.status.lock().unwrap() = (format!("Done: {p}"), 1.0);
                        self.last_output = Some(p);
                    }
                    JobResult::Error(e) => {
                        crate::log(&format!("ERROR: {e}"));
                        *self.status.lock().unwrap() = (format!("Interrupted: {e}"), 0.0);
                    }
                }
            }
        }
        if let Some((frame, rx)) = &self.preview_job {
            if let Ok(r) = rx.try_recv() {
                let frame = *frame;
                self.preview_job = None;
                match r {
                    Ok(p) => {
                        let v = &self.analysis.as_ref().unwrap().info;
                        let (w, h) = (v.w, v.h);
                        let (sb, sa) = match (self.s.show_seam, p.seam) {
                            (true, Some((b, a))) => (Some(b), Some(a)),
                            _ => (None, None),
                        };
                        self.preview = Some(PreviewTextures {
                            frame,
                            before: ctx.load_texture("before", display_image(&p.original, w, h, 960, sb), egui::TextureOptions::LINEAR),
                            after: ctx.load_texture("after", display_image(&p.repaired, w, h, 960, sa), egui::TextureOptions::LINEAR),
                        });
                    }
                    Err(e) => crate::log(&format!("Preview failed: {e}")),
                }
                if self.preview_again {
                    self.preview_again = false;
                    self.request_preview(self.frame);
                }
            }
        }
    }

    fn select(&mut self, i: usize) {
        self.selected = Some(i);
        if let Some(an) = &self.analysis {
            let r = &an.runs[i];
            self.frame = (r.start + r.end) / 2;
            self.request_preview(self.frame);
        }
    }

    fn settings_panel(&mut self, ui: &mut egui::Ui) {
        let busy = self.job.is_some();
        ui.heading("Repair");
        ui.add_enabled_ui(!busy, |ui| {
            ui.checkbox(&mut self.s.use_model, "Slip classifier (neural network)")
                .on_hover_text("Tells real slips from straight horizontal edges (horizons, window frames). models\\slip_classifier.onnx");
            ui.horizontal(|ui| {
                ui.label("Classifier threshold");
                ui.add(egui::DragValue::new(&mut self.s.model_threshold).range(0.05..=0.95).speed(0.01));
            });
            ui.horizontal(|ui| {
                ui.label("Confidence threshold");
                ui.add(egui::DragValue::new(&mut self.s.threshold).range(0.05..=1.2).speed(0.01))
                    .on_hover_text("Runs near the picture edge are decided by this (default 0.45)");
            });
            ui.checkbox(&mut self.s.assemble, "Rebuild frames from neighbours")
                .on_hover_text("The missing part is taken from the neighbouring frame that continues the picture best, so the seam does not show. Off: roll only.");
            ui.checkbox(&mut self.s.refine, "Automatic seam refinement (experimental)")
                .on_hover_text("Checks every repaired frame against the previous one and corrects the seam position when the tracker locked onto an edge in the picture. Takes effect on the next analysis.");
        });
        ui.separator();
        ui.heading("Stabilization");
        ui.add_enabled_ui(!busy, |ui| {
            ui.checkbox(&mut self.s.stabilize, "Stabilize after repairing");
            ui.horizontal(|ui| {
                ui.label("Smoothing");
                ui.add(egui::DragValue::new(&mut self.s.smoothing).range(1.0..=40.0).speed(0.2).suffix(" frames"))
                    .on_hover_text("Larger = steadier, but slow camera moves are damped too");
            });
            ui.horizontal(|ui| {
                ui.label("Max shift");
                ui.add(egui::DragValue::new(&mut self.s.max_shift).range(0.5..=10.0).speed(0.1).suffix(" %"));
            });
            ui.horizontal(|ui| {
                ui.label("Scene threshold");
                ui.add(egui::DragValue::new(&mut self.s.scene_threshold).range(5.0..=100.0).speed(0.5));
            });
        });
        ui.separator();
        ui.heading("Output");
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal(|ui| {
                ui.label("Quality (CRF)");
                ui.add(egui::DragValue::new(&mut self.s.crf).range(0..=40));
            });
            ui.horizontal(|ui| {
                ui.label("Speed");
                egui::ComboBox::from_id_salt("preset").selected_text(self.s.preset.clone()).show_ui(ui, |ui| {
                    for p in ["ultrafast", "veryfast", "fast", "medium", "slow", "slower"] {
                        ui.selectable_value(&mut self.s.preset, p.to_string(), p);
                    }
                });
            });
            ui.checkbox(&mut self.s.gpu, "Use the GPU for the classifier (DirectML)");
        });
        ui.separator();
        if ui.button("Restore defaults").clicked() {
            let video = self.s.video.clone();
            self.s = Settings { video, ..Default::default() };
        }
    }

    fn runs_panel(&mut self, ui: &mut egui::Ui) {
        let Some(an) = &mut self.analysis else {
            ui.add_space(30.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("Choose a video and press Analyze.").size(18.0));
                ui.label("The program finds the places where the frame line has slipped into the picture and lists them here.");
                ui.label("You can check every run in the preview, decide whether it is repaired, and correct it by hand.");
            });
            return;
        };
        let fps = an.info.fps;
        let (nr, nf) = an.repaired_count();
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{} runs, repairing {nr} ({nf} frames)", an.runs.len())).strong());
            if ui.small_button("All").clicked() {
                for i in 0..an.runs.len() {
                    an.set_repair(i, true);
                }
                changed = true;
            }
            if ui.small_button("None").clicked() {
                for i in 0..an.runs.len() {
                    an.set_repair(i, false);
                }
                changed = true;
            }
            ui.separator();
            ui.label("Show:");
            for (v, t) in [(Show::All, "all"), (Show::Repaired, "repaired"), (Show::Skipped, "skipped"), (Show::Edited, "edited")] {
                ui.selectable_value(&mut self.show, v, t);
            }
        });
        let mut pick = None;
        let show = self.show;
        egui::ScrollArea::vertical().id_salt("runs").max_height(ui.available_height() * 0.34).auto_shrink([false, true]).show(ui, |ui| {
            egui::Grid::new("run_table").striped(true).num_columns(9).spacing([14.0, 3.0]).show(ui, |ui| {
                for h in ["Repair", "Start", "End", "Frames", "Confidence", "Model", "Frame line", "Status", "Edits"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for i in 0..an.runs.len() {
                    let edited = !an.edits[i].is_default();
                    let visible = match show {
                        Show::All => true,
                        Show::Repaired => an.runs[i].repair,
                        Show::Skipped => !an.runs[i].repair,
                        Show::Edited => edited,
                    };
                    if !visible {
                        continue;
                    }
                    let mut rep = an.runs[i].repair;
                    if ui.checkbox(&mut rep, "").changed() {
                        an.set_repair(i, rep);
                        changed = true;
                    }
                    let r = &an.runs[i];
                    if ui.selectable_label(self.selected == Some(i), timestamp(r.start, fps)).clicked() {
                        pick = Some(i);
                    }
                    ui.label(timestamp(r.end, fps));
                    ui.label(format!("{}", r.end - r.start));
                    ui.label(format!("{:.2}", r.confidence));
                    ui.label(r.p.map(|p| format!("{p:.2}")).unwrap_or_else(|| "-".into()));
                    ui.label(format!("{:.0} %", 100.0 * r.position as f64 / crate::analysis::AH as f64));
                    let (t, c) = match (r.repair, r.manual, r.p) {
                        (true, true, _) => ("slip (by user)", Color32::from_rgb(90, 150, 220)),
                        (true, false, _) => ("slip", Color32::from_rgb(90, 180, 90)),
                        (false, true, _) => ("skipped by user", Color32::from_rgb(220, 150, 60)),
                        (false, false, Some(_)) => ("not a slip (model)", Color32::GRAY),
                        (false, false, None) => ("not a slip", Color32::GRAY),
                    };
                    ui.label(RichText::new(t).color(c));
                    let refined = (r.start..r.end).filter(|&k| an.auto_adjust[k] != 0).count();
                    let mut note = String::new();
                    if edited {
                        note.push_str("edited ");
                    }
                    if refined > 0 && !an.edits[i].no_refine {
                        note.push_str(&format!("auto {refined}"));
                    }
                    ui.label(RichText::new(note).small());
                    ui.end_row();
                }
            });
        });
        if let Some(i) = pick {
            self.select(i);
        }
        if changed {
            self.edited();
        }
        ui.separator();
        self.preview_panel(ui);
    }

    fn run_editor(&mut self, ui: &mut egui::Ui, ri: usize) {
        let frame = self.frame;
        let Some(an) = &mut self.analysis else { return };
        let mut changed = false;
        let auto = an.auto_adjust[frame];
        let base = an.base_shift(ri, frame);
        let refined = an.refined;
        let final_shift = an.shift(ri, frame);
        let edit_file = an.edit_file().file_name().unwrap_or_default().to_string_lossy().into_owned();
        let mut set_prev = false;
        let mut to_end = self.to_run_end;
        let e = &mut an.edits[ri];
        egui::Grid::new("editor").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
            ui.label("Seam, whole run").on_hover_text("Moves the picture of every frame in the run up (-) or down (+), in video rows. Use this when the repaired frames are too high or too low.");
            ui.horizontal(|ui| {
                for (t, d) in [("-10", -10), ("-1", -1)] {
                    if ui.small_button(t).clicked() {
                        e.offset += d;
                        changed = true;
                    }
                }
                changed |= ui.add(egui::DragValue::new(&mut e.offset).range(-400..=400).suffix(" px")).changed();
                for (t, d) in [("+1", 1), ("+10", 10)] {
                    if ui.small_button(t).clicked() {
                        e.offset += d;
                        changed = true;
                    }
                }
            });
            ui.end_row();
            ui.label("Seam, this frame").on_hover_text("An extra correction for the current frame only.");
            ui.horizontal(|ui| {
                let mut f = e.frame_offsets.get(&frame).copied().unwrap_or(0);
                let before = f;
                for (t, d) in [("-10", -10), ("-1", -1)] {
                    if ui.small_button(t).clicked() {
                        f += d;
                    }
                }
                ui.add(egui::DragValue::new(&mut f).range(-400..=400).suffix(" px"));
                for (t, d) in [("+1", 1), ("+10", 10)] {
                    if ui.small_button(t).clicked() {
                        f += d;
                    }
                }
                if f != before {
                    if f == 0 {
                        e.frame_offsets.remove(&frame);
                    } else {
                        e.frame_offsets.insert(frame, f);
                    }
                    changed = true;
                }
            });
            ui.end_row();
            ui.label("Fill missing part from");
            egui::ComboBox::from_id_salt("fill").selected_text(e.fill.label()).show_ui(ui, |ui| {
                for f in Fill::ALL {
                    changed |= ui.selectable_value(&mut e.fill, f, f.label()).changed();
                }
            });
            ui.end_row();
            ui.label("Filled part offset").on_hover_text("Moves the filled part (vertical, horizontal) when it does not line up with the rest of the picture.");
            ui.horizontal(|ui| {
                ui.label("Y");
                changed |= ui.add(egui::DragValue::new(&mut e.nudge_y).range(-100..=100).suffix(" px")).changed();
                ui.label("X");
                changed |= ui.add(egui::DragValue::new(&mut e.nudge_x).range(-200..=200).suffix(" px")).changed();
            });
            ui.end_row();
            ui.label("Seam tools").on_hover_text("Click the real frame line in the Before picture to place the seam there. The buttons and the click change this frame, or with the tick also the rest of the run.");
            ui.horizontal(|ui| {
                ui.add_enabled_ui(frame > e.start, |ui| {
                    if ui.button("Same as previous frame").clicked() {
                        set_prev = true;
                    }
                });
                ui.checkbox(&mut to_end, "to run end").on_hover_text("Also change the following frames of the run");
            });
            ui.end_row();
            if refined {
                ui.label("Automatic refinement");
                ui.horizontal(|ui| {
                    let mut use_auto = !e.no_refine;
                    if ui.checkbox(&mut use_auto, "use").changed() {
                        e.no_refine = !use_auto;
                        changed = true;
                    }
                    ui.label(RichText::new(format!("this frame: {auto:+} px")).small().color(Color32::GRAY));
                });
                ui.end_row();
            }
            ui.label("Tracked shift");
            ui.label(RichText::new(format!("{base} px, final {final_shift} px")).color(Color32::GRAY));
            ui.end_row();
        });
        ui.horizontal(|ui| {
            if ui.button("Reset this run").clicked() {
                let (s, en, rep) = (e.start, e.end, e.repair);
                *e = repair::RunEdit { start: s, end: en, repair: rep, ..Default::default() };
                changed = true;
            }
            ui.label(RichText::new(format!("Edits are saved next to the video: {edit_file}")).small().color(Color32::GRAY));
        });
        self.to_run_end = to_end;
        if set_prev {
            let end = if to_end { an.runs[ri].end } else { frame + 1 };
            let target = an.shift(ri, frame - 1) as i64;
            an.set_shift(ri, frame, end, target);
            changed = true;
        }
        if changed {
            self.edited();
        }
    }

    fn preview_panel(&mut self, ui: &mut egui::Ui) {
        let (Some(an), Some(i)) = (&self.analysis, self.selected) else {
            ui.label("Select a run in the table to see the preview.");
            return;
        };
        let r = an.runs[i].clone();
        let fps = an.info.fps;
        let n_runs = an.runs.len();
        let h = an.info.h;
        let mut frame = self.frame;
        let mut go_run: Option<usize> = None;
        // keyboard: left/right = frame, page up/down = run (not while typing)
        if !ui.ctx().egui_wants_keyboard_input() {
            ui.input(|inp| {
                if inp.key_pressed(egui::Key::ArrowLeft) && frame > r.start {
                    frame -= 1;
                }
                if inp.key_pressed(egui::Key::ArrowRight) && frame + 1 < r.end {
                    frame += 1;
                }
                if inp.key_pressed(egui::Key::PageUp) && i > 0 {
                    go_run = Some(i - 1);
                }
                if inp.key_pressed(egui::Key::PageDown) && i + 1 < n_runs {
                    go_run = Some(i + 1);
                }
            });
        }
        ui.horizontal(|ui| {
            if ui.small_button("< Run").on_hover_text("Previous run (Page Up)").clicked() && i > 0 {
                go_run = Some(i - 1);
            }
            if ui.small_button("<").on_hover_text("Previous frame (Left arrow)").clicked() && frame > r.start {
                frame -= 1;
            }
            let resp = ui.add(egui::Slider::new(&mut frame, r.start..=r.end.saturating_sub(1)).show_value(false));
            if ui.small_button(">").on_hover_text("Next frame (Right arrow)").clicked() && frame + 1 < r.end {
                frame += 1;
            }
            if ui.small_button("Run >").on_hover_text("Next run (Page Down)").clicked() && i + 1 < n_runs {
                go_run = Some(i + 1);
            }
            ui.label(format!("{} (#{frame}, {} / {})", timestamp(frame, fps), frame - r.start + 1, r.end - r.start));
            // while dragging only the number follows; the preview is computed on release
            if resp.dragged() {
                self.frame = frame;
            } else if frame != self.frame || (resp.drag_stopped() && self.preview.as_ref().is_none_or(|p| p.frame != frame)) {
                self.frame = frame;
                self.request_preview(frame);
            }
            if ui.checkbox(&mut self.s.show_seam, "Show seam").changed() {
                self.request_preview(self.frame);
            }
            if self.preview_job.is_some() {
                ui.spinner();
            }
            if !r.repair {
                ui.label(RichText::new("This run is not repaired - the right side shows the same picture").color(Color32::GRAY));
            }
        });
        if let Some(g) = go_run {
            self.select(g);
            return;
        }
        const EDITOR_W: f32 = 430.0;
        let room = ui.available_size();
        let pictures_w = (room.x - EDITOR_W - 24.0).max(200.0);
        let mut clicked_row = None;
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(pictures_w);
                clicked_row = self.pictures(ui, egui::vec2(pictures_w, room.y - 24.0), r.repair, h, fps);
            });
            ui.separator();
            ui.vertical(|ui| {
                ui.set_width(EDITOR_W);
                ui.label(RichText::new("Run editor - manual corrections").strong());
                ui.add_enabled_ui(r.repair, |ui| self.run_editor(ui, i));
                if r.repair {
                    ui.label(RichText::new("Tip: click the real frame line in the Before picture.").small().color(Color32::GRAY));
                } else {
                    ui.label(RichText::new("Tick Repair in the table to edit this run.").color(Color32::GRAY));
                }
            });
        });
        if let (Some(row), Some(an)) = (clicked_row, &mut self.analysis) {
            let frame = self.frame;
            let end = if self.to_run_end { an.runs[i].end } else { frame + 1 };
            let target = an.shift_for_line(i, row);
            an.set_shift(i, frame, end, target);
            self.edited();
        }
    }

    /// The before/after pictures in `room`. Returns the video row clicked in the Before picture.
    fn pictures(&self, ui: &mut egui::Ui, room: egui::Vec2, clickable: bool, h: usize, fps: f64) -> Option<usize> {
        let p = self.preview.as_ref()?;
        let width = ((room.x - 16.0) / 2.0).max(100.0);
        let size = p.before.size_vec2();
        let k = (width / size.x).min(room.y.max(80.0) / (size.y + 24.0));
        let shown = size * k;
        let mut clicked = None;
        ui.horizontal(|ui| {
            for (title, t) in [("Before", &p.before), ("After", &p.after)] {
                ui.vertical(|ui| {
                    ui.label(RichText::new(format!("{title} - {}", timestamp(p.frame, fps))).strong());
                    let img = egui::Image::new(egui::load::SizedTexture::new(t.id(), shown));
                    if title == "Before" && clickable {
                        let resp = ui.add(img.sense(egui::Sense::click())).on_hover_text("Click the real frame line to place the seam there");
                        if let Some(pos) = resp.interact_pointer_pos().filter(|_| resp.clicked()) {
                            let y = ((pos.y - resp.rect.top()) / resp.rect.height()).clamp(0.0, 1.0);
                            clicked = Some(((y * h as f32) as usize).min(h - 1));
                        }
                        if resp.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
                        }
                    } else {
                        ui.add(img);
                    }
                });
            }
        });
        clicked
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_results(ctx);
        if self.test_start && Path::new(&self.s.video).is_file() {
            self.test_start = false;
            self.analyze();
        }
        if self.job.is_some() || self.preview_job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        let dropped: Option<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).find(|p| !p.as_os_str().is_empty()));
        if let Some(p) = dropped {
            if self.job.is_none() {
                self.s.video = p.to_string_lossy().into_owned();
                self.analysis = None;
                self.preview = None;
            }
        }
        if ctx.input(|i| i.viewport().close_requested()) {
            crate::STOP.store(true, Ordering::Relaxed);
            self.s.save();
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let busy = self.job.is_some();
        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("Super8 Fixer").size(20.0).strong());
                ui.separator();
                ui.label("Video:");
                let w = (ui.available_width() - 330.0).max(200.0);
                ui.add_enabled(!busy, egui::TextEdit::singleline(&mut self.s.video).desired_width(w).hint_text("Choose or drop a video file into the window"));
                if ui.add_enabled(!busy, egui::Button::new("Browse...")).clicked() {
                    let mut d = rfd::FileDialog::new().set_title("Choose a digitized film").add_filter("Video", VIDEO_TYPES);
                    if let Some(k) = Path::new(&self.s.video).parent().filter(|p| p.is_dir()) {
                        d = d.set_directory(k);
                    }
                    if let Some(p) = d.pick_file() {
                        self.s.video = p.to_string_lossy().into_owned();
                        self.analysis = None;
                        self.preview = None;
                    }
                }
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let has_video = Path::new(&self.s.video).is_file();
                if ui.add_enabled(!busy && has_video, egui::Button::new(RichText::new("1. Analyze").strong())).clicked() {
                    self.analyze();
                }
                let t = if self.s.stabilize { "2. Repair and stabilize" } else { "2. Repair" };
                if ui.add_enabled(!busy && self.analysis.is_some(), egui::Button::new(RichText::new(t).strong())).clicked() {
                    self.repair();
                }
                if ui
                    .add_enabled(!busy && has_video, egui::Button::new("Stabilize only"))
                    .on_hover_text("Stabilize the chosen video as it is (no slip repair)")
                    .clicked()
                {
                    self.stabilize_only();
                }
                if ui.add_enabled(busy, egui::Button::new("Stop")).clicked() {
                    crate::STOP.store(true, Ordering::Relaxed);
                }
                if let Some(p) = self.last_output.clone() {
                    ui.separator();
                    if ui.button("Open result").clicked() {
                        let _ = std::process::Command::new("explorer").arg(&p).spawn();
                    }
                    if ui.button("Show in folder").clicked() {
                        let _ = std::process::Command::new("explorer").arg(format!("/select,{p}")).spawn();
                    }
                }
            });
            ui.add_space(4.0);
        });
        egui::Panel::bottom("bottom").resizable(true).default_size(130.0).show(ui, |ui| {
            let (t, p) = self.status.lock().unwrap().clone();
            ui.add_space(4.0);
            ui.add(egui::ProgressBar::new(p).desired_height(14.0));
            ui.label(t);
            egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                for row in self.log.lock().unwrap().iter() {
                    ui.label(RichText::new(row).monospace().size(12.0));
                }
            });
        });
        egui::Panel::left("settings").resizable(false).default_size(280.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.settings_panel(ui));
        });
        egui::CentralPanel::default().show(ui, |ui| self.runs_panel(ui));
    }
}
