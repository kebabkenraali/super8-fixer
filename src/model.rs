//! The slip classifier (models\slip_classifier.onnx) on ONNX Runtime: DirectML GPU, CPU as fallback.

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Result, anyhow};
use ort::ep::DirectML;
use ort::session::Session;
use ort::value::Tensor;

pub const MW: usize = 96; // classifier input
pub const MH: usize = 144;
pub const MODEL_FILE: &str = "slip_classifier.onnx";
pub const PERSONAL_FILE: &str = "slip_classifier_personal.onnx";

pub fn model_path(name: &str) -> PathBuf {
    crate::video::app_dir().join("models").join(name)
}

fn environment() -> Result<&'static ort::environment::Environment> {
    static E: OnceLock<std::result::Result<ort::environment::Environment, String>> = OnceLock::new();
    E.get_or_init(|| ort::init().with_name("super8fixer").build().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| anyhow!("ONNX Runtime did not start: {e}"))
}

fn create_session(path: &PathBuf, gpu: bool) -> Result<Session> {
    let mut b = Session::builder(environment()?)?;
    if gpu {
        b = b
            .with_memory_pattern(false)
            .map_err(|e| anyhow!(e.message().to_string()))?
            .with_parallel_execution(false)
            .map_err(|e| anyhow!(e.message().to_string()))?
            .with_execution_providers([DirectML::default().build().error_on_failure()])
            .map_err(|e| anyhow!(e.message().to_string()))?;
    }
    Ok(b.commit_from_file(path)?)
}

pub struct Classifier {
    session: Session,
    /// mean probability of a run that counts as 0.5 (from <model>.json, "operating_point")
    operating_point: f32,
}

impl Classifier {
    /// Loads the model on the GPU when possible, otherwise on the CPU.
    pub fn load(gpu: bool) -> Result<Classifier> {
        // a model fine-tuned with the user's own corrections (training\finetune_from_edits.py) wins
        let personal = model_path(PERSONAL_FILE);
        let p = if personal.is_file() { personal } else { model_path(MODEL_FILE) };
        if !p.is_file() {
            return Err(anyhow!("Model file missing: {}", p.display()));
        }
        let operating_point = std::fs::read_to_string(p.with_extension("json"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v["operating_point"].as_f64())
            .map(|v| v.clamp(0.01, 0.99) as f32)
            .unwrap_or(0.5);
        crate::log(&format!("Slip classifier model: {} (operating point {operating_point})", p.file_name().unwrap_or_default().to_string_lossy()));
        if gpu {
            match create_session(&p, true) {
                Ok(session) => {
                    crate::log("Slip classifier: GPU (DirectML)");
                    return Ok(Classifier { session, operating_point });
                }
                Err(e) => crate::log(&format!("Slip classifier: GPU not available ({e}), using the CPU.")),
            }
        }
        crate::log("Slip classifier: CPU");
        Ok(Classifier { session: create_session(&p, false)?, operating_point })
    }

    /// Calibrates the mean probability of a run: the model's operating point becomes 0.5.
    pub fn calibrate(&self, mean_p: f32) -> f32 {
        let logit = |p: f32| (p.clamp(1e-6, 1.0 - 1e-6) / (1.0 - p.clamp(1e-6, 1.0 - 1e-6))).ln();
        1.0 / (1.0 + (-(logit(mean_p) - logit(self.operating_point))).exp())
    }

    /// Images MH x MW x 3 (RGB, u8) -> probability that the frame has a slip.
    pub fn predict(&mut self, images: &[Vec<u8>]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(images.len());
        for batch in images.chunks(64) {
            let mut x = Vec::with_capacity(batch.len() * MH * MW * 3);
            for k in batch {
                x.extend(k.iter().map(|&v| v as f32));
            }
            let t = Tensor::from_array(([batch.len(), MH, MW, 3], x))?;
            let res = self.session.run(ort::inputs![t])?;
            let (_, p) = res[0].try_extract_tensor::<f32>()?;
            out.extend_from_slice(p);
        }
        Ok(out)
    }
}
