//! Safetensors Weight Resolver
//!
//! Loads real model weights from HuggingFace safetensors files for
//! insertion into the Core ML mlpackage during proto-direct emission.
//!
//! ## Architecture
//!
//! 1. The Python tracer exports `safetensors_files` (paths to .safetensors
//!    files in the HuggingFace cache) and `weight_name_map` (mapping from
//!    torch.fx node names to HuggingFace parameter names).
//!
//! 2. The SIR builder uses `module_path` to produce HuggingFace-style
//!    weight names (e.g., "model.layers.0.self_attn.q_proj.weight") instead
//!    of synthetic names (e.g., "weight_linear1").
//!
//! 3. This resolver reads all safetensors files at construction time,
//!    building an in-memory index of tensor name → raw bytes + shape.
//!
//! 4. When `mir_to_compat` encounters a `MILConst { value_path, .. }`,
//!    it calls `resolve(value_path)` which looks up the tensor by its
//!    HuggingFace parameter name.
//!
//! ## Data Layout
//!
//! Safetensors stores tensors in their original dtype (FP16, BF16, FP32, etc.).
//! The resolver returns raw bytes as-is — dtype conversion happens downstream
//! in the CoreML emission layer.

use crate::mir_to_compat::{WeightData, WeightResolver};
use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Weight resolver backed by HuggingFace safetensors files.
///
/// Reads all specified safetensors files at construction time and builds
/// an in-memory index of tensor name → raw bytes + shape. This allows
/// fast O(1) lookups during `mir_graph_to_compat()`.
#[derive(Debug, Clone)]
pub struct SafetensorsWeightResolver {
    /// Indexed tensors: name → (raw_bytes, shape).
    tensors: HashMap<String, TensorEntry>,
}

#[derive(Debug, Clone)]
struct TensorEntry {
    /// Raw tensor bytes (in safetensors storage format, e.g., FP16 little-endian).
    data: Vec<u8>,
    /// Shape of the tensor.
    shape: Vec<usize>,
}

impl SafetensorsWeightResolver {
    /// Create a resolver by reading safetensors files from the traced graph metadata.
    ///
    /// Args:
    /// - `safetensors_files`: Paths to .safetensors files (from `TracedGraph::safetensors_files`)
    ///
    /// If no files are provided or files can't be read, the resolver will return
    /// `None` for all lookups (falling back to zero-filled weights).
    pub fn from_safetensors_files(safetensors_files: &[String]) -> Self {
        let mut tensors = HashMap::new();

        for path_str in safetensors_files {
            if let Err(e) = Self::load_safetensors_file(path_str, &mut tensors) {
                eprintln!("Warning: failed to load safetensors file '{}': {}", path_str, e);
            }
        }

        Self { tensors }
    }

    /// Create a resolver by scanning a directory for .safetensors files.
    ///
    /// This is useful when the `model_cache_dir` is known but
    /// `safetensors_files` wasn't populated (e.g., older Python tracer).
    pub fn from_cache_dir(cache_dir: &str) -> Self {
        let path = Path::new(cache_dir);
        if !path.is_dir() {
            return Self { tensors: HashMap::new() };
        }

        let mut files = Vec::new();
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                    if let Some(s) = p.to_str() {
                        files.push(s.to_string());
                    }
                }
            }
        }

        Self::from_safetensors_files(&files)
    }

    /// Create a resolver by locating safetensors files from a HuggingFace model ID.
    ///
    /// This is the primary fallback when the Python tracer's strategies fail.
    /// It scans the standard HuggingFace cache directory structure:
    ///
    /// ```text
    /// ~/.cache/huggingface/hub/
    ///   models--Qwen--Qwen3-0.6B/
    ///     snapshots/
    ///       <commit-hash>/
    ///         model.safetensors  (or model-00001-of-000NN.safetensors)
    /// ```
    ///
    /// The model ID is converted to a directory name by replacing `/` with `--`
    /// and prepending `models--`. For example, `Qwen/Qwen3-0.6B` becomes
    /// `models--Qwen--Qwen3-0.6B`.
    ///
    /// This method does NOT require the `huggingface_hub` Python package —
    /// it directly walks the filesystem, which is more reliable.
    pub fn from_hf_model_id(model_id: &str) -> Self {
        let safetensors_files = discover_hf_safetensors(model_id);
        if safetensors_files.is_empty() {
            return Self { tensors: HashMap::new() };
        }
        Self::from_safetensors_files(&safetensors_files)
    }

    /// Create a resolver by locating safetensors from a HuggingFace model ID,
    /// with additional fallback to a cache directory.
    ///
    /// This is the comprehensive resolver that tries multiple strategies:
    /// 1. Explicit safetensors file paths (from Python tracer)
    /// 2. Model cache directory (from Python tracer)
    /// 3. HuggingFace model ID → automatic cache discovery
    ///
    /// Returns the resolver and a description of which strategy succeeded.
    pub fn from_traced_graph(
        safetensors_files: &[String],
        model_cache_dir: Option<&str>,
        model_id: &str,
    ) -> (Self, String) {
        // Strategy 1: Use explicit safetensors file paths from the tracer
        if !safetensors_files.is_empty() {
            let resolver = Self::from_safetensors_files(safetensors_files);
            if !resolver.is_empty() {
                return (resolver, format!("explicit paths ({} files)", safetensors_files.len()));
            }
        }

        // Strategy 2: Scan cache directory from the tracer
        if let Some(cache_dir) = model_cache_dir {
            let resolver = Self::from_cache_dir(cache_dir);
            if !resolver.is_empty() {
                return (resolver, format!("cache dir: {}", cache_dir));
            }
            // The cache_dir might be the HF hub root (not the snapshot dir).
            // Try walking snapshots/ subdirectories.
            let resolver = Self::from_cache_dir_recursive(cache_dir);
            if !resolver.is_empty() {
                return (resolver, format!("cache dir (recursive): {}", cache_dir));
            }
        }

        // Strategy 3: Automatic HF cache discovery from model ID
        let resolver = Self::from_hf_model_id(model_id);
        if !resolver.is_empty() {
            return (resolver, format!("HF model ID auto-discovery: {}", model_id));
        }

        // T-79 (I-54): Previously returned an empty resolver without any warning,
        // causing all weights to become zero-filled placeholders silently. Now
        // we log a warning so the user knows weight resolution failed.
        log::warn!(
            "safetensors resolver is empty — all weights will be zero-filled: {}",
            "no weights found"
        );
        (Self::empty(), "no weights found".to_string())
    }

    /// Create an empty resolver that returns `None` for all lookups.
    /// Equivalent to `EmptyWeightResolver` but in the same type for API consistency.
    pub fn empty() -> Self {
        Self { tensors: HashMap::new() }
    }

    /// Load a single safetensors file into the tensor index.
    fn load_safetensors_file(path: &str, tensors: &mut HashMap<String, TensorEntry>) -> Result<()> {
        let data = fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&data)?;

        for (name, view) in st.tensors() {
            // Convert bf16 → fp16 if needed (ANE uses FP16, not BF16).
            let raw_data = view.data();
            let shape = view.shape().to_vec();

            // Check dtype and convert BF16 → FP16 if necessary
            let final_data = match view.dtype() {
                safetensors::Dtype::BF16 => {
                    // BF16 and FP16 are both 16-bit but have different exponent sizes.
                    // BF16: 8-bit exponent, 7-bit mantissa
                    // FP16: 5-bit exponent, 10-bit mantissa
                    // Convert via: bf16 → f32 → fp16
                    convert_bf16_to_fp16(raw_data)
                }
                safetensors::Dtype::F32 => {
                    // For ANE, we typically want FP16. But let the downstream
                    // decide — return raw bytes and let the emission layer handle it.
                    raw_data.to_vec()
                }
                safetensors::Dtype::F16 => {
                    // Already in the right format
                    raw_data.to_vec()
                }
                _ => {
                    // For other dtypes (int, etc.), pass through as-is
                    raw_data.to_vec()
                }
            };

            tensors.insert(name, TensorEntry { data: final_data, shape });
        }

        Ok(())
    }

    /// Number of tensors loaded.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Create a resolver by recursively scanning a directory for .safetensors files.
    ///
    /// Unlike `from_cache_dir()` which only scans the top level, this method
    /// walks all subdirectories. This is needed for HuggingFace cache directories
    /// where safetensors files are inside `snapshots/<hash>/` subdirectories.
    pub fn from_cache_dir_recursive(cache_dir: &str) -> Self {
        let path = Path::new(cache_dir);
        if !path.is_dir() {
            return Self { tensors: HashMap::new() };
        }

        let mut files = Vec::new();
        walk_for_safetensors(path, &mut files);
        Self::from_safetensors_files(&files)
    }

    /// Whether any tensors were loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get the names of all loaded tensors (for diagnostic output).
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    /// Total byte size of all loaded weight data.
    pub fn total_weight_bytes(&self) -> usize {
        self.tensors.values().map(|e| e.data.len()).sum()
    }

    /// Get all weight names and their shapes as a HashMap.
    /// Used by the mil_lower pass to seed shape inference for weight-backed
    /// ops like Gather (embedding lookup) where the weight tensor isn't an
    /// AIR graph node but its shape is needed for output shape inference.
    pub fn weight_shapes(&self) -> std::collections::HashMap<String, Vec<usize>> {
        self.tensors.iter().map(|(name, entry)| (name.clone(), entry.shape.clone())).collect()
    }
}

impl WeightResolver for SafetensorsWeightResolver {
    fn resolve(&self, value_path: &str) -> Option<WeightData> {
        // Direct lookup first
        if let Some(entry) = self.tensors.get(value_path) {
            return Some(WeightData { data: entry.data.clone(), shape: entry.shape.clone() });
        }

        // Virtual shard weight: "lm_head.shard_N.weight"
        // The lm_head vocab projection is too large for the ANE execution planner
        // (error -5 for a single linear with 151936 output channels). The compiler
        // shards it into N smaller linears. Each shard references a virtual weight
        // name that this resolver resolves by slicing the original lm_head.weight.
        if let Some((base_weight, shard_index)) = parse_shard_weight_name(value_path) {
            return self.resolve_shard(&base_weight, shard_index);
        }

        None
    }
}

impl SafetensorsWeightResolver {
    /// Resolve a virtual shard weight by slicing the original weight tensor.
    ///
    /// The shard naming convention is: `<base>.shard_<N>.weight`
    /// (e.g., `lm_head.shard_0.weight`, `lm_head.shard_1.weight`, ...).
    ///
    /// Each shard takes `LM_HEAD_SHARD_SIZE` rows from the original weight's
    /// first dimension (except the last shard, which may be smaller).
    fn resolve_shard(&self, base_weight: &str, shard_index: usize) -> Option<WeightData> {
        // T-73 (I-48): Previously, LM_HEAD_SHARD_SIZE was hardcoded to 19000,
        // specific to Qwen3-0.6B (vocab_size=151936, 151936/8≈18992). Other
        // models with different vocab sizes would get wrong shard sizes. Now we
        // derive the shard size from the actual vocab_size and a target shard
        // count (8), with a floor of 1 to avoid division by zero.
        const TARGET_SHARD_COUNT: usize = 8;

        let entry = self.tensors.get(base_weight)?;
        if entry.shape.len() != 2 {
            log::warn!(
                "shard weight '{}' references non-2D tensor with shape {:?}",
                base_weight,
                entry.shape
            );
            return None;
        }

        let vocab_size = entry.shape[0];
        let hidden_size = entry.shape[1];

        // Derive shard size from vocab_size, aiming for ~8 shards
        let shard_size = (vocab_size / TARGET_SHARD_COUNT).max(1);

        let start_row = shard_index * shard_size;
        let end_row = (start_row + shard_size).min(vocab_size);

        if start_row >= vocab_size {
            log::warn!(
                "shard index {} out of range for weight '{}' (vocab_size={})",
                shard_index,
                base_weight,
                vocab_size
            );
            return None;
        }

        let shard_rows = end_row - start_row;

        // T-74 (I-49): Previously assumed FP16 (2 bytes per element) with
        // `hidden_size * 2`. For F32 weights that would be 4 bytes, INT8 would
        // be 1 byte — silently slicing the wrong portion of weight data. Now
        // we derive the element size from the actual data length and shape.
        let total_elements: usize = entry.shape.iter().product();
        let bytes_per_element = if total_elements > 0 {
            entry.data.len() / total_elements
        } else {
            2 // fallback to FP16 if shape is degenerate
        };
        let bytes_per_row = hidden_size * bytes_per_element;
        let start_byte = start_row * bytes_per_row;
        let end_byte = end_row * bytes_per_row;

        if end_byte > entry.data.len() {
            log::warn!(
                "shard byte range {}..{} exceeds data length {} for weight '{}' — data may be corrupted",
                start_byte, end_byte, entry.data.len(), base_weight
            );
        }

        let shard_data = entry.data.get(start_byte..end_byte)?.to_vec();
        Some(WeightData { data: shard_data, shape: vec![shard_rows, hidden_size] })
    }
}

/// Parse a virtual shard weight name into (base_weight_name, shard_index).
///
/// Pattern: `<prefix>.shard_<N>.weight` → `(prefix + ".weight", N)`
///
/// Examples:
/// - `"lm_head.shard_0.weight"` → `("lm_head.weight", 0)`
/// - `"lm_head.shard_7.weight"` → `("lm_head.weight", 7)`
/// - `"lm_head.weight"` → `None` (not a shard name)
/// - `"model.layers.0.self_attn.q_proj.weight"` → `None` (not a shard name)
fn parse_shard_weight_name(value_path: &str) -> Option<(String, usize)> {
    // Match pattern: <prefix>.shard_<N>.weight
    if !value_path.contains(".shard_") {
        return None;
    }

    // Try to extract: prefix + ".shard_N" + ".weight"
    let weight_suffix = ".weight";
    if !value_path.ends_with(weight_suffix) {
        return None;
    }

    let without_suffix = &value_path[..value_path.len() - weight_suffix.len()];

    // Find the last ".shard_" segment
    if let Some(shard_start) = without_suffix.rfind(".shard_") {
        let prefix = &without_suffix[..shard_start]; // e.g., "lm_head"
        let shard_part = &without_suffix[shard_start + ".shard_".len()..]; // e.g., "0", "7"

        if let Ok(index) = shard_part.parse::<usize>() {
            let base_weight = format!("{}.weight", prefix);
            return Some((base_weight, index));
        }
    }

    None
}

/// Convert BF16 tensor data to FP16.
///
/// BF16 has the same structure as FP32 but with the lower 16 bits truncated.
/// FP16 has 5 exponent bits and 10 mantissa bits.
/// Conversion: BF16 bits → F32 → F16 bits (via the `half` crate).
fn convert_bf16_to_fp16(bf16_data: &[u8]) -> Vec<u8> {
    // BF16 is 2 bytes per element, little-endian
    let num_elements = bf16_data.len() / 2;
    let mut fp16_data = Vec::with_capacity(num_elements * 2);

    for i in 0..num_elements {
        let bf16_bits = u16::from_le_bytes([bf16_data[i * 2], bf16_data[i * 2 + 1]]);
        // BF16 → F32: shift left 16 bits
        let f32_bits = (bf16_bits as u32) << 16;
        let f32_val = f32::from_bits(f32_bits);
        // F32 → F16 via the `half` crate — correctly handles subnormals and NaN payloads.
        let fp16_val = half::f16::from_f32(f32_val);
        fp16_data.extend_from_slice(&fp16_val.to_bits().to_le_bytes());
    }

    fp16_data
}

// The hand-rolled f32_to_f16 has been replaced by the `half` crate's
// `half::f16::from_f32()`, which correctly handles subnormals and NaN
// payloads. See `convert_bf16_to_fp16` above for usage.

/// Discover safetensors files in the HuggingFace cache from a model ID.
///
/// Walks the standard HF cache directory structure:
/// ```text
/// ~/.cache/huggingface/hub/
///   models--Qwen--Qwen3-0.6B/
///     snapshots/
///       <commit-hash>/
///         model.safetensors
/// ```
///
/// Also respects `HF_HOME` and `HUGGINGFACE_HUB_CACHE` environment variables
/// for custom cache locations.
fn discover_hf_safetensors(model_id: &str) -> Vec<String> {
    // Convert model ID to directory name: "Qwen/Qwen3-0.6B" → "models--Qwen--Qwen3-0.6B"
    let repo_dir_name = format!("models--{}", model_id.replace('/', "--"));

    // Determine the HuggingFace cache root directory.
    // Priority: HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > HF_HOME/hub > ~/.cache/huggingface/hub
    let cache_root = if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        PathBuf::from(v)
    } else if let Ok(v) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        PathBuf::from(v)
    } else if let Ok(v) = std::env::var("HF_HOME") {
        PathBuf::from(v).join("hub")
    } else {
        // Default: ~/.cache/huggingface/hub
        dirs_home_cache().join("huggingface").join("hub")
    };

    let repo_dir = cache_root.join(&repo_dir_name);
    if !repo_dir.is_dir() {
        eprintln!("  HF cache repo dir not found: {} (model_id={})", repo_dir.display(), model_id);
        return Vec::new();
    }

    // Walk the snapshots/ subdirectory to find the latest snapshot with safetensors files
    let snapshots_dir = repo_dir.join("snapshots");
    if !snapshots_dir.is_dir() {
        eprintln!("  No snapshots/ directory in {}", repo_dir.display());
        return Vec::new();
    }

    // Find all snapshot directories, sorted by modification time (newest first)
    let mut snapshot_dirs: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = fs::read_dir(&snapshots_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                snapshot_dirs.push(p);
            }
        }
    }

    // Sort by modification time, newest first
    snapshot_dirs.sort_by(|a, b| {
        let mt_a = a.metadata().and_then(|m| m.modified()).ok();
        let mt_b = b.metadata().and_then(|m| m.modified()).ok();
        mt_b.cmp(&mt_a) // newest first
    });

    // Try each snapshot directory, returning the first one with safetensors files
    for snapshot_dir in &snapshot_dirs {
        let mut safetensors_files = Vec::new();
        if let Ok(entries) = fs::read_dir(snapshot_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                    if let Some(s) = p.to_str() {
                        safetensors_files.push(s.to_string());
                    }
                }
            }
        }
        if !safetensors_files.is_empty() {
            safetensors_files.sort();
            return safetensors_files;
        }
    }

    eprintln!("  No .safetensors files found in any snapshot of {}", model_id);
    Vec::new()
}

/// Get the default cache directory for the current platform.
///
/// HuggingFace's `huggingface_hub` uses XDG-style paths on all platforms:
/// - macOS: ~/.cache  (NOT ~/Library/Caches)
/// - Linux: ~/.cache
/// - Windows: %LOCALAPPDATA%
///
/// This matches the behavior of `huggingface_hub.constants.HF_HUB_CACHE`
/// which uses `os.path.expanduser("~/.cache")` on macOS and Linux.
fn dirs_home_cache() -> PathBuf {
    // Check XDG_CACHE_HOME first (Linux and power users on macOS)
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg);
    }

    // Default: ~/.cache (used by huggingface_hub on both macOS and Linux)
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache");
    }

    // Fallback: use a temp-like location
    PathBuf::from("/tmp/.cache")
}

/// Recursively walk a directory tree collecting .safetensors file paths.
fn walk_for_safetensors(dir: &Path, files: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk_for_safetensors(&p, files);
            } else if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                if let Some(s) = p.to_str() {
                    files.push(s.to_string());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_to_f16_roundtrip() {
        // Test some known values using the `half` crate
        let test_cases = vec![
            (0.0f32, 0x0000u16),
            (1.0f32, 0x3C00u16),
            (2.0f32, 0x4000u16),
            (0.5f32, 0x3800u16),
            (-1.0f32, 0xBC00u16),
            (0.1f32, 0x2E66u16), // Approximate
        ];

        for (f32_val, expected) in test_cases {
            let f16 = half::f16::from_f32(f32_val).to_bits();
            assert_eq!(
                f16, expected,
                "half::f16::from_f32({}) = 0x{:04X}, expected 0x{:04X}",
                f32_val, f16, expected
            );
        }
    }

    // ─── T-83 (I-58): BF16→FP16 Edge-Case Tests ──────────────────────────
    //
    // These tests exercise the convert_bf16_to_fp16() function through the
    // same conversion path used by the safetensors loader: BF16 bits → F32
    // → half::f16::from_f32(). The `half` crate handles subnormals and NaN
    // payloads correctly, but we verify that the full pipeline preserves
    // these special values.

    /// Helper: convert a single BF16 value (as u16) to FP16 (as u16 bits)
    /// using the same pipeline as `convert_bf16_to_fp16()`.
    fn bf16_to_fp16_bits(bf16_bits: u16) -> u16 {
        let f32_bits = (bf16_bits as u32) << 16;
        let f32_val = f32::from_bits(f32_bits);
        half::f16::from_f32(f32_val).to_bits()
    }

    #[test]
    fn test_bf16_to_fp16_nan_preservation() {
        // BF16 quiet NaN: exponent=0xFF, mantissa MSB=1 → 0x7FC0
        // BF16 signaling NaN: exponent=0xFF, mantissa MSB=0, rest≠0 → 0x7F81
        let qnan_bf16 = 0x7FC0u16;
        let snan_bf16 = 0x7F81u16;

        let qnan_fp16 = half::f16::from_bits(bf16_to_fp16_bits(qnan_bf16));
        let snan_fp16 = half::f16::from_bits(bf16_to_fp16_bits(snan_bf16));

        assert!(qnan_fp16.is_nan(), "BF16 quiet NaN should produce FP16 NaN");
        assert!(snan_fp16.is_nan(), "BF16 signaling NaN should produce FP16 NaN");
    }

    #[test]
    fn test_bf16_to_fp16_infinity_preservation() {
        // BF16 +Inf: 0x7F80, -Inf: 0xFF80
        let pos_inf_bf16 = 0x7F80u16;
        let neg_inf_bf16 = 0xFF80u16;

        let pos_inf_fp16 = half::f16::from_bits(bf16_to_fp16_bits(pos_inf_bf16));
        let neg_inf_fp16 = half::f16::from_bits(bf16_to_fp16_bits(neg_inf_bf16));

        assert!(
            pos_inf_fp16.is_infinite() && pos_inf_fp16.is_sign_positive(),
            "BF16 +Inf should produce FP16 +Inf"
        );
        assert!(
            neg_inf_fp16.is_infinite() && neg_inf_fp16.is_sign_negative(),
            "BF16 -Inf should produce FP16 -Inf"
        );
    }

    #[test]
    fn test_bf16_to_fp16_negative_zero() {
        // BF16 -0.0: sign=1, exponent=0, mantissa=0 → 0x8000
        let neg_zero_bf16 = 0x8000u16;
        let pos_zero_bf16 = 0x0000u16;

        let neg_zero_fp16 = half::f16::from_bits(bf16_to_fp16_bits(neg_zero_bf16));
        let pos_zero_fp16 = half::f16::from_bits(bf16_to_fp16_bits(pos_zero_bf16));

        // Both should be zero
        assert_eq!(neg_zero_fp16.to_bits(), 0x8000u16, "BF16 -0 should produce FP16 -0");
        assert_eq!(pos_zero_fp16.to_bits(), 0x0000u16, "BF16 +0 should produce FP16 +0");

        // -0 should be signed
        assert!(neg_zero_fp16.is_sign_negative(), "FP16 -0 should be sign negative");
        assert!(pos_zero_fp16.is_sign_positive(), "FP16 +0 should be sign positive");
    }

    #[test]
    fn test_bf16_to_fp16_subnormal_handling() {
        // BF16 has no subnormals — its smallest positive normal is 2^-126 ≈ 1.175e-38.
        // FP16's smallest positive normal is 2^-14 ≈ 6.104e-5, and its smallest
        // subnormal is 2^-24 ≈ 5.96e-8. Values between these map to FP16 subnormals.
        //
        // BF16 value 2^-126 → F32 = 1.175494e-38 → FP16 subnormal
        // BF16 bits: sign=0, exponent=1 (bias 127 → true exp = -126), mantissa=0
        let small_normal_bf16 = 0x0080u16; // 2^-126
        let fp16_result = half::f16::from_bits(bf16_to_fp16_bits(small_normal_bf16));

        // This BF16 value is far below FP16's normal range, so it becomes
        // a subnormal or flushes to zero. The `half` crate handles this correctly.
        assert!(
            fp16_result.is_normal() == false || fp16_result.to_bits() == 0x0000,
            "BF16 small normal should map to FP16 subnormal or zero, got {:?}",
            fp16_result
        );
    }

    #[test]
    fn test_bf16_to_fp16_max_finite_value() {
        // BF16 max finite: sign=0, exponent=0xFE (254), mantissa=0x7F → 0x7F7F
        // Value = (2 - 2^-7) * 2^127 ≈ 3.389e+38
        // This overflows FP16 max (65504) and should produce +Inf
        let max_bf16 = 0x7F7Fu16;
        let fp16_result = half::f16::from_bits(bf16_to_fp16_bits(max_bf16));

        assert!(
            fp16_result.is_infinite() && fp16_result.is_sign_positive(),
            "BF16 max finite should overflow FP16 to +Inf, got {:?}",
            fp16_result
        );
    }

    #[test]
    fn test_bf16_to_fp16_bulk_conversion() {
        // Test the full convert_bf16_to_fp16 function with a multi-element buffer
        // Construct a BF16 buffer with: [1.0, -1.0, +0.0, -0.0]
        let bf16_values: Vec<u16> = vec![
            0x3F80, // 1.0
            0xBF80, // -1.0
            0x0000, // +0.0
            0x8000, // -0.0
        ];

        // Serialize as little-endian bytes
        let mut bf16_bytes = Vec::new();
        for &val in &bf16_values {
            bf16_bytes.extend_from_slice(&val.to_le_bytes());
        }

        let fp16_bytes = convert_bf16_to_fp16(&bf16_bytes);
        let num_elements = fp16_bytes.len() / 2;
        assert_eq!(num_elements, 4, "Should have 4 FP16 elements");

        // Decode the results
        let fp16_values: Vec<u16> = (0..num_elements)
            .map(|i| {
                let offset = i * 2;
                u16::from_le_bytes([fp16_bytes[offset], fp16_bytes[offset + 1]])
            })
            .collect();

        // Verify: 1.0 → 0x3C00, -1.0 → 0xBC00, +0.0 → 0x0000, -0.0 → 0x8000
        assert_eq!(fp16_values[0], 0x3C00, "1.0 BF16→FP16");
        assert_eq!(fp16_values[1], 0xBC00, "-1.0 BF16→FP16");
        assert_eq!(fp16_values[2], 0x0000, "+0.0 BF16→FP16");
        assert_eq!(fp16_values[3], 0x8000, "-0.0 BF16→FP16");
    }

    #[test]
    fn test_empty_resolver() {
        let resolver = SafetensorsWeightResolver::empty();
        assert!(resolver.is_empty());
        assert!(resolver.resolve("any_name").is_none());
    }

    #[test]
    fn test_from_nonexistent_dir() {
        let resolver = SafetensorsWeightResolver::from_cache_dir("/nonexistent/path");
        assert!(resolver.is_empty());
    }

    #[test]
    fn test_from_hf_model_id_nonexistent() {
        // This should not panic, just return empty
        let resolver = SafetensorsWeightResolver::from_hf_model_id("nonexistent/model-12345");
        assert!(resolver.is_empty());
    }

    #[test]
    fn test_from_traced_graph_no_weights() {
        let (resolver, strategy) =
            SafetensorsWeightResolver::from_traced_graph(&[], None, "nonexistent/model-12345");
        assert!(resolver.is_empty());
        assert_eq!(strategy, "no weights found");
    }

    #[test]
    fn test_from_cache_dir_recursive_nonexistent() {
        let resolver = SafetensorsWeightResolver::from_cache_dir_recursive("/nonexistent/path");
        assert!(resolver.is_empty());
    }

    #[test]
    fn test_parse_shard_weight_name() {
        // Valid shard names
        assert_eq!(
            parse_shard_weight_name("lm_head.shard_0.weight"),
            Some(("lm_head.weight".to_string(), 0))
        );
        assert_eq!(
            parse_shard_weight_name("lm_head.shard_7.weight"),
            Some(("lm_head.weight".to_string(), 7))
        );

        // Not a shard name
        assert_eq!(parse_shard_weight_name("lm_head.weight"), None);
        assert_eq!(parse_shard_weight_name("model.layers.0.self_attn.q_proj.weight"), None);
        assert_eq!(parse_shard_weight_name("some_tensor"), None);
    }

    #[test]
    fn test_resolve_shard_weight() {
        use crate::mir_to_compat::WeightResolver;

        // Create a resolver with a fake "lm_head.weight" tensor: shape [40000, 1024] fp16
        let vocab_size = 40000usize;
        let hidden_size = 1024usize;
        let total_elements = vocab_size * hidden_size;
        let total_bytes = total_elements * 2; // fp16
        let fake_data: Vec<u8> = (0..total_bytes).map(|i| (i % 256) as u8).collect();

        let mut resolver = SafetensorsWeightResolver::empty();
        resolver.tensors.insert(
            "lm_head.weight".to_string(),
            TensorEntry { data: fake_data, shape: vec![vocab_size, hidden_size] },
        );

        // T-73: Shard size is now derived from vocab_size / TARGET_SHARD_COUNT.
        // For vocab_size=40000 and TARGET_SHARD_COUNT=8: shard_size = 5000
        let shard_size = vocab_size / 8; // 5000

        // Resolve shard 0: rows 0..5000
        let shard_0 = resolver.resolve("lm_head.shard_0.weight").expect("shard_0 should resolve");
        assert_eq!(shard_0.shape, vec![shard_size, hidden_size]);
        assert_eq!(shard_0.data.len(), shard_size * hidden_size * 2);

        // Resolve shard 7 (last): rows 35000..40000 → 5000 rows
        let shard_7 = resolver.resolve("lm_head.shard_7.weight").expect("shard_7 should resolve");
        assert_eq!(shard_7.shape, vec![shard_size, hidden_size]);
        assert_eq!(shard_7.data.len(), shard_size * hidden_size * 2);

        // Out of range shard (shard 8 would start at row 40000 = vocab_size)
        assert!(resolver.resolve("lm_head.shard_8.weight").is_none());

        // Non-shard name still works
        let original = resolver.resolve("lm_head.weight").expect("original should resolve");
        assert_eq!(original.shape, vec![vocab_size, hidden_size]);
    }

    #[test]
    fn test_resolve_shard_weight_f32() {
        use crate::mir_to_compat::WeightResolver;

        // T-74: Test shard resolution for F32 (4 bytes/element) weights.
        // Previously, `bytes_per_row = hidden_size * 2` which would produce
        // wrong byte offsets for F32 weights.
        let vocab_size = 16000usize;
        let hidden_size = 512usize;
        let total_elements = vocab_size * hidden_size;
        let total_bytes = total_elements * 4; // fp32 = 4 bytes per element
        let fake_data: Vec<u8> = (0..total_bytes).map(|i| (i % 256) as u8).collect();

        let mut resolver = SafetensorsWeightResolver::empty();
        resolver.tensors.insert(
            "lm_head.weight".to_string(),
            TensorEntry { data: fake_data, shape: vec![vocab_size, hidden_size] },
        );

        // Shard size = 16000 / 8 = 2000
        let shard_size = vocab_size / 8;

        let shard_0 = resolver.resolve("lm_head.shard_0.weight").expect("shard_0 should resolve");
        assert_eq!(shard_0.shape, vec![shard_size, hidden_size]);
        assert_eq!(shard_0.data.len(), shard_size * hidden_size * 4); // F32: 4 bytes/element

        // Verify the byte offsets are correct: first row starts at byte 0
        // and shard_0 data should be the first shard_size * hidden_size * 4 bytes
        assert_eq!(shard_0.data[0], 0u8);
        assert_eq!(shard_0.data[1], 1u8);
    }

    #[test]
    fn test_resolve_shard_qwen3_vocab() {
        use crate::mir_to_compat::WeightResolver;

        // T-73 regression test: Qwen3-0.6B has vocab_size=151936.
        // With TARGET_SHARD_COUNT=8: shard_size = 151936 / 8 = 18992.
        // This matches the old hardcoded 19000 closely (off by 8 rows
        // on the last shard, which is fine — the last shard is smaller anyway).
        let vocab_size = 151936usize;
        let hidden_size = 1024usize;
        let total_elements = vocab_size * hidden_size;
        let total_bytes = total_elements * 2; // fp16
        let fake_data: Vec<u8> = vec![0u8; total_bytes];

        let mut resolver = SafetensorsWeightResolver::empty();
        resolver.tensors.insert(
            "lm_head.weight".to_string(),
            TensorEntry { data: fake_data, shape: vec![vocab_size, hidden_size] },
        );

        let shard_size = vocab_size / 8; // 18992
        assert_eq!(shard_size, 18992);

        let shard_0 = resolver.resolve("lm_head.shard_0.weight").expect("shard_0 should resolve");
        assert_eq!(shard_0.shape, vec![18992, hidden_size]);

        // Last shard (7) should get the remaining rows
        let shard_7 = resolver.resolve("lm_head.shard_7.weight").expect("shard_7 should resolve");
        let expected_last_rows = vocab_size - 7 * shard_size;
        assert_eq!(shard_7.shape, vec![expected_last_rows, hidden_size]);
    }
}
