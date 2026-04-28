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
use std::path::Path;

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
                eprintln!(
                    "Warning: failed to load safetensors file '{}': {}",
                    path_str, e
                );
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

    /// Create an empty resolver that returns `None` for all lookups.
    /// Equivalent to `EmptyWeightResolver` but in the same type for API consistency.
    pub fn empty() -> Self {
        Self { tensors: HashMap::new() }
    }

    /// Load a single safetensors file into the tensor index.
    fn load_safetensors_file(
        path: &str,
        tensors: &mut HashMap<String, TensorEntry>,
    ) -> Result<()> {
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
}

impl WeightResolver for SafetensorsWeightResolver {
    fn resolve(&self, value_path: &str) -> Option<WeightData> {
        self.tensors.get(value_path).map(|entry| WeightData {
            data: entry.data.clone(),
            shape: entry.shape.clone(),
        })
    }
}

/// Convert BF16 tensor data to FP16.
///
/// BF16 has the same structure as FP32 but with the lower 16 bits truncated.
/// FP16 has 5 exponent bits and 10 mantissa bits.
/// Conversion: BF16 bits → F32 → F16 bits
fn convert_bf16_to_fp16(bf16_data: &[u8]) -> Vec<u8> {
    // BF16 is 2 bytes per element, little-endian
    let num_elements = bf16_data.len() / 2;
    let mut fp16_data = Vec::with_capacity(num_elements * 2);

    for i in 0..num_elements {
        let bf16_bits = u16::from_le_bytes([bf16_data[i * 2], bf16_data[i * 2 + 1]]);
        // BF16 → F32: shift left 16 bits
        let f32_bits = (bf16_bits as u32) << 16;
        let f32_val = f32::from_bits(f32_bits);
        // F32 → F16 via half crate or manual conversion
        let fp16_val = f32_to_f16(f32_val);
        fp16_data.extend_from_slice(&fp16_val.to_le_bytes());
    }

    fp16_data
}

/// Convert f32 to f16 (IEEE 754 half-precision).
///
/// This is a software implementation since Rust's standard library
/// doesn't include f16. We implement the conversion using bit manipulation.
fn f32_to_f16(val: f32) -> u16 {
    const EXPONENT_BIAS_F32: i32 = 127;
    const EXPONENT_BIAS_F16: i32 = 15;
    const MANTISSA_BITS_F32: u32 = 23;
    const MANTISSA_BITS_F16: u32 = 10;

    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exponent = ((bits >> MANTISSA_BITS_F32) & 0xFF) as i32;
    let mantissa = bits & ((1 << MANTISSA_BITS_F32) - 1);

    if exponent == 0 {
        // Zero or subnormal f32 → zero f16
        return (sign as u16) << 15;
    }

    if exponent == 255 {
        // Inf or NaN
        let fp16_mantissa = if mantissa != 0 { 0x200 } else { 0 };
        return ((sign as u16) << 15) | (0x1F << MANTISSA_BITS_F16) | (fp16_mantissa as u16);
    }

    let new_exp = exponent - EXPONENT_BIAS_F32 + EXPONENT_BIAS_F16;

    if new_exp <= 0 {
        // Underflow to zero
        return (sign as u16) << 15;
    }

    if new_exp >= 0x1F {
        // Overflow to infinity
        return ((sign as u16) << 15) | (0x1F << MANTISSA_BITS_F16);
    }

    // Round mantissa: shift right and apply round-to-nearest-even
    let shift = MANTISSA_BITS_F32 - MANTISSA_BITS_F16;
    let round_bit = 1u32 << (shift - 1);
    let truncated_mantissa = (mantissa + round_bit) >> shift;

    ((sign as u16) << 15) | ((new_exp as u16) << MANTISSA_BITS_F16) | (truncated_mantissa as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_to_f16_roundtrip() {
        // Test some known values
        let test_cases = vec![
            (0.0f32, 0x0000u16),
            (1.0f32, 0x3C00u16),
            (2.0f32, 0x4000u16),
            (0.5f32, 0x3800u16),
            (-1.0f32, 0xBC00u16),
            (0.1f32, 0x2E66u16), // Approximate
        ];

        for (f32_val, expected) in test_cases {
            let f16 = f32_to_f16(f32_val);
            assert_eq!(
                f16, expected,
                "f32_to_f16({}) = 0x{:04X}, expected 0x{:04X}",
                f32_val, f16, expected
            );
        }
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
}
