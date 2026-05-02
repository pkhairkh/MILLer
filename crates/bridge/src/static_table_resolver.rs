//! Static Table Resolver — computes RoPE cos/sin/eye/mask tables at compile time.
//!
//! Derived from `pkhairkh/qwen3-coreml-palettized`'s `rotary_tables.py`:
//! pre-computes sin_tab, cos_tab, eye_tab, and mask_tab as fp16 constants.
//!
//! ## Why static tables?
//!
//! Computing `cos(pos * inv_freq)` and `sin(pos * inv_freq)` at runtime in
//! fp16 on the ANE introduces precision loss, especially under long-context
//! decode. Pre-computing them in float64 and storing as fp16 constants
//! eliminates this source of error. Additionally, the ANE has limited support
//! for dynamic transcendental functions — static tables + `mb.gather` is
//! ANE-friendly.
//!
//! ## Table shapes
//!
//! | Table | Shape | dtype | Purpose |
//! |-------|-------|-------|---------|
//! | sin_tab | [1, 1, seq_len, head_dim] | float16 | RoPE sin values per position |
//! | cos_tab | [1, 1, seq_len, head_dim] | float16 | RoPE cos values per position |
//! | eye_tab | [seq_len, seq_len] | float16 | Identity for KV-cache write mask (Gather+Reshape) |
//! | mask_tab | [seq_len, seq_len] | float16 | Causal attention mask (Gather+Reshape+Add) |
//! | arange_tab | [seq_len] | int32 | Position indices (legacy, for prefill fallback) |
//!
//! The sin/cos table shape `[1, 1, seq_len, head_dim]` is chosen for broadcast
//! compatibility with the Q/K tensor shape `[1, num_heads, seq_len, head_dim]`.
//! Core ML broadcasting aligns dimensions right-to-left, so this shape broadcasts
//! correctly across the heads dimension (1 → num_heads).
//!
//! ## Mathematical formula
//!
//! ```text
//! inv_freq[i] = 1 / theta^(2i/d)   for i = 0..d/2-1
//! freqs[pos, i] = pos * inv_freq[i]
//! emb[pos, :] = cat(freqs[pos], freqs[pos])  (duplicate to full head_dim)
//! cos_tab[0, 0, pos, :] = cos(emb[pos, :])
//! sin_tab[0, 0, pos, :] = sin(emb[pos, :])
//! ```

use crate::mir_to_compat::{WeightData, WeightResolver};
use std::collections::HashMap;

/// Configuration for RoPE table computation.
///
/// Generic parameters — not tied to any specific model architecture.
/// The tracer/compiler pipeline populates these from HuggingFace config
/// fields (`rope_theta`, `head_dim`, `max_position_embeddings`).
#[derive(Debug, Clone)]
pub struct RopeTableConfig {
    /// RoPE base frequency (theta). Default: 10,000 (standard transformer).
    /// Models like Qwen3 use 1,000,000; Llama uses 500,000.
    pub rope_theta: f64,
    /// Dimension per attention head. Default: 64 (standard transformer).
    /// Models like Qwen3-0.6B use 128; Llama-2 uses 128.
    pub head_dim: usize,
    /// Maximum sequence length (determines table rows). Default: 2048.
    pub seq_len: usize,
}

impl Default for RopeTableConfig {
    fn default() -> Self {
        Self {
            rope_theta: 10_000.0,
            head_dim: 64,
            seq_len: 2048,
        }
    }
}

impl RopeTableConfig {
    pub fn new(rope_theta: f64, head_dim: usize, seq_len: usize) -> Self {
        Self { rope_theta, head_dim, seq_len }
    }
}

/// Weight resolver that computes and caches static RoPE tables on demand.
///
/// This resolver handles `value_path` strings of the form:
/// `static_tables/{tables_ref}/sin_tab`
/// `static_tables/{tables_ref}/cos_tab`
/// `static_tables/{tables_ref}/eye_tab`
/// `static_tables/{tables_ref}/mask_tab`
///
/// It computes the tables in float64 precision and converts to fp16 for
/// storage, matching the HuggingFace reference implementation's approach
/// of computing in float32 and storing as fp16.
///
/// Unknown value_paths are returned as `None`, allowing a chained resolver
/// (e.g., `SafetensorsWeightResolver`) to handle them.
#[derive(Debug, Clone)]
pub struct StaticTableResolver {
    /// Configuration for table computation.
    config: RopeTableConfig,
    /// Lazily computed tables, keyed by value_path.
    cache: HashMap<String, WeightData>,
}

impl StaticTableResolver {
    /// Create a new static table resolver with the given configuration.
    pub fn new(config: RopeTableConfig) -> Self {
        Self { config, cache: HashMap::new() }
    }

    /// Create a resolver with parameters from a model configuration.
    /// This is the preferred way to create a resolver — it reads all
    /// parameters from the model config rather than hardcoding model-specific values.
    pub fn from_model_config(rope_theta: f64, head_dim: usize, seq_len: usize) -> Self {
        Self::new(RopeTableConfig {
            rope_theta,
            head_dim,
            seq_len,
        })
    }

    /// Ensure the tables for a given tables_ref are computed and cached.
    ///
    /// All four tables (sin, cos, eye, mask) share the same seq_len and
    /// head_dim parameters, so we compute them together and cache each
    /// under its value_path key.
    pub fn ensure_tables_computed(&mut self, tables_ref: &str) {
        // Check if any table for this ref is already cached
        let sin_key = format!("static_tables/{}/sin_tab", tables_ref);
        if self.cache.contains_key(&sin_key) {
            return;
        }

        let seq = self.config.seq_len;
        let hd = self.config.head_dim;
        let theta = self.config.rope_theta;

        // Step 1: Compute inverse frequencies
        // inv_freq[i] = 1 / theta^(2i/d) for i = 0..hd/2-1
        let half_dim = hd / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let exponent = (2.0 * i as f64) / (hd as f64);
            inv_freq.push(1.0 / theta.powf(exponent));
        }

        // Step 2: Compute freqs = position × inv_freq
        // freqs[pos, i] = pos * inv_freq[i]
        // emb[pos, :] = cat(freqs[pos, :], freqs[pos, :])  — duplicate to full head_dim
        // cos/sin = cos(emb) / sin(emb)
        //
        // Shape: sin_tab and cos_tab = [1, 1, seq_len, head_dim] in fp16
        // The [1, 1, S, D] shape broadcasts correctly with Q/K [1, H, S, D]
        // where H = num_heads. Core ML broadcasting aligns right-to-left.
        let mut sin_bytes = Vec::with_capacity(1 * 1 * seq * hd * 2);
        let mut cos_bytes = Vec::with_capacity(1 * 1 * seq * hd * 2);

        for _batch in 0..1 {
            for _heads in 0..1 {
                for pos in 0..seq {
                    for i in 0..hd {
                        // Determine the frequency index: first half and second half
                        // share the same frequencies (duplicated)
                        let freq_idx = i % half_dim;
                        let angle = pos as f64 * inv_freq[freq_idx];

                        let sin_val = half::f16::from_f64(angle.sin());
                        let cos_val = half::f16::from_f64(angle.cos());

                        sin_bytes.extend_from_slice(&sin_val.to_bits().to_le_bytes());
                        cos_bytes.extend_from_slice(&cos_val.to_bits().to_le_bytes());
                    }
                }
            }
        }

        // Step 3: Compute identity table (eye_tab) — [seq, seq] fp16
        // Precomputed identity matrix for KV-cache write masking.
        // The decode_step path uses Gather(eye_tab, pos, axis=0) to get
        // a one-hot row for the write position — this is ANE-legal and
        // replaces the ANE-illegal Equal+Cast pattern.
        //
        // ONLY compute when seq_len is small enough to be practical.
        // For seq_len > 8192, the eye_tab would be > 128 MB (seq×seq×2 bytes)
        // which may be too large for mobile deployment. The embedding path
        // (small seq_len) always uses eye_tab for prefill attention masking.
        // The decode_step path requires it for KV write masking.
        let compute_large_tables = seq <= 8192;

        let mut eye_bytes = Vec::new();
        if compute_large_tables {
            eye_bytes = Vec::with_capacity(seq * seq * 2);
            for row in 0..seq {
                for col in 0..seq {
                    let val = half::f16::from_f64(if row == col { 1.0 } else { 0.0 });
                    eye_bytes.extend_from_slice(&val.to_bits().to_le_bytes());
                }
            }
        }

        // Step 4: Compute causal mask (mask_tab) — [seq, seq] fp16
        // Precomputed causal attention mask for additive masking.
        // The decode_step path uses Gather(mask_tab, pos, axis=0) to get
        // the causal mask row for the current position, then applies it
        // via mb.add(logits, mask) — fully ANE-legal, replacing the
        // ANE-illegal LessEqual+Fill+Select pattern.
        //
        // Same seq_len guard as eye_tab — mask_tab is [seq, seq] and grows
        // quadratically. The decode_step ALWAYS uses the precomputed table
        // approach now (ISSUE-001 fix), so this table is mandatory for
        // seq_len ≤ 8192.
        let mut mask_bytes = Vec::new();
        if compute_large_tables {
            // Upper-triangular: the last (idx+1) positions are unmasked (0.0),
            // the rest are -inf. This matches the reversed ring-buffer pattern
            // from the HuggingFace reference.
            // mask_tab[idx, seq-(idx+1):] = 0.0, rest = -inf
            let neg_inf_f16 = half::f16::from_f64(f64::NEG_INFINITY);
            let zero_f16 = half::f16::from_f64(0.0);
            mask_bytes = Vec::with_capacity(seq * seq * 2);
            for idx in 0..seq {
                let unmask_start = seq - (idx + 1);
                for col in 0..seq {
                    let val = if col >= unmask_start { zero_f16 } else { neg_inf_f16 };
                    mask_bytes.extend_from_slice(&val.to_bits().to_le_bytes());
                }
            }
        }

        // Step 5: Compute arange table (arange_tab) — [seq_len] int32
        // Used for computing one-hot KV write masks and causal masks at runtime
        // via Equal/LessEqual + Cast/Select, instead of storing huge [seq, seq]
        // eye_tab/mask_tab tables (which would be 3+ GB for seq_len=40960).
        let mut arange_bytes = Vec::with_capacity(seq * 4);
        for i in 0..seq {
            arange_bytes.extend_from_slice(&(i as i32).to_le_bytes());
        }

        // Step 6: Compute fp16 arange table (arange_fp16_tab) — [seq_len] fp16
        // Used for pure-arithmetic mask computation without Gather.
        // Replaces the Gather(eye_tab, pos) and Gather(mask_tab, pos) pattern
        // with Abs/Sub/Minimum/Maximum-based computation that is fully ANE-legal
        // and works for any seq_len (no quadratic memory cost).
        let mut arange_fp16_bytes = Vec::with_capacity(seq * 2);
        for i in 0..seq {
            let val = half::f16::from_f64(i as f64);
            arange_fp16_bytes.extend_from_slice(&val.to_bits().to_le_bytes());
        }

        // Cache all tables
        // cos/sin shape: [1, 1, seq_len, head_dim] — broadcasts with [B, H, S, D]
        self.cache.insert(
            format!("static_tables/{}/sin_tab", tables_ref),
            WeightData { data: sin_bytes, shape: vec![1, 1, seq, hd] },
        );
        self.cache.insert(
            format!("static_tables/{}/cos_tab", tables_ref),
            WeightData { data: cos_bytes, shape: vec![1, 1, seq, hd] },
        );
        if compute_large_tables {
            self.cache.insert(
                format!("static_tables/{}/eye_tab", tables_ref),
                WeightData { data: eye_bytes, shape: vec![seq, seq] },
            );
            self.cache.insert(
                format!("static_tables/{}/mask_tab", tables_ref),
                WeightData { data: mask_bytes, shape: vec![seq, seq] },
            );
        }
        // arange shape: [seq_len] int32 — position indices for mask computation
        self.cache.insert(
            format!("static_tables/{}/arange_tab", tables_ref),
            WeightData { data: arange_bytes, shape: vec![seq] },
        );
        // arange_fp16 shape: [seq_len] fp16 — position indices for arithmetic mask computation
        self.cache.insert(
            format!("static_tables/{}/arange_fp16_tab", tables_ref),
            WeightData { data: arange_fp16_bytes, shape: vec![seq] },
        );
    }
}

impl StaticTableResolver {
    /// Resolve a scalar constant from a value_path of the form `scalar://fp16/{value}`
    /// or `scalar://fp32/{value}`.
    ///
    /// This produces a 1-element WeightData with the scalar value,
    /// which broadcasts correctly with tensors of any shape in
    /// CoreML MIL `mb.add`, `mb.mul`, etc.
    fn resolve_scalar(value_path: &str) -> Option<WeightData> {
        if let Some(rest) = value_path.strip_prefix("scalar://fp16/") {
            let val: f32 = rest.parse().ok()?;
            let f16_val = half::f16::from_f32(val);
            let mut bytes = Vec::with_capacity(2);
            bytes.extend_from_slice(&f16_val.to_bits().to_le_bytes());
            Some(WeightData { data: bytes, shape: vec![1] })
        } else if let Some(rest) = value_path.strip_prefix("scalar://fp32/") {
            let val: f32 = rest.parse().ok()?;
            let mut bytes = Vec::with_capacity(4);
            bytes.extend_from_slice(&val.to_le_bytes());
            Some(WeightData { data: bytes, shape: vec![1] })
        } else {
            None
        }
    }
}

impl WeightResolver for StaticTableResolver {
    fn resolve(&self, value_path: &str) -> Option<WeightData> {
        // First check the cache
        if let Some(data) = self.cache.get(value_path) {
            return Some(data.clone());
        }

        // Try scalar constant resolution
        if let Some(data) = Self::resolve_scalar(value_path) {
            return Some(data);
        }

        // If it's a static_tables path but not cached, it won't be found
        // (we can't mutably compute here). The caller should use
        // resolve_or_compute() instead, or pre-compute tables.
        None
    }
}

/// A composite resolver that chains a `StaticTableResolver` with another resolver.
///
/// Static table paths (starting with `static_tables/`) are handled by the
/// static table resolver; all other paths fall through to the fallback resolver
/// (typically a `SafetensorsWeightResolver`).
#[derive(Debug, Clone)]
pub struct ChainedResolver<FB: WeightResolver> {
    static_tables: StaticTableResolver,
    fallback: FB,
}

impl<FB: WeightResolver> ChainedResolver<FB> {
    /// Create a new chained resolver.
    pub fn new(static_tables: StaticTableResolver, fallback: FB) -> Self {
        Self { static_tables, fallback }
    }

    /// Pre-compute all static tables for the given tables_refs.
    ///
    /// Call this before the first `resolve()` to ensure all static table
    /// entries are cached and available.
    pub fn precompute_tables(&mut self, tables_refs: &[&str]) {
        for tables_ref in tables_refs {
            self.static_tables.ensure_tables_computed(tables_ref);
        }
    }

    /// Get a reference to the fallback resolver.
    pub fn fallback(&self) -> &FB {
        &self.fallback
    }

    /// Get a mutable reference to the static table resolver.
    pub fn static_tables_mut(&mut self) -> &mut StaticTableResolver {
        &mut self.static_tables
    }
}

impl<FB: WeightResolver> WeightResolver for ChainedResolver<FB> {
    fn resolve(&self, value_path: &str) -> Option<WeightData> {
        // Try static tables first
        if let Some(data) = self.static_tables.resolve(value_path) {
            return Some(data);
        }
        // Fall through to the model weight resolver
        self.fallback.resolve(value_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_table_computation() {
        let mut resolver = StaticTableResolver::new(RopeTableConfig {
            rope_theta: 1_000_000.0,
            head_dim: 128,
            seq_len: 8, // Small for testing
        });
        resolver.ensure_tables_computed("rope_tables_0");

        // Check sin_tab
        let sin_data = resolver.resolve("static_tables/rope_tables_0/sin_tab").unwrap();
        assert_eq!(sin_data.shape, vec![1, 1, 8, 128]);
        assert_eq!(sin_data.data.len(), 1 * 1 * 8 * 128 * 2); // fp16 = 2 bytes

        // Check cos_tab
        let cos_data = resolver.resolve("static_tables/rope_tables_0/cos_tab").unwrap();
        assert_eq!(cos_data.shape, vec![1, 1, 8, 128]);
        assert_eq!(cos_data.data.len(), 1 * 1 * 8 * 128 * 2);

        // Check eye_tab
        let eye_data = resolver.resolve("static_tables/rope_tables_0/eye_tab").unwrap();
        assert_eq!(eye_data.shape, vec![8, 8]);
        assert_eq!(eye_data.data.len(), 8 * 8 * 2);

        // Check mask_tab
        let mask_data = resolver.resolve("static_tables/rope_tables_0/mask_tab").unwrap();
        assert_eq!(mask_data.shape, vec![8, 8]);
        assert_eq!(mask_data.data.len(), 8 * 8 * 2);
    }

    #[test]
    fn test_cos_sin_values_at_position_zero() {
        let mut resolver = StaticTableResolver::new(RopeTableConfig {
            rope_theta: 1_000_000.0,
            head_dim: 128,
            seq_len: 4,
        });
        resolver.ensure_tables_computed("rope_tables_test");

        let cos_data = resolver.resolve("static_tables/rope_tables_test/cos_tab").unwrap();
        let sin_data = resolver.resolve("static_tables/rope_tables_test/sin_tab").unwrap();

        // At position 0, angle = 0 for all frequencies → cos = 1.0, sin = 0.0
        // Read first element (position 0, dim 0) as fp16
        let cos_first = half::f16::from_bits(u16::from_le_bytes([
            cos_data.data[0], cos_data.data[1],
        ]));
        let sin_first = half::f16::from_bits(u16::from_le_bytes([
            sin_data.data[0], sin_data.data[1],
        ]));

        assert!((cos_first.to_f32() - 1.0).abs() < 0.01, "cos(0) should be ~1.0, got {}", cos_first.to_f32());
        assert!(sin_first.to_f32().abs() < 0.01, "sin(0) should be ~0.0, got {}", sin_first.to_f32());
    }

    #[test]
    fn test_chained_resolver() {
        use crate::mir_to_compat::HashMapWeightResolver;

        let mut static_resolver = StaticTableResolver::new(RopeTableConfig {
            rope_theta: 1_000_000.0,
            head_dim: 128,
            seq_len: 4,
        });
        static_resolver.ensure_tables_computed("rope_tables_0");

        let mut fallback = HashMapWeightResolver::new();
        fallback.add("model.weight".to_string(), vec![1, 2, 3, 4], vec![2, 2]);

        let chained = ChainedResolver::new(static_resolver, fallback);

        // Static table path → resolved by static tables
        assert!(chained.resolve("static_tables/rope_tables_0/sin_tab").is_some());

        // Model weight path → resolved by fallback
        let weight = chained.resolve("model.weight").unwrap();
        assert_eq!(weight.data, vec![1, 2, 3, 4]);

        // Unknown path → None
        assert!(chained.resolve("nonexistent").is_none());
    }

    #[test]
    fn test_non_static_path_returns_none() {
        let resolver = StaticTableResolver::new(RopeTableConfig::default());
        assert!(resolver.resolve("model.layers.0.weight").is_none());
    }

    #[test]
    fn test_scalar_fp16_resolution() {
        let resolver = StaticTableResolver::new(RopeTableConfig::default());
        let data = resolver.resolve("scalar://fp16/0.000001").unwrap();
        assert_eq!(data.shape, vec![1]);
        assert_eq!(data.data.len(), 2); // fp16 = 2 bytes
        let val = half::f16::from_bits(u16::from_le_bytes([data.data[0], data.data[1]]));
        assert!((val.to_f32() - 0.000001).abs() < 1e-7, "Expected ~1e-6, got {}", val.to_f32());
    }

    #[test]
    fn test_scalar_fp32_resolution() {
        let resolver = StaticTableResolver::new(RopeTableConfig::default());
        let data = resolver.resolve("scalar://fp32/1.5").unwrap();
        assert_eq!(data.shape, vec![1]);
        assert_eq!(data.data.len(), 4); // fp32 = 4 bytes
        let val = f32::from_le_bytes([data.data[0], data.data[1], data.data[2], data.data[3]]);
        assert!((val - 1.5).abs() < 1e-7);
    }

    #[test]
    fn test_scalar_zero_resolution() {
        let resolver = StaticTableResolver::new(RopeTableConfig::default());
        let data = resolver.resolve("scalar://fp16/0").unwrap();
        let val = half::f16::from_bits(u16::from_le_bytes([data.data[0], data.data[1]]));
        assert_eq!(val.to_f32(), 0.0);
    }

    #[test]
    fn test_scalar_invalid_path_returns_none() {
        let resolver = StaticTableResolver::new(RopeTableConfig::default());
        assert!(resolver.resolve("scalar://fp16/notanumber").is_none());
        assert!(resolver.resolve("scalar://int32/5").is_none());
    }
}
