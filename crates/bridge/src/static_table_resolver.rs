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
//! | eye_tab | [seq_len, seq_len] | float16 | Identity for KV-cache ring buffer (legacy, large) |
//! | mask_tab | [seq_len, seq_len] | float16 | Causal attention mask (legacy, large) |
//! | arange_tab | [seq_len] | int32 | Position indices for computed masks |
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

    /// Create a resolver with parameters typical for modern large language models
    /// (high rope_theta, large head_dim). The caller should provide the actual
    /// model-specific parameters via `RopeTableConfig::new()` when available.
    #[deprecated(note = "Use RopeTableConfig::new() with model-specific parameters instead")]
    pub fn for_qwen3_0_6b(seq_len: usize) -> Self {
        Self::new(RopeTableConfig {
            rope_theta: 1_000_000.0,
            head_dim: 128,
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
        let mut eye_bytes = Vec::with_capacity(seq * seq * 2);
        for row in 0..seq {
            for col in 0..seq {
                let val = half::f16::from_f64(if row == col { 1.0 } else { 0.0 });
                eye_bytes.extend_from_slice(&val.to_bits().to_le_bytes());
            }
        }

        // Step 4: Compute causal mask (mask_tab) — [seq, seq] fp16
        // Upper-triangular: the last (idx+1) positions are unmasked (0.0),
        // the rest are -inf. This matches the reversed ring-buffer pattern
        // from the HuggingFace reference.
        // mask_tab[idx, seq-(idx+1):] = 0.0, rest = -inf
        let neg_inf_f16 = half::f16::from_f64(f64::NEG_INFINITY);
        let zero_f16 = half::f16::from_f64(0.0);
        let mut mask_bytes = Vec::with_capacity(seq * seq * 2);
        for idx in 0..seq {
            let unmask_start = seq - (idx + 1);
            for col in 0..seq {
                let val = if col >= unmask_start { zero_f16 } else { neg_inf_f16 };
                mask_bytes.extend_from_slice(&val.to_bits().to_le_bytes());
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
        self.cache.insert(
            format!("static_tables/{}/eye_tab", tables_ref),
            WeightData { data: eye_bytes, shape: vec![seq, seq] },
        );
        self.cache.insert(
            format!("static_tables/{}/mask_tab", tables_ref),
            WeightData { data: mask_bytes, shape: vec![seq, seq] },
        );
        // arange shape: [seq_len] int32 — position indices for mask computation
        self.cache.insert(
            format!("static_tables/{}/arange_tab", tables_ref),
            WeightData { data: arange_bytes, shape: vec![seq] },
        );
    }
}

impl WeightResolver for StaticTableResolver {
    fn resolve(&self, value_path: &str) -> Option<WeightData> {
        // First check the cache
        if let Some(data) = self.cache.get(value_path) {
            return Some(data.clone());
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
}
