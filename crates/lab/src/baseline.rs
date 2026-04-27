//! Baseline Computation
//!
//! Computes reference outputs for profiling tasks using
//! deterministic FP32 CPU computation. The baseline is the
//! ground-truth reference against which drift is measured.
//!
//! For synthetic linear projection tasks, the baseline is computed
//! by: (1) materializing weight and bias tensors from a deterministic
//! seed, (2) performing the matrix multiply + bias add in FP32.
//!
//! Note: The MIL emission path now uses `mb.linear` instead of separate
//! matmul + add ops (Sprint 31), but the baseline computation remains
//! `x @ W + b` since it is the mathematical reference — `mb.linear` and
//! `x @ W + b` are semantically equivalent. Drift measured against this
//! baseline captures quantization error, not structural differences.
//!
//! The same task spec + seed always produces the same baseline output,
//! ensuring reproducibility and stable artifact identity.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Baseline computation engine.
///
/// Computes deterministic reference outputs from task specifications
/// using pure FP32 arithmetic. No ANE, no Core ML, no GPU — just
/// honest host-side linear algebra.
pub struct BaselineComputer {
    /// Random seed for deterministic weight/bias generation.
    pub seed: u64,
}

impl BaselineComputer {
    /// Create a new baseline computer with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Create a baseline computer with the default seed (42).
    pub fn default_seed() -> Self {
        Self::new(42)
    }

    /// Compute a baseline for a linear projection task.
    ///
    /// Given the task dimensions (input_dim, output_dim, batch_size),
    /// this materializes deterministic weight and bias tensors, then
    /// computes `y = x @ W + b` in FP32, where `x` is a deterministic
    /// input tensor.
    ///
    /// The result includes the task_id linkage, the FP32 output tensor
    /// (flattened), and metadata about the computation.
    pub fn compute_linear_projection(
        &self,
        task_id: &str,
        input_dim: usize,
        output_dim: usize,
        batch_size: usize,
    ) -> Result<BaselineResult> {
        let start = std::time::Instant::now();

        // Materialize deterministic weight matrix [input_dim, output_dim]
        let weights = deterministic_tensor_2d(self.seed, input_dim, output_dim);

        // Materialize deterministic bias vector [output_dim]
        let bias = deterministic_tensor_1d(self.seed.wrapping_add(1), output_dim);

        // Materialize deterministic input [batch_size, input_dim]
        let input = deterministic_tensor_2d(self.seed.wrapping_add(2), batch_size, input_dim);

        // Compute y = x @ W + b in FP32
        // Result shape: [batch_size, output_dim]
        let mut output = Vec::with_capacity(batch_size * output_dim);
        for b in 0..batch_size {
            for j in 0..output_dim {
                let mut sum = 0.0f32;
                for k in 0..input_dim {
                    sum += input[b * input_dim + k] * weights[k * output_dim + j];
                }
                sum += bias[j];
                output.push(sum);
            }
        }

        let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(BaselineResult {
            task_id: task_id.to_string(),
            task_hash: None, // Set by caller from compute_task_hash
            input_dim,
            output_dim,
            batch_size,
            seed: self.seed,
            precision: "fp32".to_string(),
            output_tensor: output,
            output_shape: vec![batch_size, output_dim],
            compute_time_ms,
            baseline_schema_version: BASELINE_SCHEMA_VERSION.to_string(),
        })
    }

    /// Compute a baseline for a decode-step task.
    ///
    /// Models the three-part decode-step pattern (QKV projection →
    /// attention → output projection) in FP32. This is a simplified
    /// but honest model of what a transformer decode step computes:
    ///
    /// 1. QKV projection: x @ W_qkv → Q, K, V tensors
    /// 2. Scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V
    /// 3. Output projection: attn_out @ W_out → output
    ///
    /// The computation uses deterministic weight matrices and input
    /// tensors derived from the seed. The attention is computed with
    /// single-head semantics for simplicity (multi-head would partition
    /// the same computation into subspaces but produce the same overall
    /// output shape).
    ///
    /// **Note**: This baseline models the mathematical decode-step
    /// computation, not the exact MIL emission path. Drift measured
    /// against this baseline captures both quantization error and any
    /// structural approximation in the emission path.
    pub fn compute_decode_step(
        &self,
        task_id: &str,
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        kv_len: usize,
        batch_size: usize,
    ) -> Result<BaselineResult> {
        let start = std::time::Instant::now();

        let qkv_dim = 3 * embed_dim; // Concatenated Q, K, V projections

        // Materialize deterministic weight matrices
        // W_qkv: [embed_dim, qkv_dim] — projects input to Q, K, V
        let w_qkv = deterministic_tensor_2d(self.seed, embed_dim, qkv_dim);
        // W_out: [embed_dim, embed_dim] — output projection
        let w_out = deterministic_tensor_2d(self.seed.wrapping_add(3), embed_dim, embed_dim);

        // Materialize deterministic input: [batch_size, embed_dim]
        let input = deterministic_tensor_2d(self.seed.wrapping_add(2), batch_size, embed_dim);

        // Materialize deterministic KV cache for context: [2, batch_size, num_heads, kv_len, head_dim]
        // We only use K and V from the cache for attention computation.
        // For simplicity, we generate K and V as deterministic tensors of the right shape.
        let k_cache = deterministic_tensor_2d(
            self.seed.wrapping_add(4),
            batch_size * num_heads * kv_len,
            head_dim,
        );
        let v_cache = deterministic_tensor_2d(
            self.seed.wrapping_add(5),
            batch_size * num_heads * kv_len,
            head_dim,
        );

        // Step 1: QKV projection — x @ W_qkv
        // Result shape: [batch_size, qkv_dim]
        let mut qkv_output = Vec::with_capacity(batch_size * qkv_dim);
        for b in 0..batch_size {
            for j in 0..qkv_dim {
                let mut sum = 0.0f32;
                for k in 0..embed_dim {
                    sum += input[b * embed_dim + k] * w_qkv[k * qkv_dim + j];
                }
                qkv_output.push(sum);
            }
        }

        // Extract Q from the QKV output: [batch_size, embed_dim]
        // Q occupies columns [0, embed_dim) of the qkv output
        let q: Vec<f32> = (0..batch_size)
            .flat_map(|b| {
                let qkv = &qkv_output;
                (0..embed_dim).map(move |j| qkv[b * qkv_dim + j])
            })
            .collect();

        // Step 2: Simplified scaled dot-product attention
        // For the baseline, we compute a single-head attention for simplicity.
        // Q: [batch_size, embed_dim], K_cache: flattened, V_cache: flattened
        //
        // We compute: attn_out = softmax(Q @ K^T / sqrt(head_dim)) @ V
        // where K and V are from the cache, treating the full embed_dim as
        // the query/key dimension.
        //
        // For efficiency with the flattened cache, we compute attention
        // using the first head's K and V (simplified but deterministic).
        // The output shape is [batch_size, embed_dim].

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute attention scores: Q @ K^T for cached keys
        // Using K from the first head: shape [kv_len, head_dim]
        // Simplified: use the full embed_dim as the key space
        // Q_reduced: [batch_size, head_dim], K_reduced: [kv_len, head_dim]
        // We take the first head_dim columns of Q and first head of K cache
        let mut attn_output = Vec::with_capacity(batch_size * embed_dim);

        for b in 0..batch_size {
            // Attention scores for this batch element: [kv_len]
            let mut scores = Vec::with_capacity(kv_len);
            for t in 0..kv_len {
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    let q_val = q[b * embed_dim + d]; // First head_dim of Q
                    let k_val = k_cache[(b * num_heads * kv_len + t) * head_dim + d]; // First head of K cache
                    score += q_val * k_val;
                }
                scores.push(score * scale);
            }

            // Softmax over scores
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let attn_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

            // Weighted sum of V: [embed_dim]
            // We replicate the attention across all heads to produce the full output.
            // For each output dimension, we compute the weighted sum from the V cache.
            // Simplified: we compute attention output for the first head and
            // replicate/tile across heads, then project through W_out.
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for t in 0..kv_len {
                        let v_val = v_cache[(b * num_heads * kv_len + h * kv_len + t) * head_dim + d];
                        val += attn_weights[t] * v_val;
                    }
                    attn_output.push(val);
                }
            }
        }

        // Step 3: Output projection — attn_out @ W_out
        // attn_output: [batch_size, embed_dim], W_out: [embed_dim, embed_dim]
        let mut output = Vec::with_capacity(batch_size * embed_dim);
        for b in 0..batch_size {
            for j in 0..embed_dim {
                let mut sum = 0.0f32;
                for k in 0..embed_dim {
                    sum += attn_output[b * embed_dim + k] * w_out[k * embed_dim + j];
                }
                output.push(sum);
            }
        }

        let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(BaselineResult {
            task_id: task_id.to_string(),
            task_hash: None,
            input_dim: embed_dim,
            output_dim: embed_dim,
            batch_size,
            seed: self.seed,
            precision: "fp32".to_string(),
            output_tensor: output,
            output_shape: vec![batch_size, embed_dim],
            compute_time_ms,
            baseline_schema_version: BASELINE_SCHEMA_VERSION.to_string(),
        })
    }

    /// Compute a baseline for an MLP block task.
    ///
    /// Models the fused feed-forward network block pattern:
    /// - Up-projection: input @ W_up -> [batch_size, hidden_dim]
    /// - Activation: GELU or ReLU on the up-projected result
    /// - Down-projection: activated @ W_down -> [batch_size, output_dim]
    ///
    /// GELU approximation: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ///
    /// Weights are deterministic (based on seed), and the same
    /// task spec + seed always produces the same baseline output.
    pub fn compute_mlp_block(
        &self,
        task_id: &str,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        activation: &str,
        batch_size: usize,
    ) -> Result<BaselineResult> {
        let start = std::time::Instant::now();

        // Materialize deterministic weight matrices
        // W_up: [input_dim, hidden_dim] — up-projection
        let w_up = deterministic_tensor_2d(self.seed, input_dim, hidden_dim);
        // W_down: [hidden_dim, output_dim] — down-projection
        let w_down = deterministic_tensor_2d(self.seed.wrapping_add(10), hidden_dim, output_dim);

        // Materialize deterministic input: [batch_size, input_dim]
        let input = deterministic_tensor_2d(self.seed.wrapping_add(2), batch_size, input_dim);

        // Step 1: Up-projection — input @ W_up -> [batch_size, hidden_dim]
        let mut up_projected = Vec::with_capacity(batch_size * hidden_dim);
        for b in 0..batch_size {
            for j in 0..hidden_dim {
                let mut sum = 0.0f32;
                for k in 0..input_dim {
                    sum += input[b * input_dim + k] * w_up[k * hidden_dim + j];
                }
                up_projected.push(sum);
            }
        }

        // Step 2: Activation
        let mut activated = Vec::with_capacity(batch_size * hidden_dim);
        match activation {
            "gelu" => {
                // GELU approximation: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
                for &x in &up_projected {
                    let x3 = x * x * x;
                    let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
                    let gelu = x * 0.5 * (1.0 + inner.tanh());
                    activated.push(gelu);
                }
            }
            "relu" => {
                for &x in &up_projected {
                    activated.push(x.max(0.0));
                }
            }
            _ => {
                anyhow::bail!("Invalid activation '{}': must be 'gelu' or 'relu'", activation);
            }
        }

        // Step 3: Down-projection — activated @ W_down -> [batch_size, output_dim]
        let mut output = Vec::with_capacity(batch_size * output_dim);
        for b in 0..batch_size {
            for j in 0..output_dim {
                let mut sum = 0.0f32;
                for k in 0..hidden_dim {
                    sum += activated[b * hidden_dim + k] * w_down[k * output_dim + j];
                }
                output.push(sum);
            }
        }

        let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(BaselineResult {
            task_id: task_id.to_string(),
            task_hash: None,
            input_dim,
            output_dim,
            batch_size,
            seed: self.seed,
            precision: "fp32".to_string(),
            output_tensor: output,
            output_shape: vec![batch_size, output_dim],
            compute_time_ms,
            baseline_schema_version: BASELINE_SCHEMA_VERSION.to_string(),
        })
    }

    /// Compute a baseline for an attention task.
    ///
    /// Models the multi-head self-attention pattern without KV-cache
    /// semantics (unlike decode-step). The computation is:
    /// - QKV projection: x @ W_qkv → Q, K, V tensors
    /// - Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
    /// - Output projection: attn_out @ W_out → output
    ///
    /// This is a standalone attention block (no cache), modeling the
    /// pattern exercised by the Attention task family.
    ///
    /// **Note**: This baseline models the mathematical attention
    /// computation, not the exact MIL emission path. Drift measured
    /// against this baseline captures both quantization error and any
    /// structural approximation in the emission path.
    pub fn compute_attention(
        &self,
        task_id: &str,
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<BaselineResult> {
        let start = std::time::Instant::now();

        let qkv_dim = 3 * embed_dim; // Concatenated Q, K, V projections

        // Materialize deterministic weight matrices
        // W_qkv: [embed_dim, qkv_dim] — projects input to Q, K, V
        let w_qkv = deterministic_tensor_2d(self.seed, embed_dim, qkv_dim);
        // W_out: [embed_dim, embed_dim] — output projection
        let w_out = deterministic_tensor_2d(self.seed.wrapping_add(3), embed_dim, embed_dim);

        // Materialize deterministic input: [batch_size * seq_len, embed_dim]
        let input = deterministic_tensor_2d(self.seed.wrapping_add(2), batch_size * seq_len, embed_dim);

        // Step 1: QKV projection — input @ W_qkv
        // Result shape: [batch_size * seq_len, qkv_dim]
        let mut qkv_output = Vec::with_capacity(batch_size * seq_len * qkv_dim);
        for bs in 0..batch_size * seq_len {
            for j in 0..qkv_dim {
                let mut sum = 0.0f32;
                for k in 0..embed_dim {
                    sum += input[bs * embed_dim + k] * w_qkv[k * qkv_dim + j];
                }
                qkv_output.push(sum);
            }
        }

        // Extract Q, K, V from the QKV output
        // Q occupies columns [0, embed_dim)
        // K occupies columns [embed_dim, 2*embed_dim)
        // V occupies columns [2*embed_dim, 3*embed_dim)
        // Q, K, V each have shape [batch_size * seq_len, embed_dim]

        // Step 2: Multi-head scaled dot-product attention
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_output = Vec::with_capacity(batch_size * seq_len * embed_dim);

        for b in 0..batch_size {
            for h in 0..num_heads {
                // For each batch and head, compute attention over the seq_len tokens
                // Q_h: [seq_len, head_dim], K_h: [seq_len, head_dim], V_h: [seq_len, head_dim]
                // Attention scores: [seq_len, seq_len]

                // Compute attention scores: Q_h @ K_h^T / sqrt(d_k)
                let mut scores = Vec::with_capacity(seq_len * seq_len);
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            let q_idx = ((b * seq_len + i) * qkv_dim) + h * head_dim + d;
                            let k_idx = ((b * seq_len + j) * qkv_dim) + embed_dim + h * head_dim + d;
                            score += qkv_output[q_idx] * qkv_output[k_idx];
                        }
                        scores.push(score * scale);
                    }
                }

                // Softmax over each row of scores
                for i in 0..seq_len {
                    let row_start = i * seq_len;
                    let max_score = scores[row_start..row_start + seq_len]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> = scores[row_start..row_start + seq_len]
                        .iter()
                        .map(|s| (s - max_score).exp())
                        .collect();
                    let sum_exp: f32 = exp_scores.iter().sum();

                    // Weighted sum of V_h
                    for d in 0..head_dim {
                        let mut val = 0.0f32;
                        for j in 0..seq_len {
                            let v_idx = ((b * seq_len + j) * qkv_dim) + 2 * embed_dim + h * head_dim + d;
                            val += (exp_scores[j] / sum_exp) * qkv_output[v_idx];
                        }
                        // Store in the output at position [batch, seq, head, head_dim]
                        // Flattened as [batch * seq_len, embed_dim] with head-major layout
                        attn_output.push(val);
                    }
                }
            }
        }

        // Reorder attn_output from [batch][head][seq][head_dim] to [batch][seq][embed_dim]
        let mut reordered = vec![0.0f32; batch_size * seq_len * embed_dim];
        for b in 0..batch_size {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    for d in 0..head_dim {
                        // Source: batch * (num_heads * seq_len * head_dim) + head * (seq_len * head_dim) + seq * head_dim + d
                        let src_idx = b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + s * head_dim + d;
                        // Target: (batch * seq_len + seq) * embed_dim + head * head_dim + d
                        let dst_idx = (b * seq_len + s) * embed_dim + h * head_dim + d;
                        reordered[dst_idx] = attn_output[src_idx];
                    }
                }
            }
        }

        // Step 3: Output projection — attn_out @ W_out
        // reordered: [batch_size * seq_len, embed_dim], W_out: [embed_dim, embed_dim]
        let mut output = Vec::with_capacity(batch_size * seq_len * embed_dim);
        for bs in 0..batch_size * seq_len {
            for j in 0..embed_dim {
                let mut sum = 0.0f32;
                for k in 0..embed_dim {
                    sum += reordered[bs * embed_dim + k] * w_out[k * embed_dim + j];
                }
                output.push(sum);
            }
        }

        let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(BaselineResult {
            task_id: task_id.to_string(),
            task_hash: None,
            input_dim: embed_dim,
            output_dim: embed_dim,
            batch_size,
            seed: self.seed,
            precision: "fp32".to_string(),
            output_tensor: output,
            output_shape: vec![batch_size, seq_len, embed_dim],
            compute_time_ms,
            baseline_schema_version: BASELINE_SCHEMA_VERSION.to_string(),
        })
    }

    /// Compute a baseline for a sharded linear pipeline task.
    ///
    /// Models the 3-shard Entry/Interior/Exit linear pipeline in FP32:
    /// - Entry shard: input @ W_entry → [batch_size, hidden_dim]
    /// - Interior shard: entry_out @ W_interior → [batch_size, hidden_dim]
    /// - Exit shard: interior_out @ W_exit → [batch_size, output_dim]
    ///
    /// This models what a 3-shard linear pipeline computes when each shard
    /// performs a separate linear projection. The result is mathematically
    /// equivalent to composing three linear projections sequentially.
    pub fn compute_sharded_linear_pipeline(
        &self,
        task_id: &str,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        batch_size: usize,
    ) -> Result<BaselineResult> {
        let start = std::time::Instant::now();

        // Entry shard: input @ W_entry → [batch_size, hidden_dim]
        let w_entry = deterministic_tensor_2d(self.seed, input_dim, hidden_dim);
        let input = deterministic_tensor_2d(self.seed.wrapping_add(2), batch_size, input_dim);

        let mut entry_out = Vec::with_capacity(batch_size * hidden_dim);
        for b in 0..batch_size {
            for j in 0..hidden_dim {
                let mut sum = 0.0f32;
                for k in 0..input_dim {
                    sum += input[b * input_dim + k] * w_entry[k * hidden_dim + j];
                }
                entry_out.push(sum);
            }
        }

        // Interior shard: entry_out @ W_interior → [batch_size, hidden_dim]
        let w_interior = deterministic_tensor_2d(self.seed.wrapping_add(20), hidden_dim, hidden_dim);

        let mut interior_out = Vec::with_capacity(batch_size * hidden_dim);
        for b in 0..batch_size {
            for j in 0..hidden_dim {
                let mut sum = 0.0f32;
                for k in 0..hidden_dim {
                    sum += entry_out[b * hidden_dim + k] * w_interior[k * hidden_dim + j];
                }
                interior_out.push(sum);
            }
        }

        // Exit shard: interior_out @ W_exit → [batch_size, output_dim]
        let w_exit = deterministic_tensor_2d(self.seed.wrapping_add(30), hidden_dim, output_dim);

        let mut output = Vec::with_capacity(batch_size * output_dim);
        for b in 0..batch_size {
            for j in 0..output_dim {
                let mut sum = 0.0f32;
                for k in 0..hidden_dim {
                    sum += interior_out[b * hidden_dim + k] * w_exit[k * output_dim + j];
                }
                output.push(sum);
            }
        }

        let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(BaselineResult {
            task_id: task_id.to_string(),
            task_hash: None,
            input_dim,
            output_dim,
            batch_size,
            seed: self.seed,
            precision: "fp32".to_string(),
            output_tensor: output,
            output_shape: vec![batch_size, output_dim],
            compute_time_ms,
            baseline_schema_version: BASELINE_SCHEMA_VERSION.to_string(),
        })
    }

    /// Compute a baseline for a sharded decode-step task.
    ///
    /// Models the 3-shard QKV→Attention→Output pipeline in FP32,
    /// mirroring the role-specific structure of the sharded decode-step:
    /// - QKV shard: input @ W_qkv → Q, K, V projections
    /// - Attention shard: multi-head scaled dot-product attention with KV cache
    /// - Output shard: attention output @ W_out → output projection
    ///
    /// Each shard uses independent deterministic weights (seed offsets 0, 40, 50)
    /// to model the fact that in a sharded pipeline each shard has its own
    /// weight set. The KV cache is populated deterministically.
    ///
    /// This differs from `compute_decode_step` in that:
    /// 1. Each shard uses separate seed offsets (modeling independent shard weights)
    /// 2. The intermediate activations between shards are explicit
    /// 3. The computation structure mirrors the RoleMirBuilder 3-shard decomposition
    pub fn compute_sharded_decode_step(
        &self,
        task_id: &str,
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        kv_len: usize,
        batch_size: usize,
    ) -> Result<BaselineResult> {
        let start = std::time::Instant::now();

        let qkv_dim = 3 * embed_dim;

        // ── QKV Shard: input @ W_qkv ──
        // Uses seed offset 0 for QKV weights (separate from the single decode-step path)
        let w_qkv = deterministic_tensor_2d(self.seed, embed_dim, qkv_dim);
        let input = deterministic_tensor_2d(self.seed.wrapping_add(2), batch_size, embed_dim);

        let mut qkv_output = Vec::with_capacity(batch_size * qkv_dim);
        for b in 0..batch_size {
            for j in 0..qkv_dim {
                let mut sum = 0.0f32;
                for k in 0..embed_dim {
                    sum += input[b * embed_dim + k] * w_qkv[k * qkv_dim + j];
                }
                qkv_output.push(sum);
            }
        }

        // Extract Q from QKV output: [batch_size, embed_dim]
        let q: Vec<f32> = (0..batch_size)
            .flat_map(|b| {
                let qkv = &qkv_output;
                (0..embed_dim).map(move |j| qkv[b * qkv_dim + j])
            })
            .collect();

        // ── Attention Shard: multi-head scaled dot-product attention with KV cache ──
        // Uses seed offsets 40/45 for KV cache (separate from QKV and output weights)
        let k_cache = deterministic_tensor_2d(
            self.seed.wrapping_add(40),
            batch_size * num_heads * kv_len,
            head_dim,
        );
        let v_cache = deterministic_tensor_2d(
            self.seed.wrapping_add(45),
            batch_size * num_heads * kv_len,
            head_dim,
        );

        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut attn_output = Vec::with_capacity(batch_size * embed_dim);

        for b in 0..batch_size {
            for h in 0..num_heads {
                // Compute attention scores: Q_h @ K_h^T / sqrt(head_dim)
                // Q_h: [head_dim], K_h: [kv_len, head_dim]
                let mut scores = Vec::with_capacity(kv_len);
                for t in 0..kv_len {
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        let q_val = q[b * embed_dim + h * head_dim + d];
                        let k_val = k_cache[(b * num_heads * kv_len + h * kv_len + t) * head_dim + d];
                        score += q_val * k_val;
                    }
                    scores.push(score * scale);
                }

                // Softmax over scores
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let attn_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

                // Weighted sum of V: [head_dim]
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for t in 0..kv_len {
                        let v_val = v_cache[(b * num_heads * kv_len + h * kv_len + t) * head_dim + d];
                        val += attn_weights[t] * v_val;
                    }
                    attn_output.push(val);
                }
            }
        }

        // ── Output Shard: attn_out @ W_out ──
        // Uses seed offset 50 for output projection weights
        let w_out = deterministic_tensor_2d(self.seed.wrapping_add(50), embed_dim, embed_dim);

        let mut output = Vec::with_capacity(batch_size * embed_dim);
        for b in 0..batch_size {
            for j in 0..embed_dim {
                let mut sum = 0.0f32;
                for k in 0..embed_dim {
                    sum += attn_output[b * embed_dim + k] * w_out[k * embed_dim + j];
                }
                output.push(sum);
            }
        }

        let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(BaselineResult {
            task_id: task_id.to_string(),
            task_hash: None,
            input_dim: embed_dim,
            output_dim: embed_dim,
            batch_size,
            seed: self.seed,
            precision: "fp32".to_string(),
            output_tensor: output,
            output_shape: vec![batch_size, embed_dim],
            compute_time_ms,
            baseline_schema_version: BASELINE_SCHEMA_VERSION.to_string(),
        })
    }

    /// Compute a baseline for a LUT projection task.
    ///
    /// Models the grouped scalar-LUT palettized projection pattern:
    /// - Generates deterministic LUT tables (one per group)
    /// - Generates deterministic input indices
    /// - For each index, looks up the corresponding LUT value
    /// - Concatenates results across groups
    ///
    /// This is the FP32 reference for what a palettized projection
    /// would produce. The output shape is [batch_size, embed_dim].
    ///
    /// **Note**: This models the LUT-gather pattern, not a full dense
    /// linear projection. The baseline represents what the LUT-quantized
    /// weights would produce, not what a full-precision matmul would.
    /// Drift measured against this baseline captures quantization error
    /// from the LUT approximation, not from the compile path.
    pub fn compute_lut_projection(
        &self,
        task_id: &str,
        vocab_size: usize,
        embed_dim: usize,
        num_groups: usize,
        lut_bitwidth: usize,
        batch_size: usize,
    ) -> Result<BaselineResult> {
        let start = std::time::Instant::now();

        let group_size = embed_dim.max(num_groups) / num_groups;
        let lut_entries_per_group = vocab_size.min(1usize << lut_bitwidth);

        // Generate deterministic LUT tables: one table per group.
        // Each table has lut_entries_per_group entries of FP32 values.
        let mut lut_tables = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let table_seed = self.seed.wrapping_add(g as u64 * 100);
            let table = deterministic_tensor_1d(table_seed, lut_entries_per_group);
            lut_tables.push(table);
        }

        // Generate deterministic input indices [batch_size, embed_dim].
        // Each index is in range [0, lut_entries_per_group).
        let index_seed = self.seed.wrapping_add(10000);
        let raw_indices = deterministic_tensor_1d(index_seed, batch_size * embed_dim);

        // Compute output: for each element, look up the LUT value.
        let mut output = Vec::with_capacity(batch_size * embed_dim);
        for b in 0..batch_size {
            for e in 0..embed_dim {
                let group_idx = (e / group_size).min(num_groups - 1);
                let raw_val = raw_indices[b * embed_dim + e];
                // Map raw value to an index in [0, lut_entries_per_group)
                let idx = ((raw_val.abs() * (lut_entries_per_group as f32 - 1.0)) as usize)
                    .min(lut_entries_per_group - 1);
                let lut_val = lut_tables.get(group_idx)
                    .and_then(|t| t.get(idx))
                    .copied()
                    .unwrap_or(0.0);
                output.push(lut_val);
            }
        }

        let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(BaselineResult {
            task_id: task_id.to_string(),
            task_hash: None,
            input_dim: embed_dim,  // For LUT projection, input_dim = embed_dim
            output_dim: embed_dim, // For LUT projection, output_dim = embed_dim
            batch_size,
            seed: self.seed,
            precision: "fp32".to_string(),
            output_tensor: output,
            output_shape: vec![batch_size, embed_dim],
            compute_time_ms,
            baseline_schema_version: BASELINE_SCHEMA_VERSION.to_string(),
        })
    }
}

/// Schema version for the baseline artifact format.
/// Increment when the serialized structure changes incompatibly.
pub const BASELINE_SCHEMA_VERSION: &str = "1.0.0";

/// Result of a baseline computation.
///
/// This is a stable artifact format. The same task_id + seed + dimensions
/// always produces the same output_tensor values, making the baseline
/// reproducible and linkable to task identity.
///
/// The `task_hash` field links this baseline to the deterministic task
/// identity used throughout the system (manifest, knowledge update, lab run).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineResult {
    /// Schema version of this baseline format.
    pub baseline_schema_version: String,
    /// Task identifier (typically the task name from the spec).
    pub task_id: String,
    /// Deterministic task identity hash (sha256:<hex>), matching manifest/knowledge.
    /// Set by the caller after computing the task hash.
    pub task_hash: Option<String>,
    /// Input dimension of the linear projection.
    pub input_dim: usize,
    /// Output dimension of the linear projection.
    pub output_dim: usize,
    /// Batch size of the linear projection.
    pub batch_size: usize,
    /// Seed used for deterministic weight/input generation.
    pub seed: u64,
    /// Precision used for baseline computation (always "fp32").
    pub precision: String,
    /// The baseline output tensor, flattened in row-major order.
    /// Shape is [batch_size, output_dim].
    pub output_tensor: Vec<f32>,
    /// Shape of the output tensor.
    pub output_shape: Vec<usize>,
    /// Time taken to compute the baseline, in milliseconds.
    pub compute_time_ms: f64,
}

impl BaselineResult {
    /// Serialize this baseline to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Write this baseline to a JSON file.
    pub fn write_to_file(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Get the reference output as a flat slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.output_tensor
    }
}

/// Generate a deterministic 1D tensor using a simple LCG PRNG.
///
/// The values are in the range [-0.5, 0.5] to simulate typical
/// neural network weight magnitudes.
fn deterministic_tensor_1d(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    let mut result = Vec::with_capacity(len);
    for _ in 0..len {
        state = lcg_next(state);
        // Map u64 to f32 in [-0.5, 0.5]
        let normalized = (state as f64 / u64::MAX as f64) - 0.5;
        result.push(normalized as f32);
    }
    result
}

/// Generate a deterministic 2D tensor (row-major) using a simple LCG PRNG.
fn deterministic_tensor_2d(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
    deterministic_tensor_1d(seed, rows * cols)
}

/// Simple Linear Congruential Generator step.
/// Uses the same parameters as Numerical Recipes (but with u64).
/// Not cryptographically secure — only used for reproducible test data.
fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_baseline() {
        // Same inputs → same output
        let computer = BaselineComputer::new(42);
        let result1 = computer.compute_linear_projection("test", 4, 3, 1).unwrap();
        let result2 = computer.compute_linear_projection("test", 4, 3, 1).unwrap();
        assert_eq!(result1.output_tensor, result2.output_tensor,
            "Same seed and dimensions must produce identical baselines");
    }

    #[test]
    fn test_different_seeds_different_output() {
        let computer1 = BaselineComputer::new(42);
        let computer2 = BaselineComputer::new(99);
        let result1 = computer1.compute_linear_projection("test", 4, 3, 1).unwrap();
        let result2 = computer2.compute_linear_projection("test", 4, 3, 1).unwrap();
        assert_ne!(result1.output_tensor, result2.output_tensor,
            "Different seeds must produce different baselines");
    }

    #[test]
    fn test_baseline_shape() {
        let computer = BaselineComputer::new(42);
        let result = computer.compute_linear_projection("test", 64, 32, 1).unwrap();
        assert_eq!(result.output_tensor.len(), 32, "batch=1, output_dim=32 → 32 values");
        assert_eq!(result.output_shape, vec![1, 32]);
    }

    #[test]
    fn test_baseline_serialization() {
        let computer = BaselineComputer::new(42);
        let result = computer.compute_linear_projection("test", 4, 3, 1).unwrap();
        let json = result.to_json().unwrap();
        let parsed: BaselineResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.output_tensor, result.output_tensor);
        assert_eq!(parsed.baseline_schema_version, "1.0.0");
    }

    #[test]
    fn test_lut_baseline_deterministic() {
        let computer = BaselineComputer::new(42);
        let result1 = computer.compute_lut_projection("test_lut", 16, 128, 16, 4, 1).unwrap();
        let result2 = computer.compute_lut_projection("test_lut", 16, 128, 16, 4, 1).unwrap();
        assert_eq!(result1.output_tensor, result2.output_tensor,
            "Same seed and dimensions must produce identical LUT baselines");
    }

    #[test]
    fn test_lut_baseline_shape() {
        let computer = BaselineComputer::new(42);
        let result = computer.compute_lut_projection("test_lut", 16, 128, 16, 4, 1).unwrap();
        assert_eq!(result.output_tensor.len(), 128, "batch=1, embed_dim=128 → 128 values");
        assert_eq!(result.output_shape, vec![1, 128]);
    }

    #[test]
    fn test_lut_baseline_different_bitwidths() {
        let computer = BaselineComputer::new(42);
        let result_4bit = computer.compute_lut_projection("test", 16, 64, 8, 4, 1).unwrap();
        let result_8bit = computer.compute_lut_projection("test", 256, 64, 8, 8, 1).unwrap();
        // Different bitwidths should produce different baselines
        assert_ne!(result_4bit.output_tensor, result_8bit.output_tensor,
            "Different bitwidths must produce different baselines");
    }

    #[test]
    fn test_decode_step_baseline_deterministic() {
        let computer = BaselineComputer::new(42);
        let result1 = computer.compute_decode_step("test_ds", 128, 4, 32, 64, 1).unwrap();
        let result2 = computer.compute_decode_step("test_ds", 128, 4, 32, 64, 1).unwrap();
        assert_eq!(result1.output_tensor, result2.output_tensor,
            "Same seed and dimensions must produce identical decode-step baselines");
    }

    #[test]
    fn test_decode_step_baseline_shape() {
        let computer = BaselineComputer::new(42);
        let result = computer.compute_decode_step("test_ds", 128, 4, 32, 64, 1).unwrap();
        assert_eq!(result.output_tensor.len(), 128, "batch=1, embed_dim=128 → 128 values");
        assert_eq!(result.output_shape, vec![1, 128]);
    }

    #[test]
    fn test_decode_step_baseline_different_params() {
        let computer = BaselineComputer::new(42);
        let result_small = computer.compute_decode_step("test", 64, 2, 32, 32, 1).unwrap();
        let result_large = computer.compute_decode_step("test", 128, 4, 32, 64, 1).unwrap();
        // Different parameters should produce different baselines
        assert_ne!(result_small.output_tensor, result_large.output_tensor,
            "Different decode-step parameters must produce different baselines");
    }

    #[test]
    fn test_decode_step_baseline_differs_from_linear() {
        let computer = BaselineComputer::new(42);
        let linear = computer.compute_linear_projection("test", 128, 128, 1).unwrap();
        let decode_step = computer.compute_decode_step("test", 128, 4, 32, 64, 1).unwrap();
        // Decode-step baseline must differ from linear projection baseline
        // even with the same dimensions, because decode-step includes attention
        assert_ne!(linear.output_tensor, decode_step.output_tensor,
            "Decode-step baseline must differ from linear projection baseline");
    }

    #[test]
    fn test_mlp_block_baseline_deterministic() {
        let computer = BaselineComputer::new(42);
        let result1 = computer.compute_mlp_block("test_mlp", 128, 512, 128, "gelu", 1).unwrap();
        let result2 = computer.compute_mlp_block("test_mlp", 128, 512, 128, "gelu", 1).unwrap();
        assert_eq!(result1.output_tensor, result2.output_tensor,
            "Same seed and dimensions must produce identical MLP block baselines");
    }

    #[test]
    fn test_mlp_block_baseline_shape() {
        let computer = BaselineComputer::new(42);
        let result = computer.compute_mlp_block("test_mlp", 128, 512, 128, "gelu", 1).unwrap();
        assert_eq!(result.output_tensor.len(), 128, "batch=1, output_dim=128 → 128 values");
        assert_eq!(result.output_shape, vec![1, 128]);
    }

    #[test]
    fn test_mlp_block_baseline_activation_variation() {
        let computer = BaselineComputer::new(42);
        let gelu_result = computer.compute_mlp_block("test_mlp", 64, 256, 64, "gelu", 1).unwrap();
        let relu_result = computer.compute_mlp_block("test_mlp", 64, 256, 64, "relu", 1).unwrap();
        // Different activations should produce different baselines
        assert_ne!(gelu_result.output_tensor, relu_result.output_tensor,
            "GELU and ReLU must produce different baselines");
    }

    #[test]
    fn test_attention_baseline_deterministic() {
        let computer = BaselineComputer::new(42);
        let result1 = computer.compute_attention("test_attn", 128, 4, 32, 32, 1).unwrap();
        let result2 = computer.compute_attention("test_attn", 128, 4, 32, 32, 1).unwrap();
        assert_eq!(result1.output_tensor, result2.output_tensor,
            "Same seed and dimensions must produce identical attention baselines");
    }

    #[test]
    fn test_attention_baseline_shape() {
        let computer = BaselineComputer::new(42);
        let result = computer.compute_attention("test_attn", 128, 4, 32, 32, 1).unwrap();
        // Output shape: [batch_size, seq_len, embed_dim]
        assert_eq!(result.output_tensor.len(), 128 * 32, "batch=1, seq=32, embed=128 → 4096 values");
        assert_eq!(result.output_shape, vec![1, 32, 128]);
    }

    #[test]
    fn test_attention_baseline_differs_from_decode_step() {
        let computer = BaselineComputer::new(42);
        let decode_step = computer.compute_decode_step("test", 128, 4, 32, 32, 1).unwrap();
        let attention = computer.compute_attention("test", 128, 4, 32, 32, 1).unwrap();
        // Attention (no cache, full seq-to-seq) must differ from decode-step (with cache)
        assert_ne!(decode_step.output_shape, attention.output_shape,
            "Attention output shape must differ from decode-step (seq_len dimension)");
    }

    #[test]
    fn test_sharded_linear_pipeline_baseline_deterministic() {
        let computer = BaselineComputer::new(42);
        let result1 = computer.compute_sharded_linear_pipeline("test_shard_lin", 64, 48, 32, 1).unwrap();
        let result2 = computer.compute_sharded_linear_pipeline("test_shard_lin", 64, 48, 32, 1).unwrap();
        assert_eq!(result1.output_tensor, result2.output_tensor,
            "Same seed and dimensions must produce identical sharded linear pipeline baselines");
    }

    #[test]
    fn test_sharded_linear_pipeline_baseline_shape() {
        let computer = BaselineComputer::new(42);
        let result = computer.compute_sharded_linear_pipeline("test_shard_lin", 64, 48, 32, 1).unwrap();
        assert_eq!(result.output_tensor.len(), 32, "batch=1, output_dim=32 → 32 values");
        assert_eq!(result.output_shape, vec![1, 32]);
    }

    #[test]
    fn test_sharded_linear_pipeline_differs_from_single_projection() {
        let computer = BaselineComputer::new(42);
        let single = computer.compute_linear_projection("test", 64, 32, 1).unwrap();
        let sharded = computer.compute_sharded_linear_pipeline("test", 64, 48, 32, 1).unwrap();
        // The sharded pipeline with hidden_dim=48 must differ from a single projection 64→32
        // because the intermediate dimension changes the computation.
        assert_ne!(single.output_tensor, sharded.output_tensor,
            "Sharded pipeline must differ from single linear projection");
    }

    // --- Sharded decode-step baseline tests (Sprint 54) ---

    #[test]
    fn test_sharded_decode_step_baseline_deterministic() {
        let computer = BaselineComputer::new(42);
        let result1 = computer.compute_sharded_decode_step("test_shard_ds", 128, 4, 32, 64, 1).unwrap();
        let result2 = computer.compute_sharded_decode_step("test_shard_ds", 128, 4, 32, 64, 1).unwrap();
        assert_eq!(result1.output_tensor, result2.output_tensor,
            "Same seed and dimensions must produce identical sharded decode-step baselines");
    }

    #[test]
    fn test_sharded_decode_step_baseline_shape() {
        let computer = BaselineComputer::new(42);
        let result = computer.compute_sharded_decode_step("test_shard_ds", 128, 4, 32, 64, 1).unwrap();
        assert_eq!(result.output_tensor.len(), 128, "batch=1, embed_dim=128 → 128 values");
        assert_eq!(result.output_shape, vec![1, 128]);
    }

    #[test]
    fn test_sharded_decode_step_differs_from_single_decode_step() {
        let computer = BaselineComputer::new(42);
        let single = computer.compute_decode_step("test", 128, 4, 32, 64, 1).unwrap();
        let sharded = computer.compute_sharded_decode_step("test", 128, 4, 32, 64, 1).unwrap();
        // The sharded decode-step uses different seed offsets for KV cache and output
        // projection (40/45/50 vs 3/4/5), so it must produce different output.
        assert_ne!(single.output_tensor, sharded.output_tensor,
            "Sharded decode-step must differ from single decode-step due to different weight seeds");
    }
}
