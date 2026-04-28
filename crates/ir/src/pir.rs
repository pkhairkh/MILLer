//! Package/Deployment IR (PIR)
//!
//! Full deployment artifact representation: which MIL programs
//! go in which packages, how they connect, state ownership.
//! Includes multifunction package seam.
//!
//! ## Shard Role Model (Sprint 9, S9.1)
//!
//! The shard role model captures the five logical roles that a package
//! can play in a sharded deployment:
//!
//! - **Entry**: First decoder shard — receives embedded tokens from the IO model,
//!   applies the first N transformer layers, produces intermediate activations.
//! - **Interior**: Middle decoder shard(s) — applies intermediate transformer
//!   layers. A model may have zero, one, or multiple interior shards.
//! - **Exit**: Last decoder shard — applies the final transformer layers and
//!   produces the hidden state that feeds back to the IO model for logit
//!   projection and sampling.
//! - **Io**: I/O model — handles embedding lookup and logit projection.
//!   Typically placed on CPU+GPU because these ops are ANE-hostile or
//!   ANE-suboptimal (embedding gather, large matmul for LM head).
//! - **Sampler**: Sampling model — applies temperature, top-p, repetition
//!   penalty, and token selection. Placed on CPU+GPU because sampling
//!   involves dynamic control flow and scalar operations.
//!
//! These roles are represented at two levels:
//!
//! 1. `ShardRole` — a unified enum of all five roles. This is the role model
//!    that shard templates, partition specs, and knowledge entries use.
//!    It is the primary role representation for the compiler.
//!
//! 2. `PackageRole` — a package-level classification that groups roles by
//!    their deployment category: IO packages, DecoderShard packages (with
//!    their sub-role), and Sampler packages. This captures the architectural
//!    distinction between "packages that run on CPU+GPU" and "packages that
//!    target CPU+NE (ANE)."
//!
//! The two representations are isomorphic: `PackageRole::from_shard_role()`
//! converts any `ShardRole` to the corresponding `PackageRole`, and
//! `PackageRole::to_shard_role()` converts back.

use serde::{Deserialize, Serialize};

// Sprint 58 (S58.3): ComputeUnits was removed. PIR now uses
// `ComputeUnitHint` from the mir module directly, eliminating the duplicate type.
pub use super::mir::ComputeUnitHint;
pub use super::sir::{KvCacheLayout, SamplerSpec, IoModelSpec, QuantizationStrategy};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Package {
    pub name: String,
    pub role: PackageRole,
    pub compute_units: ComputeUnitHint,
    /// Reference to the MIL program for the primary function.
    pub mil_program_ref: String,
    /// Named functions within this package (multifunction seam).
    /// For single-function packages, this contains one entry named "main".
    /// For multifunction packages, this contains multiple named functions
    /// that share weights/backbone within a single mlpackage.
    /// This field is the architectural seam for future multifunction support;
    /// it does not need to be fully exercised in the current vertical slice.
    pub functions: Vec<FunctionEntry>,
}

/// A named function within a package.
/// In a multifunction mlpackage, each function has its own I/O signature
/// but shares the package's weight storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionEntry {
    /// Function name (e.g., "main", "encode", "decode").
    pub name: String,
    /// Input tensor specifications for this function.
    pub inputs: Vec<TensorSpec>,
    /// Output tensor specifications for this function.
    pub outputs: Vec<TensorSpec>,
    /// Whether this function uses persistent state.
    pub stateful: bool,
}

/// Tensor shape and dtype specification for function I/O.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Package-level role classification.
///
/// Groups shard roles by their deployment category:
/// - `IO` — I/O model (embedding + LM head), typically CPU+GPU
/// - `DecoderShard(ShardRole)` — decoder shard with explicit sub-role
///   (Entry, Interior, Exit), typically CPU+NE
/// - `Sampler` — sampling model, typically CPU+GPU
///
/// This classification captures the architectural split between
/// ANE-targeted packages (decoder shards) and CPU+GPU packages
/// (IO and sampler). It is used in PIR `Package` to declare what
/// each compiled mlpackage is for.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PackageRole {
    /// I/O model: handles embedding lookup and logit projection.
    /// Placed on CPU+GPU because embedding gather and large matmul
    /// for LM head are ANE-hostile or ANE-suboptimal.
    IO,
    /// Decoder shard with explicit sub-role (Entry, Interior, Exit, Io, Sampler).
    /// The sub-role determines the shard's position in the decode sequence.
    /// Most decoder shards use Entry/Interior/Exit; Io and Sampler sub-roles
    /// exist for completeness but are rarely used in PackageRole context
    /// (prefer `PackageRole::IO` or `PackageRole::Sampler` directly).
    DecoderShard(ShardRole),
    /// Sampling model: applies temperature, top-p, repetition penalty.
    /// Placed on CPU+GPU because sampling involves dynamic control flow.
    Sampler,
}

impl PackageRole {
    /// Convert a `ShardRole` to the corresponding `PackageRole`.
    ///
    /// - `ShardRole::Entry` / `Interior` / `Exit` → `DecoderShard(role)`
    /// - `ShardRole::Io` → `PackageRole::IO`
    /// - `ShardRole::Sampler` → `PackageRole::Sampler`
    pub fn from_shard_role(role: ShardRole) -> Self {
        match role {
            ShardRole::Io => PackageRole::IO,
            ShardRole::Sampler => PackageRole::Sampler,
            other => PackageRole::DecoderShard(other),
        }
    }

    /// Convert this `PackageRole` to the corresponding `ShardRole`.
    ///
    /// - `PackageRole::IO` → `ShardRole::Io`
    /// - `PackageRole::Sampler` → `ShardRole::Sampler`
    /// - `PackageRole::DecoderShard(role)` → the inner `ShardRole`
    pub fn to_shard_role(&self) -> ShardRole {
        match self {
            PackageRole::IO => ShardRole::Io,
            PackageRole::Sampler => ShardRole::Sampler,
            PackageRole::DecoderShard(role) => role.clone(),
        }
    }

    /// Whether this package targets ANE (CPU+NE) deployment.
    pub fn is_ane_targeted(&self) -> bool {
        matches!(
            self,
            PackageRole::DecoderShard(ShardRole::Entry | ShardRole::Interior | ShardRole::Exit)
        )
    }

    /// Whether this package typically runs on CPU+GPU.
    pub fn is_cpu_gpu(&self) -> bool {
        matches!(self, PackageRole::IO | PackageRole::Sampler)
    }
}

/// Unified shard role enum representing all five logical roles
/// that a package can play in a sharded deployment.
///
/// This is the primary role model used by:
/// - `ShardPartitionEntry` in `ShardTemplate`
/// - Shard template seed loading
/// - Knowledge store shard template entries
/// - The compiler's partition planner
///
/// The five roles correspond to the Qwen3 five-package decomposition:
/// `io_model` (Io), three decoder shards (Entry/Interior/Exit),
/// and `sampler_model` (Sampler).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ShardRole {
    /// I/O model: handles embedding lookup and logit projection.
    /// In the Qwen3 deployment, this is `io_model.mlpackage` on CPU+GPU.
    Io,
    /// First decoder shard: receives embedded tokens from the IO model,
    /// applies the first N transformer layers.
    /// In the Qwen3 deployment, this is the "left" shard on CPU+NE.
    Entry,
    /// Middle decoder shard(s): applies intermediate transformer layers.
    /// In the Qwen3 deployment, this is the "mid" shard on CPU+NE.
    /// A deployment may have zero, one, or multiple interior shards.
    Interior,
    /// Last decoder shard: applies the final transformer layers,
    /// produces the hidden state that feeds back to the IO model.
    /// In the Qwen3 deployment, this is the "right" shard on CPU+NE.
    Exit,
    /// Sampling model: applies temperature, top-p, repetition penalty,
    /// and token selection.
    /// In the Qwen3 deployment, this is `sampler_model.mlpackage` on CPU+GPU.
    Sampler,
}

impl ShardRole {
    /// Whether this role targets ANE (CPU+NE) deployment.
    ///
    /// Entry, Interior, and Exit shards are ANE-targeted.
    /// Io and Sampler are typically CPU+GPU.
    pub fn is_ane_targeted(&self) -> bool {
        matches!(self, ShardRole::Entry | ShardRole::Interior | ShardRole::Exit)
    }

    /// Whether this role typically runs on CPU+GPU.
    ///
    /// Io and Sampler roles involve operations that are ANE-hostile
    /// or ANE-suboptimal (embedding gather, sampling control flow).
    pub fn is_cpu_gpu(&self) -> bool {
        matches!(self, ShardRole::Io | ShardRole::Sampler)
    }

    /// Whether this role is a decoder shard (Entry/Interior/Exit).
    pub fn is_decoder_shard(&self) -> bool {
        matches!(self, ShardRole::Entry | ShardRole::Interior | ShardRole::Exit)
    }

    /// Parse a shard role from its string representation.
    ///
    /// Used when loading shard template seeds from JSON.
    /// Accepts both PascalCase (Entry, Interior, Exit, Io, Sampler)
    /// and lowercase (entry, interior, exit, io, sampler).
    pub fn from_str_flexible(s: &str) -> Option<Self> {
        match s {
            "Entry" | "entry" => Some(ShardRole::Entry),
            "Interior" | "interior" => Some(ShardRole::Interior),
            "Exit" | "exit" => Some(ShardRole::Exit),
            "Io" | "io" | "IO" => Some(ShardRole::Io),
            "Sampler" | "sampler" => Some(ShardRole::Sampler),
            _ => None,
        }
    }

    /// Returns the canonical string name for this role.
    pub fn canonical_name(&self) -> &'static str {
        match self {
            ShardRole::Io => "Io",
            ShardRole::Entry => "Entry",
            ShardRole::Interior => "Interior",
            ShardRole::Exit => "Exit",
            ShardRole::Sampler => "Sampler",
        }
    }

    /// Returns the default compute units for this role.
    pub fn default_compute_units(&self) -> ComputeUnitHint {
        match self {
            ShardRole::Entry | ShardRole::Interior | ShardRole::Exit => ComputeUnitHint::CPUAndNE,
            ShardRole::Io | ShardRole::Sampler => ComputeUnitHint::CPUAndGPU,
        }
    }
}

// Sprint 58 (S58.3): The `ComputeUnits` enum and its impl block have been removed.
// `ComputeUnitHint` is now the unified type, defined in mir.rs and re-exported above.
// The `from_str_flexible()` and `to_coreml_string()` methods are on `ComputeUnitHint` in mir.rs.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDeclaration {
    pub state_id: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub owner_package: String,
}

/// Kind of inter-shard handoff at runtime.
///
/// This enum captures the concrete runtime semantics of how data
/// moves between packages during execution. It is not just metadata —
/// it determines what happens at the shard boundary.
///
/// - `TensorPassThrough`: The output tensor of the source package is
///   fed directly as the input tensor of the target package. This is
///   the simplest handoff: a single forward pass through the pipeline.
///
/// - `StateWriteRead`: The source package writes to a shared state
///   tensor, and the target package reads from it. This is used for
///   KV-cache-style patterns where persistent state must be communicated
///   between shards across multiple predict() calls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HandoffKind {
    /// Direct tensor pass-through: source output → target input.
    TensorPassThrough,
    /// State-mediated: source writes state, target reads it.
    StateWriteRead,
}

/// A concrete inter-shard handoff with runtime semantics.
///
/// Unlike the original purely-structural handoff (which was just metadata),
/// this model captures what actually happens at the shard boundary during
/// execution: which output connects to which input, in what order, and
/// through what mechanism (direct pass-through vs. state-mediated).
///
/// The `execution_order` field defines the sequence in which handoffs
/// must occur during a single forward pass. For a three-shard Entry →
/// Interior → Exit pipeline, the handoff from Entry to Interior has
/// order 0, and the handoff from Interior to Exit has order 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Handoff {
    /// Package that produces the tensor.
    pub from_package: String,
    /// Package that consumes the tensor.
    pub to_package: String,
    /// Name of the tensor being handed off (backward compatible).
    pub tensor_name: String,
    /// Shape of the handed-off tensor.
    pub shape: Vec<usize>,
    /// Data type of the handed-off tensor.
    pub dtype: String,
    /// Concrete handoff kind: direct pass-through or state-mediated.
    pub handoff_kind: HandoffKind,
    /// Execution order in the pipeline sequence (0 = first handoff).
    /// For N shards there are N-1 handoffs, ordered 0..N-2.
    pub execution_order: usize,
    /// Named output in the source package that produces this tensor.
    /// Maps to `FunctionEntry.outputs[].name` in the source package.
    pub source_output_name: String,
    /// Named input in the target package that consumes this tensor.
    /// Maps to `FunctionEntry.inputs[].name` in the target package.
    pub target_input_name: String,
}

/// A shard template describing a proven partitioning pattern.
///
/// Shard templates capture deployment patterns that have been validated
/// on real hardware. The Qwen3 three-shard template is the primary example:
/// Entry (layers 0-10), Interior (layers 11-19), Exit (layers 20-27),
/// plus an Io package and a Sampler package.
///
/// Templates are loaded from seed knowledge files and can also be derived
/// from successful compilation runs. They are used by the shard planner
/// to propose initial partitionings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardTemplate {
    pub template_id: String,
    /// Partition entries for decoder shards (Entry/Interior/Exit).
    /// Io and Sampler are described by separate fields because they
    /// have different structure (no layer range, different defaults).
    pub partition_spec: Vec<ShardPartitionEntry>,
    /// Compute units for the I/O model, if present.
    pub io_compute_units: Option<ComputeUnitHint>,
    /// Compute units for the sampler model, if present.
    pub sampler_compute_units: Option<ComputeUnitHint>,
    /// State configuration (e.g., "per_shard_kv_reverse_ring_buffer").
    pub state_config: Option<String>,
    /// Context length this template supports.
    pub context_length: usize,
}

/// A single partition entry in a shard template.
///
/// Describes one decoder shard: its role, layer range, and
/// target compute units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardPartitionEntry {
    pub role: ShardRole,
    pub layer_start: usize,
    pub layer_end: usize,
    pub compute_units: ComputeUnitHint,
}

/// A generalized shard descriptor within a multi-shard pipeline.
///
/// Unlike `ShardDesc` (which is specific to linear projection), this type
/// describes any shard in any pipeline using `TensorSpec` for I/O rather
/// than scalar dimensions. This makes it applicable to decode-step,
/// attention, and other non-linear shard types.
///
/// Sprint 23 (S23.1): extracted from the linear-specific `ShardDesc` to
/// make the handoff/runtime model reusable across task families.
///
/// Sprint 43: `op_profile` makes roles affect op structure, not just
/// dimensions. Different roles now carry genuinely different op sequences:
/// Entry shards include a reshape for handoff preparation, Interior shards
/// include activation functions, and Exit shards include normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardSpec {
    /// Shard name (e.g., "pipeline_entry", "pipeline_interior").
    pub shard_name: String,
    /// Shard role (Entry, Interior, Exit, Io, Sampler).
    pub role: ShardRole,
    /// Input tensor specifications for this shard.
    pub input_specs: Vec<TensorSpec>,
    /// Output tensor specifications for this shard.
    pub output_specs: Vec<TensorSpec>,
    /// Compute units for this shard.
    pub compute_units: ComputeUnitHint,
    /// Op profile describing the operation sequence for this shard.
    ///
    /// This makes the shard's op structure explicit rather than implicit.
    /// Two shards with the same role but different op profiles produce
    /// genuinely different MIR graphs, not just different dimensions.
    pub op_profile: ShardOpProfile,
}

/// Describes the operation sequence that a shard performs.
///
/// This is the key structural difference between roles: it's not just that
/// Entry/Interior/Exit have different I/O dimensions — they perform
/// fundamentally different computations. Entry shards prepare handoff tensors,
/// Interior shards apply activation functions, and Exit shards apply
/// normalization. Io and Sampler shards have their own distinct profiles.
///
/// Before Sprint 43, all three decoder shards in a linear pipeline had
/// identical op structure (just a single MILLinear), differing only in
/// dimensions. Now each role has a distinct op sequence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ShardOpProfile {
    /// Entry shard: Linear + Reshape for handoff preparation.
    /// The reshape ensures the output tensor is in the correct layout
    /// for the next shard's consumption.
    EntryLinear {
        /// Whether a reshape is needed after the linear projection.
        needs_reshape: bool,
        /// Target shape for the reshape (if needed).
        reshape_target: Option<Vec<usize>>,
    },
    /// Interior shard: Linear + Activation (GELU).
    /// The activation function represents the non-linearity between
    /// transformer layers that the interior shard is responsible for.
    InteriorLinear {
        /// Activation function to apply after linear projection.
        activation: ActivationType,
    },
    /// Exit shard: Linear + LayerNorm.
    /// The layer normalization at the output is critical for numerical
    /// stability in the final shard's output to the IO model.
    ExitLinear {
        /// Epsilon for layer normalization.
        ln_epsilon: f32,
    },
    /// QKV projection shard (decode-step Entry): Linear producing Q, K, V.
    QkvProjection {
        /// Number of attention heads.
        num_heads: usize,
        /// Dimension per head.
        head_dim: usize,
    },
    /// Attention computation shard (decode-step Interior): SDPA + state.
    AttentionComputation {
        /// Whether causal masking is applied.
        causal: bool,
        /// Whether KV cache state is used.
        stateful: bool,
    },
    /// Output projection shard (decode-step Exit): Linear + optional norm.
    OutputProjection {
        /// Whether layer normalization is applied after projection.
        with_norm: bool,
        /// Epsilon for normalization (if applied).
        ln_epsilon: Option<f32>,
    },
    /// IO model shard: Embedding lookup + optional LM head projection.
    IoEmbedding {
        /// Whether LM head projection is included.
        with_lm_head: bool,
    },
    /// Sampler shard: Top-k + softmax + gather for token selection.
    SamplerTopk {
        /// K value for top-k sampling.
        k: usize,
    },
    /// Generic linear-only shard (backward compatible with pre-Sprint-43 behavior).
    LinearOnly,
}

/// Activation function types for shard op profiles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    /// GELU with tanh approximation (mb.gelu mode="TANH_APPROXIMATION").
    GeluTanh,
    /// ReLU (mb.relu).
    Relu,
    /// No activation function.
    None,
}

/// A complete multi-shard pipeline specification.
///
/// This is the compiler-level representation of a multi-shard decomposition,
/// independent of any specific task family. It describes the shards, their
/// inter-shard handoffs, and the overall pipeline structure.
///
/// A `ShardPipelineSpec` can be constructed:
/// - directly from a task spec (e.g., `ShardedLinearPipeline` → 3-shard linear spec)
/// - from a `ShardTemplate` with dimension context (knowledge-driven planning)
///
/// Sprint 23 (S23.1 + S23.2): this type replaces the raw-dimension parameters
/// in `ShardPlanPass::build_sharded_plan`, making the shard planner consume
/// typed inputs generically rather than hardcoding the 3-shard linear pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardPipelineSpec {
    /// Pipeline name (typically the task name).
    pub pipeline_name: String,
    /// Data type for all tensors in the pipeline.
    pub dtype: String,
    /// Batch size for the pipeline.
    pub batch_size: usize,
    /// Ordered list of shards in the pipeline.
    pub shards: Vec<ShardSpec>,
    /// Inter-shard handoffs connecting consecutive shards.
    pub handoffs: Vec<Handoff>,
    /// State declarations for stateful shards (e.g., KV cache).
    pub state_declarations: Vec<StateDeclaration>,
    /// Optional shard template reference for provenance.
    pub shard_template: Option<ShardTemplate>,
    /// Opset version for the pipeline.
    pub opset_version: String,
}

impl ShardPipelineSpec {
    /// Build a 3-shard linear pipeline specification.
    ///
    /// This is the legacy decomposition: Entry (input_dim → hidden_dim),
    /// Interior (hidden_dim → hidden_dim), Exit (hidden_dim → output_dim).
    /// Each shard uses `TensorPassThrough` handoffs.
    ///
    /// This method preserves backward compatibility with the previous
    /// `ShardPlanPass::build_sharded_plan(task_name, input_dim, hidden_dim,
    /// output_dim, batch_size, dtype)` interface.
    pub fn three_shard_linear(
        task_name: &str,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        batch_size: usize,
        dtype: &str,
    ) -> Self {
        let shard_names = [
            format!("{}_entry", task_name),
            format!("{}_interior", task_name),
            format!("{}_exit", task_name),
        ];

        let shards = vec![
            ShardSpec {
                shard_name: shard_names[0].clone(),
                role: ShardRole::Entry,
                input_specs: vec![TensorSpec {
                    name: "x".into(),
                    shape: vec![batch_size, input_dim],
                    dtype: dtype.into(),
                }],
                output_specs: vec![TensorSpec {
                    name: "output".into(),
                    shape: vec![batch_size, hidden_dim],
                    dtype: dtype.into(),
                }],
                compute_units: ShardRole::Entry.default_compute_units(),
                op_profile: ShardOpProfile::EntryLinear {
                    needs_reshape: input_dim != hidden_dim,
                    reshape_target: if input_dim != hidden_dim {
                        Some(vec![batch_size, hidden_dim])
                    } else {
                        None
                    },
                },
            },
            ShardSpec {
                shard_name: shard_names[1].clone(),
                role: ShardRole::Interior,
                input_specs: vec![TensorSpec {
                    name: "x".into(),
                    shape: vec![batch_size, hidden_dim],
                    dtype: dtype.into(),
                }],
                output_specs: vec![TensorSpec {
                    name: "output".into(),
                    shape: vec![batch_size, hidden_dim],
                    dtype: dtype.into(),
                }],
                compute_units: ShardRole::Interior.default_compute_units(),
                op_profile: ShardOpProfile::InteriorLinear { activation: ActivationType::GeluTanh },
            },
            ShardSpec {
                shard_name: shard_names[2].clone(),
                role: ShardRole::Exit,
                input_specs: vec![TensorSpec {
                    name: "x".into(),
                    shape: vec![batch_size, hidden_dim],
                    dtype: dtype.into(),
                }],
                output_specs: vec![TensorSpec {
                    name: "output".into(),
                    shape: vec![batch_size, output_dim],
                    dtype: dtype.into(),
                }],
                compute_units: ShardRole::Exit.default_compute_units(),
                op_profile: ShardOpProfile::ExitLinear { ln_epsilon: 1e-5 },
            },
        ];

        let handoffs = vec![
            Handoff {
                from_package: shard_names[0].clone(),
                to_package: shard_names[1].clone(),
                tensor_name: "output".into(),
                shape: vec![batch_size, hidden_dim],
                dtype: dtype.into(),
                handoff_kind: HandoffKind::TensorPassThrough,
                execution_order: 0,
                source_output_name: "output".into(),
                target_input_name: "x".into(),
            },
            Handoff {
                from_package: shard_names[1].clone(),
                to_package: shard_names[2].clone(),
                tensor_name: "output".into(),
                shape: vec![batch_size, hidden_dim],
                dtype: dtype.into(),
                handoff_kind: HandoffKind::TensorPassThrough,
                execution_order: 1,
                source_output_name: "output".into(),
                target_input_name: "x".into(),
            },
        ];

        let shard_template = ShardTemplate {
            template_id: format!("{}_3shard_template", task_name),
            partition_spec: vec![
                ShardPartitionEntry {
                    role: ShardRole::Entry,
                    layer_start: 0,
                    layer_end: 0,
                    compute_units: ComputeUnitHint::CPUAndNE,
                },
                ShardPartitionEntry {
                    role: ShardRole::Interior,
                    layer_start: 1,
                    layer_end: 1,
                    compute_units: ComputeUnitHint::CPUAndNE,
                },
                ShardPartitionEntry {
                    role: ShardRole::Exit,
                    layer_start: 2,
                    layer_end: 2,
                    compute_units: ComputeUnitHint::CPUAndNE,
                },
            ],
            io_compute_units: None,
            sampler_compute_units: None,
            state_config: None,
            context_length: 0,
        };

        Self {
            pipeline_name: task_name.into(),
            dtype: dtype.into(),
            batch_size,
            shards,
            handoffs,
            state_declarations: vec![],
            shard_template: Some(shard_template),
            opset_version: "iOS18".into(),
        }
    }

    /// Build a 3-shard decode-step pipeline specification.
    ///
    /// The decode step is decomposed into three shards:
    /// - Entry: QKV projection (embed_dim → 3 * embed_dim)
    /// - Interior: attention computation (3 * embed_dim → embed_dim)
    /// - Exit: output projection (embed_dim → embed_dim)
    ///
    /// This models the dominant execution pattern in autoregressive LLM
    /// inference: projection → attention → output projection. Each shard
    /// is a separate package that can be independently compiled and
    /// placed on different compute units.
    ///
    /// Sprint 23 (S23.3): this is the second multi-unit pipeline spec,
    /// proving the generalized shard model works beyond linear projection.
    pub fn three_shard_decode_step(
        task_name: &str,
        embed_dim: usize,
        num_heads: usize,
        head_dim: usize,
        kv_len: usize,
        batch_size: usize,
        dtype: &str,
    ) -> Self {
        let qkv_dim = 3 * embed_dim; // Q, K, V projections concatenated
        let shard_names = [
            format!("{}_qkv_proj", task_name),
            format!("{}_attention", task_name),
            format!("{}_out_proj", task_name),
        ];

        let shards = vec![
            ShardSpec {
                shard_name: shard_names[0].clone(),
                role: ShardRole::Entry,
                input_specs: vec![TensorSpec {
                    name: "x".into(),
                    shape: vec![batch_size, embed_dim],
                    dtype: dtype.into(),
                }],
                output_specs: vec![TensorSpec {
                    name: "qkv".into(),
                    shape: vec![batch_size, qkv_dim],
                    dtype: dtype.into(),
                }],
                compute_units: ShardRole::Entry.default_compute_units(),
                op_profile: ShardOpProfile::QkvProjection { num_heads, head_dim },
            },
            ShardSpec {
                shard_name: shard_names[1].clone(),
                role: ShardRole::Interior,
                input_specs: vec![TensorSpec {
                    name: "qkv".into(),
                    shape: vec![batch_size, qkv_dim],
                    dtype: dtype.into(),
                }],
                output_specs: vec![TensorSpec {
                    name: "attn_out".into(),
                    shape: vec![batch_size, embed_dim],
                    dtype: dtype.into(),
                }],
                compute_units: ShardRole::Interior.default_compute_units(),
                op_profile: ShardOpProfile::AttentionComputation { causal: true, stateful: true },
            },
            ShardSpec {
                shard_name: shard_names[2].clone(),
                role: ShardRole::Exit,
                input_specs: vec![TensorSpec {
                    name: "attn_out".into(),
                    shape: vec![batch_size, embed_dim],
                    dtype: dtype.into(),
                }],
                output_specs: vec![TensorSpec {
                    name: "output".into(),
                    shape: vec![batch_size, embed_dim],
                    dtype: dtype.into(),
                }],
                compute_units: ShardRole::Exit.default_compute_units(),
                op_profile: ShardOpProfile::OutputProjection {
                    with_norm: true,
                    ln_epsilon: Some(1e-5),
                },
            },
        ];

        // Decode-step handoffs carry the QKV and attention output tensors.
        // The Entry → Interior handoff carries the concatenated QKV output.
        // The Interior → Exit handoff carries the attention output.
        //
        // Sprint 48 / S36.2: The Interior (attention) shard is stateful (KV cache).
        // The Entry → Interior handoff is a tensor pass-through (QKV data flows
        // directly). But the Interior → Exit handoff is StateWriteRead: the
        // attention shard writes its updated KV cache to shared state, and the
        // next invocation reads it. This models the real runtime behavior where
        // KV cache persists across decode steps.
        let handoffs = vec![
            Handoff {
                from_package: shard_names[0].clone(),
                to_package: shard_names[1].clone(),
                tensor_name: "qkv".into(),
                shape: vec![batch_size, qkv_dim],
                dtype: dtype.into(),
                handoff_kind: HandoffKind::TensorPassThrough,
                execution_order: 0,
                source_output_name: "qkv".into(),
                target_input_name: "qkv".into(),
            },
            Handoff {
                from_package: shard_names[1].clone(),
                to_package: shard_names[2].clone(),
                tensor_name: "attn_out".into(),
                shape: vec![batch_size, embed_dim],
                dtype: dtype.into(),
                handoff_kind: HandoffKind::StateWriteRead,
                execution_order: 1,
                source_output_name: "attn_out".into(),
                target_input_name: "attn_out".into(),
            },
        ];

        // KV cache state declaration for the attention shard.
        // In a real deployment, the attention shard would read from and
        // write to a shared KV cache. For v0, we declare the state
        // but the synthetic emission path uses linear projection.
        let state_declarations = vec![StateDeclaration {
            state_id: format!("{}_kv_cache", task_name),
            shape: vec![2, batch_size, num_heads, kv_len, head_dim],
            dtype: dtype.into(),
            owner_package: shard_names[1].clone(),
        }];

        let shard_template = ShardTemplate {
            template_id: format!("{}_3shard_decode_template", task_name),
            partition_spec: vec![
                ShardPartitionEntry {
                    role: ShardRole::Entry,
                    layer_start: 0,
                    layer_end: 0,
                    compute_units: ComputeUnitHint::CPUAndNE,
                },
                ShardPartitionEntry {
                    role: ShardRole::Interior,
                    layer_start: 1,
                    layer_end: 1,
                    compute_units: ComputeUnitHint::CPUAndNE,
                },
                ShardPartitionEntry {
                    role: ShardRole::Exit,
                    layer_start: 2,
                    layer_end: 2,
                    compute_units: ComputeUnitHint::CPUAndNE,
                },
            ],
            io_compute_units: None,
            sampler_compute_units: None,
            state_config: Some("per_shard_kv_reverse_ring_buffer".into()),
            context_length: kv_len,
        };

        Self {
            pipeline_name: task_name.into(),
            dtype: dtype.into(),
            batch_size,
            shards,
            handoffs,
            state_declarations,
            shard_template: Some(shard_template),
            opset_version: "iOS18".into(),
        }
    }

    /// Number of shards in this pipeline.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Whether this is a multi-shard pipeline (more than one shard).
    pub fn is_multi_shard(&self) -> bool {
        self.shards.len() > 1
    }

    /// Convert this pipeline spec into a PIR graph.
    ///
    /// This is the generalized PIR construction that works for any
    /// pipeline spec, not just the 3-shard linear decomposition.
    pub fn to_pir_graph(&self) -> PirGraph {
        let packages: Vec<Package> = self
            .shards
            .iter()
            .map(|shard| Package {
                name: shard.shard_name.clone(),
                role: PackageRole::DecoderShard(shard.role.clone()),
                compute_units: shard.compute_units.clone(),
                mil_program_ref: shard.shard_name.clone(),
                functions: vec![FunctionEntry {
                    name: "main".into(),
                    inputs: shard.input_specs.clone(),
                    outputs: shard.output_specs.clone(),
                    stateful: !self.state_declarations.is_empty()
                        && self
                            .state_declarations
                            .iter()
                            .any(|s| s.owner_package == shard.shard_name),
                }],
            })
            .collect();

        PirGraph {
            packages,
            state_declarations: self.state_declarations.clone(),
            handoffs: self.handoffs.clone(),
            shard_template: self.shard_template.clone(),
            context_length: self.shard_template.as_ref().map(|t| t.context_length).unwrap_or(0),
            opset_version: self.opset_version.clone(),
            minimum_deployment_target: self.opset_version.clone(),
            kv_cache_layout: KvCacheLayout::default(),
            sampler_spec: None,
            io_model_spec: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PirGraph {
    pub packages: Vec<Package>,
    pub state_declarations: Vec<StateDeclaration>,
    pub handoffs: Vec<Handoff>,
    pub shard_template: Option<ShardTemplate>,
    pub context_length: usize,
    pub opset_version: String,
    pub minimum_deployment_target: String,
    /// KV cache layout strategy. Default is Naive; set to ReverseRingBuffer
    /// for ANE-optimized KV cache updates via masked blending.
    /// Derived from pkhairkh/qwen3-coreml-palettized's reverse ring-buffer approach.
    pub kv_cache_layout: KvCacheLayout,
    /// Specification for the on-device sampler model, if present.
    /// When set, the deployment includes a dedicated sampler MLProgram
    /// instead of relying on host-side sampling.
    pub sampler_spec: Option<SamplerSpec>,
    /// Specification for the conditional IO model (embedding + LM head).
    /// When set with tied_weights=true, embedding and logit projection
    /// share weights in a single MLProgram with a mode switch.
    pub io_model_spec: Option<IoModelSpec>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S36.2 test: decode-step shard plan uses StateWriteRead for the
    /// Interior → Exit handoff, because the attention shard maintains
    /// KV-cache state that must persist across decode steps.
    ///
    /// Before Sprint 48, all handoffs were TensorPassThrough, which
    /// meant StateWriteRead was declared but never exercised.
    #[test]
    fn test_decode_step_uses_state_write_read_for_attention_handoff() {
        let spec =
            ShardPipelineSpec::three_shard_decode_step("test_task", 128, 4, 32, 64, 1, "fp16");

        // There should be exactly 2 handoffs: Entry→Interior and Interior→Exit
        assert_eq!(spec.handoffs.len(), 2, "Three-shard decode step must have 2 handoffs");

        // Entry → Interior: tensor pass-through (QKV data flows directly)
        assert_eq!(
            spec.handoffs[0].handoff_kind,
            HandoffKind::TensorPassThrough,
            "Entry → Interior handoff must be TensorPassThrough (QKV data)"
        );

        // Interior → Exit: state write-read (attention shard's KV cache persists)
        assert_eq!(
            spec.handoffs[1].handoff_kind,
            HandoffKind::StateWriteRead,
            "Interior → Exit handoff must be StateWriteRead (KV cache state persistence)"
        );
    }

    /// Verify that linear pipeline handoffs remain TensorPassThrough
    /// (no state-mediated handoff needed for linear-only shards).
    #[test]
    fn test_linear_pipeline_uses_tensor_pass_through() {
        let spec = ShardPipelineSpec::three_shard_linear("test_linear", 64, 48, 32, 1, "fp16");

        assert_eq!(spec.handoffs.len(), 2);
        assert_eq!(
            spec.handoffs[0].handoff_kind,
            HandoffKind::TensorPassThrough,
            "Linear pipeline Entry → Interior must be TensorPassThrough"
        );
        assert_eq!(
            spec.handoffs[1].handoff_kind,
            HandoffKind::TensorPassThrough,
            "Linear pipeline Interior → Exit must be TensorPassThrough"
        );
    }

    /// Verify that decode-step plan has state declarations for KV cache.
    #[test]
    fn test_decode_step_has_kv_cache_state_declaration() {
        let spec =
            ShardPipelineSpec::three_shard_decode_step("test_task", 128, 4, 32, 64, 1, "fp16");

        assert!(
            !spec.state_declarations.is_empty(),
            "Decode-step plan must have state declarations for KV cache"
        );
        assert!(
            spec.state_declarations.iter().any(|s| s.state_id.contains("kv_cache")),
            "At least one state declaration must reference KV cache"
        );
    }
}
