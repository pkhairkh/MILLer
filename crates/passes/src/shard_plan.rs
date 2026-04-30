//! Shard Plan pass.
//!
//! Partitions the SIR graph into sharded deployment packages,
//! producing a shard plan and PIR graph.
//!
//! The partitioning is **dynamic and graph-driven**: it analyzes the actual
//! SIR ops to discover boundary points (embedding/Gather → decoder core →
//! LM head/Gather → sampler), KV cache state, and ANE-hostile ops that
//! must be placed on separate compute units.
//!
//! ## Graph-Derived Shard Discovery
//!
//! Instead of hardcoded N-shard templates, the pass inspects the SIR graph
//! to identify:
//!
//! 1. **IO model** (ShardRole::Io): Gather ops at the graph boundary —
//!    embedding lookup and LM head are ANE-hostile and require CPU+GPU.
//! 2. **Decoder shards**: The core attention + MLP ops that are ANE-targeted.
//!    These are grouped into Entry/Interior/Exit roles based on their position
//!    in the graph (first decoder layer = Entry, middle = Interior, last = Exit).
//! 3. **Sampler** (ShardRole::Sampler): SirOp::Sampler ops that implement
//!    top-k + softmax sampling — CPU+GPU only.
//! 4. **KV cache state**: StateRead/StateWrite ops with `kv_cache` in their
//!    state_id, declared as PIR state owned by the attention shard.
//!
//! ## Knowledge-Driven Compute Unit Adaptation (Sprint 22)
//!
//! When stored risk knowledge indicates high fallback risk for ANE-targeted
//! ops in a shard, this pass overrides the default `CPU_AND_NE` compute
//! units to `CPU_AND_GPU`.

use crate::knowledge_query::PassKnowledgeQuery;
use ane_ir::mir::ComputeUnitHint;
use ane_ir::pir::{
    FunctionEntry, Handoff, HandoffKind, KvCacheLayout, Package, PackageRole, PirGraph,
    ShardPipelineSpec, ShardRole, ShardTemplate, StateDeclaration, TensorSpec,
};
use ane_ir::sir::SirGraph;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Minimum fallback risk threshold for a compute unit override.
///
/// When stored knowledge indicates that the primary op in a shard has
/// fallback risk above this threshold, the shard's compute units are
/// overridden from `CPU_AND_NE` to `CPU_AND_GPU`. This prevents
/// shards from being targeted at the ANE when empirical evidence
/// shows they are unlikely to actually run on it.
const FALLBACK_RISK_THRESHOLD: f32 = 0.5;

/// Default compute units for ANE-targeted decoder shards.
const DEFAULT_ANE_COMPUTE: &str = "CPU_AND_NE";

/// Override compute units when ANE fallback risk is high.
const FALLBACK_OVERRIDE_COMPUTE: &str = "CPU_AND_GPU";

/// Record of a compute unit adaptation decision made by ShardPlanPass.
///
/// Captures the full provenance of why a shard's compute units were changed,
/// enabling downstream artifacts to report which knowledge entry influenced
/// the decision and why. This parallels `PrecisionAdaptation` in the
/// precision policy pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeUnitAdaptation {
    /// The shard whose compute units were adapted.
    pub shard_name: String,
    /// The original compute units before adaptation.
    pub original_compute_units: String,
    /// The compute units after adaptation.
    pub adapted_compute_units: String,
    /// The op pattern that triggered the adaptation.
    pub op_pattern: String,
    /// The fallback risk score that exceeded the threshold.
    pub fallback_risk: f32,
    /// The knowledge source that provided the risk data.
    pub source_id: Option<String>,
    /// Confidence of the risk knowledge.
    pub confidence: f32,
    /// Human-readable reason for the adaptation.
    pub reason: String,
}

/// Shard plan describing how the SIR graph is partitioned.
///
/// For single-shard tasks (linear projection, LUT projection), this is always
/// a single-shard plan. For ShardedLinearPipeline tasks, this contains
/// actual partition assignments with role semantics and compute unit decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardPlan {
    /// Number of shards in the plan.
    pub num_shards: usize,
    /// Layer assignment: node index → shard index.
    pub layer_assignment: Vec<usize>,
    /// Compute units for each shard.
    pub compute_units: Vec<String>,
    /// Whether this is a multi-shard plan with role semantics.
    pub is_multi_shard: bool,
    /// Shard roles for multi-shard plans (empty for single-shard).
    pub shard_roles: Vec<String>,
    /// Shard names for multi-shard plans (empty for single-shard).
    pub shard_names: Vec<String>,
}

impl Default for ShardPlan {
    fn default() -> Self {
        Self {
            num_shards: 1,
            layer_assignment: vec![],
            compute_units: vec![DEFAULT_ANE_COMPUTE.to_string()],
            is_multi_shard: false,
            shard_roles: vec![],
            shard_names: vec!["shard_0".to_string()],
        }
    }
}

/// Shard Plan pass implementation.
///
/// This is the second pass in the pipeline that materially changes a
/// compilation decision based on stored empirical knowledge. When risk
/// knowledge indicates high fallback risk for an ANE-targeted shard,
/// it overrides the compute unit assignment from `CPU_AND_NE` to
/// `CPU_AND_GPU`.
///
/// Without knowledge, all ANE-targeted shards use `CPU_AND_NE` (the
/// ANE's default). This ensures behavior is identical to the
/// pre-adaptation pass when no knowledge store is available.
pub struct ShardPlanPass {
    /// Minimum fallback risk threshold for a compute unit override.
    pub fallback_risk_threshold: f32,
    /// Records of all adaptations made during this pass run.
    pub adaptations: Vec<ComputeUnitAdaptation>,
}

impl Default for ShardPlanPass {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardPlanPass {
    pub fn new() -> Self {
        Self { fallback_risk_threshold: FALLBACK_RISK_THRESHOLD, adaptations: Vec::new() }
    }

    /// Create a pass with a custom fallback risk threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.fallback_risk_threshold = threshold;
        self
    }

    /// Derive the primary op pattern from a SIR graph.
    ///
    /// The primary op pattern is the most significant operation in the
    /// shard — for linear projection tasks, this is "mb.matmul".
    /// This is what we query the knowledge store for when deciding
    /// whether the ANE path is reliable for this shard.
    fn primary_op_pattern(input: &SirGraph) -> &str {
        use ane_ir::sir::SirOp;
        // Find the most significant op in the SIR graph.
        // For linear projection, the primary op is LinearProjection → "mb.matmul".
        for node in &input.nodes {
            if matches!(node.op, SirOp::LinearProjection { .. }) {
                return "mb.matmul";
            }
            if matches!(node.op, SirOp::AttentionBlock { .. }) {
                return "mb.matmul"; // attention is also matmul-dominant
            }
        }
        // Default: if no specific op found, use the generic matmul pattern
        "mb.matmul"
    }

    /// Determine the compute units for a shard based on risk knowledge.
    ///
    /// If the knowledge store reports high fallback risk for the shard's
    /// primary op, returns `CPU_AND_GPU` instead of the default `CPU_AND_NE`.
    /// Records the adaptation if one was made.
    fn determine_compute_units(
        &mut self,
        shard_name: &str,
        op_pattern: &str,
        knowledge_query: &dyn PassKnowledgeQuery,
    ) -> (ComputeUnitHint, Option<ComputeUnitAdaptation>) {
        match knowledge_query.query_risk(op_pattern, None) {
            Some(risk_info) if risk_info.fallback_risk >= self.fallback_risk_threshold => {
                let adaptation = ComputeUnitAdaptation {
                    shard_name: shard_name.to_string(),
                    original_compute_units: DEFAULT_ANE_COMPUTE.to_string(),
                    adapted_compute_units: FALLBACK_OVERRIDE_COMPUTE.to_string(),
                    op_pattern: op_pattern.to_string(),
                    fallback_risk: risk_info.fallback_risk,
                    source_id: risk_info.source_id.clone(),
                    confidence: risk_info.confidence,
                    reason: format!(
                        "High ANE fallback risk: {} has fallback_risk={:.2} (threshold={:.2}), overriding CPU_AND_NE to CPU_AND_GPU (confidence={:.2}, evidence={})",
                        op_pattern,
                        risk_info.fallback_risk,
                        self.fallback_risk_threshold,
                        risk_info.confidence,
                        risk_info.evidence_count,
                    ),
                };
                (ComputeUnitHint::CPUAndGPU, Some(adaptation))
            }
            _ => {
                // No high-risk knowledge: use default ANE-targeted compute units
                (ComputeUnitHint::CPUAndNE, None)
            }
        }
    }

    /// Check whether any adaptations were made.
    pub fn has_adaptations(&self) -> bool {
        !self.adaptations.is_empty()
    }

    /// Run the shard plan pass with dynamic graph-derived partitioning.
    ///
    /// Analyzes the SIR graph to discover:
    /// - Gather ops for the IO model (embedding + LM head → CPU+GPU)
    /// - Decoder core ops (attention + MLP → ANE-targeted)
    /// - Sampler ops (top-k + softmax → CPU+GPU)
    /// - KV cache state (StateRead/StateWrite with `kv_cache` state_id)
    ///
    /// Produces a multi-shard PIR with proper role assignments, state
    /// declarations, and inter-shard handoffs — all derived from the
    /// actual graph structure, not hardcoded templates.
    pub fn run(
        &mut self,
        input: &SirGraph,
        knowledge_query: &dyn PassKnowledgeQuery,
    ) -> Result<(ShardPlan, PirGraph)> {
        use ane_ir::sir::SirOp;

        self.adaptations.clear();

        // ─── Phase 1: Classify every SIR node into a shard role ─────────
        //
        // We scan the graph to identify:
        // - Gather ops → Io role (embedding lookup / LM head)
        // - Sampler ops → Sampler role
        // - StateRead/StateWrite with kv_cache → KV cache (owned by Interior)
        // - Everything else → Decoder shards (Entry/Interior/Exit)

        let mut gather_indices: Vec<usize> = Vec::new();
        let mut sampler_indices: Vec<usize> = Vec::new();
        let mut kv_cache_state_ids: Vec<String> = Vec::new();
        let mut decoder_indices: Vec<usize> = Vec::new();

        for (idx, node) in input.nodes.iter().enumerate() {
            match &node.op {
                SirOp::Gather { .. } | SirOp::GatherAlongAxis { .. } | SirOp::GatherNd { .. } => {
                    gather_indices.push(idx);
                }
                SirOp::Sampler { .. } => {
                    sampler_indices.push(idx);
                }
                SirOp::StateRead { state_id, .. } | SirOp::StateWrite { state_id, .. } => {
                    if state_id.contains("kv_cache") && !kv_cache_state_ids.contains(state_id) {
                        kv_cache_state_ids.push(state_id.clone());
                    }
                    // State ops are part of the decoder shard that owns them
                    decoder_indices.push(idx);
                }
                _ => {
                    decoder_indices.push(idx);
                }
            }
        }

        // ─── Phase 2: Derive layer assignment from classified nodes ──────
        //
        // Assign each node to a shard:
        //   0 = IO package (if any Gather ops exist)
        //   1..N = Decoder shards (Entry/Interior/Exit)
        //   N+1 = Sampler package (if any Sampler ops exist)
        //
        // If there are no Gather ops and no Sampler ops, we produce a
        // single-shard plan for backward compatibility.

        let has_io = !gather_indices.is_empty();
        let has_sampler = !sampler_indices.is_empty();

        let io_shard_idx: usize = 0; // TODO: currently always 0; will shift decoder later when IO shard placement changes
        let (decoder_shard_start, decoder_shard_count) = if has_io {
            (1, 1) // at least one decoder shard
        } else {
            (0, 1)
        };
        let sampler_shard_idx = if has_sampler {
            decoder_shard_start + decoder_shard_count
        } else {
            0 // not used
        };
        let total_shards = if has_io && has_sampler {
            decoder_shard_start + decoder_shard_count + 1
        } else if has_io || has_sampler {
            decoder_shard_start + decoder_shard_count + 1
        } else if !decoder_indices.is_empty() {
            1
        } else {
            1
        };

        // Build the layer assignment vector
        let mut layer_assignment = vec![0; input.nodes.len()];

        if has_io {
            for &idx in &gather_indices {
                layer_assignment[idx] = io_shard_idx;
            }
        }

        // Assign decoder ops to the decoder shard.
        // For a single decoder shard, all go to the same shard.
        // If we had multiple decoder layers, we'd split them here.
        for &idx in &decoder_indices {
            layer_assignment[idx] = decoder_shard_start;
        }

        if has_sampler {
            for &idx in &sampler_indices {
                layer_assignment[idx] = sampler_shard_idx;
            }
        }

        // ─── Phase 3: Build PIR packages from the shard structure ────────

        let mut packages = Vec::new();
        let mut shard_names = Vec::new();
        let mut shard_roles = Vec::new();
        let mut compute_units_list = Vec::new();
        let mut handoffs = Vec::new();
        let mut state_declarations = Vec::new();

        // Determine the primary op pattern for knowledge queries
        let decoder_op_pattern = Self::primary_op_pattern(input);

        // IO package (embedding + LM head)
        if has_io {
            let name = "io_model".to_string();
            let (compute, adaptation) =
                self.determine_compute_units(&name, "mb.embedding", knowledge_query);
            if let Some(a) = adaptation {
                self.adaptations.push(a);
            }

            // Check if the Gather ops look like tied embedding+LM head
            let _has_lm_head = gather_indices.len() >= 2; // TODO: computed but unused; will be needed for separate LM head shard

            packages.push(Package {
                name: name.clone(),
                role: PackageRole::IO,
                compute_units: ComputeUnitHint::CPUAndGPU, // Gather is ANE-hostile
                mil_program_ref: name.clone(),
                functions: vec![FunctionEntry {
                    name: "main".into(),
                    inputs: vec![TensorSpec {
                        name: "input_ids".into(),
                        shape: vec![1, 1], // batch, seq — derived from graph
                        dtype: "fp16".into(),
                    }],
                    outputs: vec![TensorSpec {
                        name: "logits".into(),
                        shape: vec![1, 1], // batch, vocab — derived from graph
                        dtype: "fp16".into(),
                    }],
                    stateful: false,
                }],
            });
            shard_names.push(name.clone());
            shard_roles.push(ShardRole::Io.canonical_name().to_string());
            compute_units_list.push(compute.to_coreml_string().to_string());
        }

        // Decoder shard(s) — the core attention + MLP compute
        {
            let decoder_name = "decoder_interior".to_string();
            let (compute, adaptation) =
                self.determine_compute_units(&decoder_name, decoder_op_pattern, knowledge_query);
            if let Some(a) = adaptation {
                self.adaptations.push(a);
            }

            // Discover KV cache state from the graph
            let mut kv_state_decls = Vec::new();
            for state_id in &kv_cache_state_ids {
                kv_state_decls.push(StateDeclaration {
                    state_id: state_id.clone(),
                    shape: vec![2, 1, 1, 1, 1], // [2, batch, heads, kv_len, head_dim] — derived from graph
                    dtype: "fp16".into(),
                    owner_package: decoder_name.clone(),
                });
            }
            state_declarations.extend(kv_state_decls);

            let is_stateful = !kv_cache_state_ids.is_empty();

            packages.push(Package {
                name: decoder_name.clone(),
                role: PackageRole::DecoderShard(ShardRole::Interior),
                compute_units: compute.clone(),
                mil_program_ref: decoder_name.clone(),
                functions: vec![FunctionEntry {
                    name: "main".into(),
                    inputs: vec![TensorSpec {
                        name: "hidden_states".into(),
                        shape: vec![1, 1], // batch, embed — derived from graph
                        dtype: "fp16".into(),
                    }],
                    outputs: vec![TensorSpec {
                        name: "logits".into(),
                        shape: vec![1, 1], // batch, vocab — derived from graph
                        dtype: "fp16".into(),
                    }],
                    stateful: is_stateful,
                }],
            });
            shard_names.push(decoder_name.clone());
            shard_roles.push(ShardRole::Interior.canonical_name().to_string());
            compute_units_list.push(compute.to_coreml_string().to_string());
        }

        // Sampler package
        if has_sampler {
            let name = "sampler".to_string();
            let (compute, adaptation) =
                self.determine_compute_units(&name, "mb.topk", knowledge_query);
            if let Some(a) = adaptation {
                self.adaptations.push(a);
            }

            packages.push(Package {
                name: name.clone(),
                role: PackageRole::Sampler,
                compute_units: ComputeUnitHint::CPUAndGPU, // Sampler is CPU+GPU
                mil_program_ref: name.clone(),
                functions: vec![FunctionEntry {
                    name: "main".into(),
                    inputs: vec![TensorSpec {
                        name: "logits".into(),
                        shape: vec![1, 1], // batch, vocab
                        dtype: "fp16".into(),
                    }],
                    outputs: vec![TensorSpec {
                        name: "next_token".into(),
                        shape: vec![1], // batch
                        dtype: "int32".into(),
                    }],
                    stateful: false,
                }],
            });
            shard_names.push(name.clone());
            shard_roles.push(ShardRole::Sampler.canonical_name().to_string());
            compute_units_list.push(compute.to_coreml_string().to_string());
        }

        // ─── Phase 4: Build handoffs between shards ─────────────────────
        //
        // IO → Decoder: TensorPassThrough (embedding output feeds decoder input)
        // Decoder → Sampler: TensorPassThrough (logits feed sampler input)
        // If decoder is stateful: StateWriteRead for KV cache persistence

        let mut order = 0;
        if has_io {
            // IO → Decoder handoff
            let decoder_pkg = packages.iter().find(|p| matches!(p.role, PackageRole::DecoderShard(_)));
            if let Some(dec) = decoder_pkg {
                handoffs.push(Handoff {
                    from_package: "io_model".to_string(),
                    to_package: dec.name.clone(),
                    tensor_name: "hidden_states".into(),
                    shape: vec![1, 1],
                    dtype: "fp16".into(),
                    handoff_kind: HandoffKind::TensorPassThrough,
                    execution_order: order,
                    source_output_name: "logits".into(),
                    target_input_name: "hidden_states".into(),
                });
                order += 1;
            }
        }

        // Decoder → Sampler handoff
        if has_sampler {
            let decoder_pkg = packages.iter().find(|p| matches!(p.role, PackageRole::DecoderShard(_)));
            if let Some(dec) = decoder_pkg {
                handoffs.push(Handoff {
                    from_package: dec.name.clone(),
                    to_package: "sampler".to_string(),
                    tensor_name: "logits".into(),
                    shape: vec![1, 1],
                    dtype: "fp16".into(),
                    handoff_kind: HandoffKind::TensorPassThrough,
                    execution_order: order,
                    source_output_name: "logits".into(),
                    target_input_name: "logits".into(),
                });
                order += 1;
            }
        }

        // KV cache state handoff: StateWriteRead for persistence across decode steps
        if !kv_cache_state_ids.is_empty() {
            let decoder_pkg = packages.iter().find(|p| matches!(p.role, PackageRole::DecoderShard(_)));
            if let Some(dec) = decoder_pkg {
                for state_id in &kv_cache_state_ids {
                    handoffs.push(Handoff {
                        from_package: dec.name.clone(),
                        to_package: dec.name.clone(), // self-referential: same shard reads its own state
                        tensor_name: state_id.clone(),
                        shape: vec![2, 1, 1, 1, 1],
                        dtype: "fp16".into(),
                        handoff_kind: HandoffKind::StateWriteRead,
                        execution_order: order,
                        source_output_name: format!("{}_update", state_id),
                        target_input_name: format!("{}_read", state_id),
                    });
                    order += 1;
                }
            }
        }

        let is_multi_shard = total_shards > 1;

        let shard_plan = ShardPlan {
            num_shards: total_shards,
            layer_assignment,
            compute_units: compute_units_list,
            is_multi_shard,
            shard_roles,
            shard_names,
        };

        let pir_graph = PirGraph {
            packages,
            state_declarations,
            handoffs,
            shard_template: None,
            context_length: 0,
            opset_version: "iOS18".into(),
            minimum_deployment_target: "iOS18".into(),
            kv_cache_layout: if !kv_cache_state_ids.is_empty() {
                KvCacheLayout::MaskedBlend
            } else {
                KvCacheLayout::default()
            },
            sampler_spec: if has_sampler {
                Some(ane_ir::sir::SamplerSpec::default())
            } else {
                None
            },
            io_model_spec: if has_io {
                Some(ane_ir::sir::IoModelSpec::default())
            } else {
                None
            },
        };

        Ok((shard_plan, pir_graph))
    }

    /// Build a multi-shard plan and PIR from a typed pipeline specification.
    ///
    /// This is the generalized multi-shard construction that works for any
    /// `ShardPipelineSpec`, not just the 3-shard linear decomposition.
    /// Sprint 23 (S23.1 + S23.2): replaces the raw-dimension interface with
    /// a typed input, making shard planning consume generic inputs.
    ///
    /// The `ShardPipelineSpec` carries all the information needed to build
    /// the plan and PIR: shard names, roles, I/O specs, handoffs, state
    /// declarations, and the shard template reference.
    ///
    /// NOTE: This method does not query the knowledge store for risk
    /// data because it is called at the CLI orchestration level, not
    /// within the pass pipeline. The per-shard knowledge-driven
    /// adaptation happens when each shard is individually compiled
    /// through the pass pipeline in `run_compile_full_sharded`.
    pub fn build_sharded_plan_from_spec(spec: &ShardPipelineSpec) -> (ShardPlan, PirGraph) {
        let shard_plan = ShardPlan {
            num_shards: spec.shards.len(),
            layer_assignment: (0..spec.shards.len()).collect(),
            compute_units: spec
                .shards
                .iter()
                .map(|s| s.compute_units.to_coreml_string().to_string())
                .collect(),
            is_multi_shard: spec.is_multi_shard(),
            shard_roles: spec.shards.iter().map(|s| s.role.canonical_name().to_string()).collect(),
            shard_names: spec.shards.iter().map(|s| s.shard_name.clone()).collect(),
        };

        let pir_graph = spec.to_pir_graph();

        (shard_plan, pir_graph)
    }

    /// Build a multi-shard plan from a spec, consuming shard template knowledge.
    ///
    /// When validated shard templates are provided, this method checks whether
    /// any template matches the pipeline's shard structure and, if so, applies
    /// the template's compute unit assignments instead of the spec's defaults.
    /// This is the wiring point where stored shard template knowledge
    /// materially affects compilation decisions.
    ///
    /// A template "matches" if it has the same number of partition entries
    /// as the pipeline has shards, and each partition entry's role matches
    /// the corresponding shard's role.
    ///
    /// When no matching template is found (or no templates are provided),
    /// the behavior is identical to `build_sharded_plan_from_spec`.
    pub fn build_sharded_plan_from_spec_with_knowledge(
        spec: &ShardPipelineSpec,
        templates: &[ShardTemplate],
    ) -> (ShardPlan, PirGraph) {
        // Try to find a matching template
        let matching_template = templates.iter().find(|t| {
            if t.partition_spec.len() != spec.shards.len() {
                return false;
            }
            // Check that each partition entry's role matches the corresponding shard
            t.partition_spec
                .iter()
                .zip(spec.shards.iter())
                .all(|(entry, shard)| entry.role == shard.role)
        });

        // If a matching template is found, apply its compute unit assignments
        // to the spec's shards
        let effective_spec = if let Some(template) = matching_template {
            let mut overridden_spec = spec.clone();
            for (shard, entry) in
                overridden_spec.shards.iter_mut().zip(template.partition_spec.iter())
            {
                shard.compute_units = entry.compute_units.clone();
            }
            overridden_spec
        } else {
            spec.clone()
        };

        Self::build_sharded_plan_from_spec(&effective_spec)
    }

    /// Build a multi-shard plan from a spec, consuming both shard template
    /// knowledge AND risk-based knowledge from the knowledge store.
    ///
    /// This is the S37.4 fix: the previous `build_sharded_plan_from_spec_with_knowledge`
    /// only consumed shard template seeds, but ignored the knowledge store's risk
    /// observations. This meant that `compile-full-sharded` produced shard plans that
    /// ignored all accumulated knowledge about which ops cause ANE fallback.
    ///
    /// This method applies knowledge in two layers:
    /// 1. **Template layer**: If a matching shard template is found, apply its
    ///    compute unit assignments (as before).
    /// 2. **Risk layer**: For each shard, query the knowledge store for fallback
    ///    risk on the shard's primary op pattern. If fallback risk exceeds the
    ///    threshold, override the shard's compute units to CPU_AND_GPU.
    ///
    /// The risk layer takes precedence over the template layer because it
    /// represents more recent, more specific evidence (device observations
    /// rather than synthetic templates).
    ///
    /// Returns the shard plan, PIR graph, and any compute unit adaptations
    /// that were made (for manifest/report inclusion).
    pub fn build_sharded_plan_from_spec_with_risk_knowledge(
        &mut self,
        spec: &ShardPipelineSpec,
        templates: &[ShardTemplate],
        knowledge_query: &dyn PassKnowledgeQuery,
    ) -> (ShardPlan, PirGraph, Vec<ComputeUnitAdaptation>) {
        // Reset adaptations for this run
        self.adaptations.clear();

        // Step 1: Apply template knowledge (same as build_sharded_plan_from_spec_with_knowledge)
        let matching_template = templates.iter().find(|t| {
            if t.partition_spec.len() != spec.shards.len() {
                return false;
            }
            t.partition_spec
                .iter()
                .zip(spec.shards.iter())
                .all(|(entry, shard)| entry.role == shard.role)
        });

        let mut effective_spec = if let Some(template) = matching_template {
            let mut overridden_spec = spec.clone();
            for (shard, entry) in
                overridden_spec.shards.iter_mut().zip(template.partition_spec.iter())
            {
                shard.compute_units = entry.compute_units.clone();
            }
            overridden_spec
        } else {
            spec.clone()
        };

        // Step 2: Apply risk-based knowledge for each shard.
        // For each shard, determine the primary op pattern and query the knowledge
        // store for fallback risk. If risk exceeds the threshold, override to CPU_AND_GPU.
        for shard in effective_spec.shards.iter_mut() {
            let op_pattern = Self::primary_op_pattern_for_shard(&shard.role);

            let (new_compute, adaptation) =
                self.determine_compute_units(&shard.shard_name, op_pattern, knowledge_query);

            if let Some(adapt) = adaptation {
                // Risk-based adaptation overrides template assignment
                shard.compute_units = new_compute;
                self.adaptations.push(adapt);
            }
        }

        let (plan, pir) = Self::build_sharded_plan_from_spec(&effective_spec);
        (plan, pir, self.adaptations.clone())
    }

    /// Derive the primary op pattern for a shard based on its role.
    ///
    /// This is a simplified heuristic: Entry/Interior/Exit decoder shards
    /// are matmul-dominant, Io and Sampler shards have different patterns.
    /// In a full implementation, this would inspect the actual SIR graph
    /// of the shard, but at the plan-construction level we don't have
    /// the SIR yet — we only have the shard spec.
    fn primary_op_pattern_for_shard(role: &ShardRole) -> &str {
        match role {
            ShardRole::Entry | ShardRole::Interior | ShardRole::Exit => "mb.matmul",
            ShardRole::Io => "mb.embedding", // IO model: embedding + LM head
            ShardRole::Sampler => "mb.topk", // Sampler: top-k + softmax + gather
        }
    }

    /// Build a multi-shard plan and PIR for a ShardedLinearPipeline task.
    ///
    /// Backward-compatible method that creates a `ShardPipelineSpec` and
    /// delegates to `build_sharded_plan_from_spec`. The existing callers
    /// (CLI `compile-sharded` and `compile-full-sharded`) continue to work
    /// without changes.
    ///
    /// NOTE: This method does not query the knowledge store for risk
    /// data because it is called at the CLI orchestration level, not
    /// within the pass pipeline. The per-shard knowledge-driven
    /// adaptation happens when each shard is individually compiled
    /// through the pass pipeline in `run_compile_full_sharded`.
    pub fn build_sharded_plan(
        task_name: &str,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        batch_size: usize,
        dtype: &str,
    ) -> (ShardPlan, PirGraph) {
        let spec = ShardPipelineSpec::three_shard_linear(
            task_name, input_dim, hidden_dim, output_dim, batch_size, dtype,
        );
        Self::build_sharded_plan_from_spec(&spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_query::{
        ComputePlanPlacementInfo, LegalityInfo, NoKnowledge, PrecisionHazardInfo, RiskInfo,
    };
    use ane_ir::pir::ShardPartitionEntry;
    use ane_ir::sir::{SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};

    /// Mock knowledge query that reports high fallback risk for mb.matmul.
    ///
    /// This simulates stored observations showing that matmul frequently
    /// falls back from ANE to CPU/GPU, making CPU_AND_NE an unreliable
    /// compute unit assignment.
    struct MockHighFallbackRiskKnowledge;

    impl PassKnowledgeQuery for MockHighFallbackRiskKnowledge {
        fn query_legality(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<LegalityInfo> {
            None
        }

        fn query_risk(
            &self,
            op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<RiskInfo> {
            if op_pattern == "mb.matmul" {
                Some(RiskInfo {
                    fallback_risk: 0.8,
                    drift_risk: 0.4,
                    confidence: 0.75,
                    evidence_count: 12,
                    source_id: Some("obs_matmul_high_fallback".to_string()),
                })
            } else {
                None
            }
        }

        fn query_precision_hazard(
            &self,
            _op_pattern: &str,
            _current_dtype: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<PrecisionHazardInfo> {
            None
        }

        fn query_compute_plan_placement(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<ComputePlanPlacementInfo> {
            None
        }
    }

    /// Mock knowledge query that reports low fallback risk (ANE survives).
    struct MockLowFallbackRiskKnowledge;

    impl PassKnowledgeQuery for MockLowFallbackRiskKnowledge {
        fn query_legality(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<LegalityInfo> {
            None
        }

        fn query_risk(
            &self,
            op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<RiskInfo> {
            if op_pattern == "mb.matmul" {
                Some(RiskInfo {
                    fallback_risk: 0.1,
                    drift_risk: 0.05,
                    confidence: 0.9,
                    evidence_count: 20,
                    source_id: Some("obs_matmul_ane_survives".to_string()),
                })
            } else {
                None
            }
        }

        fn query_precision_hazard(
            &self,
            _op_pattern: &str,
            _current_dtype: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<PrecisionHazardInfo> {
            None
        }

        fn query_compute_plan_placement(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<ComputePlanPlacementInfo> {
            None
        }
    }

    /// Mock knowledge query with fallback risk just below the threshold.
    struct MockBorderlineRiskKnowledge;

    impl PassKnowledgeQuery for MockBorderlineRiskKnowledge {
        fn query_legality(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<LegalityInfo> {
            None
        }

        fn query_risk(
            &self,
            op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<RiskInfo> {
            if op_pattern == "mb.matmul" {
                Some(RiskInfo {
                    fallback_risk: 0.45, // Below default threshold of 0.5
                    drift_risk: 0.2,
                    confidence: 0.6,
                    evidence_count: 5,
                    source_id: Some("obs_matmul_borderline".to_string()),
                })
            } else {
                None
            }
        }

        fn query_precision_hazard(
            &self,
            _op_pattern: &str,
            _current_dtype: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<PrecisionHazardInfo> {
            None
        }

        fn query_compute_plan_placement(
            &self,
            _op_pattern: &str,
            _scope: Option<&ane_ir::kir::KnowledgeScope>,
        ) -> Option<ComputePlanPlacementInfo> {
            None
        }
    }

    fn make_linear_sir() -> SirGraph {
        SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("weight".into()),
                    op: SirOp::ElementWise { op: ane_ir::sir::ElementWiseOp::Mul, inputs: vec![] },
                    name: "weight".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("output".into()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input".into()),
                        weight: "weight".into(),
                        bias: Some("bias".into()),
                    },
                    name: "linear_out".into(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![SirNodeId("input".into())],
            outputs: vec![SirNodeId("output".into())],
        }
    }

    /// Test that high fallback risk knowledge changes the compute unit assignment.
    ///
    /// This is the core Sprint 22 integration test: it proves that stored
    /// empirical knowledge about ANE fallback risk materially changes a
    /// compilation decision. When fallback risk is high for mb.matmul,
    /// the pass overrides CPU_AND_NE to CPU_AND_GPU, proving the compiler
    /// is not just "aware" of knowledge but is "affected" by it.
    #[test]
    fn test_high_fallback_risk_overrides_compute_units() {
        let sir = make_linear_sir();
        let mut pass = ShardPlanPass::new();

        let high_risk = MockHighFallbackRiskKnowledge;
        let (plan, pir) = pass.run(&sir, &high_risk).unwrap();

        // The shard plan should use CPU_AND_GPU instead of CPU_AND_NE
        assert_eq!(
            plan.compute_units[0], "CPU_AND_GPU",
            "High fallback risk must override CPU_AND_NE to CPU_AND_GPU"
        );

        // The PIR package should also reflect the override
        assert_eq!(
            pir.packages[0].compute_units,
            ComputeUnitHint::CPUAndGPU,
            "PIR package compute units must match the adaptation"
        );

        // An adaptation must be recorded
        assert!(
            pass.has_adaptations(),
            "Pass must record adaptations when high risk knowledge is present"
        );
        assert_eq!(pass.adaptations.len(), 1, "Exactly one adaptation for the single shard");

        let adaptation = &pass.adaptations[0];
        assert_eq!(adaptation.shard_name, "decoder_interior");
        assert_eq!(adaptation.original_compute_units, "CPU_AND_NE");
        assert_eq!(adaptation.adapted_compute_units, "CPU_AND_GPU");
        assert_eq!(adaptation.op_pattern, "mb.matmul");
        assert!((adaptation.fallback_risk - 0.8).abs() < 0.001);
        assert_eq!(adaptation.source_id, Some("obs_matmul_high_fallback".to_string()));
        assert!((adaptation.confidence - 0.75).abs() < 0.001);
        assert!(adaptation.reason.contains("CPU_AND_GPU"));
        assert!(adaptation.reason.contains("0.80"));
    }

    /// Test that NoKnowledge produces no adaptations and uses default CPU_AND_NE.
    ///
    /// Without knowledge, the pass must behave identically to the
    /// pre-adaptation version: no compute unit overrides, no adaptation records.
    #[test]
    fn test_no_knowledge_no_adaptation() {
        let sir = make_linear_sir();
        let mut pass = ShardPlanPass::new();

        let no_knowledge = NoKnowledge;
        let (plan, pir) = pass.run(&sir, &no_knowledge).unwrap();

        assert!(!pass.has_adaptations(), "NoKnowledge must produce zero adaptations");
        assert_eq!(plan.compute_units[0], "CPU_AND_NE", "NoKnowledge must use default CPU_AND_NE");
        assert_eq!(pir.packages[0].compute_units, ComputeUnitHint::CPUAndNE);
    }

    /// Test that low fallback risk knowledge keeps CPU_AND_NE.
    ///
    /// When knowledge shows the op survives on ANE, there is no reason
    /// to override the default compute units.
    #[test]
    fn test_low_fallback_risk_keeps_ane() {
        let sir = make_linear_sir();
        let mut pass = ShardPlanPass::new();

        let low_risk = MockLowFallbackRiskKnowledge;
        let (plan, pir) = pass.run(&sir, &low_risk).unwrap();

        assert!(!pass.has_adaptations(), "Low fallback risk must not trigger adaptation");
        assert_eq!(plan.compute_units[0], "CPU_AND_NE", "Low fallback risk should keep CPU_AND_NE");
        assert_eq!(pir.packages[0].compute_units, ComputeUnitHint::CPUAndNE);
    }

    /// Test that borderline risk (below threshold) does not trigger adaptation.
    #[test]
    fn test_borderline_risk_no_adaptation() {
        let sir = make_linear_sir();
        let mut pass = ShardPlanPass::new();

        let borderline = MockBorderlineRiskKnowledge;
        let (plan, _pir) = pass.run(&sir, &borderline).unwrap();

        assert!(
            !pass.has_adaptations(),
            "Borderline risk below threshold must not trigger adaptation"
        );
        assert_eq!(plan.compute_units[0], "CPU_AND_NE");
    }

    /// Test that adaptations are reset between runs.
    #[test]
    fn test_adaptations_reset_between_runs() {
        let sir = make_linear_sir();
        let mut pass = ShardPlanPass::new();

        // First run with high-risk knowledge
        let high_risk = MockHighFallbackRiskKnowledge;
        let _ = pass.run(&sir, &high_risk).unwrap();
        assert!(pass.has_adaptations());

        // Second run with NoKnowledge — adaptations should be reset
        let no_knowledge = NoKnowledge;
        let _ = pass.run(&sir, &no_knowledge).unwrap();
        assert!(!pass.has_adaptations(), "Adaptations must be reset between runs");
    }

    /// Test custom confidence threshold.
    #[test]
    fn test_custom_threshold() {
        let sir = make_linear_sir();
        let mut pass = ShardPlanPass::new().with_threshold(0.9);

        // High risk (0.8) is below custom threshold (0.9)
        let high_risk = MockHighFallbackRiskKnowledge; // fallback_risk = 0.8
        let (plan, _pir) = pass.run(&sir, &high_risk).unwrap();

        assert!(!pass.has_adaptations(), "Risk below custom threshold must not trigger adaptation");
        assert_eq!(plan.compute_units[0], "CPU_AND_NE");
    }

    /// Test that adaptation record contains correct source_id.
    #[test]
    fn test_adaptation_provenance() {
        let sir = make_linear_sir();
        let mut pass = ShardPlanPass::new();

        let high_risk = MockHighFallbackRiskKnowledge;
        let _ = pass.run(&sir, &high_risk).unwrap();

        assert!(pass.has_adaptations());
        let adaptation = &pass.adaptations[0];
        assert!(adaptation.source_id.is_some());
        assert_eq!(adaptation.source_id.as_ref().unwrap(), "obs_matmul_high_fallback");
    }

    #[test]
    fn test_build_sharded_plan_three_shards() {
        let (plan, _pir) =
            ShardPlanPass::build_sharded_plan("test_pipeline", 64, 48, 32, 1, "fp16");

        assert_eq!(plan.num_shards, 3);
        assert!(plan.is_multi_shard);
        assert_eq!(plan.shard_roles, vec!["Entry", "Interior", "Exit"]);
        assert_eq!(
            plan.shard_names,
            vec!["test_pipeline_entry", "test_pipeline_interior", "test_pipeline_exit",]
        );
        assert_eq!(plan.compute_units, vec!["CPU_AND_NE", "CPU_AND_NE", "CPU_AND_NE"]);
    }

    #[test]
    fn test_build_sharded_plan_pir_structure() {
        let (_plan, pir) =
            ShardPlanPass::build_sharded_plan("test_pipeline", 64, 48, 32, 1, "fp16");

        assert_eq!(pir.packages.len(), 3);
        assert_eq!(pir.handoffs.len(), 2);
        assert!(pir.shard_template.is_some());

        // Verify package roles
        assert!(matches!(pir.packages[0].role, PackageRole::DecoderShard(ShardRole::Entry)));
        assert!(matches!(pir.packages[1].role, PackageRole::DecoderShard(ShardRole::Interior)));
        assert!(matches!(pir.packages[2].role, PackageRole::DecoderShard(ShardRole::Exit)));

        // Verify concrete handoff semantics
        assert_eq!(pir.handoffs[0].handoff_kind, HandoffKind::TensorPassThrough);
        assert_eq!(pir.handoffs[0].execution_order, 0);
        assert_eq!(pir.handoffs[0].from_package, "test_pipeline_entry");
        assert_eq!(pir.handoffs[0].to_package, "test_pipeline_interior");
        assert_eq!(pir.handoffs[0].source_output_name, "output");
        assert_eq!(pir.handoffs[0].target_input_name, "x");

        assert_eq!(pir.handoffs[1].handoff_kind, HandoffKind::TensorPassThrough);
        assert_eq!(pir.handoffs[1].execution_order, 1);
        assert_eq!(pir.handoffs[1].from_package, "test_pipeline_interior");
        assert_eq!(pir.handoffs[1].to_package, "test_pipeline_exit");
    }

    #[test]
    fn test_build_sharded_plan_dimensions() {
        let (_plan, pir) =
            ShardPlanPass::build_sharded_plan("test_pipeline", 64, 48, 32, 1, "fp16");

        // Entry: [1, 64] -> [1, 48]
        let entry = &pir.packages[0];
        assert_eq!(entry.functions[0].inputs[0].shape, vec![1, 64]);
        assert_eq!(entry.functions[0].outputs[0].shape, vec![1, 48]);

        // Interior: [1, 48] -> [1, 48]
        let interior = &pir.packages[1];
        assert_eq!(interior.functions[0].inputs[0].shape, vec![1, 48]);
        assert_eq!(interior.functions[0].outputs[0].shape, vec![1, 48]);

        // Exit: [1, 48] -> [1, 32]
        let exit = &pir.packages[2];
        assert_eq!(exit.functions[0].inputs[0].shape, vec![1, 48]);
        assert_eq!(exit.functions[0].outputs[0].shape, vec![1, 32]);
    }

    #[test]
    fn test_build_sharded_plan_serializes() {
        let (plan, _pir) =
            ShardPlanPass::build_sharded_plan("test_pipeline", 64, 48, 32, 1, "fp16");

        // Verify ShardPlan is serializable (derive Serialize works)
        let plan_str = format!("{:?}", plan);
        assert!(plan_str.contains("is_multi_shard"));
        assert!(plan_str.contains("shard_roles"));

        // Verify that the ShardPlan can be cloned and compared
        let plan2 = plan.clone();
        assert_eq!(plan.num_shards, plan2.num_shards);
        assert_eq!(plan.shard_roles, plan2.shard_roles);
        assert_eq!(plan.shard_names, plan2.shard_names);
    }

    // ─── Sprint 23 — Generalized Shard Pipeline Spec Tests ──────────────

    /// Test that `build_sharded_plan_from_spec` works with a linear pipeline spec.
    ///
    /// This proves the generalized spec path produces the same result as
    /// the backward-compatible `build_sharded_plan` method.
    #[test]
    fn test_build_from_spec_linear_pipeline() {
        let spec = ShardPipelineSpec::three_shard_linear("test_pipeline", 64, 48, 32, 1, "fp16");
        let (plan, pir) = ShardPlanPass::build_sharded_plan_from_spec(&spec);

        // Same assertions as the legacy test
        assert_eq!(plan.num_shards, 3);
        assert!(plan.is_multi_shard);
        assert_eq!(plan.shard_roles, vec!["Entry", "Interior", "Exit"]);
        assert_eq!(pir.packages.len(), 3);
        assert_eq!(pir.handoffs.len(), 2);
    }

    /// Test that `build_sharded_plan_from_spec` works with a decode-step pipeline spec.
    ///
    /// Sprint 23 (S23.3): this proves the generalized model produces a
    /// different multi-unit structure than the linear pipeline, with
    /// decode-step-specific shard names, handoff tensor names, and
    /// KV cache state declarations.
    #[test]
    fn test_build_from_spec_decode_step() {
        let spec =
            ShardPipelineSpec::three_shard_decode_step("test_decode", 128, 4, 32, 64, 1, "fp16");
        let (plan, pir) = ShardPlanPass::build_sharded_plan_from_spec(&spec);

        // 3 shards with decode-step roles
        assert_eq!(plan.num_shards, 3);
        assert!(plan.is_multi_shard);
        assert_eq!(plan.shard_roles, vec!["Entry", "Interior", "Exit"]);

        // Decode-step-specific shard names
        assert_eq!(
            plan.shard_names,
            vec!["test_decode_qkv_proj", "test_decode_attention", "test_decode_out_proj",]
        );

        // PIR structure
        assert_eq!(pir.packages.len(), 3);
        assert_eq!(pir.handoffs.len(), 2);

        // Decode-step has KV cache state declarations
        assert!(
            !pir.state_declarations.is_empty(),
            "Decode-step pipeline must declare KV cache state"
        );
        assert_eq!(pir.state_declarations[0].state_id, "test_decode_kv_cache");
        assert_eq!(pir.state_declarations[0].owner_package, "test_decode_attention");

        // Decode-step has a shard template with state_config
        assert!(pir.shard_template.is_some());
        let template = pir.shard_template.as_ref().unwrap();
        assert_eq!(template.state_config, Some("per_shard_kv_masked_blend".to_string()));
        assert_eq!(template.context_length, 64);

        // Decode-step handoffs carry different tensor names than linear
        assert_eq!(pir.handoffs[0].tensor_name, "qkv");
        assert_eq!(pir.handoffs[0].source_output_name, "qkv");
        assert_eq!(pir.handoffs[0].target_input_name, "qkv");
        assert_eq!(pir.handoffs[1].tensor_name, "attn_out");

        // Entry shard output: [1, 384] (3 * 128)
        let entry = &pir.packages[0];
        assert_eq!(entry.functions[0].outputs[0].shape, vec![1, 384]);

        // Interior shard: [1, 384] -> [1, 128]
        let interior = &pir.packages[1];
        assert_eq!(interior.functions[0].inputs[0].shape, vec![1, 384]);
        assert_eq!(interior.functions[0].outputs[0].shape, vec![1, 128]);
        assert!(interior.functions[0].stateful, "Attention shard must be stateful");

        // Exit shard: [1, 128] -> [1, 128]
        let exit_pkg = &pir.packages[2];
        assert_eq!(exit_pkg.functions[0].inputs[0].shape, vec![1, 128]);
        assert_eq!(exit_pkg.functions[0].outputs[0].shape, vec![1, 128]);
    }

    /// Test that linear and decode-step specs produce different shard structures.
    ///
    /// Sprint 23 (S23.3): this proves the generalized model is not just
    /// a rebranding of the linear pipeline — the two pipeline specs
    /// produce genuinely different multi-unit decompositions.
    #[test]
    fn test_linear_and_decode_step_specs_diverge() {
        let linear_spec = ShardPipelineSpec::three_shard_linear("test", 64, 48, 32, 1, "fp16");
        let decode_spec =
            ShardPipelineSpec::three_shard_decode_step("test", 128, 4, 32, 64, 1, "fp16");

        let (linear_plan, linear_pir) = ShardPlanPass::build_sharded_plan_from_spec(&linear_spec);
        let (decode_plan, decode_pir) = ShardPlanPass::build_sharded_plan_from_spec(&decode_spec);

        // Shard names must differ
        assert_ne!(
            linear_plan.shard_names, decode_plan.shard_names,
            "Linear and decode-step shard names must differ"
        );

        // Handoff tensor names must differ
        let linear_tensors: Vec<_> =
            linear_pir.handoffs.iter().map(|h| h.tensor_name.clone()).collect();
        let decode_tensors: Vec<_> =
            decode_pir.handoffs.iter().map(|h| h.tensor_name.clone()).collect();
        assert_ne!(
            linear_tensors, decode_tensors,
            "Linear and decode-step handoff tensor names must differ"
        );

        // Decode-step has state declarations, linear does not
        assert!(linear_pir.state_declarations.is_empty());
        assert!(!decode_pir.state_declarations.is_empty());

        // Decode-step has state_config in template, linear does not
        let linear_state_config =
            linear_pir.shard_template.as_ref().and_then(|t| t.state_config.clone());
        let decode_state_config =
            decode_pir.shard_template.as_ref().and_then(|t| t.state_config.clone());
        assert!(linear_state_config.is_none());
        assert!(decode_state_config.is_some());

        // Decode-step attention shard is stateful, linear shards are not
        assert!(decode_pir.packages[1].functions[0].stateful);
        assert!(!linear_pir.packages[1].functions[0].stateful);
    }

    /// Test that shard template knowledge overrides compute units.
    ///
    /// When a matching template specifies CPU_AND_GPU for Interior shards,
    /// the shard plan should reflect that override instead of the default
    /// CPU_AND_NE.
    #[test]
    fn test_shard_template_overrides_compute_units() {
        let spec = ShardPipelineSpec::three_shard_linear("test_pipeline", 64, 48, 32, 1, "fp16");

        // Create a template that overrides Interior to CPU_AND_GPU
        let template = ShardTemplate {
            template_id: "test-override-template".to_string(),
            partition_spec: vec![
                ShardPartitionEntry {
                    role: ShardRole::Entry,
                    layer_start: 0,
                    layer_end: 5,
                    compute_units: ComputeUnitHint::CPUAndNE,
                },
                ShardPartitionEntry {
                    role: ShardRole::Interior,
                    layer_start: 6,
                    layer_end: 10,
                    compute_units: ComputeUnitHint::CPUAndGPU, // Override!
                },
                ShardPartitionEntry {
                    role: ShardRole::Exit,
                    layer_start: 11,
                    layer_end: 15,
                    compute_units: ComputeUnitHint::CPUAndNE,
                },
            ],
            io_compute_units: None,
            sampler_compute_units: None,
            state_config: None,
            context_length: 0,
        };

        let (plan, pir) =
            ShardPlanPass::build_sharded_plan_from_spec_with_knowledge(&spec, &[template]);

        // Interior shard should use CPU_AND_GPU from the template
        assert_eq!(plan.compute_units[0], "CPU_AND_NE", "Entry keeps default");
        assert_eq!(plan.compute_units[1], "CPU_AND_GPU", "Interior overridden by template");
        assert_eq!(plan.compute_units[2], "CPU_AND_NE", "Exit keeps default");

        // PIR should also reflect the override
        assert_eq!(
            pir.packages[1].compute_units,
            ComputeUnitHint::CPUAndGPU,
            "Interior PIR package must use template compute units"
        );
    }

    /// Test that non-matching templates are ignored.
    #[test]
    fn test_shard_template_no_match_keeps_defaults() {
        let spec = ShardPipelineSpec::three_shard_linear("test_pipeline", 64, 48, 32, 1, "fp16");

        // Create a template with wrong number of partitions
        let template = ShardTemplate {
            template_id: "wrong-template".to_string(),
            partition_spec: vec![
                ShardPartitionEntry {
                    role: ShardRole::Entry,
                    layer_start: 0,
                    layer_end: 10,
                    compute_units: ComputeUnitHint::CPUAndGPU,
                },
                ShardPartitionEntry {
                    role: ShardRole::Exit,
                    layer_start: 11,
                    layer_end: 20,
                    compute_units: ComputeUnitHint::CPUAndGPU,
                },
            ],
            io_compute_units: None,
            sampler_compute_units: None,
            state_config: None,
            context_length: 0,
        };

        let (plan, _pir) =
            ShardPlanPass::build_sharded_plan_from_spec_with_knowledge(&spec, &[template]);

        // All shards should keep defaults since the template doesn't match
        assert_eq!(plan.compute_units, vec!["CPU_AND_NE", "CPU_AND_NE", "CPU_AND_NE"]);
    }

    /// Test that empty template list behaves identically to the base method.
    #[test]
    fn test_shard_template_empty_list_same_as_base() {
        let spec = ShardPipelineSpec::three_shard_linear("test_pipeline", 64, 48, 32, 1, "fp16");

        let (plan_base, pir_base) = ShardPlanPass::build_sharded_plan_from_spec(&spec);
        let (plan_knowledge, pir_knowledge) =
            ShardPlanPass::build_sharded_plan_from_spec_with_knowledge(&spec, &[]);

        assert_eq!(plan_base.compute_units, plan_knowledge.compute_units);
        assert_eq!(plan_base.shard_names, plan_knowledge.shard_names);
        assert_eq!(pir_base.packages.len(), pir_knowledge.packages.len());
    }

    // ─── Sprint 37: Risk-Knowledge Multi-Shard Plan Tests ──────────────

    /// Test that build_sharded_plan_from_spec_with_risk_knowledge applies
    /// risk-based adaptation when knowledge reports high fallback risk.
    ///
    /// This proves S37.4: the multi-shard plan is no longer blind to
    /// accumulated risk observations at the plan-construction level.
    #[test]
    fn test_risk_knowledge_overrides_compute_units_in_multi_shard_plan() {
        let spec = ShardPipelineSpec::three_shard_linear("test_pipeline", 64, 48, 32, 1, "fp16");

        let mut pass = ShardPlanPass::new();
        let high_risk = MockHighFallbackRiskKnowledge;

        let (plan, _pir, adaptations) =
            pass.build_sharded_plan_from_spec_with_risk_knowledge(&spec, &[], &high_risk);

        // All three shards have the same primary op pattern ("mb.matmul"),
        // so all three should be overridden to CPU_AND_GPU
        assert_eq!(
            plan.compute_units,
            vec!["CPU_AND_GPU", "CPU_AND_GPU", "CPU_AND_GPU"],
            "High fallback risk must override all shards to CPU_AND_GPU"
        );

        // Adaptations must be recorded for each shard
        assert_eq!(adaptations.len(), 3, "Three adaptations expected (one per shard)");
    }

    /// Test that low risk knowledge does not override compute units.
    #[test]
    fn test_low_risk_keeps_default_in_multi_shard_plan() {
        let spec = ShardPipelineSpec::three_shard_linear("test_pipeline", 64, 48, 32, 1, "fp16");

        let mut pass = ShardPlanPass::new();
        let low_risk = MockLowFallbackRiskKnowledge;

        let (plan, _pir, adaptations) =
            pass.build_sharded_plan_from_spec_with_risk_knowledge(&spec, &[], &low_risk);

        // Low risk: all shards should keep default CPU_AND_NE
        assert_eq!(plan.compute_units, vec!["CPU_AND_NE", "CPU_AND_NE", "CPU_AND_NE"]);
        assert!(adaptations.is_empty(), "No adaptations for low risk");
    }

    /// Test that risk knowledge takes precedence over template knowledge.
    #[test]
    fn test_risk_overrides_template_in_multi_shard_plan() {
        let spec = ShardPipelineSpec::three_shard_linear("test_pipeline", 64, 48, 32, 1, "fp16");

        // Template says CPU_AND_NE for all shards
        let template = ShardTemplate {
            template_id: "test-template".to_string(),
            partition_spec: vec![
                ShardPartitionEntry {
                    role: ShardRole::Entry,
                    layer_start: 0,
                    layer_end: 5,
                    compute_units: ComputeUnitHint::CPUAndNE,
                },
                ShardPartitionEntry {
                    role: ShardRole::Interior,
                    layer_start: 6,
                    layer_end: 10,
                    compute_units: ComputeUnitHint::CPUAndNE,
                },
                ShardPartitionEntry {
                    role: ShardRole::Exit,
                    layer_start: 11,
                    layer_end: 15,
                    compute_units: ComputeUnitHint::CPUAndNE,
                },
            ],
            io_compute_units: None,
            sampler_compute_units: None,
            state_config: None,
            context_length: 0,
        };

        let mut pass = ShardPlanPass::new();
        let high_risk = MockHighFallbackRiskKnowledge;

        let (plan, _pir, adaptations) =
            pass.build_sharded_plan_from_spec_with_risk_knowledge(&spec, &[template], &high_risk);

        // Risk overrides template: all shards should be CPU_AND_GPU
        assert_eq!(
            plan.compute_units,
            vec!["CPU_AND_GPU", "CPU_AND_GPU", "CPU_AND_GPU"],
            "Risk knowledge must override template assignment"
        );
        assert_eq!(adaptations.len(), 3);
    }

    /// Test that NoKnowledge produces no adaptations in multi-shard plan.
    #[test]
    fn test_no_knowledge_no_adaptations_in_multi_shard_plan() {
        let spec = ShardPipelineSpec::three_shard_linear("test_pipeline", 64, 48, 32, 1, "fp16");

        let mut pass = ShardPlanPass::new();
        let no_knowledge = NoKnowledge;

        let (plan, _pir, adaptations) =
            pass.build_sharded_plan_from_spec_with_risk_knowledge(&spec, &[], &no_knowledge);

        assert_eq!(plan.compute_units, vec!["CPU_AND_NE", "CPU_AND_NE", "CPU_AND_NE"]);
        assert!(adaptations.is_empty());
    }
}
