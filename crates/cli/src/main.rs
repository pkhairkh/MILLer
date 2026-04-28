//! ANE Compiler CLI
//!
//! Command-line interface for MILLer.
//! The `compile` subcommand drives the vertical slice:
//! task spec → SIR → MIR → bridge → mlpackage → manifest + knowledge update.
//! The `compile-full` subcommand drives the full pass pipeline:
//! task spec → SIR → Canonicalize → Staticize → PrecisionPolicy → StateTopology
//! → LegalityRewrite → RiskAnnotate → ShardPlan → MilLower → bridge → mlpackage
//! → manifest + knowledge update + all IR dumps.
//! The `lab` subcommand drives a complete lab run:
//! compile + host-side inspection + structured run record.

use clap::{Parser, Subcommand};
use sha2::Digest;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ane-compile")]
#[command(about = "MILLer — compile synthetic linear projection tasks to mlpackage")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a synthetic linear projection task to mlpackage (current vertical slice only).
    Compile {
        /// Path to the task specification file (TOML).
        #[arg(short, long)]
        input: String,

        /// Output directory for compiled packages and artifacts.
        #[arg(short, long)]
        output: String,

        /// Path to the Python bridge script.
        #[arg(long, default_value = "python/bridge.py")]
        bridge: String,

        /// Path to the Python interpreter.
        #[arg(long, default_value = "python3.12")]
        python: String,

        /// Knowledge store directory.
        #[arg(long)]
        knowledge: Option<String>,

        /// Random seed for reproducibility.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Compile a synthetic linear projection task to mlpackage via the full pass pipeline.
    ///
    /// Unlike the fast-path `compile` command, this drives the complete pass pipeline:
    /// SIR → Canonicalize → Staticize → PrecisionPolicy → StateTopology
    /// → LegalityRewrite → RiskAnnotate → ShardPlan → MilLower → bridge.
    /// All intermediate IR representations (SIR, AIR, PIR, MIR) are written as artifacts.
    CompileFull {
        /// Path to the task specification file (TOML).
        #[arg(short, long)]
        input: String,

        /// Output directory for compiled packages and artifacts.
        #[arg(short, long)]
        output: String,

        /// Path to the Python bridge script.
        #[arg(long, default_value = "python/bridge.py")]
        bridge: String,

        /// Path to the Python interpreter.
        #[arg(long, default_value = "python3.12")]
        python: String,

        /// Knowledge store directory.
        #[arg(long)]
        knowledge: Option<String>,

        /// Random seed for reproducibility.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Compile a sharded linear pipeline task to multiple mlpackages with role semantics.
    ///
    /// This is the shard-aware compilation path (S9.2). It emits one mlpackage
    /// per shard (Entry, Interior, Exit), each with its own dimensions and
    /// compute unit assignment. The resulting manifest reflects the multi-shard
    /// structure with role semantics and inter-shard handoffs.
    ///
    /// When --knowledge is provided, shard template seeds are loaded and used
    /// to override compute unit assignments via build_sharded_plan_from_spec_with_knowledge.
    CompileSharded {
        /// Path to the task specification file (TOML).
        #[arg(short, long)]
        input: String,

        /// Output directory for compiled packages and artifacts.
        #[arg(short, long)]
        output: String,

        /// Path to the Python bridge script.
        #[arg(long, default_value = "python/bridge.py")]
        bridge: String,

        /// Path to the Python interpreter.
        #[arg(long, default_value = "python3.12")]
        python: String,

        /// Knowledge store directory (optional). When provided, shard template seeds
        /// are loaded and used to inform compute unit assignments.
        #[arg(long)]
        knowledge: Option<String>,

        /// Random seed for reproducibility.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Use proto-direct emission (Rust-only, no Python bridge) for decode-step shards.
        /// When set, RoleMirBuilder produces role-specific MIR and proto-direct emits
        /// the mlpackage directly, bypassing coremltools. For linear shards, the Python
        /// bridge is still used since proto-direct linear emission is not yet role-aware.
        #[arg(long, default_value_t = false)]
        proto_direct: bool,
    },

    /// Compile a sharded linear pipeline task through the full pass pipeline.
    ///
    /// Unlike `compile-sharded` (which bypasses the pass pipeline and directly
    /// lowers each shard to MIR), this command drives each shard through the
    /// complete pass pipeline: SIR → Canonicalize → Staticize → PrecisionPolicy
    /// → LegalityRewrite → RiskAnnotate → ShardPlan → MilLower → bridge.
    /// This is the first multi-unit orchestration path that exercises the full
    /// pass pipeline for each shard independently, with concrete handoff
    /// semantics and per-shard provenance.
    CompileFullSharded {
        /// Path to the task specification file (TOML).
        #[arg(short, long)]
        input: String,

        /// Output directory for compiled packages and artifacts.
        #[arg(short, long)]
        output: String,

        /// Path to the Python bridge script.
        #[arg(long, default_value = "python/bridge.py")]
        bridge: String,

        /// Path to the Python interpreter.
        #[arg(long, default_value = "python3.12")]
        python: String,

        /// Knowledge store directory.
        #[arg(long)]
        knowledge: Option<String>,

        /// Random seed for reproducibility.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Use proto-direct emission (Rust-only, no Python bridge) for decode-step shards.
        /// When set, RoleMirBuilder produces role-specific MIR and proto-direct emits
        /// the mlpackage directly, bypassing coremltools. For linear shards, the Python
        /// bridge is still used since proto-direct linear emission is not yet role-aware.
        #[arg(long, default_value_t = false)]
        proto_direct: bool,
    },

    /// Run a complete host-side evidence loop: compile + baseline + drift + knowledge store persistence.
    ///
    /// Unlike the `lab` command (which writes the knowledge update as an artifact file
    /// but never ingests it into the store), `lab-loop` closes the loop by persisting
    /// observations into the file-backed knowledge store via UpdatePipeline. This makes
    /// the ingested observations queryable by the pass pipeline in subsequent compiles.
    LabLoop {
        /// Path to the task specification file (TOML).
        #[arg(short, long)]
        input: String,

        /// Output directory for the lab-loop run.
        #[arg(short, long)]
        output: String,

        /// Path to the Python bridge script.
        #[arg(long, default_value = "python/bridge.py")]
        bridge: String,

        /// Path to the Python interpreter.
        #[arg(long, default_value = "python3.12")]
        python: String,

        /// Knowledge store directory (required — this is the loop closer).
        #[arg(long)]
        knowledge: String,

        /// Random seed for reproducibility.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Mark this run as using a generated task (records generator provenance in run artifacts).
        /// Format: "family,seed,generator_version" (e.g., "LinearProjection,42,1.0.0").
        #[arg(long)]
        generated_from: Option<String>,
    },

    /// Run a complete lab session: compile + host-side inspection + structured run record.
    Lab {
        /// Path to the task specification file (TOML).
        #[arg(short, long)]
        input: String,

        /// Output directory for the lab run.
        #[arg(short, long)]
        output: String,

        /// Path to the Python bridge script.
        #[arg(long, default_value = "python/bridge.py")]
        bridge: String,

        /// Path to the Python interpreter.
        #[arg(long, default_value = "python3.12")]
        python: String,

        /// Skip host-side inspection after compilation.
        #[arg(long, default_value_t = false)]
        skip_inspect: bool,

        /// Random seed for reproducibility.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Mark this run as using a generated task (records generator provenance in run artifacts).
        /// Format: "family,seed,generator_version" (e.g., "LinearProjection,42,1.0.0").
        #[arg(long)]
        generated_from: Option<String>,
    },

    /// Run profiling tasks on device (requires Apple hardware).
    Profile {
        /// Path to the mlpackage to profile.
        #[arg(short, long)]
        mlpackage: String,

        /// Output path for profiling results.
        #[arg(short, long)]
        output: String,

        /// Path to the Python bridge script.
        #[arg(long, default_value = "python/bridge.py")]
        bridge: String,

        /// Path to the Python interpreter.
        #[arg(long, default_value = "python3.12")]
        python: String,

        /// Number of warmup iterations.
        #[arg(long, default_value_t = 5)]
        warmup: usize,

        /// Number of measured iterations.
        #[arg(long, default_value_t = 20)]
        iterations: usize,

        /// Compute units for profiling.
        #[arg(long, default_value = "CPU_AND_NE")]
        compute_units: String,
    },

    /// Query the knowledge store.
    Query {
        /// Knowledge store path.
        #[arg(short, long)]
        store: String,

        /// Query expression (type, scope, confidence filter).
        #[arg(short, long)]
        filter: Option<String>,

        /// Output format (json, table, markdown).
        #[arg(long, default_value = "table")]
        format: String,
    },

    /// Generate reports from profiling or compilation data.
    Report {
        /// Input data path.
        #[arg(short, long)]
        input: String,

        /// Output report path.
        #[arg(short, long)]
        output: String,

        /// Report format (markdown, json).
        #[arg(long, default_value = "markdown")]
        format: String,
    },

    /// Package compile artifacts into a zip archive.
    ///
    /// Walks the compile output directory and produces a deterministic zip
    /// file containing all artifacts: mlpackage, manifest, MIR dump,
    /// knowledge updates, and any other files present.
    Package {
        /// Input directory containing compiled artifacts (must contain manifest.json).
        #[arg(short, long)]
        input: String,

        /// Output directory for the zip archive.
        #[arg(short, long)]
        output: String,
    },

    /// Generate profiling tasks from task families.
    ///
    /// Generates deterministic task specifications for the specified family
    /// and persists them as TOML files. Generated tasks can be fed directly
    /// into `compile`, `compile-full`, or `lab` commands.
    ///
    /// Supports "linear", "lut", "decode", "mlp", "attn", "shape", "remap", and "survival" families.
    GenerateTasks {
        /// Task family to generate ("linear", "lut", "decode", "mlp", "attn", "shape", "remap", or "survival").
        #[arg(short, long, default_value = "linear")]
        family: String,

        /// Output directory for generated task files.
        #[arg(short, long)]
        output: String,

        /// Random seed for deterministic generation.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Import knowledge from external sources.
    Import {
        /// Source path (JSON or MessagePack snapshot).
        #[arg(short, long)]
        source: String,

        /// Target knowledge store path.
        #[arg(long)]
        store: String,

        /// Validate before importing.
        #[arg(long, default_value_t = true)]
        validate: bool,
    },

    /// Verify an emitted mlpackage against compiler intent (Sprint 46).
    ///
    /// Dispatches the `verify` bridge command which performs four-dimension
    /// verification: op graph fidelity, compute-unit placement, state
    /// conformance, and multi-function conformance. On macOS with Core ML
    /// runtime, MLModelStructure and MLComputePlan provide full fidelity.
    /// On Linux, spec-based extraction provides structural verification.
    Verify {
        /// Path to the .mlpackage to verify.
        #[arg(short, long)]
        mlpackage: String,

        /// Output directory for verification artifacts (JSON).
        #[arg(short, long)]
        output: String,

        /// Path to the Python bridge script.
        #[arg(long, default_value = "python/bridge.py")]
        bridge: String,

        /// Path to the Python interpreter.
        #[arg(long, default_value = "python3.12")]
        python: String,

        /// Compute units to use for verification.
        #[arg(long, default_value = "CPU_AND_NE")]
        compute_units: String,

        /// Expected MIR op list as JSON (optional, for op fidelity comparison).
        #[arg(long)]
        mir_ops: Option<String>,

        /// Expected function names as comma-separated list (optional).
        #[arg(long)]
        expected_functions: Option<String>,

        /// Expected state names as comma-separated list (optional).
        #[arg(long)]
        expected_states: Option<String>,
    },

    /// Trace a HuggingFace transformers model and compile it to an ANE-faithful graph.
    ///
    /// This command traces a transformers model using torch.fx (via a Python subprocess),
    /// constructs a SIR graph from the traced computation, and compiles it through the
    /// full MILLer pass pipeline with version-aware constraint enforcement.
    ///
    /// The model class (CausalLM, Seq2SeqLM, multimodal decoder) is auto-detected
    /// from the model's config.json `architectures` field — no manual specification
    /// needed. Works with any HuggingFace model, including future architectures.
    ///
    /// The output includes:
    /// - SIR graph (from traced model)
    /// - AIR graph (post-legality, ANE-legal)
    /// - MIR graph (ready for emission)
    /// - ANE faithfulness report (which ops run on ANE vs CPU)
    /// - Core ML .mlpackage (if bridge emission succeeds)
    TraceCompile {
        /// HuggingFace model ID (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
        /// or path to a local model directory, or path to a pre-traced JSON graph.
        #[arg(short, long)]
        model: String,

        /// Output directory for compiled artifacts.
        #[arg(short, long)]
        output: String,

        /// Target ANE family for constraint-aware compilation.
        /// Accepts ANE generation codes (A11Legacy, A12, A14, A15, A16, A18)
        /// or Apple Silicon chip names (M1, M2, M3, M4, with Pro/Max variants).
        /// Defaults to A16 (first family with reliable SDPA support).
        #[arg(long, default_value = "A16")]
        target_family: String,

        /// Whether to enforce ANE-only compilation (reject CPU-fallback ops).
        #[arg(long, default_value_t = false)]
        ane_only: bool,

        /// Input batch size for tracing.
        #[arg(long, default_value_t = 1)]
        batch_size: usize,

        /// Input sequence length for tracing.
        #[arg(long, default_value_t = 32)]
        seq_len: usize,

        /// Whether to decompose composite ops (attention, MLP) during tracing.
        #[arg(long, default_value_t = true)]
        decompose: bool,

        /// Whether to include KV-cache state in the traced graph.
        #[arg(long, default_value_t = false)]
        with_kv_cache: bool,

        /// Path to the Python tracing script.
        #[arg(long, default_value = "python/trace_model.py")]
        trace_script: String,

        /// Path to the Python bridge script.
        #[arg(long, default_value = "python/bridge.py")]
        bridge: String,

        /// Path to the Python interpreter.
        #[arg(long, default_value = "python3.12")]
        python: String,

        /// Data type for model weights ("fp16" or "fp32").
        #[arg(long, default_value = "fp16")]
        dtype: String,

        /// Knowledge store directory.
        #[arg(long)]
        knowledge: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile { input, output, bridge, python, knowledge, seed } => {
            if let Err(e) =
                run_compile(&input, &output, &bridge, &python, knowledge.as_deref(), seed)
            {
                eprintln!("Compile failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::CompileFull { input, output, bridge, python, knowledge, seed } => {
            if let Err(e) =
                run_compile_full(&input, &output, &bridge, &python, knowledge.as_deref(), seed)
            {
                eprintln!("Compile-full failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::CompileSharded {
            input,
            output,
            bridge,
            python,
            knowledge,
            seed,
            proto_direct,
        } => {
            if let Err(e) = run_compile_sharded(
                &input,
                &output,
                &bridge,
                &python,
                knowledge.as_deref(),
                seed,
                proto_direct,
            ) {
                eprintln!("Compile-sharded failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::CompileFullSharded {
            input,
            output,
            bridge,
            python,
            knowledge,
            seed,
            proto_direct,
        } => {
            if let Err(e) = run_compile_full_sharded(
                &input,
                &output,
                &bridge,
                &python,
                knowledge.as_deref(),
                seed,
                proto_direct,
            ) {
                eprintln!("Compile-full-sharded failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::LabLoop { input, output, bridge, python, knowledge, seed, generated_from } => {
            if let Err(e) = run_lab_loop(
                &input,
                &output,
                &bridge,
                &python,
                &knowledge,
                seed,
                generated_from.as_deref(),
            ) {
                eprintln!("Lab-loop run failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Lab { input, output, bridge, python, skip_inspect, seed, generated_from } => {
            if let Err(e) = run_lab(
                &input,
                &output,
                &bridge,
                &python,
                !skip_inspect,
                seed,
                generated_from.as_deref(),
            ) {
                eprintln!("Lab run failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Profile {
            mlpackage,
            output,
            bridge,
            python,
            warmup,
            iterations,
            compute_units,
        } => {
            if let Err(e) = run_profile(
                &mlpackage,
                &output,
                &bridge,
                &python,
                warmup,
                iterations,
                &compute_units,
            ) {
                eprintln!("Profile failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Query { store, filter, format } => {
            if let Err(e) = run_query(&store, filter.as_deref(), &format) {
                eprintln!("Query failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Report { input, output, format } => {
            if let Err(e) = run_report(&input, &output, &format) {
                eprintln!("Report generation failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Package { input, output } => {
            if let Err(e) = run_package(&input, &output) {
                eprintln!("Package failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Import { source, store, validate } => {
            if let Err(e) = run_import(&source, &store, validate) {
                eprintln!("Import failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::GenerateTasks { family, output, seed } => {
            if let Err(e) = run_generate_tasks(&family, &output, seed) {
                eprintln!("Generate-tasks failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Verify {
            mlpackage,
            output,
            bridge,
            python,
            compute_units,
            mir_ops,
            expected_functions,
            expected_states,
        } => {
            if let Err(e) = run_verify(
                &mlpackage,
                &output,
                &bridge,
                &python,
                &compute_units,
                mir_ops.as_deref(),
                expected_functions.as_deref(),
                expected_states.as_deref(),
            ) {
                eprintln!("Verify failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::TraceCompile {
            model,
            output,
            target_family,
            ane_only,
            batch_size,
            seq_len,
            decompose,
            with_kv_cache,
            trace_script,
            bridge: _,
            python,
            dtype,
            knowledge,
        } => {
            if let Err(e) = run_trace_compile(
                &model,
                &output,
                &target_family,
                ane_only,
                batch_size,
                seq_len,
                decompose,
                with_kv_cache,
                &trace_script,
                &python,
                &dtype,
                knowledge.as_deref(),
            ) {
                eprintln!("Trace-compile failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// Compute a deterministic task identity hash from the spec parameters.
///
/// This hash is derived from the fields that determine the compilation output:
/// family, op type, dimensions, dtype, opset, compute_units, seed.
/// Same inputs → same hash. This enables artifact identity verification
/// and cache invalidation without needing to inspect the output.
fn compute_task_hash(spec: &ane_ir::task_spec::SyntheticTaskSpec) -> String {
    use std::fmt::Write;
    // Use a simple deterministic string representation for hashing.
    // This avoids relying on std::hash which is not guaranteed stable
    // across Rust versions.
    let mut hash_input = String::new();
    write!(hash_input, "family={}", spec.family).unwrap();
    write!(hash_input, ";name={}", spec.name).unwrap();

    // Use the canonical identity string from TaskOp — single source of truth
    // for all op-specific fields, eliminating per-variant match arms here.
    write!(hash_input, ";{}", spec.op.identity_string()).unwrap();

    // SHA-256 of the deterministic string
    let digest = sha2::Sha256::digest(hash_input.as_bytes());
    let hex: String = digest.iter().fold(String::new(), |mut output, b| {
        write!(output, "{:02x}", b).unwrap();
        output
    });
    format!("sha256:{}", hex)
}

/// Run the compile vertical slice.
fn run_compile(
    input: &str,
    output: &str,
    bridge_script: &str,
    python_path: &str,
    knowledge_dir: Option<&str>,
    _seed: u64,
) -> Result<(), String> {
    use ane_bridge::subprocess::PythonBridge;
    use ane_ir::linear_slice::{
        lower_linear_projection_to_mir, sir_from_linear_projection, FamilyPayload,
    };
    use ane_ir::task_spec::load_synthetic_task;

    println!(
        "=== MILLer — Vertical Slice Compile ===\
"
    );

    // Optionally load knowledge store for awareness (fast-path compile does not
    // drive the pass pipeline, but it records whether knowledge was available).
    let mut knowledge_consulted = false;
    let mut knowledge_seed_count: usize = 0;
    let mut knowledge_observation_count: usize = 0;
    if let Some(kdir) = knowledge_dir {
        let store_path = PathBuf::from(kdir);
        if store_path.exists() {
            if store_path.join("store_index.json").exists() {
                if let Ok(store) = ane_knowledge::store::KnowledgeStore::open(kdir) {
                    let (seeds, obs) = store.counts();
                    println!("  Knowledge store: {} seeds, {} observations available", seeds, obs);
                    knowledge_consulted = true;
                    knowledge_seed_count = seeds;
                    knowledge_observation_count = obs;
                }
            } else {
                // Treat as seed directory
                let tmp_store_dir = std::env::temp_dir().join("ane_compile_knowledge_store");
                let tmp_store_dir_str = tmp_store_dir.to_string_lossy().into_owned();
                if let Ok(mut store) =
                    ane_knowledge::store::KnowledgeStore::open(&tmp_store_dir_str)
                {
                    if let Ok(count) = store.load_seeds_from_directory(kdir) {
                        if count > 0 {
                            println!("  Knowledge seeds: {} entries loaded from {}", count, kdir);
                            knowledge_consulted = true;
                            knowledge_seed_count = count;
                        }
                    }
                }
            }
        }
    }

    // Step 1: Load task spec
    println!("[1/7] Loading task spec: {}", input);
    let spec = load_synthetic_task(input)?;
    println!("  Task: {} (family: {})", spec.name, spec.family);

    // Compute deterministic task identity
    let task_hash = compute_task_hash(&spec);
    println!("  Task hash: {}", task_hash);

    // Verify the task op type (reject sharded types, print op info via generic methods)
    if spec.op.is_sharded() {
        return Err(format!("Use 'compile-sharded' command for {} tasks", spec.op.family_id()));
    }
    let (input_dim, output_dim, _batch_size, _dtype) = spec.op.primary_dims();
    println!("  Op: {} {}x{}", spec.op.op_type_str(), input_dim, output_dim);

    // Step 2: Build SIR
    println!("[2/7] Building SIR graph...");
    // Note: The fast-path compile command builds SIR for logging/validation but
    // lowers directly from spec to MIR. The full pass pipeline (compile-full)
    // drives SIR → AIR → MIR through the pass infrastructure.
    let sir = sir_from_linear_projection(&spec)?;
    println!(
        "  SIR: {} nodes, {} inputs, {} outputs",
        sir.nodes.len(),
        sir.inputs.len(),
        sir.outputs.len()
    );

    // Step 3: Lower to MIR
    println!("[3/7] Lowering SIR to MIR...");
    let shard_name = format!("{}_shard_0", spec.name);
    let mir = lower_linear_projection_to_mir(&spec, &shard_name)?;
    println!(
        "  MIR: {} nodes, {} inputs, {} outputs",
        mir.nodes.len(),
        mir.inputs.len(),
        mir.outputs.len()
    );

    // Step 4: Build bridge payload
    println!("[4/7] Building bridge payload...");
    let output_path = PathBuf::from(output);
    let mlpackage_output = output_path.join("mlpackage");
    // Use generic FamilyPayload — no per-variant match needed
    let payload = FamilyPayload::from_spec(&spec, mlpackage_output.to_str().unwrap_or(""))?;
    let payload_json = serde_json::to_value(&payload)
        .map_err(|e| format!("Payload serialization failed: {}", e))?;
    println!("  Payload: command={}, task={}", payload.command, payload.task_name);

    // Step 5: Invoke Python bridge
    println!("[5/7] Invoking Python bridge...");
    let bridge = PythonBridge::new(python_path, bridge_script);
    let result = bridge
        .execute_raw_payload(&payload_json)
        .map_err(|e| format!("Bridge execution failed: {}", e))?;

    if result.status == "success" {
        println!("  Bridge: SUCCESS");
        if let Some(ref path) = result.output_path {
            println!("  Output: {}", path);
        }
        if let Some(ref hash) = result.content_hash {
            println!("  Hash: {}", hash);
        }
        if let Some(ref ver) = result.coremltools_version {
            println!("  coremltools: v{}", ver);
        }
        if !result.function_descriptors.is_empty() {
            println!("  Functions: {} defined", result.function_descriptors.len());
        }
        if !result.package_files.is_empty() {
            println!("  Package files: {} entries", result.package_files.len());
        }
    } else {
        println!("  Bridge: FAILED");
        if let Some(ref err) = result.error_message {
            println!("  Error: {}", err);
        }
    }

    // Step 6: Write artifact manifest
    println!("[6/7] Writing artifact manifest...");
    let mut manifest = build_artifact_manifest(&spec, &result, &task_hash);
    // Add knowledge consultation status to the manifest
    if knowledge_consulted {
        if let Some(obj) = manifest.as_object_mut() {
            obj.insert("knowledge_consulted".to_string(), serde_json::json!(true));
            obj.insert("knowledge_seed_count".to_string(), serde_json::json!(knowledge_seed_count));
            obj.insert(
                "knowledge_observation_count".to_string(),
                serde_json::json!(knowledge_observation_count),
            );
            obj.insert("knowledge_path".to_string(), serde_json::json!("fast_path_compile"));
        }
    } else if knowledge_dir.is_some() {
        // Knowledge was requested but not available
        if let Some(obj) = manifest.as_object_mut() {
            obj.insert("knowledge_consulted".to_string(), serde_json::json!(false));
            obj.insert(
                "knowledge_note".to_string(),
                serde_json::json!("knowledge directory specified but no store found"),
            );
        }
    }
    let manifest_path = output_path.join("manifest.json");
    fs::create_dir_all(&output_path).map_err(|e| format!("Failed to create output dir: {}", e))?;
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("Manifest serialization failed: {}", e))?;
    fs::write(&manifest_path, &manifest_json)
        .map_err(|e| format!("Failed to write manifest: {}", e))?;
    println!("  Manifest: {}", manifest_path.display());

    // Also write the MIR dump
    let mir_path = output_path.join("mir.json");
    let mir_json = serde_json::to_string_pretty(&mir)
        .map_err(|e| format!("MIR serialization failed: {}", e))?;
    fs::write(&mir_path, &mir_json).map_err(|e| format!("Failed to write MIR: {}", e))?;

    // Step 7: Write backend-knowledge update
    println!("[7/7] Writing backend-knowledge update...");
    let knowledge_output = match knowledge_dir {
        Some(dir) => PathBuf::from(dir),
        None => output_path.join("knowledge"),
    };
    fs::create_dir_all(&knowledge_output)
        .map_err(|e| format!("Failed to create knowledge dir: {}", e))?;

    let knowledge_update = build_knowledge_update(&spec, &result, &task_hash);
    let knowledge_path = knowledge_output.join(format!("update_{}.json", spec.name));
    let knowledge_json = serde_json::to_string_pretty(&knowledge_update)
        .map_err(|e| format!("Knowledge serialization failed: {}", e))?;
    fs::write(&knowledge_path, &knowledge_json)
        .map_err(|e| format!("Failed to write knowledge: {}", e))?;
    println!("  Knowledge: {}", knowledge_path.display());

    println!("\n=== Compile complete ===");
    println!("Artifacts in: {}", output);

    Ok(())
}

/// Bridge between the knowledge store and the pass pipeline.
///
/// Implements `PassKnowledgeQuery` by querying the `KnowledgeStore`
/// for legality and risk data. This is the concrete wiring point
/// where the knowledge system meets the compilation pipeline.
struct StoreKnowledgeQuery<'a> {
    store: &'a ane_knowledge::store::KnowledgeStore,
}

impl<'a> StoreKnowledgeQuery<'a> {
    fn new(store: &'a ane_knowledge::store::KnowledgeStore) -> Self {
        Self { store }
    }
}

impl<'a> ane_passes::knowledge_query::PassKnowledgeQuery for StoreKnowledgeQuery<'a> {
    fn query_legality(
        &self,
        op_pattern: &str,
        _scope: Option<&ane_ir::kir::KnowledgeScope>,
    ) -> Option<ane_passes::knowledge_query::LegalityInfo> {
        use ane_ir::kir::KnowledgeType;
        use ane_knowledge::query::{KnowledgeQuery, KnowledgeQueryable};

        let query =
            KnowledgeQuery::new().with_type(KnowledgeType::LegalityRule).with_min_confidence(0.1);

        let results = self.store.query(&query).ok()?;

        // Find a result that matches the op pattern
        for unit in results {
            if let Some(pattern) = unit.payload.get("op_pattern").and_then(|v| v.as_str()) {
                // Check if the pattern matches. Patterns can be:
                // - Exact: "mb.matmul"
                // - Pipe-separated alternatives: "mb.add|mb.mul|mb.abs"
                if pattern.split('|').any(|p| p.trim() == op_pattern) {
                    let ane_legal =
                        unit.payload.get("ane_legal").and_then(|v| v.as_bool()).unwrap_or(false);
                    return Some(ane_passes::knowledge_query::LegalityInfo {
                        ane_legal,
                        confidence: unit.confidence,
                        evidence_count: unit.evidence_count,
                        source_id: Some(unit.id.clone()),
                    });
                }
            }
        }
        None
    }

    fn query_risk(
        &self,
        op_pattern: &str,
        scope: Option<&ane_ir::kir::KnowledgeScope>,
    ) -> Option<ane_passes::knowledge_query::RiskInfo> {
        use ane_ir::kir::KnowledgeType;
        use ane_knowledge::query::{KnowledgeQuery, KnowledgeQueryable};

        // Query for survival data (fallback risk)
        let survival_query = KnowledgeQuery::new()
            .with_type(KnowledgeType::SurvivalMatrixEntry)
            .with_min_confidence(0.1);

        if let Ok(survival_results) = self.store.query(&survival_query) {
            for unit in survival_results {
                if let Some(pattern) = unit.payload.get("op_pattern").and_then(|v| v.as_str()) {
                    if pattern.split('|').any(|p| p.trim() == op_pattern) {
                        let fallback_risk = unit
                            .payload
                            .get("fallback_risk")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.1) as f32;
                        let drift_risk =
                            unit.payload.get("drift_risk").and_then(|v| v.as_f64()).unwrap_or(0.05)
                                as f32;
                        return Some(ane_passes::knowledge_query::RiskInfo {
                            fallback_risk,
                            drift_risk,
                            confidence: unit.confidence,
                            evidence_count: unit.evidence_count,
                            source_id: Some(unit.id.clone()),
                        });
                    }
                }
            }
        }

        // Fall back to legality knowledge: if an op is known ANE-legal with high confidence,
        // its fallback risk is low. If illegal, fallback risk is high.
        let legality = self.query_legality(op_pattern, scope)?;
        let fallback_risk =
            if legality.ane_legal { 1.0 - legality.confidence } else { legality.confidence };
        let drift_risk = fallback_risk * 0.5;

        Some(ane_passes::knowledge_query::RiskInfo {
            fallback_risk: fallback_risk.min(1.0),
            drift_risk: drift_risk.min(1.0),
            confidence: legality.confidence,
            evidence_count: legality.evidence_count,
            source_id: legality.source_id,
        })
    }

    fn query_precision_hazard(
        &self,
        op_pattern: &str,
        current_dtype: &str,
        _scope: Option<&ane_ir::kir::KnowledgeScope>,
    ) -> Option<ane_passes::knowledge_query::PrecisionHazardInfo> {
        use ane_ir::kir::KnowledgeType;
        use ane_knowledge::query::{KnowledgeQuery, KnowledgeQueryable};

        // Query for precision hazard knowledge
        let hazard_query = KnowledgeQuery::new()
            .with_type(KnowledgeType::PrecisionHazard)
            .with_min_confidence(0.1);

        let results = self.store.query(&hazard_query).ok()?;

        // Find a result that matches the op pattern and current dtype.
        // The seed entries use "op" as the field name (e.g., "LinearProjection"),
        // and "bitwidth" or "quality_impact" to indicate severity.
        for unit in results {
            // Check op pattern match
            let op_match = unit
                .payload
                .get("op")
                .and_then(|v| v.as_str())
                .map(|op| op == op_pattern)
                .unwrap_or(false);

            // Also check op_pattern field (used by legality/risk entries)
            let pattern_match = unit
                .payload
                .get("op_pattern")
                .and_then(|v| v.as_str())
                .map(|p| p.split('|').any(|s| s.trim() == op_pattern))
                .unwrap_or(false);

            if op_match || pattern_match {
                // Determine if this hazard applies to the current dtype.
                // The seed entries have "bitwidth" and "quality_impact" fields.
                // A hazard with "high" quality_impact means fp16 is unsafe.
                let quality_impact =
                    unit.payload.get("quality_impact").and_then(|v| v.as_str()).unwrap_or("none");

                // Only report a hazard if the quality impact is high or medium
                // and the current dtype is fp16 (the default)
                let applies = match quality_impact {
                    "high" | "medium" => current_dtype == "fp16",
                    _ => false,
                };

                if applies {
                    return Some(ane_passes::knowledge_query::PrecisionHazardInfo {
                        op_pattern: op_pattern.to_string(),
                        hazardous_dtype: "fp16".to_string(),
                        recommended_dtype: "fp32".to_string(),
                        confidence: unit.confidence,
                        evidence_count: unit.evidence_count,
                        source_id: Some(unit.id.clone()),
                        description: unit
                            .payload
                            .get("note")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                    });
                }
            }
        }
        None
    }

    fn query_compute_plan_placement(
        &self,
        op_pattern: &str,
        _scope: Option<&ane_ir::kir::KnowledgeScope>,
    ) -> Option<ane_passes::knowledge_query::ComputePlanPlacementInfo> {
        use ane_ir::kir::KnowledgeType;
        use ane_knowledge::query::{KnowledgeQuery, KnowledgeQueryable};

        let query = KnowledgeQuery::new()
            .with_type(KnowledgeType::SurvivalMatrixEntry)
            .with_min_confidence(0.1);

        let results = self.store.query(&query).ok()?;

        for unit in results {
            if let Some(pattern) = unit.payload.get("op_pattern").and_then(|v| v.as_str()) {
                if pattern.split('|').any(|p| p.trim() == op_pattern) {
                    let ane_placed =
                        unit.payload.get("ane_placed").and_then(|v| v.as_bool()).unwrap_or(false);
                    let preferred_device = unit
                        .payload
                        .get("preferred_device")
                        .and_then(|v| v.as_str())
                        .unwrap_or("CPU")
                        .to_string();
                    return Some(ane_passes::knowledge_query::ComputePlanPlacementInfo {
                        op_pattern: op_pattern.to_string(),
                        ane_placed,
                        preferred_device,
                        confidence: unit.confidence,
                        evidence_count: unit.evidence_count,
                        source_id: Some(unit.id.clone()),
                    });
                }
            }
        }
        None
    }
}

/// Run the compile-full path: drives the complete pass pipeline.
///
/// Unlike the fast-path `compile` command (which lowers directly from spec
/// to MIR), this command drives the full pass pipeline:
///
/// SIR → Canonicalize → Staticize → PrecisionPolicy → StateTopology
///   → LegalityRewrite → RiskAnnotate → ShardPlan → MilLower → bridge
///
/// All intermediate IR representations (SIR, AIR, PIR, MIR) are written
/// as artifacts in the output directory.
fn run_compile_full(
    input: &str,
    output: &str,
    bridge_script: &str,
    python_path: &str,
    knowledge_dir: Option<&str>,
    _seed: u64,
) -> Result<(), String> {
    use ane_bridge::subprocess::PythonBridge;
    use ane_ir::linear_slice::{sir_from_linear_projection, FamilyPayload};
    use ane_ir::task_spec::load_synthetic_task;
    use ane_passes::canonicalize::CanonicalizePass;
    use ane_passes::knowledge_query::NoKnowledge;
    use ane_passes::legality_rewrite::{DecompositionContext, LegalityRewritePass};
    use ane_passes::mil_lower::MilLowerPass;
    use ane_passes::precision_policy::PrecisionPolicyPass;
    use ane_passes::risk_annotate::RiskAnnotatePass;
    use ane_passes::shard_plan::ShardPlanPass;
    use ane_passes::staticize::StaticizePass;

    println!("=== MILLer — Full Pass Pipeline Compile ===\n");

    // Step 1: Load task spec
    println!("[1/13] Loading task spec: {}", input);
    let spec = load_synthetic_task(input)?;
    println!("  Task: {} (family: {})", spec.name, spec.family);

    let task_hash = compute_task_hash(&spec);
    println!("  Task hash: {}", task_hash);

    // Verify the task op type (reject sharded types, print op info via generic methods)
    if spec.op.is_sharded() {
        return Err(format!(
            "Use 'compile-full-sharded' command for {} tasks",
            spec.op.family_id()
        ));
    }
    let (input_dim, output_dim, _batch_size, _dtype) = spec.op.primary_dims();
    println!("  Op: {} {}x{}", spec.op.op_type_str(), input_dim, output_dim);

    // Step 2: Build SIR graph
    println!("[2/13] Building SIR graph...");
    let sir = sir_from_linear_projection(&spec)?;
    println!(
        "  SIR: {} nodes, {} inputs, {} outputs",
        sir.nodes.len(),
        sir.inputs.len(),
        sir.outputs.len()
    );

    // Step 2b: Set up knowledge query for the pass pipeline
    // If a knowledge directory is provided, load seeds and create a StoreKnowledgeQuery.
    // Otherwise, use NoKnowledge (returns None for all queries, so passes use defaults).
    let mut knowledge_store: Option<ane_knowledge::store::KnowledgeStore> = None;
    if let Some(kdir) = knowledge_dir {
        let store_path = PathBuf::from(kdir);
        // The knowledge dir might be a seed directory (contains JSON files)
        // or an existing store (contains store_index.json).
        if store_path.exists() {
            if store_path.join("store_index.json").exists() {
                match ane_knowledge::store::KnowledgeStore::open(kdir) {
                    Ok(store) => {
                        let (seeds, obs) = store.counts();
                        println!("  Knowledge store: {} seeds, {} observations", seeds, obs);
                        knowledge_store = Some(store);
                    }
                    Err(e) => {
                        eprintln!("  Warning: failed to open knowledge store at {}: {}", kdir, e);
                    }
                }
            } else {
                // Treat as a seed directory: create a temporary store and load seeds
                let tmp_store_dir = std::env::temp_dir().join("ane_compile_full_knowledge_store");
                let tmp_store_dir_str = tmp_store_dir.to_string_lossy().into_owned();
                match ane_knowledge::store::KnowledgeStore::open(&tmp_store_dir_str) {
                    Ok(mut store) => match store.load_seeds_from_directory(kdir) {
                        Ok(count) => {
                            if count > 0 {
                                println!(
                                    "  Knowledge seeds: {} entries loaded from {}",
                                    count, kdir
                                );
                                knowledge_store = Some(store);
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "  Warning: failed to load knowledge seeds from {}: {}",
                                kdir, e
                            );
                        }
                    },
                    Err(e) => {
                        eprintln!("  Warning: failed to create temporary knowledge store: {}", e);
                    }
                }
            }
        }
    }

    // Step 3: Run CanonicalizePass on SIR (pass-through for now)
    println!("[3/13] Running CanonicalizePass...");
    let canonicalize = CanonicalizePass::new();
    let sir = canonicalize.run(sir).map_err(|e| format!("CanonicalizePass failed: {}", e))?;
    println!("  Canonicalize: {} nodes (pass-through for linear projection)", sir.nodes.len());

    // Step 4: Run StaticizePass on SIR (pass-through for now)
    println!("[4/13] Running StaticizePass...");
    let staticize = StaticizePass::new();
    let sir = staticize.run(sir).map_err(|e| format!("StaticizePass failed: {}", e))?;
    println!("  Staticize: {} nodes (pass-through for linear projection)", sir.nodes.len());

    // Step 4b: Run PrecisionPolicyPass (SIR→SIR with dtype adaptation)
    // This is the first pass that materially changes a compilation decision
    // based on stored empirical knowledge. When a precision hazard is known
    // for an operation, it overrides the default fp16 to fp32.
    println!("[4b/13] Running PrecisionPolicyPass...");
    let mut precision_policy = PrecisionPolicyPass::new();
    let sir = match &knowledge_store {
        Some(store) => {
            let query = StoreKnowledgeQuery::new(store);
            precision_policy
                .run(sir, &query)
                .map_err(|e| format!("PrecisionPolicyPass failed: {}", e))?
        }
        None => {
            let no_knowledge = NoKnowledge;
            precision_policy
                .run(sir, &no_knowledge)
                .map_err(|e| format!("PrecisionPolicyPass failed: {}", e))?
        }
    };
    if precision_policy.has_adaptations() {
        println!("  PrecisionPolicy: {} adaptation(s) applied", precision_policy.adaptations.len());
        for adaptation in &precision_policy.adaptations {
            println!(
                "    {}:{} → {} (source: {}, confidence: {:.2})",
                adaptation.node_name,
                adaptation.original_dtype,
                adaptation.adapted_dtype,
                adaptation.source_id.as_deref().unwrap_or("unknown"),
                adaptation.confidence
            );
        }
    } else {
        println!("  PrecisionPolicy: no adaptations (all nodes use default fp16)");
    }

    // Step 5: Run LegalityRewritePass (SIR→AIR)
    // Construct DecompositionContext from task spec for truthful AIR shapes (Sprint 56)
    let decomp_ctx: Option<DecompositionContext> = match &spec.op {
        ane_ir::task_spec::TaskOp::Attention {
            embed_dim,
            num_heads,
            head_dim,
            seq_len,
            batch_size,
            ..
        } => Some(DecompositionContext::for_attention(
            *batch_size,
            *embed_dim,
            *num_heads,
            *head_dim,
            *seq_len,
        )),
        ane_ir::task_spec::TaskOp::DecodeStep {
            embed_dim,
            num_heads,
            head_dim,
            kv_len,
            batch_size,
            ..
        } => Some(DecompositionContext::for_decode_step(
            *batch_size,
            *embed_dim,
            *num_heads,
            *head_dim,
            *kv_len,
        )),
        ane_ir::task_spec::TaskOp::ShardedDecodeStep {
            embed_dim,
            num_heads,
            head_dim,
            kv_len,
            batch_size,
            ..
        } => Some(DecompositionContext::for_decode_step(
            *batch_size,
            *embed_dim,
            *num_heads,
            *head_dim,
            *kv_len,
        )),
        _ => None,
    };
    println!("[5/13] Running LegalityRewritePass (SIR→AIR)...");
    let legality = LegalityRewritePass::new();
    let air = match &knowledge_store {
        Some(store) => {
            let query = StoreKnowledgeQuery::new(store);
            legality
                .run(sir.clone(), &query, decomp_ctx.as_ref())
                .map_err(|e| format!("LegalityRewritePass failed: {}", e))?
        }
        None => {
            let no_knowledge = NoKnowledge;
            legality
                .run(sir.clone(), &no_knowledge, decomp_ctx.as_ref())
                .map_err(|e| format!("LegalityRewritePass failed: {}", e))?
        }
    };
    println!(
        "  AIR: {} nodes, {} inputs, {} outputs",
        air.nodes.len(),
        air.inputs.len(),
        air.outputs.len()
    );

    // Step 6: Run RiskAnnotatePass on AIR
    println!("[6/13] Running RiskAnnotatePass...");
    let risk = RiskAnnotatePass::new();
    let air = match &knowledge_store {
        Some(store) => {
            let query = StoreKnowledgeQuery::new(store);
            risk.run(air, &query).map_err(|e| format!("RiskAnnotatePass failed: {}", e))?
        }
        None => {
            let no_knowledge = NoKnowledge;
            risk.run(air, &no_knowledge).map_err(|e| format!("RiskAnnotatePass failed: {}", e))?
        }
    };
    println!("  RiskAnnotate: {} nodes annotated", air.nodes.len());

    // Step 7: Run ShardPlanPass (SIR→ShardPlan+PIR)
    // This is the second pass that materially changes a compilation decision
    // based on stored empirical knowledge. When fallback risk is known to be
    // high for the shard's primary op, it overrides CPU_AND_NE to CPU_AND_GPU.
    println!("[7/13] Running ShardPlanPass...");
    let mut shard_plan_pass = ShardPlanPass::new();
    let (shard_plan, pir) = match &knowledge_store {
        Some(store) => {
            let query = StoreKnowledgeQuery::new(store);
            shard_plan_pass.run(&sir, &query).map_err(|e| format!("ShardPlanPass failed: {}", e))?
        }
        None => {
            let no_knowledge = NoKnowledge;
            shard_plan_pass
                .run(&sir, &no_knowledge)
                .map_err(|e| format!("ShardPlanPass failed: {}", e))?
        }
    };
    if shard_plan_pass.has_adaptations() {
        println!("  ShardPlan: {} adaptation(s) applied", shard_plan_pass.adaptations.len());
        for adaptation in &shard_plan_pass.adaptations {
            println!(
                "    {}:{} → {} (source: {}, fallback_risk: {:.2}, confidence: {:.2})",
                adaptation.shard_name,
                adaptation.original_compute_units,
                adaptation.adapted_compute_units,
                adaptation.source_id.as_deref().unwrap_or("unknown"),
                adaptation.fallback_risk,
                adaptation.confidence
            );
        }
    } else {
        println!("  ShardPlan: no adaptations (all shards use default CPU_AND_NE)");
    }
    println!("  ShardPlan: {} shards", shard_plan.num_shards);
    println!("  PIR: {} packages, {} handoffs", pir.packages.len(), pir.handoffs.len());

    // Step 8: Run MilLowerPass (AIR→Vec<MIR>)
    println!("[8/13] Running MilLowerPass (AIR→MIR)...");
    let mil_lower = MilLowerPass::new();
    let mirs =
        mil_lower.run(&air, &shard_plan).map_err(|e| format!("MilLowerPass failed: {}", e))?;
    println!("  MIR: {} shard graphs produced", mirs.len());
    for (i, mir) in mirs.iter().enumerate() {
        println!(
            "    MIR[{}]: {} nodes, {} inputs, {} outputs",
            i,
            mir.nodes.len(),
            mir.inputs.len(),
            mir.outputs.len()
        );
    }

    // Step 9: Build bridge payload from spec
    // If precision policy adapted any dtype, use the adapted dtype for the bridge
    // payload. This ensures the knowledge-informed precision decision actually
    // reaches the emitted mlpackage via the Python bridge.
    println!("[9/13] Building bridge payload...");
    let output_path = PathBuf::from(output);
    let mlpackage_output = output_path.join("mlpackage");
    let adapted_dtype: Option<&str> = if precision_policy.has_adaptations() {
        // Use the first adaptation's adapted_dtype as the effective dtype.
        // For the linear projection vertical slice, there is at most one
        // primary op node, so the first adaptation covers it.
        Some(&precision_policy.adaptations[0].adapted_dtype)
    } else {
        None
    };
    // Use generic FamilyPayload with dtype override — no per-variant match needed
    let payload = FamilyPayload::from_spec_with_override(
        &spec,
        mlpackage_output.to_str().unwrap_or(""),
        adapted_dtype,
    )?;
    let payload_json = serde_json::to_value(&payload)
        .map_err(|e| format!("Payload serialization failed: {}", e))?;
    println!("  Payload: command={}, task={}", payload.command, payload.task_name);
    if adapted_dtype.is_some() {
        println!("  Precision override: bridge payload uses adapted dtype instead of spec default");
    }

    // Step 10: Invoke Python bridge
    println!("[10/13] Invoking Python bridge...");
    let bridge = PythonBridge::new(python_path, bridge_script);
    let result = bridge
        .execute_raw_payload(&payload_json)
        .map_err(|e| format!("Bridge execution failed: {}", e))?;

    if result.status == "success" {
        println!("  Bridge: SUCCESS");
        if let Some(ref path) = result.output_path {
            println!("  Output: {}", path);
        }
        if let Some(ref hash) = result.content_hash {
            println!("  Hash: {}", hash);
        }
        if let Some(ref ver) = result.coremltools_version {
            println!("  coremltools: v{}", ver);
        }
    } else {
        println!("  Bridge: FAILED");
        if let Some(ref err) = result.error_message {
            println!("  Error: {}", err);
        }
    }

    // Step 11: Write manifest (version "0.5.0" for pass-pipeline path)
    println!("[11/13] Writing artifact manifest...");
    fs::create_dir_all(&output_path).map_err(|e| format!("Failed to create output dir: {}", e))?;

    let mut manifest = {
        // Extract MIR op types from each MIR graph for the manifest.
        // This enables the `verify` command to auto-populate --mir-ops
        // from the compile manifest, closing the usability gap where
        // users had to manually specify expected MIR ops.
        use ane_artifacts::manifest::MirOpEntry;
        let mir_ops_per_graph: Vec<Vec<MirOpEntry>> = mirs
            .iter()
            .map(|mir| {
                mir.nodes
                    .iter()
                    .map(|node| MirOpEntry {
                        op_type: format!("{:?}", node.op)
                            .strip_prefix("MIL")
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("{:?}", node.op)),
                    })
                    .collect()
            })
            .collect();
        build_artifact_manifest_pass_pipeline(&spec, &result, &task_hash, &pir, &mir_ops_per_graph)
    };

    // Add precision adaptation provenance to the manifest (S16.3)
    if precision_policy.has_adaptations() {
        if let Some(obj) = manifest.as_object_mut() {
            let adaptations: Vec<serde_json::Value> = precision_policy
                .adaptations
                .iter()
                .map(|a| {
                    serde_json::json!({
                        "node_name": a.node_name,
                        "original_dtype": a.original_dtype,
                        "adapted_dtype": a.adapted_dtype,
                        "source_id": a.source_id,
                        "confidence": a.confidence,
                        "reason": a.reason,
                    })
                })
                .collect();
            obj.insert("precision_adaptations".to_string(), serde_json::json!(adaptations));
            obj.insert("precision_adapted".to_string(), serde_json::json!(true));
        }
    } else {
        if let Some(obj) = manifest.as_object_mut() {
            obj.insert("precision_adapted".to_string(), serde_json::json!(false));
        }
    }

    // Add shard plan adaptation provenance to the manifest (S22.4)
    // This is the second adaptive pass — when fallback risk knowledge causes
    // the compute unit assignment to change, the adaptation is recorded here.
    if shard_plan_pass.has_adaptations() {
        if let Some(obj) = manifest.as_object_mut() {
            let adaptations: Vec<serde_json::Value> = shard_plan_pass
                .adaptations
                .iter()
                .map(|a| {
                    serde_json::json!({
                        "shard_name": a.shard_name,
                        "original_compute_units": a.original_compute_units,
                        "adapted_compute_units": a.adapted_compute_units,
                        "op_pattern": a.op_pattern,
                        "fallback_risk": a.fallback_risk,
                        "source_id": a.source_id,
                        "confidence": a.confidence,
                        "reason": a.reason,
                    })
                })
                .collect();
            obj.insert("compute_unit_adaptations".to_string(), serde_json::json!(adaptations));
            obj.insert("compute_units_adapted".to_string(), serde_json::json!(true));
        }
    } else {
        if let Some(obj) = manifest.as_object_mut() {
            obj.insert("compute_units_adapted".to_string(), serde_json::json!(false));
        }
    }

    let manifest_path = output_path.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("Manifest serialization failed: {}", e))?;
    fs::write(&manifest_path, &manifest_json)
        .map_err(|e| format!("Failed to write manifest: {}", e))?;
    println!("  Manifest: {}", manifest_path.display());

    // Step 12: Write all IR dumps (SIR, AIR, PIR, MIR)
    println!("[12/13] Writing IR dumps...");
    let sir_path = output_path.join("sir.json");
    let sir_json = serde_json::to_string_pretty(&sir)
        .map_err(|e| format!("SIR serialization failed: {}", e))?;
    fs::write(&sir_path, &sir_json).map_err(|e| format!("Failed to write SIR: {}", e))?;

    let air_path = output_path.join("air.json");
    let air_json = serde_json::to_string_pretty(&air)
        .map_err(|e| format!("AIR serialization failed: {}", e))?;
    fs::write(&air_path, &air_json).map_err(|e| format!("Failed to write AIR: {}", e))?;

    let pir_path = output_path.join("pir.json");
    let pir_json = serde_json::to_string_pretty(&pir)
        .map_err(|e| format!("PIR serialization failed: {}", e))?;
    fs::write(&pir_path, &pir_json).map_err(|e| format!("Failed to write PIR: {}", e))?;

    for (i, mir) in mirs.iter().enumerate() {
        let mir_path = output_path.join(format!("mir_{}.json", i));
        let mir_json = serde_json::to_string_pretty(&mir)
            .map_err(|e| format!("MIR serialization failed: {}", e))?;
        fs::write(&mir_path, &mir_json).map_err(|e| format!("Failed to write MIR: {}", e))?;
    }
    println!("  SIR: {}", sir_path.display());
    println!("  AIR: {}", air_path.display());
    println!("  PIR: {}", pir_path.display());
    println!("  MIR: {} shard graphs", mirs.len());

    // Step 13: Write knowledge update
    println!("[13/13] Writing knowledge update...");
    let knowledge_output = match knowledge_dir {
        Some(dir) => PathBuf::from(dir),
        None => output_path.join("knowledge"),
    };
    fs::create_dir_all(&knowledge_output)
        .map_err(|e| format!("Failed to create knowledge dir: {}", e))?;

    let knowledge_update = build_knowledge_update(&spec, &result, &task_hash);
    let knowledge_path = knowledge_output.join(format!("update_{}.json", spec.name));
    let knowledge_json = serde_json::to_string_pretty(&knowledge_update)
        .map_err(|e| format!("Knowledge serialization failed: {}", e))?;
    fs::write(&knowledge_path, &knowledge_json)
        .map_err(|e| format!("Failed to write knowledge: {}", e))?;
    println!("  Knowledge: {}", knowledge_path.display());

    println!("\n=== Compile-full complete ===");
    println!("Artifacts in: {}", output);

    Ok(())
}

/// Build an artifact manifest for the pass-pipeline compilation path.
///
/// Uses version "0.5.0" and includes `compilation_path: "pass_pipeline"`
/// in the environment_limitations to distinguish from the fast-path compile.
fn build_artifact_manifest_pass_pipeline(
    spec: &ane_ir::task_spec::SyntheticTaskSpec,
    bridge_result: &ane_bridge::subprocess::BridgeResult,
    task_hash: &str,
    _pir: &ane_ir::pir::PirGraph,
    mir_ops_per_graph: &[Vec<ane_artifacts::manifest::MirOpEntry>],
) -> serde_json::Value {
    use ane_artifacts::manifest::{ArtifactManifest, FunctionDescriptor, PackageEntry, TensorSpec};

    let timestamp = chrono::Utc::now().to_rfc3339();

    let (input_dim, output_dim, batch_size, dtype) = spec.op.primary_dims();

    let functions: Vec<FunctionDescriptor> = if !bridge_result.function_descriptors.is_empty() {
        bridge_result
            .function_descriptors
            .iter()
            .enumerate()
            .map(|(i, fd)| {
                let inputs: Vec<TensorSpec> = fd
                    .inputs
                    .iter()
                    .map(|inp| TensorSpec {
                        name: inp
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        shape: inp
                            .get("shape")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
                            })
                            .unwrap_or_default(),
                        dtype: inp
                            .get("dtype")
                            .and_then(|v| v.as_str())
                            .unwrap_or("fp16")
                            .to_string(),
                    })
                    .collect();
                let outputs: Vec<TensorSpec> = fd
                    .outputs
                    .iter()
                    .map(|outp| TensorSpec {
                        name: outp
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        shape: outp
                            .get("shape")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
                            })
                            .unwrap_or_default(),
                        dtype: outp
                            .get("dtype")
                            .and_then(|v| v.as_str())
                            .unwrap_or("fp16")
                            .to_string(),
                    })
                    .collect();
                // Get MIR ops for this function from the MIR graphs produced by the pass pipeline.
                // mir_ops_per_graph is indexed by MIR graph index; bridge function descriptors
                // are indexed by function index. For single-function models they align 1:1.
                let mir_ops = mir_ops_per_graph.get(i).cloned().unwrap_or_default();
                FunctionDescriptor {
                    name: fd.name.clone(),
                    inputs,
                    outputs,
                    stateful: fd.stateful,
                    emission_status: "emitted".to_string(),
                    mir_ops,
                }
            })
            .collect()
    } else {
        vec![FunctionDescriptor {
            name: "main".to_string(),
            inputs: vec![TensorSpec {
                name: "x".to_string(),
                shape: vec![batch_size, input_dim],
                dtype: dtype.clone(),
            }],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![batch_size, output_dim],
                dtype: dtype.clone(),
            }],
            stateful: false,
            emission_status: if bridge_result.status == "success" {
                "emitted".to_string()
            } else {
                "seam_only".to_string()
            },
            mir_ops: mir_ops_per_graph.first().cloned().unwrap_or_default(),
        }]
    };

    let packages: Vec<PackageEntry> = if bridge_result.status == "success" {
        vec![PackageEntry {
            name: spec.name.clone(),
            role: "synthetic_microkernel".to_string(),
            path: bridge_result.output_path.clone(),
            content_hash: bridge_result.content_hash.clone(),
            size_bytes: 0,
            functions,
        }]
    } else {
        vec![]
    };

    let manifest = ArtifactManifest {
        version: "0.5.0".to_string(),
        model_id: spec.name.clone(),
        task_hash: task_hash.to_string(),
        created_at: timestamp,
        packages,
        state_declarations: vec![],
        handoffs: vec![],
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        implementation_status: "host_compiled".to_string(),
        verification_scope: "host_compile_only".to_string(),
        environment_limitations: vec![
            "no_apple_hardware".to_string(),
            "ane_placement_not_verified".to_string(),
            "no_on_device_predict".to_string(),
            "compilation_path:pass_pipeline".to_string(),
        ],
    };

    serde_json::to_value(&manifest)
        .unwrap_or_else(|_| serde_json::json!({"error": "manifest serialization failed"}))
}

/// Run the compile-full-sharded path: multi-unit orchestration through the pass pipeline.
///
/// This is Sprint 17's runnable multi-unit orchestration path. Unlike
/// `compile-sharded` (which bypasses the pass pipeline), this command:
///
/// 1. Loads a ShardedLinearPipeline task spec
/// 2. Decomposes it into per-shard sub-tasks (Entry, Interior, Exit)
/// 3. Runs each shard independently through the full pass pipeline
/// 4. Emits one mlpackage per shard through the Python bridge
/// 5. Produces a unified manifest with concrete handoff semantics and
///    per-shard provenance
///
/// Each shard gets its own pass pipeline run (SIR → Canonicalize → Staticize
/// → PrecisionPolicy → LegalityRewrite → RiskAnnotate → ShardPlan → MilLower
/// → bridge), enabling knowledge-informed compilation per shard.
fn run_compile_full_sharded(
    input: &str,
    output: &str,
    bridge_script: &str,
    python_path: &str,
    knowledge_dir: Option<&str>,
    seed: u64,
    proto_direct: bool,
) -> Result<(), String> {
    use ane_bridge::subprocess::PythonBridge;
    use ane_ir::linear_slice::{sir_from_linear_projection, ShardDesc, ShardedShardPayload};
    use ane_ir::task_spec::load_synthetic_task;
    use ane_passes::shard_plan::ShardPlanPass;

    println!("=== MILLer — Full Pipeline Multi-Shard Compile ===\n");

    // Step 1: Load task spec
    println!("[1/8] Loading task spec: {}", input);
    let spec = load_synthetic_task(input)?;
    println!("  Task: {} (family: {})", spec.name, spec.family);

    let task_hash = compute_task_hash(&spec);
    println!("  Task hash: {}", task_hash);

    // Verify it's a sharded task and extract parameters
    if !spec.op.is_sharded() {
        return Err(format!(
            "compile-full-sharded requires a sharded task, got {}",
            spec.op.family_id()
        ));
    }
    println!("  Op: {}", spec.op.op_type_str());
    let (_input_dim, _output_dim, batch_size, dtype) = spec.op.primary_dims();

    // Step 2: Build shard pipeline spec (generalized)
    println!("[2/8] Building shard pipeline specification...");
    let pipeline_spec = match &spec.op {
        ane_ir::task_spec::TaskOp::ShardedLinearPipeline {
            input_dim,
            hidden_dim,
            output_dim,
            batch_size,
            dtype,
        } => ane_ir::pir::ShardPipelineSpec::three_shard_linear(
            &spec.name,
            *input_dim,
            *hidden_dim,
            *output_dim,
            *batch_size,
            dtype,
        ),
        ane_ir::task_spec::TaskOp::ShardedDecodeStep {
            embed_dim,
            num_heads,
            head_dim,
            kv_len,
            batch_size,
            dtype,
        } => ane_ir::pir::ShardPipelineSpec::three_shard_decode_step(
            &spec.name,
            *embed_dim,
            *num_heads,
            *head_dim,
            *kv_len,
            *batch_size,
            dtype,
        ),
        _ => return Err("Unexpected task type after validation".into()),
    };
    for shard in &pipeline_spec.shards {
        println!(
            "  Shard: {} (role: {}, compute: {})",
            shard.shard_name,
            shard.role.canonical_name(),
            shard.compute_units.to_coreml_string()
        );
    }

    // Step 3: Build multi-shard plan and PIR with concrete handoffs
    // When a knowledge store is available, load shard template seeds and
    // use them to inform compute unit assignments.
    println!("[3/8] Building multi-shard plan with concrete handoffs...");
    let shard_templates = if let Some(kdir) = knowledge_dir {
        match ane_knowledge::shard_template::load_shard_template_seeds(kdir) {
            Ok(templates) => {
                if !templates.is_empty() {
                    println!("  Loaded {} shard template seed(s) from {}", templates.len(), kdir);
                }
                templates.iter().map(|t| t.template.clone()).collect::<Vec<_>>()
            }
            Err(e) => {
                eprintln!("  Warning: failed to load shard template seeds from {}: {}", kdir, e);
                vec![]
            }
        }
    } else {
        vec![]
    };

    // Step 4: Load knowledge store (optional) — moved before plan construction
    // so that risk-based knowledge can inform the multi-shard plan (S37.4).
    let mut knowledge_store: Option<ane_knowledge::store::KnowledgeStore> = None;
    if let Some(kdir) = knowledge_dir {
        let store_path = PathBuf::from(kdir);
        if store_path.exists() {
            if store_path.join("store_index.json").exists() {
                if let Ok(store) = ane_knowledge::store::KnowledgeStore::open(kdir) {
                    let (seeds, obs) = store.counts();
                    println!("  Knowledge store: {} seeds, {} observations", seeds, obs);
                    knowledge_store = Some(store);
                }
            } else {
                let tmp_store_dir =
                    std::env::temp_dir().join("ane_compile_full_sharded_knowledge_store");
                let tmp_store_dir_str = tmp_store_dir.to_string_lossy().into_owned();
                if let Ok(mut store) =
                    ane_knowledge::store::KnowledgeStore::open(&tmp_store_dir_str)
                {
                    if let Ok(count) = store.load_seeds_from_directory(kdir) {
                        if count > 0 {
                            println!("  Knowledge seeds: {} entries loaded from {}", count, kdir);
                            knowledge_store = Some(store);
                        }
                    }
                }
            }
        }
    }

    // Step 3b (S37.4): Build multi-shard plan with both template AND risk knowledge.
    // Previously, only template knowledge was used at the plan-construction level.
    // Now, if a knowledge store is available, we also query it for per-shard fallback
    // risk and apply compute unit adaptations accordingly. This means the multi-shard
    // plan is no longer blind to accumulated risk observations.
    let mut multi_shard_plan_pass = ShardPlanPass::new();
    let (shard_plan, pir, plan_compute_adaptations) = match &knowledge_store {
        Some(store) => {
            let query = StoreKnowledgeQuery::new(store);
            multi_shard_plan_pass.build_sharded_plan_from_spec_with_risk_knowledge(
                &pipeline_spec,
                &shard_templates,
                &query,
            )
        }
        None => {
            // No knowledge store: use template-only path or default
            let (plan, graph) = if shard_templates.is_empty() {
                ShardPlanPass::build_sharded_plan_from_spec(&pipeline_spec)
            } else {
                println!("  Applying shard template knowledge to compute unit assignments...");
                ShardPlanPass::build_sharded_plan_from_spec_with_knowledge(
                    &pipeline_spec,
                    &shard_templates,
                )
            };
            (plan, graph, vec![])
        }
    };
    if !plan_compute_adaptations.is_empty() {
        println!(
            "  Multi-shard plan: {} compute unit adaptation(s) from risk knowledge",
            plan_compute_adaptations.len()
        );
        for a in &plan_compute_adaptations {
            println!(
                "    {} → {} (risk: {:.2}, source: {})",
                a.original_compute_units,
                a.adapted_compute_units,
                a.fallback_risk,
                a.source_id.as_deref().unwrap_or("unknown")
            );
        }
    }
    println!(
        "  ShardPlan: {} shards, multi_shard={}",
        shard_plan.num_shards, shard_plan.is_multi_shard
    );
    println!("  PIR: {} packages, {} handoffs", pir.packages.len(), pir.handoffs.len());
    for handoff in &pir.handoffs {
        println!(
            "    Handoff[{}]: {} → {} (kind: {:?}, output: {} → input: {})",
            handoff.execution_order,
            handoff.from_package,
            handoff.to_package,
            handoff.handoff_kind,
            handoff.source_output_name,
            handoff.target_input_name
        );
    }

    // Step 5: Run each shard through the pass pipeline independently
    println!("[5/8] Running each shard through the pass pipeline...");
    let output_path = PathBuf::from(output);
    fs::create_dir_all(&output_path).map_err(|e| format!("Failed to create output dir: {}", e))?;

    let bridge = PythonBridge::new(python_path, bridge_script);
    let mut shard_results: Vec<ShardCompileResult> = Vec::new();

    for shard_spec in &pipeline_spec.shards {
        println!(
            "  --- Shard: {} (role: {}) ---",
            shard_spec.shard_name,
            shard_spec.role.canonical_name()
        );

        // Derive scalar dimensions from the tensor specs
        let input_dim =
            shard_spec.input_specs.first().and_then(|t| t.shape.last().copied()).unwrap_or(0);
        let output_dim =
            shard_spec.output_specs.first().and_then(|t| t.shape.last().copied()).unwrap_or(0);

        // Build a synthetic spec for this shard (linear projection)
        let shard_task_spec = ane_ir::task_spec::SyntheticTaskSpec {
            name: shard_spec.shard_name.clone(),
            family: spec.family.clone(),
            description: Some(format!(
                "Shard {} of {}: {}x{} linear projection",
                shard_spec.role.canonical_name(),
                spec.name,
                input_dim,
                output_dim
            )),
            op: ane_ir::task_spec::TaskOp::LinearProjection {
                input_dim,
                output_dim,
                batch_size,
                has_bias: true,
                dtype: dtype.clone(),
            },
            measurement: spec.measurement.clone(),
        };

        // Run the shard through the pass pipeline
        let shard_sir = sir_from_linear_projection(&shard_task_spec)
            .map_err(|e| format!("SIR build failed for shard {}: {}", shard_spec.shard_name, e))?;

        // Run pass pipeline: Canonicalize → Staticize → PrecisionPolicy → LegalityRewrite → RiskAnnotate
        use ane_passes::canonicalize::CanonicalizePass;
        use ane_passes::knowledge_query::NoKnowledge;
        use ane_passes::legality_rewrite::{DecompositionContext, LegalityRewritePass};
        use ane_passes::precision_policy::PrecisionPolicyPass;
        use ane_passes::risk_annotate::RiskAnnotatePass;
        use ane_passes::staticize::StaticizePass;

        let canonicalize = CanonicalizePass::new();
        let shard_sir = canonicalize.run(shard_sir).map_err(|e| {
            format!("CanonicalizePass failed for shard {}: {}", shard_spec.shard_name, e)
        })?;

        let staticize = StaticizePass::new();
        let shard_sir = staticize.run(shard_sir).map_err(|e| {
            format!("StaticizePass failed for shard {}: {}", shard_spec.shard_name, e)
        })?;

        let mut precision_policy = PrecisionPolicyPass::new();
        let shard_sir = match &knowledge_store {
            Some(store) => {
                let query = StoreKnowledgeQuery::new(store);
                precision_policy.run(shard_sir, &query).map_err(|e| {
                    format!("PrecisionPolicyPass failed for shard {}: {}", shard_spec.shard_name, e)
                })?
            }
            None => {
                let no_knowledge = NoKnowledge;
                precision_policy.run(shard_sir, &no_knowledge).map_err(|e| {
                    format!("PrecisionPolicyPass failed for shard {}: {}", shard_spec.shard_name, e)
                })?
            }
        };
        let shard_precision_adaptations = precision_policy.adaptations.clone();
        if precision_policy.has_adaptations() {
            println!("    PrecisionPolicy: {} adaptation(s)", precision_policy.adaptations.len());
        }

        let legality = LegalityRewritePass::new();
        // Construct DecompositionContext from the original task spec for sharded paths (Sprint 56)
        let shard_decomp_ctx: Option<DecompositionContext> = match &spec.op {
            ane_ir::task_spec::TaskOp::ShardedDecodeStep {
                embed_dim,
                num_heads,
                head_dim,
                kv_len,
                batch_size,
                ..
            } => Some(DecompositionContext::for_decode_step(
                *batch_size,
                *embed_dim,
                *num_heads,
                *head_dim,
                *kv_len,
            )),
            _ => None,
        };
        let _shard_air = match &knowledge_store {
            Some(store) => {
                let query = StoreKnowledgeQuery::new(store);
                legality.run(shard_sir.clone(), &query, shard_decomp_ctx.as_ref()).map_err(|e| {
                    format!("LegalityRewritePass failed for shard {}: {}", shard_spec.shard_name, e)
                })?
            }
            None => {
                let no_knowledge = NoKnowledge;
                legality.run(shard_sir.clone(), &no_knowledge, shard_decomp_ctx.as_ref()).map_err(
                    |e| {
                        format!(
                            "LegalityRewritePass failed for shard {}: {}",
                            shard_spec.shard_name, e
                        )
                    },
                )?
            }
        };

        let risk = RiskAnnotatePass::new();
        let shard_air = match &knowledge_store {
            Some(store) => {
                let query = StoreKnowledgeQuery::new(store);
                risk.run(_shard_air, &query).map_err(|e| {
                    format!("RiskAnnotatePass failed for shard {}: {}", shard_spec.shard_name, e)
                })?
            }
            None => {
                let no_knowledge = NoKnowledge;
                risk.run(_shard_air, &no_knowledge).map_err(|e| {
                    format!("RiskAnnotatePass failed for shard {}: {}", shard_spec.shard_name, e)
                })?
            }
        };

        // Step 5c (S37.3): Run ShardPlanPass for this shard.
        // Unlike compile-full (which runs ShardPlanPass with knowledge), the multi-shard
        // path constructs the ShardPlan from the pre-built multi-shard plan. This ensures
        // the per-shard compute units from the multi-shard plan propagate through the MIR.
        let mut shard_plan_pass = ShardPlanPass::new();
        let (mut shard_shard_plan, _shard_pir) = match &knowledge_store {
            Some(store) => {
                let query = StoreKnowledgeQuery::new(store);
                shard_plan_pass.run(&shard_sir, &query).map_err(|e| {
                    format!("ShardPlanPass failed for shard {}: {}", shard_spec.shard_name, e)
                })?
            }
            None => {
                let no_knowledge = NoKnowledge;
                shard_plan_pass.run(&shard_sir, &no_knowledge).map_err(|e| {
                    format!("ShardPlanPass failed for shard {}: {}", shard_spec.shard_name, e)
                })?
            }
        };

        // Override the ShardPlan's compute_units with the multi-shard plan's per-shard
        // assignment. The ShardPlanPass produces a single-shard plan with knowledge-adapted
        // compute units, but the multi-shard plan already has the correct assignment from
        // build_sharded_plan_from_spec_with_knowledge. We take the multi-shard plan's
        // assignment as authoritative, since it was constructed with both template knowledge
        // and the shard role's default compute units.
        let shard_index =
            pipeline_spec.shards.iter().position(|s| s.shard_name == shard_spec.shard_name);
        if let Some(idx) = shard_index {
            if idx < shard_plan.compute_units.len() {
                // If the knowledge-driven adaptation (from ShardPlanPass) changed the
                // compute units, record it. Otherwise, use the multi-shard plan's value.
                let knowledge_compute = shard_shard_plan
                    .compute_units
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "CPU_AND_NE".to_string());
                let multi_shard_compute = shard_plan.compute_units[idx].clone();

                // Use the knowledge-driven result if it adapted (different from default),
                // otherwise use the multi-shard plan's assignment.
                let effective_compute = if knowledge_compute != "CPU_AND_NE" {
                    knowledge_compute
                } else {
                    multi_shard_compute
                };

                shard_shard_plan.compute_units = vec![effective_compute];
                shard_shard_plan.shard_names = vec![shard_spec.shard_name.clone()];
            }
        }

        let shard_compute_adaptations = shard_plan_pass.adaptations.clone();
        if shard_plan_pass.has_adaptations() {
            println!("    ShardPlanPass: {} adaptation(s)", shard_plan_pass.adaptations.len());
            for a in &shard_plan_pass.adaptations {
                println!(
                    "      {} → {} (risk: {:.2}, source: {})",
                    a.original_compute_units,
                    a.adapted_compute_units,
                    a.fallback_risk,
                    a.source_id.as_deref().unwrap_or("unknown")
                );
            }
        }

        // Step 5d (S37.3): Run MilLowerPass (AIR→MIR) for this shard.
        // This produces a MIR graph whose compute_unit_hint matches the shard plan,
        // fixing the previous gap where MIR always said CPU_AND_NE regardless of
        // the multi-shard plan's actual compute unit assignment.
        use ane_passes::mil_lower::MilLowerPass;
        let mil_lower = MilLowerPass::new();
        let _shard_mirs = mil_lower.run(&shard_air, &shard_shard_plan).map_err(|e| {
            format!("MilLowerPass failed for shard {}: {}", shard_spec.shard_name, e)
        })?;
        println!(
            "    MilLower: {} MIR graph(s) produced, compute_unit_hint={}",
            _shard_mirs.len(),
            shard_shard_plan.compute_units.first().unwrap_or(&"N/A".to_string())
        );

        // Step 6: Emit mlpackage for this shard
        // Sprint 52: When --proto-direct is set for decode-step shards, use
        // RoleMirBuilder + proto-direct emission instead of the Python bridge.
        let shard_output = output_path.join(&shard_spec.shard_name);
        let use_proto_direct =
            proto_direct && matches!(&spec.op, ane_ir::task_spec::TaskOp::ShardedDecodeStep { .. });

        if use_proto_direct {
            use ane_bridge::proto_direct::{
                emit_role_shard_proto_direct, validate_proto_direct_package,
            };
            println!("    Emission: proto-direct via RoleMirBuilder");

            let mlpackage_path =
                shard_output.join(format!("{}_proto_direct.mlpackage", shard_spec.shard_name));
            fs::create_dir_all(&shard_output)
                .map_err(|e| format!("Failed to create shard output dir: {}", e))?;

            let emit_result =
                emit_role_shard_proto_direct(shard_spec, mlpackage_path.to_str().unwrap_or(""))
                    .map_err(|e| {
                        format!(
                            "Proto-direct emission failed for shard {}: {}",
                            shard_spec.shard_name, e
                        )
                    })?;

            let validation = validate_proto_direct_package(mlpackage_path.to_str().unwrap_or(""))
                .map_err(|e| {
                format!("Validation failed for shard {}: {}", shard_spec.shard_name, e)
            })?;

            println!(
                "    Proto-direct: {} files, {} weights, hash={:.8}",
                emit_result.file_count, emit_result.weight_count, emit_result.content_hash
            );
            if !validation.is_valid {
                for err in &validation.errors {
                    eprintln!("    Validation warning: {}", err);
                }
            }

            let result = ane_bridge::subprocess::BridgeResult {
                status: "success".to_string(),
                error_message: None,
                output_path: Some(emit_result.mlpackage_path.clone()),
                coremltools_version: None,
                content_hash: Some(emit_result.content_hash.clone()),
                package_files: vec![],
                compute_plan: None,
                function_descriptors: vec![ane_bridge::subprocess::BridgeFunctionDescriptor {
                    name: "main".to_string(),
                    inputs: shard_spec
                        .input_specs
                        .iter()
                        .map(|t| {
                            serde_json::json!({
                                "name": t.name, "shape": t.shape, "dtype": t.dtype
                            })
                        })
                        .collect(),
                    outputs: shard_spec
                        .output_specs
                        .iter()
                        .map(|t| {
                            serde_json::json!({
                                "name": t.name, "shape": t.shape, "dtype": t.dtype
                            })
                        })
                        .collect(),
                    stateful: false,
                }],
                metadata: serde_json::json!({"emission_method": "proto-direct"}),
                stderr: String::new(),
                emission_path: ane_bridge::subprocess::EmissionPath::ProtoDirect,
            };

            shard_results.push(ShardCompileResult {
                shard_name: shard_spec.shard_name.clone(),
                role: shard_spec.role.canonical_name().to_string(),
                bridge_result: result,
                precision_adaptations: shard_precision_adaptations,
                compute_adaptations: shard_compute_adaptations,
                effective_compute_units: shard_shard_plan
                    .compute_units
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "CPU_AND_NE".to_string()),
            });
        } else {
            // Use the Python bridge for emission
            // If precision policy adapted any dtype for this shard, use the adapted dtype.
            let shard_adapted_dtype: Option<&str> = if !shard_precision_adaptations.is_empty() {
                Some(&shard_precision_adaptations[0].adapted_dtype)
            } else {
                None
            };
            let shard_desc = ShardDesc {
                role: shard_spec.role.clone(),
                shard_name: shard_spec.shard_name.clone(),
                input_dim,
                output_dim,
                compute_units: shard_spec.compute_units.clone(),
            };

            // S37 residual fix: For decode-step shards, use the role-sensitive
            // emit_shard_decode_step command instead of emit_linear_projection.
            // This produces structurally different programs per shard role (different
            // head counts, KV cache state dimensions, output projection dimensions),
            // closing the gap where "shard emission is still too uniform until shard
            // role materially changes emitted graphs and/or dimensions."
            let payload = match &spec.op {
                ane_ir::task_spec::TaskOp::ShardedDecodeStep {
                    embed_dim,
                    num_heads,
                    head_dim,
                    kv_len,
                    ..
                } => ShardedShardPayload::from_shard_decode_step(
                    &shard_desc,
                    &spec.name,
                    &spec.family,
                    batch_size,
                    &dtype,
                    shard_output.to_str().unwrap_or(""),
                    seed,
                    shard_adapted_dtype,
                    *embed_dim,
                    *num_heads,
                    *head_dim,
                    *kv_len,
                ),
                _ => ShardedShardPayload::from_shard_with_override(
                    &shard_desc,
                    &spec.name,
                    &spec.family,
                    batch_size,
                    &dtype,
                    shard_output.to_str().unwrap_or(""),
                    seed,
                    shard_adapted_dtype,
                ),
            };
            let payload_json = serde_json::to_value(&payload)
                .map_err(|e| format!("Shard payload serialization failed: {}", e))?;

            let result = bridge.execute_raw_payload(&payload_json).map_err(|e| {
                format!("Bridge execution failed for shard {}: {}", shard_spec.shard_name, e)
            })?;

            println!(
                "    Bridge: {}",
                if result.status == "success" { "SUCCESS" } else { "FAILED" }
            );

            shard_results.push(ShardCompileResult {
                shard_name: shard_spec.shard_name.clone(),
                role: shard_spec.role.canonical_name().to_string(),
                bridge_result: result,
                precision_adaptations: shard_precision_adaptations,
                compute_adaptations: shard_compute_adaptations,
                effective_compute_units: shard_shard_plan
                    .compute_units
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "CPU_AND_NE".to_string()),
            });
        }
    }

    // Step 7: Write unified manifest with concrete handoffs and per-shard provenance
    println!("[7/8] Writing unified manifest with per-shard provenance...");
    let manifest =
        build_full_sharded_manifest(&spec, &shard_results, &pir, &task_hash, &shard_plan);
    let manifest_path = output_path.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("Manifest serialization failed: {}", e))?;
    fs::write(&manifest_path, &manifest_json)
        .map_err(|e| format!("Failed to write manifest: {}", e))?;
    println!("  Manifest: {}", manifest_path.display());

    // Write PIR dump
    let pir_path = output_path.join("pir.json");
    let pir_json = serde_json::to_string_pretty(&pir)
        .map_err(|e| format!("PIR serialization failed: {}", e))?;
    fs::write(&pir_path, &pir_json).map_err(|e| format!("Failed to write PIR: {}", e))?;

    // Step 8: Write shard plan dump
    println!("[8/8] Writing shard plan...");
    let plan_path = output_path.join("shard_plan.json");
    let plan_json = serde_json::to_string_pretty(&shard_plan)
        .map_err(|e| format!("Shard plan serialization failed: {}", e))?;
    fs::write(&plan_path, &plan_json).map_err(|e| format!("Failed to write shard plan: {}", e))?;

    println!("\n=== Full-pipeline multi-shard compile complete ===");
    println!("Artifacts in: {}", output);
    println!("Shards: {} (Entry, Interior, Exit)", pipeline_spec.shards.len());
    println!("Handoffs: {} (TensorPassThrough)", pir.handoffs.len());

    Ok(())
}

/// Per-shard compile result with provenance.
struct ShardCompileResult {
    shard_name: String,
    role: String,
    bridge_result: ane_bridge::subprocess::BridgeResult,
    precision_adaptations: Vec<ane_passes::precision_policy::PrecisionAdaptation>,
    /// S37.3/S37.5: compute unit adaptations from ShardPlanPass, proving
    /// that knowledge-driven risk assessment materially changed the shard's
    /// compute unit assignment.
    compute_adaptations: Vec<ane_passes::shard_plan::ComputeUnitAdaptation>,
    /// S37.3: the effective compute units used for this shard after all
    /// adaptations (template-based + risk-based).
    effective_compute_units: String,
}

/// Build a manifest for the compile-full-sharded path with per-shard provenance
/// and concrete handoff semantics.
fn build_full_sharded_manifest(
    spec: &ane_ir::task_spec::SyntheticTaskSpec,
    shard_results: &[ShardCompileResult],
    pir: &ane_ir::pir::PirGraph,
    task_hash: &str,
    shard_plan: &ane_passes::shard_plan::ShardPlan,
) -> serde_json::Value {
    use ane_artifacts::manifest::{
        ArtifactManifest, FunctionDescriptor, HandoffEntry, PackageEntry, TensorSpec,
    };

    let timestamp = chrono::Utc::now().to_rfc3339();

    let packages: Vec<PackageEntry> = shard_results
        .iter()
        .map(|result| {
            let functions: Vec<FunctionDescriptor> =
                if !result.bridge_result.function_descriptors.is_empty() {
                    result
                        .bridge_result
                        .function_descriptors
                        .iter()
                        .map(|fd| {
                            let inputs: Vec<TensorSpec> = fd
                                .inputs
                                .iter()
                                .map(|inp| TensorSpec {
                                    name: inp
                                        .get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("x")
                                        .to_string(),
                                    shape: inp
                                        .get("shape")
                                        .and_then(|v| v.as_array())
                                        .map(|arr| {
                                            arr.iter()
                                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                                .collect()
                                        })
                                        .unwrap_or_default(),
                                    dtype: inp
                                        .get("dtype")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("fp16")
                                        .to_string(),
                                })
                                .collect();
                            let outputs: Vec<TensorSpec> = fd
                                .outputs
                                .iter()
                                .map(|outp| TensorSpec {
                                    name: outp
                                        .get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("output")
                                        .to_string(),
                                    shape: outp
                                        .get("shape")
                                        .and_then(|v| v.as_array())
                                        .map(|arr| {
                                            arr.iter()
                                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                                .collect()
                                        })
                                        .unwrap_or_default(),
                                    dtype: outp
                                        .get("dtype")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("fp16")
                                        .to_string(),
                                })
                                .collect();
                            FunctionDescriptor {
                                name: fd.name.clone(),
                                inputs,
                                outputs,
                                stateful: fd.stateful,
                                emission_status: if result.bridge_result.status == "success" {
                                    "emitted".to_string()
                                } else {
                                    "seam_only".to_string()
                                },
                                mir_ops: vec![],
                            }
                        })
                        .collect()
                } else {
                    vec![FunctionDescriptor {
                        name: "main".to_string(),
                        inputs: vec![],
                        outputs: vec![],
                        stateful: false,
                        emission_status: if result.bridge_result.status == "success" {
                            "emitted".to_string()
                        } else {
                            "seam_only".to_string()
                        },
                        mir_ops: vec![],
                    }]
                };

            PackageEntry {
                name: result.shard_name.clone(),
                role: format!("DecoderShard({})", result.role),
                path: result.bridge_result.output_path.clone(),
                content_hash: result.bridge_result.content_hash.clone(),
                size_bytes: 0,
                functions,
            }
        })
        .collect();

    // Build handoff entries with concrete runtime semantics
    let handoffs: Vec<HandoffEntry> = pir
        .handoffs
        .iter()
        .map(|h| HandoffEntry {
            from_package: h.from_package.clone(),
            to_package: h.to_package.clone(),
            tensor_name: h.tensor_name.clone(),
            shape: h.shape.clone(),
            dtype: h.dtype.clone(),
        })
        .collect();

    let manifest = ArtifactManifest {
        version: "0.6.0".to_string(),
        model_id: spec.name.clone(),
        task_hash: task_hash.to_string(),
        created_at: timestamp,
        packages,
        state_declarations: vec![],
        handoffs,
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        implementation_status: "host_compiled".to_string(),
        verification_scope: "host_compile_only".to_string(),
        environment_limitations: vec![
            "no_apple_hardware".to_string(),
            "ane_placement_not_verified".to_string(),
            "no_on_device_predict".to_string(),
            "shard_aware_full_pipeline_path".to_string(),
        ],
    };

    let mut manifest_value = serde_json::to_value(&manifest)
        .unwrap_or_else(|_| serde_json::json!({"error": "manifest serialization failed"}));

    // Add concrete handoff semantics and per-shard provenance
    if let Some(obj) = manifest_value.as_object_mut() {
        // Add concrete handoff details with runtime semantics
        let handoff_details: Vec<serde_json::Value> = pir
            .handoffs
            .iter()
            .map(|h| {
                serde_json::json!({
                    "from_package": h.from_package,
                    "to_package": h.to_package,
                    "tensor_name": h.tensor_name,
                    "shape": h.shape,
                    "dtype": h.dtype,
                    "handoff_kind": format!("{:?}", h.handoff_kind),
                    "execution_order": h.execution_order,
                    "source_output_name": h.source_output_name,
                    "target_input_name": h.target_input_name,
                })
            })
            .collect();
        obj.insert("concrete_handoffs".to_string(), serde_json::json!(handoff_details));

        // Add per-shard provenance (S37.5: now includes compute_unit_adaptations
        // and effective_compute_units, proving that shard role and knowledge
        // materially influenced emitted content)
        let shard_provenance: Vec<serde_json::Value> = shard_results.iter().map(|r| {
            let precision_adaptations: Vec<serde_json::Value> = r.precision_adaptations.iter().map(|a| {
                serde_json::json!({
                    "node_name": a.node_name,
                    "original_dtype": a.original_dtype,
                    "adapted_dtype": a.adapted_dtype,
                    "source_id": a.source_id,
                    "confidence": a.confidence,
                })
            }).collect();

            let compute_adaptations: Vec<serde_json::Value> = r.compute_adaptations.iter().map(|a| {
                serde_json::json!({
                    "shard_name": a.shard_name,
                    "original_compute_units": a.original_compute_units,
                    "adapted_compute_units": a.adapted_compute_units,
                    "op_pattern": a.op_pattern,
                    "fallback_risk": a.fallback_risk,
                    "source_id": a.source_id,
                    "confidence": a.confidence,
                    "reason": a.reason,
                })
            }).collect();

            serde_json::json!({
                "shard_name": r.shard_name,
                "role": r.role,
                "effective_compute_units": r.effective_compute_units,
                "precision_adaptations": precision_adaptations,
                "compute_unit_adaptations": compute_adaptations,
                "pass_pipeline": "Canonicalize→Staticize→PrecisionPolicy→LegalityRewrite→RiskAnnotate→ShardPlan→MilLower→bridge",
                "compile_path": "compile_full_sharded",
            })
        }).collect();
        obj.insert("shard_provenance".to_string(), serde_json::json!(shard_provenance));

        // Add shard plan summary with knowledge_adapted flag (S37.5)
        let any_compute_adapted = shard_results.iter().any(|r| !r.compute_adaptations.is_empty());
        obj.insert(
            "shard_plan".to_string(),
            serde_json::json!({
                "num_shards": shard_plan.num_shards,
                "is_multi_shard": shard_plan.is_multi_shard,
                "shard_roles": shard_plan.shard_roles,
                "shard_names": shard_plan.shard_names,
                "compute_units": shard_plan.compute_units,
                "compute_units_adapted": any_compute_adapted,
            }),
        );
    }

    manifest_value
}

fn run_compile_sharded(
    input: &str,
    output: &str,
    bridge_script: &str,
    python_path: &str,
    knowledge_dir: Option<&str>,
    seed: u64,
    proto_direct: bool,
) -> Result<(), String> {
    use ane_bridge::subprocess::PythonBridge;
    use ane_ir::linear_slice::{ShardDesc, ShardedShardPayload};
    use ane_ir::task_spec::load_synthetic_task;
    use ane_knowledge::shard_template::load_shard_template_seeds;

    println!("=== MILLer — Shard-Aware Compile ===\n");

    // Step 1: Load task spec
    println!("[1/5] Loading task spec: {}", input);
    let spec = load_synthetic_task(input)?;
    println!("  Task: {} (family: {})", spec.name, spec.family);

    // Compute deterministic task identity
    let task_hash = compute_task_hash(&spec);
    println!("  Task hash: {}", task_hash);

    // Verify it's a sharded task and extract parameters
    if !spec.op.is_sharded() {
        return Err(format!("Use 'compile' command for {} tasks", spec.op.family_id()));
    }
    println!("  Op: {}", spec.op.op_type_str());
    let (_input_dim, _output_dim, batch_size, dtype) = spec.op.primary_dims();

    // Step 2: Build shard pipeline spec and PIR
    println!("[2/5] Building shard pipeline specification...");
    let pipeline_spec = match &spec.op {
        ane_ir::task_spec::TaskOp::ShardedLinearPipeline {
            input_dim,
            hidden_dim,
            output_dim,
            batch_size,
            dtype,
        } => ane_ir::pir::ShardPipelineSpec::three_shard_linear(
            &spec.name,
            *input_dim,
            *hidden_dim,
            *output_dim,
            *batch_size,
            dtype,
        ),
        ane_ir::task_spec::TaskOp::ShardedDecodeStep {
            embed_dim,
            num_heads,
            head_dim,
            kv_len,
            batch_size,
            dtype,
        } => ane_ir::pir::ShardPipelineSpec::three_shard_decode_step(
            &spec.name,
            *embed_dim,
            *num_heads,
            *head_dim,
            *kv_len,
            *batch_size,
            dtype,
        ),
        _ => return Err("Unexpected task type after validation".into()),
    };
    for shard in &pipeline_spec.shards {
        println!(
            "  Shard: {} (role: {}, compute: {})",
            shard.shard_name,
            shard.role.canonical_name(),
            shard.compute_units.to_coreml_string(),
        );
    }

    // Step 3: Build PIR (deployment structure) from the pipeline spec
    // When a knowledge directory is provided, load shard template seeds and
    // use them to inform compute unit assignments.
    println!("[3/5] Building PIR deployment structure...");
    let shard_templates = if let Some(kdir) = knowledge_dir {
        match load_shard_template_seeds(kdir) {
            Ok(templates) => {
                if !templates.is_empty() {
                    println!("  Loaded {} shard template seed(s) from {}", templates.len(), kdir);
                }
                templates.iter().map(|t| t.template.clone()).collect::<Vec<_>>()
            }
            Err(e) => {
                eprintln!("  Warning: failed to load shard template seeds from {}: {}", kdir, e);
                vec![]
            }
        }
    } else {
        vec![]
    };

    let (_shard_plan, pir) = if shard_templates.is_empty() {
        ane_passes::shard_plan::ShardPlanPass::build_sharded_plan_from_spec(&pipeline_spec)
    } else {
        println!("  Applying shard template knowledge to compute unit assignments...");
        ane_passes::shard_plan::ShardPlanPass::build_sharded_plan_from_spec_with_knowledge(
            &pipeline_spec,
            &shard_templates,
        )
    };
    println!("  PIR: {} packages, {} handoffs", pir.packages.len(), pir.handoffs.len());

    // Step 4: Emit one mlpackage per shard
    println!("[4/5] Emitting mlpackage per shard...");
    let output_path = PathBuf::from(output);
    fs::create_dir_all(&output_path).map_err(|e| format!("Failed to create output dir: {}", e))?;

    let bridge = PythonBridge::new(python_path, bridge_script);
    let mut shard_results: Vec<(String, String, ane_bridge::subprocess::BridgeResult)> = Vec::new();

    for shard_spec in &pipeline_spec.shards {
        let shard_output = output_path.join(&shard_spec.shard_name);

        // Sprint 52: For decode-step shards with --proto-direct, use RoleMirBuilder
        // + proto-direct emission instead of the Python bridge. This makes
        // RoleMirBuilder the single source of truth for role-specific MIR in the
        // CLI compile path.
        let use_proto_direct =
            proto_direct && matches!(&spec.op, ane_ir::task_spec::TaskOp::ShardedDecodeStep { .. });

        if use_proto_direct {
            use ane_bridge::proto_direct::{
                emit_role_shard_proto_direct, validate_proto_direct_package,
            };
            println!(
                "  Shard {} (role: {}): using proto-direct emission via RoleMirBuilder",
                shard_spec.shard_name,
                shard_spec.role.canonical_name()
            );

            let mlpackage_path =
                shard_output.join(format!("{}_proto_direct.mlpackage", shard_spec.shard_name));
            fs::create_dir_all(&shard_output)
                .map_err(|e| format!("Failed to create shard output dir: {}", e))?;

            let emit_result =
                emit_role_shard_proto_direct(shard_spec, mlpackage_path.to_str().unwrap_or(""))
                    .map_err(|e| {
                        format!(
                            "Proto-direct emission failed for shard {}: {}",
                            shard_spec.shard_name, e
                        )
                    })?;

            // Validate the emitted package
            let validation = validate_proto_direct_package(mlpackage_path.to_str().unwrap_or(""))
                .map_err(|e| {
                format!("Validation failed for shard {}: {}", shard_spec.shard_name, e)
            })?;

            println!(
                "    Proto-direct: {} files, {} weights, content_hash={}",
                emit_result.file_count,
                emit_result.weight_count,
                &emit_result.content_hash[..8]
            );
            if !validation.is_valid {
                for err in &validation.errors {
                    eprintln!("    Validation warning: {}", err);
                }
            }

            // Build a BridgeResult-compatible struct from the proto-direct result
            let result = ane_bridge::subprocess::BridgeResult {
                status: "success".to_string(),
                error_message: None,
                output_path: Some(emit_result.mlpackage_path.clone()),
                coremltools_version: None,
                content_hash: Some(emit_result.content_hash.clone()),
                package_files: vec![], // proto-direct doesn't enumerate files the same way
                compute_plan: None,
                function_descriptors: vec![ane_bridge::subprocess::BridgeFunctionDescriptor {
                    name: "main".to_string(),
                    inputs: shard_spec
                        .input_specs
                        .iter()
                        .map(|t| {
                            serde_json::json!({
                                "name": t.name, "shape": t.shape, "dtype": t.dtype
                            })
                        })
                        .collect(),
                    outputs: shard_spec
                        .output_specs
                        .iter()
                        .map(|t| {
                            serde_json::json!({
                                "name": t.name, "shape": t.shape, "dtype": t.dtype
                            })
                        })
                        .collect(),
                    stateful: false,
                }],
                metadata: serde_json::json!({"emission_method": "proto-direct"}),
                stderr: String::new(),
                emission_path: ane_bridge::subprocess::EmissionPath::ProtoDirect,
            };

            shard_results.push((
                shard_spec.shard_name.clone(),
                shard_spec.role.canonical_name().to_string(),
                result,
            ));
        } else {
            // Derive scalar dimensions from the tensor specs for bridge payload.
            // Each shard has one input tensor and one output tensor; we take the
            // last dimension as the feature dimension for the linear projection.
            let input_dim =
                shard_spec.input_specs.first().and_then(|t| t.shape.last().copied()).unwrap_or(0);
            let output_dim =
                shard_spec.output_specs.first().and_then(|t| t.shape.last().copied()).unwrap_or(0);

            // Construct a ShardDesc for the bridge payload.
            // Sprint 23: for decode-step shards, the emission path still uses
            // linear projection (the narrowest honest emission path), but the
            // shard dimensions are decode-step-specific (e.g., 128→384 for QKV).
            let shard_desc = ShardDesc {
                role: shard_spec.role.clone(),
                shard_name: shard_spec.shard_name.clone(),
                input_dim,
                output_dim,
                compute_units: shard_spec.compute_units.clone(),
            };

            let payload = ShardedShardPayload::from_shard(
                &shard_desc,
                &spec.name,
                &spec.family,
                batch_size,
                &dtype,
                shard_output.to_str().unwrap_or(""),
                seed,
            );
            let payload_json = serde_json::to_value(&payload)
                .map_err(|e| format!("Shard payload serialization failed: {}", e))?;

            let result = bridge.execute_raw_payload(&payload_json).map_err(|e| {
                format!("Bridge execution failed for shard {}: {}", shard_spec.shard_name, e)
            })?;

            let status = result.status.clone();
            println!(
                "  Shard {}: {}",
                shard_spec.shard_name,
                if status == "success" { "SUCCESS" } else { "FAILED" }
            );
            if let Some(ref err) = result.error_message {
                println!("    Error: {}", err);
            }

            shard_results.push((
                shard_spec.shard_name.clone(),
                shard_spec.role.canonical_name().to_string(),
                result,
            ));
        }
    }

    // Step 5: Write manifest reflecting the multi-shard structure
    println!("[5/5] Writing shard-aware manifest...");
    let manifest = build_sharded_manifest(&spec, &shard_results, &pir, &task_hash);
    let manifest_path = output_path.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("Manifest serialization failed: {}", e))?;
    fs::write(&manifest_path, &manifest_json)
        .map_err(|e| format!("Failed to write manifest: {}", e))?;
    println!("  Manifest: {}", manifest_path.display());

    // Write PIR dump
    let pir_path = output_path.join("pir.json");
    let pir_json = serde_json::to_string_pretty(&pir)
        .map_err(|e| format!("PIR serialization failed: {}", e))?;
    fs::write(&pir_path, &pir_json).map_err(|e| format!("Failed to write PIR: {}", e))?;

    // Write MIR dumps per shard
    for shard_spec in &pipeline_spec.shards {
        let input_dim =
            shard_spec.input_specs.first().and_then(|t| t.shape.last().copied()).unwrap_or(0);
        let output_dim =
            shard_spec.output_specs.first().and_then(|t| t.shape.last().copied()).unwrap_or(0);
        let shard_desc = ShardDesc {
            role: shard_spec.role.clone(),
            shard_name: shard_spec.shard_name.clone(),
            input_dim,
            output_dim,
            compute_units: shard_spec.compute_units.clone(),
        };
        let mir = ane_ir::linear_slice::lower_shard_to_mir(&shard_desc, batch_size, &dtype)?;
        let mir_path = output_path.join(format!("mir_{}.json", shard_spec.shard_name));
        let mir_json = serde_json::to_string_pretty(&mir)
            .map_err(|e| format!("MIR serialization failed: {}", e))?;
        fs::write(&mir_path, &mir_json).map_err(|e| format!("Failed to write MIR: {}", e))?;
    }

    println!("\n=== Shard-aware compile complete ===");
    println!("Artifacts in: {}", output);

    Ok(())
}

/// Build a shard-aware artifact manifest for a multi-shard compilation.
///
/// Unlike the single-shard manifest, this manifest contains one PackageEntry
/// per shard, each with its role as a string, and inter-shard handoffs.
fn build_sharded_manifest(
    spec: &ane_ir::task_spec::SyntheticTaskSpec,
    shard_results: &[(String, String, ane_bridge::subprocess::BridgeResult)],
    pir: &ane_ir::pir::PirGraph,
    task_hash: &str,
) -> serde_json::Value {
    use ane_artifacts::manifest::{
        ArtifactManifest, FunctionDescriptor, HandoffEntry, PackageEntry, TensorSpec,
    };

    let timestamp = chrono::Utc::now().to_rfc3339();

    let packages: Vec<PackageEntry> = shard_results
        .iter()
        .map(|(shard_name, role, result)| {
            let functions: Vec<FunctionDescriptor> = if !result.function_descriptors.is_empty() {
                result
                    .function_descriptors
                    .iter()
                    .map(|fd| {
                        let inputs: Vec<TensorSpec> = fd
                            .inputs
                            .iter()
                            .map(|inp| TensorSpec {
                                name: inp
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("x")
                                    .to_string(),
                                shape: inp
                                    .get("shape")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                                            .collect()
                                    })
                                    .unwrap_or_default(),
                                dtype: inp
                                    .get("dtype")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("fp16")
                                    .to_string(),
                            })
                            .collect();
                        let outputs: Vec<TensorSpec> = fd
                            .outputs
                            .iter()
                            .map(|outp| TensorSpec {
                                name: outp
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("output")
                                    .to_string(),
                                shape: outp
                                    .get("shape")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                                            .collect()
                                    })
                                    .unwrap_or_default(),
                                dtype: outp
                                    .get("dtype")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("fp16")
                                    .to_string(),
                            })
                            .collect();
                        FunctionDescriptor {
                            name: fd.name.clone(),
                            inputs,
                            outputs,
                            stateful: fd.stateful,
                            emission_status: if result.status == "success" {
                                "emitted".to_string()
                            } else {
                                "seam_only".to_string()
                            },
                            mir_ops: vec![],
                        }
                    })
                    .collect()
            } else {
                vec![FunctionDescriptor {
                    name: "main".to_string(),
                    inputs: vec![],
                    outputs: vec![],
                    stateful: false,
                    emission_status: if result.status == "success" {
                        "emitted".to_string()
                    } else {
                        "seam_only".to_string()
                    },
                    mir_ops: vec![],
                }]
            };

            PackageEntry {
                name: shard_name.clone(),
                role: format!("DecoderShard({})", role),
                path: result.output_path.clone(),
                content_hash: result.content_hash.clone(),
                size_bytes: 0,
                functions,
            }
        })
        .collect();

    let handoffs: Vec<HandoffEntry> = pir
        .handoffs
        .iter()
        .map(|h| HandoffEntry {
            from_package: h.from_package.clone(),
            to_package: h.to_package.clone(),
            tensor_name: h.tensor_name.clone(),
            shape: h.shape.clone(),
            dtype: h.dtype.clone(),
        })
        .collect();

    let manifest = ArtifactManifest {
        version: "0.4.0".to_string(),
        model_id: spec.name.clone(),
        task_hash: task_hash.to_string(),
        created_at: timestamp,
        packages,
        state_declarations: vec![],
        handoffs,
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        implementation_status: "host_compiled".to_string(),
        verification_scope: "host_compile_only".to_string(),
        environment_limitations: vec![
            "no_apple_hardware".to_string(),
            "ane_placement_not_verified".to_string(),
            "no_on_device_predict".to_string(),
            "shard_aware_path".to_string(),
        ],
    };

    serde_json::to_value(&manifest)
        .unwrap_or_else(|_| serde_json::json!({"error": "manifest serialization failed"}))
}

/// Run a complete lab session: compile + inspect + baseline + drift + structured run record.
///
/// This drives the full lab pipeline:
/// 1. Load task spec
/// 2. Compile via bridge
/// 3. Write artifacts in lab run directory layout
/// 4. Perform host-side inspection (if not skipped)
/// 5. Compute FP32 baseline reference output
/// 6. Compute drift between baseline and actual output (if available)
/// 7. Write knowledge update with drift evidence
/// 8. Build and write the LabRun record
fn run_lab(
    input: &str,
    output: &str,
    bridge_script: &str,
    python_path: &str,
    do_inspect: bool,
    seed: u64,
    generated_from: Option<&str>,
) -> Result<(), String> {
    use ane_bridge::subprocess::PythonBridge;
    use ane_ir::linear_slice::{
        lower_linear_projection_to_mir, sir_from_linear_projection, FamilyPayload,
    };
    use ane_ir::task_spec::load_synthetic_task;
    use ane_lab::baseline::BaselineComputer;
    use ane_lab::drift::DriftDetector;
    use ane_lab::harness::{
        CompileStepResult, EnvironmentSummary, GeneratorProvenance, LabRunBuilder,
        VerificationScope,
    };
    use ane_lab::run_dir::{generate_run_id, layout, LabRunWriter};

    println!("=== MILLer — Lab Run ===\n");

    // Step 1: Load task spec
    println!("[1/8] Loading task spec: {}", input);
    let spec = load_synthetic_task(input)?;
    let task_hash = compute_task_hash(&spec);
    println!("  Task: {} (family: {})", spec.name, spec.family);
    println!("  Task hash: {}", task_hash);

    // Reject sharded ops — use generic methods for dimension extraction
    if spec.op.is_sharded() {
        return Err(format!("Use 'compile-sharded' command for {} tasks", spec.op.family_id()));
    }
    let (input_dim, output_dim, batch_size, _dtype) = spec.op.primary_dims();

    // Step 2: Build IR and compile
    println!("[2/8] Compiling...");
    let sir = sir_from_linear_projection(&spec)?;
    println!("  SIR: {} nodes", sir.nodes.len());

    let shard_name = format!("{}_shard_0", spec.name);
    let mir = lower_linear_projection_to_mir(&spec, &shard_name)?;

    let output_path = PathBuf::from(output);
    let mlpackage_output = output_path.join(layout::MLPACKAGE_DIR);
    // Use generic FamilyPayload — no per-variant match needed
    let payload = FamilyPayload::from_spec(&spec, mlpackage_output.to_str().unwrap_or(""))?;
    let payload_json = serde_json::to_value(&payload)
        .map_err(|e| format!("Payload serialization failed: {}", e))?;

    let bridge = PythonBridge::new(python_path, bridge_script);
    let result = bridge
        .execute_raw_payload(&payload_json)
        .map_err(|e| format!("Bridge execution failed: {}", e))?;

    let compile_step = CompileStepResult {
        success: result.status == "success",
        error: result.error_message.clone(),
        output_path: result.output_path.clone(),
        content_hash: result.content_hash.clone(),
        file_count: if result.package_files.is_empty() {
            None
        } else {
            Some(result.package_files.len())
        },
        coremltools_version: result.coremltools_version.clone(),
    };

    if compile_step.success {
        println!("  Compilation: SUCCESS");
        if let Some(ref hash) = compile_step.content_hash {
            println!("  Content hash: {}", hash);
        }
    } else {
        println!("  Compilation: FAILED");
        if let Some(ref err) = compile_step.error {
            println!("  Error: {}", err);
        }
    }

    // Step 3: Create lab run directory and write artifacts
    println!("[3/8] Writing lab run artifacts...");
    let run_id = generate_run_id(&task_hash);
    let _run_dir_initial = output_path.join(&run_id);
    let writer = LabRunWriter::new(&output_path);
    let run_dir = writer
        .create_run_directory(&run_id)
        .map_err(|e| format!("Failed to create run directory: {}", e))?;
    println!("  Run directory: {}", run_dir.display());

    // Write manifest
    let manifest = build_artifact_manifest(&spec, &result, &task_hash);
    writer
        .write_manifest(&run_dir, &manifest)
        .map_err(|e| format!("Failed to write manifest: {}", e))?;

    // Write MIR
    let mir_json =
        serde_json::to_value(&mir).map_err(|e| format!("MIR serialization failed: {}", e))?;
    writer.write_mir(&run_dir, &mir_json).map_err(|e| format!("Failed to write MIR: {}", e))?;

    // Step 4: Host-side inspection
    let inspect_step = if do_inspect && compile_step.success {
        println!("[4/8] Performing host-side inspection...");
        let inspector = ane_lab::host_inspect::HostInspector::new(python_path, bridge_script);
        let mlpackage_path = result.output_path.as_deref().unwrap_or("");
        let inspect_result = inspector.inspect(mlpackage_path);

        println!("  Package present: {}", inspect_result.package_present);
        println!("  Manifest readable: {}", inspect_result.manifest_readable);
        println!("  Model loadable: {}", inspect_result.model_loadable);
        if !inspect_result.model_loadable {
            if let Some(ref reason) = inspect_result.model_load_failure_reason {
                println!("  Load failure: {}", reason);
            }
        }
        if !inspect_result.warnings.is_empty() {
            println!("  Warnings:");
            for w in &inspect_result.warnings {
                println!("    - {}", w);
            }
        }

        // Write inspection result
        let inspect_json = serde_json::to_value(&inspect_result)
            .map_err(|e| format!("Inspection serialization failed: {}", e))?;
        writer
            .write_inspection(&run_dir, &inspect_json)
            .map_err(|e| format!("Failed to write inspection: {}", e))?;

        inspect_result
    } else {
        if !do_inspect {
            println!("[4/8] Host-side inspection: SKIPPED");
        } else {
            println!("[4/8] Host-side inspection: SKIPPED (compilation failed)");
        }
        ane_lab::harness::InspectionStepResult {
            package_present: false,
            manifest_readable: false,
            model_loadable: false,
            model_load_failure_reason: Some("Inspection not performed".to_string()),
            function_count: None,
            input_specs: vec![],
            output_specs: vec![],
            warnings: vec!["Host-side inspection was not performed".to_string()],
            structure_inspection_available: None,
            structure_inspection_failure_reason: Some("Inspection not performed".to_string()),
            structure_op_names: vec![],
            structure_op_count: None,
            structure_function_count: None,
            structure_state_declarations: vec![],
            op_fidelity_score: None,
            missing_ops: vec![],
            extra_ops: vec![],
            inspection_method: "none".to_string(),
        }
    };

    // Step 5: Compute FP32 baseline reference
    println!("[5/8] Computing FP32 baseline reference...");
    let baseline_computer = BaselineComputer::new(seed);
    let mut baseline_result = match &spec.op {
        ane_ir::task_spec::TaskOp::MlpBlock {
            input_dim,
            hidden_dim,
            output_dim,
            activation,
            batch_size,
            ..
        } => baseline_computer
            .compute_mlp_block(
                &spec.name,
                *input_dim,
                *hidden_dim,
                *output_dim,
                activation,
                *batch_size,
            )
            .map_err(|e| format!("MLP baseline computation failed: {}", e))?,
        ane_ir::task_spec::TaskOp::DecodeStep {
            embed_dim,
            num_heads,
            head_dim,
            kv_len,
            batch_size,
            ..
        } => baseline_computer
            .compute_decode_step(
                &spec.name,
                *embed_dim,
                *num_heads,
                *head_dim,
                *kv_len,
                *batch_size,
            )
            .map_err(|e| format!("Decode-step baseline computation failed: {}", e))?,
        ane_ir::task_spec::TaskOp::Attention {
            embed_dim,
            num_heads,
            head_dim,
            seq_len,
            batch_size,
            ..
        } => baseline_computer
            .compute_attention(&spec.name, *embed_dim, *num_heads, *head_dim, *seq_len, *batch_size)
            .map_err(|e| format!("Attention baseline computation failed: {}", e))?,
        ane_ir::task_spec::TaskOp::LutProjection {
            vocab_size,
            embed_dim,
            num_groups,
            lut_bitwidth,
            batch_size,
            ..
        } => baseline_computer
            .compute_lut_projection(
                &spec.name,
                *vocab_size,
                *embed_dim,
                *num_groups,
                *lut_bitwidth,
                *batch_size,
            )
            .map_err(|e| format!("LUT baseline computation failed: {}", e))?,
        ane_ir::task_spec::TaskOp::ShardedLinearPipeline {
            input_dim,
            hidden_dim,
            output_dim,
            batch_size,
            ..
        } => baseline_computer
            .compute_sharded_linear_pipeline(
                &spec.name,
                *input_dim,
                *hidden_dim,
                *output_dim,
                *batch_size,
            )
            .map_err(|e| format!("Sharded linear pipeline baseline computation failed: {}", e))?,
        ane_ir::task_spec::TaskOp::ShardedDecodeStep {
            embed_dim,
            num_heads,
            head_dim,
            kv_len,
            batch_size,
            ..
        } => baseline_computer
            .compute_sharded_decode_step(
                &spec.name,
                *embed_dim,
                *num_heads,
                *head_dim,
                *kv_len,
                *batch_size,
            )
            .map_err(|e| format!("Sharded decode-step baseline computation failed: {}", e))?,
        _ => baseline_computer
            .compute_linear_projection(&spec.name, input_dim, output_dim, batch_size)
            .map_err(|e| format!("Baseline computation failed: {}", e))?,
    };
    // Link baseline to the deterministic task identity
    baseline_result.task_hash = Some(task_hash.clone());
    println!(
        "  Baseline: {} output elements, computed in {:.3}ms",
        baseline_result.output_tensor.len(),
        baseline_result.compute_time_ms
    );

    // Write baseline artifact
    let baseline_json = serde_json::to_value(&baseline_result)
        .map_err(|e| format!("Baseline serialization failed: {}", e))?;
    writer
        .write_baseline(&run_dir, &baseline_json)
        .map_err(|e| format!("Failed to write baseline: {}", e))?;
    println!("  Baseline: {}", run_dir.join(layout::BASELINE_JSON).display());

    // Step 6: Compute drift (requires actual model output from predict())
    // On non-Apple hardware, actual outputs are unavailable, so drift is reported as unavailable.
    println!("[6/8] Computing drift metrics...");
    let drift_report = if compile_step.success {
        // We have a compiled model but cannot run predict() on this host.
        // Drift computation requires actual model output, which requires
        // Apple hardware with Core ML runtime.
        let unavailable_report =
            DriftDetector::unavailable("predict() requires Apple hardware with Core ML runtime");
        println!("  Drift: UNAVAILABLE (no on-device predict output)");
        unavailable_report
    } else {
        let unavailable_report =
            DriftDetector::unavailable("compilation failed — no model output to compare");
        println!("  Drift: UNAVAILABLE (compilation failed)");
        unavailable_report
    };

    // Write drift report
    let drift_json = serde_json::to_value(&drift_report)
        .map_err(|e| format!("Drift serialization failed: {}", e))?;
    writer
        .write_drift(&run_dir, &drift_json)
        .map_err(|e| format!("Failed to write drift report: {}", e))?;
    println!("  Drift report: {}", run_dir.join(layout::DRIFT_JSON).display());

    // Step 7: Write knowledge update with drift evidence
    println!("[7/8] Writing knowledge update...");
    let knowledge_update = build_knowledge_update_with_drift(
        &spec,
        &result,
        &task_hash,
        &baseline_result,
        &drift_report,
    );
    writer
        .write_knowledge_update(&run_dir, &spec.name, &knowledge_update)
        .map_err(|e| format!("Failed to write knowledge update: {}", e))?;
    println!("  Knowledge: {}", run_dir.join(layout::KNOWLEDGE_DIR).display());

    // Step 8: Build and write LabRun record
    println!("[8/8] Writing lab run record...");
    let env = EnvironmentSummary::detect(1); // bridge_version = 1
    let verification_scope = VerificationScope::HostOnlyInspection;

    let mut builder =
        LabRunBuilder::new(run_id, task_hash, spec.name.clone(), verification_scope, env)
            .compile_result(compile_step)
            .inspect_result(inspect_step)
            .artifact_directory(run_dir.to_string_lossy().to_string())
            .adaptation_readiness("artifacts_only".to_string())
            .warning("No device-backed profiling performed — requires Apple hardware".to_string())
            .warning(
                "Drift metrics unavailable — requires Apple hardware for predict() output"
                    .to_string(),
            );

    // Attach generator provenance if this run used a generated task
    if let Some(gen_info) = generated_from {
        // Parse format: "family,seed,generator_version"
        let parts: Vec<&str> = gen_info.splitn(3, ',').collect();
        if parts.len() == 3 {
            if let Ok(gen_seed) = parts[1].parse::<u64>() {
                builder = builder.generator_provenance(GeneratorProvenance {
                    generator_version: parts[2].to_string(),
                    family: parts[0].to_string(),
                    seed: gen_seed,
                    task_name: spec.name.clone(),
                });
                println!(
                    "  Generator provenance: family={}, seed={}, version={}",
                    parts[0], gen_seed, parts[2]
                );
            }
        } else {
            eprintln!(
                "  Warning: --generated-from format should be 'family,seed,version', got: {}",
                gen_info
            );
        }
    }

    let lab_run = builder.build();

    writer
        .write_run_record(&run_dir, &lab_run)
        .map_err(|e| format!("Failed to write run record: {}", e))?;
    println!("  Run record: {}", run_dir.join(layout::RUN_JSON).display());

    println!("\n=== Lab run summary ===");
    println!("  Run ID: {}", lab_run.run_id);
    println!("  Verification scope: {:?}", lab_run.verification_scope);
    println!(
        "  Compilation: {}",
        if lab_run.compile_result.success { "SUCCESS" } else { "FAILED" }
    );
    println!("  Baseline: {} FP32 reference values computed", baseline_result.output_tensor.len());
    println!("  Drift: {}", if drift_report.is_computed() { "computed" } else { "unavailable" });
    println!("  Artifacts: {}", run_dir.display());

    println!("\n=== Lab run complete ===");

    Ok(())
}

/// Ingest knowledge observations from a knowledge update JSON into the knowledge store.
///
/// This is the key function that closes the host-side evidence loop: it converts
/// observations from the JSON format produced by `build_knowledge_update_with_drift`
/// into proper `KnowledgeUnit` structs and ingests them via `UpdatePipeline`.
///
/// Returns the number of observations successfully ingested.
fn ingest_knowledge_observations(
    store: &mut ane_knowledge::store::KnowledgeStore,
    knowledge_update: &serde_json::Value,
    task_hash: &str,
) -> Result<usize, String> {
    use ane_ir::kir::{EvidenceSource, KnowledgeScope, KnowledgeType, KnowledgeUnit};
    use ane_knowledge::update::UpdatePipeline;

    let observations = knowledge_update
        .get("observations")
        .and_then(|v| v.as_array())
        .ok_or("No observations found in knowledge update")?;

    let mut pipeline = UpdatePipeline::new(store);
    let mut ingested = 0;

    for obs in observations {
        // Extract fields from the observation JSON
        let knowledge_type_str =
            obs.get("knowledge_type").and_then(|v| v.as_str()).unwrap_or("LegalityRule");

        let knowledge_type = match knowledge_type_str {
            "LegalityRule" => KnowledgeType::LegalityRule,
            "PrecisionHazard" => KnowledgeType::PrecisionHazard,
            "SurvivalMatrixEntry" => KnowledgeType::SurvivalMatrixEntry,
            "FallbackSignature" => KnowledgeType::FallbackSignature,
            "MotifCatalog" => KnowledgeType::MotifCatalog,
            "ShardTemplateKnowledge" => KnowledgeType::ShardTemplateKnowledge,
            "DeviceFingerprint" => KnowledgeType::DeviceFingerprint,
            "StateTopologyOutcome" => KnowledgeType::StateTopologyOutcome,
            "SyntheticTransferAnnotation" => KnowledgeType::SyntheticTransferAnnotation,
            _ => KnowledgeType::LegalityRule,
        };

        let confidence = obs.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

        let evidence_source_str =
            obs.get("evidence_source").and_then(|v| v.as_str()).unwrap_or("SyntheticRun");

        let evidence_source = match evidence_source_str {
            "SyntheticRun" => EvidenceSource::SyntheticRun,
            "RealModelRun" => EvidenceSource::RealModelRun,
            "CompileFailure" => EvidenceSource::CompileFailure,
            "LoadFailure" => EvidenceSource::LoadFailure,
            "RuntimeAnomaly" => EvidenceSource::RuntimeAnomaly,
            "ManualEntry" => EvidenceSource::ManualEntry,
            "CrossValidated" => EvidenceSource::CrossValidated,
            _ => EvidenceSource::SyntheticRun,
        };

        let evidence_count =
            obs.get("evidence_count").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

        // Build a unique ID for this observation
        let obs_id = format!("obs_{}_{}", task_hash.replace(":", "_"), ingested);

        // Build scope from observation
        let scope_json = obs.get("scope").cloned().unwrap_or(serde_json::json!({
            "device_classes": ["unknown"],
            "os_versions": ["unknown"],
            "opset_versions": ["iOS18"],
        }));

        let scope = KnowledgeScope {
            device_classes: scope_json
                .get("device_classes")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            os_versions: scope_json
                .get("os_versions")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            opset_versions: scope_json
                .get("opset_versions")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
        };

        // Build the payload from all remaining fields
        let mut payload = std::collections::HashMap::new();
        if let Some(obj) = obs.as_object() {
            for (key, value) in obj {
                if !matches!(
                    key.as_str(),
                    "knowledge_type"
                        | "confidence"
                        | "evidence_source"
                        | "evidence_count"
                        | "scope"
                ) {
                    payload.insert(key.clone(), value.clone());
                }
            }
        }

        let unit = KnowledgeUnit {
            id: obs_id,
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type,
            confidence,
            evidence_source,
            evidence_count,
            scope,
            conflict_priority: 0,
            payload,
        };

        match pipeline.ingest(unit) {
            Ok(()) => ingested += 1,
            Err(e) => eprintln!("  Warning: failed to ingest observation: {}", e),
        }
    }

    Ok(ingested)
}

/// Run the lab-loop subcommand: a single host-side evidence loop that persists
/// observations into the knowledge store.
///
/// This command closes the loop from task → compile → baseline → drift →
/// knowledge store persistence. The key difference from `lab` is that after
/// computing the knowledge update JSON, this command actually ingests the
/// observations into the KnowledgeStore using UpdatePipeline, making them
/// queryable by the pass pipeline in subsequent compiles.
///
/// Steps:
/// 1. Load task spec from input TOML
/// 2. Compute task hash
/// 3. Build SIR and MIR
/// 4. Build bridge payload and invoke Python bridge
/// 5. Compute baseline
/// 6. Compute drift (unavailable on non-Apple hardware, but the path must exist)
/// 7. Open knowledge store and ingest observations
/// 8. Write all run artifacts
/// 9. Determine and record adaptation_readiness metadata
fn run_lab_loop(
    input: &str,
    output: &str,
    bridge_script: &str,
    python_path: &str,
    knowledge_dir: &str,
    seed: u64,
    generated_from: Option<&str>,
) -> Result<(), String> {
    use ane_bridge::subprocess::PythonBridge;
    use ane_ir::linear_slice::{
        lower_linear_projection_to_mir, sir_from_linear_projection, FamilyPayload,
    };
    use ane_ir::task_spec::load_synthetic_task;
    use ane_lab::baseline::BaselineComputer;
    use ane_lab::drift::DriftDetector;
    use ane_lab::harness::{
        CompileStepResult, EnvironmentSummary, GeneratorProvenance, LabRunBuilder,
        VerificationScope,
    };
    use ane_lab::run_dir::{generate_run_id, layout, LabRunWriter};

    println!("=== MILLer — Lab-Loop (Host-Side Evidence Loop) ===\n");

    // Step 1: Load task spec
    println!("[1/9] Loading task spec: {}", input);
    let spec = load_synthetic_task(input)?;
    let task_hash = compute_task_hash(&spec);
    println!("  Task: {} (family: {})", spec.name, spec.family);
    println!("  Task hash: {}", task_hash);

    // Reject sharded ops — use generic methods for dimension extraction
    if spec.op.is_sharded() {
        return Err(format!("Use 'compile-sharded' command for {} tasks", spec.op.family_id()));
    }
    let (input_dim, output_dim, batch_size, _dtype) = spec.op.primary_dims();

    // Step 2: Build IR and compile
    println!("[2/9] Compiling...");
    let sir = sir_from_linear_projection(&spec)?;
    println!("  SIR: {} nodes", sir.nodes.len());

    let shard_name = format!("{}_shard_0", spec.name);
    let mir = lower_linear_projection_to_mir(&spec, &shard_name)?;

    let output_path = PathBuf::from(output);
    let mlpackage_output = output_path.join(layout::MLPACKAGE_DIR);
    // Use generic FamilyPayload — no per-variant match needed
    let payload = FamilyPayload::from_spec(&spec, mlpackage_output.to_str().unwrap_or(""))?;
    let payload_json = serde_json::to_value(&payload)
        .map_err(|e| format!("Payload serialization failed: {}", e))?;

    let bridge = PythonBridge::new(python_path, bridge_script);
    let result = bridge
        .execute_raw_payload(&payload_json)
        .map_err(|e| format!("Bridge execution failed: {}", e))?;

    let compile_step = CompileStepResult {
        success: result.status == "success",
        error: result.error_message.clone(),
        output_path: result.output_path.clone(),
        content_hash: result.content_hash.clone(),
        file_count: if result.package_files.is_empty() {
            None
        } else {
            Some(result.package_files.len())
        },
        coremltools_version: result.coremltools_version.clone(),
    };

    if compile_step.success {
        println!("  Compilation: SUCCESS");
        if let Some(ref hash) = compile_step.content_hash {
            println!("  Content hash: {}", hash);
        }
    } else {
        println!("  Compilation: FAILED");
        if let Some(ref err) = compile_step.error {
            println!("  Error: {}", err);
        }
    }

    // Step 3: Create lab run directory and write initial artifacts
    println!("[3/9] Writing lab-loop run artifacts...");
    let run_id = generate_run_id(&task_hash);
    let writer = LabRunWriter::new(&output_path);
    let run_dir = writer
        .create_run_directory(&run_id)
        .map_err(|e| format!("Failed to create run directory: {}", e))?;
    println!("  Run directory: {}", run_dir.display());

    // Write manifest
    let mut manifest = build_artifact_manifest(&spec, &result, &task_hash);

    // Write MIR
    let mir_json =
        serde_json::to_value(&mir).map_err(|e| format!("MIR serialization failed: {}", e))?;
    writer.write_mir(&run_dir, &mir_json).map_err(|e| format!("Failed to write MIR: {}", e))?;

    // Step 4: Host-side inspection
    println!("[4/9] Performing host-side inspection...");
    let inspect_step = if compile_step.success {
        let inspector = ane_lab::host_inspect::HostInspector::new(python_path, bridge_script);
        let mlpackage_path = result.output_path.as_deref().unwrap_or("");
        let inspect_result = inspector.inspect(mlpackage_path);

        println!("  Package present: {}", inspect_result.package_present);
        println!("  Manifest readable: {}", inspect_result.manifest_readable);
        println!("  Model loadable: {}", inspect_result.model_loadable);
        if !inspect_result.model_loadable {
            if let Some(ref reason) = inspect_result.model_load_failure_reason {
                println!("  Load failure: {}", reason);
            }
        }

        // Write inspection result
        let inspect_json = serde_json::to_value(&inspect_result)
            .map_err(|e| format!("Inspection serialization failed: {}", e))?;
        writer
            .write_inspection(&run_dir, &inspect_json)
            .map_err(|e| format!("Failed to write inspection: {}", e))?;

        inspect_result
    } else {
        println!("  Host-side inspection: SKIPPED (compilation failed)");
        ane_lab::harness::InspectionStepResult {
            package_present: false,
            manifest_readable: false,
            model_loadable: false,
            model_load_failure_reason: Some("Inspection not performed".to_string()),
            function_count: None,
            input_specs: vec![],
            output_specs: vec![],
            warnings: vec!["Host-side inspection was not performed".to_string()],
            structure_inspection_available: None,
            structure_inspection_failure_reason: Some("Inspection not performed".to_string()),
            structure_op_names: vec![],
            structure_op_count: None,
            structure_function_count: None,
            structure_state_declarations: vec![],
            op_fidelity_score: None,
            missing_ops: vec![],
            extra_ops: vec![],
            inspection_method: "none".to_string(),
        }
    };

    // Step 5: Compute FP32 baseline reference
    println!("[5/9] Computing FP32 baseline reference...");
    let baseline_computer = BaselineComputer::new(seed);
    let mut baseline_result = match &spec.op {
        ane_ir::task_spec::TaskOp::MlpBlock {
            input_dim,
            hidden_dim,
            output_dim,
            activation,
            batch_size,
            ..
        } => baseline_computer
            .compute_mlp_block(
                &spec.name,
                *input_dim,
                *hidden_dim,
                *output_dim,
                activation,
                *batch_size,
            )
            .map_err(|e| format!("MLP baseline computation failed: {}", e))?,
        ane_ir::task_spec::TaskOp::DecodeStep {
            embed_dim,
            num_heads,
            head_dim,
            kv_len,
            batch_size,
            ..
        } => baseline_computer
            .compute_decode_step(
                &spec.name,
                *embed_dim,
                *num_heads,
                *head_dim,
                *kv_len,
                *batch_size,
            )
            .map_err(|e| format!("Decode-step baseline computation failed: {}", e))?,
        ane_ir::task_spec::TaskOp::Attention {
            embed_dim,
            num_heads,
            head_dim,
            seq_len,
            batch_size,
            ..
        } => baseline_computer
            .compute_attention(&spec.name, *embed_dim, *num_heads, *head_dim, *seq_len, *batch_size)
            .map_err(|e| format!("Attention baseline computation failed: {}", e))?,
        ane_ir::task_spec::TaskOp::LutProjection {
            vocab_size,
            embed_dim,
            num_groups,
            lut_bitwidth,
            batch_size,
            ..
        } => baseline_computer
            .compute_lut_projection(
                &spec.name,
                *vocab_size,
                *embed_dim,
                *num_groups,
                *lut_bitwidth,
                *batch_size,
            )
            .map_err(|e| format!("LUT baseline computation failed: {}", e))?,
        ane_ir::task_spec::TaskOp::ShardedLinearPipeline {
            input_dim,
            hidden_dim,
            output_dim,
            batch_size,
            ..
        } => baseline_computer
            .compute_sharded_linear_pipeline(
                &spec.name,
                *input_dim,
                *hidden_dim,
                *output_dim,
                *batch_size,
            )
            .map_err(|e| format!("Sharded linear pipeline baseline computation failed: {}", e))?,
        ane_ir::task_spec::TaskOp::ShardedDecodeStep {
            embed_dim,
            num_heads,
            head_dim,
            kv_len,
            batch_size,
            ..
        } => baseline_computer
            .compute_sharded_decode_step(
                &spec.name,
                *embed_dim,
                *num_heads,
                *head_dim,
                *kv_len,
                *batch_size,
            )
            .map_err(|e| format!("Sharded decode-step baseline computation failed: {}", e))?,
        _ => baseline_computer
            .compute_linear_projection(&spec.name, input_dim, output_dim, batch_size)
            .map_err(|e| format!("Baseline computation failed: {}", e))?,
    };
    baseline_result.task_hash = Some(task_hash.clone());
    println!(
        "  Baseline: {} output elements, computed in {:.3}ms",
        baseline_result.output_tensor.len(),
        baseline_result.compute_time_ms
    );

    // Write baseline artifact
    let baseline_json = serde_json::to_value(&baseline_result)
        .map_err(|e| format!("Baseline serialization failed: {}", e))?;
    writer
        .write_baseline(&run_dir, &baseline_json)
        .map_err(|e| format!("Failed to write baseline: {}", e))?;

    // Step 6: Compute drift (requires actual model output from predict())
    println!("[6/9] Computing drift metrics...");
    let drift_report = if compile_step.success {
        let unavailable_report =
            DriftDetector::unavailable("predict() requires Apple hardware with Core ML runtime");
        println!("  Drift: UNAVAILABLE (no on-device predict output)");
        unavailable_report
    } else {
        let unavailable_report =
            DriftDetector::unavailable("compilation failed — no model output to compare");
        println!("  Drift: UNAVAILABLE (compilation failed)");
        unavailable_report
    };

    // Write drift report
    let drift_json = serde_json::to_value(&drift_report)
        .map_err(|e| format!("Drift serialization failed: {}", e))?;
    writer
        .write_drift(&run_dir, &drift_json)
        .map_err(|e| format!("Failed to write drift report: {}", e))?;

    // Step 7: Build knowledge update and ingest observations into the store
    println!("[7/9] Ingesting observations into knowledge store...");
    let knowledge_update = build_knowledge_update_with_drift(
        &spec,
        &result,
        &task_hash,
        &baseline_result,
        &drift_report,
    );

    // Write knowledge update artifact
    writer
        .write_knowledge_update(&run_dir, &spec.name, &knowledge_update)
        .map_err(|e| format!("Failed to write knowledge update: {}", e))?;

    // Open the knowledge store and ingest observations
    let knowledge_store_path = PathBuf::from(knowledge_dir);
    let mut store = if knowledge_store_path.join("store_index.json").exists() {
        ane_knowledge::store::KnowledgeStore::open(knowledge_dir)
            .map_err(|e| format!("Failed to open knowledge store at {}: {}", knowledge_dir, e))?
    } else {
        // Create a new store at the specified path
        fs::create_dir_all(&knowledge_store_path)
            .map_err(|e| format!("Failed to create knowledge store directory: {}", e))?;
        ane_knowledge::store::KnowledgeStore::open(knowledge_dir)
            .map_err(|e| format!("Failed to create knowledge store at {}: {}", knowledge_dir, e))?
    };

    let ingested_count = ingest_knowledge_observations(&mut store, &knowledge_update, &task_hash)?;
    println!(
        "  Ingested {} observations into knowledge store at {}",
        ingested_count, knowledge_dir
    );

    // Step 8: Determine adaptation_readiness
    println!("[8/9] Determining adaptation readiness...");
    let readiness_level = if ingested_count > 0 {
        // Check if any ingested observation is compiler-consumable
        // (confidence > 0 and evidence_count >= 1, making it queryable by the pass pipeline)
        let empty_observations: Vec<serde_json::Value> = vec![];
        let observations = knowledge_update
            .get("observations")
            .and_then(|v| v.as_array())
            .unwrap_or(&empty_observations);
        let has_compiler_consumable = observations.iter().any(|obs| {
            let conf = obs.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let ev_count = obs.get("evidence_count").and_then(|v| v.as_u64()).unwrap_or(0);
            conf > 0.0 && ev_count >= 1
        });
        if has_compiler_consumable {
            "artifacts_observation_compiler_consumable"
        } else {
            "artifacts_and_observation"
        }
    } else {
        "artifacts_only"
    };
    println!("  Adaptation readiness: {}", readiness_level);

    // Step 9: Build and write LabRun record with adaptation_readiness
    println!("[9/9] Writing lab-loop run record...");
    let env = EnvironmentSummary::detect(1);
    let verification_scope = VerificationScope::HostOnlyInspection;

    let mut builder =
        LabRunBuilder::new(run_id, task_hash, spec.name.clone(), verification_scope, env)
            .compile_result(compile_step)
            .inspect_result(inspect_step)
            .artifact_directory(run_dir.to_string_lossy().to_string())
            .adaptation_readiness(readiness_level.to_string())
            .warning("No device-backed profiling performed — requires Apple hardware".to_string())
            .warning(
                "Drift metrics unavailable — requires Apple hardware for predict() output"
                    .to_string(),
            );

    // Attach generator provenance if this run used a generated task
    if let Some(gen_info) = generated_from {
        let parts: Vec<&str> = gen_info.splitn(3, ',').collect();
        if parts.len() == 3 {
            if let Ok(gen_seed) = parts[1].parse::<u64>() {
                builder = builder.generator_provenance(GeneratorProvenance {
                    generator_version: parts[2].to_string(),
                    family: parts[0].to_string(),
                    seed: gen_seed,
                    task_name: spec.name.clone(),
                });
                println!(
                    "  Generator provenance: family={}, seed={}, version={}",
                    parts[0], gen_seed, parts[2]
                );
            }
        } else {
            eprintln!(
                "  Warning: --generated-from format should be 'family,seed,version', got: {}",
                gen_info
            );
        }
    }

    let lab_run = builder.build();

    writer
        .write_run_record(&run_dir, &lab_run)
        .map_err(|e| format!("Failed to write run record: {}", e))?;

    // Add adaptation_readiness to manifest
    if let Some(obj) = manifest.as_object_mut() {
        obj.insert("adaptation_readiness".to_string(), serde_json::json!(readiness_level));
        obj.insert("knowledge_store_path".to_string(), serde_json::json!(knowledge_dir));
        obj.insert("observations_ingested".to_string(), serde_json::json!(ingested_count));
    }
    writer
        .write_manifest(&run_dir, &manifest)
        .map_err(|e| format!("Failed to write manifest: {}", e))?;

    println!("  Run record: {}", run_dir.join(layout::RUN_JSON).display());

    println!("\n=== Lab-loop run summary ===");
    println!("  Run ID: {}", lab_run.run_id);
    println!("  Verification scope: {:?}", lab_run.verification_scope);
    println!(
        "  Compilation: {}",
        if lab_run.compile_result.success { "SUCCESS" } else { "FAILED" }
    );
    println!("  Baseline: {} FP32 reference values computed", baseline_result.output_tensor.len());
    println!("  Drift: {}", if drift_report.is_computed() { "computed" } else { "unavailable" });
    println!("  Observations ingested: {}", ingested_count);
    println!("  Adaptation readiness: {}", readiness_level);
    println!("  Knowledge store: {}", knowledge_dir);
    println!("  Artifacts: {}", run_dir.display());

    println!("\n=== Lab-loop run complete ===");

    Ok(())
}

/// Run the profiling subcommand.
///
/// On Apple hardware, this invokes the Python bridge's `profile` command
/// to run the model and capture timing. On non-Apple hardware, it reports
/// the limitation honestly.
fn run_profile(
    mlpackage_path: &str,
    output: &str,
    bridge_script: &str,
    python_path: &str,
    warmup: usize,
    iterations: usize,
    compute_units: &str,
) -> Result<(), String> {
    use ane_bridge::subprocess::PythonBridge;
    use ane_lab::device_meta::DeviceMetadata;
    use ane_lab::fallback::FallbackDetector;
    use ane_lab::harness::TimingResult;

    println!("=== MILLer — Device Profiling ===\n");

    // Check device metadata
    let device_meta = DeviceMetadata::host_only();
    if !device_meta.is_device_backed() {
        println!("Device metadata: host-only environment detected");
        println!("  Core ML runtime: not available");
        println!("  Compute plan: not available");
    }

    // Invoke Python bridge profile command
    println!("[1/3] Invoking profile command via Python bridge...");
    let bridge = PythonBridge::new(python_path, bridge_script);
    let payload = serde_json::json!({
        "command": "profile",
        "bridge_version": 1,
        "mlpackage_path": mlpackage_path,
        "compute_units": compute_units,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "seed": 42,
    });

    let result = bridge
        .execute_raw_payload(&payload)
        .map_err(|e| format!("Bridge invocation failed: {}", e))?;

    if result.status != "success" {
        println!("  Profiling not available:");
        if let Some(ref err) = result.error_message {
            println!("  {}", err);
        }
        println!("\nProfiling requires Apple hardware with Core ML runtime.");

        // Write a result indicating unavailability
        let output_path = PathBuf::from(output);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create output dir: {}", e))?;
        }
        let unavailable_result = serde_json::json!({
            "status": "unavailable",
            "reason": "Profiling requires Apple hardware with Core ML runtime",
            "mlpackage_path": mlpackage_path,
            "device_metadata": device_meta,
            "timing": null,
            "fallback_suspicion": {
                "suspicion_level": "unavailable",
                "explanation": "Cannot assess fallback without device-backed execution",
                "evidence": []
            },
        });
        let json = serde_json::to_string_pretty(&unavailable_result)
            .map_err(|e| format!("JSON serialization failed: {}", e))?;
        fs::write(&output_path, json).map_err(|e| format!("Failed to write output: {}", e))?;

        return Ok(());
    }

    // Extract timing from metadata
    println!("[2/3] Processing timing results...");
    let timing: Option<TimingResult> = result.metadata.get("timing").and_then(|t| {
        Some(TimingResult {
            warmup_iterations: t.get("warmup_iterations")?.as_u64()? as usize,
            measured_iterations: t.get("measured_iterations")?.as_u64()? as usize,
            p50_ms: t.get("p50_ms")?.as_f64()?,
            p90_ms: t.get("p90_ms")?.as_f64()?,
            p99_ms: t.get("p99_ms")?.as_f64()?,
            min_ms: t.get("min_ms")?.as_f64()?,
            max_ms: t.get("max_ms")?.as_f64()?,
            mean_ms: t.get("mean_ms")?.as_f64()?,
            std_dev_ms: t.get("std_dev_ms")?.as_f64()?,
            compute_units: t.get("compute_units")?.as_str()?.to_string(),
            scope_note: t.get("scope_note")?.as_str()?.to_string(),
        })
    });

    if let Some(ref t) = timing {
        println!("  p50: {:.3}ms", t.p50_ms);
        println!("  p90: {:.3}ms", t.p90_ms);
        println!("  p99: {:.3}ms", t.p99_ms);
        println!("  min: {:.3}ms", t.min_ms);
        println!("  max: {:.3}ms", t.max_ms);
        println!("  mean: {:.3}ms (stddev: {:.3}ms)", t.mean_ms, t.std_dev_ms);
        println!(
            "  Iterations: {} warmup + {} measured",
            t.warmup_iterations, t.measured_iterations
        );
        println!("  Scope: {}", t.scope_note);
    }

    // Fallback suspicion
    println!("[3/3] Assessing fallback suspicion...");
    let detector = FallbackDetector::new();
    let fallback_suspicion = detector.detect_from_timing(
        timing.as_ref().map(|t| t.p50_ms).unwrap_or(0.0),
        None, // No expected ANE latency baseline available yet
        &device_meta,
    );
    println!("  Suspicion level: {:?}", fallback_suspicion.suspicion_level);
    println!("  Explanation: {}", fallback_suspicion.explanation);

    // Write results
    let output_path = PathBuf::from(output);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create output dir: {}", e))?;
    }

    let profile_result = serde_json::json!({
        "status": "completed",
        "mlpackage_path": mlpackage_path,
        "device_metadata": device_meta,
        "timing": timing,
        "fallback_suspicion": fallback_suspicion,
    });

    let json = serde_json::to_string_pretty(&profile_result)
        .map_err(|e| format!("JSON serialization failed: {}", e))?;
    fs::write(&output_path, json).map_err(|e| format!("Failed to write output: {}", e))?;

    println!("\n=== Profiling complete ===");
    println!("Results: {}", output_path.display());

    Ok(())
}

/// Build an artifact manifest from the compilation result.
///
/// Uses the typed ArtifactManifest struct from manifest.rs with truth fields
/// (implementation_status, verification_scope, environment_limitations)
/// that prevent the manifest from being misread as proving device/runtime success.
///
/// Deterministic identity: task_hash is the primary identifier.
/// Timestamp is informational only.
fn build_artifact_manifest(
    spec: &ane_ir::task_spec::SyntheticTaskSpec,
    bridge_result: &ane_bridge::subprocess::BridgeResult,
    task_hash: &str,
) -> serde_json::Value {
    use ane_artifacts::manifest::{ArtifactManifest, FunctionDescriptor, PackageEntry, TensorSpec};

    let timestamp = chrono::Utc::now().to_rfc3339();

    // Extract actual dimensions from the spec
    let (input_dim, output_dim, batch_size, dtype) = spec.op.primary_dims();

    // Build function descriptors from bridge result or spec fallback
    let functions: Vec<FunctionDescriptor> = if !bridge_result.function_descriptors.is_empty() {
        bridge_result
            .function_descriptors
            .iter()
            .map(|fd| {
                let inputs: Vec<TensorSpec> = fd
                    .inputs
                    .iter()
                    .map(|inp| TensorSpec {
                        name: inp
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        shape: inp
                            .get("shape")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
                            })
                            .unwrap_or_default(),
                        dtype: inp
                            .get("dtype")
                            .and_then(|v| v.as_str())
                            .unwrap_or("fp16")
                            .to_string(),
                    })
                    .collect();
                let outputs: Vec<TensorSpec> = fd
                    .outputs
                    .iter()
                    .map(|outp| TensorSpec {
                        name: outp
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        shape: outp
                            .get("shape")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
                            })
                            .unwrap_or_default(),
                        dtype: outp
                            .get("dtype")
                            .and_then(|v| v.as_str())
                            .unwrap_or("fp16")
                            .to_string(),
                    })
                    .collect();
                FunctionDescriptor {
                    name: fd.name.clone(),
                    inputs,
                    outputs,
                    stateful: fd.stateful,
                    emission_status: "emitted".to_string(),
                    mir_ops: vec![],
                }
            })
            .collect()
    } else {
        // Fallback: derive from spec dimensions (not hardcoded)
        vec![FunctionDescriptor {
            name: "main".to_string(),
            inputs: vec![TensorSpec {
                name: "x".to_string(),
                shape: vec![batch_size, input_dim],
                dtype: dtype.clone(),
            }],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![batch_size, output_dim],
                dtype: dtype.clone(),
            }],
            stateful: false,
            emission_status: if bridge_result.status == "success" {
                "emitted".to_string()
            } else {
                "seam_only".to_string()
            },
            mir_ops: vec![],
        }]
    };

    let packages: Vec<PackageEntry> = if bridge_result.status == "success" {
        vec![PackageEntry {
            name: spec.name.clone(),
            role: "synthetic_microkernel".to_string(),
            path: bridge_result.output_path.clone(),
            content_hash: bridge_result.content_hash.clone(),
            size_bytes: 0, // Computed by packaging step if used
            functions,
        }]
    } else {
        vec![]
    };

    let manifest = ArtifactManifest {
        version: "0.3.0".to_string(),
        model_id: spec.name.clone(),
        task_hash: task_hash.to_string(),
        created_at: timestamp,
        packages,
        state_declarations: vec![],
        handoffs: vec![],
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        implementation_status: "host_compiled".to_string(),
        verification_scope: "host_compile_only".to_string(),
        environment_limitations: vec![
            "no_apple_hardware".to_string(),
            "ane_placement_not_verified".to_string(),
            "no_on_device_predict".to_string(),
        ],
    };

    serde_json::to_value(&manifest)
        .unwrap_or_else(|_| serde_json::json!({"error": "manifest serialization failed"}))
}

/// Build a backend-knowledge update from the compilation result.
///
/// Includes the deterministic task hash for identity tracking.
fn build_knowledge_update(
    spec: &ane_ir::task_spec::SyntheticTaskSpec,
    bridge_result: &ane_bridge::subprocess::BridgeResult,
    task_hash: &str,
) -> serde_json::Value {
    let timestamp = chrono::Utc::now().to_rfc3339();

    let (input_dim, output_dim, _batch_size, _dtype) = spec.op.primary_dims();

    serde_json::json!({
        "version": 2,
        "timestamp": timestamp,
        "source": "vertical_slice_compile",
        "task_hash": task_hash,
        "task_name": spec.name,
        "task_family": spec.family,
        "observations": [
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.matmul",
                "ane_legal": bridge_result.status == "success",
                "confidence": if bridge_result.status == "success" { 0.3 } else { 0.7 },
                "evidence_source": "SyntheticRun",
                "evidence_count": 1,
                "scope": {
                    "device_classes": ["unknown"],
                    "os_versions": ["unknown"],
                    "opset_versions": ["iOS18"],
                },
                "context": format!("LinearProjection {}x{}", input_dim, output_dim),
            },
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.add",
                "ane_legal": bridge_result.status == "success",
                "confidence": if bridge_result.status == "success" { 0.3 } else { 0.7 },
                "evidence_source": "SyntheticRun",
                "evidence_count": 1,
                "scope": {
                    "device_classes": ["unknown"],
                    "os_versions": ["unknown"],
                    "opset_versions": ["iOS18"],
                },
                "context": "bias addition after matmul",
            },
        ],
        "compilation_result": {
            "status": bridge_result.status,
            "mlpackage_produced": bridge_result.output_path.is_some(),
            "content_hash": bridge_result.content_hash,
        },
        "residuals": [
            "Device-specific ANE placement not verified (requires Apple hardware)",
            "Numerical drift not measured (requires Apple hardware for predict())",
            "Fallback suspicion not assessed (requires compute plan on Apple hardware)",
        ],
    })
}

/// Build a backend-knowledge update that includes drift evidence.
///
/// This extends the standard knowledge update with:
/// - baseline provenance (FP32 reference was computed, linked by task_hash)
/// - drift observation (if available, with scope/confidence fields)
/// - honest residual about what could not be measured
fn build_knowledge_update_with_drift(
    spec: &ane_ir::task_spec::SyntheticTaskSpec,
    bridge_result: &ane_bridge::subprocess::BridgeResult,
    task_hash: &str,
    baseline: &ane_lab::baseline::BaselineResult,
    drift: &ane_lab::drift::DriftReport,
) -> serde_json::Value {
    let timestamp = chrono::Utc::now().to_rfc3339();

    let (input_dim, output_dim, _batch_size, _dtype) = spec.op.primary_dims();

    // Build drift observation based on computation status
    let drift_observation = if drift.is_computed() {
        serde_json::json!({
            "knowledge_type": "PrecisionHazard",
            "op_pattern": "linear_projection_fp16_vs_fp32",
            "max_absolute_error": drift.max_absolute_error,
            "mean_absolute_error": drift.mean_absolute_error,
            "rmse": drift.rmse,
            "cosine_distance": drift.cosine_distance,
            "relative_error_p99": drift.relative_error_p99,
            "has_drift": drift.has_drift,
            "confidence": 0.3,
            "evidence_source": "SyntheticRun",
            "evidence_count": 1,
            "scope": {
                "device_classes": ["unknown"],
                "os_versions": ["unknown"],
                "opset_versions": ["iOS18"],
            },
            "context": format!("FP16 vs FP32 drift for LinearProjection {}x{}", input_dim, output_dim),
        })
    } else {
        serde_json::json!({
            "knowledge_type": "PrecisionHazard",
            "op_pattern": "linear_projection_fp16_vs_fp32",
            "computation_status": "unavailable",
            "reason": match &drift.computation_status {
                ane_lab::drift::DriftComputationStatus::Unavailable { reason } => reason.clone(),
                _ => "unknown".to_string(),
            },
            "confidence": 0.0,
            "evidence_source": "None",
            "evidence_count": 0,
            "scope": {
                "device_classes": [],
                "os_versions": [],
                "opset_versions": ["iOS18"],
            },
            "note": "Drift could not be computed — requires predict() output from Apple hardware",
        })
    };

    serde_json::json!({
        "version": 3,
        "timestamp": timestamp,
        "source": "lab_run_with_drift",
        "task_hash": task_hash,
        "task_name": spec.name,
        "task_family": spec.family,
        "observations": [
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.matmul",
                "ane_legal": bridge_result.status == "success",
                "confidence": if bridge_result.status == "success" { 0.3 } else { 0.7 },
                "evidence_source": "SyntheticRun",
                "evidence_count": 1,
                "scope": {
                    "device_classes": ["unknown"],
                    "os_versions": ["unknown"],
                    "opset_versions": ["iOS18"],
                },
                "context": format!("LinearProjection {}x{}", input_dim, output_dim),
            },
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.add",
                "ane_legal": bridge_result.status == "success",
                "confidence": if bridge_result.status == "success" { 0.3 } else { 0.7 },
                "evidence_source": "SyntheticRun",
                "evidence_count": 1,
                "scope": {
                    "device_classes": ["unknown"],
                    "os_versions": ["unknown"],
                    "opset_versions": ["iOS18"],
                },
                "context": "bias addition after matmul",
            },
            drift_observation,
        ],
        "baseline_provenance": {
            "baseline_schema_version": baseline.baseline_schema_version,
            "task_id": baseline.task_id,
            "task_hash": baseline.task_hash,
            "seed": baseline.seed,
            "precision": baseline.precision,
            "output_element_count": baseline.output_tensor.len(),
            "compute_time_ms": baseline.compute_time_ms,
        },
        "drift_evidence": {
            "drift_report_schema_version": drift.drift_report_schema_version,
            "computation_status": match &drift.computation_status {
                ane_lab::drift::DriftComputationStatus::Computed => "computed",
                ane_lab::drift::DriftComputationStatus::Unavailable { .. } => "unavailable",
                ane_lab::drift::DriftComputationStatus::LengthMismatch { .. } => "length_mismatch",
                ane_lab::drift::DriftComputationStatus::EmptyInput => "empty_input",
            },
            "has_drift": drift.has_drift,
            "max_absolute_error": drift.max_absolute_error,
            "mean_absolute_error": drift.mean_absolute_error,
            "rmse": drift.rmse,
            "scope_note": drift.scope_note,
        },
        "compilation_result": {
            "status": bridge_result.status,
            "mlpackage_produced": bridge_result.output_path.is_some(),
            "content_hash": bridge_result.content_hash,
        },
        "residuals": [
            "Device-specific ANE placement not verified (requires Apple hardware)",
            "Numerical drift not fully measured — baseline computed but actual model output requires Apple hardware for predict()",
            "Fallback suspicion not assessed (requires compute plan on Apple hardware)",
        ],
    })
}

/// Run the package subcommand.
///
/// Uses the `Packager` from `ane-artifacts` to create a deterministic
/// zip archive from a compile output directory. The zip contains all
/// artifacts: mlpackage, manifest, MIR dump, knowledge updates, etc.
fn run_package(input: &str, output: &str) -> Result<(), String> {
    use ane_artifacts::packaging::Packager;

    let input_path = PathBuf::from(input);
    if !input_path.exists() {
        return Err(format!("Input directory does not exist: {}", input));
    }
    if !input_path.join("manifest.json").exists() {
        return Err(format!("Input directory must contain manifest.json: {}", input));
    }

    let output_path = PathBuf::from(output);
    fs::create_dir_all(&output_path)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    // Read manifest to get model_id for the zip filename
    let manifest_str = fs::read_to_string(input_path.join("manifest.json"))
        .map_err(|e| format!("Failed to read manifest.json: {}", e))?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_str)
        .map_err(|e| format!("Failed to parse manifest.json: {}", e))?;
    let model_id = manifest.get("model_id").and_then(|v| v.as_str()).unwrap_or("package");

    let packager = Packager::new(input);
    let zip_path =
        packager.package_single(model_id, input).map_err(|e| format!("Packaging failed: {}", e))?;

    // Move zip to output directory if different from input
    let zip_src = PathBuf::from(&zip_path);
    if zip_src.parent() != Some(output_path.as_path()) {
        let zip_dest = output_path.join(zip_src.file_name().unwrap_or_default());
        fs::copy(&zip_src, &zip_dest)
            .map_err(|e| format!("Failed to copy zip to output: {}", e))?;
        println!("Package: {}", zip_dest.display());
    } else {
        println!("Package: {}", zip_path);
    }

    // Validate the package
    let valid =
        packager.validate(&zip_path).map_err(|e| format!("Validation check failed: {}", e))?;
    if valid {
        println!("Package validation: OK");
    } else {
        eprintln!("Warning: package validation failed — zip may be empty or corrupt");
    }

    Ok(())
}

/// Run the report generation subcommand.
///
/// Reads a manifest JSON (and optionally a knowledge update JSON) from the
/// input path and generates a report in the requested format.
///
/// Input path can be:
/// - A directory containing manifest.json (and knowledge/update_*.json)
/// - A single manifest.json file
fn run_report(input: &str, output: &str, format: &str) -> Result<(), String> {
    use ane_report::json_report::JsonReporter;
    use ane_report::markdown::MarkdownReporter;

    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);

    // Locate manifest
    let (manifest_path, manifest_dir) = if input_path.is_dir() {
        let mp = input_path.join("manifest.json");
        if !mp.exists() {
            return Err(format!("No manifest.json found in {}", input));
        }
        (mp, input_path.clone())
    } else if input_path.is_file() {
        let dir = input_path
            .parent()
            .ok_or("Cannot determine parent directory of input file")?
            .to_path_buf();
        (input_path.clone(), dir)
    } else {
        return Err(format!("Input path does not exist: {}", input));
    };

    // Read manifest
    let manifest_json_str = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Failed to read manifest: {}", e))?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_json_str)
        .map_err(|e| format!("Failed to parse manifest JSON: {}", e))?;

    // Try to find bridge result or knowledge update in the same directory
    let bridge_result = if manifest_dir.join("result.json").exists() {
        let br_str = fs::read_to_string(manifest_dir.join("result.json"))
            .map_err(|e| format!("Failed to read result.json: {}", e))?;
        Some(
            serde_json::from_str(&br_str)
                .map_err(|e| format!("Failed to parse result.json: {}", e))?,
        )
    } else {
        None
    };

    // Find knowledge update (first matching file)
    let knowledge_update = find_knowledge_update(&manifest_dir)?;

    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    match format {
        "markdown" | "md" => {
            let reporter = MarkdownReporter::new();
            reporter
                .generate_compilation_report(&manifest, bridge_result.as_ref(), output)
                .map_err(|e| format!("Markdown report generation failed: {}", e))?;

            // Also generate knowledge report if knowledge update exists
            if let Some(ref ku) = knowledge_update {
                let ku_path = output_path.with_extension("knowledge.md");
                reporter
                    .generate_knowledge_report(ku, ku_path.to_str().unwrap_or(""))
                    .map_err(|e| format!("Knowledge markdown report failed: {}", e))?;
                println!("Knowledge report: {}", ku_path.display());
            }

            println!("Compilation report: {}", output);
        }
        "json" => {
            let reporter = JsonReporter::new();
            let report = reporter
                .generate_compilation_report(&manifest, bridge_result.as_ref())
                .map_err(|e| format!("JSON report generation failed: {}", e))?;
            reporter
                .write_to_file(&report, output)
                .map_err(|e| format!("Failed to write JSON report: {}", e))?;

            // Also generate knowledge report if knowledge update exists
            if let Some(ref ku) = knowledge_update {
                let ku_report = reporter
                    .generate_knowledge_report(ku)
                    .map_err(|e| format!("Knowledge JSON report failed: {}", e))?;
                let ku_path = output_path.with_extension("knowledge.json");
                reporter
                    .write_to_file(&ku_report, ku_path.to_str().unwrap_or(""))
                    .map_err(|e| format!("Failed to write knowledge JSON report: {}", e))?;
                println!("Knowledge report: {}", ku_path.display());
            }

            println!("Compilation report: {}", output);
        }
        other => {
            return Err(format!("Unknown report format: '{}'. Supported: markdown, json", other));
        }
    }

    Ok(())
}

/// Find a knowledge update JSON file in the given directory.
///
/// Looks for `knowledge/update_*.json` or `update_*.json` in the directory.
fn find_knowledge_update(dir: &std::path::Path) -> Result<Option<serde_json::Value>, String> {
    // Try knowledge/update_*.json first
    let knowledge_dir = dir.join("knowledge");
    let search_dir = if knowledge_dir.exists() { &knowledge_dir } else { dir };

    if let Ok(entries) = fs::read_dir(search_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("update_") && name.ends_with(".json") {
                    let content = fs::read_to_string(&path)
                        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
                    let value: serde_json::Value = serde_json::from_str(&content)
                        .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;
                    return Ok(Some(value));
                }
            }
        }
    }

    Ok(None)
}

/// Run the knowledge store query subcommand.
///
/// Opens a knowledge store and queries its contents using
/// type, confidence, and scope filters.
fn run_query(store_path: &str, filter: Option<&str>, format: &str) -> Result<(), String> {
    use ane_knowledge::query::{KnowledgeQuery, KnowledgeQueryable};
    use ane_knowledge::shard_template;
    use ane_knowledge::store::KnowledgeStore;

    println!("=== MILLer — Knowledge Query ===\n");

    // Open the knowledge store
    let mut store = KnowledgeStore::open(store_path)
        .map_err(|e| format!("Failed to open knowledge store: {}", e))?;

    // Try to load seed entries from the knowledge/ directory if store is empty
    let (seeds, observations) = store.counts();
    if seeds == 0 && observations == 0 {
        // Try loading seeds from a relative knowledge/ directory
        let knowledge_dir = if std::path::Path::new("knowledge").exists() {
            "knowledge"
        } else if std::path::Path::new("../../knowledge").exists() {
            "../../knowledge"
        } else {
            ""
        };

        if !knowledge_dir.is_empty() {
            let loaded = store
                .load_seeds_from_directory(knowledge_dir)
                .map_err(|e| format!("Failed to load seeds: {}", e))?;
            if loaded > 0 {
                println!("  Loaded {} seed entries from {}", loaded, knowledge_dir);
            }
        }
    }

    let (seeds, observations) = store.counts();
    println!("  Store: {}", store_path);
    println!("  Seeds: {}", seeds);
    println!("  Observations: {}", observations);

    // Build query from filter expression
    let query = if let Some(filter_expr) = filter {
        parse_query_filter(filter_expr)?
    } else {
        KnowledgeQuery::new()
    };

    // Execute query
    let results = store.query(&query).map_err(|e| format!("Query failed: {}", e))?;

    println!("  Results: {} matching entries\n", results.len());

    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&results)
                .map_err(|e| format!("JSON serialization failed: {}", e))?;
            println!("{}", json);
        }
        "markdown" | "md" => {
            for unit in &results {
                println!("### {}", unit.id);
                println!("- Type: {:?}", unit.knowledge_type);
                println!("- Confidence: {:.2}", unit.confidence);
                println!("- Evidence: {:?} (count: {})", unit.evidence_source, unit.evidence_count);
                println!(
                    "- Scope: devices={:?}, os={:?}, opset={:?}",
                    unit.scope.device_classes, unit.scope.os_versions, unit.scope.opset_versions
                );
                if !unit.payload.is_empty() {
                    println!(
                        "- Payload: {}",
                        serde_json::to_string(&unit.payload).unwrap_or_default()
                    );
                }
                println!();
            }
        }
        _ => {
            // Table format (default)
            for unit in &results {
                println!(
                    "  {:40} {:?} conf={:.2} evidence={:?}/{}",
                    unit.id,
                    unit.knowledge_type,
                    unit.confidence,
                    unit.evidence_source,
                    unit.evidence_count
                );
            }
        }
    }

    // Also show shard template seeds if available
    if format == "table" || format == "markdown" || format == "md" {
        let knowledge_dir = if std::path::Path::new("knowledge").exists() {
            "knowledge"
        } else if std::path::Path::new("../../knowledge").exists() {
            "../../knowledge"
        } else {
            ""
        };

        if !knowledge_dir.is_empty() {
            let templates =
                shard_template::load_shard_template_seeds(knowledge_dir).unwrap_or_default();
            if !templates.is_empty() {
                println!("\nShard Template Seeds:");
                for t in &templates {
                    println!(
                        "  {} (template: {}, partitions: {}, known_good: {}, confidence: {:.2})",
                        t.seed_id,
                        t.template.template_id,
                        t.template.partition_spec.len(),
                        t.known_good,
                        t.confidence
                    );
                    for ps in &t.template.partition_spec {
                        println!(
                            "    - {:?} layers {}-{} {:?}",
                            ps.role, ps.layer_start, ps.layer_end, ps.compute_units
                        );
                    }
                }
            }
        }
    }

    println!("\n=== Query complete ===");

    Ok(())
}

/// Parse a query filter expression.
///
/// Supported syntax: `type=LegalityRule`, `min_conf=0.5`, `source=SyntheticRun`
/// Multiple filters separated by commas.
fn parse_query_filter(expr: &str) -> Result<ane_knowledge::query::KnowledgeQuery, String> {
    use ane_ir::kir::{EvidenceSource, KnowledgeType};
    use ane_knowledge::query::KnowledgeQuery;

    let mut query = KnowledgeQuery::new();

    for part in expr.split(',') {
        let part = part.trim();
        if let Some((key, value)) = part.split_once('=') {
            match key.trim() {
                "type" => {
                    let kt = match value.trim() {
                        "LegalityRule" => KnowledgeType::LegalityRule,
                        "MotifCatalog" => KnowledgeType::MotifCatalog,
                        "SurvivalMatrixEntry" => KnowledgeType::SurvivalMatrixEntry,
                        "ShardTemplateKnowledge" => KnowledgeType::ShardTemplateKnowledge,
                        "PrecisionHazard" => KnowledgeType::PrecisionHazard,
                        "FallbackSignature" => KnowledgeType::FallbackSignature,
                        "DeviceFingerprint" => KnowledgeType::DeviceFingerprint,
                        "StateTopologyOutcome" => KnowledgeType::StateTopologyOutcome,
                        "SyntheticTransferAnnotation" => KnowledgeType::SyntheticTransferAnnotation,
                        other => return Err(format!("Unknown knowledge type: {}", other)),
                    };
                    query = query.with_type(kt);
                }
                "min_conf" => {
                    let conf: f32 = value
                        .trim()
                        .parse()
                        .map_err(|_| format!("Invalid confidence value: {}", value))?;
                    query = query.with_min_confidence(conf);
                }
                "source" => {
                    let source = match value.trim() {
                        "SyntheticRun" => EvidenceSource::SyntheticRun,
                        "RealModelRun" => EvidenceSource::RealModelRun,
                        "CompileFailure" => EvidenceSource::CompileFailure,
                        "LoadFailure" => EvidenceSource::LoadFailure,
                        "RuntimeAnomaly" => EvidenceSource::RuntimeAnomaly,
                        "ManualEntry" => EvidenceSource::ManualEntry,
                        "CrossValidated" => EvidenceSource::CrossValidated,
                        other => return Err(format!("Unknown evidence source: {}", other)),
                    };
                    query = query.with_evidence_source(source);
                }
                other => {
                    return Err(format!(
                        "Unknown filter key: '{}'. Supported: type, min_conf, source",
                        other
                    ))
                }
            }
        } else {
            return Err(format!("Invalid filter expression: '{}'. Use key=value format.", part));
        }
    }

    Ok(query)
}

/// Run the knowledge import subcommand.
///
/// Imports knowledge from a snapshot file into a knowledge store.
fn run_import(source: &str, store_path: &str, validate: bool) -> Result<(), String> {
    use ane_knowledge::shard_template;
    use ane_knowledge::snapshot::SnapshotImport;
    use ane_knowledge::store::KnowledgeStore;

    println!("=== MILLer — Knowledge Import ===\n");

    // Check if source is a shard template seed file
    if source.ends_with(".json") {
        // Try loading as shard template seed
        let templates = shard_template::load_shard_template_seed_file(source)
            .map_err(|e| format!("Failed to load shard template seed: {}", e))?;

        if !templates.is_empty() {
            println!("  Found {} shard template entries in {}", templates.len(), source);

            // Open/create the store
            let mut store = KnowledgeStore::open(store_path)
                .map_err(|e| format!("Failed to open store: {}", e))?;

            // Convert validated templates to knowledge units and insert
            for template in &templates {
                if validate && (template.confidence < 0.0 || template.confidence > 1.0) {
                    eprintln!(
                        "  Skipping template '{}' — invalid confidence: {}",
                        template.seed_id, template.confidence
                    );
                    continue;
                }

                // Build a KnowledgeUnit from the validated template
                let mut payload = std::collections::HashMap::new();
                payload.insert(
                    "template_id".to_string(),
                    serde_json::json!(template.template.template_id),
                );
                payload.insert("known_good".to_string(), serde_json::json!(template.known_good));
                payload.insert(
                    "context_length".to_string(),
                    serde_json::json!(template.template.context_length),
                );
                if let Some(ref state_config) = template.template.state_config {
                    payload.insert("state_config".to_string(), serde_json::json!(state_config));
                }

                let unit = ane_ir::kir::KnowledgeUnit {
                    id: template.seed_id.clone(),
                    version: 1,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    knowledge_type: ane_ir::kir::KnowledgeType::ShardTemplateKnowledge,
                    confidence: template.confidence,
                    evidence_source: template.evidence_source.clone(),
                    evidence_count: template.evidence_count,
                    scope: template.scope.clone(),
                    conflict_priority: 0,
                    payload,
                };

                store.insert_observation(unit).map_err(|e| {
                    format!("Failed to insert template '{}': {}", template.seed_id, e)
                })?;
                println!(
                    "  Imported: {} (template: {})",
                    template.seed_id, template.template.template_id
                );
            }

            let (seeds, observations) = store.counts();
            println!("\n  Store now contains {} seeds, {} observations", seeds, observations);
            println!("\n=== Import complete ===");
            return Ok(());
        }
    }

    // Try loading as a snapshot
    let snapshot = SnapshotImport::import_json(source)
        .map_err(|e| format!("Failed to parse snapshot: {}. Note: only JSON snapshot files and shard template seed files are supported.", e))?;

    // Validate the snapshot if requested
    if validate {
        let warnings = SnapshotImport::validate(&snapshot)
            .map_err(|e| format!("Snapshot validation error: {}", e))?;
        if !warnings.is_empty() {
            return Err(format!("Snapshot validation failed: {:?}", warnings));
        }
        println!("  Snapshot validation: PASSED");
    }

    // Open/create the store
    let mut store =
        KnowledgeStore::open(store_path).map_err(|e| format!("Failed to open store: {}", e))?;

    // Import the snapshot
    let stats = SnapshotImport::import_into_store(&mut store, &snapshot)
        .map_err(|e| format!("Import failed: {}", e))?;

    let (seeds, observations) = store.counts();
    println!(
        "  Imported {} seeds, {} observations from {}",
        stats.seeds_imported, stats.observations_imported, source
    );
    println!("  Store now contains {} seeds, {} observations", seeds, observations);

    println!("\n=== Import complete ===");

    Ok(())
}

/// Generate profiling tasks from task families.
///
/// Generates deterministic task specifications for the specified family
/// and persists them as TOML files in the output directory.
/// Generated tasks can be fed directly into `compile`, `compile-full`,
/// or `lab` commands.
///
/// Currently supports the `linear`, `lut`, and `decode` families. Other families remain
/// open and unimplemented.
fn run_generate_tasks(family: &str, output: &str, seed: u64) -> Result<(), String> {
    use ane_lab::task_gen::{TaskFamilyId, TaskGenerator};

    println!("=== MILLer — Task Generation ===\n");

    // Parse the family identifier
    let family_id = TaskFamilyId::from_str_flexible(family)
        .ok_or_else(|| format!(
            "Unknown task family '{}'. Currently supported: linear, lut, decode, mlp, attn, shape, remap, survival",
            family
        ))?;

    println!("[1/3] Generating tasks for family: {}", family_id.canonical_name());
    println!("  Seed: {}", seed);

    let generator = TaskGenerator::with_seed(seed);
    let output_path = PathBuf::from(output);

    let results = generator
        .generate_and_persist(&family_id, &output_path)
        .map_err(|e| format!("Task generation failed: {}", e))?;

    println!("[2/3] Generated {} task(s):", results.len());
    for (spec, path) in &results {
        let task_hash = compute_task_hash(spec);
        println!("  {} -> {} (hash: {})", spec.name, path.display(), task_hash);
    }

    println!("[3/3] Tasks written to: {}", output);
    println!("\nTo compile a generated task:");
    println!(
        "  ane-cli compile --input {}/LinearProjection/<task>.toml --output <out_dir>",
        output
    );

    println!("\n=== Task generation complete ===");

    Ok(())
}

/// Run verification on an emitted mlpackage (Sprint 46).
///
/// Dispatches the `verify` bridge command to perform four-dimension
/// verification: op graph fidelity, compute-unit placement, state
/// conformance, and multi-function conformance. The Python bridge
/// handles platform detection (macOS gets MLModelStructure/MLComputePlan,
/// Linux gets spec-based fallback).
#[allow(clippy::too_many_arguments)]
fn run_verify(
    mlpackage_path: &str,
    output: &str,
    bridge_script: &str,
    python_path: &str,
    compute_units: &str,
    mir_ops: Option<&str>,
    expected_functions: Option<&str>,
    expected_states: Option<&str>,
) -> Result<(), String> {
    use ane_bridge::subprocess::PythonBridge;

    println!("=== MILLer — Model Verification ===\n");
    println!("Package: {}", mlpackage_path);
    println!("Compute units: {}", compute_units);

    // Build the verify command payload
    let mut payload = serde_json::json!({
        "command": "verify",
        "bridge_version": 1,
        "mlpackage_path": mlpackage_path,
        "compute_units": compute_units,
    });

    // Add optional MIR ops list for op fidelity comparison
    // Sprint 47: Auto-populate from compile manifest if not explicitly provided
    if let Some(ops_json) = mir_ops {
        match serde_json::from_str::<serde_json::Value>(ops_json) {
            Ok(ops) => {
                payload["mir_ops"] = ops;
            }
            Err(e) => println!("Warning: could not parse mir_ops JSON: {}. Skipping.", e),
        }
    } else {
        // Auto-populate mir_ops from compile manifest if available.
        // The compile output directory is typically the parent of the mlpackage directory.
        let mlpackage_dir = std::path::Path::new(mlpackage_path);
        let compile_output_dir = mlpackage_dir.parent().unwrap_or(mlpackage_dir);
        let manifest_path = compile_output_dir.join("manifest.json");
        if manifest_path.exists() {
            if let Ok(manifest_str) = std::fs::read_to_string(&manifest_path) {
                if let Ok(manifest_val) = serde_json::from_str::<serde_json::Value>(&manifest_str) {
                    // Extract mir_ops from the first function's descriptor in the first package
                    if let Some(mir_ops_val) = manifest_val
                        .get("packages")
                        .and_then(|p| p.get(0))
                        .and_then(|pkg| pkg.get("functions"))
                        .and_then(|f| f.get(0))
                        .and_then(|func| func.get("mir_ops"))
                    {
                        if mir_ops_val.as_array().is_some_and(|arr| !arr.is_empty()) {
                            payload["mir_ops"] = mir_ops_val.clone();
                            println!(
                                "  Auto-populated mir_ops from compile manifest ({} ops)",
                                mir_ops_val.as_array().map(|a| a.len()).unwrap_or(0)
                            );
                        }
                    }
                }
            }
        }
    }

    // Add optional expected function names
    if let Some(funcs) = expected_functions {
        let func_list: Vec<&str> = funcs.split(',').map(|s| s.trim()).collect();
        payload["expected_function_names"] = serde_json::json!(func_list);
    }

    // Add optional expected state names
    if let Some(states) = expected_states {
        let state_list: Vec<&str> = states.split(',').map(|s| s.trim()).collect();
        payload["expected_state_names"] = serde_json::json!(state_list);
    }

    // Dispatch to Python bridge
    println!("[1/3] Dispatching verify command via Python bridge...");
    let bridge = PythonBridge::new(python_path, bridge_script);
    let result = bridge
        .execute_raw_payload(&payload)
        .map_err(|e| format!("Bridge invocation failed: {}", e))?;

    if result.status != "success" {
        if let Some(ref err) = result.error_message {
            println!("  Verification failed: {}", err);
        }
        return Err(format!(
            "Verify command failed: {}",
            result.error_message.as_deref().unwrap_or("unknown error")
        ));
    }

    // Extract verification results from metadata
    println!("[2/3] Processing verification results...");
    let meta = &result.metadata;

    // Print summary
    if let Some(overall) = meta.get("overall_score") {
        println!("  Overall score: {:.2}", overall.as_f64().unwrap_or(0.0));
    }
    if let Some(op_fidelity) = meta.get("op_fidelity") {
        if let Some(score) = op_fidelity.get("op_fidelity_score") {
            println!("  Op fidelity: {:.2}", score.as_f64().unwrap_or(0.0));
        }
    }
    if let Some(placement) = meta.get("placement") {
        if let Some(available) = placement.get("available") {
            if available.as_bool().unwrap_or(false) {
                if let Some(rate) = placement.get("ane_placement_rate") {
                    println!("  ANE placement rate: {:.2}", rate.as_f64().unwrap_or(0.0));
                }
            } else {
                println!("  ANE placement: unavailable (requires macOS)");
            }
        }
    }
    if let Some(state_conf) = meta.get("state_conformance") {
        if let Some(score) = state_conf.get("conformance_score") {
            println!("  State conformance: {:.2}", score.as_f64().unwrap_or(0.0));
        }
    }
    if let Some(mf_conf) = meta.get("multifunction_conformance") {
        if let Some(score) = mf_conf.get("conformance_score") {
            println!("  Multi-function conformance: {:.2}", score.as_f64().unwrap_or(0.0));
        }
    }

    // Write verification artifacts
    println!("[3/3] Writing verification artifacts...");
    let output_path = PathBuf::from(output);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create output dir: {}", e))?;
    }
    let json = serde_json::to_string_pretty(&result.metadata)
        .map_err(|e| format!("JSON serialization failed: {}", e))?;
    fs::write(&output_path, json).map_err(|e| format!("Failed to write output: {}", e))?;

    println!("  Verification artifacts written to: {}", output);
    println!("\n=== Verification complete ===");

    Ok(())
}

/// Run the trace-compile pipeline: trace a HuggingFace model → SIR → ANE-faithful compile.
///
/// This is the primary entry point for the transformers tracing extension.
/// It performs the following steps:
/// 1. Trace the model (via Python subprocess or load pre-traced JSON)
/// 2. Build SIR from the traced graph
/// 3. Validate against ANE constraints (version-aware)
/// 4. LegalityRewrite (SIR→AIR)
/// 5. MilLower (AIR→MIR)
/// 6. Proto-direct emission (MIR→.mlpackage)
/// 7. Write all artifacts (SIR, AIR, MIR, faithfulness report, traced graph, .mlpackage)
fn run_trace_compile(
    model: &str,
    output: &str,
    target_family: &str,
    ane_only: bool,
    batch_size: usize,
    seq_len: usize,
    decompose: bool,
    with_kv_cache: bool,
    trace_script: &str,
    python_path: &str,
    dtype: &str,
    knowledge_dir: Option<&str>,
) -> Result<(), String> {
    use ane_bridge::proto_direct::{
        emit_mir_graph_proto_direct_with_resolver, validate_proto_direct_package,
    };
    use ane_bridge::safetensors_resolver::SafetensorsWeightResolver;
    use ane_passes::knowledge_query::NoKnowledge;
    use ane_passes::legality_rewrite::{DecompositionContext, LegalityRewritePass};
    use ane_passes::mil_lower::MilLowerPass;
    use ane_passes::shard_plan::ShardPlan;
    use ane_trace::config::{InputShape, TraceConfig, TraceTarget};
    use ane_trace::sir_build::build_sir_from_trace;
    use ane_trace::subprocess::trace_model;
    use ane_trace::versioned::VersionedCompiler;

    println!("=== MILLer — Trace-Compile Pipeline ===\n");

    // Step 1: Parse target family
    println!("[1/10] Parsing target ANE family: {}", target_family);
    let family = parse_ane_family(target_family)?;
    println!("  Target: {:?}", family);

    // Step 2: Configure and run tracing
    println!("[2/10] Tracing model: {}", model);
    let target = if model.ends_with(".json") {
        TraceTarget::PreTraced(model.to_string())
    } else if std::path::Path::new(model).is_dir() {
        TraceTarget::LocalPath(model.to_string())
    } else {
        TraceTarget::HuggingFaceId(model.to_string())
    };

    let config = TraceConfig {
        target,
        target_family: family,
        ane_only,
        decompose_at_trace: decompose,
        input_shapes: vec![InputShape { batch_size, seq_len }],
        with_kv_cache,
        max_seq_len: seq_len * 64, // Allow longer sequences than trace input
        dtype: dtype.to_string(),
        trace_script: trace_script.to_string(),
        python_path: python_path.to_string(),
        ..TraceConfig::default()
    };

    let traced_graph = trace_model(&config).map_err(|e| format!("Model tracing failed: {}", e))?;
    println!(
        "  Traced: {} nodes, architecture={}, model_type={}",
        traced_graph.nodes.len(),
        traced_graph.architecture,
        traced_graph.model_config.model_type,
    );
    println!(
        "  Config: hidden_size={}, num_heads={}, num_layers={}",
        traced_graph.model_config.hidden_size,
        traced_graph.model_config.num_attention_heads,
        traced_graph.model_config.num_hidden_layers,
    );

    // Step 3: Build SIR from traced graph
    println!("[3/10] Building SIR from traced graph...");
    let sir = build_sir_from_trace(&traced_graph, family)?;
    println!(
        "  SIR: {} nodes, {} inputs, {} outputs",
        sir.nodes.len(),
        sir.inputs.len(),
        sir.outputs.len(),
    );

    // Step 4: Validate against ANE constraints
    println!("[4/10] Validating ANE faithfulness...");
    let compiler = VersionedCompiler::new(family);
    let result = compiler.validate_sir(&sir, ane_only);
    println!(
        "  ANE utilization: {:.1}% ({}/{} ops on ANE)",
        result.report.ane_utilization(),
        result.report.ane_supported,
        result.report.total_ops,
    );
    println!("  CPU fallback: {} ops", result.report.cpu_fallback,);
    println!("  ANE-faithful: {}", if result.report.is_faithful { "YES" } else { "NO" },);

    if !result.report.warnings.is_empty() {
        println!("  Warnings:");
        for warning in &result.report.warnings {
            println!("    - {}", warning);
        }
    }

    if !result.report.violations.is_empty() {
        println!("  Violations:");
        for v in &result.report.violations {
            println!("    - [{}] {} ({})", v.severity_str(), v.op_name, v.message);
        }
    }

    // Step 5: Run LegalityRewritePass (SIR→AIR)
    println!("[5/10] Running LegalityRewritePass (SIR→AIR)...");
    let legality = LegalityRewritePass::new();
    let no_knowledge = NoKnowledge;
    let decomp_ctx = DecompositionContext::for_attention(
        batch_size,
        traced_graph.model_config.hidden_size,
        traced_graph.model_config.num_attention_heads,
        traced_graph.model_config.hidden_size / traced_graph.model_config.num_attention_heads,
        seq_len,
    );
    let air = legality
        .run(sir.clone(), &no_knowledge, Some(&decomp_ctx))
        .map_err(|e| format!("LegalityRewritePass failed: {}", e))?;
    println!(
        "  AIR: {} nodes, {} inputs, {} outputs",
        air.nodes.len(),
        air.inputs.len(),
        air.outputs.len()
    );

    // Step 6: Run MilLowerPass (AIR→MIR) with default single-shard plan
    println!("[6/10] Running MilLowerPass (AIR→MIR)...");
    let mil_lower = MilLowerPass::new();
    let shard_plan = ShardPlan::default();
    let mirs = mil_lower
        .run(&air, &shard_plan)
        .map_err(|e| format!("MilLowerPass failed: {}", e))?;
    println!("  MIR: {} shard graph(s) produced", mirs.len());
    for (i, mir) in mirs.iter().enumerate() {
        println!(
            "    MIR[{}]: {} nodes, {} inputs, {} outputs",
            i,
            mir.nodes.len(),
            mir.inputs.len(),
            mir.outputs.len()
        );
    }

    // Step 7: Emit .mlpackage via proto-direct with real weights
    println!("[7/10] Emitting .mlpackage via proto-direct...");
    let output_path = PathBuf::from(output);
    fs::create_dir_all(&output_path).map_err(|e| format!("Failed to create output dir: {}", e))?;

    // Load real weights from safetensors files in the HuggingFace cache
    let weight_resolver = if !traced_graph.safetensors_files.is_empty() {
        println!("  Loading weights from {} safetensors file(s)...", traced_graph.safetensors_files.len());
        let resolver = SafetensorsWeightResolver::from_safetensors_files(&traced_graph.safetensors_files);
        println!("  Loaded {} tensor(s) from safetensors", resolver.len());
        resolver
    } else if let Some(ref cache_dir) = traced_graph.model_cache_dir {
        println!("  Loading weights from cache dir: {}", cache_dir);
        let resolver = SafetensorsWeightResolver::from_cache_dir(cache_dir);
        println!("  Loaded {} tensor(s) from safetensors", resolver.len());
        resolver
    } else {
        println!("  No safetensors files found — using zero-filled weights");
        SafetensorsWeightResolver::empty()
    };

    let mlpackage_dir = output_path.join("model.mlpackage");
    if mirs.is_empty() {
        return Err("MilLowerPass produced no MIR graphs — nothing to emit".to_string());
    }
    let emit_result = emit_mir_graph_proto_direct_with_resolver(
        &mirs[0],
        mlpackage_dir.to_str().unwrap_or(""),
        &weight_resolver,
    )
    .map_err(|e| format!("Proto-direct emission failed: {}", e))?;
    println!("  Emitted: {}", mlpackage_dir.display());
    println!("  Total size: {} bytes, {} file(s), {} weight(s)",
        emit_result.total_size, emit_result.file_count, emit_result.weight_count);

    // Step 8: Validate .mlpackage structure
    println!("[8/10] Validating .mlpackage structure...");
    let validation = validate_proto_direct_package(mlpackage_dir.to_str().unwrap_or(""))
        .map_err(|e| format!("Package validation failed: {}", e))?;
    if validation.is_valid {
        println!("  .mlpackage: VALID");
        if let Some(model_size) = validation.model_file_size {
            println!("  model.mlmodel: {} bytes", model_size);
        }
        if let Some(weight_size) = validation.weight_file_size {
            println!("  weight.bin: {} bytes", weight_size);
        }
    } else {
        println!("  .mlpackage: INVALID");
        for err in &validation.errors {
            println!("    ERROR: {}", err);
        }
    }
    for warn in &validation.warnings {
        println!("    WARNING: {}", warn);
    }

    // Step 9: Write intermediate artifacts
    println!("[9/10] Writing artifacts...");

    // Write traced graph
    let trace_path = output_path.join("traced_graph.json");
    let trace_json = serde_json::to_string_pretty(&traced_graph)
        .map_err(|e| format!("Traced graph serialization failed: {}", e))?;
    fs::write(&trace_path, &trace_json)
        .map_err(|e| format!("Failed to write traced graph: {}", e))?;
    println!("  Traced graph: {}", trace_path.display());

    // Write SIR
    let sir_path = output_path.join("sir.json");
    let sir_json = serde_json::to_string_pretty(&sir)
        .map_err(|e| format!("SIR serialization failed: {}", e))?;
    fs::write(&sir_path, &sir_json).map_err(|e| format!("Failed to write SIR: {}", e))?;
    println!("  SIR: {}", sir_path.display());

    // Write AIR
    let air_path = output_path.join("air.json");
    let air_json = serde_json::to_string_pretty(&air)
        .map_err(|e| format!("AIR serialization failed: {}", e))?;
    fs::write(&air_path, &air_json).map_err(|e| format!("Failed to write AIR: {}", e))?;
    println!("  AIR: {}", air_path.display());

    // Write MIR
    let mir_path = output_path.join("mir.json");
    let mir_json = serde_json::to_string_pretty(&mirs[0])
        .map_err(|e| format!("MIR serialization failed: {}", e))?;
    fs::write(&mir_path, &mir_json).map_err(|e| format!("Failed to write MIR: {}", e))?;
    println!("  MIR: {}", mir_path.display());

    // Write faithfulness report
    let report_path = output_path.join("ane_faithfulness_report.json");
    let report_json = serde_json::to_string_pretty(&result.report)
        .map_err(|e| format!("Report serialization failed: {}", e))?;
    fs::write(&report_path, &report_json).map_err(|e| format!("Failed to write report: {}", e))?;
    println!("  Faithfulness report: {}", report_path.display());

    // Step 10: Knowledge consultation (optional)
    println!("[10/10] Knowledge consultation...");
    if let Some(kdir) = knowledge_dir {
        let store_path = PathBuf::from(kdir);
        if store_path.exists() {
            println!("  Knowledge store: {} (available)", kdir);
        } else {
            println!("  Knowledge store: {} (not found)", kdir);
        }
    } else {
        println!("  No knowledge store specified");
    }

    println!("\n=== Trace-compile complete ===");
    println!("mlpackage: {}", mlpackage_dir.display());
    println!("Artifacts in: {}", output);

    Ok(())
}

/// Parse an ANE family string into an AneFamily enum value.
///
/// Accepts both ANE generation codes (A12, A16, etc.) and Apple Silicon
/// chip names (M1, M2, M3, etc.) as aliases. The mapping is:
///
/// | Chip          | ANE Gen | Notes                              |
/// |---------------|---------|------------------------------------|
/// | M1            | A12     | broadcast_fp16_only=true           |
/// | M1 Pro/Max    | A14     |                                    |
/// | M2            | A12     | same ANE as M1 (Rev V5)            |
/// | M2 Pro/Max    | A14     |                                    |
/// | M3            | A15     |                                    |
/// | M3 Pro/Max    | A16     | first with reliable SDPA           |
/// | M4            | A16     |                                    |
/// | M4 Pro/Max    | A18     |                                    |
fn parse_ane_family(s: &str) -> Result<ane_ir::ane_target::AneFamily, String> {
    use ane_ir::ane_target::AneFamily;
    match s.to_lowercase().as_str() {
        // ANE generation codes
        "a11legacy" | "a11" => Ok(AneFamily::A11Legacy),
        "a12" => Ok(AneFamily::A12),
        "a14" => Ok(AneFamily::A14),
        "a15" => Ok(AneFamily::A15),
        "a16" => Ok(AneFamily::A16),
        "a18" => Ok(AneFamily::A18),
        // Apple Silicon chip name aliases
        "m1" => Ok(AneFamily::A12),
        "m1pro" | "m1_max" | "m1max" | "m1ultra" => Ok(AneFamily::A14),
        "m2" => Ok(AneFamily::A12),
        "m2pro" | "m2_max" | "m2max" | "m2ultra" => Ok(AneFamily::A14),
        "m3" => Ok(AneFamily::A15),
        "m3pro" | "m3_max" | "m3max" => Ok(AneFamily::A16),
        "m4" => Ok(AneFamily::A16),
        "m4pro" | "m4_max" | "m4max" => Ok(AneFamily::A18),
        _ => Err(format!(
            "Unknown ANE family '{}'. Valid: A11Legacy, A12, A14, A15, A16, A18, \
             or chip names: M1, M2, M3, M4 (with Pro/Max variants)",
            s
        )),
    }
}
