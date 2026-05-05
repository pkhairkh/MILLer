//! ANE Compiler Bridge
//!
//! Bridge to the Core ML emission backends, supporting both:
//! - **Python subprocess** — shells out to `python/bridge.py` via coremltools
//! - **Proto-direct** — native Rust emission via `ane-coreml-emit` (Sprint 41)
//!
//! The proto-direct path bypasses the Python subprocess entirely, producing
//! `.mlpackage` artifacts directly from Rust. This enables true weight sharing
//! across function boundaries (which coremltools 9.0 cannot do) and eliminates
//! the Python dependency for emission.

pub mod mir_to_compat;
pub mod proto_direct;
pub mod safetensors_resolver;
pub mod shape_inference;
pub mod static_table_resolver;
pub mod subprocess;

/// Errors produced by the bridge layer during MIR-to-compat conversion
/// and proto-direct emission.
///
/// T-P2-05: These typed error variants enable programmatic error handling
/// by callers — they can match on specific error kinds rather than parsing
/// string messages from `anyhow::Error`.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    /// A weight referenced by a MIR op could not be resolved.
    ///
    /// This is a hard error when `allow_missing_weights` is false (the default).
    /// When `allow_missing_weights` is true, missing weights produce zero-filled
    /// placeholders with a warning instead.
    #[error("Weight '{path}' not found in resolver. This produces a silently broken model \
             with zero-filled weights. Use --allow-missing-weights to opt into zero-fill \
             (NOT recommended for production).")]
    UnresolvedWeight {
        /// The weight path that was not found (e.g., "model.layers.0.self_attn.q_proj.weight").
        path: String,
    },
}
