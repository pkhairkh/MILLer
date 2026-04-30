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
pub mod subprocess;
