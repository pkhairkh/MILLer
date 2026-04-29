//! Core ML FFI Bindings (Sprint 41)
//!
//! This crate provides Rust FFI bindings for the Core ML C API,
//! enabling future on-device model compilation, loading, and prediction
//! without going through Python at all.
//!
//! ## Strategic Purpose
//!
//! The current compilation path is:
//!
//! ```text
//! Rust MIR → Python bridge → coremltools → .mlpackage on disk
//! ```
//!
//! The long-term target path is:
//!
//! ```text
//! Rust MIR → Core ML C API → .mlpackage on disk
//!               ↕
//!         (on-device predict)
//! ```
//!
//! This crate defines the FFI interface that will enable this path.
//! The actual implementation requires macOS with the Core ML framework,
//! which is not available in the current environment.
//!
//! ## Core ML C API
//!
//! Apple provides a C API for Core ML that includes:
//! - `MLModel` loading and compilation
//! - `MLModel` prediction (inference)
//! - `MLModel` metadata inspection
//! - `MLComputePlan` inspection
//! - `MLModelStructure` structural introspection
//!
//! The C API headers are available at:
//! `/System/Library/Frameworks/CoreML.framework/Headers/`
//!
//! ## Build
//!
//! This crate compiles on all platforms but the FFI functions are only
//! available on macOS with Core ML framework installed. On other platforms,
//! the functions return "unavailable" errors.
//!
//! **Note**: Rust toolchain is not available in the current environment,
//! so this crate cannot be compiled here.

pub mod api;
pub mod capi;
pub mod error;
pub mod model;

pub use api::CoreMlApi;
pub use capi::{
    CoreMlCompileResult, CoreMlModelHandle, CoreMlModelInfo, CoreMlPredictResult, CoreMlStatus,
};
pub use error::FfiError;
pub use model::FfiModel;
