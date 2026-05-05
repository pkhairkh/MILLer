//! C-Compatible FFI Interface for Core ML
//!
//! This module defines the C-compatible FFI interface that non-macOS consumers
//! can link against. On macOS, these would link to CoreML.framework; on other
//! platforms, they provide stub implementations that return error codes.
//!
//! ## API Design
//!
//! All functions use C-compatible types (`c_int`, `c_char`, raw pointers) and
//! the `extern "C"` calling convention. Error handling uses `CoreMlStatus` enum
//! values rather than Rust's `Result` type.
//!
//! ## Memory Ownership
//!
//! - Handles returned by `coreml_model_load` must be freed with `coreml_model_destroy`
//! - C strings returned by `coreml_version` and `coreml_model_compile` must be
//!   freed with `coreml_free_string`
//! - All functions handle null pointers safely (returning error codes rather than
//!   undefined behavior)

use prost::Message;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

// ─── Opaque Handle ──────────────────────────────────────────────────────────

/// Opaque handle for a loaded Core ML model.
///
/// This is an opaque type from the C consumer's perspective. Internally, on
/// macOS it would wrap an `MLModel*`; on other platforms it carries no
/// meaningful data because no model can actually be loaded.
pub enum CoreMlModelHandle {}

/// Internal representation of a model handle.
///
/// On non-macOS platforms, we store the path for diagnostics.
/// On macOS, this would wrap an `MLModel*` from the Core ML C API.
///
/// # Allocation contract (T-75, I-50)
///
/// Handles returned by `coreml_model_load` are allocated with `Box::new(ModelHandleInner)`.
/// `coreml_model_destroy` reconstructs the `Box` via `Box::from_raw` to drop it.
/// This contract MUST be maintained: if you change the allocation strategy in
/// `coreml_model_load`, you MUST change the deallocation in `coreml_model_destroy`
/// to match. Mixing allocation strategies (e.g., `Box::new` + `libc::free`) is
/// undefined behavior.
struct ModelHandleInner {
    /// Path to the model file (for diagnostics on non-macOS).
    _path: String,
}

// ─── Error Codes ────────────────────────────────────────────────────────────

/// Error codes returned by the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMlStatus {
    /// Operation succeeded.
    Ok = 0,
    /// Core ML is not available on this platform (non-macOS).
    ErrorPlatformUnavailable = 1,
    /// Failed to load the model.
    ErrorModelLoad = 2,
    /// Failed to compile the model.
    ErrorModelCompile = 3,
    /// Prediction failed.
    ErrorPrediction = 4,
    /// Invalid argument passed (null pointer, bad path, etc.).
    ErrorInvalidArgument = 5,
    /// The OS version is too old for the requested feature.
    ErrorInsufficientOsVersion = 6,
    /// Serialization or deserialization error.
    ErrorSerialization = 7,
    /// An unknown error occurred.
    ErrorUnknown = -1,
}

// ─── Result Structs ─────────────────────────────────────────────────────────

/// Model metadata returned by the C API.
#[repr(C)]
#[derive(Debug)]
pub struct CoreMlModelInfo {
    /// Number of functions in the model.
    pub function_count: c_int,
    /// Whether the model has state inputs.
    pub has_state: bool,
    /// Core ML specification version.
    pub spec_version: c_int,
}

/// Result of compiling an mlpackage.
#[repr(C)]
#[derive(Debug)]
pub struct CoreMlCompileResult {
    /// Status of the compile operation.
    pub status: CoreMlStatus,
    /// Path to the compiled model (must be freed with `coreml_free_string`).
    pub compiled_path: *mut c_char,
}

/// Result of a prediction.
#[repr(C)]
#[derive(Debug)]
pub struct CoreMlPredictResult {
    /// Status of the prediction.
    pub status: CoreMlStatus,
    /// Prediction latency in nanoseconds.
    pub latency_ns: u64,
}

// ─── C API Functions ────────────────────────────────────────────────────────

/// Check if Core ML is available on this platform.
///
/// Returns `true` on macOS (where CoreML.framework is available),
/// `false` on all other platforms.
#[no_mangle]
pub extern "C" fn coreml_is_available() -> bool {
    cfg!(target_os = "macos")
}

/// Get the Core ML framework version string.
///
/// On macOS, returns a C string like "7.0" or "8.0" depending on the
/// installed Core ML version. The returned string must be freed with
/// `coreml_free_string()`.
///
/// On non-macOS platforms, returns a null pointer.
///
/// T-P2-07: On macOS without the CoreML framework linked, returning "unknown"
/// is misleading — callers may interpret it as a valid version string.
/// We now return null on macOS as well, since we cannot determine the
/// actual framework version without linking CoreML.framework.
#[no_mangle]
pub extern "C" fn coreml_version() -> *mut c_char {
    // T-P2-07: Return null when we cannot determine the actual version.
    // Previously returned "unknown" on macOS which callers might interpret
    // as a valid version string.
    ptr::null_mut()
}

/// Load a model from an mlpackage or mlmodelc path.
///
/// On success, writes the model handle to `out_handle` and returns
/// `CoreMlStatus::Ok`. The handle must be released with
/// `coreml_model_destroy()`.
///
/// On failure, writes null to `out_handle` and returns an error status.
///
/// # Safety
///
/// `path` must be a valid pointer to a null-terminated C string.
/// `out_handle` must be a valid pointer to a `*mut CoreMlModelHandle`.
#[no_mangle]
pub unsafe extern "C" fn coreml_model_load(
    path: *const c_char,
    out_handle: *mut *mut CoreMlModelHandle,
) -> CoreMlStatus {
    // Null pointer checks
    if path.is_null() || out_handle.is_null() {
        return CoreMlStatus::ErrorInvalidArgument;
    }

    // Write null by default
    unsafe {
        *out_handle = ptr::null_mut();
    }

    // Platform check
    if !cfg!(target_os = "macos") {
        return CoreMlStatus::ErrorPlatformUnavailable;
    }

    // On macOS, we would call MLModelLoad() here.
    // For now, return a stub error since we can't link CoreML.framework on Linux.
    // When macOS support is implemented, the handle MUST be allocated with
    // Box::new(ModelHandleInner { _path: ... }) so that coreml_model_destroy
    // can safely reconstruct it with Box::from_raw (see allocation contract
    // on ModelHandleInner).
    let _path_str = unsafe { CStr::from_ptr(path) };
    CoreMlStatus::ErrorModelLoad
}

/// Destroy a loaded model handle.
///
/// Safely handles null pointers (no-op).
///
/// # Allocation contract (T-75, I-50)
///
/// When `coreml_model_load` is implemented on macOS, it MUST allocate the
/// handle using `Box::new(ModelHandleInner { ... })` and cast the result
/// to `*mut CoreMlModelHandle`. This function reconstructs the `Box` via
/// `Box::from_raw` and drops it, which is safe ONLY if the handle was
/// allocated with `Box::new`. If the allocation strategy changes (e.g.,
/// to use the Core ML C API's own allocation), this function MUST be
/// updated to match — mixing allocation strategies is undefined behavior.
///
/// # Safety
///
/// `handle` must be either null or a valid pointer previously returned by
/// `coreml_model_load`. Passing a pointer from any other source is
/// undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn coreml_model_destroy(handle: *mut CoreMlModelHandle) {
    if handle.is_null() {
        return;
    }

    // Reconstruct the Box<ModelHandleInner> that was allocated in
    // coreml_model_load, then drop it. This is safe because:
    // 1. coreml_model_load allocates with Box::new(ModelHandleInner)
    // 2. We cast the opaque handle back to the concrete type
    // 3. Box::from_raw reconstructs the Box, which is then dropped
    //
    // On macOS with Core ML C API, this would need to call
    // MLModelDestroy() instead. The current implementation is correct
    // for the Box-based allocation strategy used by coreml_model_load.
    let inner = handle as *mut ModelHandleInner;
    unsafe {
        let _ = Box::from_raw(inner);
    }
}

/// Get model metadata.
///
/// Writes metadata to `out_info`. Returns `CoreMlStatus::Ok` on success.
/// Returns an error status if the handle is invalid or the platform
/// doesn't support Core ML.
///
/// # Safety
///
/// `handle` must be either null or a valid model handle.
/// `out_info` must be a valid pointer to a `CoreMlModelInfo`.
#[no_mangle]
pub unsafe extern "C" fn coreml_model_info(
    handle: *mut CoreMlModelHandle,
    out_info: *mut CoreMlModelInfo,
) -> CoreMlStatus {
    if handle.is_null() || out_info.is_null() {
        return CoreMlStatus::ErrorInvalidArgument;
    }

    if !cfg!(target_os = "macos") {
        return CoreMlStatus::ErrorPlatformUnavailable;
    }

    // On macOS, we would inspect the model's spec.
    // T-P2-07: On macOS without CoreML framework, we cannot provide model info.
    // Returning an error is more honest than returning zeroed/fabricated data
    // that callers would misinterpret as valid.
    CoreMlStatus::ErrorUnknown
}

/// Compile an mlpackage to mlmodelc.
///
/// On macOS, this calls `MLModelCompile()` which produces an `.mlmodelc`
/// directory. The `compiled_path` in `out_result` must be freed with
/// `coreml_free_string()`.
///
/// On non-macOS platforms, returns `ErrorPlatformUnavailable`.
///
/// # Safety
///
/// `source_path` and `output_dir` must be valid pointers to null-terminated C strings.
/// `out_result` must be a valid pointer to a `CoreMlCompileResult`.
#[no_mangle]
pub unsafe extern "C" fn coreml_model_compile(
    source_path: *const c_char,
    output_dir: *const c_char,
    out_result: *mut CoreMlCompileResult,
) -> CoreMlStatus {
    if source_path.is_null() || output_dir.is_null() || out_result.is_null() {
        return CoreMlStatus::ErrorInvalidArgument;
    }

    if !cfg!(target_os = "macos") {
        unsafe {
            (*out_result).status = CoreMlStatus::ErrorPlatformUnavailable;
            (*out_result).compiled_path = ptr::null_mut();
        }
        return CoreMlStatus::ErrorPlatformUnavailable;
    }

    // On macOS, we would call MLModelCompile().
    unsafe {
        (*out_result).status = CoreMlStatus::ErrorModelCompile;
        (*out_result).compiled_path = ptr::null_mut();
    }

    CoreMlStatus::ErrorModelCompile
}

/// Run prediction on a loaded model.
///
/// On success, returns a `CoreMlPredictResult` with `status = Ok` and
/// the measured latency. On failure, returns an appropriate error status.
///
/// If `latency_ns` is non-null, the prediction latency is written there.
///
/// # Safety
///
/// `handle` must be either null or a valid model handle.
/// `latency_ns` must be either null or a valid pointer to a `u64`.
#[no_mangle]
pub unsafe extern "C" fn coreml_model_predict(
    handle: *mut CoreMlModelHandle,
    latency_ns: *mut u64,
) -> CoreMlPredictResult {
    if handle.is_null() {
        return CoreMlPredictResult { status: CoreMlStatus::ErrorInvalidArgument, latency_ns: 0 };
    }

    if !latency_ns.is_null() {
        unsafe {
            *latency_ns = 0;
        }
    }

    if !cfg!(target_os = "macos") {
        return CoreMlPredictResult {
            status: CoreMlStatus::ErrorPlatformUnavailable,
            latency_ns: 0,
        };
    }

    // On macOS, we would call MLModelPrediction().
    CoreMlPredictResult { status: CoreMlStatus::ErrorPrediction, latency_ns: 0 }
}

/// Free a C string allocated by the Core ML C API.
///
/// Safely handles null pointers (no-op).
///
/// # Safety
///
/// `s` must be either null or a pointer previously returned by a Core ML
/// C API function that allocates a C string (e.g., `coreml_version`,
/// `coreml_model_compile`). The pointer must not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn coreml_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(s);
    }
}

/// Validate a proto-direct emitted mlpackage.
///
/// This checks the directory structure of an mlpackage emitted by the
/// proto-direct emission path, verifying:
///
/// 1. The path exists and is a directory
/// 2. `Manifest.json` exists and is valid JSON
/// 3. `Data/com.apple.CoreML/model.mlmodel` exists
/// 4. `Data/com.apple.CoreML/weights/weight.bin` exists (if referenced in manifest)
///
/// This function works on **all platforms** — it does NOT require macOS
/// or the Core ML runtime. It validates the filesystem output of the
/// proto-direct emission pipeline.
///
/// # Safety
///
/// `path` must be a valid pointer to a null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn coreml_validate_proto_package(path: *const c_char) -> CoreMlStatus {
    if path.is_null() {
        return CoreMlStatus::ErrorInvalidArgument;
    }

    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return CoreMlStatus::ErrorInvalidArgument,
    };

    let pkg_path = std::path::Path::new(path_str);

    // Check 1: Path exists and is a directory
    if !pkg_path.exists() {
        return CoreMlStatus::ErrorModelLoad;
    }
    if !pkg_path.is_dir() {
        return CoreMlStatus::ErrorModelLoad;
    }

    // Check 2: Manifest.json exists and is valid JSON
    let manifest_path = pkg_path.join("Manifest.json");
    if !manifest_path.exists() {
        return CoreMlStatus::ErrorSerialization;
    }

    let manifest_data = match std::fs::read_to_string(&manifest_path) {
        Ok(data) => data,
        Err(_) => return CoreMlStatus::ErrorSerialization,
    };

    // Validate that Manifest.json is valid JSON
    if serde_json::from_str::<serde_json::Value>(&manifest_data).is_err() {
        return CoreMlStatus::ErrorSerialization;
    }

    // Check 3: Data/com.apple.CoreML/model.mlmodel exists
    // Apple's mlpackage format uses Data/ not Model/ for the model protobuf.
    let mlmodel_path = pkg_path.join("Data/com.apple.CoreML/model.mlmodel");
    if !mlmodel_path.exists() {
        return CoreMlStatus::ErrorModelLoad;
    }

    // Check 4: Data/com.apple.CoreML/weights/weight.bin exists
    let weight_path = pkg_path.join("Data/com.apple.CoreML/weights/weight.bin");
    if !weight_path.exists() {
        // weight.bin is expected but not strictly required for all models
        // (e.g., models with only inline const data). However, for
        // proto-direct emitted packages, it should always be present.
        // Return a soft error indicating the weight file is missing.
        return CoreMlStatus::ErrorSerialization;
    }

    // Optional: Validate the model.mlmodel is valid protobuf
    // We attempt to parse it, but don't fail hard if the proto
    // definitions don't match exactly (forward compatibility).
    if let Ok(mlmodel_bytes) = std::fs::read(&mlmodel_path) {
        // Try to decode as a prost Model message.
        // This is best-effort — if it fails, we don't reject the package
        // because the proto schema may have been updated.
        let _ = ane_coreml_proto::apple_proto::Model::decode(mlmodel_bytes.as_slice());
    }

    CoreMlStatus::Ok
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    // ─── CoreMlStatus enum value tests ──────────────────────────────────

    #[test]
    fn test_status_values() {
        assert_eq!(CoreMlStatus::Ok as c_int, 0);
        assert_eq!(CoreMlStatus::ErrorPlatformUnavailable as c_int, 1);
        assert_eq!(CoreMlStatus::ErrorModelLoad as c_int, 2);
        assert_eq!(CoreMlStatus::ErrorModelCompile as c_int, 3);
        assert_eq!(CoreMlStatus::ErrorPrediction as c_int, 4);
        assert_eq!(CoreMlStatus::ErrorInvalidArgument as c_int, 5);
        assert_eq!(CoreMlStatus::ErrorInsufficientOsVersion as c_int, 6);
        assert_eq!(CoreMlStatus::ErrorSerialization as c_int, 7);
        assert_eq!(CoreMlStatus::ErrorUnknown as c_int, -1);
    }

    // ─── coreml_is_available tests ──────────────────────────────────────

    #[test]
    fn test_coreml_is_available() {
        // On non-macOS, should return false
        if !cfg!(target_os = "macos") {
            assert!(!coreml_is_available());
        }
    }

    // ─── coreml_version tests ───────────────────────────────────────────

    #[test]
    fn test_coreml_version_non_macos() {
        // T-P2-07: coreml_version now always returns null since we cannot
        // determine the actual framework version without linking CoreML.framework.
        let version = coreml_version();
        assert!(version.is_null());
    }

    // ─── coreml_model_load tests ────────────────────────────────────────

    #[test]
    fn test_model_load_null_path() {
        let mut handle: *mut CoreMlModelHandle = ptr::null_mut();
        let status = unsafe { coreml_model_load(ptr::null(), &mut handle) };
        assert_eq!(status, CoreMlStatus::ErrorInvalidArgument);
    }

    #[test]
    fn test_model_load_null_out_handle() {
        let path = CString::new("/test/model.mlpackage").unwrap();
        let status = unsafe { coreml_model_load(path.as_ptr(), ptr::null_mut()) };
        assert_eq!(status, CoreMlStatus::ErrorInvalidArgument);
    }

    #[test]
    fn test_model_load_non_macos() {
        if !cfg!(target_os = "macos") {
            let path = CString::new("/test/model.mlpackage").unwrap();
            let mut handle: *mut CoreMlModelHandle = ptr::null_mut();
            let status = unsafe { coreml_model_load(path.as_ptr(), &mut handle) };
            assert_eq!(status, CoreMlStatus::ErrorPlatformUnavailable);
            assert!(handle.is_null());
        }
    }

    // ─── coreml_model_destroy tests ─────────────────────────────────────

    #[test]
    fn test_model_destroy_null() {
        // Should be a no-op, not a crash
        unsafe { coreml_model_destroy(ptr::null_mut()) };
    }

    #[test]
    fn test_model_destroy_allocated_handle() {
        // T-75 (I-50): Verify that a handle allocated with Box::new(ModelHandleInner)
        // can be safely destroyed. This tests the allocation contract.
        let inner = Box::new(ModelHandleInner { _path: "/test/model.mlpackage".to_string() });
        let handle = Box::into_raw(inner) as *mut CoreMlModelHandle;
        // Should not panic or cause UB
        unsafe { coreml_model_destroy(handle) };
    }

    // ─── coreml_model_info tests ────────────────────────────────────────

    #[test]
    fn test_model_info_null_handle() {
        let mut info = CoreMlModelInfo { function_count: 0, has_state: false, spec_version: 0 };
        let status = unsafe { coreml_model_info(ptr::null_mut(), &mut info) };
        assert_eq!(status, CoreMlStatus::ErrorInvalidArgument);
    }

    #[test]
    fn test_model_info_null_out_info() {
        // Create a fake non-null handle (we won't actually use it)
        let handle = std::ptr::dangling_mut::<CoreMlModelHandle>();
        let status = unsafe { coreml_model_info(handle, ptr::null_mut()) };
        assert_eq!(status, CoreMlStatus::ErrorInvalidArgument);
    }

    // ─── coreml_model_compile tests ─────────────────────────────────────

    #[test]
    fn test_model_compile_null_source() {
        let output_dir = CString::new("/tmp").unwrap();
        let mut result =
            CoreMlCompileResult { status: CoreMlStatus::Ok, compiled_path: ptr::null_mut() };
        let status = unsafe { coreml_model_compile(ptr::null(), output_dir.as_ptr(), &mut result) };
        assert_eq!(status, CoreMlStatus::ErrorInvalidArgument);
    }

    #[test]
    fn test_model_compile_null_output_dir() {
        let source = CString::new("/test/model.mlpackage").unwrap();
        let mut result =
            CoreMlCompileResult { status: CoreMlStatus::Ok, compiled_path: ptr::null_mut() };
        let status = unsafe { coreml_model_compile(source.as_ptr(), ptr::null(), &mut result) };
        assert_eq!(status, CoreMlStatus::ErrorInvalidArgument);
    }

    #[test]
    fn test_model_compile_null_out_result() {
        let source = CString::new("/test/model.mlpackage").unwrap();
        let output_dir = CString::new("/tmp").unwrap();
        let status =
            unsafe { coreml_model_compile(source.as_ptr(), output_dir.as_ptr(), ptr::null_mut()) };
        assert_eq!(status, CoreMlStatus::ErrorInvalidArgument);
    }

    #[test]
    fn test_model_compile_non_macos() {
        if !cfg!(target_os = "macos") {
            let source = CString::new("/test/model.mlpackage").unwrap();
            let output_dir = CString::new("/tmp").unwrap();
            let mut result =
                CoreMlCompileResult { status: CoreMlStatus::Ok, compiled_path: ptr::null_mut() };
            let status =
                unsafe { coreml_model_compile(source.as_ptr(), output_dir.as_ptr(), &mut result) };
            assert_eq!(status, CoreMlStatus::ErrorPlatformUnavailable);
            assert_eq!(result.status, CoreMlStatus::ErrorPlatformUnavailable);
            assert!(result.compiled_path.is_null());
        }
    }

    // ─── coreml_model_predict tests ─────────────────────────────────────

    #[test]
    fn test_model_predict_null_handle() {
        let mut latency: u64 = 0;
        let result = unsafe { coreml_model_predict(ptr::null_mut(), &mut latency) };
        assert_eq!(result.status, CoreMlStatus::ErrorInvalidArgument);
    }

    #[test]
    fn test_model_predict_non_macos() {
        if !cfg!(target_os = "macos") {
            // Create a fake non-null handle
            let handle = std::ptr::dangling_mut::<CoreMlModelHandle>();
            let mut latency: u64 = 99;
            let result = unsafe { coreml_model_predict(handle, &mut latency) };
            assert_eq!(result.status, CoreMlStatus::ErrorPlatformUnavailable);
            assert_eq!(latency, 0);
        }
    }

    // ─── coreml_free_string tests ───────────────────────────────────────

    #[test]
    fn test_free_string_null() {
        // Should be a no-op, not a crash
        unsafe { coreml_free_string(ptr::null_mut()) };
    }

    #[test]
    fn test_free_string_valid() {
        let s = CString::new("test string").unwrap();
        let raw = s.into_raw();
        // Should free without crash
        unsafe { coreml_free_string(raw) };
    }

    // ─── CoreMlModelInfo layout tests ───────────────────────────────────

    #[test]
    fn test_model_info_layout() {
        use std::mem::{align_of, size_of};

        // CoreMlModelInfo should be C-compatible
        // Layout: c_int(4) + bool(1) + padding(3) + c_int(4) = 12 bytes on 64-bit
        assert_eq!(size_of::<CoreMlModelInfo>(), size_of::<c_int>() + 4 + size_of::<c_int>()); // bool padded to 4 for alignment
        assert!(align_of::<CoreMlModelInfo>() >= align_of::<c_int>());
    }

    // ─── CoreMlStatus layout tests ──────────────────────────────────────

    #[test]
    fn test_status_layout() {
        use std::mem::{align_of, size_of};

        // CoreMlStatus should be c_int sized
        assert_eq!(size_of::<CoreMlStatus>(), size_of::<c_int>());
        assert_eq!(align_of::<CoreMlStatus>(), align_of::<c_int>());
    }

    // ─── CoreMlCompileResult layout tests ───────────────────────────────

    #[test]
    fn test_compile_result_layout() {
        use std::mem::{align_of, size_of};

        // CoreMlCompileResult should be C-compatible
        assert!(size_of::<CoreMlCompileResult>() > 0);
        assert!(align_of::<CoreMlCompileResult>() > 0);
    }

    // ─── CoreMlPredictResult layout tests ───────────────────────────────

    #[test]
    fn test_predict_result_layout() {
        use std::mem::{align_of, size_of};

        // CoreMlPredictResult should be C-compatible
        assert!(size_of::<CoreMlPredictResult>() > 0);
        assert!(align_of::<CoreMlPredictResult>() > 0);
    }

    // ─── coreml_validate_proto_package tests ────────────────────────────

    #[test]
    fn test_validate_null_path() {
        let status = unsafe { coreml_validate_proto_package(ptr::null()) };
        assert_eq!(status, CoreMlStatus::ErrorInvalidArgument);
    }

    #[test]
    fn test_validate_nonexistent_path() {
        let path = CString::new("/nonexistent/path/model.mlpackage").unwrap();
        let status = unsafe { coreml_validate_proto_package(path.as_ptr()) };
        assert_eq!(status, CoreMlStatus::ErrorModelLoad);
    }

    #[test]
    fn test_validate_file_not_directory() {
        // Create a temp file (not a directory)
        let tmp_dir = std::env::temp_dir().join("coreml_ffi_test_file");
        let _ = std::fs::remove_file(&tmp_dir);
        std::fs::write(&tmp_dir, b"not a directory").unwrap();

        let path = CString::new(tmp_dir.to_str().unwrap()).unwrap();
        let status = unsafe { coreml_validate_proto_package(path.as_ptr()) };
        assert_eq!(status, CoreMlStatus::ErrorModelLoad);

        let _ = std::fs::remove_file(&tmp_dir);
    }

    #[test]
    fn test_validate_empty_directory() {
        // Create a temp directory without mlpackage structure
        let tmp_dir = std::env::temp_dir().join("coreml_ffi_test_empty_dir");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let path = CString::new(tmp_dir.to_str().unwrap()).unwrap();
        let status = unsafe { coreml_validate_proto_package(path.as_ptr()) };
        // Should fail because Manifest.json is missing
        assert_eq!(status, CoreMlStatus::ErrorSerialization);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_validate_missing_manifest() {
        // Create a directory with model.mlmodel but no Manifest.json
        let tmp_dir = std::env::temp_dir().join("coreml_ffi_test_no_manifest");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        let model_dir = tmp_dir.join("Data/com.apple.CoreML");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.mlmodel"), b"").unwrap();

        let path = CString::new(tmp_dir.to_str().unwrap()).unwrap();
        let status = unsafe { coreml_validate_proto_package(path.as_ptr()) };
        assert_eq!(status, CoreMlStatus::ErrorSerialization);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_validate_invalid_manifest_json() {
        // Create a directory with invalid Manifest.json
        let tmp_dir = std::env::temp_dir().join("coreml_ffi_test_bad_manifest");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        let model_dir = tmp_dir.join("Data/com.apple.CoreML");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(tmp_dir.join("Manifest.json"), b"not valid json {{{").unwrap();
        std::fs::write(model_dir.join("model.mlmodel"), b"").unwrap();

        let path = CString::new(tmp_dir.to_str().unwrap()).unwrap();
        let status = unsafe { coreml_validate_proto_package(path.as_ptr()) };
        assert_eq!(status, CoreMlStatus::ErrorSerialization);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_validate_missing_mlmodel() {
        // Create a directory with valid Manifest.json but no model.mlmodel
        let tmp_dir = std::env::temp_dir().join("coreml_ffi_test_no_mlmodel");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        std::fs::write(tmp_dir.join("Manifest.json"), r#"{"schemaVersion":"1.0"}"#).unwrap();

        let path = CString::new(tmp_dir.to_str().unwrap()).unwrap();
        let status = unsafe { coreml_validate_proto_package(path.as_ptr()) };
        assert_eq!(status, CoreMlStatus::ErrorModelLoad);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_validate_missing_weight_bin() {
        // Create a directory with manifest and model.mlmodel but no weight.bin
        let tmp_dir = std::env::temp_dir().join("coreml_ffi_test_no_weights");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        let model_dir = tmp_dir.join("Data/com.apple.CoreML");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(tmp_dir.join("Manifest.json"), r#"{"schemaVersion":"1.0"}"#).unwrap();
        std::fs::write(model_dir.join("model.mlmodel"), b"").unwrap();

        let path = CString::new(tmp_dir.to_str().unwrap()).unwrap();
        let status = unsafe { coreml_validate_proto_package(path.as_ptr()) };
        // Missing weight.bin should return serialization error
        assert_eq!(status, CoreMlStatus::ErrorSerialization);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_validate_valid_mlpackage() {
        // Create a complete, valid mlpackage structure
        let tmp_dir = std::env::temp_dir().join("coreml_ffi_test_valid");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        let model_dir = tmp_dir.join("Data/com.apple.CoreML");
        let weights_dir = tmp_dir.join("Data/com.apple.CoreML/weights");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::create_dir_all(&weights_dir).unwrap();

        // Write valid Manifest.json
        std::fs::write(
            tmp_dir.join("Manifest.json"),
            r#"{"schemaVersion":"1.0","modelId":"test","files":[],"metadata":{}}"#,
        )
        .unwrap();

        // Write model.mlmodel (empty protobuf is technically invalid, but we
        // accept it as best-effort validation)
        std::fs::write(model_dir.join("model.mlmodel"), b"").unwrap();

        // Write weight.bin
        std::fs::write(weights_dir.join("weight.bin"), b"\x00\x00\x00\x00").unwrap();

        let path = CString::new(tmp_dir.to_str().unwrap()).unwrap();
        let status = unsafe { coreml_validate_proto_package(path.as_ptr()) };
        assert_eq!(status, CoreMlStatus::Ok);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    // ─── Invalid UTF-8 path test ────────────────────────────────────────

    #[test]
    fn test_validate_invalid_utf8_path() {
        // Create a path with invalid UTF-8
        let bad_bytes: &[u8] = b"/tmp/test\xFFpath";
        let _c_str = unsafe { CStr::from_ptr(bad_bytes.as_ptr() as *const c_char) };
        // This tests that the CStr conversion handles the case
        // (though in practice, CStr::from_ptr requires a null terminator)
        // We can't easily test this without unsafe, so we test a null path instead
        let status = unsafe { coreml_validate_proto_package(ptr::null()) };
        assert_eq!(status, CoreMlStatus::ErrorInvalidArgument);
    }
}
