//! Build script for ane-coreml-proto
//!
//! Compiles two sets of Core ML .proto files into Rust code using prost-build:
//!
//! 1. **Legacy** (`coreml/`): The original custom proto format (package `coreml`).
//!    Kept for backward compatibility with existing tests.
//!
//! 2. **Apple-compatible** (`coremlv2/`): Matches Apple's actual wire format
//!    (packages `CoreML.Specification` and `CoreML.Specification.MILSpec`).
//!    This is the format that Core ML's runtime can actually decode.

use std::io::Result;

fn main() -> Result<()> {
    let proto_include = "proto";

    // ─── Legacy proto files (package: coreml) ────────────────────────────
    let legacy_dir = "proto/coreml";
    let legacy_files = [
        format!("{}/DataStructures.proto", legacy_dir),
        format!("{}/MIL.proto", legacy_dir),
        format!("{}/Model.proto", legacy_dir),
    ];

    prost_build::Config::new()
        // The include path must be "proto/" so that `import "coreml/..."` resolves.
        .compile_protos(&legacy_files, &[proto_include.to_string()])
        .expect("Failed to compile legacy Core ML proto files");

    for proto_file in &legacy_files {
        println!("cargo:rerun-if-changed={}", proto_file);
    }

    // ─── Apple-compatible proto files ────────────────────────────────────
    // Package CoreML.Specification (Model.proto, FeatureTypes.proto)
    // Package CoreML.Specification.MILSpec (MIL.proto)
    let v2_dir = "proto/coremlv2";
    let v2_files = [
        format!("{}/FeatureTypes.proto", v2_dir),
        format!("{}/MIL.proto", v2_dir),
        format!("{}/Model.proto", v2_dir),
    ];

    prost_build::Config::new()
        .compile_protos(&v2_files, &[proto_include.to_string()])
        .expect("Failed to compile Apple-compatible Core ML proto files");

    for proto_file in &v2_files {
        println!("cargo:rerun-if-changed={}", proto_file);
    }

    Ok(())
}
