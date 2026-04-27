//! Build script for ane-coreml-proto
//!
//! Compiles the Core ML .proto files into Rust code using prost-build.
//! The proto files define the Core ML model format used by Apple's
//! Core ML framework, enabling direct Rust-to-mlpackage emission
//! without the Python bridge.

use std::io::Result;

fn main() -> Result<()> {
    let proto_include = "proto";           // Root for import resolution
    let proto_dir = "proto/coreml";        // Where the .proto files live
    let proto_files = [
        format!("{}/DataStructures.proto", proto_dir),
        format!("{}/MIL.proto", proto_dir),
        format!("{}/Model.proto", proto_dir),
    ];

    prost_build::Config::new()
        // The include path must be "proto/" so that `import "coreml/DataStructures.proto"`
        // inside MIL.proto and Model.proto resolves correctly.
        .compile_protos(&proto_files, &[proto_include.to_string()])
        .expect("Failed to compile Core ML proto files");

    // Rerun if protos change
    for proto_file in &proto_files {
        println!("cargo:rerun-if-changed={}", proto_file);
    }

    Ok(())
}
