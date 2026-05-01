//! ML Package Writer
//!
//! Writes a `.mlpackage` directory to disk with the correct structure
//! matching Apple's Core ML package format:
//!
//! ```text
//! model.mlpackage/
//! ├── Manifest.json           — Package metadata (Apple schema)
//! └── Data/
//!     └── com.apple.CoreML/
//!         ├── model.mlmodel   — Protobuf model definition
//!         └── weights/
//!             └── weight.bin  — MILBlob Storage format (blob_v1)
//! ```
//!
//! The `weight.bin` file uses Apple's MILBlob Storage format (version 2,
//! aka "blob_v1"), which is the only format accepted by CoreML's Espresso/EIR
//! execution planner and StorageReader.

use crate::weights::WeightBinBuilder;
use ane_coreml_proto::{CoreMlModel, ManifestItemInfo, PackageManifest};
use anyhow::Result;
use std::fs;
use std::path::Path;

/// Writer for `.mlpackage` directory structures.
///
/// This handles the complete lifecycle of writing an mlpackage:
/// 1. Create the directory structure
/// 2. Build and write the `weight.bin` file
/// 3. Serialize and write the `model.mlmodel` protobuf file
/// 4. Generate and write the `Manifest.json` file
pub struct MlPackageWriter;

/// Result of writing an mlpackage to disk.
#[derive(Debug, Clone)]
pub struct MlPackageResult {
    /// Path to the written mlpackage directory.
    pub path: String,
    /// SHA-256 content hash of the entire mlpackage directory.
    pub content_hash: String,
    /// Total size of the mlpackage directory in bytes.
    pub total_size: u64,
    /// Number of files written.
    pub file_count: usize,
    /// Number of unique weights in weight.bin.
    pub weight_count: usize,
    /// Number of functions in the model.
    pub function_count: usize,
    /// Whether the model has shared weights across functions.
    pub has_shared_weights: bool,
    /// Size comparison with coremltools equivalent (if available).
    pub size_comparison: Option<SizeComparison>,
}

/// Size comparison between proto-direct and coremltools emission.
#[derive(Debug, Clone)]
pub struct SizeComparison {
    /// Size of the proto-direct weight.bin.
    pub proto_weight_bin_size: u64,
    /// Size of the coremltools weight.bin (if measured).
    pub coremltools_weight_bin_size: Option<u64>,
    /// Whether the proto-direct variant is smaller.
    pub proto_is_smaller: Option<bool>,
    /// Bytes saved by deduplication.
    pub bytes_saved: u64,
}

impl MlPackageWriter {
    /// Write a CoreMlModel to disk as an mlpackage directory.
    ///
    /// This creates the complete mlpackage structure with all required files.
    /// If the target directory already exists, it is removed and recreated.
    pub fn write(model: &CoreMlModel, output_path: &str) -> Result<MlPackageResult> {
        let pkg_path = Path::new(output_path);

        // Remove existing package if present
        if pkg_path.exists() {
            fs::remove_dir_all(pkg_path)?;
        }

        // Create directory structure — Apple's mlpackage puts EVERYTHING under Data/
        let data_dir = pkg_path.join("Data/com.apple.CoreML");
        let weights_dir = data_dir.join("weights");

        fs::create_dir_all(&data_dir)?;
        fs::create_dir_all(&weights_dir)?;

        // Step 1: Build and write weight.bin (MILBlob Storage format — blob_v1)
        // Enable content deduplication for multi-function models where
        // different functions may produce differently-named weights with
        // identical content (e.g., embedding and decode_step sharing a
        // projection weight).
        let has_shared = !model.shared_weights.is_empty();
        let mut weight_builder = WeightBinBuilder::new().with_content_dedup();
        for weight in &model.weights {
            weight_builder.add_weight(
                &weight.name,
                weight.shape.clone(),
                weight.dtype,
                weight.data.clone(),
            )?;
        }

        // Handle shared weights
        let has_shared_weights = has_shared;
        // The shared weight was already added via add_weight above.
        // The SharedWeightRef just tracks which functions reference it.
        // We don't need to add it again — the dedup in add_weight
        // handles this.

        let weight_result = weight_builder.build();
        let weight_bin_path = weights_dir.join("weight.bin");
        fs::write(&weight_bin_path, &weight_result.data)?;

        // Step 2: Serialize and write the model protobuf
        let model_proto =
            crate::mir_to_proto::model_to_protobuf_bytes(model, &weight_result.entries)?;
        let mlmodel_path = data_dir.join("model.mlmodel");
        fs::write(&mlmodel_path, &model_proto)?;

        // Step 3: Generate and write Manifest.json
        let manifest = Self::build_manifest(model);
        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        let manifest_path = pkg_path.join("Manifest.json");
        fs::write(&manifest_path, manifest_json)?;

        // Step 4: Compute content hash
        let content_hash = Self::hash_directory(pkg_path)?;

        // Step 5: Compute total size
        let total_size = Self::directory_size(pkg_path)?;
        let file_count = Self::count_files(pkg_path)?;

        Ok(MlPackageResult {
            path: output_path.to_string(),
            content_hash,
            total_size,
            file_count,
            weight_count: weight_result.entries.len(),
            function_count: model.functions.len(),
            has_shared_weights,
            size_comparison: None,
        })
    }

    /// Build the Manifest.json content from a CoreMlModel using Apple's schema.
    ///
    /// Apple's Manifest.json requires:
    /// - `fileFormatVersion`: Must be `"1.0.0"`
    /// - `itemInfoEntries`: Map of UUID → {path, name, author, description}
    /// - `rootModelIdentifier`: UUID of the model spec entry
    ///
    /// The `path` in each entry is relative to the `Data/` directory.
    fn build_manifest(model: &CoreMlModel) -> PackageManifest {
        // Deterministic UUIDs: same model content always produces the same manifest.
        // Uses UUID v5 (SHA-1 name-based) with a stable namespace so that two
        // consecutive builds of the same model produce identical Manifest.json files.
        let namespace = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, b"miller.coreml-emit.internal");
        let model_uuid = uuid::Uuid::new_v5(&namespace, model.default_function_name.as_bytes());
        let weights_uuid = uuid::Uuid::new_v5(&namespace, b"weights");

        let mut item_info_entries = HashMap::new();

        // Model spec entry
        item_info_entries.insert(
            model_uuid.to_string().to_uppercase(),
            ManifestItemInfo {
                path: "com.apple.CoreML/model.mlmodel".to_string(),
                name: "model.mlmodel".to_string(),
                author: "com.apple.CoreML".to_string(),
                description: "CoreML Model Specification".to_string(),
            },
        );

        // Weights entry (only if there are weights)
        if !model.weights.is_empty() {
            item_info_entries.insert(
                weights_uuid.to_string().to_uppercase(),
                ManifestItemInfo {
                    path: "com.apple.CoreML/weights".to_string(),
                    name: "weights".to_string(),
                    author: "com.apple.CoreML".to_string(),
                    description: "CoreML Model Weights".to_string(),
                },
            );
        }

        PackageManifest {
            file_format_version: "1.0.0".to_string(),
            item_info_entries,
            root_model_identifier: model_uuid.to_string().to_uppercase(),
        }
    }

    /// Compute SHA-256 hash of all files in a directory.
    fn hash_directory(path: &Path) -> Result<String> {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        Self::hash_directory_recursive(path, path, &mut hasher)?;
        let hash = hasher.finalize();
        Ok(format!("sha256:{:x}", hash))
    }

    fn hash_directory_recursive(base: &Path, dir: &Path, hasher: &mut sha2::Sha256) -> Result<()> {
        use sha2::Digest;

        let mut entries: Vec<_> = fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                Self::hash_directory_recursive(base, &path, hasher)?;
            } else {
                let rel = path.strip_prefix(base)?.to_string_lossy();
                hasher.update(rel.as_bytes());
                hasher.update(b":");
                let data = fs::read(&path)?;
                hasher.update(&data);
                hasher.update(b";");
            }
        }
        Ok(())
    }

    /// Compute total size of all files in a directory.
    fn directory_size(path: &Path) -> Result<u64> {
        let mut total: u64 = 0;
        Self::size_recursive(path, &mut total)?;
        Ok(total)
    }

    fn size_recursive(dir: &Path, total: &mut u64) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::size_recursive(&path, total)?;
            } else {
                *total += entry.metadata()?.len();
            }
        }
        Ok(())
    }

    /// Count files in a directory.
    fn count_files(path: &Path) -> Result<usize> {
        let mut count = 0;
        Self::count_recursive(path, &mut count)?;
        Ok(count)
    }

    fn count_recursive(dir: &Path, count: &mut usize) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::count_recursive(&path, count)?;
            } else {
                *count += 1;
            }
        }
        Ok(())
    }
}

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    use ane_coreml_proto::{CoreMlComputeUnit, SpecVersion};

    #[test]
    fn test_build_manifest() {
        let model = CoreMlModel {
            spec_version: SpecVersion::V10,
            description: ane_coreml_proto::ModelDescriptionCompat {
                inputs: vec![],
                outputs: vec![],
                states: vec![],
            },
            functions: vec![],
            default_function_name: "main".to_string(),
            weights: vec![],
            shared_weights: vec![],
            compute_unit: CoreMlComputeUnit::CpuAndNe,
            user_defined_metadata: HashMap::new(),
        };

        let manifest = MlPackageWriter::build_manifest(&model);
        assert_eq!(manifest.file_format_version, "1.0.0");
        assert_eq!(manifest.item_info_entries.len(), 1); // Only model entry (no weights)
        assert!(!manifest.root_model_identifier.is_empty());
        // The root model identifier must be one of the entry UUIDs
        assert!(manifest.item_info_entries.contains_key(&manifest.root_model_identifier));
    }

    #[test]
    fn test_write_mlpackage_apple_format() {
        use crate::mir_to_proto::{build_linear_projection_mir, convert_mir_to_proto};
        use ane_coreml_proto::mir_compat::MilDtypeCompat;

        let graph = build_linear_projection_mir(
            "test_apple_proto_disk",
            64,
            32,
            1,
            MilDtypeCompat::Fp16,
            42,
        );
        let model =
            convert_mir_to_proto(&graph, SpecVersion::V7, CoreMlComputeUnit::CpuAndNe).unwrap();

        let output_path = "/tmp/miller_test_validate.mlpackage";
        let result = MlPackageWriter::write(&model, output_path).unwrap();

        assert_eq!(result.file_count, 3);
        assert!(result.weight_count > 0);
        assert!(result.total_size > 0);

        // Verify the protobuf file was written and is parseable
        let mlmodel_path = format!("{output_path}/Data/com.apple.CoreML/model.mlmodel");
        let data = std::fs::read(&mlmodel_path).unwrap();
        assert!(!data.is_empty());

        // Parse with Apple's protobuf format
        let parsed: ane_coreml_proto::apple_proto::Model =
            prost::Message::decode(data.as_slice()).unwrap();
        assert_eq!(parsed.specification_version, 7);
        assert!(parsed.description.is_some());

        // Verify mlProgram is present (field 502)
        let model_type = parsed.r#type.as_ref().unwrap();
        match model_type {
            ane_coreml_proto::apple_proto::model::Type::MlProgram(program) => {
                assert!(program.functions.contains_key("main"));
                let func = program.functions.get("main").unwrap();
                assert!(!func.block_specializations.is_empty());
            }
        }

        // Verify weight.bin was written with actual weight data
        let weight_bin_path = format!("{output_path}/Data/com.apple.CoreML/weights/weight.bin");
        let weight_data = std::fs::read(&weight_bin_path).unwrap();
        assert!(weight_data.len() > 0, "weight.bin should have data");

        // Verify Manifest.json uses Apple's schema
        let manifest_path = format!("{output_path}/Manifest.json");
        let manifest_data = std::fs::read_to_string(&manifest_path).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&manifest_data).unwrap();
        assert_eq!(manifest["fileFormatVersion"], "1.0.0");
        assert!(manifest["itemInfoEntries"].is_object());
        assert!(manifest["rootModelIdentifier"].is_string());
    }
}
