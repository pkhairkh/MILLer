//! Weight Binary Builder
//!
//! Constructs the `weight.bin` file that lives inside an mlpackage at
//! `Data/com.apple.CoreML/weights/weight.bin`. This file contains all
//! weight data for constant tensors, concatenated into a single binary
//! blob. The model protobuf references weights by offset into this file.
//!
//! ## Weight Sharing
//!
//! The key feature of this builder is support for **shared weights across
//! functions**. When two functions reference the same weight tensor (e.g.,
//! the embedding and decode_step functions sharing a projection weight),
//! they reference the same offset in weight.bin rather than each getting
//! their own copy. This produces smaller mlpackages than coremltools 9.0,
//! which duplicates constants per function boundary.

use ane_coreml_proto::{CoreMlDataType, SharedWeightRef, WeightEntry};
use anyhow::{bail, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// SHA-256 content hash for weight deduplication.
type ContentHash = [u8; 32];

/// Compute a SHA-256 hash of weight data.
fn content_hash(data: &[u8]) -> ContentHash {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Builder for the weight.bin file inside an mlpackage.
///
/// This manages the layout of weight tensors in the binary file,
/// tracking offsets and sizes for protobuf reference.
///
/// ## Deduplication
///
/// Two levels of deduplication are supported:
///
/// 1. **Name-based** (always active): When `add_weight()` encounters a name
///    that already exists with matching shape/dtype, it returns the existing
///    offset without duplicating data. This is the mechanism for cross-function
///    weight sharing (e.g., embedding and decode_step sharing a projection weight).
///
/// 2. **Content-hash** (opt-in via `with_content_dedup()`): When enabled and a
///    name is not found, the builder hashes the weight data with SHA-256 and
///    checks whether an existing weight has identical content. If found, the new
///    weight name is aliased to the existing entry's offset. This closes the gap
///    where coremltools 9.0 produces differently-named weights with identical data
///    (e.g., `embedding_projection_w` and `decode_step_projection_w` that happen
///    to share the same values).
///
/// Both deduplication levels are tracked separately in the build result metrics.
#[derive(Debug, Clone)]
pub struct WeightBinBuilder {
    /// Entries in the order they will appear in the binary file.
    entries: Vec<WeightEntry>,
    /// Map from weight name to its index in the entries vector.
    name_to_index: HashMap<String, usize>,
    /// Map from content hash to the first entry index with that content.
    /// Only populated when content deduplication is enabled.
    content_hash_to_index: HashMap<ContentHash, usize>,
    /// Aliases: names that were content-deduped to an existing entry.
    /// Maps alias name → index of the canonical entry.
    content_aliases: HashMap<String, usize>,
    /// Whether content-hash deduplication is enabled.
    enable_content_dedup: bool,
    /// Current offset (in bytes) into the binary file.
    current_offset: u64,
    /// Alignment requirement for weight data (16 bytes for ANE).
    alignment: u64,
    /// Number of weight additions that were deduplicated by name.
    name_dedup_count: usize,
    /// Bytes saved by name-based deduplication.
    name_dedup_bytes_saved: u64,
    /// Number of weight additions that were deduplicated by content hash.
    content_dedup_count: usize,
    /// Bytes saved by content-hash deduplication.
    content_dedup_bytes_saved: u64,
}

/// Result of building a weight.bin file.
#[derive(Debug, Clone)]
pub struct WeightBinResult {
    /// The raw binary data for weight.bin.
    pub data: Vec<u8>,
    /// Entries with updated offsets.
    pub entries: Vec<WeightEntry>,
    /// Total size of the weight.bin file.
    pub total_size: u64,
    /// Number of name-deduplicated weights.
    pub deduplicated_count: usize,
    /// Bytes saved by name-based deduplication.
    pub deduplicated_bytes: u64,
    /// Number of content-hash-deduplicated weights (different names, same content).
    pub content_deduplicated_count: usize,
    /// Bytes saved by content-hash deduplication.
    pub content_deduplicated_bytes: u64,
}

impl Default for WeightBinBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightBinBuilder {
    /// Create a new weight binary builder with 16-byte alignment (ANE requirement).
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            name_to_index: HashMap::new(),
            content_hash_to_index: HashMap::new(),
            content_aliases: HashMap::new(),
            enable_content_dedup: false,
            current_offset: 0,
            alignment: 16,
            name_dedup_count: 0,
            name_dedup_bytes_saved: 0,
            content_dedup_count: 0,
            content_dedup_bytes_saved: 0,
        }
    }

    /// Create a new weight binary builder with custom alignment.
    pub fn with_alignment(alignment: u64) -> Self {
        Self {
            entries: Vec::new(),
            name_to_index: HashMap::new(),
            content_hash_to_index: HashMap::new(),
            content_aliases: HashMap::new(),
            enable_content_dedup: false,
            current_offset: 0,
            alignment,
            name_dedup_count: 0,
            name_dedup_bytes_saved: 0,
            content_dedup_count: 0,
            content_dedup_bytes_saved: 0,
        }
    }

    /// Enable content-hash deduplication.
    ///
    /// When enabled, if `add_weight()` encounters a weight with a new name
    /// but identical content (same SHA-256 hash) and matching shape/dtype as
    /// an existing weight, it returns the existing entry's offset instead of
    /// storing the data again. This is semantically safe: two weights that
    /// happen to have identical bytes at the same shape/dtype can share storage
    /// because the protobuf references weights by offset, not by name.
    ///
    /// Use this when building multi-function mlpackages where different
    /// functions may produce differently-named constants with identical values
    /// — the exact scenario where coremltools 9.0 duplicates weight data.
    pub fn with_content_dedup(mut self) -> Self {
        self.enable_content_dedup = true;
        self
    }

    /// Add a weight tensor to the binary file.
    ///
    /// If a weight with the same name already exists, this returns the
    /// existing entry's offset without duplicating the data (name-based
    /// deduplication). This is the mechanism for cross-function weight sharing.
    ///
    /// If content-hash deduplication is enabled and the name is new but the
    /// content (SHA-256 hash) matches an existing weight with the same
    /// shape/dtype, this also returns the existing entry's offset. The new
    /// name is recorded as a content alias.
    ///
    /// Returns the offset of the weight in the binary file.
    pub fn add_weight(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: CoreMlDataType,
        data: Vec<u8>,
    ) -> Result<u64> {
        // Check for name-based duplicate (shared weight)
        if let Some(&idx) = self.name_to_index.get(name) {
            let existing = &self.entries[idx];
            // Verify the shapes and dtypes match
            if existing.shape != shape || existing.dtype != dtype {
                bail!(
                    "Weight '{}' already exists with different shape/dtype. \
                     Existing: shape={:?}, dtype={:?}. \
                     New: shape={:?}, dtype={:?}.",
                    name,
                    existing.shape,
                    existing.dtype,
                    shape,
                    dtype
                );
            }
            // Track name-based deduplication metrics
            self.name_dedup_count += 1;
            self.name_dedup_bytes_saved += data.len() as u64;
            return Ok(existing.offset);
        }

        // Check for content-hash deduplication (opt-in)
        if self.enable_content_dedup {
            let hash = content_hash(&data);
            if let Some(&idx) = self.content_hash_to_index.get(&hash) {
                let existing = &self.entries[idx];
                // Verify shape/dtype match — content alone is not enough;
                // two weights with the same bytes but different shapes would
                // produce different tensor values when read by the runtime.
                if existing.shape == shape && existing.dtype == dtype {
                    // Content-hash match: alias this name to the existing entry
                    self.content_aliases.insert(name.to_string(), idx);
                    self.name_to_index.insert(name.to_string(), idx);
                    self.content_dedup_count += 1;
                    self.content_dedup_bytes_saved += data.len() as u64;
                    return Ok(existing.offset);
                }
                // Shape/dtype mismatch with same content hash: do NOT deduplicate.
                // This is an unusual case (collision or genuinely different tensors
                // that happen to have the same bytes). Store separately.
            }
        }

        // Align the offset
        let aligned_offset = align_up(self.current_offset, self.alignment);

        // Add padding bytes if needed
        let _padding = (aligned_offset - self.current_offset) as usize;

        let entry = WeightEntry {
            name: name.to_string(),
            offset: aligned_offset,
            size: data.len() as u64,
            shape,
            dtype,
            data,
        };

        let entry_index = self.entries.len();

        self.name_to_index.insert(name.to_string(), entry_index);

        // Record content hash for future content-dedup lookups
        if self.enable_content_dedup {
            let hash = content_hash(&entry.data);
            self.content_hash_to_index.insert(hash, entry_index);
        }

        self.entries.push(entry);
        self.current_offset = aligned_offset + self.entries.last().unwrap().size;

        Ok(aligned_offset)
    }

    /// Add a shared weight that will be referenced by multiple functions.
    ///
    /// This is a convenience method that calls `add_weight` and then
    /// creates a `SharedWeightRef` tracking which functions reference it.
    pub fn add_shared_weight(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: CoreMlDataType,
        data: Vec<u8>,
        referencing_functions: Vec<String>,
    ) -> Result<SharedWeightRef> {
        let _offset = self.add_weight(name, shape, dtype, data)?;

        let entry = self.entries[self.name_to_index[name]].clone();

        Ok(SharedWeightRef { weight: entry, referencing_functions })
    }

    /// Build the weight.bin binary data.
    ///
    /// This concatenates all weight tensors with appropriate alignment
    /// padding and returns the complete binary data plus updated entries.
    /// Both name-based and content-hash deduplication metrics are reported.
    pub fn build(self) -> WeightBinResult {
        let total_entries = self.entries.len();
        let mut data = Vec::new();
        let mut current_pos: u64 = 0;

        // First pass: calculate total size
        let mut total_size: u64 = 0;
        for entry in &self.entries {
            let aligned_offset = align_up(total_size, self.alignment);
            total_size = aligned_offset + entry.size;
        }

        // Second pass: build the binary
        let mut updated_entries = Vec::with_capacity(total_entries);
        let deduplicated_count = self.name_dedup_count;
        let deduplicated_bytes = self.name_dedup_bytes_saved;
        let content_deduplicated_count = self.content_dedup_count;
        let content_deduplicated_bytes = self.content_dedup_bytes_saved;

        for mut entry in self.entries {
            let aligned_offset = align_up(current_pos, self.alignment);

            // Add padding
            if aligned_offset > current_pos {
                let padding = (aligned_offset - current_pos) as usize;
                data.extend(std::iter::repeat_n(0u8, padding));
            }

            entry.offset = aligned_offset;
            data.extend_from_slice(&entry.data);
            current_pos = aligned_offset + entry.size;

            updated_entries.push(entry);
        }

        WeightBinResult {
            data,
            entries: updated_entries,
            total_size: current_pos,
            deduplicated_count,
            deduplicated_bytes,
            content_deduplicated_count,
            content_deduplicated_bytes,
        }
    }

    /// Check if a weight with the given name already exists.
    pub fn has_weight(&self, name: &str) -> bool {
        self.name_to_index.contains_key(name)
    }

    /// Get the offset of an existing weight, if it exists.
    pub fn get_weight_offset(&self, name: &str) -> Option<u64> {
        self.name_to_index.get(name).map(|&idx| self.entries[idx].offset)
    }

    /// Number of unique weights in the binary.
    pub fn weight_count(&self) -> usize {
        self.entries.len()
    }

    /// Total size of the binary data (with alignment padding).
    pub fn estimated_size(&self) -> u64 {
        self.current_offset
    }
}

/// Align a value up to the given alignment boundary.
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_bin_builder_basic() {
        let mut builder = WeightBinBuilder::new();

        let data = vec![1u8; 64];
        let offset = builder
            .add_weight("weight_0", vec![4, 16], CoreMlDataType::Float16, data.clone())
            .unwrap();

        assert_eq!(offset, 0); // First weight starts at offset 0

        let result = builder.build();
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.total_size, 64);
    }

    #[test]
    fn test_weight_bin_builder_shared() {
        let mut builder = WeightBinBuilder::new();

        let data = vec![42u8; 128];
        let offset1 = builder
            .add_weight("shared_weight", vec![8, 16], CoreMlDataType::Float16, data.clone())
            .unwrap();

        // Adding the same weight again should return the same offset
        let offset2 = builder
            .add_weight("shared_weight", vec![8, 16], CoreMlDataType::Float16, data.clone())
            .unwrap();

        assert_eq!(offset1, offset2);

        let result = builder.build();
        assert_eq!(result.entries.len(), 1); // Only one entry, not two
    }

    #[test]
    fn test_weight_bin_builder_alignment() {
        let mut builder = WeightBinBuilder::with_alignment(16);

        // Add a weight that's not a multiple of 16
        let data1 = vec![1u8; 10];
        let offset1 =
            builder.add_weight("weight_0", vec![10], CoreMlDataType::UInt8, data1).unwrap();

        // Second weight should be aligned to 16 bytes
        let data2 = vec![2u8; 32];
        let offset2 =
            builder.add_weight("weight_1", vec![16], CoreMlDataType::Float16, data2).unwrap();

        assert_eq!(offset1, 0);
        assert_eq!(offset2, 16); // Aligned to 16 bytes

        let result = builder.build();
        assert_eq!(result.total_size, 48); // 16 (aligned) + 32
    }

    #[test]
    fn test_weight_bin_builder_mismatch_rejected() {
        let mut builder = WeightBinBuilder::new();

        let data = vec![1u8; 64];
        builder.add_weight("weight_0", vec![4, 16], CoreMlDataType::Float16, data).unwrap();

        // Same name, different shape — should fail
        let data2 = vec![1u8; 64];
        let result = builder.add_weight(
            "weight_0",
            vec![8, 8], // Different shape
            CoreMlDataType::Float16,
            data2,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_weight_bin_builder_shared_weight_ref() {
        let mut builder = WeightBinBuilder::new();

        let data = vec![42u8; 256];
        let shared = builder
            .add_shared_weight(
                "shared_projection_weight",
                vec![128, 128],
                CoreMlDataType::Float16,
                data,
                vec!["embedding".to_string(), "decode_step".to_string()],
            )
            .unwrap();

        assert_eq!(shared.referencing_functions.len(), 2);
        assert_eq!(shared.weight.name, "shared_projection_weight");
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }

    /// Test that deduplication metrics are correctly tracked.
    ///
    /// This proves that the proto-direct path can measure and report
    /// exactly how much weight data was saved by deduplication —
    /// a capability that coremltools 9.0 lacks.
    #[test]
    fn test_deduplication_metrics_tracked() {
        let mut builder = WeightBinBuilder::new();

        // Add a unique weight
        let data_a = vec![1u8; 64];
        builder
            .add_weight("weight_a", vec![4, 16], CoreMlDataType::Float16, data_a.clone())
            .unwrap();

        // Add a shared weight (first occurrence)
        let data_shared = vec![42u8; 128];
        builder
            .add_weight("shared_proj", vec![8, 16], CoreMlDataType::Float16, data_shared.clone())
            .unwrap();

        // Deduplicate: add the same shared weight again (second occurrence)
        builder
            .add_weight("shared_proj", vec![8, 16], CoreMlDataType::Float16, data_shared.clone())
            .unwrap();

        // Deduplicate again: third occurrence of the same weight
        builder
            .add_weight("shared_proj", vec![8, 16], CoreMlDataType::Float16, data_shared.clone())
            .unwrap();

        let result = builder.build();

        // 2 unique entries (weight_a + shared_proj), not 4
        assert_eq!(result.entries.len(), 2);

        // 2 deduplication events (2nd and 3rd additions of shared_proj)
        assert_eq!(
            result.deduplicated_count, 2,
            "Should track 2 deduplication events for 3 total additions of the same name"
        );

        // 128 bytes saved per dedup × 2 = 256 bytes saved
        assert_eq!(
            result.deduplicated_bytes, 256,
            "Should track 256 bytes saved (128 bytes × 2 dedup events)"
        );

        // Total size should be 64 + 128 = 192 (not 64 + 128*3 = 448)
        assert_eq!(
            result.total_size, 192,
            "Total size should reflect deduplicated storage, not naive concatenation"
        );
    }

    /// Test that no deduplication produces zero metrics.
    #[test]
    fn test_no_deduplication_zero_metrics() {
        let mut builder = WeightBinBuilder::new();

        builder.add_weight("w0", vec![4], CoreMlDataType::Float16, vec![0u8; 8]).unwrap();
        builder.add_weight("w1", vec![8], CoreMlDataType::Float16, vec![0u8; 16]).unwrap();
        builder.add_weight("w2", vec![16], CoreMlDataType::Float16, vec![0u8; 32]).unwrap();

        let result = builder.build();

        assert_eq!(result.deduplicated_count, 0, "No deduplication should produce count=0");
        assert_eq!(result.deduplicated_bytes, 0, "No deduplication should produce bytes=0");
        assert_eq!(result.entries.len(), 3, "All 3 unique weights should be present");
    }

    /// Test content-hash based deduplication: different names but identical content
    /// can be deduplicated when explicitly opted in via `with_content_dedup()`.
    ///
    /// This goes beyond name-based deduplication (which coremltools 9.0 cannot do
    /// at all) and adds the ability to detect that two differently-named weights
    /// have the same binary content, sharing storage.
    #[test]
    fn test_content_hash_deduplication() {
        // --- Without content-hash dedup (default): different names, identical content → stored separately ---
        let mut builder_no_dedup = WeightBinBuilder::new();
        let content = vec![42u8; 256];
        builder_no_dedup
            .add_weight(
                "embedding_projection_w",
                vec![128, 16],
                CoreMlDataType::Float16,
                content.clone(),
            )
            .unwrap();
        builder_no_dedup
            .add_weight(
                "decode_step_projection_w",
                vec![128, 16],
                CoreMlDataType::Float16,
                content.clone(),
            )
            .unwrap();

        let result_no_dedup = builder_no_dedup.build();
        assert_eq!(result_no_dedup.entries.len(), 2,
            "Without content dedup: different names are stored separately even with identical content");
        assert_eq!(result_no_dedup.deduplicated_count, 0);
        assert_eq!(result_no_dedup.content_deduplicated_count, 0);

        // --- With content-hash dedup enabled: different names, identical content → deduplicated ---
        let mut builder = WeightBinBuilder::new().with_content_dedup();

        builder
            .add_weight(
                "embedding_projection_w",
                vec![128, 16],
                CoreMlDataType::Float16,
                content.clone(),
            )
            .unwrap();
        builder
            .add_weight(
                "decode_step_projection_w",
                vec![128, 16],
                CoreMlDataType::Float16,
                content.clone(),
            )
            .unwrap();

        let result = builder.build();

        // Only one entry should exist (second was content-deduped)
        assert_eq!(
            result.entries.len(),
            1,
            "Content-hash dedup: different names with identical content should produce one entry"
        );
        assert_eq!(
            result.deduplicated_count, 0,
            "No name-based dedup expected (names are different)"
        );
        assert_eq!(result.content_deduplicated_count, 1, "One content-hash dedup event");
        assert_eq!(result.content_deduplicated_bytes, 256, "256 bytes saved by content-hash dedup");

        // Verify the content-deduped entry has the correct size
        assert_eq!(result.entries[0].size, 256);
    }

    /// Test that name-based deduplication still works when content dedup is enabled.
    #[test]
    fn test_name_dedup_with_content_dedup_enabled() {
        let mut builder = WeightBinBuilder::new().with_content_dedup();

        let content = vec![7u8; 128];
        builder
            .add_weight("shared_w", vec![8, 16], CoreMlDataType::Float16, content.clone())
            .unwrap();
        builder
            .add_weight("shared_w", vec![8, 16], CoreMlDataType::Float16, content.clone())
            .unwrap();

        let result = builder.build();

        assert_eq!(result.entries.len(), 1, "Same name → one entry");
        assert_eq!(result.deduplicated_count, 1, "One name-based dedup event");
        assert_eq!(result.deduplicated_bytes, 128, "128 bytes saved by name dedup");
        assert_eq!(
            result.content_deduplicated_count, 0,
            "No content-hash dedup (name matched first)"
        );
    }

    /// Test that content-hash dedup does NOT deduplicate when shapes differ.
    #[test]
    fn test_content_dedup_shape_mismatch_not_deduped() {
        let mut builder = WeightBinBuilder::new().with_content_dedup();

        // Same bytes, different shapes → NOT deduplicated
        let content = vec![0u8; 64];
        builder
            .add_weight("weight_a", vec![4, 16], CoreMlDataType::Float16, content.clone())
            .unwrap();
        builder
            .add_weight("weight_b", vec![8, 8], CoreMlDataType::Float16, content.clone())
            .unwrap();

        let result = builder.build();

        assert_eq!(result.entries.len(), 2, "Same content, different shape → stored separately");
        assert_eq!(
            result.content_deduplicated_count, 0,
            "No content-hash dedup when shapes differ"
        );
    }

    /// Test that content-hash dedup does NOT deduplicate when dtypes differ.
    #[test]
    fn test_content_dedup_dtype_mismatch_not_deduped() {
        let mut builder = WeightBinBuilder::new().with_content_dedup();

        // Same bytes, same shape, different dtype → NOT deduplicated
        let content = vec![0u8; 32];
        builder.add_weight("weight_a", vec![8], CoreMlDataType::Float16, content.clone()).unwrap();
        builder.add_weight("weight_b", vec![4], CoreMlDataType::Float32, content.clone()).unwrap();

        let result = builder.build();

        assert_eq!(result.entries.len(), 2, "Same content, different dtype → stored separately");
        assert_eq!(
            result.content_deduplicated_count, 0,
            "No content-hash dedup when dtypes differ"
        );
    }

    /// Test the coremltools gap: different function namespaces produce
    /// differently-named weights with identical data.
    ///
    /// This is the real-world scenario that motivates content-hash dedup:
    /// coremltools 9.0's `add_function()` duplicates weight data per function,
    /// but proto-direct emission with content dedup produces one copy.
    #[test]
    fn test_content_dedup_coremltools_scenario() {
        let weight_data = vec![7u8; 1024]; // 512×512 fp16 weight matrix

        // --- coremltools 9.0 path: no content dedup → duplicated ---
        let mut no_dedup = WeightBinBuilder::new();
        no_dedup
            .add_weight(
                "embedding_projection_w",
                vec![512, 512],
                CoreMlDataType::Float16,
                weight_data.clone(),
            )
            .unwrap();
        no_dedup
            .add_weight(
                "decode_step_projection_w",
                vec![512, 512],
                CoreMlDataType::Float16,
                weight_data.clone(),
            )
            .unwrap();
        let no_dedup_result = no_dedup.build();

        // --- Proto-direct with content dedup: shared ---
        let mut with_dedup = WeightBinBuilder::new().with_content_dedup();
        with_dedup
            .add_weight(
                "embedding_projection_w",
                vec![512, 512],
                CoreMlDataType::Float16,
                weight_data.clone(),
            )
            .unwrap();
        with_dedup
            .add_weight(
                "decode_step_projection_w",
                vec![512, 512],
                CoreMlDataType::Float16,
                weight_data.clone(),
            )
            .unwrap();
        let with_dedup_result = with_dedup.build();

        // Content-dedup path produces one entry, coremltools-style produces two
        assert_eq!(
            with_dedup_result.entries.len(),
            1,
            "Content dedup: one entry for two differently-named weights with identical content"
        );
        assert_eq!(
            no_dedup_result.entries.len(),
            2,
            "No content dedup: two entries for two differently-named weights"
        );

        // Proto-direct + content dedup is smaller by exactly one weight's worth
        assert!(
            with_dedup_result.total_size < no_dedup_result.total_size,
            "Content-dedup path ({}) must be smaller than coremltools-style path ({})",
            with_dedup_result.total_size,
            no_dedup_result.total_size
        );
        assert_eq!(with_dedup_result.content_deduplicated_count, 1);
        assert_eq!(with_dedup_result.content_deduplicated_bytes, 1024);
    }

    /// Test that distinct content with different names is NOT deduplicated.
    #[test]
    fn test_content_dedup_different_content_not_deduped() {
        let mut builder = WeightBinBuilder::new().with_content_dedup();

        builder
            .add_weight("weight_a", vec![8, 16], CoreMlDataType::Float16, vec![1u8; 256])
            .unwrap();
        builder
            .add_weight("weight_b", vec![8, 16], CoreMlDataType::Float16, vec![2u8; 256])
            .unwrap();

        let result = builder.build();

        assert_eq!(result.entries.len(), 2, "Different content → stored separately");
        assert_eq!(result.content_deduplicated_count, 0);
    }

    /// Test multi-function weight sharing scenario (the coremltools 9.0 gap).
    ///
    /// When an embedding function and a decode_step function share a projection
    /// weight matrix, coremltools 9.0 duplicates the weight in weight.bin.
    /// Our proto-direct path stores it once and both functions reference the
    /// same offset, producing a smaller mlpackage.
    #[test]
    fn test_multifunction_weight_sharing_saves_space() {
        let weight_data = vec![7u8; 1024]; // Simulate a 512×512 fp16 weight matrix

        // --- Proto-direct path: shared weight ---
        let mut shared_builder = WeightBinBuilder::new();
        shared_builder
            .add_weight(
                "projection_w",
                vec![512, 512],
                CoreMlDataType::Float16,
                weight_data.clone(),
            )
            .unwrap();
        // Second function references the same weight (deduplicated)
        shared_builder
            .add_weight(
                "projection_w",
                vec![512, 512],
                CoreMlDataType::Float16,
                weight_data.clone(),
            )
            .unwrap();
        let shared_result = shared_builder.build();

        // --- coremltools 9.0 path: duplicated weight ---
        let mut dup_builder = WeightBinBuilder::new();
        dup_builder
            .add_weight(
                "embedding_projection_w",
                vec![512, 512],
                CoreMlDataType::Float16,
                weight_data.clone(),
            )
            .unwrap();
        dup_builder
            .add_weight(
                "decode_step_projection_w",
                vec![512, 512],
                CoreMlDataType::Float16,
                weight_data.clone(),
            )
            .unwrap();
        let dup_result = dup_builder.build();

        // Proto-direct saves exactly the weight size (1024 bytes)
        assert!(
            shared_result.total_size < dup_result.total_size,
            "Proto-direct shared path ({}) must be smaller than duplicated path ({})",
            shared_result.total_size,
            dup_result.total_size
        );
        assert_eq!(shared_result.deduplicated_count, 1, "One deduplication event");
        assert_eq!(shared_result.deduplicated_bytes, 1024, "1024 bytes saved");
        assert_eq!(dup_result.deduplicated_count, 0, "No dedup in coremltools-style path");
    }
}
