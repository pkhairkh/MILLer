//! Weight Binary Builder (MILBlob Storage Format — blob_v1)
//!
//! Constructs the `weight.bin` file that lives inside an mlpackage at
//! `Data/com.apple.CoreML/weights/weight.bin`. This file uses Apple's
//! **MILBlob Storage format** (version 2, aka "blob_v1"), which is the
//! only format accepted by CoreML's Espresso/EIR execution planner.
//!
//! ## File Layout
//!
//! ```text
//! |<storage_header>|<blob_metadata 0>|<data 0>|...|<blob_metadata k>|<data k>|
//! ```
//!
//! Every structure is **64-byte aligned**.
//!
//! ### storage_header (64 bytes)
//!
//! | Offset | Size | Field     | Value                     |
//! |--------|------|-----------|---------------------------|
//! | 0      | 4    | count     | Number of blob entries    |
//! | 4      | 4    | version   | Must be 2                 |
//! | 8      | 56   | reserved  | All zeros                 |
//!
//! ### blob_metadata (64 bytes, one per weight tensor)
//!
//! | Offset | Size | Field                 | Value                        |
//! |--------|------|-----------------------|------------------------------|
//! | 0      | 4    | sentinel              | 0xDEADBEEF                   |
//! | 4      | 4    | mil_dtype             | BlobDataType enum value      |
//! | 8      | 8    | sizeInBytes           | Size of raw data             |
//! | 16     | 8    | offset                | Absolute file offset to data |
//! | 24     | 8    | padding_size_in_bits  | 0 for byte-aligned types     |
//! | 32     | 32   | reserved              | All zeros                    |
//!
//! ## Weight Sharing
//!
//! The key feature of this builder is support for **shared weights across
//! functions**. When two functions reference the same weight tensor (e.g.,
//! the embedding and decode_step functions sharing a projection weight),
//! they reference the same offset in weight.bin rather than each getting
//! their own copy. This produces smaller mlpackages than coremltools 9.0,
//! which duplicates constants per function boundary.
//!
//! ## Offset Semantics
//!
//! The `WeightEntry.offset` field stores the **blob_metadata offset**
//! (not the raw data offset). This is because the protobuf's
//! `BlobFileValue.offset` must point to the `blob_metadata` header —
//! the runtime's `StorageReader` reads the metadata at that offset to
//! find where the actual data lives.

use ane_coreml_proto::{CoreMlDataType, SharedWeightRef, WeightEntry};
use anyhow::{bail, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

// ─── MILBlob Constants ──────────────────────────────────────────────────────

/// MILBlob Storage format version. Must be 2 for the runtime to accept it.
const BLOB_STORAGE_VERSION: u32 = 2;

/// Magic sentinel value for each blob_metadata entry.
const BLOB_METADATA_SENTINEL: u32 = 0xDEADBEEF;

/// Size of the storage_header in bytes (64-byte aligned).
const STORAGE_HEADER_SIZE: u64 = 64;

/// Size of each blob_metadata entry in bytes (64-byte aligned).
const BLOB_METADATA_SIZE: u64 = 64;

/// Alignment for all structures in the blob file (64 bytes).
const BLOB_ALIGNMENT: u64 = 64;

// ─── BlobDataType Enum (mirrors Apple's BlobDataType.hpp) ───────────────────

/// Data type enum for the MILBlob format.
///
/// These values match Apple's `BlobDataType` enum from
/// `mlmodel/src/MILBlob/Blob/BlobDataType.hpp` in the coremltools source.
///
/// T-35: Added UInt16, Int4, UInt4, Float8E4M3FN, Float8E5M2 variants
/// matching Apple's full BlobDataType enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum BlobDataType {
    Float16 = 1,
    Float32 = 2,
    UInt8 = 3,
    Int8 = 4,
    UInt16 = 7,
    Int4 = 8,
    UInt4 = 11,
    Int32 = 14,
    Float8E4M3FN = 16,
    Float8E5M2 = 17,
}

/// Map CoreMlDataType to MILBlob BlobDataType enum value.
///
/// This is a free function because we cannot add inherent methods to
/// `CoreMlDataType` which is defined in the `ane-coreml-proto` crate.
fn coreml_dtype_to_blob_dtype(dtype: &CoreMlDataType) -> u32 {
    match dtype {
        CoreMlDataType::Float16 => BlobDataType::Float16 as u32,
        CoreMlDataType::Float32 => BlobDataType::Float32 as u32,
        CoreMlDataType::UInt8 => BlobDataType::UInt8 as u32,
        CoreMlDataType::Int8 => BlobDataType::Int8 as u32,
        CoreMlDataType::Int32 => BlobDataType::Int32 as u32,
        // T-35: new dtype blob mappings
        CoreMlDataType::UInt16 => BlobDataType::UInt16 as u32,
        CoreMlDataType::Int4 => BlobDataType::Int4 as u32,
        CoreMlDataType::UInt4 => BlobDataType::UInt4 as u32,
        CoreMlDataType::E4M3 => BlobDataType::Float8E4M3FN as u32,
        CoreMlDataType::E5M2 => BlobDataType::Float8E5M2 as u32,
        // Conservatively map unknown/unsupported types to Float32
        CoreMlDataType::Float64 | CoreMlDataType::Bool | CoreMlDataType::Unknown => {
            BlobDataType::Float32 as u32
        }
    }
}

// ─── Deduplication ──────────────────────────────────────────────────────────

/// SHA-256 content hash for weight deduplication.
type ContentHash = [u8; 32];

/// Compute a SHA-256 hash of weight data.
fn content_hash(data: &[u8]) -> ContentHash {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

// ─── Builder ────────────────────────────────────────────────────────────────

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
    /// The raw binary data for weight.bin (in MILBlob Storage format).
    pub data: Vec<u8>,
    /// Entries with updated offsets (offset points to blob_metadata header).
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
    /// Create a new weight binary builder.
    ///
    /// The MILBlob Storage format requires 64-byte alignment for all
    /// structures (storage_header, blob_metadata, and data sections).
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            name_to_index: HashMap::new(),
            content_hash_to_index: HashMap::new(),
            content_aliases: HashMap::new(),
            enable_content_dedup: false,
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
    /// Note: The offset is set during `build()`; before that, it is 0.
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

        let entry = WeightEntry {
            name: name.to_string(),
            offset: 0, // Will be set during build()
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

        // Return 0 placeholder — the actual offset is computed during build()
        Ok(0)
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

    /// Build the weight.bin binary data in MILBlob Storage format (blob_v1).
    ///
    /// This produces the binary format expected by CoreML's Espresso/EIR
    /// execution planner and StorageReader:
    ///
    /// ```text
    /// | storage_header (64B) | blob_metadata_0 (64B) | data_0 (padded) | blob_metadata_1 (64B) | data_1 (padded) | ...
    /// ```
    ///
    /// The `WeightEntry.offset` in the result points to the `blob_metadata`
    /// header for that weight — this is the offset that must go into the
    /// protobuf's `BlobFileValue.offset` field, because the runtime's
    /// StorageReader reads the metadata at that offset to locate the data.
    pub fn build(self) -> WeightBinResult {
        let num_entries = self.entries.len();
        let mut buf: Vec<u8> = Vec::new();

        let deduplicated_count = self.name_dedup_count;
        let deduplicated_bytes = self.name_dedup_bytes_saved;
        let content_deduplicated_count = self.content_dedup_count;
        let content_deduplicated_bytes = self.content_dedup_bytes_saved;

        // ── Step 1: Write storage_header (64 bytes) ─────────────────────
        // We write a placeholder count first; if the file is empty we leave
        // it as-is, otherwise we patch it after writing all entries.
        write_storage_header(&mut buf, num_entries as u32);
        let mut current_pos: u64 = STORAGE_HEADER_SIZE;

        // ── Step 2: For each entry, write blob_metadata + data ──────────
        let mut updated_entries = Vec::with_capacity(num_entries);

        for mut entry in self.entries {
            // Align to 64-byte boundary for the blob_metadata header
            let metadata_offset = align_up(current_pos, BLOB_ALIGNMENT);
            if metadata_offset > current_pos {
                let padding = (metadata_offset - current_pos) as usize;
                buf.extend(std::iter::repeat_n(0u8, padding));
            }

            // The raw data starts immediately after the blob_metadata header.
            // Since metadata_offset is 64-byte aligned and BLOB_METADATA_SIZE
            // is 64, data_offset = metadata_offset + 64 is also 64-byte aligned.
            let data_offset = metadata_offset + BLOB_METADATA_SIZE;

            // Write blob_metadata (64 bytes)
            write_blob_metadata(
                &mut buf,
                coreml_dtype_to_blob_dtype(&entry.dtype),
                entry.size,
                data_offset,
            );

            // Write raw weight data
            buf.extend_from_slice(&entry.data);

            // Pad data to 64-byte boundary
            let data_end = data_offset + entry.size;
            let padded_end = align_up(data_end, BLOB_ALIGNMENT);
            if padded_end > data_end {
                let padding = (padded_end - data_end) as usize;
                buf.extend(std::iter::repeat_n(0u8, padding));
            }

            current_pos = padded_end;

            // Store the metadata offset in the entry — this is what the
            // protobuf's BlobFileValue.offset must reference.
            entry.offset = metadata_offset;
            updated_entries.push(entry);
        }

        // ── Step 3: If there are zero entries, we still need a valid header ─
        // (Already written above with count=0.)

        WeightBinResult {
            data: buf,
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
    ///
    /// Note: The offset is only valid after `build()` has been called.
    /// Before build(), offsets are 0 placeholders.
    pub fn get_weight_offset(&self, name: &str) -> Option<u64> {
        self.name_to_index.get(name).map(|&idx| self.entries[idx].offset)
    }

    /// Number of unique weights in the binary.
    pub fn weight_count(&self) -> usize {
        self.entries.len()
    }

    /// Estimated size of the binary data (with alignment and metadata overhead).
    pub fn estimated_size(&self) -> u64 {
        if self.entries.is_empty() {
            return STORAGE_HEADER_SIZE;
        }
        let data_total: u64 = self.entries.iter().map(|e| e.size).sum();
        let padding_per_entry: u64 = self
            .entries
            .iter()
            .map(|e| {
                let data_end = e.size;
                align_up(data_end, BLOB_ALIGNMENT) - data_end
            })
            .sum();
        STORAGE_HEADER_SIZE
            + (BLOB_METADATA_SIZE * self.entries.len() as u64)
            + data_total
            + padding_per_entry
    }
}

// ─── Binary Writing Helpers ─────────────────────────────────────────────────

/// Write the MILBlob storage_header (64 bytes) to the buffer.
///
/// Layout:
/// - bytes 0-3:   count (u32 LE) — number of blob entries
/// - bytes 4-7:   version (u32 LE) — must be 2
/// - bytes 8-63:  reserved — all zeros
fn write_storage_header(buf: &mut Vec<u8>, count: u32) {
    buf.extend_from_slice(&count.to_le_bytes());
    buf.extend_from_slice(&BLOB_STORAGE_VERSION.to_le_bytes());
    // Reserved: 56 bytes of zeros (7 × u64)
    buf.extend(std::iter::repeat_n(0u8, 56));
}

/// Write a blob_metadata entry (64 bytes) to the buffer.
///
/// Layout:
/// - bytes 0-3:   sentinel (u32 LE) — 0xDEADBEEF
/// - bytes 4-7:   mil_dtype (u32 LE) — BlobDataType enum value
/// - bytes 8-15:  sizeInBytes (u64 LE)
/// - bytes 16-23: offset (u64 LE) — absolute file offset to raw data
/// - bytes 24-31: padding_size_in_bits (u64 LE) — 0 for byte-aligned types
/// - bytes 32-63: reserved — all zeros (4 × u64)
fn write_blob_metadata(buf: &mut Vec<u8>, mil_dtype: u32, size_in_bytes: u64, data_offset: u64) {
    buf.extend_from_slice(&BLOB_METADATA_SENTINEL.to_le_bytes());
    buf.extend_from_slice(&mil_dtype.to_le_bytes());
    buf.extend_from_slice(&size_in_bytes.to_le_bytes());
    buf.extend_from_slice(&data_offset.to_le_bytes());
    // padding_size_in_bits = 0 (byte-aligned types)
    buf.extend_from_slice(&0u64.to_le_bytes());
    // Reserved: 32 bytes of zeros (4 × u64)
    buf.extend(std::iter::repeat_n(0u8, 32));
}

/// Align a value up to the given alignment boundary.
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_v1_storage_header() {
        let mut builder = WeightBinBuilder::new();
        builder
            .add_weight("weight_0", vec![4, 16], CoreMlDataType::Float16, vec![1u8; 128])
            .unwrap();

        let result = builder.build();

        // File must start with storage_header
        assert!(result.data.len() >= 64, "File must be at least 64 bytes for header");

        // Check storage_header fields
        let count = u32::from_le_bytes(result.data[0..4].try_into().unwrap());
        let version = u32::from_le_bytes(result.data[4..8].try_into().unwrap());
        assert_eq!(count, 1, "Storage header count should be 1");
        assert_eq!(version, BLOB_STORAGE_VERSION, "Storage header version must be 2");

        // Reserved bytes 8-63 must be zero
        for i in 8..64 {
            assert_eq!(result.data[i], 0, "Reserved byte {} must be zero", i);
        }
    }

    #[test]
    fn test_blob_v1_metadata_sentinel() {
        let mut builder = WeightBinBuilder::new();
        builder
            .add_weight("weight_0", vec![4, 16], CoreMlDataType::Float16, vec![1u8; 128])
            .unwrap();

        let result = builder.build();

        // First blob_metadata starts at offset 64 (after storage_header)
        let sentinel = u32::from_le_bytes(result.data[64..68].try_into().unwrap());
        assert_eq!(sentinel, BLOB_METADATA_SENTINEL, "Sentinel must be 0xDEADBEEF");
    }

    #[test]
    fn test_blob_v1_metadata_dtype() {
        let mut builder = WeightBinBuilder::new();
        builder
            .add_weight("fp16_weight", vec![4, 16], CoreMlDataType::Float16, vec![1u8; 128])
            .unwrap();

        let result = builder.build();

        // Check dtype field in metadata (offset 68-71)
        let dtype = u32::from_le_bytes(result.data[68..72].try_into().unwrap());
        assert_eq!(dtype, BlobDataType::Float16 as u32, "dtype should be Float16 (1)");
    }

    #[test]
    fn test_blob_v1_metadata_data_offset() {
        let mut builder = WeightBinBuilder::new();
        let weight_data = vec![0xABu8; 128];
        builder
            .add_weight("weight_0", vec![4, 32], CoreMlDataType::Float16, weight_data.clone())
            .unwrap();

        let result = builder.build();

        // The data offset should be metadata_offset + 64 = 64 + 64 = 128
        let data_offset = u64::from_le_bytes(result.data[80..88].try_into().unwrap());
        assert_eq!(data_offset, 128u64, "Data should start at offset 128");

        // Verify the actual data is at that offset
        assert_eq!(&result.data[128..256], &weight_data[..], "Data at offset must match");
    }

    #[test]
    fn test_blob_v1_offset_is_metadata_offset() {
        let mut builder = WeightBinBuilder::new();
        builder
            .add_weight("weight_0", vec![4, 16], CoreMlDataType::Float16, vec![1u8; 128])
            .unwrap();

        let result = builder.build();

        // The entry.offset should point to the blob_metadata, not the raw data
        assert_eq!(result.entries[0].offset, 64, "Offset should point to metadata at byte 64");
    }

    #[test]
    fn test_blob_v1_multiple_entries() {
        let mut builder = WeightBinBuilder::new();
        builder
            .add_weight("weight_0", vec![4, 16], CoreMlDataType::Float16, vec![1u8; 128])
            .unwrap();
        builder
            .add_weight("weight_1", vec![8, 16], CoreMlDataType::Float16, vec![2u8; 256])
            .unwrap();

        let result = builder.build();

        // Check count in header
        let count = u32::from_le_bytes(result.data[0..4].try_into().unwrap());
        assert_eq!(count, 2);

        // First entry: metadata at 64, data at 128, data ends at 256, padded to 256
        assert_eq!(result.entries[0].offset, 64);

        // Second entry: metadata at 256 (next 64-byte boundary after data_0),
        // data at 320
        assert_eq!(result.entries[1].offset, 256);

        // Verify second entry's metadata sentinel
        let sentinel2 = u32::from_le_bytes(result.data[256..260].try_into().unwrap());
        assert_eq!(sentinel2, BLOB_METADATA_SENTINEL);

        // Verify second entry's data offset
        let data_offset2 = u64::from_le_bytes(result.data[272..280].try_into().unwrap());
        assert_eq!(data_offset2, 320u64);
    }

    #[test]
    fn test_blob_v1_shared_weight_same_offset() {
        let mut builder = WeightBinBuilder::new();

        let data = vec![42u8; 128];
        let _offset1 = builder
            .add_weight("shared_weight", vec![8, 16], CoreMlDataType::Float16, data.clone())
            .unwrap();

        // Adding the same weight again should deduplicate
        let _offset2 = builder
            .add_weight("shared_weight", vec![8, 16], CoreMlDataType::Float16, data.clone())
            .unwrap();

        let result = builder.build();
        assert_eq!(result.entries.len(), 1, "Only one entry, not two");
        assert_eq!(result.deduplicated_count, 1, "One dedup event");
    }

    #[test]
    fn test_blob_v1_dtype_mappings() {
        // Test each supported dtype maps correctly
        assert_eq!(coreml_dtype_to_blob_dtype(&CoreMlDataType::Float16), 1);
        assert_eq!(coreml_dtype_to_blob_dtype(&CoreMlDataType::Float32), 2);
        assert_eq!(coreml_dtype_to_blob_dtype(&CoreMlDataType::UInt8), 3);
        assert_eq!(coreml_dtype_to_blob_dtype(&CoreMlDataType::Int8), 4);
        assert_eq!(coreml_dtype_to_blob_dtype(&CoreMlDataType::Int32), 14);
    }

    #[test]
    fn test_blob_v1_small_weight_padding() {
        let mut builder = WeightBinBuilder::new();
        // 10 bytes of data — needs padding to 64-byte boundary
        builder.add_weight("small_weight", vec![10], CoreMlDataType::UInt8, vec![1u8; 10]).unwrap();

        let result = builder.build();

        // metadata at 64, data at 128, data is 10 bytes, padded to 192
        assert_eq!(result.entries[0].offset, 64);

        let size_in_bytes = u64::from_le_bytes(result.data[72..80].try_into().unwrap());
        assert_eq!(size_in_bytes, 10u64, "sizeInBytes should be 10 (unpadded)");

        // Total size: header(64) + metadata(64) + data_padded(64) = 192
        assert_eq!(result.total_size, 192);
    }

    #[test]
    fn test_blob_v1_empty_model() {
        let builder = WeightBinBuilder::new();
        let result = builder.build();

        // Should still produce a valid header
        assert_eq!(result.entries.len(), 0);
        assert_eq!(result.data.len(), 64, "Empty model should have 64-byte header");

        let count = u32::from_le_bytes(result.data[0..4].try_into().unwrap());
        let version = u32::from_le_bytes(result.data[4..8].try_into().unwrap());
        assert_eq!(count, 0);
        assert_eq!(version, BLOB_STORAGE_VERSION);
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
    fn test_align_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(63, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(128, 64), 128);
        assert_eq!(align_up(129, 64), 192);
    }

    /// Test that deduplication metrics are correctly tracked with blob format.
    #[test]
    fn test_deduplication_metrics_tracked() {
        let mut builder = WeightBinBuilder::new();

        // Add a unique weight
        builder
            .add_weight("weight_a", vec![4, 16], CoreMlDataType::Float16, vec![1u8; 64])
            .unwrap();

        // Add a shared weight (first occurrence)
        builder
            .add_weight("shared_proj", vec![8, 16], CoreMlDataType::Float16, vec![42u8; 128])
            .unwrap();

        // Deduplicate: add the same shared weight again (second occurrence)
        builder
            .add_weight("shared_proj", vec![8, 16], CoreMlDataType::Float16, vec![42u8; 128])
            .unwrap();

        // Deduplicate again: third occurrence of the same weight
        builder
            .add_weight("shared_proj", vec![8, 16], CoreMlDataType::Float16, vec![42u8; 128])
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

    /// Test content-hash based deduplication with blob format.
    #[test]
    fn test_content_hash_deduplication() {
        let content = vec![42u8; 256];

        // --- Without content-hash dedup (default): different names, identical content → stored separately ---
        let mut builder_no_dedup = WeightBinBuilder::new();
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
    #[test]
    fn test_content_dedup_coremltools_scenario() {
        let weight_data = vec![7u8; 1024];

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

    /// Verify binary output matches expected blob_v1 format exactly.
    #[test]
    fn test_blob_v1_binary_format_exact() {
        let mut builder = WeightBinBuilder::new();
        let weight_data = vec![0xAAu8; 64]; // 64 bytes = 32 fp16 elements
        builder
            .add_weight("test_weight", vec![2, 16], CoreMlDataType::Float16, weight_data.clone())
            .unwrap();

        let result = builder.build();

        // Total: header(64) + metadata(64) + data(64) = 192
        // (data is already 64-byte aligned, no padding needed)
        assert_eq!(result.data.len(), 192);

        // ── storage_header ──
        assert_eq!(u32::from_le_bytes(result.data[0..4].try_into().unwrap()), 1); // count
        assert_eq!(u32::from_le_bytes(result.data[4..8].try_into().unwrap()), 2); // version
                                                                                  // bytes 8-63: reserved zeros
        assert!(&result.data[8..64].iter().all(|&b| b == 0));

        // ── blob_metadata at offset 64 ──
        assert_eq!(u32::from_le_bytes(result.data[64..68].try_into().unwrap()), 0xDEADBEEF); // sentinel
        assert_eq!(u32::from_le_bytes(result.data[68..72].try_into().unwrap()), 1); // Float16
        assert_eq!(u64::from_le_bytes(result.data[72..80].try_into().unwrap()), 64); // sizeInBytes
        assert_eq!(u64::from_le_bytes(result.data[80..88].try_into().unwrap()), 128); // data offset
        assert_eq!(u64::from_le_bytes(result.data[88..96].try_into().unwrap()), 0); // padding_size_in_bits
                                                                                    // bytes 96-127: reserved zeros
        assert!(&result.data[96..128].iter().all(|&b| b == 0));

        // ── data at offset 128 ──
        assert_eq!(&result.data[128..192], &weight_data[..]);

        // Entry offset should be 64 (metadata offset)
        assert_eq!(result.entries[0].offset, 64);
    }
}
