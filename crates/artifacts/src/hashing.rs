//! Content Hashing
//!
//! SHA-256 content hashing for artifact integrity verification
//! and cache invalidation.

use anyhow::Result;
use sha2::Digest;

/// Compute the SHA-256 hash of a file.
///
/// Reads the entire file into memory and hashes it.
/// Returns the hash as a lowercase hex string prefixed with "sha256:".
pub fn hash_file(path: &str) -> Result<String> {
    let content = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", path, e))?;
    Ok(hash_bytes(&content))
}

/// Compute the SHA-256 hash of byte content.
///
/// Returns the hash as a lowercase hex string prefixed with "sha256:".
/// This format matches the Python bridge's `_hash_directory` convention.
pub fn hash_bytes(content: &[u8]) -> String {
    use std::fmt::Write;
    let digest = sha2::Sha256::digest(content);
    let hex: String = digest.iter().fold(String::new(), |mut output, b| {
        write!(output, "{:02x}", b).unwrap();
        output
    });
    format!("sha256:{}", hex)
}

/// Verify that a file matches an expected hash.
///
/// The expected hash should be in the format "sha256:<hex>".
pub fn verify_hash(path: &str, expected: &str) -> Result<bool> {
    let actual = hash_file(path)?;
    Ok(actual == expected)
}
