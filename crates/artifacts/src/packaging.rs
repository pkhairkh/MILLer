//! Packaging
//!
//! Assembles .mlpackage bundles from compiled artifacts into
//! deployable zip archives. The packager walks the compile output
//! directory and produces a deterministic zip file containing all
//! artifacts: mlpackage, manifest, MIR dump, and knowledge updates.

use crate::manifest::ArtifactManifest;
use anyhow::Result;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Artifact packager.
pub struct Packager {
    /// Output directory for packages.
    pub output_dir: String,
}

impl Packager {
    /// Create a new packager.
    pub fn new(output_dir: &str) -> Self {
        Self { output_dir: output_dir.to_string() }
    }

    /// Package all artifacts described in the manifest into a zip archive.
    ///
    /// Walks the compile output directory and adds every file to a zip archive.
    /// Returns the path to the created zip file.
    ///
    /// The zip file is placed in the output directory with the name
    /// `{model_id}.zip`.
    pub fn package(&self, manifest: &ArtifactManifest) -> Result<String> {
        let source_dir = Path::new(&self.output_dir);
        let zip_path = source_dir.join(format!("{}.zip", manifest.model_id));
        self.create_zip_from_directory(source_dir, &zip_path)?;
        Ok(zip_path.to_string_lossy().to_string())
    }

    /// Package a single .mlpackage from a directory into a zip archive.
    ///
    /// This is for packaging individual mlpackage bundles, e.g., for
    /// distribution or upload. The zip contains only the mlpackage contents.
    pub fn package_single(&self, name: &str, source_dir: &str) -> Result<String> {
        let source_path = Path::new(source_dir);
        let output_dir = Path::new(&self.output_dir);
        let zip_path = output_dir.join(format!("{}.zip", name));
        self.create_zip_from_directory(source_path, &zip_path)?;
        Ok(zip_path.to_string_lossy().to_string())
    }

    /// Validate that a package (directory or zip) is well-formed.
    ///
    /// For directories: checks that the directory exists and contains
    /// at least one file.
    /// For zip files: checks that the zip can be opened and contains
    /// at least one entry.
    pub fn validate(&self, package_path: &str) -> Result<bool> {
        let path = Path::new(package_path);

        if !path.exists() {
            return Ok(false);
        }

        if path.is_dir() {
            // Directory validation: must contain at least one file
            let file_count = count_files_recursive(path);
            Ok(file_count > 0)
        } else if path.extension().is_some_and(|e| e == "zip") {
            // Zip validation: must be a valid zip with at least one entry
            self.validate_zip(path)
        } else {
            Ok(false)
        }
    }

    /// Create a zip archive from a directory.
    ///
    /// All files within the directory are added with relative paths.
    /// The zip is deterministic: files are added in sorted order.
    fn create_zip_from_directory(&self, source_dir: &Path, zip_path: &Path) -> Result<()> {
        // Collect all files with their relative paths
        let mut entries: Vec<(PathBuf, PathBuf)> = Vec::new();
        collect_files_recursive(source_dir, source_dir, &mut entries);

        // Sort by relative path for determinism
        entries.sort_by(|a, b| a.1.cmp(&b.1));

        // Create the zip file
        if let Some(parent) = zip_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = fs::File::create(zip_path)?;
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        for (full_path, rel_path) in &entries {
            let rel_str = rel_path.to_string_lossy();
            let mut f = fs::File::open(full_path)?;
            let mut buffer = Vec::new();
            f.read_to_end(&mut buffer)?;

            zip.start_file(rel_str.as_ref(), options)?;
            zip.write_all(&buffer)?;
        }

        zip.finish()?;
        Ok(())
    }

    /// Validate a zip file by opening it and checking for entries.
    fn validate_zip(&self, zip_path: &Path) -> Result<bool> {
        let file = fs::File::open(zip_path)?;
        let archive = zip::ZipArchive::new(file)?;
        Ok(!archive.is_empty())
    }
}

/// Recursively collect files in a directory, storing (full_path, relative_path) pairs.
fn collect_files_recursive(base: &Path, current: &Path, entries: &mut Vec<(PathBuf, PathBuf)>) {
    if let Ok(dir_entries) = fs::read_dir(current) {
        for entry in dir_entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_files_recursive(base, &path, entries);
            } else {
                if let Ok(rel) = path.strip_prefix(base) {
                    entries.push((path.clone(), rel.to_path_buf()));
                }
            }
        }
    }
}

/// Count files recursively in a directory.
fn count_files_recursive(dir: &Path) -> usize {
    let mut count = 0;
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                count += count_files_recursive(&path);
            } else {
                count += 1;
            }
        }
    }
    count
}
