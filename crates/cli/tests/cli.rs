//! Integration test: CLI basic invocations
//!
//! Tests help, version, and error-on-missing-args for the
//! `ane-compile` CLI binary. These tests run the binary as a
//! subprocess to verify actual CLI behavior.

use std::process::Command;

/// Helper to get the CLI binary path.
fn cli_binary() -> Command {
    // The binary name is `ane-cli` (from the crate name) or `ane-compile`
    // depending on how it was built. We try `ane-cli` first.
    let cmd = Command::new(env!("CARGO_BIN_EXE_ane-cli"));
    cmd
}

#[test]
fn test_help_flag() {
    let output = cli_binary().arg("--help").output().expect("Failed to run CLI with --help");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "--help should exit successfully");
    assert!(
        stdout.contains("ane-compile") || stdout.contains("MILLer"),
        "Help output should mention the tool name"
    );
    assert!(
        stdout.contains("compile") || stdout.contains("Compile"),
        "Help output should list compile subcommand"
    );
}

#[test]
fn test_version_flag() {
    let output = cli_binary().arg("--version").output().expect("Failed to run CLI with --version");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "--version should exit successfully");
    // Version output should contain the binary name
    assert!(
        stdout.contains("ane-compile") || stdout.contains("ane-cli"),
        "Version output should contain the binary name"
    );
}

#[test]
fn test_no_args_shows_help() {
    let output = cli_binary().output().expect("Failed to run CLI with no args");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // With clap, running with no subcommand prints help to stdout.
    // The important thing is it doesn't crash and shows usage info.
    let combined = format!("{}{}", stdout, stderr);
    assert!(
        combined.contains("Usage") || combined.contains("COMMAND") || combined.contains("compile"),
        "Output should show usage information with available commands"
    );
}

#[test]
fn test_compile_missing_input() {
    let output = cli_binary()
        .arg("compile")
        .arg("--output")
        .arg("/tmp/test_output")
        .output()
        .expect("Failed to run CLI compile with missing input");

    // Should fail because --input is required
    assert!(
        !output.status.success(),
        "compile without --input should fail"
    );
}

#[test]
fn test_compile_missing_output() {
    let output = cli_binary()
        .arg("compile")
        .arg("--input")
        .arg("/tmp/nonexistent.toml")
        .output()
        .expect("Failed to run CLI compile with missing output");

    // Should fail because --output is required
    assert!(
        !output.status.success(),
        "compile without --output should fail"
    );
}
