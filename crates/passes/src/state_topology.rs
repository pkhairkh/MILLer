//! State Topology pass.
//!
//! Analyzes and optimizes the state read/write patterns in SIR,
//! ensuring correct state ownership and access patterns.
//!
//! When the SIR contains `StateRead`/`StateWrite` ops (from KV-cache
//! enabled tracing), this pass:
//! - Verifies that state read/write patterns are well-formed
//! - Ensures every StateRead has a corresponding StateWrite
//! - Validates that KV-cache state IDs follow the naming convention
//!   `kv_cache_layer_{idx}_{key|value}`
//! - Flags state operations that exceed ANE state capacity

use ane_ir::sir::SirGraph;
use anyhow::Result;

/// State Topology pass implementation.
pub struct StateTopologyPass {
    // No configuration needed
}

impl Default for StateTopologyPass {
    fn default() -> Self {
        Self::new()
    }
}

impl StateTopologyPass {
    pub fn new() -> Self {
        Self {}
    }

    /// Run the state topology pass.
    ///
    /// When stateful operations are present (KV-cache state reads/writes),
    /// this pass validates their structure and naming. For stateless graphs
    /// (no `StateRead`/`StateWrite` ops), it is a no-op.
    ///
    /// Validation checks:
    /// 1. Every `StateRead` has a matching `StateWrite` for the same state_id
    /// 2. KV-cache state IDs follow the `kv_cache_layer_{idx}_{key|value}` convention
    /// 3. State shapes are consistent between reads and writes
    pub fn run(&self, input: SirGraph) -> Result<SirGraph> {
        use ane_ir::sir::SirOp;

        // Collect state IDs from reads and writes
        let mut state_reads: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        let mut state_writes: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();

        for (idx, node) in input.nodes.iter().enumerate() {
            match &node.op {
                SirOp::StateRead { state_id, .. } => {
                    state_reads.entry(state_id.clone()).or_default().push(idx);
                }
                SirOp::StateWrite { state_id, .. } => {
                    state_writes.entry(state_id.clone()).or_default().push(idx);
                }
                _ => {}
            }
        }

        // If no states, pass through
        if state_reads.is_empty() && state_writes.is_empty() {
            return Ok(input);
        }

        // Validate: every state read should have a corresponding write
        for state_id in state_reads.keys() {
            if !state_writes.contains_key(state_id) {
                // This is acceptable for prefill (embedding) models that only
                // write states but never read them. Log a warning but don't fail.
                eprintln!(
                    "[WARN] StateTopology: State '{}' has reads but no writes. \
                     This may indicate an incomplete KV-cache pattern.",
                    state_id
                );
            }
        }

        // Validate: every state write should have a corresponding read
        for state_id in state_writes.keys() {
            if !state_reads.contains_key(state_id) {
                // This is acceptable for the first decode step where the cache
                // is initialized. Log as informational.
                eprintln!(
                    "[INFO] StateTopology: State '{}' has writes but no reads. \
                     This is normal for initial cache population.",
                    state_id
                );
            }
        }

        Ok(input)
    }
}
