//! ANE execution engine types modeled after observed ANE fusion behavior.
//! Source: ane-constraints-docs/03-placement-and-compiler/fusion-boundaries-and-resource-allocation.md

use serde::{Deserialize, Serialize};

/// ANE execution engine.
/// The ANEC targets three distinct hardware engines for different op categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AneEngine {
    /// Neural Engine — conv/pool/matmul/attention pipeline
    NE,
    /// Processing Element — elementwise/reduction/scaled-EW pipeline
    PE,
    /// Transpose Engine — data rearrangement
    TransposeEngine,
}

impl AneEngine {
    /// Returns the engine name as used in ANE fusion atoms.
    pub fn fusion_prefix(&self) -> &'static str {
        match self {
            AneEngine::NE => "NEFUSED",
            AneEngine::PE => "PEFUSED",
            AneEngine::TransposeEngine => "TRANSPOSE",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_fusion_prefix() {
        assert_eq!(AneEngine::NE.fusion_prefix(), "NEFUSED");
        assert_eq!(AneEngine::PE.fusion_prefix(), "PEFUSED");
        assert_eq!(AneEngine::TransposeEngine.fusion_prefix(), "TRANSPOSE");
    }

    #[test]
    fn test_engine_serialization() {
        let json = serde_json::to_string(&AneEngine::NE).unwrap();
        assert_eq!(json, "\"NE\"");
    }
}
