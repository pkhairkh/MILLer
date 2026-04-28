//! Serialization utilities for all IR types.

use crate::{air::AirGraph, mir::MirGraph, pir::PirGraph, sir::SirGraph};

/// Serialize a SIR graph to MessagePack bytes.
pub fn serialize_sir(graph: &SirGraph) -> Result<Vec<u8>, String> {
    rmp_serde::to_vec(graph).map_err(|e| format!("SIR serialization failed: {}", e))
}

/// Deserialize a SIR graph from MessagePack bytes.
pub fn deserialize_sir(bytes: &[u8]) -> Result<SirGraph, String> {
    rmp_serde::from_slice(bytes).map_err(|e| format!("SIR deserialization failed: {}", e))
}

/// Serialize an AIR graph to MessagePack bytes.
pub fn serialize_air(graph: &AirGraph) -> Result<Vec<u8>, String> {
    rmp_serde::to_vec(graph).map_err(|e| format!("AIR serialization failed: {}", e))
}

/// Deserialize an AIR graph from MessagePack bytes.
pub fn deserialize_air(bytes: &[u8]) -> Result<AirGraph, String> {
    rmp_serde::from_slice(bytes).map_err(|e| format!("AIR deserialization failed: {}", e))
}

/// Serialize a MIR graph to MessagePack bytes.
pub fn serialize_mir(graph: &MirGraph) -> Result<Vec<u8>, String> {
    rmp_serde::to_vec(graph).map_err(|e| format!("MIR serialization failed: {}", e))
}

/// Deserialize a MIR graph from MessagePack bytes.
pub fn deserialize_mir(bytes: &[u8]) -> Result<MirGraph, String> {
    rmp_serde::from_slice(bytes).map_err(|e| format!("MIR deserialization failed: {}", e))
}

/// Serialize a PIR graph to MessagePack bytes.
pub fn serialize_pir(graph: &PirGraph) -> Result<Vec<u8>, String> {
    rmp_serde::to_vec(graph).map_err(|e| format!("PIR serialization failed: {}", e))
}

/// Deserialize a PIR graph from MessagePack bytes.
pub fn deserialize_pir(bytes: &[u8]) -> Result<PirGraph, String> {
    rmp_serde::from_slice(bytes).map_err(|e| format!("PIR deserialization failed: {}", e))
}
