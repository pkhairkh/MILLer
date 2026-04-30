//! Serialization utilities for all IR types.

use serde::de::DeserializeOwned;
use serde::Serialize;

/// Serialize any IR graph to MessagePack bytes.
pub fn serialize_graph<T: Serialize>(graph: &T) -> Result<Vec<u8>, String> {
    rmp_serde::to_vec(graph).map_err(|e| format!("IR serialization failed: {}", e))
}

/// Deserialize any IR graph from MessagePack bytes.
pub fn deserialize_graph<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    rmp_serde::from_slice(bytes).map_err(|e| format!("IR deserialization failed: {}", e))
}

// Keep type-specific convenience wrappers for backward compatibility
use crate::{air::AirGraph, mir::MirGraph, pir::PirGraph, sir::SirGraph};

pub fn serialize_sir(graph: &SirGraph) -> Result<Vec<u8>, String> {
    serialize_graph(graph)
}
pub fn deserialize_sir(bytes: &[u8]) -> Result<SirGraph, String> {
    deserialize_graph(bytes)
}
pub fn serialize_air(graph: &AirGraph) -> Result<Vec<u8>, String> {
    serialize_graph(graph)
}
pub fn deserialize_air(bytes: &[u8]) -> Result<AirGraph, String> {
    deserialize_graph(bytes)
}
pub fn serialize_mir(graph: &MirGraph) -> Result<Vec<u8>, String> {
    serialize_graph(graph)
}
pub fn deserialize_mir(bytes: &[u8]) -> Result<MirGraph, String> {
    deserialize_graph(bytes)
}
pub fn serialize_pir(graph: &PirGraph) -> Result<Vec<u8>, String> {
    serialize_graph(graph)
}
pub fn deserialize_pir(bytes: &[u8]) -> Result<PirGraph, String> {
    deserialize_graph(bytes)
}
