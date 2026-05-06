//! Multi-ANE Device and Firmware Model (T-D-04 / F-FW-01)
//!
//! This module models the multi-ANE capabilities discovered through
//! binary forensic analysis. The ANEC binary reveals:
//! - Multi-ANE device enumeration (up to 16 NEs across multiple ANE instances)
//! - 4 firmware images per ANE instance (boot, runtime, debug, recovery)
//! - SubType matching for chip-specific firmware selection
//! - Program chaining for multi-ANE execution sequences
//!
//! This is a foundational model — actual firmware loading and multi-ANE
//! scheduling require hardware integration that is out of current scope.

use crate::ane_hw_limits::AneSubVariant;
use crate::ane_target::AneRevision;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A single ANE device instance in a multi-ANE system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneDevice {
    /// Unique device identifier within the system.
    pub device_id: u32,
    /// ANE revision for this device.
    pub revision: AneRevision,
    /// Sub-variant (chip-level SKU).
    pub sub_variant: AneSubVariant,
    /// Number of NEs on this device.
    pub num_nes: u32,
    /// Firmware images for this device.
    pub firmware: AneFirmwareSet,
    /// Whether this device is the primary ANE.
    pub is_primary: bool,
}

/// Set of firmware images for an ANE device.
/// The ANEC binary loads 4 firmware images per ANE instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneFirmwareSet {
    /// Boot firmware — loaded at ANE power-on.
    pub boot: FirmwareImage,
    /// Runtime firmware — loaded for normal operation.
    pub runtime: FirmwareImage,
    /// Debug firmware — loaded for diagnostics/testing.
    pub debug: FirmwareImage,
    /// Recovery firmware — loaded when runtime firmware fails.
    pub recovery: FirmwareImage,
}

/// A firmware image for an ANE device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirmwareImage {
    /// Firmware version string (e.g., "2024.1.0").
    pub version: String,
    /// Path to the firmware binary relative to the ANEC bundle.
    pub path: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// SHA-256 hash of the firmware binary.
    pub content_hash: String,
    /// SubType this firmware is compatible with.
    /// Empty string means compatible with all sub-types.
    pub compatible_sub_type: String,
}

/// SubType matching descriptor.
/// The ANEC binary uses subType strings to match firmware and
/// hardware configurations to specific chip variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTypeDescriptor {
    /// SubType identifier string (e.g., "t6020", "t8112").
    pub sub_type: String,
    /// The ANE revision this sub-type maps to.
    pub revision: AneRevision,
    /// The HAL sub-variant this sub-type maps to.
    pub hal_variant: AneSubVariant,
    /// Number of ANE instances on this chip.
    pub ane_count: u32,
    /// Total NEs across all ANE instances.
    pub total_nes: u32,
}

/// A chained program for multi-ANE execution.
/// Program chaining allows splitting a model across multiple ANE
/// instances, with intermediate results transferred between them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainedProgram {
    /// Unique identifier for this chain.
    pub chain_id: String,
    /// The sequence of program segments, one per ANE device.
    pub segments: Vec<ProgramSegment>,
    /// Intermediate data transfer descriptors between segments.
    pub transfers: Vec<InterDeviceTransfer>,
}

/// A segment of a chained program, targeting a single ANE device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramSegment {
    /// Target device ID.
    pub device_id: u32,
    /// Ops assigned to this segment (MIR node IDs).
    pub op_names: Vec<String>,
    /// Input tensor names for this segment.
    pub inputs: Vec<String>,
    /// Output tensor names for this segment.
    pub outputs: Vec<String>,
}

/// Intermediate data transfer between ANE devices in a chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterDeviceTransfer {
    /// Source device ID.
    pub source_device_id: u32,
    /// Destination device ID.
    pub dest_device_id: u32,
    /// Tensor name being transferred.
    pub tensor_name: String,
    /// Transfer method.
    pub method: TransferMethod,
}

/// Method for transferring data between ANE devices.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferMethod {
    /// Direct DMA transfer between ANE devices.
    DirectDma,
    /// Transfer via shared system memory (CPU bounce buffer).
    SharedMemory,
    /// Transfer via on-chip interconnect (fastest, requires adjacent devices).
    OnChipInterconnect,
}

/// A multi-ANE system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAneSystem {
    /// All ANE devices in the system.
    pub devices: Vec<AneDevice>,
    /// Known sub-type descriptors.
    pub sub_types: Vec<SubTypeDescriptor>,
}

impl MultiAneSystem {
    /// Create a single-ANE system (most mobile devices).
    pub fn single_ane(revision: AneRevision, sub_variant: AneSubVariant, num_nes: u32) -> Self {
        let device = AneDevice {
            device_id: 0,
            revision,
            sub_variant,
            num_nes,
            firmware: AneFirmwareSet::default_for(revision),
            is_primary: true,
        };
        Self { devices: vec![device], sub_types: vec![] }
    }

    /// Create a multi-ANE system for Apple Silicon Macs.
    pub fn mac_multi_ane(revision: AneRevision, num_devices: u32, nes_per_device: u32) -> Self {
        let devices: Vec<AneDevice> = (0..num_devices)
            .map(|id| AneDevice {
                device_id: id,
                revision,
                sub_variant: AneSubVariant::Mac,
                num_nes: nes_per_device,
                firmware: AneFirmwareSet::default_for(revision),
                is_primary: id == 0,
            })
            .collect();
        Self { devices, sub_types: vec![] }
    }

    /// Get the primary ANE device.
    pub fn primary_device(&self) -> Option<&AneDevice> {
        self.devices.iter().find(|d| d.is_primary)
    }

    /// Total NEs across all devices.
    pub fn total_nes(&self) -> u32 {
        self.devices.iter().map(|d| d.num_nes).sum()
    }

    /// Find a device by ID.
    pub fn device_by_id(&self, id: u32) -> Option<&AneDevice> {
        self.devices.iter().find(|d| d.device_id == id)
    }

    /// Validate a chained program against this system.
    pub fn validate_chain(&self, chain: &ChainedProgram) -> Result<(), ChainValidationError> {
        for segment in &chain.segments {
            if self.device_by_id(segment.device_id).is_none() {
                return Err(ChainValidationError::UnknownDevice { device_id: segment.device_id });
            }
        }
        for transfer in &chain.transfers {
            if self.device_by_id(transfer.source_device_id).is_none() {
                return Err(ChainValidationError::UnknownDevice {
                    device_id: transfer.source_device_id,
                });
            }
            if self.device_by_id(transfer.dest_device_id).is_none() {
                return Err(ChainValidationError::UnknownDevice {
                    device_id: transfer.dest_device_id,
                });
            }
        }
        Ok(())
    }
}

/// Errors that can occur when validating a chained program against a system.
#[derive(Debug, Clone)]
pub enum ChainValidationError {
    /// A segment or transfer references a device ID that doesn't exist.
    UnknownDevice { device_id: u32 },
    /// A transfer between two devices is invalid for the given reason.
    InvalidTransfer { source: u32, dest: u32, reason: String },
}

impl AneFirmwareSet {
    /// Create a default firmware set for a given ANE revision.
    /// The paths follow the ANEC binary's expected structure.
    pub fn default_for(revision: AneRevision) -> Self {
        let rev_str = format!("{:?}", revision).to_lowercase();
        Self {
            boot: FirmwareImage {
                version: "0.0.0".into(),
                path: format!("firmware/{}/boot.bin", rev_str),
                size_bytes: 0,
                content_hash: String::new(),
                compatible_sub_type: String::new(),
            },
            runtime: FirmwareImage {
                version: "0.0.0".into(),
                path: format!("firmware/{}/runtime.bin", rev_str),
                size_bytes: 0,
                content_hash: String::new(),
                compatible_sub_type: String::new(),
            },
            debug: FirmwareImage {
                version: "0.0.0".into(),
                path: format!("firmware/{}/debug.bin", rev_str),
                size_bytes: 0,
                content_hash: String::new(),
                compatible_sub_type: String::new(),
            },
            recovery: FirmwareImage {
                version: "0.0.0".into(),
                path: format!("firmware/{}/recovery.bin", rev_str),
                size_bytes: 0,
                content_hash: String::new(),
                compatible_sub_type: String::new(),
            },
        }
    }
}

// ─── Display Implementations ──────────────────────────────────────────

impl fmt::Display for AneDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ANE#{} {:?}/{:?} ({} NEs{})",
            self.device_id,
            self.revision,
            self.sub_variant,
            self.num_nes,
            if self.is_primary { " [primary]" } else { "" }
        )
    }
}

impl fmt::Display for FirmwareImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (v={}, {} bytes, sub_type={})",
            self.path,
            self.version,
            self.size_bytes,
            if self.compatible_sub_type.is_empty() { "*" } else { &self.compatible_sub_type }
        )
    }
}

impl fmt::Display for TransferMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransferMethod::DirectDma => write!(f, "DirectDMA"),
            TransferMethod::SharedMemory => write!(f, "SharedMemory"),
            TransferMethod::OnChipInterconnect => write!(f, "OnChipInterconnect"),
        }
    }
}

impl fmt::Display for ChainedProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Chain '{}':", self.chain_id)?;
        for (i, seg) in self.segments.iter().enumerate() {
            writeln!(
                f,
                "  Segment {}: device={}, ops={:?}, inputs={:?}, outputs={:?}",
                i, seg.device_id, seg.op_names, seg.inputs, seg.outputs
            )?;
        }
        for xfer in &self.transfers {
            writeln!(
                f,
                "  Transfer: {} -> {} '{}' via {}",
                xfer.source_device_id, xfer.dest_device_id, xfer.tensor_name, xfer.method
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for MultiAneSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "MultiANE System ({} device{}, {} total NEs)",
            self.devices.len(),
            if self.devices.len() == 1 { "" } else { "s" },
            self.total_nes()
        )?;
        for dev in &self.devices {
            writeln!(f, "  {}", dev)?;
        }
        if !self.sub_types.is_empty() {
            writeln!(f, "  SubTypes:")?;
            for st in &self.sub_types {
                writeln!(
                    f,
                    "    {} -> {:?}/{:?} ({} ANEs, {} NEs)",
                    st.sub_type, st.revision, st.hal_variant, st.ane_count, st.total_nes
                )?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for ChainValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChainValidationError::UnknownDevice { device_id } => {
                write!(f, "unknown device ID {}", device_id)
            }
            ChainValidationError::InvalidTransfer { source, dest, reason } => {
                write!(f, "invalid transfer {} -> {}: {}", source, dest, reason)
            }
        }
    }
}

impl std::error::Error for ChainValidationError {}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_ane_system_creation() {
        let sys = MultiAneSystem::single_ane(AneRevision::V10, AneSubVariant::Standard, 4);
        assert_eq!(sys.devices.len(), 1);
        assert_eq!(sys.devices[0].device_id, 0);
        assert_eq!(sys.devices[0].revision, AneRevision::V10);
        assert_eq!(sys.devices[0].sub_variant, AneSubVariant::Standard);
        assert_eq!(sys.devices[0].num_nes, 4);
        assert!(sys.devices[0].is_primary);
        assert!(sys.sub_types.is_empty());
    }

    #[test]
    fn test_mac_multi_ane_system_creation() {
        let sys = MultiAneSystem::mac_multi_ane(AneRevision::V17, 2, 6);
        assert_eq!(sys.devices.len(), 2);
        // First device is primary
        assert!(sys.devices[0].is_primary);
        assert_eq!(sys.devices[0].device_id, 0);
        assert_eq!(sys.devices[0].sub_variant, AneSubVariant::Mac);
        assert_eq!(sys.devices[0].num_nes, 6);
        // Second device is not primary
        assert!(!sys.devices[1].is_primary);
        assert_eq!(sys.devices[1].device_id, 1);
        assert_eq!(sys.devices[1].revision, AneRevision::V17);
    }

    #[test]
    fn test_primary_device() {
        // Single-ANE system
        let sys = MultiAneSystem::single_ane(AneRevision::V11, AneSubVariant::Pro, 4);
        let primary = sys.primary_device().unwrap();
        assert_eq!(primary.device_id, 0);
        assert!(primary.is_primary);

        // Multi-ANE system: only device 0 is primary
        let sys = MultiAneSystem::mac_multi_ane(AneRevision::V17, 3, 6);
        let primary = sys.primary_device().unwrap();
        assert_eq!(primary.device_id, 0);
        assert!(primary.is_primary);
        // Devices 1 and 2 are not primary
        assert!(!sys.devices[1].is_primary);
        assert!(!sys.devices[2].is_primary);
    }

    #[test]
    fn test_total_nes() {
        // Single device with 4 NEs
        let sys = MultiAneSystem::single_ane(AneRevision::V10, AneSubVariant::Standard, 4);
        assert_eq!(sys.total_nes(), 4);

        // Multi device: 2 * 6 = 12
        let sys = MultiAneSystem::mac_multi_ane(AneRevision::V17, 2, 6);
        assert_eq!(sys.total_nes(), 12);

        // Multi device: 3 * 8 = 24
        let sys = MultiAneSystem::mac_multi_ane(AneRevision::Vu1, 3, 8);
        assert_eq!(sys.total_nes(), 24);
    }

    #[test]
    fn test_device_by_id() {
        let sys = MultiAneSystem::mac_multi_ane(AneRevision::V17, 3, 6);

        // Existing devices
        assert!(sys.device_by_id(0).is_some());
        assert!(sys.device_by_id(1).is_some());
        assert!(sys.device_by_id(2).is_some());

        // Non-existing device
        assert!(sys.device_by_id(3).is_none());
        assert!(sys.device_by_id(99).is_none());

        // Verify correct device is returned
        let dev = sys.device_by_id(1).unwrap();
        assert_eq!(dev.device_id, 1);
    }

    #[test]
    fn test_validate_chain_valid() {
        let sys = MultiAneSystem::mac_multi_ane(AneRevision::V17, 2, 6);

        let chain = ChainedProgram {
            chain_id: "test_chain".into(),
            segments: vec![
                ProgramSegment {
                    device_id: 0,
                    op_names: vec!["conv1".into()],
                    inputs: vec!["input".into()],
                    outputs: vec!["intermediate".into()],
                },
                ProgramSegment {
                    device_id: 1,
                    op_names: vec!["conv2".into()],
                    inputs: vec!["intermediate".into()],
                    outputs: vec!["output".into()],
                },
            ],
            transfers: vec![InterDeviceTransfer {
                source_device_id: 0,
                dest_device_id: 1,
                tensor_name: "intermediate".into(),
                method: TransferMethod::DirectDma,
            }],
        };

        assert!(sys.validate_chain(&chain).is_ok());
    }

    #[test]
    fn test_validate_chain_unknown_device_in_segment() {
        let sys = MultiAneSystem::single_ane(AneRevision::V10, AneSubVariant::Standard, 4);

        let chain = ChainedProgram {
            chain_id: "bad_chain".into(),
            segments: vec![ProgramSegment {
                device_id: 5, // doesn't exist
                op_names: vec!["conv1".into()],
                inputs: vec!["input".into()],
                outputs: vec!["output".into()],
            }],
            transfers: vec![],
        };

        let result = sys.validate_chain(&chain);
        assert!(result.is_err());
        match result.unwrap_err() {
            ChainValidationError::UnknownDevice { device_id } => {
                assert_eq!(device_id, 5);
            }
            other => panic!("Expected UnknownDevice, got {:?}", other),
        }
    }

    #[test]
    fn test_validate_chain_unknown_device_in_transfer_source() {
        let sys = MultiAneSystem::single_ane(AneRevision::V10, AneSubVariant::Standard, 4);

        let chain = ChainedProgram {
            chain_id: "bad_transfer".into(),
            segments: vec![ProgramSegment {
                device_id: 0,
                op_names: vec!["conv1".into()],
                inputs: vec!["input".into()],
                outputs: vec!["output".into()],
            }],
            transfers: vec![InterDeviceTransfer {
                source_device_id: 3, // doesn't exist
                dest_device_id: 0,
                tensor_name: "data".into(),
                method: TransferMethod::SharedMemory,
            }],
        };

        let result = sys.validate_chain(&chain);
        assert!(result.is_err());
        match result.unwrap_err() {
            ChainValidationError::UnknownDevice { device_id } => {
                assert_eq!(device_id, 3);
            }
            other => panic!("Expected UnknownDevice, got {:?}", other),
        }
    }

    #[test]
    fn test_validate_chain_unknown_device_in_transfer_dest() {
        let sys = MultiAneSystem::single_ane(AneRevision::V10, AneSubVariant::Standard, 4);

        let chain = ChainedProgram {
            chain_id: "bad_dest".into(),
            segments: vec![ProgramSegment {
                device_id: 0,
                op_names: vec!["conv1".into()],
                inputs: vec!["input".into()],
                outputs: vec!["output".into()],
            }],
            transfers: vec![InterDeviceTransfer {
                source_device_id: 0,
                dest_device_id: 7, // doesn't exist
                tensor_name: "data".into(),
                method: TransferMethod::OnChipInterconnect,
            }],
        };

        let result = sys.validate_chain(&chain);
        assert!(result.is_err());
        match result.unwrap_err() {
            ChainValidationError::UnknownDevice { device_id } => {
                assert_eq!(device_id, 7);
            }
            other => panic!("Expected UnknownDevice, got {:?}", other),
        }
    }

    #[test]
    fn test_firmware_set_default_for() {
        let fw = AneFirmwareSet::default_for(AneRevision::V10);

        // Check that paths use the lowercase revision string
        assert_eq!(fw.boot.path, "firmware/v10/boot.bin");
        assert_eq!(fw.runtime.path, "firmware/v10/runtime.bin");
        assert_eq!(fw.debug.path, "firmware/v10/debug.bin");
        assert_eq!(fw.recovery.path, "firmware/v10/recovery.bin");

        // Check defaults
        assert_eq!(fw.boot.version, "0.0.0");
        assert_eq!(fw.boot.size_bytes, 0);
        assert_eq!(fw.boot.content_hash, "");
        assert_eq!(fw.boot.compatible_sub_type, "");

        // Test another revision
        let fw_v17 = AneFirmwareSet::default_for(AneRevision::V17);
        assert_eq!(fw_v17.boot.path, "firmware/v17/boot.bin");

        let fw_vu1 = AneFirmwareSet::default_for(AneRevision::Vu1);
        assert_eq!(fw_vu1.runtime.path, "firmware/vu1/runtime.bin");
    }

    #[test]
    fn test_chained_program_construction() {
        let chain = ChainedProgram {
            chain_id: "resnet_split".into(),
            segments: vec![
                ProgramSegment {
                    device_id: 0,
                    op_names: vec!["conv1".into(), "relu1".into()],
                    inputs: vec!["image".into()],
                    outputs: vec!["feat1".into()],
                },
                ProgramSegment {
                    device_id: 1,
                    op_names: vec!["conv2".into(), "relu2".into()],
                    inputs: vec!["feat1".into()],
                    outputs: vec!["feat2".into()],
                },
            ],
            transfers: vec![InterDeviceTransfer {
                source_device_id: 0,
                dest_device_id: 1,
                tensor_name: "feat1".into(),
                method: TransferMethod::OnChipInterconnect,
            }],
        };

        assert_eq!(chain.chain_id, "resnet_split");
        assert_eq!(chain.segments.len(), 2);
        assert_eq!(chain.transfers.len(), 1);

        // Verify segment details
        assert_eq!(chain.segments[0].device_id, 0);
        assert_eq!(chain.segments[0].op_names, vec!["conv1", "relu1"]);
        assert_eq!(chain.segments[1].device_id, 1);
        assert_eq!(chain.segments[1].inputs, vec!["feat1"]);

        // Verify transfer details
        let xfer = &chain.transfers[0];
        assert_eq!(xfer.source_device_id, 0);
        assert_eq!(xfer.dest_device_id, 1);
        assert_eq!(xfer.tensor_name, "feat1");
        assert_eq!(xfer.method, TransferMethod::OnChipInterconnect);
    }

    #[test]
    fn test_transfer_method_equality() {
        assert_eq!(TransferMethod::DirectDma, TransferMethod::DirectDma);
        assert_eq!(TransferMethod::SharedMemory, TransferMethod::SharedMemory);
        assert_eq!(TransferMethod::OnChipInterconnect, TransferMethod::OnChipInterconnect);
        assert_ne!(TransferMethod::DirectDma, TransferMethod::SharedMemory);
    }

    #[test]
    fn test_sub_type_descriptor() {
        let desc = SubTypeDescriptor {
            sub_type: "t6020".into(),
            revision: AneRevision::V17,
            hal_variant: AneSubVariant::Mac,
            ane_count: 2,
            total_nes: 12,
        };

        assert_eq!(desc.sub_type, "t6020");
        assert_eq!(desc.revision, AneRevision::V17);
        assert_eq!(desc.hal_variant, AneSubVariant::Mac);
        assert_eq!(desc.ane_count, 2);
        assert_eq!(desc.total_nes, 12);
    }

    #[test]
    fn test_display_ane_device() {
        let dev = AneDevice {
            device_id: 0,
            revision: AneRevision::V10,
            sub_variant: AneSubVariant::Standard,
            num_nes: 4,
            firmware: AneFirmwareSet::default_for(AneRevision::V10),
            is_primary: true,
        };
        let s = format!("{}", dev);
        assert!(s.contains("ANE#0"));
        assert!(s.contains("[primary]"));
        assert!(s.contains("4 NEs"));

        // Non-primary
        let dev2 = AneDevice {
            device_id: 1,
            revision: AneRevision::V10,
            sub_variant: AneSubVariant::Standard,
            num_nes: 4,
            firmware: AneFirmwareSet::default_for(AneRevision::V10),
            is_primary: false,
        };
        let s2 = format!("{}", dev2);
        assert!(!s2.contains("[primary]"));
    }

    #[test]
    fn test_display_firmware_image() {
        let fw = FirmwareImage {
            version: "1.2.3".into(),
            path: "firmware/v10/runtime.bin".into(),
            size_bytes: 4096,
            content_hash: "abc123".into(),
            compatible_sub_type: String::new(),
        };
        let s = format!("{}", fw);
        assert!(s.contains("firmware/v10/runtime.bin"));
        assert!(s.contains("v=1.2.3"));
        assert!(s.contains("4096 bytes"));
        assert!(s.contains("sub_type=*"));

        // With specific sub_type
        let fw2 = FirmwareImage {
            version: "1.0.0".into(),
            path: "firmware/v10/boot.bin".into(),
            size_bytes: 2048,
            content_hash: String::new(),
            compatible_sub_type: "H14g".into(),
        };
        let s2 = format!("{}", fw2);
        assert!(s2.contains("sub_type=H14g"));
    }

    #[test]
    fn test_display_transfer_method() {
        assert_eq!(format!("{}", TransferMethod::DirectDma), "DirectDMA");
        assert_eq!(format!("{}", TransferMethod::SharedMemory), "SharedMemory");
        assert_eq!(format!("{}", TransferMethod::OnChipInterconnect), "OnChipInterconnect");
    }

    #[test]
    fn test_display_multi_ane_system() {
        let sys = MultiAneSystem::mac_multi_ane(AneRevision::V17, 2, 6);
        let s = format!("{}", sys);
        assert!(s.contains("MultiANE System"));
        assert!(s.contains("2 devices"));
        assert!(s.contains("12 total NEs"));
    }

    #[test]
    fn test_display_chain_validation_error() {
        let err = ChainValidationError::UnknownDevice { device_id: 42 };
        assert!(format!("{}", err).contains("42"));

        let err2 = ChainValidationError::InvalidTransfer {
            source: 1,
            dest: 5,
            reason: "not adjacent".into(),
        };
        let s = format!("{}", err2);
        assert!(s.contains("1 -> 5"));
        assert!(s.contains("not adjacent"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let sys = MultiAneSystem::mac_multi_ane(AneRevision::V17, 2, 6);
        let json = serde_json::to_string(&sys).expect("serialization failed");
        let deserialized: MultiAneSystem =
            serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(deserialized.devices.len(), 2);
        assert_eq!(deserialized.total_nes(), 12);
    }

    #[test]
    fn test_chained_program_serialization() {
        let chain = ChainedProgram {
            chain_id: "test".into(),
            segments: vec![ProgramSegment {
                device_id: 0,
                op_names: vec!["op1".into()],
                inputs: vec!["in".into()],
                outputs: vec!["out".into()],
            }],
            transfers: vec![InterDeviceTransfer {
                source_device_id: 0,
                dest_device_id: 1,
                tensor_name: "mid".into(),
                method: TransferMethod::DirectDma,
            }],
        };
        let json = serde_json::to_string(&chain).expect("serialization failed");
        let deserialized: ChainedProgram =
            serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(deserialized.chain_id, "test");
        assert_eq!(deserialized.segments.len(), 1);
        assert_eq!(deserialized.transfers.len(), 1);
        assert_eq!(deserialized.transfers[0].method, TransferMethod::DirectDma);
    }
}
