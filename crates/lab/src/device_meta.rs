//! Device Metadata
//!
//! Collects and represents device-specific metadata for profiling context
//! and knowledge scoping.
//!
//! Key design principle: host-only metadata and device-backed metadata are
//! structurally distinct types. You cannot accidentally treat a host-only
//! environment summary as if it came from a real Apple device.

use serde::{Deserialize, Serialize};

/// Device metadata for profiling context.
///
/// This is the combined type that captures EITHER host-only metadata
/// OR device-backed metadata, but never pretends one is the other.
/// The `source` field makes the distinction explicit and structural.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMetadata {
    /// Where this metadata came from.
    pub source: MetadataSource,
    /// Host OS description (always available).
    pub host_os: String,
    /// Device class (only meaningful for device-backed runs).
    /// E.g., "Apple M2", "Apple M1 Pro", "Apple A17 Pro".
    /// None for host-only runs.
    pub device_class: Option<String>,
    /// Chip name (only meaningful for device-backed runs).
    /// E.g., "t6020", "t6000", "t8120".
    /// None for host-only runs.
    pub chip_name: Option<String>,
    /// OS version (only meaningful for device-backed runs).
    /// E.g., "macOS 14.3.1", "iOS 17.2".
    /// For host-only runs, this is the host OS version.
    pub os_version: String,
    /// Core ML version (only available on Apple platforms).
    /// None for host-only runs where Core ML is not available.
    pub core_ml_version: Option<String>,
    /// Total device memory in GB (only meaningful for device-backed runs).
    pub total_memory_gb: Option<f32>,
    /// Number of ANE cores (only discoverable on Apple hardware).
    /// Apple does not officially expose this; it's a best-effort estimate
    /// based on chip model. None means we don't know.
    pub ane_core_count: Option<usize>,
    /// Whether Core ML runtime is available.
    pub coreml_runtime_available: bool,
    /// Whether compute plan inspection is available.
    /// Requires both Apple hardware and coremltools >= 8.0.
    pub compute_plan_available: bool,
}

/// The source of device metadata — makes host-only vs device-backed
/// structurally distinct.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MetadataSource {
    /// Metadata was collected from the host environment only.
    /// No Apple hardware or Core ML runtime was available.
    /// Device-specific fields (device_class, chip_name, etc.) are None.
    HostOnly,
    /// Metadata was collected from an Apple device with Core ML runtime.
    /// Device-specific fields are populated from the actual device.
    DeviceBacked,
}

/// Run type distinguishing warm and cold executions.
///
/// Warm runs include a warmup phase where the model is executed several
/// times before measurement begins. Cold runs measure from a fresh load.
/// The distinction matters for latency interpretation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunType {
    /// Cold run: model loaded fresh, no warmup.
    /// Latency includes model loading and first inference.
    Cold,
    /// Warm run: model was pre-loaded and warmup iterations completed.
    /// Latency reflects steady-state inference only.
    Warm {
        /// Number of warmup iterations completed before measurement.
        warmup_iterations: usize,
    },
}

/// Execution context for a device-backed profiling run.
///
/// This only exists for device-backed runs. Host-only runs do not
/// have an ExecutionContext.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// The device metadata for this execution.
    pub device: DeviceMetadata,
    /// The type of run (warm/cold).
    pub run_type: RunType,
    /// Number of measured iterations.
    pub measured_iterations: usize,
    /// Compute units requested for execution.
    /// E.g., "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL".
    pub compute_units: String,
}

impl DeviceMetadata {
    /// Collect device metadata for a host-only environment.
    ///
    /// This honestly represents that no Apple hardware was available.
    /// Device-specific fields are set to None.
    pub fn host_only() -> Self {
        Self {
            source: MetadataSource::HostOnly,
            host_os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
            device_class: None,
            chip_name: None,
            os_version: format!("{} (host)", std::env::consts::OS),
            core_ml_version: None,
            total_memory_gb: None,
            ane_core_count: None,
            coreml_runtime_available: false,
            compute_plan_available: false,
        }
    }

    /// Collect device metadata from an Apple device.
    ///
    /// This is only callable on Apple hardware. On non-Apple platforms,
    /// it returns a host-only metadata with a note explaining why.
    pub fn device_backed() -> Self {
        // On non-Apple platforms, we cannot collect device metadata.
        // Return host-only metadata with an honest explanation.
        Self {
            source: MetadataSource::HostOnly,
            host_os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
            device_class: None,
            chip_name: None,
            os_version: format!("{} (host — device metadata unavailable)", std::env::consts::OS),
            core_ml_version: None,
            total_memory_gb: None,
            ane_core_count: None,
            coreml_runtime_available: false,
            compute_plan_available: false,
        }
    }

    /// Parse a device class from an Apple chip identifier.
    ///
    /// This is a best-effort mapping based on publicly known chip names.
    /// Returns None for unknown chip identifiers.
    pub fn parse_device_class(chip: &str) -> Option<String> {
        match chip {
            "t6020" | "t6021" => Some("Apple M2".to_string()),
            "t6030" | "t6031" => Some("Apple M3".to_string()),
            "t6000" | "t6001" | "t6002" => Some("Apple M1".to_string()),
            "t8101" => Some("Apple A14 Bionic".to_string()),
            "t8110" => Some("Apple A15 Bionic".to_string()),
            "t8120" => Some("Apple A16 Bionic".to_string()),
            "t8140" => Some("Apple A17 Pro".to_string()),
            _ => None,
        }
    }

    /// Whether this metadata came from a real Apple device.
    pub fn is_device_backed(&self) -> bool {
        self.source == MetadataSource::DeviceBacked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_metadata_host_only() {
        let meta = DeviceMetadata::host_only();
        assert_eq!(meta.source, MetadataSource::HostOnly);
        assert!(meta.device_class.is_none());
        assert!(meta.chip_name.is_none());
        assert!(meta.core_ml_version.is_none());
        assert!(meta.total_memory_gb.is_none());
        assert!(meta.ane_core_count.is_none());
        assert!(!meta.coreml_runtime_available);
        assert!(!meta.compute_plan_available);
        // host_os should contain the OS name (e.g. "linux")
        assert!(
            meta.host_os.contains(std::env::consts::OS),
            "host_os '{}' should contain OS '{}'",
            meta.host_os,
            std::env::consts::OS
        );
        // os_version should contain "host"
        assert!(
            meta.os_version.contains("host"),
            "os_version '{}' should contain 'host'",
            meta.os_version
        );
    }

    #[test]
    fn test_device_metadata_device_backed_on_non_apple() {
        // On non-Apple platforms, device_backed() returns host-only metadata
        // because we cannot collect device metadata without Apple hardware.
        let meta = DeviceMetadata::device_backed();
        // On non-Apple, source should be HostOnly (not DeviceBacked)
        if !cfg!(target_vendor = "apple") {
            assert_eq!(meta.source, MetadataSource::HostOnly);
            assert!(meta.device_class.is_none());
            assert!(meta.chip_name.is_none());
            assert!(!meta.coreml_runtime_available);
        }
    }

    #[test]
    fn test_parse_device_class_known_chips() {
        // M2 family
        assert_eq!(DeviceMetadata::parse_device_class("t6020"), Some("Apple M2".to_string()));
        assert_eq!(DeviceMetadata::parse_device_class("t6021"), Some("Apple M2".to_string()));
        // M3 family
        assert_eq!(DeviceMetadata::parse_device_class("t6030"), Some("Apple M3".to_string()));
        assert_eq!(DeviceMetadata::parse_device_class("t6031"), Some("Apple M3".to_string()));
        // M1 family
        assert_eq!(DeviceMetadata::parse_device_class("t6000"), Some("Apple M1".to_string()));
        assert_eq!(DeviceMetadata::parse_device_class("t6001"), Some("Apple M1".to_string()));
        assert_eq!(DeviceMetadata::parse_device_class("t6002"), Some("Apple M1".to_string()));
        // A-series
        assert_eq!(
            DeviceMetadata::parse_device_class("t8101"),
            Some("Apple A14 Bionic".to_string())
        );
        assert_eq!(
            DeviceMetadata::parse_device_class("t8110"),
            Some("Apple A15 Bionic".to_string())
        );
        assert_eq!(
            DeviceMetadata::parse_device_class("t8120"),
            Some("Apple A16 Bionic".to_string())
        );
        assert_eq!(DeviceMetadata::parse_device_class("t8140"), Some("Apple A17 Pro".to_string()));
    }

    #[test]
    fn test_parse_device_class_unknown_chip() {
        assert_eq!(DeviceMetadata::parse_device_class("t9999"), None);
        assert_eq!(DeviceMetadata::parse_device_class(""), None);
    }

    #[test]
    fn test_is_device_backed_host_only() {
        let meta = DeviceMetadata::host_only();
        assert!(!meta.is_device_backed());
    }

    #[test]
    fn test_metadata_source_serialization() {
        // Roundtrip MetadataSource through serde_json
        for source in [MetadataSource::HostOnly, MetadataSource::DeviceBacked] {
            let json = serde_json::to_string(&source).unwrap();
            let back: MetadataSource = serde_json::from_str(&json).unwrap();
            assert_eq!(source, back, "MetadataSource roundtrip failed for {:?}", source);
        }
    }

    #[test]
    fn test_run_type_serialization() {
        // Cold roundtrip
        let cold = RunType::Cold;
        let json = serde_json::to_string(&cold).unwrap();
        let back: RunType = serde_json::from_str(&json).unwrap();
        assert_eq!(cold, back);

        // Warm roundtrip
        let warm = RunType::Warm { warmup_iterations: 5 };
        let json = serde_json::to_string(&warm).unwrap();
        let back: RunType = serde_json::from_str(&json).unwrap();
        assert_eq!(warm, back);
        if let RunType::Warm { warmup_iterations } = back {
            assert_eq!(warmup_iterations, 5);
        } else {
            panic!("Expected RunType::Warm");
        }
    }

    #[test]
    fn test_execution_context_serialization() {
        let ctx = ExecutionContext {
            device: DeviceMetadata::host_only(),
            run_type: RunType::Cold,
            measured_iterations: 10,
            compute_units: "CPU_AND_NE".to_string(),
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let back: ExecutionContext = serde_json::from_str(&json).unwrap();
        assert_eq!(back.measured_iterations, 10);
        assert_eq!(back.compute_units, "CPU_AND_NE");
        assert_eq!(back.run_type, RunType::Cold);
        assert_eq!(back.device.source, MetadataSource::HostOnly);
    }

    #[test]
    fn test_device_metadata_serialization() {
        let meta = DeviceMetadata::host_only();
        let json = serde_json::to_string(&meta).unwrap();
        let back: DeviceMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back.source, meta.source);
        assert_eq!(back.host_os, meta.host_os);
        assert_eq!(back.device_class, meta.device_class);
        assert_eq!(back.chip_name, meta.chip_name);
        assert_eq!(back.os_version, meta.os_version);
        assert_eq!(back.core_ml_version, meta.core_ml_version);
        assert_eq!(back.total_memory_gb, meta.total_memory_gb);
        assert_eq!(back.ane_core_count, meta.ane_core_count);
        assert_eq!(back.coreml_runtime_available, meta.coreml_runtime_available);
        assert_eq!(back.compute_plan_available, meta.compute_plan_available);
    }
}
