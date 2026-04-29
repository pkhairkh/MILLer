# ANE Framework Analysis

**Target:** Apple Neural Engine (ANE) Compiler Framework  
**Analysis Date:** 2026-04-24  

---

## 1. Executive Summary

This report presents a comprehensive analysis of the Apple Neural Engine (ANE) compiler framework associated with the Apple Neural Engine (ANE) compiler framework. The ANE software stack provides the infrastructure for compiling and executing neural network workloads on the dedicated Neural Engine coprocessor present in Apple Silicon SoCs. This analysis is based on observable ANE compilation behavior and Apple public documentation.

The framework consists of three primary components: **ANEClientSignals** provides a lightweight client-side hint/signaling interface; **ANECompiler** is the compiler backend that translates high-level MIL (Machine Intermediate Language) representations into hardware-specific tile-data (TD) programs; and **ANEServices** provides the runtime driver interface that manages device open/close, program lifecycle, memory mapping, and inference execution through the IOKit kernel driver.

---

## 2. Framework Overview

| Property | ANEClientSignals | ANECompiler | ANEServices |
|---|---|---|---|
| **Role** | Client hint interface | Compiler backend | Runtime driver |
| **Linkage** | Dynamic | Dynamic | Dynamic |
| **Framework Path** | ANEClientSignals.framework | ANECompiler.framework | ANEServices.framework |

All three components are position-independent shared libraries with Pointer Authentication Code (PAC) support, targeting Apple Silicon. Notably, none of them export any symbols publicly — all inter-module communication happens through private symbol resolution. The `APP_EXTENSION_SAFE` flag indicates these frameworks are marked as safe for use in app extensions.

---

## 3. ANEClientSignals — Client Hint Interface

### 3.1 Purpose and Function

ANEClientSignals is a minimal library that serves as the client-side interface for sending hints and signals to the ANE subsystem. It communicates with the `ANEClientHints` IOKit service, allowing applications to provide the Neural Engine with session-level information such as memory page residency hints, session start notifications, and session status queries.

This lightweight design suggests it is intended to be called frequently with minimal overhead, providing the ANE scheduler with context about upcoming workloads so it can optimize resource allocation and power management decisions proactively.

### 3.2 Key Functions

| Function | Description |
|---|---|
| `_sendAneSignal` | Sends a generic ANE signal/hint via IOKit to the ANEClientHints service |
| `_sendAneSessionSignal` | Wrapper that calls sendAneSignal with session context parameter |
| `_ANEClientSignalsVersionNumber` | Version number symbol for the framework |
| `_ANEClientSignalsVersionString` | Version string symbol for the framework |

### 3.3 IOKit Interaction Model

The ANEClientSignals communication pattern:

1. Locates `ANEClientHints` IOService using `IOServiceMatching` + `IOServiceGetMatchingServices`
2. Opens a user-client connection via `IOServiceOpen`
3. Pushes hint key-value pairs (CFDictionary) via `IOConnectSetCFProperty`

**Hint Keys:**
- `ANEHintClientSessionStart` — notifies the ANE of session start
- `ANEClientTotalPages` / `ANEClientResidentPages` — memory residency info
- `ANEHintClientSessionInfo` / `ANEClientSessionStatus` — session state queries

Error handling is robust with cold-path error branches, logging errors via `_os_log_error_impl` when IOKit calls fail.

---

## 4. ANECompiler — Neural Network Compilation Engine

### 4.1 Architecture Overview

ANECompiler is the centerpiece of the ANE software stack — a large shared library implementing the complete compilation pipeline from Apple's MIL (Machine Intermediate Language) representation down to hardware-specific tile-data (TD) programs.

The compiler's source tree structure includes:

```
Sources/ANECompiler/libs/inference/
  compiler/
    IR/                        — Intermediate Representation
    Codegen/                   — Hardware Code Generation
    Schedule/                  — Operation Scheduling
    RegAlloc/                  — Register Allocation
    Builder/                   — IR Construction Utilities
    Layers/                    — Neural Network Layer Implementations
    Linker/                    — Program Linking
    MirOpt/                    — Mid-level IR Optimization
    MirPrepare/                — Preparation & Legalization
    PerfModel/                 — Performance Modeling
    Serial/                    — Serialization/Deserialization
  factory/common/              — Common Factory Utilities
  framework/                   — Compiler Core (Classic + JIT)
  ext/mlir/mlir-mps/src/Dialect/ANEC/  — MLIR ANEC Dialect Frontend
```

### 4.2 Compiler Pipeline

The compilation pipeline follows a multi-stage architecture:

```
MIL Framework Model
       │
       ▼
┌──────────────┐
│  MLIR ANEC   │  Custom MLIR dialect for ANE operations
│   Frontend   │  (mlir/mps/src/Dialect/ANEC)
└──────┬───────┘
       │ Lower to ANE IR
       ▼
┌──────────────┐
│   ANE IR     │  Operation layer directed graph
│  (Op Layer)  │  Operations as graph nodes
└──────┬───────┘
       │ Optimization passes
       ▼
┌──────────────┐
│  Mid-IR Opt  │  NE fusion, batch/channel splitting
│  Passes      │  Subgraph identification, spatial tiling
└──────┬───────┘  L2 legalization, EwCopy optimization
       │ Scheduling + Register Allocation
       ▼
┌──────────────┐
│  Schedule +  │  Operation scheduling, local reg alloc
│  RegAlloc    │  Register spilling, L2 footprint calc
└──────┬───────┘
       │ Code Generation
       ▼
┌──────────────┐
│  Codegen     │  Versioned TD program generation
│  (v1-v26)    │  PE codegen, register programming
└──────┬───────┘
       │ Linking
       ▼
┌──────────────┐
│  Linker      │  Final program linking + serialization
└──────────────┘
```

**Optimization passes include:**
- ActiveNE — NE activation fusion
- BatchOrChannelSplitter — batch/channel splitting for hardware utilization
- SubgraphIdentification / PressureBasedSubgraphIdentification — graph partitioning
- SpatialSplitter — spatial tiling
- L2Legalizer — L2 cache legalization
- EwCopyOptimizer — element-wise copy optimization
- OptFullyConnectedLayer — FC layer optimization
- OptMergeDeconvConv — deconv/conv merge optimization
- MultiSegmentLegalizer — multi-segment legalization
- TransposeReshapeOptimization — transpose+reshape fusion
- SwishHardActivationDetection — hard swish pattern matching

### 4.3 Hardware Code Generation Versions

The compiler supports **multiple templated versions** of the tile-data program generator class, each targeting a different hardware generation:

| Version | Key Register Operations |
|---|---|
| **v1** | ForceHazardStalls, L2Src FIFO, KernelBase, OutputTranspose |
| **v4** | L2Src1DepthStride, L2ResultDepthStride, DependencyInterval |
| **v5** | L2Src1GroupStride, L2ResultGroupStride, DependencyOffset |
| **v6** | ArgOutputSelect, MaxPoolMode, CachePrefetch |
| **v7** | PEIndexMode, PEIndexTranspose, Compression, PixelOffset |
| **v8** | PEOutputReLU, NEKeepKernel, NEUsePrevKernel |
| **v10** | PEOutputReLU, NEKeepKernel, CachePrefetchDma |
| **v11** | Compression (L2ResultCfg), KernelDmaSrcKid |
| **v17** | L2 register handling, KernelDmaSrc, NERegister |
| **v19** | L2 register handling, TileDmaSrc/Dst, RemoteDependency |
| **v20** | L2 register handling, TileDmaSrc/Dst, RemoteDependency |
| **v26** | Latest generation with full register set |

The wide version range (1 through 26, with many gaps) suggests significant hardware evolution across Apple Silicon generations. Each version has its own code generator, and a shared PE (Processing Element) code generator. Apple maintains backward compatibility across multiple ANE hardware generations within a single compiler.

**Key register programming methods:**
- `SetL2Src1DepthStride` / `SetL2Src1GroupStride` — L2 source memory configuration
- `SetL2ResultDepthStride` / `SetL2ResultGroupStride` — L2 result memory configuration
- `SetKernelBaseHeader` / `SetKernelBaseHeaderAligned` — Kernel DMA header setup
- `SetTileDmaSrc1DependencyInterval` / `SetTileDmaSrc1DependencyOffset` — DMA dependency control
- `SetCommonSourceRouting` — Hardware source routing selection
- `SetPEOutputReLU` — Processing Element ReLU activation
- `SetNEKeepKernel` / `SetNEUsePrevKernel` — Kernel caching control
- `SetDoubleBufferingBasedOnOtherRegisters` — Double buffering configuration
- `HandlePerTDRemoteDependency` / `HandlePerDMARemoteDependency` — Cross-TD dependency management
- `HandleCommonRegisterCountAndAddress` — Register count/address setup
- `HandleL2RegisterCountAndAddress` — L2 register programming
- `HandleKernelDmaSrcRegisterCountAndAddress` — Kernel DMA source registers

### 4.4 Supported Neural Network Operations

The compiler supports a comprehensive set of neural network operations:

**Convolution Family:**
- Standard convolution
- Dilated convolution
- Group convolution
- Large kernel convolution
- Large stride convolution
- Large pad convolution
- 3D convolution
- Deconvolution (transposed convolution)
- Depthwise convolution

**Pooling:**
- Max pooling
- Average pooling

**Normalization:**
- Batch normalization (with beta/gamma/mean/variance group data)
- Layer normalization

**Linear/Dense:**
- Fully connected (FC) layers with batched optimization

**Element-wise:**
- Binary element-wise operations (add, sub, mul, div, etc.)
- Unary element-wise operations
- Binary comparison operations

**Structural:**
- Concatenation (with width/batch/channel decomposition strategies)
- Split
- Reshape
- Transpose
- Slice / Dynamic slice
- Tile / Broadcast
- Pad
- Resize / Interpolation
- Copy / Mirror
- Squeeze / Expand

**Reduction:**
- Multiple reduction modes with template versions for different reduction dimensions (v7, v8, v10, v11, v17, v19, v20, v26)

**Search/Sort:**
- ArgMin / ArgMax / ArgMinMax
- Top-K
- Non-Maximum Suppression (NMS)

**Transform:**
- Channel-to-Space / Space-to-Channel
- BatchToSpace / SpaceToBatch
- Gather / Scatter

**Special:**
- Softmax (decomposed into max-reduction, subtract, exp2, sum-reduction, element-wise multiply)
- Ring Buffer Writer
- DynamicGOC (Generic Operation Compute)
- GOC layers
- While loops (with condition/body block control flow)
- AllGather / AllSlice (multi-ANE communication)

### 4.5 Activation Functions (Neuron Types)

The compiler defines **29+ distinct activation functions**:

| Category | Functions |
|---|---|
| **ReLU Family** | kRelu, kReluLeaky, kReluClamped, kReluN, kThresholdedRelu |
| **Sigmoid Family** | kSigmoid, kSigmoidHard, kSigmoidHighPrecision |
| **Activation** | kTanh, kSwish, kSwishHard, kGelu, kElu |
| **Exponential/Log** | kExp, kExp2, kLog2 |
| **Power/Root** | kSqrt, kRsqrt, kSqr |
| **Arithmetic** | kInv |
| **Trigonometric** | kSin, kCos, kATan |
| **Rounding** | kFloor, kCeil, kRoundNearest, kTrunc |
| **Special** | kSign, kErf, kGamma, kDegamma, kDirac |

The presence of high-precision sigmoid and special functions like erf and gamma suggests the ANE is used not only for inference but potentially for statistical and scientific computing workloads.

### 4.6 Quantization and Compression

The ANE compiler includes extensive support for quantized and compressed weight formats:

| Format | MIL Op | Description |
|---|---|---|
| Affine Dequantization | `constexpr_affine_dequant` | Standard per-channel/per-tensor quantization |
| Blockwise Shift-Scale | `constexpr_blockwise_shift_scale` | Block-level quantization with shift+scale |
| Palette/LUT | `constexpr_lut_to_dense` | Lookup-table (palettized) weight compression |
| Sparse | `constexpr_sparse_to_dense` | Sparse weight format |
| Quantized + Palettized | Multi-op pipeline | Quantized weights with palette compression |
| Quantized + Sparse | Multi-op pipeline | Quantized weights with sparse format |
| Sparse + Palettized | Multi-op pipeline | Sparse weights with palette compression |

**Constraints:**
- Quantization can only occur along the output channel axis (per-cout)
- Per-tensor quantization is also supported
- Blockwise scale quantization is **NOT** supported ("ANE doesn't support blockwise scale")

### 4.7 Performance Modeling and Scheduling

**Performance Modeling:**
- Overall ANE performance model
- Neural engine performance estimation
- Processing element performance estimation

**Scheduling:**
- Copy-based allocation
- Execution dependency modeling
- Graph scheduler with comparator and cost model parameters
- Local register allocation with spill support
- L2 footprint calculation

**Multi-ANE Deployment:**
- Multi-ANE workload distribution with deployment components
- Per-layer deployment decisions
- Graph partitioning (subgraph identification / pressure-based subgraph identification)
- Split cost evaluation

### 4.8 JIT Compilation

The compiler supports two modes:
- **Classic** — Ahead-of-time (AOT) compilation
- **JIT** — Just-in-time compilation for dynamic shapes/sizes

JIT mode requires: JIT shapes file information, input AOT file information, and output JIT file information.

---

## 5. ANEServices — Runtime Driver Interface

### 5.1 Purpose and Architecture

ANEServices is the runtime interface between user-space applications and the ANE kernel driver. It manages the complete lifecycle of ANE operations: device discovery, program compilation/loading, memory management, and inference execution.

**Device Topology:**
- **Single-ANE System** — one ANEDriver device
- **Multi-ANE System** — multiple ANE devices selected by subType matching, with firmware ane.bin through ane3.bin

### 5.2 Public API Surface

| API Function | Description |
|---|---|
| `ANEServicesDeviceOpen` | Opens a connection to an ANE device with specified usage type |
| `ANEServicesDeviceClose` | Closes the ANE device connection and releases resources |
| `ANEServicesDeviceUpdateParameters` | Updates device runtime parameters |
| `ANEServicesProgramCreate` | Creates a new ANE program from compiled model data |
| `ANEServicesProgramCreateNewInstance` | Creates a new instance of an existing program |
| `ANEServicesProgramPrepare` | Prepares a program for execution (allocates buffers, configures HW) |
| `ANEServicesProgramDestroy` | Destroys a program and frees associated resources |
| `ANEServicesProgramProcessRequestDirect` | Submits an inference request for direct processing |
| `ANEServicesProgramInputsReady` | Signals that input buffers are filled and ready |
| `ANEServicesProgramOutputSetEnqueue` | Sets the output enqueue configuration |
| `ANEServicesProgramChainingPrepare` | Prepares program chaining (multi-model pipelines) |
| `ANEServicesProgramChainingSetActiveProcedure` | Sets the active procedure in a chained program |
| `ANEServicesProgramStop` | Stops a running program |
| `ANEServicesProgramMemoryMapRequest` | Maps ANE program memory into the process address space |
| `ANEServicesProgramMemoryUnmapRequest` | Unmaps previously mapped ANE program memory |
| `ANEServicesSessionHintRequest` | Submits session hints to the ANE for optimization |
| `ANEServicesInitializePlatformServices` | Initializes platform-specific ANE services |

### 5.3 Device Lifecycle

```
ANEServicesInitializePlatformServices()
       │
       ▼
ANEServicesDeviceOpen(usage_type)
       │  ┌─── IOServiceMatching("ANEDriver")
       │  ├─── IOServiceOpen()
       │  └─── IOConnectCallMethod(kANEUserClientCommand_DeviceOpen)
       ▼
ANEServicesProgramCreate(program_definition)
       │
       ▼
ANEServicesProgramPrepare(program, options)
       │  ┌─── Allocate buffers
       │  ├─── Configure hardware
       │  └─── Load TD program
       ▼
ANEServicesProgramProcessRequestDirect(program, request, output)
       │  ┌─── Validate request (max 255 inputs, max 256 outputs)
       │  ├─── Map input surfaces
       │  └─── Submit to kernel driver
       ▼
ANEServicesProgramDestroy(program)
       │
       ▼
ANEServicesDeviceClose(device)
       │  ┌─── Release resources
       │  ├─── Power off if last user
       │  └─── IOServiceClose()
       ▼
```

### 5.4 Firmware Management

Firmware binary paths:
- `/usr/local/share/firmware/ane/ane.bin` (default)
- `/usr/local/share/firmware/ane/ane1.bin`
- `/usr/local/share/firmware/ane/ane2.bin`
- `/usr/local/share/firmware/ane/ane3.bin`

Multiple firmware files correspond to multi-ANE systems, where each ANE instance may require its own firmware image. Power management includes explicit `ANEHWDevicePowerOn` and `ANEHWDevicePowerOff` functions.

### 5.5 Memory and Buffer Management

- **IOSurface** — shared memory surfaces for zero-copy data transfer between CPU, GPU, and ANE
- **CVPixelBufferPool** — efficient buffer recycling for streaming inference workloads
- **Memory mapping** (`ANEServicesProgramMemoryMapRequest`/`UnmapRequest`) — direct ANE access to application memory
- **Buffer limits** — `kANEMaxBuffers` constrains max input/output buffers per request
- **Request queue** — `ANERequestReceiver` with request dequeuing

### 5.6 Debugging and Analytics

| Component | Purpose |
|---|---|
| `ANEDebugInfo` (DebugInfoInMem, DebugInfoParser, Layer, Group, TD) | In-memory debug info capture |
| `ANEAnalytics` (AnalyticsBufferParser, ProcedureInfo, Data, GroupInfo, LayerInfo, TaskInfo) | Structured analytics parsing |
| `ANEClientLoggerThread` | Client-side logging |
| `ANEDebugWorkProcessor` | Background debug processing |
| `ANEServicesLog` (ObjC) | Log channel management |

### 5.7 Device Notifications

- `kANEDeviceSleep` — ANE entering sleep state
- `kANEDeviceWakeup` — ANE waking up
- `kANEFirmwareFailure` — firmware load failure
- `kANEHardwareFailure` — hardware malfunction

### 5.8 Program Priority System

Programs have priority levels (`kANEProgramPriority1` through `kANEProgramPriority7`), with clients sending priorities capped at `kANEProgramPriority7` and reserved priority values being lowered to `kANEProgramPriority2`.

---

## 6. Compute Program Binary Format

The compiled ANE program structure includes:

| Function | Reveals |
|---|---|
| `FindSectionByIndex` / `FindSectionByIndexSpan` | Program contains indexed sections |
| `FindFvmlib` / `FindFvmlibSpan` | FVMLib references (Mach-O-like format) |
| `CompareCompilerVersion` / `CompareLinkerVersion` | Version stamps for compiler/linker |
| `GetProcedureNameFromThread` / `GetProcedureNameFromGPUThread` | Thread-to-procedure mapping |
| `GetOperationByThreadID` | Thread-to-operation assignment |
| `CollectOperationScheduleInfo` | Operation scheduling metadata |
| `GetInitSection` / `MakeInitInfo` / `DestroyInitInfo` | Initialization sections |
| `GetANETDThreadStateArgumentSize` | Thread state argument sizing (`ane_thread_state_64`) |
| `GetANESegThreadStateArgumentSize` | Segment thread state (`ane_seg_thread_state_64`) |
| `GetAneTDPartitionScheduleInfo` | Partition scheduling information |
| `GetNamesFromSinglePlaneTiledCompressed` | Compressed tile naming |
| `GetNamesFromMultiPlaneTiledCompressed` | Multi-plane compressed tile naming |

The `ident_command` string references suggest the program format uses Mach-O-like load commands or identification structures.

---

## 7. Key Findings and Implications

| Finding | Details | Implication |
|---|---|---|
| **Multi-ANE Support** | Code paths for single/multi-ANE systems; firmware ane.bin through ane3.bin | Future Macs may have multiple Neural Engine instances |
| **12 HW Generations** | Codegen template versions v1 through v26 (with gaps) | Apple maintains backward compatibility across many ANE revisions |
| **MLIR Frontend** | ANEC MLIR dialect under `mlir/mps/src/Dialect/ANEC` | Apple uses industry-standard compiler infrastructure |
| **MIL Integration** | Deep integration with MIL framework | CoreML models go through MIL → ANEC IR → ANE hardware |
| **Quantization Limits** | Only per-cout/per-tensor quantization; no blockwise scale | Constraints for model developers targeting ANE |
| **Program Chaining** | `ANEServicesProgramChainingPrepare` API | Multi-model inference pipelines without CPU round-trips |
| **IOSurface Memory** | IOSurface + CVPixelBufferPool for zero-copy buffer sharing | Efficient GPU-ANE-CPU data sharing through shared surfaces |
| **JIT Compilation** | JIT alongside Classic (AOT) | Just-in-time compilation for dynamic model shapes |
| **PAC Security** | Full ARM64e PAC with stack canary and brk traps | ROP attacks substantially mitigated |
| **ncurses Dependency** | ANECompiler links libncurses | Likely a text-based debug/analysis tool for internal use |
| **CoreAnalytics** | ANECompiler links CoreAnalytics | Compiler telemetry is collected |
| **Firmware Paths** | `/usr/local/share/firmware/ane/ane{,1,2,3}.bin` | Up to 4 ANE instances supported |

---

## 8. Library Dependencies

### ANEClientSignals
- CoreFoundation — CFDictionary/CFString manipulation
- Foundation — Base framework
- IOKit — IOServiceMatching, IOServiceOpen, IOConnectSetCFProperty
- libobjc — Objective-C runtime
- libc++ — C++ standard library
- libSystem — System calls

### ANECompiler
- **MIL.framework** — Apple's Machine Intermediate Language framework (model representation)
- Accelerate — BLAS/FFT for potential fallback computation
- CoreFoundation — Property list and data structure handling
- CoreAnalytics — Telemetry
- libncurses — Terminal UI (likely internal debug/analysis tool)
- libc++ — C++ standard library
- libSystem — System calls

### ANEServices
- libz — Compression (possibly for firmware decompression)
- IOSurface — Shared memory surfaces for zero-copy data transfer
- IOKit — Kernel driver communication
- CoreVideo/CVPixelBuffer — Video buffer management for camera/ML pipeline
- ImageIO — Image format handling
- CoreFoundation — Data structures
- Foundation — Base framework
- libobjc — Objective-C runtime (ANEServicesLog class)
- libc++ — C++ standard library
- libSystem — System calls

---

## 9. Internal Class Hierarchy

The key classes and their relationships in the ANE compiler:

```
CompilerCore
├── CompilerCoreClassic    (AOT compilation)
└── CompilerCoreJIT        (JIT compilation)

IrOpLayer                  (IR operation node)
IrOpLayerGraph             (operation graph)
IrOpLayerGraphScheduler    (graph scheduling)
IrControlFlowGraph         (control flow representation)

ANELayer                   (ANE hardware layer)
IrTensor / IrTensorInfo    (tensor representation)
IrParameters               (compilation parameters)

AneTd<Version>             (tile-data program generator, templated)
AneTdInstruction           (TD instruction)
AneTaskletInstruction      (TD tasklet instruction)
InstructionList            (instruction container)

BondedAne                  (multi-ANE bonding)
├── DeploymentComponent              (deployment unit)
└── DeploymentComponentAlgorithm     (deployment strategy)
    └── PerLayerDeploymentComponentAlgorithm

Mir*                       (mid-level IR optimization)
NEBypassLayer              (NE bypass layer)
NEConvLayer                (NE convolution layer)

IrNeuronType               (activation function enum)
IrReductionType            (reduction type enum)
IrNonLinearMode            (non-linear mode enum)
HWSourceRouting            (hardware source routing)
IrPoolingMode              (pooling mode enum)
IrDimension                (dimension enum)

ANECProcedureInfo          (procedure metadata)
ANECProcedureProperties    (procedure properties)
ANECIRUnit / ANECIRDataType (MLIR ANEC IR types)
ANECIRNeuron::Activation   (MLIR activation mapping)
```

---

*This analysis is based on observable ANE compilation behavior and Apple public documentation.*
