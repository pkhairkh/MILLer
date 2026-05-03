# Worklog Entry — T-24

**Task ID:** T-24
**Agent:** main
**Task:** Fix V6 (A13 Silicon) → A14 Family Mapping (I-03, AUDIT B-3/CQ-13)

## Work Log
- Read ane_target.rs, ane_hw_limits.rs, versioned.rs, strategy.rs, CLI main.rs, dtype_constraints.rs
- Read ANE constraint docs to understand A12/A13/A14 differences
- Searched all code that pattern-matches on AneFamily for impact analysis
- Added AneFamily::A13 variant with correct constraint profile
- Mapped AneRevision::V6 → AneFamily::A13 (was incorrectly grouped with V7 under A14)
- Added uses_a14minus_converters() and supports_reducemin_all_dtypes() methods
- Updated ReduceMin gate in versioned.rs to use supports_reducemin_all_dtypes()
- Updated all 6 exhaustive match sites in versioned.rs
- Updated strategy.rs KvCache masked_blend benefit
- Updated CLI parse_ane_family with A13 mapping, iPhone aliases, corrected Mac mapping
- Fixed chip comments throughout
- Updated dtype_constraints.rs tests to include A13
- Added 7 new tests for A13 constraint profile
- All 660+ tests pass

## Stage Summary
- AneFamily::A13 variant added, V6→A13 mapping corrected
- All P0 (CRITICAL) issues now resolved (I-01, I-02, I-03)
- Mac chip-to-family mapping corrected (M1→A14, M2→A15, M3→A16, M4→A18)
