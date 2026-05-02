# MILLer Worklog

---
Task ID: 1
Agent: Main Agent
Task: Comprehensive codebase audit and issue resolution

Work Log:
- Read all source files in crates/passes, crates/coreml-proto, crates/bridge, crates/trace, crates/cli
- Identified ISSUE-001 as the root cause: ANE-illegal mask ops (Equal/Cast/LessEqual/Fill/Select)
- Identified ISSUE-006: hardcoded Qwen3 values (rope_theta, uses_rope, has_qk_norm)
- Identified ISSUE-011: Where→Select rewrite (both CPU-only)
- Identified ISSUE-013: deprecated for_qwen3_0_6b() factory
- Mapped dependency graph between all 20 issues

Stage Summary:
- 8 P0/P1 issues identified as fixable in single pass
- 7 P2/P3 issues documented for future work
- Key insight: reference implementation uses precomputed static tables + Gather for all mask computation

---
Task ID: 2
Agent: Main Agent
Task: Fix ISSUE-001 — Replace ANE-illegal mask ops with precomputed tables + Gather

Work Log:
- Modified crates/passes/src/legality_rewrite.rs apply_rope_decode()
- Replaced Equal/Cast/LessEqual/Fill/Select with Const(eye_tab)+Gather and Const(mask_tab)+Gather
- Added shared eye_tab and mask_tab Const nodes (deduplicated across layers)
- Marked arange_tab as legacy (kept for backward compatibility)
- Added fill, fill_like, select to CPU_ONLY list in cpu_only_ops.rs
- Updated mil_lower.rs: removed Where→Select rewrite, updated Fill handling
- Updated static_table_resolver.rs: raised seq_len threshold to 8192, added from_model_config()
- Added rope_theta and has_qk_norm fields to ModelConfig in graph.rs
- Updated CLI to read rope_theta/uses_rope/has_qk_norm from ModelConfig
- Fixed all ModelConfig initializers in sir_build.rs tests
- Updated test_where_lowering to match new behavior (Where no longer rewritten to Select)

Stage Summary:
- 8 issues fixed: ISSUE-001, 002, 003, 006, 007, 011, 013, 019
- 5 issues partially fixed: ISSUE-004, 005, 008, 009, 018
- 7 issues remain open: ISSUE-010, 012, 014, 015, 016, 017, 020
- All 501+ tests pass
- Build succeeds with no errors

---
Task ID: 3
Agent: Main Agent
Task: Write ISSUES.md and update PROBLEM.md

Work Log:
- Rewrote ISSUES.md with current status of all 20 issues
- Updated PROBLEM.md to reflect current state (concat interleave fixed, new static table approach)
- Documented all changes made this session
- Listed remaining work in priority order

Stage Summary:
- ISSUES.md: 8 fixed, 5 partially fixed, 7 open
- PROBLEM.md: updated for current compilation state
- Worklog updated
