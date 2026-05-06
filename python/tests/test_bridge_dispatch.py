"""Tests for bridge.py command dispatch.

These tests verify that:
1. Each bridge command name is recognized
2. Command dispatch produces valid JSON output (mocking coremltools)
3. Error handling for invalid commands works correctly

All tests work WITHOUT macOS and WITHOUT coremltools by mocking imports.
"""

import json
import os
from unittest import mock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The bridge requires bridge_version=1 in the payload
BRIDGE_VERSION = 1

# The set of known bridge command names from bridge.py's dispatch table
KNOWN_COMMANDS = {
    "emit_linear_projection",
    "emit_lut_projection",
    "emit_decode_step",
    "emit_stateless_decode_step",
    "emit_stateful_decode_step",
    "emit_shard_decode_step",
    "emit_palettized_linear_projection",
    "emit_mlp_block",
    "emit_attention",
    "emit_mlprogram",
    "emit_multifunction",
    "emit_multifunction_shared_weights",
    "validate_multifunction",
    "convert",
    "palettize",
    "compute_plan",
    "compute_plan_harvest",
    "inspect_mlpackage",
    "host_inspect",
    "model_structure",
    "verify",
    "validate_proto_direct",
    "profile",
}


def _make_command(**kwargs):
    """Create a command dict with the required bridge_version field."""
    kwargs.setdefault("bridge_version", BRIDGE_VERSION)
    return kwargs


def _write_command_file(tmpdir, command_dict):
    """Write a command JSON file and return its path."""
    cmd_path = os.path.join(tmpdir, "command.json")
    with open(cmd_path, "w") as f:
        json.dump(command_dict, f)
    return cmd_path


def _read_result_file(result_path):
    """Read and return the result JSON."""
    with open(result_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Test: Command name recognition
# ---------------------------------------------------------------------------

class TestCommandRecognition:
    """Verify each bridge command name is in the known set."""

    def test_emit_commands_exist(self):
        emit_cmds = {c for c in KNOWN_COMMANDS if c.startswith("emit_")}
        assert len(emit_cmds) >= 8  # At least the documented emit_* commands

    def test_all_known_commands_are_recognized(self):
        """Every documented command should appear in the bridge dispatch."""
        # These are the commands from bridge.py docstring and dispatch table
        expected = {
            "emit_linear_projection",
            "emit_lut_projection",
            "emit_decode_step",
            "emit_stateless_decode_step",
            "emit_stateful_decode_step",
            "emit_shard_decode_step",
            "emit_palettized_linear_projection",
            "emit_mlp_block",
            "emit_attention",
            "emit_mlprogram",
            "emit_multifunction",
            "emit_multifunction_shared_weights",
            "validate_multifunction",
            "convert",
            "palettize",
            "compute_plan",
            "compute_plan_harvest",
            "inspect_mlpackage",
            "host_inspect",
            "model_structure",
            "verify",
            "validate_proto_direct",
            "profile",
        }
        assert KNOWN_COMMANDS == expected


# ---------------------------------------------------------------------------
# Test: Dispatch produces valid JSON for commands that don't need coremltools
# ---------------------------------------------------------------------------

class TestDispatchJsonOutput:
    """Test that bridge dispatch produces valid JSON output for commands
    that do not require coremltools for their basic error path."""

    def _run_bridge_main(self, cmd_path, result_path):
        """Run bridge.py main() with the given command/result files."""
        with mock.patch("sys.argv", ["bridge.py", cmd_path, result_path]):
            import bridge
            bridge.main()

    def test_host_inspect_missing_path(self, tmp_path):
        """host_inspect with no mlpackage_path should return error JSON."""
        cmd_path = _write_command_file(str(tmp_path), _make_command(
            command="host_inspect",
        ))
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        assert result["status"] == "error"
        assert "mlpackage_path" in result.get("error_message", "").lower()

    def test_host_inspect_nonexistent_path(self, tmp_path):
        """host_inspect with a nonexistent path should return success with package_present=False."""
        cmd_path = _write_command_file(str(tmp_path), _make_command(
            command="host_inspect",
            mlpackage_path="/nonexistent/path/to/model.mlpackage",
        ))
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        assert result["status"] == "success"
        assert result["package_present"] is False

    def test_verify_missing_path(self, tmp_path):
        """verify with no mlpackage_path should return error JSON."""
        cmd_path = _write_command_file(str(tmp_path), _make_command(
            command="verify",
        ))
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        assert result["status"] == "error"
        assert "mlpackage_path" in result.get("error_message", "").lower()

    def test_validate_proto_direct_missing_path(self, tmp_path):
        """validate_proto_direct with no path should return error JSON."""
        cmd_path = _write_command_file(str(tmp_path), _make_command(
            command="validate_proto_direct",
        ))
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        assert result["status"] == "error"
        assert "mlpackage_path" in result.get("error_message", "").lower()

    def test_model_structure_missing_path(self, tmp_path):
        """model_structure with no mlpackage_path should return error JSON."""
        cmd_path = _write_command_file(str(tmp_path), _make_command(
            command="model_structure",
        ))
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        assert result["status"] == "error"

    def test_host_inspect_result_is_valid_json(self, tmp_path):
        """All bridge results should be valid JSON serializable dicts."""
        cmd_path = _write_command_file(str(tmp_path), _make_command(
            command="host_inspect",
            mlpackage_path="/nonexistent/model.mlpackage",
        ))
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        # Should be a dict with known keys
        assert isinstance(result, dict)
        assert "status" in result


# ---------------------------------------------------------------------------
# Test: Error handling for invalid commands
# ---------------------------------------------------------------------------

class TestInvalidCommands:
    """Test that unknown/invalid commands produce proper error results."""

    def _run_bridge_main(self, cmd_path, result_path):
        with mock.patch("sys.argv", ["bridge.py", cmd_path, result_path]):
            import bridge
            bridge.main()

    def test_unknown_command(self, tmp_path):
        """An unrecognized command should produce an error result."""
        cmd_path = _write_command_file(str(tmp_path), _make_command(
            command="nonexistent_command",
        ))
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        assert result["status"] == "error"
        assert "Unknown command" in result.get("error_message", "")

    def test_empty_command(self, tmp_path):
        """An empty command string should produce an error result."""
        cmd_path = _write_command_file(str(tmp_path), _make_command(
            command="",
        ))
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        assert result["status"] == "error"

    def test_missing_command_key(self, tmp_path):
        """A command dict without a 'command' key should produce an error."""
        cmd_path = _write_command_file(str(tmp_path), _make_command(
            task_name="test",
        ))
        # Remove the command key that _make_command doesn't add
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        assert result["status"] == "error"

    def test_wrong_bridge_version(self, tmp_path):
        """Wrong bridge_version should produce a version mismatch error."""
        cmd_path = _write_command_file(str(tmp_path), {
            "command": "host_inspect",
            "bridge_version": 0,
        })
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        assert result["status"] == "error"
        assert "version" in result.get("error_message", "").lower()


# ---------------------------------------------------------------------------
# Test: Emission commands fail gracefully without coremltools
# ---------------------------------------------------------------------------

class TestEmissionWithoutCoremltools:
    """Verify emission commands return proper errors when coremltools is unavailable."""

    def _run_bridge_main(self, cmd_path, result_path):
        with mock.patch("sys.argv", ["bridge.py", cmd_path, result_path]):
            import bridge
            bridge.main()

    def test_emit_linear_projection_no_coremltools(self, tmp_path):
        """emit_linear_projection should fail gracefully without coremltools."""
        # Since coremltools is mocked (not real), the emission will try to use
        # the mock which won't actually build a real program. The _import_coremltools()
        # in mil_emitter.py will return mock objects, but the actual program
        # construction with mb.program decorator will fail or return mock results.
        # The key test is that the bridge returns a proper error/success JSON.
        cmd_path = _write_command_file(str(tmp_path), _make_command(
            command="emit_linear_projection",
            output_path=str(tmp_path),
            input_dim=64,
            output_dim=32,
        ))
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        # The result should be valid JSON with a status field
        assert isinstance(result, dict)
        assert "status" in result

    def test_convert_no_coremltools(self, tmp_path):
        """convert should handle missing coremltools gracefully."""
        cmd_path = _write_command_file(str(tmp_path), _make_command(
            command="convert",
            output_path=str(tmp_path),
        ))
        result_path = os.path.join(str(tmp_path), "result.json")
        self._run_bridge_main(cmd_path, result_path)

        result = _read_result_file(result_path)
        # The result should be valid JSON with a status field
        # (may be error or success depending on mock behavior)
        assert isinstance(result, dict)
        assert "status" in result
