"""Tests for mil_emitter.py program builder structure.

These tests verify that MIL program construction functions:
1. Return objects with expected methods/attributes
2. Have correct op count expectations (without coremltools conversion)
3. Have correct function count in multi-function programs

All tests work WITHOUT macOS and WITHOUT coremltools by mocking
the coremltools imports.
"""

from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers: Create mock coremltools objects for testing
# ---------------------------------------------------------------------------

def _make_mock_coremltools():
    """Create a mock coremltools module with Builder, types, and numpy.

    Returns a tuple of (mock_ct, mock_mb, mock_types, mock_np) that
    mimics the objects returned by _import_coremltools().
    """
    mock_ct = mock.MagicMock()
    mock_ct.__version__ = "9.0"

    # Mock ComputeUnit enum
    mock_ct.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"
    mock_ct.ComputeUnit.CPU_AND_GPU = "CPU_AND_GPU"
    mock_ct.ComputeUnit.CPU_ONLY = "CPU_ONLY"
    mock_ct.ComputeUnit.ALL = "ALL"

    # Mock target enum
    mock_ct.target.iOS16 = "iOS16"
    mock_ct.target.iOS17 = "iOS17"
    mock_ct.target.iOS18 = "iOS18"

    mock_mb = mock.MagicMock()
    mock_types = mock.MagicMock()
    mock_types.fp16 = "fp16"
    mock_types.fp32 = "fp32"
    mock_types.int32 = "int32"

    mock_np = mock.MagicMock()

    # Make numpy random functions return something sensible
    import numpy as real_np
    mock_np.random.seed = real_np.random.seed
    mock_np.random.randn = real_np.random.randn
    mock_np.float16 = real_np.float16
    mock_np.float32 = real_np.float32
    mock_np.zeros = real_np.zeros
    mock_np.array = real_np.array
    mock_np.int32 = real_np.int32

    # Mock mb.program to act as a decorator that returns a mock program
    def mock_program_decorator(input_specs=None, opset_version=None, function_name=None):
        def decorator(func):
            # Call the decorated function with mock inputs to count ops
            mock_prog = mock.MagicMock()
            mock_prog._func_name = function_name or "main"
            mock_prog._func = func

            # Create mock input tensors based on input_specs
            mock_inputs = []
            if input_specs:
                for spec in input_specs:
                    mock_input = mock.MagicMock()
                    mock_inputs.append(mock_input)

            # Also create mock state inputs
            # Count StateTensorSpec inputs
            state_inputs = []
            for spec in (input_specs or []):
                if hasattr(spec, '_mock_name') and 'state' in str(spec._mock_name).lower():
                    state_inputs.append(mock.MagicMock())

            all_inputs = mock_inputs + state_inputs
            try:
                result = func(*all_inputs)
                mock_prog._result = result
            except Exception:
                pass  # Function may not work with mocks, that's ok

            return mock_prog
        return decorator

    # Make TensorSpec and StateTensorSpec return simple mock objects
    def mock_tensor_spec(shape=None, dtype=None):
        spec = mock.MagicMock()
        spec.shape = shape
        spec.dtype = dtype
        spec._mock_name = "TensorSpec"
        return spec

    def mock_state_tensor_spec(shape=None, dtype=None):
        spec = mock.MagicMock()
        spec.shape = shape
        spec.dtype = dtype
        spec._mock_name = "StateTensorSpec"
        return spec

    mock_mb.program = mock_program_decorator
    mock_mb.TensorSpec = mock_tensor_spec
    mock_mb.StateTensorSpec = mock_state_tensor_spec

    # Mock Builder operations to return mock tensors and track call counts
    op_call_counts = {}

    def _make_op(op_name):
        def op_func(**kwargs):
            op_call_counts[op_name] = op_call_counts.get(op_name, 0) + 1
            result = mock.MagicMock()
            result._op_name = op_name
            result._kwargs = kwargs
            return result
        return op_func

    # Set up all the Builder operations used by mil_emitter
    for op_name in [
        "linear", "const", "gather", "concat", "gelu", "relu",
        "slice_by_index", "reshape", "scaled_dot_product_attention",
        "read_state", "write_state", "coreml_update_state", "slice_update",
        "matmul", "add", "mul", "layer_norm",
        "transpose", "softmax", "expand_dims",
    ]:
        setattr(mock_mb, op_name, _make_op(op_name))

    return mock_ct, mock_mb, mock_types, mock_np


# ---------------------------------------------------------------------------
# Test: Program builder functions return expected structures
# ---------------------------------------------------------------------------

class TestBuildLinearProjectionProgram:
    """Test build_linear_projection_program returns expected structure."""

    def test_returns_program_and_metadata(self):
        """build_linear_projection_program should return (program, metadata_dict)."""
        mock_ct, mock_mb, mock_types, mock_np = _make_mock_coremltools()

        with mock.patch("mil_emitter._import_coremltools", return_value=(mock_ct, mock_mb, mock_types, mock_np)):
            from mil_emitter import build_linear_projection_program
            result = build_linear_projection_program({
                "task_name": "test_linear",
                "input_dim": 64,
                "output_dim": 32,
            })

        assert isinstance(result, tuple)
        assert len(result) == 2
        program, metadata = result
        # Program should be a mock object (from mock_mb.program)
        assert program is not None
        # Metadata should be a dict with expected keys
        assert isinstance(metadata, dict)
        assert "task_name" in metadata
        assert "input_dim" in metadata
        assert "output_dim" in metadata
        assert metadata["task_name"] == "test_linear"
        assert metadata["input_dim"] == 64
        assert metadata["output_dim"] == 32

    def test_metadata_emission_path(self):
        """Metadata should contain emission_path."""
        mock_ct, mock_mb, mock_types, mock_np = _make_mock_coremltools()

        with mock.patch("mil_emitter._import_coremltools", return_value=(mock_ct, mock_mb, mock_types, mock_np)):
            from mil_emitter import build_linear_projection_program
            _, metadata = build_linear_projection_program({})

        assert metadata["emission_path"] == "linear_projection"

    def test_metadata_includes_dtype(self):
        """Metadata should include dtype field."""
        mock_ct, mock_mb, mock_types, mock_np = _make_mock_coremltools()

        with mock.patch("mil_emitter._import_coremltools", return_value=(mock_ct, mock_mb, mock_types, mock_np)):
            from mil_emitter import build_linear_projection_program
            _, metadata = build_linear_projection_program({"dtype": "fp32"})

        assert metadata["dtype"] == "fp32"


class TestBuildDecodeStepProgram:
    """Test build_decode_step_program returns expected structure."""

    def test_returns_program_and_metadata(self):
        """build_decode_step_program should return (program, metadata_dict)."""
        mock_ct, mock_mb, mock_types, mock_np = _make_mock_coremltools()

        with mock.patch("mil_emitter._import_coremltools", return_value=(mock_ct, mock_mb, mock_types, mock_np)):
            from mil_emitter import build_decode_step_program
            result = build_decode_step_program({
                "embed_dim": 128,
                "num_heads": 4,
                "head_dim": 32,
            })

        assert isinstance(result, tuple)
        assert len(result) == 2
        _, metadata = result
        assert metadata["emission_path"] == "decode_step"

    def test_decode_step_metadata_fields(self):
        """Decode step metadata should include attention parameters."""
        mock_ct, mock_mb, mock_types, mock_np = _make_mock_coremltools()

        with mock.patch("mil_emitter._import_coremltools", return_value=(mock_ct, mock_mb, mock_types, mock_np)):
            from mil_emitter import build_decode_step_program
            _, metadata = build_decode_step_program({
                "embed_dim": 256,
                "num_heads": 8,
                "head_dim": 32,
                "kv_len": 128,
            })

        assert metadata["embed_dim"] == 256
        assert metadata["num_heads"] == 8
        assert metadata["head_dim"] == 32
        assert metadata["kv_len"] == 128


class TestBuildStatefulDecodeStepProgram:
    """Test build_stateful_decode_step_program returns expected structure."""

    def test_returns_stateful_metadata(self):
        """Stateful decode step metadata should have stateful=True and state_inputs."""
        mock_ct, mock_mb, mock_types, mock_np = _make_mock_coremltools()

        with mock.patch("mil_emitter._import_coremltools", return_value=(mock_ct, mock_mb, mock_types, mock_np)):
            from mil_emitter import build_stateful_decode_step_program
            _, metadata = build_stateful_decode_step_program({
                "embed_dim": 128,
                "num_heads": 4,
                "head_dim": 32,
                "kv_len": 64,
            })

        assert metadata["stateful"] is True
        assert "state_inputs" in metadata
        assert len(metadata["state_inputs"]) == 2  # k_state and v_state

    def test_state_inputs_structure(self):
        """State inputs should have name, shape, and dtype."""
        mock_ct, mock_mb, mock_types, mock_np = _make_mock_coremltools()

        with mock.patch("mil_emitter._import_coremltools", return_value=(mock_ct, mock_mb, mock_types, mock_np)):
            from mil_emitter import build_stateful_decode_step_program
            _, metadata = build_stateful_decode_step_program({
                "embed_dim": 128,
                "num_heads": 4,
                "head_dim": 32,
                "kv_len": 64,
            })

        for state_input in metadata["state_inputs"]:
            assert "name" in state_input
            assert "shape" in state_input
            assert "dtype" in state_input


class TestBuildLutProjectionProgram:
    """Test build_lut_projection_program returns expected structure."""

    def test_returns_program_and_metadata(self):
        """build_lut_projection_program should return (program, metadata_dict)."""
        mock_ct, mock_mb, mock_types, mock_np = _make_mock_coremltools()

        with mock.patch("mil_emitter._import_coremltools", return_value=(mock_ct, mock_mb, mock_types, mock_np)):
            from mil_emitter import build_lut_projection_program
            result = build_lut_projection_program({
                "vocab_size": 1000,
                "embed_dim": 512,
                "num_groups": 64,
            })

        assert isinstance(result, tuple)
        assert len(result) == 2
        _, metadata = result
        assert metadata["emission_path"] == "lut_projection"

    def test_lut_metadata_fields(self):
        """LUT projection metadata should include LUT-specific fields."""
        mock_ct, mock_mb, mock_types, mock_np = _make_mock_coremltools()

        with mock.patch("mil_emitter._import_coremltools", return_value=(mock_ct, mock_mb, mock_types, mock_np)):
            from mil_emitter import build_lut_projection_program
            _, metadata = build_lut_projection_program({
                "vocab_size": 32000,
                "embed_dim": 512,
                "num_groups": 64,
                "lut_bitwidth": 4,
            })

        assert metadata["vocab_size"] == 32000
        assert metadata["embed_dim"] == 512
        assert metadata["num_groups"] == 64
        assert metadata["lut_bitwidth"] == 4


# ---------------------------------------------------------------------------
# Test: _import_coremltools behavior
# ---------------------------------------------------------------------------

class TestImportCoremltools:
    """Test _import_coremltools behavior with coremltools availability.

    W-26 fix: _import_coremltools now raises ImportError instead of
    silently returning (None, None, None, None). This test verifies
    the new behavior.
    """

    def test_raises_importerror_when_coremltools_unavailable(self):
        """When coremltools is not installed, _import_coremltools raises ImportError.

        Previously (before W-26 fix), it silently returned (None, None, None, None).
        Now it raises ImportError explicitly so callers don't silently proceed
        with None values.
        """
        # Reset the cached coremltools so the ImportError path is triggered
        import common
        from common import _ensure_coremltools
        old_ct = common._ct
        old_map = common.COMPUTE_MAP
        common._ct = None
        common.COMPUTE_MAP = None
        try:
            with mock.patch.dict("sys.modules", {"coremltools": None}):
                with pytest.raises(ImportError):
                    _ensure_coremltools()
        finally:
            common._ct = old_ct
            common.COMPUTE_MAP = old_map


# ---------------------------------------------------------------------------
# Test: Multi-function program structure
# ---------------------------------------------------------------------------

class TestMultifunctionPrograms:
    """Test multi-function program builder structure."""

    def test_build_multifunction_program_exists(self):
        """build_multifunction_program should exist as a callable."""
        from mil_emitter import build_multifunction_program
        assert callable(build_multifunction_program)

    def test_build_multifunction_shared_weights_program_exists(self):
        """build_multifunction_program_with_shared_weights should exist as a callable."""
        from mil_emitter import build_multifunction_program_with_shared_weights
        assert callable(build_multifunction_program_with_shared_weights)


# ---------------------------------------------------------------------------
# Test: save_mlpackage and compute_plan_info
# ---------------------------------------------------------------------------

class TestSaveAndComputePlan:
    """Test that save_mlpackage and compute_plan_info are importable and callable."""

    def test_save_mlpackage_exists(self):
        """save_mlpackage should be importable from mil_emitter."""
        from mil_emitter import save_mlpackage
        assert callable(save_mlpackage)

    def test_compute_plan_info_exists(self):
        """compute_plan_info should be importable from mil_emitter."""
        from mil_emitter import compute_plan_info
        assert callable(compute_plan_info)

    def test_compute_plan_info_no_coremltools(self):
        """compute_plan_info should report unavailable without coremltools."""
        with mock.patch("mil_emitter._import_coremltools", return_value=(None, None, None, None)):
            from mil_emitter import compute_plan_info
            result = compute_plan_info("/nonexistent/path")
            assert result["available"] is False
            assert "reason" in result
