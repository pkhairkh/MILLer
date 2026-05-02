"""Tests for verify.py verification logic.

These tests verify:
1. predict_placement_from_ops() with known ops
2. ANE-predicted ops and CPU-predicted ops are classified correctly
3. PlacementResult and VerificationResult structures

All tests work WITHOUT macOS and WITHOUT coremltools by mocking
the coremltools imports.
"""

from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Test: predict_placement_from_ops
# ---------------------------------------------------------------------------

class TestPredictPlacementFromOps:
    """Test compute_plan.predict_placement_from_ops() with known ops."""

    def test_predict_with_none_ops(self):
        """predict_placement_from_ops with None should return unavailable."""
        from compute_plan import predict_placement_from_ops
        result = predict_placement_from_ops(None)
        assert result["available"] is False
        assert result["total_ops"] == 0

    def test_predict_with_empty_ops(self):
        """predict_placement_from_ops with empty list should return unavailable."""
        from compute_plan import predict_placement_from_ops
        result = predict_placement_from_ops([])
        assert result["available"] is False

    def test_predict_linear_ops_to_ane(self):
        """Linear ops should be predicted for NeuralEngine placement."""
        from compute_plan import predict_placement_from_ops
        mir_ops = [
            {"op_type": "linear", "name": "proj_1"},
            {"op_type": "linear", "name": "proj_2"},
        ]
        result = predict_placement_from_ops(mir_ops)
        assert result["available"] is True
        assert result["total_ops"] == 2
        # Both linear ops should be predicted for NeuralEngine
        ane_ops = [op for op in result["per_op_placement"]
                   if "NeuralEngine" in op.get("preferred_device", "")]
        assert len(ane_ops) == 2

    def test_predict_gather_ops_to_cpu(self):
        """Gather ops should be predicted for CPU placement."""
        from compute_plan import predict_placement_from_ops
        mir_ops = [
            {"op_type": "gather", "name": "embed"},
        ]
        result = predict_placement_from_ops(mir_ops)
        assert result["available"] is True
        cpu_ops = [op for op in result["per_op_placement"]
                   if op.get("preferred_device", "") == "CPU"]
        assert len(cpu_ops) == 1

    def test_predict_mixed_ops(self):
        """Mixed ANE and CPU ops should produce correct placement rate."""
        from compute_plan import predict_placement_from_ops
        mir_ops = [
            {"op_type": "linear", "name": "proj"},
            {"op_type": "gather", "name": "embed"},
            {"op_type": "gelu", "name": "act"},
            {"op_type": "reshape", "name": "reshape_1"},
        ]
        result = predict_placement_from_ops(mir_ops)
        assert result["available"] is True
        assert result["total_ops"] == 4
        # linear, gelu, reshape -> ANE; gather -> CPU
        assert result["ane_placement_rate"] > 0.0
        # Should be 3/4 = 0.75
        assert abs(result["ane_placement_rate"] - 0.75) < 0.01

    def test_predict_unknown_ops_default_to_cpu(self):
        """Unknown op types should default to CPU with low confidence."""
        from compute_plan import predict_placement_from_ops
        mir_ops = [
            {"op_type": "custom_mysterious_op", "name": "custom_1"},
        ]
        result = predict_placement_from_ops(mir_ops)
        assert result["available"] is True
        assert result["per_op_placement"][0]["preferred_device"] == "CPU"
        # Low confidence for unknown ops (uses prediction_confidence per entry)
        assert result["per_op_placement"][0]["prediction_confidence"] < 0.5

    def test_predict_mil_prefix_stripped(self):
        """Op types with 'MIL' prefix should have it stripped for lookup."""
        from compute_plan import predict_placement_from_ops
        mir_ops = [
            {"op_type": "MILlinear", "name": "proj"},
        ]
        result = predict_placement_from_ops(mir_ops)
        assert result["available"] is True
        # Should match "linear" after stripping "MIL"
        assert any("NeuralEngine" in op.get("preferred_device", "")
                    for op in result["per_op_placement"])

    def test_prediction_has_confidence(self):
        """Prediction result should include prediction_confidence."""
        from compute_plan import predict_placement_from_ops
        mir_ops = [{"op_type": "linear", "name": "proj"}]
        result = predict_placement_from_ops(mir_ops)
        assert "prediction_confidence" in result
        assert 0.0 <= result["prediction_confidence"] <= 1.0

    def test_per_op_placement_has_expected_keys(self):
        """Each per-op placement entry should have expected keys."""
        from compute_plan import predict_placement_from_ops
        mir_ops = [{"op_type": "linear", "name": "proj"}]
        result = predict_placement_from_ops(mir_ops)
        for op_entry in result["per_op_placement"]:
            assert "op_type" in op_entry
            assert "op_name" in op_entry
            assert "preferred_device" in op_entry
            assert "prediction_confidence" in op_entry


# ---------------------------------------------------------------------------
# Test: Specific ANE-predicted ops
# ---------------------------------------------------------------------------

class TestAnePredictedOps:
    """Test that specific known ANE-friendly ops are predicted for NeuralEngine."""

    @pytest.mark.parametrize("op_type", [
        "linear", "matmul", "gelu", "softmax", "layer_norm",
        "reshape", "transpose", "relu", "concat", "add", "mul",
        "scaled_dot_product_attention",
    ])
    def test_ane_friendly_ops(self, op_type):
        """Known ANE-friendly ops should be predicted for NeuralEngine."""
        from compute_plan import predict_placement_from_ops
        mir_ops = [{"op_type": op_type, "name": f"{op_type}_0"}]
        result = predict_placement_from_ops(mir_ops)
        assert result["available"] is True
        assert "NeuralEngine" in result["per_op_placement"][0]["preferred_device"]


# ---------------------------------------------------------------------------
# Test: Specific CPU-predicted ops
# ---------------------------------------------------------------------------

class TestCpuPredictedOps:
    """Test that specific known CPU-bound ops are predicted for CPU."""

    @pytest.mark.parametrize("op_type", [
        "embedding", "gather", "topk", "slice_by_index",
        "read_state", "write_state",
    ])
    def test_cpu_bound_ops(self, op_type):
        """Known CPU-bound ops should be predicted for CPU."""
        from compute_plan import predict_placement_from_ops
        mir_ops = [{"op_type": op_type, "name": f"{op_type}_0"}]
        result = predict_placement_from_ops(mir_ops)
        assert result["available"] is True
        assert result["per_op_placement"][0]["preferred_device"] == "CPU"


# ---------------------------------------------------------------------------
# Test: PlacementResult structure
# ---------------------------------------------------------------------------

class TestPlacementResultStructure:
    """Test PlacementResult dataclass structure."""

    def test_placement_result_attributes(self):
        """PlacementResult should have all expected attributes."""
        from verify import PlacementResult
        result = PlacementResult()
        assert hasattr(result, "ane_placement_rate")
        assert hasattr(result, "total_ops")
        assert hasattr(result, "ane_placed_ops")
        assert hasattr(result, "per_op_placement")
        assert hasattr(result, "available")
        assert hasattr(result, "reason")
        assert hasattr(result, "verification_method")
        assert hasattr(result, "prediction_confidence")

    def test_placement_result_defaults(self):
        """PlacementResult should have sensible defaults."""
        from verify import PlacementResult
        result = PlacementResult()
        assert result.ane_placement_rate == 0.0
        assert result.total_ops == 0
        assert result.ane_placed_ops == 0
        assert result.per_op_placement == []
        assert result.available is False
        assert result.verification_method == "unavailable"

    def test_placement_result_to_dict(self):
        """PlacementResult.to_dict() should return a dict with all fields."""
        from verify import PlacementResult
        result = PlacementResult()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ane_placement_rate" in d
        assert "total_ops" in d
        assert "per_op_placement" in d
        assert "available" in d
        assert "verification_method" in d
        assert "prediction_confidence" in d


# ---------------------------------------------------------------------------
# Test: VerificationResult structure
# ---------------------------------------------------------------------------

class TestVerificationResultStructure:
    """Test VerificationResult dataclass structure."""

    def test_verification_result_attributes(self):
        """VerificationResult should have all four dimension results."""
        from verify import VerificationResult
        result = VerificationResult()
        assert hasattr(result, "op_fidelity")
        assert hasattr(result, "placement")
        assert hasattr(result, "state_conformance")
        assert hasattr(result, "multifunction_conformance")
        assert hasattr(result, "overall_score")

    def test_verification_result_sub_results(self):
        """VerificationResult sub-results should be proper types."""
        from verify import (
            VerificationResult, OpFidelityResult,
            PlacementResult, StateConformanceResult,
            MultifunctionResult,
        )
        result = VerificationResult()
        assert isinstance(result.op_fidelity, OpFidelityResult)
        assert isinstance(result.placement, PlacementResult)
        assert isinstance(result.state_conformance, StateConformanceResult)
        assert isinstance(result.multifunction_conformance, MultifunctionResult)

    def test_verification_result_to_dict(self):
        """VerificationResult.to_dict() should include all sub-results."""
        from verify import VerificationResult
        result = VerificationResult()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "op_fidelity" in d
        assert "placement" in d
        assert "state_conformance" in d
        assert "multifunction_conformance" in d
        assert "overall_score" in d

    def test_op_fidelity_result_structure(self):
        """OpFidelityResult should have all expected attributes."""
        from verify import OpFidelityResult
        result = OpFidelityResult()
        assert hasattr(result, "op_fidelity_score")
        assert hasattr(result, "mir_op_count")
        assert hasattr(result, "structure_op_count")
        assert hasattr(result, "matched_ops")
        assert hasattr(result, "available")
        d = result.to_dict()
        assert "op_fidelity_score" in d

    def test_state_conformance_result_structure(self):
        """StateConformanceResult should have all expected attributes."""
        from verify import StateConformanceResult
        result = StateConformanceResult()
        assert hasattr(result, "stateful_model")
        assert hasattr(result, "has_read_state")
        assert hasattr(result, "has_update_state")
        assert hasattr(result, "conformance_score")

    def test_multifunction_result_structure(self):
        """MultifunctionResult should have all expected attributes."""
        from verify import MultifunctionResult
        result = MultifunctionResult()
        assert hasattr(result, "is_multifunction")
        assert hasattr(result, "actual_function_count")
        assert hasattr(result, "function_names_match")


# ---------------------------------------------------------------------------
# Test: save_verification_result
# ---------------------------------------------------------------------------

class TestSaveVerificationResult:
    """Test save_verification_result artifact persistence."""

    def test_save_creates_artifact_files(self, tmp_path):
        """save_verification_result should create JSON artifact files."""
        from verify import VerificationResult, save_verification_result

        result = VerificationResult()
        result.mlpackage_path = "/test/path"

        artifact_paths = save_verification_result(
            result, str(tmp_path), artifact_name="test_verification"
        )

        assert "artifact_path" in artifact_paths
        assert "summary_path" in artifact_paths

        import os
        assert os.path.exists(artifact_paths["artifact_path"])
        assert os.path.exists(artifact_paths["summary_path"])

    def test_saved_artifact_is_valid_json(self, tmp_path):
        """Saved artifact should be valid JSON."""
        import json
        from verify import VerificationResult, save_verification_result

        result = VerificationResult()
        artifact_paths = save_verification_result(
            result, str(tmp_path), artifact_name="test_verification"
        )

        with open(artifact_paths["artifact_path"]) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "overall_score" in data

    def test_saved_summary_is_valid_json(self, tmp_path):
        """Saved summary should be valid JSON with key metrics."""
        import json
        from verify import VerificationResult, save_verification_result

        result = VerificationResult()
        artifact_paths = save_verification_result(
            result, str(tmp_path), artifact_name="test_verification"
        )

        with open(artifact_paths["summary_path"]) as f:
            summary = json.load(f)
        assert isinstance(summary, dict)
        assert "overall_score" in summary
