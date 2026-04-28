"""Conftest — shared fixtures for all Python tests.

Since coremltools is not available on Linux, we need to mock it
before importing any MILLer Python modules that depend on it.
"""

import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True, scope="session")
def mock_coremltools():
    """Mock coremltools and numpy so MILLer modules can be imported on Linux.

    This runs once per test session and inserts mock modules into sys.modules
    before any MILLer module is imported.
    """
    # Create mock coremltools module with minimal attributes
    mock_ct = mock.MagicMock()
    mock_ct.__version__ = "9.0"
    mock_ct.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"
    mock_ct.ComputeUnit.CPU_AND_GPU = "CPU_AND_GPU"
    mock_ct.ComputeUnit.CPU_ONLY = "CPU_ONLY"
    mock_ct.ComputeUnit.ALL = "ALL"
    mock_ct.target.iOS16 = "iOS16"
    mock_ct.target.iOS17 = "iOS17"
    mock_ct.target.iOS18 = "iOS18"

    # Create mock coremltools.models.compute_plan submodule
    mock_ct.models.compute_plan.MLComputePlan = mock.MagicMock()

    # Create mock coremltools.converters.mil submodule
    mock_mb = mock.MagicMock()
    mock_types = mock.MagicMock()
    mock_ct.converters.mil.Builder = mock_mb
    mock_ct.converters.mil.mil.types = mock_types

    # Only mock if not already available (don't override on macOS)
    modules_to_mock = {}
    if "coremltools" not in sys.modules:
        modules_to_mock["coremltools"] = mock_ct
        modules_to_mock["coremltools.models"] = mock_ct.models
        modules_to_mock["coremltools.models.compute_plan"] = mock_ct.models.compute_plan
        modules_to_mock["coremltools.converters"] = mock_ct.converters
        modules_to_mock["coremltools.converters.mil"] = mock_ct.converters.mil
        modules_to_mock["coremltools.converters.mil.Builder"] = mock_mb
        modules_to_mock["coremltools.converters.mil.mil"] = mock_ct.converters.mil.mil
        modules_to_mock["coremltools.converters.mil.mil.types"] = mock_types

    sys.modules.update(modules_to_mock)

    yield

    # Cleanup: remove our mock modules
    for key in modules_to_mock:
        sys.modules.pop(key, None)
