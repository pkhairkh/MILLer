"""Tests for T-D-02 (M-032): MIR specification compliance verification."""
from verify import MirSpecViolation, pre_emit_verification, verify_mir_spec_compliance


def test_no_issues_for_valid_ops():
    ops = [
        {'output_name': 'x', 'type': 'input', 'inputs': []},
        {'output_name': 'y', 'type': 'relu', 'inputs': ['x']},
    ]
    assert pre_emit_verification(ops) == []

def test_duplicate_output_name():
    ops = [
        {'output_name': 'x', 'type': 'input', 'inputs': []},
        {'output_name': 'x', 'type': 'relu', 'inputs': ['x']},  # duplicate
    ]
    issues = pre_emit_verification(ops)
    assert any('Duplicate' in i for i in issues)

def test_dangling_input():
    ops = [
        {'output_name': 'y', 'type': 'relu', 'inputs': ['nonexistent']},
    ]
    issues = pre_emit_verification(ops)
    assert any('Dangling' in i for i in issues)

def test_const_and_weight_inputs_not_flagged():
    """Inputs starting with 'const_' or 'weight_' should not be flagged as dangling."""
    ops = [
        {'output_name': 'y', 'type': 'linear', 'inputs': ['const_w', 'weight_b']},
    ]
    issues = pre_emit_verification(ops)
    assert issues == []

def test_graph_inputs_from_mir_spec():
    """Inputs listed in mir_spec['graph_inputs'] should not be flagged as dangling."""
    ops = [
        {'output_name': 'y', 'type': 'relu', 'inputs': ['x']},
    ]
    mir_spec = {'graph_inputs': ['x']}
    issues = pre_emit_verification(ops, mir_spec)
    assert issues == []

def test_mir_spec_compliance_no_program():
    """verify_mir_spec_compliance should return empty list when program is None."""
    mir_spec = {
        'inputs': [{'name': 'x', 'shape': [1, 64], 'dtype': 'fp16'}],
        'outputs': [{'name': 'output', 'shape': [1, 32], 'dtype': 'fp16'}],
        'ops': [{'name': 'linear', 'type': 'linear', 'inputs': ['x'], 'outputs': ['output']}],
    }
    violations = verify_mir_spec_compliance(None, mir_spec)
    assert violations == []

def test_mir_spec_compliance_none_spec():
    """verify_mir_spec_compliance should return empty list when mir_spec is None."""
    violations = verify_mir_spec_compliance(None, None)
    assert violations == []

def test_mir_spec_violation_to_dict():
    """MirSpecViolation.to_dict() should return expected keys."""
    v = MirSpecViolation(check="input_count", message="Mismatch", severity="error")
    d = v.to_dict()
    assert d['check'] == 'input_count'
    assert d['message'] == 'Mismatch'
    assert d['severity'] == 'error'

def test_mir_spec_violation_repr():
    """MirSpecViolation __repr__ should include check and message."""
    v = MirSpecViolation(check="input_count", message="Mismatch", severity="error")
    r = repr(v)
    assert 'input_count' in r
    assert 'Mismatch' in r

if __name__ == '__main__':
    test_no_issues_for_valid_ops()
    test_duplicate_output_name()
    test_dangling_input()
    test_const_and_weight_inputs_not_flagged()
    test_graph_inputs_from_mir_spec()
    test_mir_spec_compliance_no_program()
    test_mir_spec_compliance_none_spec()
    test_mir_spec_violation_to_dict()
    test_mir_spec_violation_repr()
    print("All M-032 verification tests passed!")
