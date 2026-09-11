"""Independent reference comparisons for registered sparse graph equations."""

import numpy as np

from pure_gnn_v31.technical_checks import reference_equivalence_report


TOLERANCE = 1e-5


def test_local_sparse_matches_explicit_reference_output_and_input_gradient():
    report = reference_equivalence_report()
    np.testing.assert_allclose(report["local_output_max_abs_error"], 0.0, atol=TOLERANCE, rtol=0.0)
    np.testing.assert_allclose(report["local_gradient_max_abs_error"], 0.0, atol=TOLERANCE, rtol=0.0)


def test_coarse_g05_matches_explicit_uniform_reference():
    report = reference_equivalence_report()
    np.testing.assert_allclose(report["coarse_g05_output_max_abs_error"], 0.0, atol=TOLERANCE, rtol=0.0)


def test_coarse_g1_matches_explicit_receiver_normalized_reference():
    report = reference_equivalence_report()
    np.testing.assert_allclose(report["coarse_g1_output_max_abs_error"], 0.0, atol=TOLERANCE, rtol=0.0)
