"""Contracts for deterministic point-to-interval estimators."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'assets/tools/proteolysis_data_manager/src'))
from proteolysis_data_manager.analysis.half_life.intervals import (
    FAMILIES, fit_interval_model, validate_interval_estimator,
)


@pytest.fixture(params=FAMILIES)
def exact_intervals(request):
    """Build intervals with analytically known positive width laws."""
    family = request.param
    points = np.geomspace(20.0, 2000.0, 8)  # (N,)
    reference = np.exp(np.log(points).mean())
    normalized = points / reference  # (N,)
    coordinate = np.log(normalized) if family == 'logarithmic' else normalized  # (N,)
    if family == 'exponential':
        lower_distance = np.exp(np.log(.3) + .1 * coordinate)  # (N,)
        upper_distance = np.exp(np.log(.5) + .15 * coordinate)  # (N,)
    else:
        lower_distance = .3 + .1 * coordinate  # (N,)
        upper_distance = .5 + .15 * coordinate  # (N,)
    return pd.DataFrame({
        'sequence_id': [f'peptide_{i}' for i in range(len(points))],
        'experiment_id': [f'experiment_{i}' for i in range(len(points))],
        'half_life_seconds': points,
        'ci_lower_seconds': points * np.exp(-lower_distance),
        'ci_upper_seconds': points * np.exp(upper_distance),
        'source': 'synthetic', 'species': 'human', 'matrix': 'serum', 'matrix_fraction': .5,
        'in_vivo_in_vitro': 'in vitro', 'temperature_c': 37., 'ph': None, 'assay': 'HPLC',
        'replicate_count': 3, 'peptide_concentration_um': 500., 'ci_level': .95,
    }), family


def test_recovers_each_analytic_family(exact_intervals):
    frame, family = exact_intervals
    model = fit_interval_model(frame, family)
    lower, upper = model.predict(frame.half_life_seconds.to_numpy())
    np.testing.assert_allclose(lower, frame.ci_lower_seconds, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(upper, frame.ci_upper_seconds, rtol=1e-10, atol=1e-10)
    points = frame.half_life_seconds.to_numpy()
    assert np.all((0 < lower) & (lower < points) & (points < upper))


def test_deterministic_and_time_unit_equivariant(exact_intervals):
    frame, family = exact_intervals
    model = fit_interval_model(frame, family)
    first = model.predict(frame.half_life_seconds)
    repeated = model.predict(frame.half_life_seconds)
    for left, right in zip(first, repeated):
        np.testing.assert_array_equal(left, right)

    scaled = frame.copy()
    time_columns = ['half_life_seconds', 'ci_lower_seconds', 'ci_upper_seconds']
    scaled[time_columns] *= 60
    scaled_model = fit_interval_model(scaled, family)
    scaled_prediction = scaled_model.predict(scaled.half_life_seconds)
    for prediction, expected in zip(scaled_prediction, first):
        np.testing.assert_allclose(prediction, 60 * expected, rtol=1e-10, atol=1e-10)


def test_leave_one_peptide_out_does_not_fit_held_out_bounds(exact_intervals):
    frame, family = exact_intervals
    _, baseline, _ = validate_interval_estimator(frame, family)
    changed = frame.copy()
    changed.loc[0, 'ci_lower_seconds'] *= .8
    changed.loc[0, 'ci_upper_seconds'] *= 1.2
    metrics, altered, _ = validate_interval_estimator(changed, family)
    baseline_row = baseline.set_index('experiment_id').loc['experiment_0']
    altered_row = altered.set_index('experiment_id').loc['experiment_0']
    np.testing.assert_allclose(
        [baseline_row.estimated_lower_seconds, baseline_row.estimated_upper_seconds],
        [altered_row.estimated_lower_seconds, altered_row.estimated_upper_seconds],
        rtol=0, atol=0,
    )
    assert metrics.family.item() == family
    assert metrics.held_out_peptides.item() == frame.sequence_id.nunique()
    assert metrics.multiplicative_error_factor.item() >= 1


def test_rejects_invalid_endpoints_mixed_design_and_extrapolation(exact_intervals):
    frame, family = exact_intervals
    invalid = frame.copy()
    invalid.loc[0, 'ci_lower_seconds'] = invalid.loc[0, 'half_life_seconds']
    with pytest.raises(ValueError, match='lower < point < upper'):
        fit_interval_model(invalid, family)

    mixed = frame.copy()
    mixed.loc[0, 'peptide_concentration_um'] = 100
    with pytest.raises(ValueError, match='one assay design'):
        fit_interval_model(mixed, family)

    model = fit_interval_model(frame, family)
    with pytest.raises(ValueError, match='outside the fitted domain'):
        model.predict(frame.half_life_seconds.min() / 2)
