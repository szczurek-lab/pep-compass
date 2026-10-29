"""Deterministic point-to-interval estimators fitted to reported endpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)
FAMILIES = ('linear', 'logarithmic', 'exponential')
DESIGN_COLUMNS = ['source', 'species', 'matrix', 'matrix_fraction', 'in_vivo_in_vitro',
                  'temperature_c', 'ph', 'assay', 'replicate_count', 'peptide_concentration_um', 'ci_level']


def reported_intervals(experiments, sequences):
    """Select exact point estimates with published confidence-interval endpoints.

    :param pandas.DataFrame experiments: Processed experiment table.
    :param pandas.DataFrame sequences: Processed peptide registry.
    :return: Positive point estimates with strictly enclosing reported bounds.
    :rtype: pandas.DataFrame
    :raises ValueError: If a selected endpoint triplet is invalid.
    """
    keep = (experiments.measurement_kind.eq('half_life') & experiments.relation.eq('eq')
            & experiments.half_life_seconds.gt(0) & experiments.ci_level.notna()
            & experiments.ci_lower_seconds.notna() & experiments.ci_upper_seconds.notna())
    frame = experiments.loc[keep].merge(sequences, on='sequence_id', how='left', validate='many_to_one')
    if not frame.empty:
        _endpoint_arrays(frame)
    return frame.sort_values(['source', 'sequence_id', 'experiment_id']).reset_index(drop=True)


def _endpoint_arrays(frame):
    values = frame[['half_life_seconds', 'ci_lower_seconds', 'ci_upper_seconds']].to_numpy(dtype=float)  # (N, 3)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite, nonempty point estimates and interval endpoints are required')
    point, lower, upper = values.T  # each (N,)
    if not ((0 < lower) & (lower < point) & (point < upper)).all():
        raise ValueError('Reported bounds must satisfy 0 < lower < point < upper')
    return point, lower, upper


@dataclass
class IntervalModel:
    """Deterministic map from a half-life point estimate to lower and upper bounds.

    :param str family: ``linear``, ``logarithmic`` or ``exponential`` width model.
    :param float reference_seconds: Geometric mean of training point estimates.
    :param list coefficients: Width-model coefficients, one column per interval side.
    :param tuple domain_seconds: Minimum and maximum training point estimates.
    :param dict design: Assay design used to fit this empirical estimator.
    """

    family: str
    reference_seconds: float
    coefficients: list
    domain_seconds: tuple
    design: dict

    def predict(self, half_life_seconds, *, allow_extrapolation=False):
        """Return deterministic lower and upper bounds for one or more point estimates.

        The fitted positive distances are ``log(t / L)`` and ``log(U / t)``.
        Their family-specific functions produce ``L = t * exp(-d_lower)`` and
        ``U = t * exp(d_upper)``; no confidence interval is computed around the
        function or its parameters.

        :param half_life_seconds: Positive scalar or one-dimensional array of points.
        :param bool allow_extrapolation: Permit points outside the fitted domain.
        :return: Lower and upper bound arrays, each with shape ``(N,)``.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        :raises ValueError: If input is invalid, extrapolation is disabled, or bounds fail.
        """
        points = np.atleast_1d(np.asarray(half_life_seconds, dtype=float))  # (N,)
        if points.ndim != 1 or not np.isfinite(points).all() or (points <= 0).any():
            raise ValueError('Half-life points must be finite and positive')
        if not allow_extrapolation and ((points < self.domain_seconds[0]) | (points > self.domain_seconds[1])).any():
            raise ValueError('Half-life point lies outside the fitted domain; extrapolation is disabled')
        normalized = points / self.reference_seconds  # (N,)
        if self.family == 'logarithmic':
            predictor = np.column_stack([np.ones_like(normalized), np.log(normalized)])  # (N, 2)
        else:
            predictor = np.column_stack([np.ones_like(normalized), normalized])  # (N, 2)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            distances = predictor @ np.asarray(self.coefficients)  # (N, 2)
            if self.family == 'exponential':
                distances = np.exp(distances)  # (N, 2)
            lower = points * np.exp(-distances[:, 0])  # (N,)
            upper = points * np.exp(distances[:, 1])  # (N,)
        if not (np.isfinite(distances).all() and (distances > 0).all()
                and np.isfinite(lower).all() and np.isfinite(upper).all()
                and ((0 < lower) & (lower < points) & (points < upper)).all()):
            raise ValueError('The fitted width function does not produce finite enclosing bounds')
        return lower, upper

    def to_dict(self):
        """Return model coefficients, domain and fit design as serializable data.

        :return: Serializable model description.
        :rtype: dict
        """
        return asdict(self)


def _design_matrix(points, reference, family):
    normalized = points / reference  # (N,)
    coordinate = np.log(normalized) if family == 'logarithmic' else normalized  # (N,)
    return np.column_stack([np.ones_like(coordinate), coordinate])  # (N, 2)


def _design_metadata(frame):
    metadata = {}
    for column in DESIGN_COLUMNS:
        values = frame[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(f'Model requires one assay design; varying field: {column}')
        value = values.iloc[0]
        metadata[column] = None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
    if any(metadata[field] is None for field in ['replicate_count', 'peptide_concentration_um', 'ci_level']):
        raise ValueError('Documented replicate count, concentration and interval level are required')
    return metadata


def fit_interval_model(frame, family='linear'):
    """Fit positive relative interval widths for one documented assay design.

    For each side, the response is ``d_lower = log(t / L)`` or
    ``d_upper = log(U / t)``. ``linear`` fits ``d = a + b * (t / t_ref)``;
    ``logarithmic`` fits ``d = a + b * log(t / t_ref)``; ``exponential`` fits
    ``d = exp(a + b * (t / t_ref))``. All three fits minimize squared residuals
    in ``d``; these residuals equal log-endpoint errors. The resulting estimator
    maps ``t`` deterministically to ``L(t)`` and ``U(t)``.

    :param pandas.DataFrame frame: Reported point and bound columns plus design metadata.
    :param str family: One of :data:`FAMILIES`.
    :return: Fitted deterministic interval estimator.
    :rtype: IntervalModel
    :raises ValueError: If data, assay design, family or fitted widths are invalid.

    For N records and fixed two-parameter regressions, linear/logarithmic fitting
    is O(N); exponential fitting is O(KN) for K optimizer iterations. Input
    space is O(N), and prediction for Q points is O(Q) time and space.
    """
    if family not in FAMILIES or len(frame) < 3:
        raise ValueError('A supported family and at least three records are required')
    points, lower, upper = _endpoint_arrays(frame)
    metadata = _design_metadata(frame)
    reference = float(np.exp(np.log(points).mean()))
    design = _design_matrix(points, reference, family)  # (N, 2)
    distances = np.column_stack([np.log(points / lower), np.log(upper / points)])  # (N, 2)
    coefficients = np.empty((2, 2), dtype=float)  # (2 coefficients, lower/upper side)
    if family == 'exponential':
        log_design = design
        for side in range(2):
            initial = np.linalg.lstsq(log_design, np.log(distances[:, side]), rcond=None)[0]  # (2,)
            result = least_squares(
                lambda params: np.exp(log_design @ params) - distances[:, side],
                initial,
                method='lm',
            )
            if not result.success or not np.isfinite(result.x).all():
                raise ValueError('Exponential interval-width fit did not converge')
            coefficients[:, side] = result.x
    else:
        coefficients = np.linalg.lstsq(design, distances, rcond=None)[0]  # (2, 2)
    model = IntervalModel(family, reference, coefficients.tolist(),
                          (float(points.min()), float(points.max())), metadata)
    model.predict(points)
    return model


def _predict_held_out(frame, family):
    lower = np.empty(len(frame), dtype=float)  # (N,)
    upper = np.empty(len(frame), dtype=float)  # (N,)
    outside_domain = np.empty(len(frame), dtype=bool)  # (N,)
    for sequence_id in sorted(frame.sequence_id.unique()):
        held_out = frame.sequence_id.eq(sequence_id).to_numpy(dtype=bool)  # (N,)
        training = frame.loc[~held_out]
        model = fit_interval_model(training, family)
        points = frame.loc[held_out, 'half_life_seconds'].to_numpy(dtype=float)  # (N_held_out,)
        lower[held_out], upper[held_out] = model.predict(points, allow_extrapolation=True)
        train_points = training.half_life_seconds.to_numpy(dtype=float)  # (N_train,)
        outside_domain[held_out] = (points < train_points.min()) | (points > train_points.max())
    return lower, upper, outside_domain


def _score_predictions(lower, upper, frame):
    observed_lower = frame.ci_lower_seconds.to_numpy(dtype=float)  # (N,)
    observed_upper = frame.ci_upper_seconds.to_numpy(dtype=float)  # (N,)
    log_errors = np.concatenate([np.log(lower / observed_lower), np.log(upper / observed_upper)])  # (2N,)
    mean_absolute_log_error = float(np.abs(log_errors).mean())
    return {
        'mean_squared_log_endpoint_error': float(np.square(log_errors).mean()),
        'mean_absolute_log_endpoint_error': mean_absolute_log_error,
        'multiplicative_error_factor': float(np.exp(mean_absolute_log_error)),
        'median_absolute_lower_relative_error': float(np.median(np.abs(lower / observed_lower - 1))),
        'median_absolute_upper_relative_error': float(np.median(np.abs(upper / observed_upper - 1))),
    }


def validate_interval_estimator(frame, family='linear'):
    """Evaluate one deterministic point-to-interval family by peptide holdout.

    Each peptide is held out as a group; the remaining peptides fit the two
    relative-width functions. The held-out point estimate is then mapped to
    predicted bounds and compared with its reported bounds.

    :param pandas.DataFrame frame: Reported intervals from one assay design.
    :param str family: One of :data:`FAMILIES`.
    :return: One-row metric table, held-out predictions and full-data model.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame, IntervalModel]
    :raises ValueError: If fewer than five distinct peptide identities are present.

    With N distinct peptides, fixed two-parameter fits cost O(N^2) time and
    O(N) retained prediction space.
    """
    if frame.sequence_id.nunique() < 5:
        raise ValueError('At least five peptide identities are required for leave-one-peptide-out validation')
    ordered = frame.sort_values(['sequence_id', 'experiment_id']).reset_index(drop=True)
    lower, upper, outside_domain = _predict_held_out(ordered, family)
    predictions = ordered.copy()
    predictions['estimated_lower_seconds'] = lower
    predictions['estimated_upper_seconds'] = upper
    predictions['outside_training_domain'] = outside_domain
    metrics = _score_predictions(lower, upper, ordered)
    metrics = pd.DataFrame([{
        'family': family,
        'held_out_peptides': ordered.sequence_id.nunique(),
        **metrics,
    }])
    model = fit_interval_model(ordered, family)
    logger.info('Evaluated %s deterministic interval estimator on %s peptides', family,
                ordered.sequence_id.nunique())
    return metrics, predictions, model
