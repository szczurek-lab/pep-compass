"""Kernel ambient metric for PepCompass MUTANG neighborhood generation.

Implements PepCompass appendix D.3 ("Extension to biologically meaningful ambient
metrics by mutation kernels", arXiv 2510.01988 p.18) as an in-process
monkey-patch. **No file under the pep-compass checkout is modified.**

D.3 is an unrealised *proposal*: the paper gives no mutation kernel, no beta and
no results ("We plan to explore such metrics in future work"). What follows is
one instantiation of its framework, not a reproduction of a published run.

The construction
----------------
The ambient space is the flattened decoder output ``R^{L*V}`` with ``L = 25``
positions, ``V = 21`` tokens, position-major (``index = position * V + token``),
token ``0`` being HydrAMP padding. With a residue kernel ``K`` (20x20, strictly
PD) the ambient Gram matrix is the block lift

    G_K = I_L (x) B,     B = diag_block(1, K)         (21, 21)

i.e. the same residue block at every position, padding left Euclidean, no
positional coupling. D.3 permits but does not require positional coupling; this
is the choice the repo's own (deleted) ``ambient_metric`` module also made.

Phase 1 verified numerically that this block lift and a literal D.3
``V_Mut / V_Mut^perp`` projector-form Gram matrix induce the *same* pullback
metric (2e-06 relative, same kappa-stable eigenvalues), because the decoder's
softmax makes the Jacobian columns exactly zero-sum per position. The projector
form is therefore not needed here.

Deviation from D.3, deliberate
------------------------------
D.3 writes the equivalent modified decoder as ``Dec_K(z) = Sigma_K^{1/2} U_K^T
Dec(z)``. This module uses the **symmetric** square root ``G_K^{1/2}`` instead.
Both satisfy ``B^T B = G_K``, so the induced Riemannian metric on latent space is
*identical*; they differ by a left orthogonal factor and therefore give different
left singular vectors. That difference is load-bearing: MUTANG scores
*coordinates* of the left vectors in the token basis
(``max_j |U[(l,a),j]|``, ``MaxAbsoluteLoading``), which ``Sigma^{1/2} U^T`` would
rotate out of that basis. The repo's own deleted implementation made the same
choice.

The two patches
---------------
1. ``pep_compass.autoencoder.geometry.compute_tangent_decomposition`` --- the sole
   place in the live pipeline where a decoder Jacobian becomes an SVD. Patched to

       J~ = (I_L (x) B^{1/2}) J ;  U~, S, V = svd(J~) ;  U = (I_L (x) B^{-1/2}) U~

   ``S`` and ``V`` from the whitened SVD are already the correct pullback-metric
   quantities. ``U`` is mapped back to one-hot coordinates so MUTANG's per-token
   loadings keep their meaning; the returned columns are G-orthonormal
   (``U^T G U = I``) rather than Euclidean-orthonormal.

2. ``StableTangentGeometry.project_ambient_to_active_latent`` --- the SORBES
   acceleration pairing (consumed at ``sorbes.py:32``). The live code applies
   ``pinv(J_h)`` by pairing the ambient vector with ``left_vectors`` under a plain
   Euclidean dot product. With back-mapped left vectors ``G^{-1/2} U~`` that
   computes ``U~^T G^{-1/2} v``, whereas the G-inner product requires
   ``U~^T G^{1/2} v``. The patch pre-multiplies the ambient vector by ``G``:

       (G^{-1/2} U~)^T G v  =  U~^T G^{1/2} v.

   This reproduces exactly the algebra of the repo's own metric-aware reference
   implementation, whose ``SubRiemannianTangentSpace`` right-multiplied its
   pseudoinverse by ``ambient_transform = G^{1/2}`` (identified from the recovered
   bytecode, 1.6e-15 relative).

Flags
-----
``enable_metric=False`` installs nothing at all, so the unpatched path is
bit-identical by construction. ``identity=True`` installs the patches with
``K = I``, which exercises the patched code path and must reproduce the baseline
bit-identically --- the validation gate. ``correct_acceleration`` selects patch 2
independently of patch 1.

Every patched call is counted in :data:`CALL_COUNTS` so a silent Euclidean bypass
is detectable after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pathlib import Path

import numpy as np
import torch

import kernel_gram as kg


def _default_matrix_path() -> str:
    """Return the AMP-BLOSUM file shipped in this checkout."""
    relative = Path(
        "src/pep_compass/optimization/components/helpers/"
        "substitution_matrices/matrices/AMPBLOSUM62_float.txt"
    )
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(relative)


#: AMP-BLOSUM substitution matrix shipped with the pep-compass checkout.
AMP_BLOSUM_PATH = _default_matrix_path()

#: Ambient sequence length and token-alphabet size of the HydrAMP decoder.
MAX_LEN = 25
ALPHABET_SIZE = len(kg.PADDED_ALPHABET)

#: Kernel inverse-temperature fixed by the user for this arm.
DEFAULT_BETA = 4.0

#: Counts of patched-path executions, for bypass detection.
CALL_COUNTS: dict[str, int] = {"tangent_decomposition": 0, "ambient_projection": 0}


def reset_call_counts() -> None:
    """Zero :data:`CALL_COUNTS`."""
    for key in CALL_COUNTS:
        CALL_COUNTS[key] = 0


@dataclass(frozen=True)
class AmbientMetric:
    """Hold the token-level Gram block and its roots, plus cached ambient lifts.

    :param token_gram: ``(21, 21)`` Gram block ``B``.
    :param token_sqrt: ``(21, 21)`` symmetric square root ``B^{1/2}``.
    :param token_sqrt_inverse: ``(21, 21)`` inverse square root ``B^{-1/2}``.
    :param max_len: Ambient sequence length ``L``.
    :param description: Human-readable provenance of the residue kernel.
    """

    token_gram: np.ndarray
    token_sqrt: np.ndarray
    token_sqrt_inverse: np.ndarray
    max_len: int = MAX_LEN
    description: str = ""
    _cache: dict[tuple[str, torch.dtype, str], torch.Tensor] = field(
        default_factory=dict, repr=False, compare=False
    )

    def ambient(
        self, which: str, *, dtype: torch.dtype, device: torch.device | str
    ) -> torch.Tensor:
        """Return a cached ``(L*21, L*21)`` ambient lift of one token block.

        :param which: ``"gram"``, ``"sqrt"`` or ``"sqrt_inverse"``.
        :param dtype: Target tensor dtype.
        :param device: Target tensor device.
        :return: ``I_L (x) block`` as a tensor.
        """
        key = (which, dtype, str(device))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        block = {
            "gram": self.token_gram,
            "sqrt": self.token_sqrt,
            "sqrt_inverse": self.token_sqrt_inverse,
        }[which]
        identity = torch.eye(self.max_len, dtype=dtype, device=device)
        lift = torch.kron(identity, torch.as_tensor(block, dtype=dtype, device=device))
        self._cache[key] = lift
        return lift

    @property
    def is_identity(self) -> bool:
        """Return whether the token block is exactly the identity."""
        return bool(np.array_equal(self.token_gram, np.eye(self.token_gram.shape[0])))


def build_ambient_metric(
    *,
    matrix_path: str = AMP_BLOSUM_PATH,
    beta: float = DEFAULT_BETA,
    max_len: int = MAX_LEN,
    identity: bool = False,
    residue_gram: np.ndarray | None = None,
) -> AmbientMetric:
    """Build the ambient metric from a substitution matrix or an explicit kernel.

    The default path is the user-fixed construction ``K = exp(beta * (S + cI))``
    entrywise on AMP-BLOSUM with ``c = -min eig(S)``, taken from
    :mod:`kernel_gram`. This is *not* the repo's
    ``build_metric(construction="entrywise_exp")``, which applies the exponential
    to an indefinite matrix and then discards negative eigenvalue mass by PSD
    projection.

    :param matrix_path: NCBI-format substitution matrix, ignored if
        ``residue_gram`` or ``identity`` is given.
    :param beta: Kernel inverse-temperature.
    :param max_len: Ambient sequence length ``L``.
    :param identity: Use ``K = I_20`` (the validation control).
    :param residue_gram: Explicit ``(20, 20)`` strictly-PD residue kernel, e.g.
        one produced by the repo's own construction for a replication check.
    :return: The assembled :class:`AmbientMetric`.
    """
    if identity:
        residue = np.eye(len(kg.AMINO_ACIDS), dtype=np.float64)
        description = "K = I (identity control)"
    elif residue_gram is not None:
        residue = np.asarray(residue_gram, dtype=np.float64)
        residue = 0.5 * (residue + residue.T)
        description = "explicit residue kernel"
    else:
        scores = kg.read_score_matrix(matrix_path)
        residue = kg.mutation_kernel(scores, beta=beta)
        description = (
            f"entrywise exp(beta*(S+cI)) on {matrix_path.rsplit('/', 1)[-1]}, "
            f"beta={beta}, c={kg.diagonal_shift(scores):.10f}"
        )

    root = kg.symmetric_sqrt(residue)
    token_gram = kg.pad_residue_block(residue)
    token_sqrt = kg.pad_residue_block(root)
    token_sqrt_inverse = kg.pad_residue_block(np.linalg.inv(root))
    return AmbientMetric(
        token_gram=token_gram,
        token_sqrt=token_sqrt,
        token_sqrt_inverse=token_sqrt_inverse,
        max_len=max_len,
        description=description,
    )


def whitened_decomposition(
    jacobians: torch.Tensor, metric: AmbientMetric
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD decoder Jacobians under the ambient metric, in one-hot coordinates.

    :param jacobians: Decoder Jacobians ``(B, A, D)``.
    :param metric: Ambient metric to whiten with.
    :return: ``(left, singular, right)`` with ``left`` mapped back through
        ``G^{-1/2}`` so its rows stay one-hot token coordinates.
    """
    forward = metric.ambient("sqrt", dtype=jacobians.dtype, device=jacobians.device)
    backward = metric.ambient(
        "sqrt_inverse", dtype=jacobians.dtype, device=jacobians.device
    )
    left, singular, right = torch.linalg.svd(forward @ jacobians, full_matrices=False)
    return backward @ left, singular, right


def gram_orthonormality_residual(
    left: torch.Tensor, metric: AmbientMetric
) -> tuple[float, float]:
    """Return ``max|U^T G U - I|`` and ``max|U^T U - I|`` for one batch element.

    :param left: Left vectors ``(A, K)`` in one-hot coordinates.
    :param metric: Ambient metric.
    :return: G-orthonormality residual and Euclidean-orthonormality residual.
    """
    gram = metric.ambient("gram", dtype=left.dtype, device=left.device)
    identity = torch.eye(left.shape[1], dtype=left.dtype, device=left.device)
    g_residual = (left.T @ gram @ left - identity).abs().max().item()
    e_residual = (left.T @ left - identity).abs().max().item()
    return g_residual, e_residual


@dataclass
class InstalledPatches:
    """Handle for restoring the unpatched pep-compass functions.

    :param metric: The installed ambient metric, or ``None`` if disabled.
    :param patched_tangent: Whether the tangent decomposition was replaced.
    :param patched_acceleration: Whether the SORBES ambient pairing was replaced.
    """

    metric: AmbientMetric | None
    patched_tangent: bool
    patched_acceleration: bool
    _restore: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def uninstall(self) -> None:
        """Restore every replaced attribute."""
        for restore in reversed(self._restore):
            restore()
        self._restore.clear()
        self.patched_tangent = False
        self.patched_acceleration = False

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of what is installed."""
        return {
            "patched_tangent_decomposition": self.patched_tangent,
            "patched_ambient_projection": self.patched_acceleration,
            "metric": None if self.metric is None else self.metric.description,
            "metric_is_identity": None
            if self.metric is None
            else self.metric.is_identity,
            "call_counts": dict(CALL_COUNTS),
        }


def install(
    *,
    enable_metric: bool = True,
    correct_acceleration: bool = True,
    beta: float = DEFAULT_BETA,
    matrix_path: str = AMP_BLOSUM_PATH,
    max_len: int = MAX_LEN,
    identity: bool = False,
    residue_gram: np.ndarray | None = None,
) -> InstalledPatches:
    """Monkey-patch the kernel ambient metric into the live pipeline.

    :param enable_metric: If ``False``, install nothing; the original code runs
        untouched and bit-identically.
    :param correct_acceleration: Also correct the SORBES ambient pairing
        (patch 2). ``False`` leaves ``sorbes.py:32`` pairing under the Euclidean
        dot product, which is metrically inconsistent; kept as a flag only so the
        two variants can be compared.
    :param beta: Kernel inverse-temperature.
    :param matrix_path: Substitution matrix for the kernel.
    :param max_len: Ambient sequence length ``L``.
    :param identity: Install with ``K = I`` (validation control).
    :param residue_gram: Explicit residue kernel, overriding ``matrix_path``.
    :return: Handle exposing :meth:`InstalledPatches.uninstall`.
    """
    reset_call_counts()
    if not enable_metric:
        return InstalledPatches(metric=None, patched_tangent=False, patched_acceleration=False)

    from pep_compass.autoencoder import geometry as geometry_module

    metric = build_ambient_metric(
        matrix_path=matrix_path,
        beta=beta,
        max_len=max_len,
        identity=identity,
        residue_gram=residue_gram,
    )
    restores: list[Callable[[], None]] = []

    original_tangent = geometry_module.compute_tangent_decomposition

    def patched_tangent_decomposition(autoencoder, latent_positions):
        """Whitened replacement for ``compute_tangent_decomposition``."""
        if latent_positions.ndim == 1:
            latent_positions = latent_positions.unsqueeze(0)
        jacobians = autoencoder.decoder_jacobian(latent_positions)  # (B, A, D)
        left, singular, right = whitened_decomposition(jacobians, metric)
        CALL_COUNTS["tangent_decomposition"] += 1
        return geometry_module.TangentDecomposition(left, singular, right)

    patched_tangent_decomposition.__wrapped__ = original_tangent
    geometry_module.compute_tangent_decomposition = patched_tangent_decomposition
    restores.append(
        lambda: setattr(
            geometry_module, "compute_tangent_decomposition", original_tangent
        )
    )

    if not correct_acceleration:
        return InstalledPatches(
            metric=metric,
            patched_tangent=True,
            patched_acceleration=False,
            _restore=restores,
        )

    original_project = geometry_module.StableTangentGeometry.project_ambient_to_active_latent

    def patched_project(self, ambient_vectors: torch.Tensor) -> torch.Tensor:
        """Pair the ambient vector through ``G`` before the Euclidean pinv."""
        gram = metric.ambient(
            "gram", dtype=ambient_vectors.dtype, device=ambient_vectors.device
        )
        CALL_COUNTS["ambient_projection"] += 1
        return original_project(self, ambient_vectors @ gram)  # G symmetric

    geometry_module.StableTangentGeometry.project_ambient_to_active_latent = (
        patched_project
    )
    restores.append(
        lambda: setattr(
            geometry_module.StableTangentGeometry,
            "project_ambient_to_active_latent",
            original_project,
        )
    )
    return InstalledPatches(
        metric=metric,
        patched_tangent=True,
        patched_acceleration=True,
        _restore=restores,
    )
