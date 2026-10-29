"""Adapters for the published PeptiVerse half-life regressors.

The upstream repository ships eight trained half-life artifacts, combining two
estimator families (a transformer regression head and XGBoost) with four input
representations (ESM-2 sequence embeddings on a ``log1p`` and on a raw target,
PeptideCLM SMILES embeddings and ChemBERTa SMILES embeddings).

The upstream inference script scores one peptide per call. This adapter loads
the same artifacts and re-implements the forward path in batched form, which
is what the pipeline of the main library requires; :func:`verify_batch_equivalence`
checks the batched path against the per-peptide path numerically.

Reference: PeptiVerse, Chatterjee Lab, https://huggingface.co/ChatterjeeLab/PeptiVerse
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from model_benchmark.config import PEPTIVERSE_DIR
from model_benchmark.predictors.base import HalfLifePredictor, PredictorMetadata

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600.0
## REMARK: Raw-target artifacts can emit non-positive hours, which have no
## logarithm. Such predictions are clipped to this floor and counted, because
## their frequency is itself a result about model calibration.
MINIMUM_HOURS = 1e-4


@dataclass(frozen=True)
class PeptiVerseVariant:
    """Description of one published half-life artifact.

    :param str key: Adapter identifier.
    :param str folder: Artifact directory below ``training_classifiers/half_life``.
    :param str estimator: ``transformer`` or ``xgb``.
    :param str backbone: Representation backbone key.
    :param str target: ``log1p`` when the artifact predicts ``log1p(hours)``, else ``raw``.
    :param str input_kind: ``sequence`` or ``smiles``.
    """

    key: str
    folder: str
    estimator: str
    backbone: str
    target: str
    input_kind: str


VARIANTS = {
    variant.key: variant
    for variant in (
        PeptiVerseVariant("peptiverse_transformer_esm_log", "transformer_wt_log", "transformer", "esm", "log1p", "sequence"),
        PeptiVerseVariant("peptiverse_transformer_esm_raw", "transformer_wt_raw", "transformer", "esm", "raw", "sequence"),
        PeptiVerseVariant("peptiverse_xgb_esm_log", "xgb_wt_log", "xgb", "esm", "log1p", "sequence"),
        PeptiVerseVariant("peptiverse_xgb_esm_raw", "xgb_wt_raw", "xgb", "esm", "raw", "sequence"),
        PeptiVerseVariant("peptiverse_transformer_peptideclm", "transformer_peptideclm", "transformer", "peptideclm", "raw", "smiles"),
        PeptiVerseVariant("peptiverse_xgb_peptideclm", "xgb_peptideclm", "xgb", "peptideclm", "raw", "smiles"),
        PeptiVerseVariant("peptiverse_transformer_chemberta", "transformer_chemberta", "transformer", "chemberta", "raw", "smiles"),
        PeptiVerseVariant("peptiverse_xgb_chemberta", "xgb_chemberta", "xgb", "chemberta", "raw", "smiles"),
    )
}

BACKBONE_NAMES = {
    "esm": "facebook/esm2_t33_650M_UR50D",
    "peptideclm": "aaronfeller/PeptideCLM-23M-all",
    "chemberta": "DeepChem/ChemBERTa-77M-MLM",
}


# ---------------------------------------------------------------------------
# Regression heads, reproduced from the published inference script
# ---------------------------------------------------------------------------
class TransformerHead(nn.Module):
    """Masked-mean transformer regression head over token embeddings."""

    def __init__(self, in_dim, d_model=256, nhead=8, layers=2, ff=512, dropout=0.1):
        """Build the head with the architecture stored in the checkpoint.

        :param int in_dim: Backbone embedding width.
        :param int d_model: Internal model width.
        :param int nhead: Attention heads.
        :param int layers: Encoder layers.
        :param int ff: Feed-forward width.
        :param float dropout: Dropout probability.
        :rtype: None
        """
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, embeddings, mask):
        """Map token embeddings to one scalar per peptide.

        :param torch.Tensor embeddings: Token embeddings, shape ``(B, L, D)``.
        :param torch.Tensor mask: Valid-token mask, shape ``(B, L)``.
        :return: One prediction per peptide, shape ``(B,)``.
        :rtype: torch.Tensor
        """
        encoded = self.enc(self.proj(embeddings), src_key_padding_mask=~mask)  # (B, L, d_model)
        weights = mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)  # (B, d_model)
        return self.head(pooled).squeeze(-1)  # (B,)


def _infer_transformer_architecture(state_dict):
    """Recover head hyperparameters from a checkpoint state dictionary.

    :param dict state_dict: Checkpoint tensors.
    :return: Input width, model width, layer count and feed-forward width.
    :rtype: tuple[int, int, int, int]
    :raises KeyError: If the projection weight is missing.
    """
    if "proj.weight" not in state_dict:
        raise KeyError("Checkpoint lacks proj.weight; not a transformer head")
    in_dim = int(state_dict["proj.weight"].shape[1])
    d_model = int(state_dict["proj.weight"].shape[0])
    indices = {
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"enc\.layers\.(\d+)\.", key))
    }
    layers = max(indices) + 1 if indices else 1
    ff = int(state_dict.get("enc.layers.0.linear1.weight", torch.empty(4 * d_model, 1)).shape[0])
    return in_dim, d_model, layers, ff


def _pick_head_count(d_model):
    """Choose the attention-head count the published builder would choose.

    :param int d_model: Model width.
    :return: Largest supported head count dividing ``d_model``.
    :rtype: int
    """
    for count in (8, 6, 4, 3, 2, 1):
        if d_model % count == 0:
            return count
    return 1


def build_transformer_head(checkpoint, device):
    """Instantiate the published transformer head from its checkpoint.

    :param dict checkpoint: Loaded ``best_model.pt`` payload.
    :param torch.device device: Target device.
    :return: Head in evaluation mode with loaded weights.
    :rtype: TransformerHead
    """
    state_dict = checkpoint["state_dict"]
    parameters = checkpoint.get("best_params", {})
    in_dim, d_model, layers, ff = _infer_transformer_architecture(state_dict)
    head = TransformerHead(
        in_dim=int(checkpoint.get("in_dim", in_dim)),
        d_model=d_model,
        nhead=int(parameters.get("nhead", _pick_head_count(d_model))),
        layers=int(parameters.get("layers", layers)),
        ff=int(parameters.get("ff", ff)),
        dropout=float(parameters.get("dropout", 0.1)),
    )
    ## REMARK: Strict loading is deliberate: a silent architecture mismatch
    ## would produce plausible but wrong predictions.
    head.load_state_dict(state_dict)
    return head.to(device).eval()


# ---------------------------------------------------------------------------
# Representation backbones, batched
# ---------------------------------------------------------------------------
class _Backbone:
    """Batched token embedder reproducing the published pooling rule."""

    def __init__(self, tokenizer, model, device, max_length):
        """Store the loaded tokenizer and encoder.

        :param tokenizer: Hugging Face tokenizer of the backbone.
        :param torch.nn.Module model: Encoder returning ``last_hidden_state``.
        :param torch.device device: Device holding the encoder.
        :param int max_length: Truncation length.
        :rtype: None
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = max_length
        candidates = [
            getattr(tokenizer, f"{kind}_token_id", None)
            for kind in ("pad", "cls", "sep", "bos", "eos", "mask")
        ]
        special = sorted({int(value) for value in candidates if value is not None})
        self.special_ids = (
            torch.tensor(special, device=device, dtype=torch.long) if special else None
        )

    @property
    def width(self):
        """Return the embedding width of the backbone.

        :return: Embedding dimension.
        :rtype: int
        """
        return int(self.model.config.hidden_size)

    def _tokenize(self, items):
        """Tokenize a batch with padding and truncation.

        :param list[str] items: Sequences or SMILES strings.
        :return: Token identifiers and attention mask on the backbone device.
        :rtype: dict[str, torch.Tensor]
        """
        encoded = self.tokenizer(
            items, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        return encoded

    def _valid_mask(self, encoded):
        """Build the mask of content tokens, excluding special tokens.

        :param dict encoded: Tokenizer output.
        :return: Boolean mask of shape ``(B, L)``.
        :rtype: torch.Tensor
        """
        valid = encoded["attention_mask"].bool()  # (B, L)
        if self.special_ids is not None and self.special_ids.numel():
            valid = valid & ~torch.isin(encoded["input_ids"], self.special_ids)  # (B, L)
        return valid

    def _hidden_states(self, encoded):
        """Run the encoder on a tokenized batch.

        :param dict encoded: Tokenizer output.
        :return: Token embeddings of shape ``(B, L, D)``.
        :rtype: torch.Tensor
        """
        return self.model(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        ).last_hidden_state  # (B, L, D)

    @torch.no_grad()
    def embed_pooled(self, items):
        """Return mean-pooled embeddings of content tokens.

        :param list[str] items: Sequences or SMILES strings.
        :return: Pooled embeddings of shape ``(B, D)``.
        :rtype: numpy.ndarray
        """
        encoded = self._tokenize(items)
        hidden = self._hidden_states(encoded)  # (B, L, D)
        weights = self._valid_mask(encoded).unsqueeze(-1).float()  # (B, L, 1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)  # (B, D)
        features = pooled.detach().cpu().numpy().astype(np.float32)  # (B, D)
        features = np.nan_to_num(features, nan=0.0)
        return np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)

    @torch.no_grad()
    def embed_tokens(self, items):
        """Return left-packed content-token embeddings and their mask.

        Content tokens are packed to the front so that a padded batch yields
        exactly the per-peptide tensors of the published implementation.

        :param list[str] items: Sequences or SMILES strings.
        :return: Embeddings ``(B, L_max, D)`` and mask ``(B, L_max)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        encoded = self._tokenize(items)
        hidden = self._hidden_states(encoded)  # (B, L, D)
        valid = self._valid_mask(encoded)  # (B, L)
        return _pack_valid_tokens(hidden, valid)


def _pack_valid_tokens(hidden, valid):
    """Pack valid token embeddings to the front of each row.

    :param torch.Tensor hidden: Token embeddings, shape ``(B, L, D)``.
    :param torch.Tensor valid: Valid-token mask, shape ``(B, L)``.
    :return: Packed embeddings ``(B, L_max, D)`` and mask ``(B, L_max)``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: If a row contains no valid token.
    """
    counts = valid.sum(dim=1)  # (B,)
    if int(counts.min()) < 1:
        raise ValueError("A peptide produced no content tokens")
    width = int(counts.max())
    packed = hidden.new_zeros((hidden.shape[0], width, hidden.shape[2]))  # (B, L_max, D)
    mask = torch.zeros((hidden.shape[0], width), dtype=torch.bool, device=hidden.device)
    for row in range(hidden.shape[0]):
        selected = hidden[row][valid[row]]  # (L_row, D)
        packed[row, : selected.shape[0]] = selected
        mask[row, : selected.shape[0]] = True
    return packed, mask


class EsmBackbone(_Backbone):
    """ESM-2 650M sequence backbone used by the ``wt`` artifacts."""

    def __init__(self, device, name=None):
        """Load the ESM-2 encoder.

        :param torch.device device: Target device.
        :param str name: Optional backbone override.
        :rtype: None
        """
        from transformers import EsmModel, EsmTokenizer

        model_name = name or BACKBONE_NAMES["esm"]
        logger.info("Loading sequence backbone %s", model_name)
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).to(device).eval()
        super().__init__(tokenizer, model, device, max_length=1022)


class ChemBertaBackbone(_Backbone):
    """ChemBERTa SMILES backbone used by the ``chemberta`` artifacts."""

    def __init__(self, device, name=None):
        """Load the ChemBERTa encoder.

        :param torch.device device: Target device.
        :param str name: Optional backbone override.
        :rtype: None
        """
        from transformers import AutoModel, AutoTokenizer

        model_name = name or BACKBONE_NAMES["chemberta"]
        logger.info("Loading SMILES backbone %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        super().__init__(tokenizer, model, device, max_length=512)


class PeptideClmBackbone(_Backbone):
    """PeptideCLM SMILES backbone with the published SMILES-pair tokenizer."""

    def __init__(self, device, root=PEPTIVERSE_DIR, name=None):
        """Load the PeptideCLM encoder and its vocabulary files.

        :param torch.device device: Target device.
        :param pathlib.Path root: Downloaded PeptiVerse snapshot.
        :param str name: Optional backbone override.
        :rtype: None
        """
        import sys

        from transformers import AutoModelForMaskedLM

        ## REMARK: The published tokenizer module ships inside the snapshot and
        ## is imported from there, so no upstream code enters this repository.
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer

        model_name = name or BACKBONE_NAMES["peptideclm"]
        logger.info("Loading SMILES backbone %s", model_name)
        tokenizer = SMILES_SPE_Tokenizer(
            str(root / "tokenizer/new_vocab.txt"), str(root / "tokenizer/new_splits.txt")
        )
        model = AutoModelForMaskedLM.from_pretrained(model_name).roformer.to(device).eval()
        super().__init__(tokenizer, model, device, max_length=512)


# ---------------------------------------------------------------------------
# Conformal calibration bundles
# ---------------------------------------------------------------------------
class _PassthroughRegressor:
    """Stub of the upstream estimator referenced inside calibration pickles."""

    def __init__(self, preds=None):
        """Store the stored predictions of the pickled estimator.

        :param numpy.ndarray preds: Predictions carried by the pickle.
        :rtype: None
        """
        self.preds = preds

    def fit(self, features, target):
        """Return self; the stub is never refitted.

        :param features: Ignored.
        :param target: Ignored.
        :return: This estimator.
        :rtype: _PassthroughRegressor
        """
        return self

    def predict(self, features):
        """Return the stored predictions.

        :param features: Input whose length selects the stored predictions.
        :return: Stored predictions.
        :rtype: numpy.ndarray
        """
        return self.preds[: len(features)]


def _register_pickle_stubs():
    """Make the upstream pickle classes importable before unpickling.

    The calibration bundles were pickled from a training script, so they
    reference ``__main__.PassthroughRegressor``.

    :rtype: None
    """
    import sys

    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "PassthroughRegressor"):
        main_module.PassthroughRegressor = _PassthroughRegressor


def _conformal_quantile(scores, alpha):
    """Compute the finite-sample conformal half-width from conformity scores.

    :param numpy.ndarray scores: Absolute conformity scores, shape ``(n,)``.
    :param float alpha: Miscoverage level of the published bundle.
    :return: Half-width and the quantile level actually used.
    :rtype: tuple[float, float]
    :raises ValueError: If no finite conformity score is present.
    """
    finite = np.abs(np.asarray(scores, dtype=float).ravel())  # (n,)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("Calibration bundle carries no finite conformity score")
    level = min(1.0, np.ceil((finite.size + 1) * (1.0 - alpha)) / finite.size)
    return float(np.quantile(finite, level)), float(level)


def load_conformal_calibration(path):
    """Read the conformal half-width published next to an artifact.

    Two bundle formats occur upstream: a plain dictionary carrying the
    half-width, and a pickled MAPIE regressor from which the half-width is
    recovered through its conformity scores.

    :param pathlib.Path path: Calibration bundle path.
    :return: Half-width with its provenance, or None when unavailable.
    :rtype: dict | None
    """
    if not path.is_file():
        return None

    import joblib

    _register_pickle_stubs()
    try:
        bundle = joblib.load(path)
    except Exception as error:  # noqa: BLE001 - upstream formats vary by artifact
        logger.warning("Could not load calibration %s: %s", path, error)
        return None

    alpha = float(bundle.get("alpha", 0.1)) if isinstance(bundle, dict) else 0.1

    ## Plain half-width bundle
    if isinstance(bundle, dict) and "quantile" in bundle:
        return {
            "quantile": float(bundle["quantile"]),
            "alpha": alpha,
            "format": "plain_quantile",
            "calibration_size": None,
        }

    ## Pickled MAPIE regressor
    ## REMARK: The published inference script only accepts the plain format, so
    ## for these artifacts its conformal interval is unavailable; the
    ## half-width is recovered here directly from the conformity scores.
    estimator = bundle.get("mapie") if isinstance(bundle, dict) else bundle
    inner = getattr(estimator, "_mapie_regressor", estimator)
    scores = getattr(inner, "conformity_scores_", None)
    if scores is None:
        logger.warning("Calibration %s has no conformity scores", path)
        return None
    quantile, level = _conformal_quantile(scores, alpha)
    return {
        "quantile": quantile,
        "alpha": alpha,
        "format": "mapie_object",
        "calibration_size": int(np.isfinite(np.asarray(scores, dtype=float).ravel()).sum()),
        "quantile_level": level,
        "accepted_by_published_loader": False,
    }


_BACKBONE_CACHE = {}


def get_backbone(key, device):
    """Return a cached backbone instance.

    :param str key: Backbone key of :data:`BACKBONE_NAMES`.
    :param torch.device device: Target device.
    :return: Shared backbone instance.
    :rtype: _Backbone
    :raises KeyError: If the backbone key is unknown.
    """
    cache_key = (key, str(device))
    if cache_key in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[cache_key]
    builders = {
        "esm": EsmBackbone,
        "chemberta": ChemBertaBackbone,
        "peptideclm": PeptideClmBackbone,
    }
    if key not in builders:
        raise KeyError(f"Unknown backbone {key!r}")
    backbone = builders[key](device)
    _BACKBONE_CACHE[cache_key] = backbone
    return backbone


def sequence_to_smiles(sequence):
    """Convert a linear peptide sequence to a canonical SMILES string.

    :param str sequence: Peptide sequence over the canonical alphabet.
    :return: Canonical SMILES of the free-termini L-peptide.
    :rtype: str
    :raises ValueError: If the sequence cannot be converted.
    """
    from rdkit import Chem

    molecule = Chem.MolFromSequence(sequence)
    if molecule is None:
        raise ValueError(f"RDKit could not build a peptide from {sequence!r}")
    return Chem.MolToSmiles(molecule)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class PeptiVersePredictor(HalfLifePredictor):
    """One published PeptiVerse half-life artifact behind the batch interface."""

    default_batch_size = 32

    def __init__(
        self,
        variant_key,
        device=None,
        batch_size=None,
        root=PEPTIVERSE_DIR,
        apply_scaler=False,
        output_convention="trained_target",
    ):
        """Load one published artifact and its calibration bundle.

        :param str variant_key: Key of :data:`VARIANTS`.
        :param str device: Torch device string; CUDA when available by default.
        :param int batch_size: Peptides per forward pass.
        :param pathlib.Path root: Downloaded PeptiVerse snapshot.
        :param bool apply_scaler: Apply the ``scaler.joblib`` shipped beside the
            XGBoost artifacts. The published inference script never applies it;
            both behaviours are measured in this study.
        :param str output_convention: ``trained_target`` inverts the ``log1p``
            target transform, which is how the artifact was trained.
            ``published_inference`` reproduces the deployed script, which omits
            that inversion and therefore reports ``log1p(hours)`` as hours.
        :rtype: None
        :raises KeyError: If the variant key is unknown.
        :raises ValueError: If the output convention is unknown.
        :raises FileNotFoundError: If the artifact is missing from the snapshot.
        """
        if variant_key not in VARIANTS:
            raise KeyError(f"Unknown variant {variant_key!r}; available: {sorted(VARIANTS)}")
        if output_convention not in {"trained_target", "published_inference"}:
            raise ValueError("output_convention must be 'trained_target' or 'published_inference'")
        variant = VARIANTS[variant_key]
        self.variant = variant
        self.root = root
        self.apply_scaler = bool(apply_scaler)
        self.output_convention = output_convention
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.clipped_predictions = 0

        model_dir = root / "training_classifiers/half_life" / variant.folder
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Missing artifact directory {model_dir}; run scripts/download_peptiverse.py"
            )
        self.model_dir = model_dir

        # Estimator
        self.scaler = None
        if variant.estimator == "transformer":
            checkpoint = torch.load(model_dir / "best_model.pt", map_location=self.device, weights_only=False)
            self.head = build_transformer_head(checkpoint, self.device)
            self.booster = None
            architecture = "transformer regression head on token embeddings"
        else:
            import xgboost as xgb

            self.booster = xgb.Booster()
            self.booster.load_model(str(model_dir / "best_model.json"))
            self.head = None
            architecture = "XGBoost on mean-pooled embeddings"
            scaler_path = model_dir / "scaler.joblib"
            if self.apply_scaler and scaler_path.is_file():
                import joblib

                self.scaler = joblib.load(scaler_path)
                logger.info("Applying %s for %s", scaler_path.name, variant.key)

        # Conformal calibration
        self.conformal = load_conformal_calibration(model_dir / "mapie_calibration.joblib")

        self.backbone = get_backbone(variant.backbone, self.device)
        super().__init__(
            PredictorMetadata(
                name=variant.key,
                publication="PeptiVerse (Chatterjee Lab), Nature Communications 2026",
                artifact=f"ChatterjeeLab/PeptiVerse:training_classifiers/half_life/{variant.folder}",
                representation=f"{BACKBONE_NAMES[variant.backbone]} ({variant.input_kind})",
                architecture=architecture,
                training_records="130 sequence-based / 245 SMILES-based (reported)",
                native_units="hours",
                supports_gradient=variant.estimator == "transformer",
                details={
                    "target_transform": variant.target,
                    "output_convention": output_convention,
                    "device": str(self.device),
                    "has_conformal_interval": self.conformal is not None,
                    "conformal_calibration": self.conformal,
                    "scaler_applied": self.scaler is not None,
                    "published_benchmark": self._published_benchmark(),
                },
            ),
            batch_size=batch_size,
        )

    def _published_benchmark(self):
        """Return the benchmark the authors stored beside the artifact.

        :return: Reported benchmark values, empty when the artifact has none.
        :rtype: dict
        """
        path = self.model_dir / "final_model_benchmark.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _model_inputs(self, sequences):
        """Translate peptide sequences into the representation the artifact expects.

        :param list[str] sequences: Peptide sequences.
        :return: Strings consumed by the backbone tokenizer.
        :rtype: list[str]
        """
        if self.variant.input_kind == "smiles":
            return [sequence_to_smiles(sequence) for sequence in sequences]
        return list(sequences)

    def _raw_predictions(self, sequences):
        """Produce the artifact's native output for one batch of peptides.

        :param list[str] sequences: Peptide sequences of one batch.
        :return: Native predictions in hours, shape ``(B,)``.
        :rtype: numpy.ndarray
        """
        return self.predict_from_model_inputs(self._model_inputs(sequences))

    def predict_from_model_inputs(self, items):
        """Score inputs already expressed in the artifact's representation.

        This bypasses the sequence-to-SMILES conversion, which is needed to
        score the authors' own training table, where structures are published
        as SMILES.

        :param list[str] items: Sequences or SMILES strings, matching the artifact.
        :return: Native predictions in hours, shape ``(B,)``.
        :rtype: numpy.ndarray
        """
        ## Estimator forward pass
        if self.variant.estimator == "transformer":
            embeddings, mask = self.backbone.embed_tokens(items)  # (B, L, D), (B, L)
            with torch.no_grad():
                raw = self.head(embeddings, mask).detach().cpu().numpy().astype(float)  # (B,)
        else:
            import xgboost as xgb

            features = self.backbone.embed_pooled(items)  # (B, D)
            if self.scaler is not None:
                features = self.scaler.transform(features)  # (B, D)
            raw = np.asarray(self.booster.predict(xgb.DMatrix(features)), dtype=float)  # (B,)

        ## Target inverse transform
        ## REMARK: The log artifacts were trained on log1p(hours), so hours are
        ## recovered with expm1. The published inference script never performs
        ## this step: it tests ``"log" in model_name`` where model_name is the
        ## manifest alias ("transformer", "xgb"), so the branch is unreachable
        ## and the deployed predictor reports log1p(hours) as hours. Both
        ## conventions are available, and the difference is measured in this
        ## study; see results/upstream_comparison.csv.
        if self.variant.target == "log1p" and self.output_convention == "trained_target":
            raw = np.expm1(raw)  # (B,)
        return np.atleast_1d(raw)

    def _predict_log10_seconds(self, sequences):
        """Predict half-life in ``log10`` seconds for one batch.

        :param list[str] sequences: Peptide sequences of one batch.
        :return: Predictions in ``log10`` seconds, shape ``(B,)``.
        :rtype: numpy.ndarray
        """
        hours = self._raw_predictions(sequences)  # (B,)
        non_positive = int((hours < MINIMUM_HOURS).sum())
        if non_positive:
            self.clipped_predictions += non_positive
            logger.debug("%s clipped %s non-positive predictions", self.name, non_positive)
        hours = np.clip(hours, MINIMUM_HOURS, None)  # (B,)
        return np.log10(hours * SECONDS_PER_HOUR)  # (B,)

    def predict_hours(self, sequences):
        """Predict half-life in the published unit.

        :param sequences: Iterable of peptide sequences.
        :return: Predicted half-life in hours, shape ``(N,)``.
        :rtype: numpy.ndarray
        """
        return 10.0 ** self.predict_log10_seconds(sequences) / SECONDS_PER_HOUR  # (N,)

    def predict_interval_log10_seconds(self, sequences, space="target"):  # noqa: D417
        """Return the published conformal interval of the artifact, in ``log10`` seconds.

        The calibration bundle stores one symmetric half-width for the space in
        which the artifact was trained, that is ``log1p(hours)`` for the ``log``
        artifacts and hours for the others.

        :param sequences: Iterable of peptide sequences.
        :param str space: ``target`` applies the half-width in the training
            target space and then inverts the transform, which is the space the
            calibration was fitted in. ``output`` reproduces the published
            inference script, which adds the half-width to the already
            back-transformed prediction in hours.
        :return: Bounds in ``log10`` seconds of shape ``(N, 2)``, or None when
            the artifact ships no calibration bundle.
        :rtype: numpy.ndarray | None
        :raises ValueError: If the requested space is unknown.
        """
        if self.conformal is None:
            return None
        space = space or "target"
        if space not in {"target", "output"}:
            raise ValueError("space must be 'target' or 'output'")

        items = [str(sequence).strip() for sequence in sequences]
        width = float(self.conformal["quantile"])
        centre_hours = self.predict_hours(items)  # (N,)

        # Bound construction in the requested space
        if space == "target" and self.variant.target == "log1p":
            transformed = np.log1p(centre_hours)  # (N,)
            lower_hours = np.expm1(transformed - width)  # (N,)
            upper_hours = np.expm1(transformed + width)  # (N,)
        else:
            lower_hours = centre_hours - width  # (N,)
            upper_hours = centre_hours + width  # (N,)

        bounds = np.column_stack(
            [
                np.clip(lower_hours, MINIMUM_HOURS, None),
                np.clip(upper_hours, MINIMUM_HOURS, None),
            ]
        )  # (N, 2)
        return np.log10(bounds * SECONDS_PER_HOUR)  # (N, 2)

    def per_residue_gradient(self, sequence):
        """Return the gradient magnitude of the prediction per content token.

        The gradient is taken with respect to the backbone token embeddings,
        which is the input the regression head actually consumes.

        :param str sequence: Peptide sequence.
        :return: Gradient L2 norm per content token, shape ``(L,)``.
        :rtype: numpy.ndarray
        :raises NotImplementedError: If the artifact is tree-based.
        """
        if self.variant.estimator != "transformer":
            raise NotImplementedError(f"{self.name} is tree-based and has no input gradient")

        items = self._model_inputs([sequence])
        embeddings, mask = self.backbone.embed_tokens(items)  # (1, L, D), (1, L)
        embeddings = embeddings.detach().clone().requires_grad_(True)
        prediction = self.head(embeddings, mask)  # (1,)
        prediction.backward()
        gradient = embeddings.grad[0][mask[0]].norm(dim=-1)  # (L,)
        return gradient.detach().cpu().numpy()

    def unload(self):
        """Release the estimator and free device memory.

        :rtype: None
        """
        self.head = None
        self.booster = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def available_variants(root=PEPTIVERSE_DIR):
    """List the published variants present in the local snapshot.

    :param pathlib.Path root: Downloaded PeptiVerse snapshot.
    :return: Variant keys whose artifact directory exists.
    :rtype: list[str]
    """
    base = root / "training_classifiers/half_life"
    return [
        key for key, variant in VARIANTS.items() if (base / variant.folder).is_dir()
    ]


def verify_batch_equivalence(predictor, sequences, tolerance=1e-4):
    """Check the batched forward path against single-peptide calls.

    The published implementation scores one peptide per call. Batching changes
    padding and attention masking, so equivalence must be demonstrated rather
    than assumed.

    :param PeptiVersePredictor predictor: Loaded adapter.
    :param list[str] sequences: Peptides to compare, at least two.
    :param float tolerance: Allowed absolute deviation in ``log10`` seconds.
    :return: Per-peptide batched and single predictions with their deviation.
    :rtype: dict
    :raises ValueError: If fewer than two peptides are supplied.
    """
    if len(sequences) < 2:
        raise ValueError("Batch equivalence needs at least two peptides")

    batched = predictor.predict_log10_seconds(sequences)  # (N,)
    single = np.concatenate(
        [predictor.predict_log10_seconds([sequence]) for sequence in sequences]
    )  # (N,)
    deviation = np.abs(batched - single)  # (N,)
    return {
        "predictor": predictor.name,
        "peptides": list(sequences),
        "batched_log10_seconds": batched.tolist(),
        "single_log10_seconds": single.tolist(),
        "max_absolute_deviation": float(deviation.max()),
        "within_tolerance": bool(deviation.max() <= tolerance),
        "tolerance": tolerance,
    }
