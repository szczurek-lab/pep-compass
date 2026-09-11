"""Tests for explicit autoencoder method and model registration."""

import pep_compass.autoencoder.strategies  # noqa: F401
import pytest

from pep_compass.autoencoder.factory import AutoencoderFactory
from pep_compass.autoencoder.registry import AutoencoderRegistry


def test_hydramp_method_and_named_model_are_registered_separately() -> None:
    """Implementation selection and weight/model selection must be independent."""
    assert AutoencoderRegistry.methods() == ("hydramp",)
    assert AutoencoderRegistry.models("hydramp") == ("article_25",)
    assert callable(AutoencoderRegistry.method("hydramp"))
    assert AutoencoderRegistry.model("hydramp", "article_25").parameters == {
        "model_name": "article_25"
    }


def test_hydramp_article_weights_load_and_execute() -> None:
    """The bundled model must load both weight files and encode a peptide."""
    autoencoder = AutoencoderFactory.build(
        "hydramp",
        model="article_25",
        device="cpu",
        jacobian_mode="approx",
        jacobian_eps=0.05,
        field_eps=0.05,
    )

    latent = autoencoder.encode_peptides(["FLYKWWIRIGRLKL"])

    assert latent.shape == (1, autoencoder.latent_dim)
    assert latent.isfinite().all()


def test_hydramp_rejects_missing_weight_directory() -> None:
    """A registered implementation must fail before use when weights are absent."""
    from pep_compass.autoencoder.strategies.hydramp.adapter import HydrampAutoencoder

    with pytest.raises(FileNotFoundError, match="weights directory does not exist"):
        HydrampAutoencoder(
            model_name="missing_test_model",
            device="cpu",
            jacobian_mode="approx",
            jacobian_eps=0.05,
            field_eps=0.05,
        )
