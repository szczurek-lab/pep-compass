from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def to_one_hot(x: float, pad: int = 26) -> list[object]:
    """To one hot.

    :param x: X.
    :type x: float
    :param pad: Pad.
    :type pad: int
    :return: Result of this operation.
    :rtype: list[object]
    """
    alphabet = list(" ACDEFGHIKLMNPQRSTVWY")
    classes = range(0, 21)
    aa_encoding = dict(zip(alphabet, classes))
    if len(x) > 25:
        x = x[:25]
    else:
        x = x + "".join([" " * (25 - len(x))])
    return [aa_encoding[aa] for aa in x]


def translate_generated_peptide(decoded_peptide: torch.Tensor) -> str:
    # Expects a tensor of shape (1, 25, 21)
    """Translate generated peptide.

    :param decoded_peptide: Decoded peptide.
    :type decoded_peptide: torch.Tensor
    :return: Result of this operation.
    :rtype: str
    """
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")

    # Get argmax indices
    indices = decoded_peptide[0].argmax(axis=1)

    # Build peptide sequence, handling empty positions
    peptide_chars = []
    for el in indices:
        if el == 0:  # Space/padding character
            break  # Stop at first padding (natural peptide end)
        else:
            peptide_chars.append(alphabet[el - 1])

    result = "".join(peptide_chars)

    # Ensure we always return at least one character to prevent completely empty peptides
    if len(result) == 0:
        # Fallback: return the most probable non-space character from first position
        first_pos_probs = decoded_peptide[0][0]  # First position probabilities
        non_space_probs = first_pos_probs[1:]  # Exclude space (index 0)
        if len(non_space_probs) > 0:
            best_aa_idx = (
                non_space_probs.argmax().item() + 1
            )  # +1 because we excluded index 0
            result = alphabet[best_aa_idx - 1]
        else:
            result = "A"  # Ultimate fallback

    return result
