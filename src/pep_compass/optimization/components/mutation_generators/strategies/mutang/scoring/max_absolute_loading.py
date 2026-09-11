"""Baseline MUTANG score matrix."""


class MaxAbsoluteLoading:
    """Return ``max_j abs(U[(l,a),j])`` for selected directions."""

    def __init__(self, max_len=25, alphabet_size=21):
        self.max_len, self.alphabet_size = max_len, alphabet_size

    def score(self, left_vectors, selected):
        table = left_vectors[:, selected.indices].abs()
        return table.reshape(self.max_len, self.alphabet_size, -1).amax(dim=-1)
