
from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from poli.core.abstract_black_box import AbstractBlackBox, BlackBoxInformation


class NegativeBlackBox(AbstractBlackBox):
    """A wrapper for a black box that negates the objective function.

    If you construct a black-box function f for maximizing, then -f is
    a black-box function for minimizing. This class is a wrapper for
    implementing the latter.

    The only difference is that the __call__ method returns -f(x) instead
    of f(x). The _black_box method is the same as the original black box.
    """

    def __init__(self, f: AbstractBlackBox):
        self.f = f
        super().__init__(
            batch_size=f.batch_size,
            parallelize=f.parallelize,
            num_workers=f.num_workers,
            evaluation_budget=cast(int | None, f.evaluation_budget),
        )

    def __call__(self, x: NDArray[np.str_], context=None):
        return -self.f.__call__(x, context)

    def _black_box(self, x: NDArray[np.str_], context=None):
        return self.f._black_box(x, context)

    def __str__(self) -> str:
        return f"NegativeBlackBox({self.f})"

    def __repr__(self) -> str:
        return f"<NegativeBlackBox({self.f})>"
    
    def get_black_box_info(self) -> BlackBoxInformation:
        return self.f.get_black_box_info()