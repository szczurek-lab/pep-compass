from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Self


def set_seed(seed: int) -> None:
    """Seed all supported generators through the backend RNG contract.

    :param seed: Non-negative integer seed.

    :type seed: int
    :rtype: None
    """
    from pep_compass.utils.randomness import seed_generators

    seed_generators(seed)


class Timer:
    def __init__(self) -> None:
        """Initialize this object with the declared configuration.

        :rtype: None
        """
        self.times = {}
        self._name = None
        self._start = None

    def __call__(self, name: str) -> Self:
        """Allow usage like: with timer("task"): ...

        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :return: Result of this operation.
        :rtype: Self
        """
        self._name = name
        return self

    def __enter__(self) -> Self:
        """Enter the managed resource scope.

        :return: Result of this operation.
        :rtype: Self
        """
        self._start = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize the managed scope and propagate exceptions.

        :param exc_type: Original exception type, if present.
        :type exc_type: type[BaseException] | None
        :param exc_value: Original exception instance, if present.
        :type exc_value: BaseException | None
        :param traceback: Original exception traceback, if present.
        :type traceback: TracebackType | None
        :rtype: None
        """
        elapsed = time.time() - self._start
        self.times[self._name] = elapsed

    def __str__(self) -> str:
        """Str.

        :return: Result of this operation.
        :rtype: str
        """
        return (
            "{\n\t"
            + "\n\t".join(
                f"{name}: {elapsed:.4f}s" for name, elapsed in self.times.items()
            )
            + "\n}"
        )

    def reset(self) -> None:
        """Reset.

        :rtype: None
        """
        self.times = {}
        self._name = None
        self._start = None
