"""Content-based code provenance for replay across working-tree changes."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


def source_digest() -> str:
    """Fingerprint Python sources; cache hashing while file metadata is unchanged.

    :return: Result of this operation.
    :rtype: str
    """
    root = Path(__file__).resolve().parents[2]
    signature = tuple(
        (str(path.relative_to(root)), path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(root.rglob("*.py"))
    )
    return _digest(str(root), signature)


@lru_cache(maxsize=1)
def _digest(root: str, signature: tuple[tuple[str, int, int], ...]) -> str:
    """Digest.

    :param root: Root resource directory or root graph node, as declared by its type.
    :type root: str
    :param signature: Signature.
    :type signature: tuple[tuple[str, int, int], ...]
    :return: Result of this operation.
    :rtype: str
    """
    digest = hashlib.sha256()
    for relative, _, _ in signature:
        digest.update(relative.encode())
        digest.update((Path(root) / relative).read_bytes())
    return digest.hexdigest()
