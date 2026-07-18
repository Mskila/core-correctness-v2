"""Fail-closed compatibility surface for removed multi-island training."""

from .semantics import ArtifactCompatibilityError


class IslandAlphaEngine:
    """Reject the identity-incompatible legacy multi-island training path."""

    def __init__(
        self,
        data_manager,
        n_islands: int | None = None,
        migration_interval: int | None = None,
        migration_top_k: int | None = None,
    ):
        raise ArtifactCompatibilityError(
            "V2 supports only single-symbol training; use train_file.py, Web, "
            "or train_single.py"
        )
