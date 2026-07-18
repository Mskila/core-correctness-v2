"""Rejected legacy island training entrypoint under V2 semantics."""

from model_core.semantics import ArtifactCompatibilityError


def main() -> None:
    raise ArtifactCompatibilityError(
        "V2 supports only single-symbol training; use train_file.py, Web, "
        "or train_single.py"
    )


if __name__ == "__main__":
    main()
