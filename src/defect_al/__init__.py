"""Reproducibility safeguards for future defect active-learning runs.

The package deliberately does not rewrite historical manifests or frozen
evidence.  New experiments can opt in to the deterministic ``cid:v1`` identity
and the strict frozen-run configuration contract exposed here.
"""

from .config import FrozenRunConfig, ManifestReference, load_frozen_run_config

__all__ = [
    "FrozenRunConfig",
    "ManifestReference",
    "load_frozen_run_config",
]

