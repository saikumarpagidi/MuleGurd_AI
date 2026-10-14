# This module has been replaced by inference.py
# Kept as a redirect for any lingering imports during transition.
from .inference import score_transaction  # noqa: F401

__all__ = ["score_transaction"]
