"""Where posts actually go, and the one check that happens on the way out."""
from .base import Platform, DryRun, PublishError

__all__ = ["Platform", "DryRun", "PublishError"]
