"""AET Neural Wave 2026: reconstruct quarter-hourly price shapes from hourly curves.

Flow: hourly curve -> mean-preserving baseline -> learned zero-sum residual
-> fallback chain (learned -> baseline -> flat) -> exact hourly-mean projection.
"""

__version__ = "0.1.0"
