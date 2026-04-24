"""Mathematical helper functions.

Small, stateless utilities used across the feature extraction pipeline.
"""

from __future__ import annotations


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Divide safely, returning *default* when *denominator* is zero.

    Parameters
    ----------
    numerator:
        Dividend.
    denominator:
        Divisor.  When exactly ``0`` the function returns *default*
        instead of raising ``ZeroDivisionError``.
    default:
        Value returned when division by zero would occur.
    """
    return numerator / denominator if denominator != 0 else default
