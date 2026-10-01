"""Odds conversions. American odds in, probabilities / payouts out. Works on scalars or pandas/numpy arrays."""
from __future__ import annotations

import numpy as np
import pandas as pd


def american_to_decimal(ml):
    """+150 -> 2.50 ; -200 -> 1.50  (decimal odds = total return per 1 staked)."""
    if np.isscalar(ml):
        return 1 + ml / 100.0 if ml > 0 else 1 + 100.0 / -ml
    s = pd.to_numeric(pd.Series(ml), errors="coerce")
    d = pd.Series(np.where(s > 0, 1 + s / 100.0, 1 + 100.0 / -s), index=s.index).where(s.notna())
    return d if isinstance(ml, pd.Series) else d.to_numpy()


def implied_prob(ml):
    """Break-even win probability of a price, INCLUDING the bookmaker's vig."""
    if np.isscalar(ml):
        return -ml / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)
    s = pd.to_numeric(pd.Series(ml), errors="coerce")
    p = np.where(s < 0, -s / (-s + 100.0), 100.0 / (s + 100.0))
    out = pd.Series(p, index=s.index).where(s.notna())
    return out if isinstance(ml, pd.Series) else out.to_numpy()


def devig_two_way(ml_a, ml_b):
    """Remove the vig from a two-outcome market (proportional method).

    Returns (fair_prob_a, fair_prob_b). Proportional devig slightly over-states
    longshots; good enough for moneylines, revisit (power/Shin) if it matters."""
    pa, pb = implied_prob(ml_a), implied_prob(ml_b)
    total = pa + pb
    return pa / total, pb / total


def overround(ml_a, ml_b):
    """Bookmaker margin: implied probabilities sum to 1 + overround."""
    return implied_prob(ml_a) + implied_prob(ml_b) - 1.0


def prob_to_american(p):
    p = np.asarray(p, dtype=float)
    return np.where(p >= 0.5, -100.0 * p / (1 - p), 100.0 * (1 - p) / p)


def profit_per_unit(ml):
    """Profit on a winning 1-unit bet (losing bet = -1)."""
    return american_to_decimal(ml) - 1.0
