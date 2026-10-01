"""When is each feature's information actually known?

early   : known early in the week (only completed games + schedule facts)
gameday : needs same-week news (injury reports, announced QB) or game-day weather
Using a 'gameday' feature in a test against early-week lines leaks news the
early line could not have priced in (found in the early-line test).
"""

from .team_features import MODEL_FEATURES_BASE
from .qb import QB_FEATURES
from .injuries import INJURY_FEATURES

AVAILABILITY = {f: "early" for f in MODEL_FEATURES_BASE}
AVAILABILITY.update({f: "gameday" for f in QB_FEATURES + INJURY_FEATURES})
AVAILABILITY.update({f"{s}_{m}_{d}": "early" for s in "ha" for m in ("pts", "yds", "epa", "giveaways")
                     for d in ("for", "against")})
AVAILABILITY.update({
    "qb_changed_h": "gameday", "qb_changed_a": "gameday", "qb_swap": "gameday",
    "spread_line": "market", "total_line": "market",
    "dome": "early",            # stadium/roof type is a schedule fact
    "wind_eff": "gameday", "temp_eff": "gameday",
})


def usable_at(features, asof="early"):
    """Drop features whose information is not yet known at `asof` ('early' or 'gameday')."""
    ok = {"early": {"early", "market"}, "gameday": {"early", "market", "gameday"}}[asof]
    missing = [f for f in features if f not in AVAILABILITY]
    if missing:
        raise KeyError(f"untagged features (add to AVAILABILITY): {missing}")
    return [f for f in features if AVAILABILITY[f] in ok]
