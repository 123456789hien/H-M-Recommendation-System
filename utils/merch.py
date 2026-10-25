"""
merch.py
================================================================================
The add / keep / reduce rule used by the product tabs. It combines two real measures:

  supply-demand gap of the topic  (share of purchases minus share of the catalogue, Section 9; the
                                   same 5pp "monitor" and 10pp "action" thresholds the notebook uses)
  seller rank of the product      (percentile of its units among all products of its topic)

Seller-rank cut-offs (top 30% = sells well, bottom 30% = sells poorly) are a design choice, not data.
================================================================================
"""
from __future__ import annotations

WELL, POOR = 70.0, 30.0
GAP_ACT, GAP_WATCH = 10.0, 5.0


def seller_label(rank_pct) -> str:
    try:
        r = float(rank_pct)
    except (TypeError, ValueError):
        return "sales n/a"
    if r != r:
        return "sales n/a"
    top = max(1, round(100 - r))
    return f"top {top}% seller in its topic" if r >= WELL else ("bottom 30% seller in its topic" if r <= POOR else "mid-range seller in its topic")


def action(gap_pp: float, rank_pct) -> tuple[str, str, str]:
    """(label, explanation, colour). Colours: green add / blue keep / amber review / red reduce."""
    try:
        r = float(rank_pct)
    except (TypeError, ValueError):
        r = float("nan")
    known = r == r
    well, poor = known and r >= WELL, known and r <= POOR
    if gap_pp >= GAP_WATCH:
        if well or not known:
            return ("Add more like this", f"Demand exceeds the range by {gap_pp:+.1f}pp and this product sells well.", "#2E8B57")
        return ("Add, but pick stronger sellers", f"The topic is under-supplied ({gap_pp:+.1f}pp) but this product is not a strong seller.", "#C9932E")
    if gap_pp <= -GAP_WATCH:
        if poor:
            return ("Reduce / rationalise", f"The range exceeds demand by {abs(gap_pp):.1f}pp and this product sells poorly.", "#C23B5E")
        return ("Keep, do not add", f"The topic is over-supplied ({gap_pp:+.1f}pp); keep proven sellers but add no more.", "#3B5FA8")
    if well:
        return ("Maintain & promote", "Supply roughly matches demand and the product sells well.", "#2E8B57")
    if poor:
        return ("Review", "Supply matches demand but this product is a weak seller for its topic.", "#C9932E")
    return ("Maintain", "Supply roughly matches demand; nothing to change.", "#3B5FA8")


def action_chip(label: str, colour: str) -> str:
    return (f"<span style='background:{colour}; color:#fff; padding:2px 9px; border-radius:11px; "
            f"font-size:0.78rem; font-weight:600; white-space:nowrap;'>{label}</span>")
