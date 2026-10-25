"""
ui.py — small shared building blocks for the four tabs (product cards, chips, key tiles).
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from utils.data_loader import image_path
from utils.catalog import price_index
from utils.merch import action, action_chip, seller_label
from utils.preference_labels import preference_name, preference_full_name
from utils.theme import intention_color
from utils.thesis_data import segment_static


def chip(text: str, colour: str = "#3B3873", fg: str = "#fff") -> str:
    return (f"<span style='background:{colour}; color:{fg}; padding:2px 9px; border-radius:11px; font-size:0.78rem; "
            f"font-weight:600; white-space:nowrap;'>{text}</span>")


def tile(label: str, value: str, note: str = "") -> str:
    """A key figure: small grey label, big value, small note."""
    return (f"<div style='padding:6px 4px;'><div style='font-size:0.78rem; color:#8A8578;'>{label}</div>"
            f"<div style='font-size:1.55rem; font-weight:700; line-height:1.15;'>{value}</div>"
            f"<div style='font-size:0.78rem; color:#8A8578;'>{note}</div></div>")


def topic_title(k: int, subtitle: str = "") -> str:
    c = intention_color(k)
    sub = f"<div style='color:#8A8578; font-size:0.9rem;'>{subtitle}</div>" if subtitle else ""
    return (f"<div style='border-left:6px solid {c}; padding:2px 0 2px 12px; margin:14px 0 6px 0;'>"
            f"<span style='font-size:1.35rem; font-weight:700;'>T{k} · {preference_full_name(k)}</span>{sub}</div>")


def product_cards(df: pd.DataFrame, cols: int = 5, show_action: bool = True, extra=None):
    """A grid of product cards. extra(row) -> optional HTML line added under the standard ones."""
    rows = [r for _, r in df.iterrows()]
    for start in range(0, len(rows), cols):
        grid = st.columns(cols)
        for c, row in zip(grid, rows[start:start + cols]):
            with c:
                p = image_path(row["article_id"])
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
                k = int(row["topic"])
                pi = price_index(row.get("avg_price"))
                lines = [f"<div style='font-weight:600; font-size:0.9rem; line-height:1.2;'>{str(row['prod_name'])[:34]}</div>",
                         f"<div style='color:#8A8578; font-size:0.78rem;'>{str(row.get('product_type_name', ''))} · {str(row.get('colour_group_name', ''))}</div>",
                         f"<div style='margin:3px 0;'>{chip(f'T{k} {preference_name(k)}', intention_color(k))}</div>"]
                if pi == pi:
                    lines.append(f"<div style='font-size:0.82rem;'>Price index <b>{pi:.0f}</b></div>")
                if row.get("units") == row.get("units"):
                    lines.append(f"<div style='font-size:0.82rem;'><b>{int(row['units']):,}</b> sold · {int(row.get('units_last_90d', 0)):,} in the last 90 days</div>"
                                 f"<div style='font-size:0.78rem; color:#8A8578;'>{seller_label(row.get('units_rank_pct'))}</div>")
                if show_action and row.get("units") == row.get("units"):
                    lab, _why, col = action(segment_static(k)["gap"], row.get("units_rank_pct"))
                    lines.append(f"<div style='margin-top:4px;'>{action_chip(lab, col)}</div>")
                if extra is not None:
                    lines.append(extra(row))
                st.markdown("".join(lines), unsafe_allow_html=True)
