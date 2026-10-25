"""
exemplars.py
================================================================================
Picks the most TYPICAL products of a preference for display (thumbnails on the
Command Center cards, "Representative products" grid on Segment Detail).

Ranking, strongest signal first:
  1. attr_match     - how fully the product carries the attributes that DEFINE the topic: the
                      categorical + TF-IDF features among the topic's top-20 LDA features
                      (Section 7-8, topic_top_features.csv), scored on the LDA input matrix.
  2. feature_match  - the same score over all 20 top features (adds the semantic components).
  3. dominant_prob  - LDA probability of the topic (K=10, article_intention_profiles.csv).
Exports made before v3 only have the intention_k probability columns, so that is the fallback.
Colour variants of the same garment (same prod_name) are shown once.
================================================================================
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd


def ranking_columns(articles: pd.DataFrame, k: int) -> list[str]:
    keys = [c for c in ("attr_match", "feature_match", "dominant_prob") if c in articles.columns]
    if not keys and f"intention_{k}" in articles.columns:
        keys = [f"intention_{k}"]
    return keys


def top_exemplars(articles: pd.DataFrame, k: int, n: int,
                  image_exists: Optional[Callable[[str], bool]] = None) -> pd.DataFrame:
    seg = articles[articles["dominant_intention"] == k]
    keys = ranking_columns(seg, k)
    if keys:
        seg = seg.assign(_primary=seg[keys[0]].round(6))
        seg = seg.sort_values(["_primary"] + keys[1:], ascending=False, kind="mergesort").drop(columns="_primary")
    if image_exists is not None:                       # keep only products that have a picture ...
        seg = seg[seg["article_id"].map(image_exists)]
    if "prod_name" in seg.columns:                     # ... then one variant per garment
        seg = seg.drop_duplicates("prod_name", keep="first")
    return seg.head(n)
