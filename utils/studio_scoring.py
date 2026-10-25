"""
studio_scoring.py
================================================================================
Scoring for the recommendation studio, with the real Three-Tower and Two-Tower models.

For a chosen product and a customer preference profile it returns, for the product and for products similar to it:
  - the Three-Tower and Two-Tower match probabilities
  - the exact contribution of each tower to the Three-Tower score, by ablation: the score is recomputed with that
    tower's input replaced by a neutral one (Tower 1: the average product image; Tower 2: the average product text and
    average customer details; Tower 3: the average customer's preference profile) and the drop is measured.
    Shares are the absolute drops as a percentage of their sum. This is the model's own arithmetic, not an estimate.

Probabilities come from a model trained with one non-purchase per purchase, so 50% means "no signal" and the numbers are best
read as match scores for ranking, not as real-world purchase probabilities.
================================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import torch

INT_COLS = [f"intention_{k}" for k in range(10)]


def typical_customer(k: int, validation: pd.DataFrame):
    """Average preference profile of the demo customers whose dominant topic is k (real customers)."""
    seg = validation[validation["dominant_intention"] == k]
    if seg.empty:
        return None
    vec = seg[INT_COLS].mean().values.astype(np.float64)
    vec = (vec / vec.sum()).astype(np.float32)
    demo = np.array([seg["age"].mean(), seg["FN"].mean(), seg["Active"].mean()], dtype=np.float32)
    return {"vec": vec, "demo": demo, "confidence": float(seg["confidence"].mean()), "n": int(len(seg)),
            "label": f"Typical customer of this topic ({len(seg)} real customers averaged)"}


def one_customer(row: pd.Series):
    vec = row[INT_COLS].astype(float).values.astype(np.float32)
    demo = np.array([row["age"], row["FN"], row["Active"]], dtype=np.float32)
    demo = np.where(np.isnan(demo), np.array([30.0, 0.0, 0.0], dtype=np.float32), demo)      # missing age / flags -> the app-wide defaults
    return {"vec": vec, "demo": demo, "confidence": float(row["confidence"]), "n": 1, "label": str(row["persona_label"])}


def neutral_inputs(visual, semantic, validation, global_prior: dict):
    prior = np.array([global_prior.get(c, 0.1) for c in INT_COLS], dtype=np.float32)
    prior = prior / prior.sum()
    demo = np.array([validation["age"].mean(), validation["FN"].mean(), validation["Active"].mean()], dtype=np.float32)
    return visual.mean(axis=0), semantic.mean(axis=0), demo, prior


def similar_ids(selected_id: str, pool: pd.DataFrame, visual, semantic, art_feat_idx, n: int):
    """Products most similar to the selected one: mean cosine similarity of image features and text features."""
    i = art_feat_idx[selected_id]
    idx = np.array([art_feat_idx[a] for a in pool["article_id"]])

    def cos(M, v):
        return (M[idx] @ v) / (np.linalg.norm(M[idx], axis=1) * np.linalg.norm(v) + 1e-9)

    sim = 0.5 * cos(visual, visual[i]) + 0.5 * cos(semantic, semantic[i])
    out = pool.assign(similarity=sim)
    return out.sort_values("similarity", ascending=False).head(n)


def _batch(three, two, vis, sem, demo, iint, uint):
    with torch.no_grad():
        p3, align, _ = three(torch.from_numpy(vis), torch.from_numpy(sem), torch.from_numpy(demo), torch.from_numpy(iint), torch.from_numpy(uint))
        p2 = two(torch.from_numpy(vis), torch.from_numpy(sem), torch.from_numpy(demo), torch.from_numpy(iint), torch.from_numpy(uint))
    return p3.numpy().astype(np.float64), p2.numpy().astype(np.float64), align.numpy()


def score_items(three, two, items: pd.DataFrame, visual, semantic, art_feat_idx, customer: dict, neutral) -> pd.DataFrame:
    """items needs article_id and intention_0..9. Returns items + three_p, two_p and tower contributions."""
    n = len(items)
    fi = np.array([art_feat_idx[a] for a in items["article_id"]])
    vis, sem = visual[fi].astype(np.float32), semantic[fi].astype(np.float32)
    iint = items[INT_COLS].values.astype(np.float32)
    demo = np.tile(customer["demo"], (n, 1)).astype(np.float32)
    uint = np.tile(customer["vec"], (n, 1)).astype(np.float32)
    vis_n, sem_n, demo_n, prior = neutral

    p3, p2, align = _batch(three, two, vis, sem, demo, iint, uint)
    q1, _, _ = _batch(three, two, np.tile(vis_n, (n, 1)).astype(np.float32), sem, demo, iint, uint)                                  # Tower 1 neutral
    q2, _, _ = _batch(three, two, vis, np.tile(sem_n, (n, 1)).astype(np.float32), np.tile(demo_n, (n, 1)).astype(np.float32), iint, uint)  # Tower 2 neutral
    q3, _, _ = _batch(three, two, vis, sem, demo, iint, np.tile(prior, (n, 1)).astype(np.float32))                                   # Tower 3 neutral
    d = np.stack([p3 - q1, p3 - q2, p3 - q3], axis=1)
    tot = np.abs(d).sum(axis=1)
    tot[tot == 0] = 1e-12
    share = np.abs(d) / tot[:, None] * 100

    out = items.copy().reset_index(drop=True)
    out["three_p"], out["two_p"] = p3, p2
    out["uplift_pp"] = (p3 - p2) * 100
    for j, nm in enumerate(["visual", "text", "preference"]):
        out[f"drop_{nm}"] = d[:, j] * 100               # percentage points of probability the tower adds
        out[f"share_{nm}"] = share[:, j]                # % of the three towers' total effect
    out["top_aligned_topic"] = np.argmax(align, axis=1)
    return out
