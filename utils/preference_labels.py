"""
preference_labels.py
================================================================================
One place for the names shown to users.

The 10 preference topics are numbered T0-T9 (the numbering comes from the topic
model and never changes). The names below are plain product-line names, so a
business reader sees "Children's Upper Body" or "Menswear Shirts", not a
technical code. They are the product-group part of the thesis topic labels
(Table 4.3); the full thesis names are built with PREFERENCE_MOTIVE. Edit them here and every page follows. The description under each name is NOT typed here: see product_line().
================================================================================
"""
from __future__ import annotations

PREFERENCE_NAMES = {
    0: "Ladieswear Full Body",
    1: "Ladieswear Upper Body",
    2: "Unisex Dark Basics",
    3: "Baby Full Body",
    4: "Unisex Lower Body",
    5: "Children's Upper Body",
    6: "Ladies Accessories & Footwear",
    7: "Ladieswear Underwear",
    8: "Ladieswear Knitwear",
    9: "Menswear Shirts",
}

# The second half of each topic name, exactly as written in the thesis (Table 4.3, 4.7, 5.1, 5.2).
PREFERENCE_MOTIVE = {
    0: "Special Occasion Dressing",
    1: "Everyday Workwear Comfort",
    2: "Utilitarian Necessity Purchase",
    3: "Infant & Nurturing Care",
    4: "Functional Versatility Seeking",
    5: "Trendy & Casual Provisioning",
    6: "Hedonic Purchase",
    7: "Intimate Self-Care",
    8: "Premium Quality Investment",
    9: "Professional Identity Expression",
}


def preference_name(k: int) -> str:
    """Short name (product line), used in charts and chips."""
    return PREFERENCE_NAMES.get(int(k), f"Topic {int(k)}")


def preference_full_name(k: int) -> str:
    """The full thesis name, e.g. 'Ladieswear Full Body: Special Occasion Dressing'."""
    k = int(k)
    return f"{PREFERENCE_NAMES[k]}: {PREFERENCE_MOTIVE[k]}" if k in PREFERENCE_NAMES else f"Topic {k}"


def product_line(k: int, n: int = 3) -> str:
    """The topic's defining product attributes, read from the Kaggle files (topic_key_attributes.csv:
    the top features of the topic from Section 7-8, decoded to names with articles_cleaned.csv).
    Returns "" when that file is absent; nothing is typed in here."""
    from utils import kaggle_data as kd          # imported here: kaggle_data itself imports data_loader
    ka = kd.key_attributes()
    if ka.empty:
        return ""
    seen, out = set(), []
    for a in ka[ka["topic"] == int(k)].sort_values("rank")["attribute"].astype(str):
        key = a.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(a.strip())
        if len(out) == n:
            break
    return " · ".join(out)


def apply_preference_names(labels: dict) -> dict:
    """Returns a copy of the label dict (from the data export) whose 'name' is the
    plain product-line name; the export's original text is kept as 'export_name'."""
    out = {}
    for key, val in labels.items():
        val = dict(val)
        if str(key).isdigit():
            val["export_name"] = val.get("name")
            val["name"] = preference_name(int(key))
        out[key] = val
    return out
