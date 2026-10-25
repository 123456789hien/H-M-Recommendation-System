# Naming in the app

The app follows the supervisor's wording and keeps the thesis title.

| Side | Word used | Where |
|---|---|---|
| **Customers** | **preference** (never intention, psychology, motivation) | topics, customer profiles, charts, tabs |
| **Brand / marketing team** | **intention** (what the brand intends to do) and **decision** | app subtitle, "Brand intention for this topic", "Supply, demand and the brand's intention" |

Thesis title (unchanged): *Deep Learning-Driven Business Intelligence for Personalized Fashion Retail: Integrating Intention Analytics and Recommendation System*.
"Intention Analytics" here means: the brand reads customer preference and forms its own intention to act (promote, add, reduce).

## The 10 topics
Names are exactly those of the thesis (Tables 4.3, 4.7, 5.1, 5.2): *product group: preference*. Headers show the full name; charts and chips
use the short product-group name so they fit. Both are set in `utils/preference_labels.py` (`PREFERENCE_NAMES`, `PREFERENCE_MOTIVE`).

| Topic | Full name |
|---|---|
| T0 | Ladieswear Full Body: Special Occasion Dressing |
| T1 | Ladieswear Upper Body: Everyday Workwear Comfort |
| T2 | Unisex Dark Basics: Utilitarian Necessity Purchase |
| T3 | Baby Full Body: Infant & Nurturing Care |
| T4 | Unisex Lower Body: Functional Versatility Seeking |
| T5 | Children's Upper Body: Trendy & Casual Provisioning |
| T6 | Ladies Accessories & Footwear: Hedonic Purchase |
| T7 | Ladieswear Underwear: Intimate Self-Care |
| T8 | Ladieswear Knitwear: Premium Quality Investment |
| T9 | Menswear Shirts: Professional Identity Expression |

The description line under each name is not typed in: it is the topic's strongest product features (Section 7-8), decoded to names
from the data (`topic_key_attributes.csv`). Column names in the data files (`dominant_intention`, `intention_k`) are technical and are never shown.
