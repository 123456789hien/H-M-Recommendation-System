# Which file feeds what

Everything below is read at run time from `data/` (unzipped from `data_export.zip`). Column names are the ones
written by the Kaggle notebooks. "App name" differs from the Kaggle name only where the Kaggle name contained
the word *intention* (PART 3 of `HM_KAGGLE_EXPORT_ALL.ipynb` renames while copying).

## `data/kaggle_outputs/`  (collected by PART 3 of `HM_KAGGLE_EXPORT_ALL.ipynb`)

| App name (Kaggle name if different) | Section | Used for |
|---|---|---|
| `dataset_overview.json` | PART 2 | Products, customers, transactions, period, price quartiles (counted from the cleaned files of Sections 1-3) |
| `lda_feature_summary.csv` | 6 | Features the topic model reads per product |
| `elbow_results.csv` | 7 | Perplexity per K (choice of 10 topics) |
| `lda_model_metadata.csv` | 8 | K, perplexity, ANOVA / Kruskal / η² |
| `topic_summary.csv` (`intention_summary.csv`) | 8 | Products per topic, catalogue share, mean price, topic concentration (`mean_dominant_prob`) |
| `topic_top_features.csv` (`intention_top_features.csv`) | 8 | The 20 strongest features of each topic |
| `topic_validation.csv` (`intention_validation.csv`) | 9 | ANOVA, Kruskal-Wallis, η², χ² of the supply-demand gap |
| `topic_price_stats.csv` (`intention_price_stats.csv`) | 9 | Price level and price tier per topic |
| `topic_gap_analysis.csv` (`intention_gap_analysis.csv`) | 9 | Catalogue share, purchase share (`user_share`), gap in percentage points |
| `topic_key_attributes.csv` | PART 3c | The topic's defining product attributes (its top features from Section 7-8 decoded to names with `articles_cleaned.csv`); the line under each topic name |
| `topic_correlation_matrix.csv` (`intention_correlation_matrix.csv`) | 9 | Kept for reference |
| `user_profiling_stats.json` | 10 | Customers, new customers, purchases per customer, price quartiles, recency decay, smoothing weights |
| `user_dominant_topic_dist.csv` (`user_dominant_intention_dist.csv`) | 10 | Customers per topic (fallback; includes new customers) |
| `sampling_statistics.csv` | 11 | Sample vs population per topic |
| `split_statistics.json` | 11 | Sampled customers, cut-off date, train/test rows, positives and negatives, warm/new test customers, feature dimensions |
| `split_visual_summary.csv` | 11 | Kept for reference |
| `three_tower_metrics.json`, `two_tower_metrics.json` | 12, 13 | Parameters, best epoch, training time, training-run test AUC |
| `three_tower_training_log.csv`, `two_tower_training_log.csv` | 12, 13 | Learning curves (`val_auc` per `epoch`) |
| `evaluation_metrics.json` | 14 | Final AUC, accuracy, precision, recall, F1, NDCG@10, MAP, t-test, Wilcoxon |
| `per_topic_metrics.csv` (`per_intention_metrics.csv`) | 14 | AUC per topic for both models; the AUC gain used across the app |
| `evaluation_extras.json`, `roc_curve_points.csv`, `confusion_matrices.json`, `cold_start_breakdown.csv` | PART 4 (optional) | Per-customer tests and Cohen's d, McNemar, ROC curve, confusion-rate difference, new-customer results |

## `data/population/`  (PART 2)
`population_summary.json`, `pop_segments.csv`, `pop_price_tier.csv`, `pop_frequency.csv`, `pop_age_groups.csv`,
`pop_age_topic.csv`, `pop_membership.csv`: customers and purchases per topic, price tiers, purchase frequency
(10+ purchases), age and membership. Aggregates only, no customer IDs.

## `data/`  (PART 1)
`articles_sample.csv`: the products of the prototype (about 5.7 thousand): a random 5% per topic (`sample_type = stratified`), the 24 most typical
products of each topic (`exemplar`, chosen by `attr_match` / `feature_match` against the topic's top-20 features) and the 30 best sellers of each topic
(`bestseller`, real units over the whole catalogue). Columns: product details, `avg_price`, `units`, `revenue`, `units_last_90d`, `units_rank_pct` (percentile among all
products of its topic, computed on the whole catalogue), `dominant_prob`, `attr_match`, `feature_match`, `sample_type`, `image_path`, `intention_0..9`.
`article_intention_profiles_sample.csv`, `article_index_sample.csv`, `visual_features_sample.npy` and `semantic_features_sample.npy` (float16, read back as float32),
`demo_personas.csv` and `customer_validation.csv` (same file: the preference profiles of 100 real customers, 10 per topic, used for the "typical customer" of each topic),
`monthly_segment_trends.csv` (revenue, units and active customers per topic per month: 100% of transactions), `intention_labels.json` (colours),
`sampling_report.json` (sampling counts and the average customer profile), `three_tower_best.pt`, `two_tower_best.pt`, `images/` (300 px, JPEG quality 70).

## Not data: display rules written in the code
The ±10pp lines on the gap chart, the 5pp thresholds of the status badge, "strong match = alignment ≥ 0.10",
the four "Suggested focus" labels (position against the median of the 10 topics on frequency and spend), and the
10 topic names in `utils/preference_labels.py`.


## Which tab reads what
| Tab | Main sources |
|---|---|
| 1 · Preference hub | `topic_gap_analysis.csv`, `topic_summary.csv`, `topic_price_stats.csv`, `population/pop_segments.csv`, `monthly_segment_trends.csv`, `per_topic_metrics.csv`, `articles_sample.csv` + `images/` |
| 2 · Customers | `population/*` (segments, price tiers, frequency, age groups, age by topic, membership), `monthly_segment_trends.csv` |
| 3 · Products and topics | `articles_sample.csv` (sales, product mix; mix on `sample_type = stratified` only), `images/`, `topic_key_attributes.csv`, `monthly_segment_trends.csv`, `topic_gap_analysis.csv` |
| 4 · Recommendation studio | `three_tower_best.pt`, `two_tower_best.pt`, `visual_features_sample.npy`, `semantic_features_sample.npy`, `customer_validation.csv` (preference profiles of 100 real customers), `sampling_report.json` (average customer profile), `articles_sample.csv`, `topic_gap_analysis.csv` |
