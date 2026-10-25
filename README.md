# H&M Preference Studio

**Intention Analytics for fashion brands.** A brand or marketing team reads what customers prefer across **10 preference topics** (found by a topic model
from product photos, descriptions, category and price), then forms its own intention to act: which product lines to promote, add or reduce, and which
products to recommend to which customers. Built on a Three-Tower neural network and the full H&M purchase history.

## The four tabs

| Tab | What it gives a brand or marketing team |
|---|---|
| **1 · Preference hub** | Every key figure of the 10 topics with real product photos: customers, purchases, revenue, price index and tier, purchases and spend per customer, loyalty, supply-demand gap |
| **2 · Customers** | Who prefers what: age, frequency, spend and price level; buying frequency against spending |
| **3 · Products and topics** | One topic at a time: best and slowest sellers, supply and demand, product mix, and the brand intention (add, keep or reduce) |
| **4 · Recommendation studio** | Pick any product and a customer preference; the Three-Tower model recommends similar products with the Three-Tower and Two-Tower match probability, the exact share of each tower, sales, supply and demand, and the action to take |

Two secondary pages: **Data and model** (every Kaggle output behind the numbers) and **Model Governance** (architecture, method, data lineage).

Every number comes from the experiments of the thesis: the Kaggle notebook collects the outputs of Sections 1-14 into one `data_export.zip`, the app
reads them, and if a required file is missing it names the file and stops. See `docs/DATA_MAP.md`.

## Deploy

1. Run `kaggle/HM_KAGGLE_EXPORT_ALL.ipynb` on Kaggle. It writes `/kaggle/working/data_export.zip` (about 60-80 MB). The last cell (PART 5c) shows a
   **download link**: click it to save the zip on your computer.
2. Upload the zip to Google Drive (*Anyone with the link: Viewer*) and copy the **File ID**.
3. Push this folder to GitHub (the folder root is the repository root, `app.py` at the top).
4. Streamlit Community Cloud: New app, main file `app.py`; **Settings > Secrets**: `GDRIVE_FILE_ID = "<your file id>"`.
5. After uploading a new zip to Drive, **Reboot** the app (a browser refresh is not enough).

`requirements.txt` installs the CPU build of PyTorch (the default build is several gigabytes and may not fit on the free tier).
Locally: `pip install -r requirements.txt`, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, `streamlit run app.py`.

## Structure

```
app.py                         the four tabs
pages/                         Data and model, Model Governance
utils/
  hub.py customers.py products.py studio.py        one module per tab
  studio_scoring.py            Three-Tower / Two-Tower scoring, similar products, exact tower shares (ablation)
  catalog.py merch.py ui.py    product data and photos, add / keep / reduce rule, cards
  topicfacts.py                one table with every key figure per topic
  kaggle_data.py               reads every Kaggle output; missing-file guard
  population.py priority.py overview.py thesis_data.py  analysis blocks
  preference_labels.py         the 10 thesis topic names (short and full)
  data_loader.py models.py theme.py trends.py charts.py exemplars.py recommender.py
docs/                          DATA_MAP.md, PREFERENCE_NAMING.md
```
