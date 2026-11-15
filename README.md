# Global Superstore Analytics Dashboard

An interactive exploratory data visualization and machine learning dashboard built with Python, Streamlit, and Plotly for the **CEN445 Introduction to Data Visualization** course.

## Project Overview
The dashboard investigates one business question: **"Where does the money come from, and where does it leak?"** Through a three-part narrative and a machine learning finale, it analyzes sales growth, geographic profitability, the margin-eroding effect of discounting, loss-making pockets, delivery performance, customer behavior, and order-level loss prediction.

## Dataset
- **Source:** [Global Superstore Dataset (Kaggle)] — the file is included in this repository as `data/superstore.csv`.
- **Raw size:** 51,290 orders × 27 columns (2011–2014, 147 countries, 7 markets).
- **After cleaning:** 51,290 orders × 28 columns.
- **Column types:** numerical (sales, profit, discount, quantity, shipping cost), categorical (category, sub-category, market, segment, ship mode, order priority), geographic (country, state, city) and temporal (order date, ship date).

### Preprocessing (`src/data_loader.py` and `notebooks/eda.ipynb`)
- Standardized column names (`Order.Date` → `order_date`) and parsed dates.
- Dropped redundant or non-informative columns: `row_id` (row index), `记录数` (constant record count), `weeknum` (derivable from the order date) and `market2` (a coarser duplicate of `market` that merges US and Canada).
- Checked for missing values, duplicate rows and negative delivery times (a data quality check is shown in the dashboard's *Data Quality Report*).
- Standardized country names (`COUNTRY_FIX`) so the choropleth map matches every country.
- **Outlier detection:** orders with extreme profit are flagged using the 1.5 × IQR rule (9,755 orders). Users can exclude them with a checkbox.
- **Engineered features:** `delivery_days`, `order_month`, `discount_group` (0%, 1–20%, 21–40%, 41%+), `profit_outlier` and the binary target `is_loss`.

## Dashboard Structure (10 Visualizations + 2 ML Models)

| # | Visualization | Part |
|---|---|---|
| 01 | Line Chart — monthly sales and profit trend | Part 1: Growth & Footprint |
| 02 | Choropleth Map — country performance | Part 1 |
| 03 | Treemap — category / sub-category profit margin | Part 1 |
| 04 | Bar Chart — discount brackets vs. profit margin | Part 2: The Discount Effect |
| 05 | Sankey Diagram — category → discount → profit/loss flow | Part 2 |
| 06 | Animated Bubble Chart — discount vs. margin by year | Part 2 |
| 07 | Parallel Coordinates — multi-dimensional order profile | Part 2 |
| 08 | Annotated Heatmap — sub-category × market margin matrix | Part 3: Pinpointing the Leak |
| 09 | Sunburst — loss pockets in dollars (market → category → sub-category) | Part 3 |
| 10 | Box Plot — delivery time by ship mode and order priority | Part 3 |
| ML 01 | K-Means customer segmentation (PCA scatter + cluster profiles) | Finale |
| ML 02 | Random Forest loss classifier (feature importance, confusion matrix, what-if simulator) | Finale |

Advanced chart types used: choropleth map, treemap, Sankey, animated bubble chart, parallel coordinates, heatmap and sunburst.

### Interactivity
- **Global filters:** year range slider, market / category / discount-group multiselects, IQR-outlier checkbox and a reset button.
- **Chart-level controls:** metric selector for the map (Sales / Profit / Margin) and a cluster-count slider (k) for K-Means.
- **Plotly interactions:** hover tooltips, zoom, pan, click-to-zoom on the treemap and sunburst, animation playback, axis brushing on parallel coordinates.
- **Dynamic insights:** KPI cards, insight boxes and "So what" takeaways update with the filters.

## Machine Learning

### ML 01 — K-Means Customer Segmentation (PCA)
- **Unit of analysis:** one row per customer (`customer_id`), aggregated from the currently filtered orders.
- **Features (5):** total sales, total profit, average discount, number of orders, and average shipping cost.
- **Method:** features are standardized (`StandardScaler`), clustered with K-Means (`random_state=42`), and projected to 2D with PCA for the scatter plot. The number of clusters (k = 2–8, default 4) is chosen with a slider.
- **Output:** an interactive scatter plot with customer-level tooltips and a cluster profile table (customers per cluster and mean sales, profit, discount, orders and shipping cost). A takeaway highlights the least profitable segment.

### ML 02 — Random Forest Loss Classifier
- **Target:** `is_loss` (1 if the order's profit is negative).
- **Features (9):** discount, sales, quantity, shipping cost, delivery days (numerical) and category, market, segment, ship mode (one-hot encoded).
- **Model:** Random Forest (100 trees, max depth 10) in a scikit-learn pipeline, trained on an 80/20 stratified train/test split. It is trained once on the full cleaned dataset, so it does not change with the dashboard filters.
- **Evaluation:** accuracy compared with a majority-class baseline, ROC-AUC, loss recall, confusion matrix and feature importances.
- **What-if simulator:** the user changes order attributes and sees the predicted loss probability on a live gauge.

## Installation & Running Locally
```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
├── app.py                  # Main Streamlit application (layout, filters, KPIs)
├── requirements.txt
├── .streamlit/config.toml  # Theme settings
├── data/superstore.csv     # Dataset
├── eda.ipynb               # Exploratory data analysis notebook
└── src/
    ├── data_loader.py      # Loading, cleaning, feature engineering, ML models
    └── charts.py           # All chart-rendering functions
```

## Team Contributions

### [Melisa Açıkbaş] — Data Preparation & Growth Analysis (Part 1)
- **Responsibilities:** data cleaning (`eda.ipynb`, Data Quality Report), global filters and the KPI strip.
- **Charts:** 01 Line Chart, 02 Choropleth Map, 03 Treemap.

### [Tunahan Küpeli] — Discount Impact & Customer Segmentation (Part 2, first half + ML 01)
- **Responsibilities:** analysis of how discounting affects profit margin, and behavioral clustering of customers.
- **Charts:** 04 Bar Chart, 05 Sankey Diagram, 06 Animated Bubble Chart.
- **ML 01:** K-Means customer segmentation with PCA.

### [Ceren KURT] — Profit Leakage, Logistics & Loss Prediction (Part 2 end + Part 3 + ML 02)
- **Responsibilities:** multi-dimensional order analysis, market × product loss pockets, delivery performance and the loss prediction model.
- **Charts:** 07 Parallel Coordinates, 08 Heatmap, 09 Sunburst, 10 Box Plot.
- **ML 02:** Random Forest loss classifier and what-if simulator.

## Course
CEN445 Introduction to Data Visualization — Assignment 2026–2027.