import sqlite3
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
DB_DIR = Path("db")
DB_PATH = DB_DIR / "olist.db"

DB_DIR.mkdir(exist_ok=True)

files = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

with sqlite3.connect(DB_PATH) as conn:
    for table, filename in files.items():
        print(f"Loading {table}...")

        df = pd.read_csv(DATA_DIR / filename)
        df.to_sql(table, conn, if_exists="replace", index=False)

        print(f"  {len(df):,} rows")

print(f"\nDatabase created: {DB_PATH}")