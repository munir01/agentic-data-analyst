import sqlite3
import pandas as pd

DB_PATH = "db/olist.db"

query = """
SELECT
    CASE
        WHEN datetime(o.order_delivered_customer_date) >
             datetime(o.order_estimated_delivery_date)
        THEN 'Late'
        ELSE 'On time'
    END AS delivery_status,

    COUNT(*) AS num_reviews,
    ROUND(AVG(r.review_score), 2) AS avg_review_score

FROM orders o
JOIN reviews r
    ON o.order_id = r.order_id

WHERE o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL

GROUP BY delivery_status;
"""

with sqlite3.connect(DB_PATH) as conn:
    result = pd.read_sql_query(query, conn)

print(result.to_string(index=False))
