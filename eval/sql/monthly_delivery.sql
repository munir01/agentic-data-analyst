WITH eligible AS (
    SELECT
        substr(order_purchase_timestamp, 1, 7) AS purchase_month,
        CASE
            WHEN datetime(order_delivered_customer_date) > datetime(order_estimated_delivery_date)
            THEN 1 ELSE 0
        END AS is_late
    FROM orders
    WHERE order_purchase_timestamp IS NOT NULL
      AND order_delivered_customer_date IS NOT NULL
      AND order_estimated_delivery_date IS NOT NULL
)
SELECT
    purchase_month,
    COUNT(*) AS delivered_orders,
    SUM(is_late) AS late_orders,
    ROUND(100.0 * SUM(is_late) / COUNT(*), 4) AS late_rate_pct
FROM eligible
GROUP BY purchase_month
HAVING COUNT(*) >= 100
ORDER BY purchase_month;