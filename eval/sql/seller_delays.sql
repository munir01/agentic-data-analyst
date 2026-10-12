WITH seller_orders AS (
    SELECT DISTINCT seller_id, order_id
    FROM order_items
),
eligible AS (
    SELECT
        so.seller_id,
        so.order_id,
        CASE
            WHEN datetime(o.order_delivered_customer_date) > datetime(o.order_estimated_delivery_date)
            THEN 1 ELSE 0
        END AS is_late
    FROM seller_orders so
    JOIN orders o ON so.order_id = o.order_id
    WHERE o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
)
SELECT
    seller_id,
    COUNT(*) AS delivered_orders,
    SUM(is_late) AS late_orders,
    ROUND(100.0 * SUM(is_late) / COUNT(*), 4) AS late_rate_pct
FROM eligible
GROUP BY seller_id
HAVING COUNT(*) >= 100
ORDER BY late_rate_pct DESC, delivered_orders DESC, seller_id ASC
LIMIT 10;
