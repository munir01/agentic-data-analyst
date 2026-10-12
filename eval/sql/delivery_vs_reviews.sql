WITH review_per_order AS (
    SELECT order_id, AVG(review_score) AS review_score
    FROM reviews
    WHERE review_score IS NOT NULL
    GROUP BY order_id
),
eligible AS (
    SELECT
        o.order_id,
        CASE
            WHEN datetime(o.order_delivered_customer_date) > datetime(o.order_estimated_delivery_date)
            THEN 'Late' ELSE 'On Time'
        END AS delivery_status,
        r.review_score
    FROM orders o
    JOIN review_per_order r ON o.order_id = r.order_id
    WHERE o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
)
SELECT
    delivery_status,
    COUNT(*) AS reviewed_orders,
    ROUND(AVG(review_score), 6) AS avg_review_score
FROM eligible
GROUP BY delivery_status
ORDER BY delivery_status;
