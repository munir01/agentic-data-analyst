WITH review_per_order AS (
    SELECT order_id, AVG(review_score) AS review_score
    FROM reviews
    WHERE review_score IS NOT NULL
    GROUP BY order_id
),
ranked_payments AS (
    SELECT
        order_id,
        payment_type,
        payment_value,
        payment_sequential,
        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY payment_value DESC, payment_sequential ASC, payment_type ASC
        ) AS rn
    FROM payments
    WHERE payment_type IS NOT NULL
),
primary_payment AS (
    SELECT order_id, payment_type
    FROM ranked_payments
    WHERE rn = 1
)
SELECT
    p.payment_type,
    COUNT(*) AS reviewed_orders,
    ROUND(AVG(r.review_score), 6) AS avg_review_score
FROM primary_payment p
JOIN review_per_order r ON p.order_id = r.order_id
GROUP BY p.payment_type
HAVING COUNT(*) >= 100
ORDER BY avg_review_score ASC, reviewed_orders DESC, p.payment_type ASC;
