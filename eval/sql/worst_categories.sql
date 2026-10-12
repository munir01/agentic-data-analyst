WITH review_per_order AS (
    SELECT order_id, AVG(review_score) AS review_score
    FROM reviews
    WHERE review_score IS NOT NULL
    GROUP BY order_id
),
order_categories AS (
    SELECT DISTINCT
        oi.order_id,
        COALESCE(ct.product_category_name_english, p.product_category_name, 'Unknown') AS category
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN category_translation ct
        ON p.product_category_name = ct.product_category_name
    WHERE p.product_category_name IS NOT NULL
),
scored AS (
    SELECT oc.category, oc.order_id, r.review_score
    FROM order_categories oc
    JOIN review_per_order r ON oc.order_id = r.order_id
)
SELECT
    category,
    COUNT(*) AS reviewed_orders,
    ROUND(AVG(review_score), 6) AS avg_review_score
FROM scored
GROUP BY category
HAVING COUNT(*) >= 100
ORDER BY avg_review_score ASC, reviewed_orders DESC, category ASC
LIMIT 10;
