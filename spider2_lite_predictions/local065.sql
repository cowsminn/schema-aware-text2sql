WITH cleaned_pizza_orders AS (
    SELECT 
        pco.order_id,
        pco.customer_id,
        pco.pizza_id,
        pco.exclusions,
        pco.extras,
        pco.order_time,
        pro.cancellation
    FROM 
        pizza_clean_customer_orders pco
    JOIN 
        pizza_runner_orders pro ON pco.order_id = pro.order_id
    WHERE 
        pro.cancellation = 'NULL'
),
pizza_prices AS (
    SELECT 
        pn.pizza_id,
        pn.pizza_name,
        CASE 
            WHEN pn.pizza_name = 'Meat Lovers' THEN 12
            WHEN pn.pizza_name = 'Vegetarian' THEN 10
            ELSE 0
        END AS base_price
    FROM 
        pizza_names pn
),
order_prices AS (
    SELECT 
        cpo.order_id,
        cpo.pizza_id,
        cpo.extras,
        pp.base_price,
        pp.base_price + LENGTH(cpo.extras) - LENGTH(REPLACE(cpo.extras, ',', '')) AS total_price
    FROM 
        cleaned_pizza_orders cpo
    JOIN 
        pizza_prices pp ON cpo.pizza_id = pp.pizza_id
)
SELECT 
    SUM(op.total_price) AS total_income
FROM 
    order_prices op
WHERE 
    op.pizza_id IN (SELECT pizza_id FROM pizza_names WHERE pizza_name IN ('Meat Lovers', 'Vegetarian'));