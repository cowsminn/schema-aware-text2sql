WITH CustomerOrderPayments AS (
  SELECT
    c.customer_unique_id,
    o.order_id,
    o.order_purchase_timestamp,
    SUM(op.payment_value) AS total_payment
  FROM customers AS c
  JOIN orders AS o
    ON c.customer_id = o.customer_id
  JOIN order_payments AS op
    ON o.order_id = op.order_id
  GROUP BY
    c.customer_unique_id,
    o.order_id,
    o.order_purchase_timestamp
), CustomerAverages AS (
  SELECT
    customer_unique_id,
    COUNT(order_id) AS total_orders,
    AVG(total_payment) AS average_payment,
    MIN(order_purchase_timestamp) AS first_purchase,
    MAX(order_purchase_timestamp) AS last_purchase
  FROM CustomerOrderPayments
  GROUP BY
    customer_unique_id
), TopCustomers AS (
  SELECT
    customer_unique_id,
    total_orders,
    average_payment,
    first_purchase,
    last_purchase,
    NTILE(100) OVER (ORDER BY average_payment DESC) AS customer_rank
  FROM CustomerAverages
  ORDER BY
    average_payment DESC
  LIMIT 3
)
SELECT
  customer_unique_id,
  total_orders,
  average_payment,
  CASE
    WHEN CAST(JULIANDAY(last_purchase) - JULIANDAY(first_purchase) AS REAL) / 7 < 1
    THEN 1.0
    ELSE CAST(JULIANDAY(last_purchase) - JULIANDAY(first_purchase) AS REAL) / 7
  END AS customer_lifespan_weeks
FROM TopCustomers;