WITH CustomerOrderTotals AS (
  SELECT
    c.customer_unique_id,
    o.order_id,
    SUM(oi.price + oi.freight_value) AS total_spend
  FROM customers AS c
  JOIN orders AS o
    ON c.customer_id = o.customer_id
  JOIN order_items AS oi
    ON o.order_id = oi.order_id
  WHERE
    o.order_status = 'delivered'
  GROUP BY
    c.customer_unique_id,
    o.order_id
), CustomerRFM AS (
  SELECT
    customer_unique_id,
    COUNT(order_id) AS Frequency,
    SUM(total_spend) AS MonetaryValue,
    MAX(order_purchase_timestamp) AS last_purchase_date
  FROM CustomerOrderTotals
  JOIN orders
    ON CustomerOrderTotals.order_id = orders.order_id
  GROUP BY
    customer_unique_id
), RFM_Scores AS (
  SELECT
    customer_unique_id,
    Frequency,
    MonetaryValue,
    last_purchase_date,
    DATE('now') AS analysis_date,
    JULIANDAY(DATE('now')) - JULIANDAY(last_purchase_date) AS Recency,
    NTILE(5) OVER (ORDER BY JULIANDAY(DATE('now')) - JULIANDAY(last_purchase_date) DESC) AS R_Score,
    NTILE(5) OVER (ORDER BY COUNT(customer_unique_id) OVER (PARTITION BY customer_unique_id)) AS F_Score,
    NTILE(5) OVER (ORDER BY SUM(MonetaryValue) OVER (PARTITION BY customer_unique_id)) AS M_Score
  FROM CustomerRFM
), RFM_Segments AS (
  SELECT
    customer_unique_id,
    R_Score,
    F_Score,
    M_Score,
    CAST(R_Score AS TEXT) || CAST(F_Score AS TEXT) || CAST(M_Score AS TEXT) AS RFM_Segment
  FROM RFM_Scores
)
SELECT
  RFM_Segment,
  COUNT(DISTINCT r.customer_unique_id) AS num_customers,
  SUM(MonetaryValue) / SUM(Frequency) AS avg_sales_per_order
FROM RFM_Segments AS r
JOIN CustomerRFM AS c
  ON r.customer_unique_id = c.customer_unique_id
GROUP BY
  RFM_Segment
ORDER BY
  RFM_Segment;