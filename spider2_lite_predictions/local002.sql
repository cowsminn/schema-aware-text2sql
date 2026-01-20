WITH DailySales AS (
  SELECT
    DATE(order_purchase_timestamp) AS sale_date,
    SUM(oi.price) AS daily_revenue
  FROM orders AS o
  JOIN order_items AS oi
    ON o.order_id = oi.order_id
  JOIN products AS p
    ON oi.product_id = p.product_id
  JOIN product_category_name_translation AS pt
    ON p.product_category_name = pt.product_category_name
  WHERE
    pt.product_category_name_english = 'toys'
    AND o.order_purchase_timestamp BETWEEN '2017-01-01' AND '2018-08-29'
  GROUP BY
    DATE(order_purchase_timestamp)
), RegressionCoefficients AS (
  SELECT
    (
      COUNT(*) * SUM(CAST(strftime('%J', sale_date) AS REAL) * daily_revenue) - SUM(CAST(strftime('%J', sale_date) AS REAL)) * SUM(daily_revenue)
    ) / (
      COUNT(*) * SUM(CAST(strftime('%J', sale_date) AS REAL) * CAST(strftime('%J', sale_date) AS REAL)) - SUM(CAST(strftime('%J', sale_date) AS REAL)) * SUM(CAST(strftime('%J', sale_date) AS REAL))
    ) AS slope,
    (
      SUM(daily_revenue) - (
        (
          COUNT(*) * SUM(CAST(strftime('%J', sale_date) AS REAL) * daily_revenue) - SUM(CAST(strftime('%J', sale_date) AS REAL)) * SUM(daily_revenue)
        ) / (
          COUNT(*) * SUM(CAST(strftime('%J', sale_date) AS REAL) * CAST(strftime('%J', sale_date) AS REAL)) - SUM(CAST(strftime('%J', sale_date) AS REAL)) * SUM(CAST(strftime('%J', sale_date) AS REAL))
        )
      ) * SUM(CAST(strftime('%J', sale_date) AS REAL))
    ) / COUNT(*) AS intercept
  FROM DailySales
), PredictedSales AS (
  SELECT
    DATE('2018-12-05', '+' || (
      ROW_NUMBER() OVER () - 1
    ) || ' days') AS predicted_date,
    (
      SELECT
        slope
      FROM RegressionCoefficients
    ) * CAST(strftime('%J', DATE('2018-12-05', '+' || (
      ROW_NUMBER() OVER () - 1
    ) || ' days')) AS REAL) + (
      SELECT
        intercept
      FROM RegressionCoefficients
    ) AS predicted_revenue
  FROM (
    SELECT
      1
    UNION ALL
    SELECT
      1
    UNION ALL
    SELECT
      1
    UNION ALL
    SELECT
      1
  )
), MovingAverages AS (
  SELECT
    predicted_date,
    AVG(predicted_revenue) OVER (ORDER BY predicted_date ASC ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) AS moving_average
  FROM PredictedSales
), FinalResult AS (
  SELECT
    SUM(moving_average) AS total_moving_average_sum
  FROM MovingAverages
)
SELECT
  total_moving_average_sum
FROM FinalResult;