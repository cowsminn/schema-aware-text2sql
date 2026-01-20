SELECT
  CAST(AVG(
    ROUND(
      ABS(
        STRFTIME('%J', final_game) - STRFTIME('%J', debut)
      ) / 365.25,
      2
    )
  ) AS REAL)
FROM player
WHERE
  debut IS NOT NULL AND final_game IS NOT NULL;