WITH PlayerStats AS (
  SELECT
    p.name_given,
    b.player_id,
    SUM(b.g) AS total_games_played,
    SUM(b.r) AS total_runs,
    SUM(b.h) AS total_hits,
    SUM(b.hr) AS total_home_runs
  FROM batting AS b
  JOIN player AS p
    ON b.player_id = p.player_id
  GROUP BY
    p.name_given,
    b.player_id
), RankedStats AS (
  SELECT
    name_given,
    player_id,
    total_games_played,
    NTILE(100) OVER (ORDER BY total_games_played DESC) AS games_played_rank,
    total_runs,
    NTILE(100) OVER (ORDER BY total_runs DESC) AS runs_rank,
    total_hits,
    NTILE(100) OVER (ORDER BY total_hits DESC) AS hits_rank,
    total_home_runs,
    NTILE(100) OVER (ORDER BY total_home_runs DESC) AS home_runs_rank
  FROM PlayerStats
)
SELECT
  name_given,
  player_id,
  total_games_played,
  games_played_rank,
  total_runs,
  runs_rank,
  total_hits,
  hits_rank,
  total_home_runs,
  home_runs_rank
FROM RankedStats
WHERE
  games_played_rank = 100 OR runs_rank = 100 OR hits_rank = 100 OR home_runs_rank = 100;