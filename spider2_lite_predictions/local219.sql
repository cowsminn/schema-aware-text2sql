WITH team_wins AS (
    SELECT 
        M.league_id,
        M.home_team_api_id AS team_api_id,
        CASE 
            WHEN M.home_team_goal > M.away_team_goal THEN 1 
            ELSE 0 
        END AS win
    FROM 
        Match M
    UNION ALL
    SELECT 
        M.league_id,
        M.away_team_api_id AS team_api_id,
        CASE 
            WHEN M.away_team_goal > M.home_team_goal THEN 1 
            ELSE 0 
        END AS win
    FROM 
        Match M
),
team_win_counts AS (
    SELECT 
        league_id,
        team_api_id,
        SUM(win) AS total_wins,
        ROW_NUMBER() OVER (PARTITION BY league_id ORDER BY SUM(win)) AS row_num
    FROM 
        team_wins
    GROUP BY 
        league_id, team_api_id
)
SELECT 
    L.name AS league_name,
    T.team_long_name AS team_name,
    TWC.total_wins
FROM 
    team_win_counts TWC
JOIN 
    League L ON TWC.league_id = L.id
JOIN 
    Team T ON TWC.team_api_id = T.team_api_id
WHERE 
    TWC.row_num = 1;