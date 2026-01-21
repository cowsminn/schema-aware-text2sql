WITH runs_scored AS (
    SELECT 
        bb.striker AS player_id,
        bb.match_id,
        SUM(bs.runs_scored) AS total_runs
    FROM 
        ball_by_ball AS bb
    JOIN 
        batsman_scored AS bs ON bb.match_id = bs.match_id AND bb.over_id = bs.over_id AND bb.ball_id = bs.ball_id AND bb.innings_no = bs.innings_no
    GROUP BY 
        bb.striker, bb.match_id
),
player_match_info AS (
    SELECT 
        pm.player_id,
        pm.match_id,
        pm.team_id,
        m.match_winner
    FROM 
        player_match AS pm
    JOIN 
        match AS m ON pm.match_id = m.match_id
)
SELECT 
    p.player_name
FROM 
    player AS p
JOIN 
    runs_scored AS rs ON p.player_id = rs.player_id
JOIN 
    player_match_info AS pm ON rs.match_id = pm.match_id AND rs.player_id = pm.player_id
WHERE 
    rs.total_runs >= 100 AND pm.team_id != pm.match_winner