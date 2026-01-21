WITH ranked_interests AS (
    SELECT 
        im.interest_name,
        im.id,
        im.interest_summary,
        i._month,
        i._year,
        i.month_year,
        i.composition,
        ROW_NUMBER() OVER (ORDER BY i.composition DESC) AS desc_rank,
        ROW_NUMBER() OVER (ORDER BY i.composition ASC) AS asc_rank
    FROM 
        interest_metrics i
    JOIN 
        interest_map im ON i.interest_id = im.id
)
SELECT 
    month_year AS time,
    interest_name,
    composition
FROM 
    ranked_interests
WHERE 
    desc_rank <= 10 OR asc_rank <= 10
ORDER BY 
    composition DESC;