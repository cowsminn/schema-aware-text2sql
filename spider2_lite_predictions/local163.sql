WITH ranked_faculty AS (
    SELECT 
        FacRank,
        FacFirstName,
        FacLastName,
        FacSalary,
        AVG(FacSalary) OVER (PARTITION BY FacRank) AS avg_salary
    FROM 
        university_faculty
),
closest_to_average AS (
    SELECT 
        FacRank,
        FacFirstName,
        FacLastName,
        FacSalary,
        ROW_NUMBER() OVER (PARTITION BY FacRank ORDER BY ABS(FacSalary - avg_salary)) AS row_num
    FROM 
        ranked_faculty
)
SELECT 
    FacRank,
    FacFirstName,
    FacLastName,
    FacSalary
FROM 
    closest_to_average
WHERE 
    row_num = 1;