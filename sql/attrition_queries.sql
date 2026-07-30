-- Attrition rate by department and job role
SELECT Department, JobRole,
       ROUND(AVG(CASE WHEN Attrition = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) AS AttritionRatePercent,
       COUNT(*) AS Headcount
FROM employees
GROUP BY Department, JobRole
ORDER BY AttritionRatePercent DESC;

-- Average tenure of employees who left vs stayed
SELECT Attrition, 
       ROUND(AVG(YearsAtCompany), 2) AS AvgTenureYears,
       ROUND(AVG(MonthlyIncome), 2) AS AvgMonthlyIncome
FROM employees
GROUP BY Attrition;

-- Attrition rate by WorkLifeBalance rating
SELECT WorkLifeBalance,
       COUNT(*) AS TotalEmployees,
       SUM(Attrition) AS AttritionCount,
       ROUND(AVG(CASE WHEN Attrition = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) AS AttritionRatePercent
FROM employees
GROUP BY WorkLifeBalance
ORDER BY WorkLifeBalance ASC;

-- Join with the Exit Interview table to analyze top exit reasons by department
SELECT e.Department,
       ei.ExitReason,
       COUNT(*) AS ExitCount,
       ROUND(AVG(e.MonthlyIncome), 2) AS AvgIncomeOfExits
FROM employees e
INNER JOIN exit_interviews ei ON e.EmployeeNumber = ei.EmployeeNumber
WHERE e.Attrition = 1
GROUP BY e.Department, ei.ExitReason
ORDER BY e.Department, ExitCount DESC;
