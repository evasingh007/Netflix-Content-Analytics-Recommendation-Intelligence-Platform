USE NetflixDB;
GO

-- 1. Data Integrity Check: Confirm record count
SELECT COUNT(*) AS total_records FROM cleaned_netflix_titles;

-- 2. Movies vs TV Shows Content Split
SELECT 
    type,
    COUNT(*) AS total_titles,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM cleaned_netflix_titles), 2) AS percentage_share
FROM cleaned_netflix_titles
GROUP BY type;

-- 3. Top 10 Content Producing Countries
SELECT TOP 10
    country,
    COUNT(*) AS title_count
FROM cleaned_netflix_titles
WHERE country <> 'Unknown'
GROUP BY country
ORDER BY title_count DESC;

-- 4. Window Function: Year-over-Year (YoY) Growth Using LAG()
WITH YearlyCatalog AS (
    SELECT 
        year_added,
        COUNT(*) AS titles_added
    FROM cleaned_netflix_titles
    WHERE year_added IS NOT NULL
    GROUP BY year_added
)
SELECT 
    year_added,
    titles_added,
    LAG(titles_added, 1) OVER (ORDER BY year_added) AS previous_year_added,
    ROUND(
        (titles_added - LAG(titles_added, 1) OVER (ORDER BY year_added)) * 100.0 / 
        NULLIF(LAG(titles_added, 1) OVER (ORDER BY year_added), 0), 2
    ) AS yoy_growth_rate
FROM YearlyCatalog
ORDER BY year_added DESC;

-- 5. Window Function: Top 3 Rated Content Categories per Country Using DENSE_RANK()
WITH CountryRatings AS (
    SELECT 
        country,
        rating,
        COUNT(*) AS total_count,
        DENSE_RANK() OVER (PARTITION BY country ORDER BY COUNT(*) DESC) AS ranking
    FROM cleaned_netflix_titles
    WHERE country <> 'Unknown' AND rating <> 'Not Rated'
    GROUP BY country, rating
)
SELECT country, rating, total_count
FROM CountryRatings
WHERE ranking <= 3
ORDER BY country, ranking;