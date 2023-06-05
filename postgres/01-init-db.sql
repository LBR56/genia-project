-- CREATE TYPE
-- DROP TYPE IF EXISTS genre;
-- CREATE TYPE genre AS ENUM (
--     'ADVENTURE',
--     'HORROR',
--     'COMEDY',
--     'ACTION',
--     'SPORTS'
-- );

-- CREATE TABLE
DROP TABLE IF EXISTS raw_data;
CREATE TABLE raw_data (
    id SERIAL PRIMARY KEY,
    title VARCHAR NOT NULL,
    link VARCHAR NOT NULL,
    describe VARCHAR NOT NULL,
    cafename VARCHAR NOT NULL,
    cafeurl VARCHAR NOT NULL,
    search_date VARCHAR NOT NULL,
    query VARCHAR NOT NULL
);

    