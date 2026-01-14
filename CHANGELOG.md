Changelog

# 0.27.0
* Introduce the non-batched mode with batched=False that allows (at the cost of more memory use) using the parallel PostgreSQL plans, which can speed up queries by a factor of few
* remove the notNamed option of sqlutil.get
* refactor the threading for converting data, to use the threadpool
* I added more tests when strings are read, so hopefully the truncation into 20 characters should be more rare now.

# 0.26.0
* Preserve uppercase columns when ingestng data
* Remove obsolet numpy functionality

# 0.25.0
* Allow querying timestamp with timezone types
* Allow sqlutilpy.connect() to the duckdb db

# 0.24.0
* In the case sqlutilpy.get() returns no rows and asDict=True, the column names will still be there

# 0.23.0
* Switch to psycopg (i.e. third version of psycopg)
* Make the code compatible with numpy 2.0

# 0.22.0
* This version migrated to pyproject.toml

# 0.21.0
* Allow uploading int8 columns
* Allow uploading data where some columns have arrays

# 0.20.0

* Increase default string length in local_join
* add intNullVal option to local_join
