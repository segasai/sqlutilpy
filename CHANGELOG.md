Changelog

# 0.27.0
* Allow parallel plans for sqlutilpy.get. This can significantly speed up some queries. Previously in some case they were impossible due to cursor not explicitly marked 'not-scrollable'

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
