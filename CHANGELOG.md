Changelog

# 0.28.0
* upload(): strings containing the delimiter, backslashes, newlines, tabs or carriage returns are now escaped and survive the upload unchanged
* get(): intNullVal=None keeps the nulls in integer columns as None (the column is returned as an object array) instead of failing
* get(): statements that do not return a result set no longer fail with sqlite3/duckdb
* get(): use a unique server-side cursor name so that several batched queries can be run over the same connection
* get() with duckdb now uses the much faster fetchnumpy() interface. As a result the integer/float columns keep the width of the database type (e.g. int -> int32, float -> float32, rather than always int64/float64), dates, timestamps with time zone and intervals are returned as datetime64/timedelta64, decimals and hugeints as float64, blobs as object arrays of bytearrays, and list columns are now supported
* add the port argument to execute() and upload()
* upload(): analyze is now run inside the try block, so a failure there rolls back the transaction
* fix the strLength docstring (strings are not truncated, the allocated length is extended as needed)
* packaging: remove numpydoc and a duplicate numpy from the dependencies; the readthedocs build uses docs/requirements.txt and python 3.12

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
