Database backends
=================

The backend is selected with the ``driver`` argument. Unknown driver
names raise :class:`sqlutilpy.SqlUtilException` before any connection
is attempted.

PostgreSQL (``driver='psycopg'``)
---------------------------------

This is the default and the most complete backend. All the functions
work with it. The connection is described by ``db``, ``host``, ``port``,
``user`` and ``password``, with libpq's environment variables and
``.pgpass`` filling in whatever is missing.

Query parameters use ``%s`` placeholders. Statements that do not return
rows (``create``, ``insert`` and so on) should be run through
:func:`sqlutilpy.execute`, since :func:`sqlutilpy.get` wraps the query in
a server-side cursor in its default batched mode.

The driver name ``'psycopg2'`` is still accepted for backwards
compatibility, with a warning, and is treated as ``'psycopg'``.

SQLite (``driver='sqlite3'``)
-----------------------------

``db`` is the path of the database file. The optional ``timeout``
argument (in seconds, 5 by default) is how long to wait for a locked
database. ``user``, ``password``, ``host`` and ``port`` are not used.

Query parameters use ``?`` placeholders. :func:`sqlutilpy.get` and
:func:`sqlutilpy.execute` work; :func:`sqlutilpy.upload` and
:func:`sqlutilpy.local_join` require PostgreSQL. ``batched``,
``strLength`` and ``nthreads`` have no effect, since the whole result
is fetched at once and the string widths are inferred from the data.

::

    a, b = sqlutilpy.get('select a, b from tmp where a > ?', (3, ),
                         db='mydata.db', driver='sqlite3')

DuckDB (``driver='duckdb'``)
----------------------------

Requires the ``duckdb`` package. ``db`` is the path of the database file,
or ``':memory:'`` for an in-memory database. As with SQLite, parameters
use ``?``, only :func:`sqlutilpy.get` and :func:`sqlutilpy.execute` are
available, and ``batched``, ``strLength`` and ``nthreads`` have no effect.

The results are fetched through DuckDB's numpy interface, so the arrays
keep the width of the database type (an ``INTEGER`` column gives
``int32``, a ``FLOAT`` column ``float32``). Dates, timestamps and
intervals come back as ``datetime64`` and ``timedelta64``, decimals and
``HUGEINT`` as ``float64``, lists as object arrays of arrays. Nulls are
handled as described on the :doc:`main page <index>`; when ``intNullVal`` does not fit
into a narrow or unsigned integer column, the column is widened to
``int64``.

::

    conn = sqlutilpy.getConnection(db=':memory:', driver='duckdb')
    sqlutilpy.execute("create table t as select * from 'data.parquet'",
                      conn=conn, driver='duckdb')
    x, y = sqlutilpy.get('select x, y from t', conn=conn, driver='duckdb')
