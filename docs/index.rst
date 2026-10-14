sqlutilpy
=========

``sqlutilpy`` is a Python module to query SQL databases and get the results
back as numpy arrays, to upload numpy arrays as tables, and to run join
queries between your local arrays and the tables in the database.
It is designed to deal efficiently with query results of millions of rows.

Three database backends are supported (see :doc:`backends`):

* **PostgreSQL** (through `psycopg <https://www.psycopg.org/>`_), with the
  full set of features: querying, uploading and local joins
* **SQLite** (through the standard library ``sqlite3`` module): querying
* **DuckDB** (through the ``duckdb`` package): querying

The source code is on `GitHub <https://github.com/segasai/sqlutilpy>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   backends
   api

Installation
------------

The package is available on PyPI::

    pip install sqlutilpy

This installs the two required dependencies, ``numpy`` and ``psycopg``.
Python 3.10 or newer and numpy 2 or newer are required. Some features
need optional packages that you install yourself:

===================  ============================================
Package              Used for
===================  ============================================
``duckdb``           the ``driver='duckdb'`` backend
``astropy``          uploading ``astropy.table.Table`` objects
``pandas``           uploading ``pandas.DataFrame`` objects
===================  ============================================

Authentication and connection information
-----------------------------------------

Every function takes the connection details as keyword arguments:
``db`` (the database name, or the file name for SQLite and DuckDB),
``driver`` (``'psycopg'`` by default), and for PostgreSQL also ``host``,
``port``, ``user`` and ``password``.

For PostgreSQL anything you do not specify is taken from the standard
libpq environment variables ``PGHOST``, ``PGPORT``, ``PGDATABASE`` and
``PGUSER``, and the password from the `.pgpass file
<https://www.postgresql.org/docs/current/libpq-pgpass.html>`_. The
examples below assume that these are set up, so that the connection
details can be omitted. If that is not the case, add ``host=...``,
``db=...``, ``user=...`` and ``password=...`` to the calls.

Querying the database and retrieving the results
------------------------------------------------

:func:`sqlutilpy.get` runs a query and returns one numpy array per column::

    import sqlutilpy
    ra, dec = sqlutilpy.get('select ra, dec from mytable',
                            host='HOST_NAME_OF_MY_PG_SERVER',
                            db='THE_NAME_OF_MY_DB')

With ``asDict=True`` you get an ordered dictionary keyed by column name
instead::

    res = sqlutilpy.get('select ra, dec, name from mytable', asDict=True)
    res['ra']

Query parameters are passed as a tuple through ``params``. The
placeholder syntax is that of the underlying driver: ``%s`` for
PostgreSQL, ``?`` for SQLite and DuckDB::

    ra, dec = sqlutilpy.get('select ra, dec from mytable where name = %s',
                            ('NGC 3166', ))

Column types and nulls
^^^^^^^^^^^^^^^^^^^^^^

Integer, floating point, boolean, string, timestamp and array columns are
converted to numpy arrays of the corresponding type. Since numpy integer
and string arrays cannot hold nulls, these are replaced:

* nulls in integer columns become ``intNullVal`` (``-9999`` by default).
  With ``intNullVal=None`` the column is returned as an object array with
  ``None`` in place of the nulls
* nulls in string columns become ``strNullVal`` (``'None'`` by default)
* nulls in floating point columns become ``NaN`` and nulls in timestamp
  columns become ``NaT``

String columns are returned as fixed-width unicode arrays, wide enough
for the longest string in the result.

Batched and non-batched mode (PostgreSQL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With PostgreSQL the results are by default fetched through a server-side
cursor in batches of ``sqlutilpy.sqlutil.config.arraysize`` rows (100000)
and converted to numpy on the fly, so that the memory use is about the
size of the final arrays. With ``batched=False`` the whole result is
fetched at once. This needs roughly twice the memory, but the query is
then eligible for PostgreSQL's parallel plans and can be several times
faster for heavy queries::

    x, = sqlutilpy.get('select sum(flux) from hugetable', batched=False)

SQLite and DuckDB
-----------------

The same :func:`sqlutilpy.get` and :func:`sqlutilpy.execute` work with
SQLite and DuckDB files, by selecting the driver and giving the file name
as the database::

    a, b = sqlutilpy.get('select a, b from mytable',
                         db='mydata.db', driver='sqlite3')
    a, b = sqlutilpy.get('select a, b from mytable',
                         db='mydata.duckdb', driver='duckdb')

Uploading tables and local joins are only available with PostgreSQL.
See :doc:`backends` for the details of each backend.

Executing statements
--------------------

:func:`sqlutilpy.execute` runs a statement that returns nothing and
commits it (unless ``noCommit=True``)::

    sqlutilpy.execute('create table mytab (a int, b real)')
    sqlutilpy.execute('insert into mytab values (%s, %s)', (1, 2.5))

Uploading your arrays as columns in a table (PostgreSQL)
--------------------------------------------------------

:func:`sqlutilpy.upload` creates a table from a tuple of arrays and a
tuple of column names::

    x = np.arange(10)
    y = x**.5
    sqlutilpy.upload('mytable', (x, y), ('xcol', 'ycol'))

This creates a table called ``mytable`` with the columns ``xcol`` and
``ycol``. An ``astropy.table.Table``, a ``pandas.DataFrame`` or a
dictionary of arrays can be given instead, in which case the column
names are taken from it::

    tab = astropy.table.Table.read('mydata.fits')
    sqlutilpy.upload('mytable', tab)

Integer, floating point, boolean, string (unicode or bytes) and
``datetime64`` columns are supported, as well as object columns holding
numpy arrays, which become array columns. Characters that are not
allowed in PostgreSQL identifiers (spaces, brackets and the like) are
replaced by underscores in the column names, with a warning.
The data is sent through ``COPY``, so uploading millions of rows is fast.

Join query involving your local data and a database table (PostgreSQL)
----------------------------------------------------------------------

Sometimes it is beneficial to run a join query involving your local data
and the data in the database. Imagine you have arrays ``myid`` and ``y``
and want all the rows of ``somebigtable`` with ``id`` equal to one of
``myid``. You could upload the arrays and run a query yourself, but
:func:`sqlutilpy.local_join` does that for you::

    myid = np.arange(10)
    y = np.random.uniform(size=10)
    R = sqlutilpy.local_join(
        '''select * from mytmptable as m, somebigtable as s
           where s.id = m.myid order by m.myid''',
        'mytmptable', (myid, y), ('myid', 'ycol'))

It executes the query as if your arrays were in ``mytmptable``. Behind
the scenes the data is uploaded into a temporary table, which only exists
for the duration of the call.

Keeping the connection open
---------------------------

By default each call opens a new connection and closes it afterwards.
Often it is beneficial to preserve an open connection. Obtain one with
:func:`sqlutilpy.getConnection` and pass it to the other functions
through the ``conn`` keyword::

    conn = sqlutilpy.getConnection(db='mydb', user='myuser',
                                   password='something', host='hostname')
    R = sqlutilpy.get('select 1', conn=conn)
    R1 = sqlutilpy.get('select 1', conn=conn)
    conn.close()

When ``conn`` is given, ``driver`` must still be passed if the connection
is not a PostgreSQL one.

How to cite the software
------------------------

If you use this package, please cite it through Zenodo:
https://doi.org/10.5281/zenodo.5160118

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
