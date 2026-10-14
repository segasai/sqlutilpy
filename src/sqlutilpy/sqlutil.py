"""Query SQL databases (PostgreSQL, SQLite, DuckDB) and get numpy
arrays back, upload numpy arrays as tables and join local arrays with
database tables
"""
import numpy
import numpy as np
import psycopg
import collections
import warnings
import uuid
import concurrent.futures

try:
    import astropy.table as atpy
except ImportError:
    # astropy is not installed
    atpy = None
try:
    import pandas
except ImportError:
    # pandas is not installed
    pandas = None

_WAIT_SELECT_TIMEOUT = 10
STRLEN_DEFAULT = 20


class config:
    arraysize = 100000


class SqlUtilException(Exception):
    pass


SUPPORTED_DRIVERS = ('psycopg', 'sqlite3', 'duckdb')


def __check_driver(driver):
    """
    Validate the driver name and return its canonical form.
    The obsolete 'psycopg2' is accepted with a warning and mapped to
    'psycopg'; any other unknown driver raises SqlUtilException
    """
    if driver == 'psycopg2':
        warnings.warn(
            'psycopg2 driver is not supported anymore using psycopg instead')
        driver = 'psycopg'
    if driver not in SUPPORTED_DRIVERS:
        raise SqlUtilException(
            'Unknown driver %r; supported drivers are %s' %
            (driver, ', '.join(SUPPORTED_DRIVERS)))
    return driver


def getConnection(db=None,
                  driver=None,
                  user=None,
                  password=None,
                  host=None,
                  port=None,
                  timeout=None):
    """
    Obtain the connection object to the DB.
    It may be useful to avoid reconnecting to the DB repeatedly.

    Parameters
    ----------
    db : string
        The name of the database (PostgreSQL) or the filename of the
        database (SQLite, DuckDB). For DuckDB ':memory:' gives an
        in-memory database. For PostgreSQL, if this and the other
        connection details are not given, the libpq defaults and
        environment variables (PGDATABASE, PGHOST, PGPORT, PGUSER) are used
    driver : string
        The database driver: 'psycopg' (PostgreSQL), 'sqlite3' or 'duckdb'
    user : string, optional
        Username (PostgreSQL only)
    password : string, optional
        Password (PostgreSQL only)
    host : string, optional
        Hostname (PostgreSQL only)
    port : integer, optional
        Connection port (PostgreSQL only, by default 5432)
    timeout : integer, optional
        How long to wait for a locked database, in seconds
        (SQLite only, by default 5)

    Returns
    -------
    conn : object
        Database connection

    Examples
    --------
    >>> conn = sqlutilpy.getConnection(db='mydb', host='myhost',
    ...                                driver='psycopg')
    >>> conn = sqlutilpy.getConnection(db='mydata.db', driver='sqlite3')
    >>> conn = sqlutilpy.getConnection(db=':memory:', driver='duckdb')
    """
    driver = __check_driver(driver)
    if driver == 'psycopg':
        conn_dict = dict()
        if db is not None:
            conn_dict['dbname'] = db
        if host is not None:
            conn_dict['host'] = host
        if port is not None:
            conn_dict['port'] = port
        if user is not None:
            conn_dict['user'] = user
        if password is not None:
            conn_dict['password'] = password
        conn = psycopg.connect(**conn_dict)
    elif driver == 'sqlite3':
        import sqlite3
        if timeout is None:
            timeout = 5
        conn = sqlite3.connect(db, timeout=timeout)
    elif driver == 'duckdb':
        import duckdb
        conn = duckdb.connect(db)
    return conn


def getCursor(conn, driver=None, preamb=None, named=True):
    """
    Retrieve the database cursor

    Parameters
    ----------
    conn : object
        Database connection
    driver : string
        The database driver: 'psycopg', 'sqlite3' or 'duckdb'
    preamb : string, optional
        SQL to be executed on the cursor before anything else
    named : bool
        If True (the default) and the driver is 'psycopg', a server-side
        (named) cursor is returned, so that the results can be fetched in
        batches. Ignored for the other drivers

    Returns
    -------
    cur : object
        Database cursor
    """
    driver = __check_driver(driver)
    if driver == 'psycopg':
        cur = conn.cursor(scrollable=False)
        if preamb is not None:
            cur.execute(preamb)
        else:
            cur.execute('set cursor_tuple_fraction TO 1')
            # this is required because otherwise PG may decide to execute a
            # different plan
        if not named:
            return cur
        # the unique cursor name allows several batched queries
        # to run over the same connection
        cur = conn.cursor(name='sqlutilcursor_' + uuid.uuid4().hex)
        cur.arraysize = config.arraysize
    elif driver == 'sqlite3':
        cur = conn.cursor()
        if preamb is not None:
            cur.execute(preamb)
    elif driver == 'duckdb':
        cur = conn.cursor()
        if preamb is not None:
            cur.execute(preamb)
    return cur


def __fromrecords(recList, dtype=None, intNullVal=None, strNullVal='None'):
    """
    This function was taken from np.core.records and updated to
    support conversion null integers to intNullVal and strings to strNullVal
    """

    descr = np.dtype((np.record, dtype))

    # Check if we must avoid fast path for strings to prevent
    # None -> 'None' conversion
    # Note: numpy's default conversion for None to string is 'None', so if
    # strNullVal is 'None', we can still use the fast path.
    force_object_strings = False
    if strNullVal is not None and strNullVal != 'None':
        if any(dtype.fields[n][0].kind in ('S', 'U') for n in dtype.names):
            force_object_strings = True

    if not force_object_strings:
        try:
            retval = np.array(recList, dtype=descr)
            return retval.view(numpy.recarray)
        except (TypeError, ValueError):
            # Failed (likely due to None in int col). Fall through to object
            # path.
            pass

    # Vectorized fallback path using object arrays
    names = dtype.names
    new_formats = []
    converters = {}

    for name in names:
        dt = dtype.fields[name][0]
        if dt.kind in ('i', 'u'):
            new_formats.append(object)
            converters[name] = intNullVal
        elif dt.kind in ('S', 'U') and strNullVal is not None:
            new_formats.append(object)
            converters[name] = strNullVal
        else:
            new_formats.append(dt)

    temp_dtype = np.dtype({'names': names, 'formats': new_formats})

    if len(recList) == 0:
        return np.recarray((0, ), dtype=descr)

    try:
        arr = np.array(recList, dtype=temp_dtype)
    except (TypeError, ValueError):
        # Handle cases where np.array fails (e.g. ragged lists)
        # by manually filling the object array
        shape = (len(recList), )
        arr = np.recarray(shape, temp_dtype)
        for k in range(len(recList)):
            arr[k] = tuple(recList[k])

    # Apply conversions
    keep_object = []
    for name, val in converters.items():
        col = arr[name]
        # Check for None efficiently
        mask = (col == None)
        if np.any(mask):
            if val is not None:
                col[mask] = val
            else:
                # no null replacement value was given, so we keep
                # the column as an object column with Nones in it
                keep_object.append(name)

    if keep_object:
        descr = np.dtype((np.record, [
            (_n, object if _n in keep_object else dtype.fields[_n][0])
            for _n in names
        ]))

    # Cast to final dtype
    retval = arr.astype(descr)
    return retval.view(numpy.recarray)


def __getDType(row, typeCodes, strLength):
    pgTypeHash = {
        16: bool,
        18: str,
        19: str,  # name type used in information schema
        20: 'i8',
        21: 'i2',
        23: 'i4',
        700: 'f4',
        701: 'f8',
        1000: bool,
        1005: 'i2',
        1007: 'i4',
        1016: 'i8',
        1021: 'f4',
        1022: 'f8',
        1700: 'f8',  # numeric
        1114: '<M8[us]',  # timestamp
        1184: '<M8[us]',  # timestamp with timezone
        1082: '<M8[us]',  # date
        25: '|U%d',
        1042: '|U%d',  # character()
        1043: '|U%d'  # varchar
    }
    strTypes = [25, 1042, 1043]

    pgTypes = []

    for i, (curv, curt) in enumerate(zip(row, typeCodes)):
        if curt not in pgTypeHash:
            raise SqlUtilException('Unknown PG type %d' % curt)
        pgType = pgTypeHash[curt]
        if curt in strTypes:
            if curv is not None:
                # if the first string value is longer than
                # strLength use that as a maximum
                curmax = max(strLength, len(curv))
            else:
                # if the first value is null
                # just use strLength
                curmax = strLength
            pgType = pgType % (curmax, )
        if curt not in strTypes:
            try:
                len(curv)
                pgType = 'O'
            except TypeError:
                pass
        pgTypes.append(('a%d' % i, pgType))
    dtype = numpy.dtype(pgTypes)
    return dtype


def __get_representative_row(batch, type_codes):
    """
    Scans the batch to find the maximum length of string columns and returns
    a representative row (based on the first row) with those maximum lengths.
    This ensures that __getDType determines the correct string length for
    the array.
    """
    first_row = list(batch[0])
    strTypes = [25, 1042, 1043]

    for i, type_code in enumerate(type_codes):
        if type_code in strTypes:
            max_len = 0
            found = False
            for row in batch:
                val = row[i]
                if val is not None:
                    max_len = max(max_len, len(val))
                    found = True
            if found:
                # Update representative row with max length string
                # __getDType uses max(strLength, len(val))
                if first_row[i] is None or len(first_row[i]) < max_len:
                    first_row[i] = 'x' * max_len
    return first_row


def __fix_object_column(col, intNullVal, strNullVal):
    """
    Convert an object column into a typed numpy array if possible,
    replacing the Nones in integer/string columns by intNullVal/strNullVal
    and by NaNs in float columns.
    The columns that cannot be converted are returned unchanged.
    """
    # Find first non-None value to guess type
    # (we avoid col != None here as that fails for columns
    # containing arrays)
    valid_mask = np.fromiter((_tmp is not None for _tmp in col),
                             dtype=bool,
                             count=len(col))
    if not np.any(valid_mask):
        # All None.
        # We can't determine type. Return as is (Object filled with
        # None)
        return col

    # Take a sample (first valid)
    first_val = col[valid_mask][0]

    # INTEGER
    if isinstance(
            first_val,
        (int, np.integer)) and not isinstance(first_val, (bool, np.bool_)):
        # Verify all valid are integers
        valid_vals = col[valid_mask]
        is_valid = all(
            isinstance(x, (int, np.integer))
            and not isinstance(x, (bool, np.bool_)) for x in valid_vals)
        if is_valid:
            # Convert
            temp_col = np.copy(col)
            temp_col[~valid_mask] = intNullVal
            try:
                return temp_col.astype(int)
            except Exception:
                pass

    # FLOAT
    if isinstance(first_val, (float, np.floating)):
        if all(
                isinstance(x, (float, np.floating, int, np.integer))
                for x in col[valid_mask]):
            temp_col = np.copy(col)
            temp_col[~valid_mask] = np.nan
            try:
                return temp_col.astype(float)
            except Exception:
                pass

    # STRING
    if isinstance(first_val, str):
        if all(isinstance(x, str) for x in col[valid_mask]):
            # Replicate PG behavior: None -> strNullVal
            temp_col = np.copy(col)
            temp_col[~valid_mask] = strNullVal
            # Let numpy infer string length
            return np.array(list(temp_col))

    return col


def __fix_numpy_record_array(res, intNullVal, strNullVal):
    if res.size == 0:
        return res

    names = res.dtype.names
    new_arrays = []

    for name in names:
        col = res[name]

        if col.dtype.kind == 'O':
            # We treat everything as candidates if it's Object type
            new_arrays.append(__fix_object_column(col, intNullVal, strNullVal))
        else:
            new_arrays.append(col)

    return numpy.rec.fromarrays(new_arrays, names=names)


def __process_duckdb_column(col, intNullVal, strNullVal):
    """
    Convert a single column returned by duckdb's fetchnumpy() into
    the sqlutilpy convention, i.e. nulls in integer/string columns
    replaced by intNullVal/strNullVal, by NaNs in float columns and
    by NaTs in datetime columns; strings converted to fixed-width arrays
    """
    if isinstance(col, np.ma.MaskedArray):
        kind = col.dtype.kind
        if kind in ('i', 'u') and intNullVal is not None:
            info = np.iinfo(col.dtype)
            if not (info.min <= intNullVal <= info.max):
                # intNullVal does not fit into the column type (e.g. a
                # negative value for an unsigned or a tinyint column),
                # so the column is widened to int64 first.
                # (we check explicitly rather than catch an exception,
                # because numpy<2 silently wraps the value instead)
                col = col.astype(np.int64)
            col = col.filled(intNullVal)
        elif kind == 'f':
            col = col.filled(np.nan)
        elif kind == 'M':
            col = col.filled(np.array('NaT', dtype=col.dtype))
        else:
            # object columns (i.e. strings), booleans and integers
            # without a null replacement value become object columns
            # with Nones
            # (note that col.filled(None) cannot be used, as None
            # there means the default fill value)
            new_col = col.data.astype(object)
            new_col[np.ma.getmaskarray(col)] = None
            col = new_col
    if col.dtype.kind == 'O':
        col = __fix_object_column(col, intNullVal, strNullVal)
    return col


def get(query,
        params=None,
        db="wsdb",
        driver="psycopg",
        user=None,
        password=None,
        host=None,
        preamb=None,
        conn=None,
        port=None,
        strLength=STRLEN_DEFAULT,
        timeout=None,
        batched=True,
        asDict=False,
        intNullVal=-9999,
        strNullVal='None',
        nthreads=1):
    """
    Execute the SQL query and return the result as a tuple or dictionary
    of numpy arrays, one array per column.

    Parameters
    ----------
    query : string
        The query to execute. It can contain placeholders for the query
        parameters: %s for PostgreSQL, ? for SQLite and DuckDB
    params : tuple, optional
        Query parameters
    db : string
        The name of the database (PostgreSQL) or the filename of the
        database (SQLite, DuckDB)
    driver : string, optional
        The database driver: 'psycopg' (PostgreSQL, the default),
        'sqlite3' or 'duckdb'
    user : string, optional
        User name for the DB connection (PostgreSQL only)
    password : string, optional
        DB connection password (PostgreSQL only)
    host : string, optional
        Hostname of the database (PostgreSQL only)
    port : integer, optional
        Port of the database (PostgreSQL only)
    preamb : string, optional
        SQL to be executed before the query
    conn : object, optional
        An existing connection to the DB (see getConnection()), to avoid
        reconnecting. The connection is left open afterwards. The driver
        argument must still match the connection
    strLength : integer, optional
        The minimum width allocated for string columns (PostgreSQL only).
        If longer strings are encountered the width is extended to
        accommodate them. With SQLite and DuckDB the width is always
        inferred from the data
    timeout : integer, optional
        How long to wait for a locked database, in seconds (SQLite only)
    batched : bool, optional
        PostgreSQL only. With the default of True the results are
        retrieved through a server-side cursor in batches of
        config.arraysize rows and converted to numpy on the fly, so the
        memory use is about the size of the final arrays. With False the
        whole result is fetched at once, which needs about twice the
        memory, but lets PostgreSQL use parallel query plans, which can be
        several times faster for heavy queries
    asDict : bool, optional
        If True, return an ordered dictionary keyed by column name instead
        of a tuple of arrays. Repeated column names get a numeric suffix
    intNullVal : integer or None, optional
        The value that replaces nulls in integer columns (-9999 by
        default). If None, integer columns with nulls are returned as
        object arrays with None in place of the nulls
    strNullVal : string, optional
        The value that replaces nulls in string columns ('None' by
        default)
    nthreads : integer, optional
        The number of threads used to convert the batches to numpy
        in the batched PostgreSQL mode

    Returns
    -------
    ret : tuple or dictionary
        A tuple with one numpy array per column of the query result, or,
        with asDict=True, an ordered dictionary mapping the column names
        to the arrays. Nulls in floating point columns become NaN, in
        timestamp columns NaT; see intNullVal and strNullVal for the
        integer and string columns

    Examples
    --------
    >>> ra, dec, d25 = sqlutilpy.get('select ra, dec, d25 from rc3',
    ...                              host='myhost', db='mydb')

    With query parameters:

    >>> ra, dec = sqlutilpy.get('select ra, dec from rc3 where name = %s',
    ...                         ('NGC 3166', ), host='myhost', db='mydb')

    From a SQLite file:

    >>> a, b = sqlutilpy.get('select a, b from tmp where a > ?', (3, ),
    ...                      db='mydata.db', driver='sqlite3')
    """

    driver = __check_driver(driver)
    connSupplied = (conn is not None)
    if not connSupplied:
        conn = getConnection(db=db,
                             driver=driver,
                             user=user,
                             password=password,
                             host=host,
                             port=port,
                             timeout=timeout)
    try:
        if batched:
            # This is related to postgresql server cursors vs
            # application cursors
            namedCursor = True
        else:
            namedCursor = False
        cur = getCursor(conn, driver=driver, preamb=preamb, named=namedCursor)

        if params is None:
            res = cur.execute(query)
        else:
            res = cur.execute(query, params)

        colNames = []
        if driver == 'psycopg':
            # Fetch first batch to determine dtype
            if batched:
                first_batch = cur.fetchmany(config.arraysize)
            else:
                first_batch = cur.fetchall()
            desc = cur.description

            # Check if we have description
            # (for non-select queries it might be None)
            if desc:
                type_codes = [_tmp.type_code for _tmp in desc]
                colNames = [_tmp.name for _tmp in desc]
            else:
                type_codes = []
                colNames = []

            if not first_batch:
                # Empty result set or no result (e.g. insert)
                if type_codes:
                    dtype = __getDType([None] * len(type_codes), type_codes,
                                       strLength)
                    res = numpy.array([], dtype=dtype)
                else:
                    # Fallback or empty
                    dtype = None
                    res = []
            else:

                def process_batch(batch):
                    # Determine dtype from the first row of the current batch
                    # This allows adapting to string lengths in later batches
                    # if the first batch had Nulls
                    if isinstance(batch, tuple):
                        batch = list(batch)

                    first_row = __get_representative_row(batch, type_codes)
                    dtype = __getDType(first_row, type_codes, strLength)
                    return __fromrecords(batch,
                                         dtype=dtype,
                                         intNullVal=intNullVal,
                                         strNullVal=strNullVal)

                def batch_iter(first):
                    yield first
                    while True:
                        batch = cur.fetchmany(config.arraysize)
                        if not batch:
                            break
                        yield batch

                if batched:
                    with concurrent.futures.ThreadPoolExecutor(
                            max_workers=nthreads) as executor:
                        results_list = list(
                            executor.map(process_batch,
                                         batch_iter(first_batch)))
                else:
                    results_list = [process_batch(first_batch)]
                # Check if results_list is valid
                if results_list:
                    res = numpy.concatenate(results_list)
                else:
                    # Should not be reached given logic above, but fallback
                    dtype = __getDType(first_batch[0], type_codes, strLength)
                    res = numpy.array([], dtype=dtype)

        elif driver == 'sqlite3':
            tups = cur.fetchall()
            if cur.description is None:
                # no result set (e.g. create/insert statement)
                colNames = []
                res = []
            else:
                colNames = [_tmp[0] for _tmp in cur.description]
                if len(tups) > 0:
                    res = numpy.rec.array(tups)
                    res = __fix_numpy_record_array(res, intNullVal, strNullVal)
                else:
                    res = [numpy.array([]) for _ in cur.description]
        elif driver == 'duckdb':
            if cur.description is None:
                # no result set
                colNames = []
                res = []
            else:
                # fetchnumpy() is much faster than fetchall() followed
                # by a conversion to numpy.
                # It returns a dict of numpy (possibly masked) arrays;
                # in the case of repeated column names the dict keys are
                # deduplicated, so we take the column names from the
                # description instead
                col_dict = cur.fetchnumpy()
                colNames = [_tmp[0] for _tmp in cur.description]
                res = [
                    __process_duckdb_column(_tmp, intNullVal, strNullVal)
                    for _tmp in col_dict.values()
                ]

        if isinstance(res, numpy.ndarray) and res.dtype.names:
            res = [res[tmp] for tmp in res.dtype.names]
        elif isinstance(res, list) and not res:
            # Handle empty result logic from original code if needed,
            # but keeping consistent with original returning array list
            pass

    except BaseException:
        failure_cleanup(conn, connSupplied)
        raise

    cur.close()

    if not connSupplied:
        conn.close()

    if asDict:
        resDict = collections.OrderedDict()
        repeats = {}
        for _n, _v in zip(colNames, res):
            if _n in resDict:
                curn = _n + '_%d' % (repeats[_n])
                repeats[_n] += 1
                warnings.warn(('Column name %s is repeated in the output, ' +
                               'new name %s assigned') % (_n, curn))
            else:
                repeats[_n] = 1
                curn = _n
            resDict[curn] = _v
        res = resDict
    return res


def execute(query,
            params=None,
            db='wsdb',
            driver="psycopg",
            user=None,
            password=None,
            host=None,
            port=None,
            conn=None,
            preamb=None,
            timeout=None,
            noCommit=False):
    """
    Execute an SQL statement without returning any results

    Parameters
    ----------
    query : string
        The statement to execute. It can contain placeholders for the
        parameters: %s for PostgreSQL, ? for SQLite and DuckDB
    params : tuple, optional
        Parameters of the statement
    db : string
        The name of the database (PostgreSQL) or the filename of the
        database (SQLite, DuckDB)
    driver : string, optional
        The database driver: 'psycopg' (PostgreSQL, the default),
        'sqlite3' or 'duckdb'
    user : string, optional
        User name for the DB connection (PostgreSQL only)
    password : string, optional
        DB connection password (PostgreSQL only)
    host : string, optional
        Hostname of the database (PostgreSQL only)
    port : integer, optional
        Port of the database (PostgreSQL only)
    conn : object, optional
        An existing connection to the DB (see getConnection()), to avoid
        reconnecting. The connection is left open afterwards
    preamb : string, optional
        SQL to be executed before the statement
    timeout : integer, optional
        How long to wait for a locked database, in seconds (SQLite only)
    noCommit : bool, optional
        By default the statement is committed. With noCommit=True the
        commit is not issued

    Examples
    --------
    >>> sqlutilpy.execute('drop table mytab', conn=conn)
    >>> sqlutilpy.execute('create table mytab (a int)', db='mydb')
    >>> sqlutilpy.execute('insert into mytab values (%s)', (1, ), db='mydb')
    """
    driver = __check_driver(driver)
    connSupplied = (conn is not None)
    if not connSupplied:
        conn = getConnection(db=db,
                             driver=driver,
                             user=user,
                             password=password,
                             host=host,
                             port=port,
                             timeout=timeout)
    try:
        cur = getCursor(conn, driver=driver, preamb=preamb, named=False)
        if params is not None:
            cur.execute(query, params)
        else:
            # sqlite3 doesn't like params here...
            cur.execute(query)
    except BaseException:
        failure_cleanup(conn, connSupplied)
        raise
    cur.close()
    if not noCommit:
        conn.commit()
    if not connSupplied:
        conn.close()  # do not close if we were given the connection


def __create_schema(tableName, arrays, names, temp=False):
    hash = dict([(np.int32, 'integer'), (np.int64, 'bigint'),
                 (np.uint64, 'bigint'), (np.int16, 'smallint'),
                 (np.uint8, 'smallint'), (np.int8, 'smallint'),
                 (np.float32, 'real'), (np.float64, 'double precision'),
                 (np.bytes_, 'varchar'), (np.str_, 'varchar'),
                 (np.bool_, 'boolean'), (np.datetime64, 'timestamp')])
    if temp:
        temp = 'temporary'
    else:
        temp = ''
    outp = 'create %s table %s ' % (temp, tableName)
    outp1 = []
    for arr, name in zip(arrays, names):
        curtyp = arr.dtype.type
        if curtyp == np.object_:
            # array
            curotyp = hash[arr[0].dtype.type] + '[]'
        else:
            curotyp = hash[arr.dtype.type]
        outp1.append(f'"{name}" {curotyp}')
    return outp + '(' + ','.join(outp1) + ')'


def __escape_strings_for_copy(arr, delimiter):
    """
    Escape the characters that are special in the text format of
    PostgreSQL's COPY FROM (backslash, newline, carriage return, tab
    and the column delimiter) in a numpy unicode string array
    """
    specials = ['\\', '\n', '\r', '\t']
    # the backslash must be escaped first
    if delimiter not in specials:
        specials.append(delimiter)
    replacements = {'\n': '\\n', '\r': '\\r', '\t': '\\t'}
    for ch in specials:
        rep = replacements.get(ch, '\\' + ch)
        arr = np.char.replace(arr, ch, rep)
    return arr


def __print_arrays(arrays, f, delimiter=' '):
    """
    print the input arrays into the open file separated by a delimiter
    """
    format_dict = dict([(np.int32, '%d'), (np.int64, '%d'), (np.int16, '%d'),
                        (np.int8, '%d'), (np.uint8, '%d'),
                        (np.float32, '%.18e'), (np.float64, '%.18e'),
                        (np.bytes_, '%s'), (np.str_, '%s'),
                        (np.datetime64, '%s'), (np.bool_, '%d')])
    fmts = []
    array_mode = False
    names = []
    for i, x in enumerate(arrays):
        names.append('f%d' % i)
        if x.dtype.type in format_dict:
            fmts.append(format_dict[x.dtype.type])
        else:
            if x.dtype.type == np.object_:
                array_mode = True
                fmts.append(None)
            else:
                raise RuntimeError(
                    'Unsupported column type %s. Please report' %
                    (x.dtype.type))
    # bytes arrays are decoded (as utf-8) into unicode arrays, since
    # otherwise they would be written as the b'...' python literals
    arrays = [
        np.char.decode(x, 'utf-8') if x.dtype.kind == 'S' else x
        for x in arrays
    ]
    arrays = [
        __escape_strings_for_copy(x, delimiter) if x.dtype.kind == 'U' else x
        for x in arrays
    ]
    recarr = np.rec.fromarrays(arrays)
    if not array_mode:
        np.savetxt(f, recarr, fmt=fmts, delimiter=delimiter)
    else:
        # this is really slow
        for row in recarr:
            for i, field in enumerate(names):
                if i != 0:
                    f.write(delimiter.encode())
                if fmts[i] is None:
                    curstr = np.array2string(
                        row[field],
                        max_line_width=np.inf,
                        threshold=None,
                        separator=',',
                        formatter={'all': lambda x: str(x)})
                    # formatter is needed because otherwise there is
                    # whitespace padding
                    curstr = '{' + curstr[1:-1] + '}'
                    f.write(curstr.encode())
                else:
                    f.write(str(row[field]).encode())
            f.write(b'\n')


def failure_cleanup(conn, connSupplied):
    try:
        conn.rollback()
    except Exception:
        pass
    if not connSupplied:
        try:
            conn.close()  # do not close if we were given the connection
        except Exception:
            pass


def upload(tableName,
           arrays,
           names=None,
           db="wsdb",
           driver="psycopg",
           user=None,
           password=None,
           host=None,
           port=None,
           conn=None,
           preamb=None,
           timeout=None,
           noCommit=False,
           temp=False,
           analyze=False,
           createTable=True,
           delimiter='|'):
    """
    Upload the data stored in a tuple of arrays (or a table) into a new
    table in the DB. PostgreSQL only.

    Parameters
    ----------
    tableName : string
        The name of the table where the data will be uploaded
    arrays : tuple, astropy.table.Table, pandas.DataFrame or dict
        A tuple of arrays that will become the columns of the table, or
        an astropy Table, pandas DataFrame or dictionary of arrays, in
        which case the column names are taken from it and the names
        argument must be omitted.
        Integer, floating point, boolean, string (unicode or bytes;
        bytes are decoded as utf-8), and datetime64 arrays are supported,
        as well as object arrays holding numpy arrays, which become
        array columns
    names : tuple, optional
        The column names (one per array). Characters that are not allowed
        in identifiers (spaces, brackets, dashes) are replaced by
        underscores, with a warning
    db : string
        Database name
    driver : string
        The database driver; only 'psycopg' is supported
    user : string, optional
        User name for the DB connection
    password : string, optional
        DB connection password
    host : string, optional
        Hostname of the database
    port : integer, optional
        Port of the database
    conn : object, optional
        An existing connection to the DB (see getConnection()), to avoid
        reconnecting. The connection is left open afterwards
    preamb : string, optional
        SQL to be executed before the upload
    timeout : integer, optional
        Unused for PostgreSQL
    noCommit : bool, optional
        If True, the commit is not executed and the table goes away
        when the connection is closed
    temp : bool, optional
        If True, a temporary table is created
    analyze : bool, optional
        If True, the table is analyzed after the upload
    createTable : bool, optional
        If True (the default), the table is created before uploading.
        With False the data is appended to an existing table
    delimiter : string, optional
        The column delimiter used when sending the data to the DB
        (the default is ``|``). Strings containing the delimiter are
        escaped, so it can be any character

    Examples
    --------
    >>> x = np.arange(10)
    >>> y = x**.5
    >>> sqlutilpy.upload('mytable', (x, y), ('xcol', 'ycol'))

    >>> T = astropy.table.Table({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
    >>> sqlutilpy.upload('mytable', T)
    """
    driver = __check_driver(driver)
    connSupplied = (conn is not None)
    if not connSupplied:
        conn = getConnection(db=db,
                             driver=driver,
                             user=user,
                             password=password,
                             host=host,
                             port=port,
                             timeout=timeout)
    if names is None:
        for i in range(1):
            # we assume that we were given a table
            if atpy is not None:
                if isinstance(arrays, atpy.Table):
                    names = arrays.columns
                    arrays = [arrays[_] for _ in names]
                    break
            if pandas is not None:
                if isinstance(arrays, pandas.DataFrame):
                    names = arrays.columns
                    arrays = [arrays[_] for _ in names]
                    break
            if isinstance(arrays, dict):
                names = arrays.keys()
                arrays = [arrays[_] for _ in names]
                break
            if names is None:
                raise Exception('you either need to give astropy \
table/pandas/dictionary or provide a separate list of arrays and their names')

    arrays = [np.asarray(_) for _ in arrays]
    if len(arrays) != len(names):
        raise RuntimeError('The column names list must have the same '
                           'length as array list')
    repl_char = {
        ' ': '_',
        '-': '_',
        '(': '_',
        ')': '_',
        '[': '_',
        ']': '_',
        '<': '_',
        '>': '_'
    }
    fixed_names = []
    for name in names:
        fixed_name = name + ''
        for k in repl_char.keys():
            fixed_name = fixed_name.replace(k, repl_char[k])
        if fixed_name != name:
            warnings.warn('''Renamed column '%s' to '%s' ''' %
                          (name, fixed_name))
        fixed_names.append(fixed_name)
    names = fixed_names
    try:
        cur = getCursor(conn, driver=driver, preamb=preamb, named=False)
        if createTable:
            query1 = __create_schema(tableName, arrays, names, temp=temp)
            cur.execute(query1)
        nsplit = 100000
        N = len(arrays[0])
        names = ','.join(['"%s"' % _ for _ in names])
        for i in range(0, N, nsplit):
            try:
                with cur.copy(f'''copy {tableName}({names}) from stdin
                        with delimiter '{delimiter}' ''') as copy:
                    __print_arrays([_[i:i + nsplit] for _ in arrays],
                                   copy,
                                   delimiter=delimiter)

            finally:
                pass
        if analyze:
            cur.execute('analyze %s' % tableName)
    except BaseException:
        failure_cleanup(conn, connSupplied)
        raise
    cur.close()
    if not noCommit:
        conn.commit()
    if not connSupplied:
        conn.close()  # do not close if we were given the connection


def local_join(query,
               tableName,
               arrays,
               names,
               db=None,
               driver="psycopg",
               user=None,
               password=None,
               host=None,
               port=None,
               conn=None,
               preamb=None,
               timeout=None,
               strLength=STRLEN_DEFAULT,
               intNullVal=-9999,
               strNullVal='None',
               asDict=False):
    """
    Join your local data in python with the data in the database.
    PostgreSQL only.

    This uploads the arrays into a temporary table and then runs the
    query, which can refer to that table. The temporary table exists
    only for the duration of the call.

    Parameters
    ----------
    query : string
        The query to be executed; it can refer to the table tableName
    tableName : string
        The name of the temporary table holding your data
    arrays : tuple
        The tuple of arrays with the data to be loaded in the DB
    names : tuple
        The column names of the temporary table (one per array)
    db : string
        Database name
    driver : string
        The database driver; only 'psycopg' is supported
    user : string, optional
        User name for the DB connection
    password : string, optional
        DB connection password
    host : string, optional
        Hostname of the database
    port : integer, optional
        Port of the database
    conn : object, optional
        An existing connection to the DB (see getConnection()), to avoid
        reconnecting. The connection is left open afterwards
    preamb : string, optional
        SQL to be executed before the query
    timeout : integer, optional
        Unused for PostgreSQL
    strLength : integer, optional
        The minimum width allocated for string columns (see get())
    intNullVal : integer or None, optional
        The value that replaces nulls in integer columns (see get())
    strNullVal : string, optional
        The value that replaces nulls in string columns (see get())
    asDict : bool, optional
        If True, return an ordered dictionary keyed by column name instead
        of a tuple of arrays

    Returns
    -------
    ret : tuple or dictionary
        The result of the query, as returned by get()

    Examples
    --------
    This extracts the rows of sometable matching the values of the
    array x:

    >>> x = np.arange(10)
    >>> y = x**.5
    >>> sqlutilpy.local_join('''
    ... select s.* from mytable as m left join sometable as s
    ... on s.x = m.x order by m.x''',
    ... 'mytable', (x, y), ('x', 'y'))
    """

    driver = __check_driver(driver)
    connSupplied = (conn is not None)
    if not connSupplied:
        conn = getConnection(db=db,
                             driver=driver,
                             user=user,
                             password=password,
                             host=host,
                             timeout=timeout,
                             port=port)
    try:
        upload(tableName,
               arrays,
               names,
               conn=conn,
               noCommit=True,
               temp=True,
               analyze=True)
    except BaseException:
        failure_cleanup(conn, connSupplied)
        raise
    try:
        res = get(query,
                  conn=conn,
                  preamb=preamb,
                  strLength=strLength,
                  asDict=asDict,
                  intNullVal=intNullVal,
                  strNullVal=strNullVal)
    except BaseException:
        failure_cleanup(conn, connSupplied)
        raise

    conn.rollback()

    if not connSupplied:
        conn.close()
    return res
