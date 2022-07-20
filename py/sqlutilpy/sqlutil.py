"""Sqlutilpy module to access SQL databases
"""
from __future__ import print_function
import numpy
import numpy as np
import psycopg2
import threading
import collections
import warnings
from numpy.core import numeric as sb
from select import select
from psycopg2.extensions import POLL_OK, POLL_READ, POLL_WRITE

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

from io import BytesIO as StringIO
import queue

_WAIT_SELECT_TIMEOUT = 10


class config:
    arraysize = 100000


class SqlUtilException(Exception):
    pass


def __wait_select_inter(conn):
    """ Make the queries interruptible by Ctrl-C

    Taken from http://initd.org/psycopg/articles/2014/07/20/cancelling-postgresql-statements-python/ # noqa
    """
    while True:
        try:
            state = conn.poll()
            if state == POLL_OK:
                break
            elif state == POLL_READ:
                select([conn.fileno()], [], [], _WAIT_SELECT_TIMEOUT)
            elif state == POLL_WRITE:
                select([], [conn.fileno()], [], _WAIT_SELECT_TIMEOUT)
            else:
                raise conn.OperationalError("bad state from poll: %s" % state)
        except KeyboardInterrupt:
            conn.cancel()
            # the loop will be broken by a server error
            continue


psycopg2.extensions.set_wait_callback(__wait_select_inter)


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
        The name of the database (in case of PostgreSQL) or filename in
        case of sqlite db
    driver :  string
        The db driver (either 'psycopg2' or 'sqlite3')
    user : string, optional
        Username
    password: string, optional
        Password
    host : string, optional
        Host-name
    port : integer
        Connection port (by default 5432 for PostgreSQL)
    timeout : integer
        Connection timeout for sqlite

    Returns
    -------
    conn : object
         Database Connection

    """
    if driver == 'psycopg2':
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
        conn = psycopg2.connect(**conn_dict)
    elif driver == 'sqlite3':
        import sqlite3
        if timeout is None:
            timeout = 5
        conn = sqlite3.connect(db, timeout=timeout)
    else:
        raise Exception("Unknown driver")
    return conn


def getCursor(conn, driver=None, preamb=None, notNamed=False):
    """
    Retrieve the database cursor
    """
    if driver == 'psycopg2':
        cur = conn.cursor()
        if preamb is not None:
            cur.execute(preamb)
        else:
            cur.execute('set cursor_tuple_fraction TO 1')
            # this is required because otherwise PG may decide to execute a
            # different plan
        if notNamed:
            return cur
        cur = conn.cursor(name='sqlutilcursor')
        cur.arraysize = config.arraysize
    elif driver == 'sqlite3':
        cur = conn.cursor()
        if preamb is not None:
            cur.execute(preamb)
    return cur


def __fromrecords(recList, dtype=None, intNullVal=None):
    """
    This function was taken from np.core.records and updated to
    support conversion null integers to intNullVal
    """

    shape = None
    descr = sb.dtype((np.core.records.record, dtype))
    try:
        retval = sb.array(recList, dtype=descr)
    except TypeError:  # list of lists instead of list of tuples
        shape = (len(recList), )
        _array = np.core.records.recarray(shape, descr)
        try:
            for k in range(_array.size):
                _array[k] = tuple(recList[k])
        except TypeError:
            convs = []
            ncols = len(dtype.fields)
            for _k in dtype.names:
                _v = dtype.fields[_k]
                if _v[0] in [np.int16, np.int32, np.int64]:
                    convs.append(lambda x: intNullVal if x is None else x)
                else:
                    convs.append(lambda x: x)
            convs = tuple(convs)

            def convF(x):
                return [convs[_](x[_]) for _ in range(ncols)]

            for k in range(k, _array.size):
                try:
                    _array[k] = tuple(recList[k])
                except TypeError:
                    _array[k] = tuple(convF(recList[k]))
        return _array
    else:
        if shape is not None and retval.shape != shape:
            retval.shape = shape

    res = retval.view(numpy.core.records.recarray)

    return res


def __converter(qIn, qOut, endEvent, dtype, intNullVal):
    """ Convert the input stream of tuples into numpy arrays """
    while (not endEvent.is_set()):
        try:
            tups = qIn.get(True, 0.1)
        except queue.Empty:
            continue
        try:
            res = __fromrecords(tups, dtype=dtype, intNullVal=intNullVal)
        except Exception:
            print('Failed to convert input data into array')
            endEvent.set()
            raise
        qOut.put(res)


def __getDType(row, typeCodes, strLength):
    pgTypeHash = {
        16: bool,
        18: str,
        19: str,  # name type used in information schema
        20: 'i8',
        21: 'i2',
        23: 'i4',
        1007: 'i4',
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


def get(query,
        params=None,
        db="wsdb",
        driver="psycopg2",
        user=None,
        password=None,
        host=None,
        preamb=None,
        conn=None,
        port=None,
        strLength=20,
        timeout=None,
        notNamed=False,
        asDict=False,
        intNullVal=-9999):
    '''
    Executes the sql query and returns the tuple or dictionary
    with the numpy arrays.

    Parameters
    ----------
    query : string
        Query you want to execute, can include question
        marks to refer to query parameters
    params : tuple
        Query parameters
    conn : object
        The connection object to the DB (optional) to avoid reconnecting
    asDict : boolean
        Flag whether to retrieve the results as a dictionary with column
        names as keys
    strLength : integer
        The maximum length of the string.
        Strings will be truncated to this length
    intNullVal : integer, optional
        All the integer columns with nulls will have null replaced by
        this value
    db : string
        The name of the database
    driver : string, optional
        The sql driver to be used (psycopg2 or sqlite3)
    user : string, optional
        User name for the DB connection
    password : string, optional
        DB connection password
    host : string, optional
        Hostname of the database
    port : integer, optional
        Port of the database
    preamb : string
        SQL code to be executed before the query

    Returns
    -------
    ret : Tuple or dictionary
        By default you get a tuple with numpy arrays for each column
        in your query.
        If you specified asDict keyword then you get an ordered dictionary with
        your columns.

    Examples
    --------
    >>> a, b, c = sqlutilpy.get('select ra,dec,d25 from rc3')

    You can also use the parameters in your query:

    >>> a, b = sqlutilpy.get('select ra,dec from rc3 where name=?', "NGC 3166")
    '''

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
        cur = getCursor(conn, driver=driver, preamb=preamb, notNamed=notNamed)

        if params is None:
            res = cur.execute(query)
        else:
            res = cur.execute(query, params)

        qIn = queue.Queue(1)
        qOut = queue.Queue()
        endEvent = threading.Event()
        nrec = 0  # keeps the number of arrays sent to the other thread
        # minus number received
        reslist = []
        proc = None
        colNames = []
        if driver == 'psycopg2':
            try:
                while (True):
                    # Iterating over the cursor, retrieving batches of results
                    # and then sending them for conversion
                    tups = cur.fetchmany()
                    desc = cur.description

                    # No more data
                    if tups == []:
                        break

                    # Send the new batch for conversion
                    qIn.put(tups)

                    # If the is just the start we need to launch the
                    # thread doing the conversion
                    if nrec == 0:
                        typeCodes = [_tmp.type_code for _tmp in desc]
                        colNames = [_tmp.name for _tmp in cur.description]
                        dtype = __getDType(tups[0], typeCodes, strLength)
                        proc = threading.Thread(target=__converter,
                                                args=(qIn, qOut, endEvent,
                                                      dtype, intNullVal))
                        proc.start()

                    # nrec is the number of batches in conversion currently
                    nrec += 1

                    # Try to retrieve one processed batch without waiting
                    # on it
                    try:
                        reslist.append(qOut.get(False))
                        nrec -= 1
                    except queue.Empty:
                        pass

                # Now we are done fetching the data from the DB, we
                # just need to assemble the converted results
                while (nrec != 0):
                    try:
                        reslist.append(qOut.get(True, 0.1))
                        nrec -= 1
                    except queue.Empty:
                        # continue looping unless the endEvent was set
                        # which should happen in the case of the crash
                        # of the converter thread
                        if endEvent.is_set():
                            raise Exception('Child thread failed')
                endEvent.set()
            except BaseException:
                endEvent.set()
                if proc is not None:
                    # notice that here the timeout is larger than the timeout
                    proc.join(0.2)
                    # in the converter process
                    if proc.is_alive():
                        proc.terminate()
                raise
            if proc is not None:
                proc.join()
            if reslist == []:
                nCols = len(desc)
                res = numpy.array([],
                                  dtype=numpy.dtype([('a%d' % i, 'f')
                                                     for i in range(nCols)]))
            else:
                res = numpy.concatenate(reslist)

        elif driver == 'sqlite3':
            tups = cur.fetchall()
            colNames = [_tmp[0] for _tmp in cur.description]
            if len(tups) > 0:
                res = numpy.core.records.array(tups)
            else:
                return [[]] * len(cur.description)

        res = [res[tmp] for tmp in res.dtype.names]

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
            driver="psycopg2",
            user=None,
            password=None,
            host=None,
            conn=None,
            preamb=None,
            timeout=None,
            noCommit=False):
    """
    Execute a given SQL command without returning the results

    Parameters
    ----------
    query: string
        The query or command you are executing
    params: tuple, optional
        Optional parameters of your query
    db : string
        Database name
    driver : string
        Driver for the DB connection ('psycopg2' or 'sqlite3')
    user : string, optional
        user name for the DB connection
    password : string, optional
        DB connection password
    host : string, optional
        Hostname of the database
    port : integer, optional
        Port of the database
    noCommit: bool
        By default execute() will commit your command.
        If you say noCommit, the commit won't be issued.

    Examples
    --------
    >>> sqlutil.execute('drop table mytab', conn=conn)
    >>> sqlutil.execute('create table mytab (a int)', db='mydb')

    """
    connSupplied = (conn is not None)
    if not connSupplied:
        conn = getConnection(db=db,
                             driver=driver,
                             user=user,
                             password=password,
                             host=host,
                             timeout=timeout)
    try:
        cur = getCursor(conn, driver=driver, preamb=preamb, notNamed=True)
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
                 (np.uint8, 'bigint'), (np.float32, 'real'),
                 (np.float64, 'double precision'), (np.string_, 'varchar'),
                 (np.str_, 'varchar'), (np.bool_, 'boolean'),
                 (np.datetime64, 'timestamp')])
    if temp:
        temp = 'temporary'
    else:
        temp = ''
    outp = 'create %s table %s ' % (temp, tableName)
    outp1 = []
    for arr, name in zip(arrays, names):
        outp1.append('"' + name + '" ' + hash[arr.dtype.type])
    return outp + '(' + ','.join(outp1) + ')'


def __print_arrays(arrays, f, delimiter=' '):
    """
    print the input arrays into the open file separated by a delimiter
    """
    format_dict = dict([(np.int32, '%d'), (np.int64, '%d'), (np.int16, '%d'),
                        (np.uint8, '%d'), (np.float32, '%.18e'),
                        (np.float64, '%.18e'), (np.string_, '%s'),
                        (np.str_, '%s'), (np.datetime64, '%s'),
                        (np.bool_, '%d')])
    fmt = [format_dict[x.dtype.type] for x in arrays]
    recarr = np.rec.fromarrays(arrays)
    np.savetxt(f, recarr, fmt=fmt, delimiter=delimiter)


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
           driver="psycopg2",
           user=None,
           password=None,
           host=None,
           conn=None,
           preamb=None,
           timeout=None,
           noCommit=False,
           temp=False,
           analyze=False,
           createTable=True,
           delimiter='|'):
    """
    Upload the data stored in the tuple of arrays in the DB

    Parameters
    ----------
    tableName : string
        The name of the table where the data will be uploaded
    arrays_or_table : tuple
        Tuple of arrays that will be columns of the new table
        If names are not specified, I this parameter can be pandas or
        astropy table
    names : tuple
    db: string
         Databas name
    driver: string
         Python database driver "psycopg2",
    user: string,
    password: string
    host: string
    conn: object
         SQL connection
    preamb: string
         The string/query to be executed before your command
    noCommit: bool
         If true, the commit is not executed and the table will go away
         after the disconnect
    temp: bool
         If true a temporary table will be created
    analyze: bool
         if True, the table will be analyzed after the upload
    createTable: bool
         if True the table will be created before uploading (default)
    delimiter: string
         the string used for delimiting the input data when ingesting into
         the db (default is |)

    Examples
    --------
    >>> x = np.arange(10)
    >>> y = x**.5
    >>> sqlutilpy.upload('mytable', (x, y), ('xcol', 'ycol'))

    >>> T = astropy.Table({'x':[1, 2, 3], 'y':['a', 'b', 'c'])
    >>> sqlutilpy.upload('mytable', T)
    """
    connSupplied = (conn is not None)
    if not connSupplied:
        conn = getConnection(db=db,
                             driver=driver,
                             user=user,
                             password=password,
                             host=host,
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
        cur = getCursor(conn, driver=driver, preamb=preamb, notNamed=True)
        if createTable:
            query1 = __create_schema(tableName, arrays, names, temp=temp)
            cur.execute(query1)
        nsplit = 100000
        N = len(arrays[0])
        for i in range(0, N, nsplit):
            f = StringIO()
            __print_arrays([_[i:i + nsplit] for _ in arrays],
                           f,
                           delimiter=delimiter)
            f.seek(0)
            try:
                thread = psycopg2.extensions.get_wait_callback()
                psycopg2.extensions.set_wait_callback(None)
                cur.copy_from(f, tableName, sep=delimiter, columns=names)
            finally:
                psycopg2.extensions.set_wait_callback(thread)
    except BaseException:
        failure_cleanup(conn, connSupplied)
        raise
    if analyze:
        cur.execute('analyze %s' % tableName)
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
               driver="psycopg2",
               user=None,
               password=None,
               host=None,
               port=None,
               conn=None,
               preamb=None,
               timeout=None,
               strLength=20,
               asDict=False):
    """
    Join your local data in python with the data in the database
    This command first uploads the data in the DB creating a temporary table
    and then runs a user specified query that can your local data.

    Parameters
    ----------
    query : String with the query to be executed
    tableName : The name of the temporary table that is going to be created
    arrays : The tuple with list of arrays with the data to be loaded in the DB
    names : The tuple with the column names for the user table

    Examples
    --------
    This will extract the rows from the table sometable matching
    to the provided array x

    >>> x = np.arange(10)
    >>> y = x**.5
    >>> sqlutilpy.local_join('''
    ... SELECT s.* FROM mytable AS m LEFT JOIN sometable AS s
    ... ON s.x = m.x ORDER BY m.xcol''',
    ... 'mytable', (x, y), ('x', 'y'))
    """

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
                  asDict=asDict)
    except BaseException:
        failure_cleanup(conn, connSupplied)
        raise

    conn.rollback()

    if not connSupplied:
        conn.close()
    return res
