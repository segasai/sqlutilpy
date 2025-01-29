import pytest
import string
import os
import psycopg
import numpy as np
import sqlutilpy as sqlutil
import time
import killer

PG_DB = os.environ['SQLUTIL_TEST_PG_DB']
PG_HOST = os.environ['SQLUTIL_TEST_PG_HOST']
PG_USER = os.environ['SQLUTIL_TEST_PG_USER']
PG_PASS = os.environ['SQLUTIL_TEST_PG_PASS']


def getrand(N, float=False):
    # simple deterministic pseudo-random number generator
    a, c = 1103515245, 12345
    m = 2**31
    arr = np.zeros(N)
    arr[0] = 1
    for i in range(1, N):
        arr[i] = (a * arr[i - 1] + c) % m
    if float:
        return arr * 1. / m
    else:
        return arr


def getconn():
    return sqlutil.getConnection(host=PG_HOST,
                                 user=PG_USER,
                                 db=PG_DB,
                                 password=PG_PASS,
                                 driver='psycopg2')


@pytest.fixture
def setup():
    conn = psycopg.connect('dbname=%s user=%s host=%s password=%s' %
                           (PG_DB, PG_USER, PG_HOST, PG_PASS))
    cur = conn.cursor()
    cur.execute('''
    create unlogged table sqlutil_test (sicol smallint, intcol int,
    bigicol bigint,
    realcol real, dpcol double precision, timecol timestamp,
    textcol varchar, boolcol bool);
    insert into sqlutil_test (sicol, intcol, bigicol, realcol, dpcol, timecol,
    textcol, boolcol)
    values( 1,2,3,4.,5.,'2018-01-01 10:00:00','tester1', true);
    insert into sqlutil_test (sicol, intcol, bigicol, realcol, dpcol, timecol,
    textcol, boolcol)
    values( 11,12,13,14.,15.,'2018-02-01 10:00:00','tester2', false);
    ''')
    conn.commit()
    kw = {'host': PG_HOST, 'user': PG_USER, 'db': PG_DB, 'password': PG_PASS}

    yield kw, conn
    conn.cursor().execute('drop table sqlutil_test;')
    conn.commit()


def test_getConn():
    conn = sqlutil.getConnection(host=PG_HOST,
                                 user=PG_USER,
                                 db=PG_DB,
                                 password=PG_PASS,
                                 driver='psycopg2')
    conn.close()
    pass


def test_getConnEnv():
    os.environ['PGHOST'] = PG_HOST
    os.environ['PGUSER'] = PG_USER
    os.environ['PGDATABASE'] = PG_DB
    conn = sqlutil.getConnection(driver='psycopg2', password=PG_PASS)
    conn.close()


def test_getConnWithPort():
    conn = sqlutil.getConnection(host=PG_HOST,
                                 user=PG_USER,
                                 db=PG_DB,
                                 port=5432,
                                 password=PG_PASS,
                                 driver='psycopg2')
    conn.close()


def test_getConnFail():
    with pytest.raises(Exception):
        sqlutil.getConnection(host=PG_HOST,
                              user=PG_USER,
                              db=PG_DB,
                              driver='psycopgXX')


def test_execute():
    conn = getconn()
    sqlutil.execute('create temp table sqlutil_test1 (a int) ', conn=conn)
    sqlutil.execute('insert into sqlutil_test1 (a) values(%s)', (1, ),
                    conn=conn)
    sqlutil.execute('drop table sqlutil_test1;', conn=conn)


def test_execute_fail(setup):
    kw, conn = setup
    with pytest.raises(Exception):
        sqlutil.execute('create xtable;', **kw)


def test_nocommit(setup):
    kw, conn = setup
    sqlutil.execute('create unlogged table sqlutil_test1 (a int)', conn=conn)
    sqlutil.execute('insert into sqlutil_test1 values(1)', conn=conn)
    sqlutil.execute('insert into sqlutil_test1 values(2)', noCommit=True, **kw)

    cnt, = sqlutil.get('select count(*) from sqlutil_test1', conn=conn)
    assert (cnt[0] == 1)
    sqlutil.execute('drop table sqlutil_test1;', conn=conn)


def test_local_join(setup):
    kw, conn = setup
    N = 1000
    rands = np.random.uniform(size=N)
    R, = sqlutil.local_join(
        '''
    select xf from sqlutil_test as s,  mytab as m
    where s.sicol = m.id''', 'mytab', [np.arange(N), rands], ['id', 'xf'],
        **kw)
    assert (len(R) == 2)


def test_big(setup):
    conn = getconn()
    sqlutil.execute(
        '''create temp table sqlutil_test_big (a int, b double precision);
        insert into sqlutil_test_big select generate_series,generate_series*2
        from generate_series(1,10000000);
    ''',
        conn=conn)

    a, b = sqlutil.get('select a,b from sqlutil_test_big;', conn=conn)
    sqlutil.execute('drop table sqlutil_test_big;', conn=conn)
    assert (len(a) == 10000000)


def notest_big_interrupt():
    # temporary disabled, as I cannot deal with sigint...
    conn = getconn()
    killer.interrupt(2)
    t1 = time.time()
    with pytest.raises(Exception):
        a, b = sqlutil.get(
            '''select generate_series,generate_series*2,generate_series*3
        from generate_series(1,10000000);''',
            conn=conn)
    t2 = time.time()
    assert ((t2 - t1) < 5)  # check interrupting work


def test_NoResults(setup):
    kw, conn = setup
    a, b = sqlutil.get('select 1, 2 where 2<1', **kw)
    assert (len(a) == 0)


def test_params(setup):
    kw, conn = setup
    xid = 5
    a, b = sqlutil.get(
        '''
    select * from (values (0,1),(10,20)) as x where column1<%s;''', (xid, ),
        **kw)
    assert (len(a) == 1)


def test_Preamb(setup):
    kw, conn = setup
    a, b = sqlutil.get('select 1, 2 where 2<1',
                       preamb='set enable_seqscan to off',
                       **kw)
    assert (len(a) == 0)


def test_StringFirstNull(setup):
    kw, conn = setup
    a, = sqlutil.get(''' values(NULL), ('abcdef')''', **kw)
    assert (len(a) == 2)
    assert (a[1] == 'abcdef')


def test_get(setup):
    kw, conn = setup
    a, b, c, d, e, f, g = sqlutil.get(
        '''select sicol,intcol,bigicol,realcol,dpcol,textcol,boolcol
        from sqlutil_test order by sicol''', **kw)
    assert ((a == np.array([1, 11])).all())
    assert ((b == np.array([2, 12])).all())
    assert ((c == np.array([3, 13])).all())
    assert ((d == np.array([4, 14])).all())
    assert ((e == np.array([5, 15])).all())
    assert ((f == np.array(['tester1', 'tester2'])).all())
    assert ((g == np.array([True, False])).all())


def test_Null(setup):
    kw, conn = setup
    a, b = sqlutil.get('values ( 1, 2.) , ( NULL, NULL) ', **kw)
    assert (np.isnan(b[1]))
    assert (a[1] == -9999)
    a, b = sqlutil.get('values (NULL,NULL), ( 1, 2.)', **kw)
    assert (np.isnan(b[0]))
    assert (a[0] == -9999)


def test_Array(setup):
    kw, conn = setup
    a, b = sqlutil.get('values ( ARRAY[1,2], 2.) , ( ARRAY[3,4], 3.) ', **kw)
    assert (a[0][0] == 1)
    assert (a[0][1] == 2)
    assert (a[1][0] == 3)
    assert (a[1][1] == 4)


def test_Array2(setup):
    kw, conn = setup
    a, b, c, d, e, f = sqlutil.get(
        '''values ( ARRAY[1::int,2::int],ARRAY[11::real,12::real],
        ARRAY[21::double precision,22::double precision],
        ARRAY[31::smallint,32::smallint], ARRAY[41::bigint, 42::bigint],
        ARRAY[true,false]) ,
        ( ARRAY[3::int,4::int], ARRAY[13::real,14::real],
        ARRAY[23::double precision, 24::double precision],
        ARRAY[33::smallint, 34::smallint], ARRAY[43::bigint, 44::bigint],
        ARRAY[false,false]) ''', **kw)
    assert (a[0][0] == 1)
    assert (b[0][0] == 11)
    assert (c[0][0] == 21)
    assert (d[0][0] == 31)
    assert (e[0][0] == 41)
    assert (f[0][0])
    assert (not f[1][0])


def test_get_dict(setup):
    kw, conn = setup
    cols = 'sicol,intcol,bigicol,realcol,dpcol,textcol'
    R0 = sqlutil.get('select %s from sqlutil_test order by sicol' % (cols, ),
                     **kw)
    Rd = sqlutil.get('select %s from sqlutil_test order by sicol' % (cols, ),
                     asDict=True,
                     **kw)
    for i, k in enumerate(cols.split(',')):
        assert ((Rd[k] == R0[i]).all())
    Rempty = sqlutil.get('select %s from sqlutil_test where sicol<-10000' %
                         (cols, ),
                         asDict=True,
                         **kw)
    for i, k in enumerate(cols.split(',')):
        assert k in Rempty


def test_get_dict_rep(setup):
    kw, conn = setup
    cols = 'sicol,sicol'
    R0 = sqlutil.get('select %s from sqlutil_test order by sicol' % (cols, ),
                     asDict=True,
                     **kw)
    assert (len(R0) == 2)


def test_error(setup):
    kw, conn = setup
    with pytest.raises(Exception):
        sqlutil.get('select 1/0 from sqlutil_test '**kw)


def test_error1(setup):
    kw, conn = setup
    with pytest.raises(sqlutil.SqlUtilException):
        sqlutil.get('''select '1'::bytea from sqlutil_test ''', **kw)


def test_version():
    sqlutil.__version__


def test_upload(setup):
    kw, conn = setup
    mytab = 'sqlutil_test_tab'
    nrows = 10
    xi16 = np.arange(nrows, dtype=np.int16)
    xi32 = np.arange(nrows, dtype=np.int32)
    xi64 = np.arange(nrows, dtype=np.int64)
    xf = getrand(nrows, True)
    xf32 = xf.astype(np.float32)
    xf64 = xf.astype(np.float64)
    strL = 15
    xstr = [
        ''.join(np.random.choice(list(string.ascii_letters), strL))
        for i in range(nrows)
    ]
    xstr[1] = ',:;'  # just to test delimiting
    xstr = np.array(xstr)
    xbool = np.arange(len(xi16)) < (len(xi16) / 2.)

    for i in range(2):
        if i == 0:
            sqlutil.upload(
                mytab, (xi16, xi32, xi64, xf32, xf64, xbool, xstr),
                ('xi16', 'xi32', 'xi64', 'xf32', 'xf64', 'xbool', 'xstr'),
                **kw)
        elif i == 1:
            sqlutil.upload(
                mytab, (xi16, xi32, xi64, xf32, xf64, xbool, xstr),
                ('xi16', 'xi32', 'xi64', 'xf32', 'xf64', 'xbool', 'xstr'),
                delimiter='*',
                **kw)
        yi16, yi32, yi64, yf32, yf64, ybool, ystr = sqlutil.get(
            '''select xi16,xi32,xi64,xf32,xf64,xbool,xstr from %s''' % (mytab),
            **kw)
        try:
            assert ((xi16 == yi16).all())
            assert ((xi32 == yi32).all())
            assert ((xi64 == yi64).all())
            assert (np.allclose(xf32, yf32))
            assert (np.allclose(xf64, yf64))
            assert ((ybool == xbool).all())
            assert ((ystr == xstr).all())
        finally:
            sqlutil.execute('drop table %s' % mytab, **kw)

    # test astropy table ingestion
    import astropy.table as atpy
    astroTab = atpy.Table({
        'xi16': xi16,
        'xi32': xi32,
        'xi64': xi64,
        'xf32': xf32,
        'xf64': xf64,
        'xbool': xbool
    })
    sqlutil.upload(mytab, astroTab, **kw)
    yi16, yi32, yi64, yf32, yf64, ybool = sqlutil.get(
        '''select xi16,xi32,xi64,xf32,xf64,xbool from %s''' % (mytab), **kw)
    try:
        assert ((xi16 == yi16).all())
        assert ((xi32 == yi32).all())
        assert ((xi64 == yi64).all())
        assert (np.allclose(xf32, yf32))
        assert (np.allclose(xf64, yf64))
        assert ((ybool == xbool).all())
    finally:
        sqlutil.execute('drop table %s' % mytab, **kw)
    sqlutil.upload(mytab, astroTab.to_pandas(), **kw)
    yi16, yi32, yi64, yf32, yf64, ybool = sqlutil.get(
        '''select xi16,xi32,xi64,xf32,xf64,xbool from %s''' % (mytab), **kw)
    try:
        assert ((xi16 == yi16).all())
        assert ((xi32 == yi32).all())
        assert ((xi64 == yi64).all())
        assert (np.allclose(xf32, yf32))
        assert (np.allclose(xf64, yf64))
        assert ((ybool == xbool).all())
    finally:
        sqlutil.execute('drop table %s' % mytab, **kw)

    sqlutil.upload(
        mytab, {
            'xi16': xi16,
            'xi32': xi32,
            'xi64': xi64,
            'xf32': xf32,
            'xf64': xf64,
            'xbool': xbool
        }, **kw)
    yi16, yi32, yi64, yf32, yf64, ybool = sqlutil.get(
        '''select xi16,xi32,xi64,xf32,xf64,xbool from %s''' % (mytab), **kw)
    try:
        assert ((xi16 == yi16).all())
        assert ((xi32 == yi32).all())
        assert ((xi64 == yi64).all())
        assert (np.allclose(xf32, yf32))
        assert (np.allclose(xf64, yf64))
        assert ((ybool == xbool).all())
    finally:
        sqlutil.execute('drop table %s' % mytab, **kw)

    with pytest.raises(Exception):
        sqlutil.upload(' a b c d', (xi16, xi32, xi64, xf32, xf64, xbool),
                       ('xi16', 'xi32', 'xi64', 'xf32', 'xf64', 'xbool'), **kw)
        # test exception handling

    with pytest.raises(Exception):
        sqlutil.upload(mytab, xi16, **kw)

    mytab1 = 'sqlutil_test_tab1'
    # try the weird names
    sqlutil.upload(mytab1, (xi16, xi32, xi64, xf32, xf64, xbool),
                   ('xi 16', 'xi(32)', 'xi[64]', 'xf32', 'xf64', 'xbool'),
                   temp=True,
                   conn=conn)
    yi16, yi32, yi64, yf32, yf64, ybool = sqlutil.get(
        '''select xi_16,xi_32_,xi_64_,xf32,xf64,xbool from %s''' % (mytab1),
        conn=conn)


def test_upload_big(setup):
    kw, conn = setup
    mytab = 'sqlutil_test_tab_big'
    nrows = 1_000_000
    xi = np.arange(nrows, dtype=np.int32)
    xf = getrand(nrows, True)
    sqlutil.upload(mytab, (xi, xf), ('xi', 'xf'), **kw)
    yi, yf = sqlutil.get('''select xi,xf from %s order by xi''' % (mytab),
                         **kw)
    try:
        assert ((xi == yi).all())
        assert (np.allclose(xf, yf))
    finally:
        sqlutil.execute('drop table %s' % mytab, **kw)


def test_upload_array(setup):
    kw, conn = setup
    mytab = 'sqlutil_test_tab'
    nrows = 10
    xi16 = np.arange(nrows, dtype=np.int16)
    xi32 = np.arange(nrows, dtype=np.int32)
    xi64 = np.arange(nrows, dtype=np.int64)
    xf = getrand(nrows, True)
    xf32 = xf.astype(np.float32)
    xf64 = xf.astype(np.float64)
    xbool = np.arange(len(xi16)) < (len(xi16) / 2.)
    xarr = np.array([np.arange(_ + 1) for _ in range(10)], dtype='object')
    for i in range(1):
        if i == 0:
            sqlutil.upload(
                mytab, (xi32, xi64, xf32, xf64, xbool, xarr, xi16),
                ('xi32', 'xi64', 'xf32', 'xf64', 'xbool', 'xarr', 'xi16'),
                **kw)
        yi16, yi32, yi64, yf32, yf64, ybool, yarr = sqlutil.get(
            '''select xi16,xi32,xi64,xf32,xf64,xbool,xarr from %s''' % (mytab),
            **kw)
        try:
            for i in range(nrows):
                assert (np.all(xarr[i] == yarr[i]))
        finally:
            sqlutil.execute('drop table %s' % mytab, **kw)
