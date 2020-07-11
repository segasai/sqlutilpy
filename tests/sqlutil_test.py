import pytest
import os
import psycopg2
import numpy as np
import sqlutilpy as sqlutil
import sqlite3

PG_DB = os.environ['SQLUTIL_TEST_PG_DB']
PG_HOST = os.environ['SQLUTIL_TEST_PG_HOST']
PG_USER = os.environ['SQLUTIL_TEST_PG_USER']


def getrand(N, float=False):
    # simple deterministic pseudo-random number generator
    a, c = 1103515245, 12345
    m = 2 ^ 31
    arr = np.zeros(N)
    arr[0] = 1
    for i in range(1, N):
        arr[i] = (a*arr[i-1]+c) % m
    if float:
        return arr*1./m
    else:
        return arr


class TestPostgres:
    def setup(self):
        conn = psycopg2.connect('dbname=%s user=%s host=%s' % (
            PG_DB, PG_USER, PG_HOST
        ))
        cur = conn.cursor()
        cur.execute(
            '''
create unlogged table sqlutil_test (sicol smallint, intcol int, bigicol bigint,
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
        self.conn = conn
        self.kw = {'host': PG_HOST, 'user': PG_USER, 'db': PG_DB}

    def teardown(self):
        self.conn.cursor().execute('drop table sqlutil_test;')
        self.conn.commit()

    def test_getConn(self):
        conn = sqlutil.getConnection(host=PG_HOST, user=PG_USER, db=PG_DB,
                                     driver='psycopg2')
        conn.close()
        pass

    def test_getConnUser(self):
        conn = sqlutil.getConnection(host=PG_HOST, db=PG_DB,
                                     driver='psycopg2', user='testuser',
                                     password='testpassword')
        conn.close()
        pass

    def test_getConn(self):
        with pytest.raises(Exception):
            conn = sqlutil.getConnection(host=PG_HOST, user=PG_USER, db=PG_DB,
                                     driver='psycopgXX')

    def test_execute(self):
        sqlutil.execute('create table aa (a int) ' ,**self.kw);
        sqlutil.execute('drop table aa;',**self.kw);

    def test_nocommit(self):
        sqlutil.execute('create table aa (a int)' ,**self.kw)
        sqlutil.execute('insert into aa values(1)' ,**self.kw)
        sqlutil.execute('insert into aa values(2)' ,noCommit=True, **self.kw)
        cnt,=sqlutil.get('select count(*) from aa', **self.kw)
        assert(cnt[0]==1)
        sqlutil.execute('drop table aa;', **self.kw)
    
    def test_local_join(self):
        R,=sqlutil.local_join('''
        select s.sicol from sqlutil_test as s,  mytab as m 
        where s.sicol = m.id''',
                    'mytab', [np.arange(10)], ['id'], **self.kw)
        assert(len(R)==1)


    def test_big(self):
        sqlutil.execute(
        '''create unlogged table sqlutil_big (a int, b double precision);
        insert into sqlutil_big select generate_series,generate_series*2 from generate_series(1,10000000);
        ''',**self.kw)

        a, b = sqlutil.get('select a,b from sqlutil_big;', **self.kw)
        sqlutil.execute('drop table sqlutil_big;',**self.kw)
        assert (len(a) == 10000000)

    def test_NoResults(self):
        a, b = sqlutil.get(
            'select 1, 2 where 2<1', **self.kw)
        assert(len(a) == 0)

    def test_Preamb(self):
        a, b = sqlutil.get(
            'select 1, 2 where 2<1', preamb='set enable_seqscan to off',
            **self.kw)
        assert(len(a) == 0)

    def test_StringFirstNull(self):
        a, = sqlutil.get(''' values(NULL), ('abcdef')''', **self.kw)
        assert(len(a) == 2)
        assert(a[1] == 'abcdef')

    def test_get(self):
        a, b, c, d, e, f, g = sqlutil.get(
            'select sicol,intcol,bigicol,realcol,dpcol,textcol,boolcol from sqlutil_test order by sicol', **self.kw)
        assert((a == np.array([1, 11])).all())
        assert((b == np.array([2, 12])).all())
        assert((c == np.array([3, 13])).all())
        assert((d == np.array([4, 14])).all())
        assert((e == np.array([5, 15])).all())
        assert((f == np.array(['tester1', 'tester2'])).all())
        assert((g == np.array([True, False])).all())        

    def test_Null(self):
        a, b = sqlutil.get('values ( 1, 2.) , ( NULL, NULL) ', **self.kw)
        assert(np.isnan(b[1]))
        assert(a[1] == -9999)
        a, b = sqlutil.get('values (NULL,NULL), ( 1, 2.)', **self.kw)
        assert(np.isnan(b[0]))
        assert(a[0] == -9999)

    def test_Array(self):
        a, b = sqlutil.get(
            'values ( ARRAY[1,2], 2.) , ( ARRAY[3,4], 3.) ', **self.kw)
        assert(a[0][0] == 1)
        assert(a[0][1] == 2)
        assert(a[1][0] == 3)
        assert(a[1][1] == 4)

    def test_Array2(self):
        a, b,c,d,e = sqlutil.get(
            '''values ( ARRAY[1::int,2::int],ARRAY[11::real,12::real],ARRAY[21::double precision,22::double precision], ARRAY[31::smallint,32::smallint], ARRAY[41::bigint, 42::bigint]) ,
            ( ARRAY[3::int,4::int], ARRAY[13::real,14::real],ARRAY[23::double precision, 24::double precision], ARRAY[33::smallint, 34::smallint], ARRAY[43::bigint, 44::bigint]) ''', **self.kw)
        assert(a[0][0] == 1)
        assert(b[0][0] == 11)
        assert(c[0][0] == 21)
        assert(d[0][0] == 31)
        assert(e[0][0] == 41)

    def test_get_dict(self):
        cols = 'sicol,intcol,bigicol,realcol,dpcol,textcol'
        R0 = sqlutil.get(
            'select %s from sqlutil_test order by sicol' % (cols,), **self.kw)
        Rd = sqlutil.get(
            'select %s from sqlutil_test order by sicol' % (cols,), asDict=True, **self.kw)
        for i, k in enumerate(cols.split(',')):
            assert((Rd[k] == R0[i]).all())

    def test_get_dict_rep(self):
        cols = 'sicol,sicol'
        R0 = sqlutil.get(
            'select %s from sqlutil_test order by sicol' % (cols,), asDict=True,
            **self.kw)
        assert(len(R0)==2)

    def test_error(self):
        cols = 'sicol,sicol'
        with pytest.raises(Exception):
            R0 = sqlutil.get(
                'select 1/0 from sqlutil_test ' 
                **self.kw)
        
    def test_version(self):
        VER = sqlutil.__version__

    def test_upload(self):
        mytab = 'sqlutil_test_tab'
        xi16 = np.arange(10,dtype=np.int16)
        xi32 = np.arange(10,dtype=np.int32)
        xi64 = np.arange(10,dtype=np.int64)
        xf = getrand(10, True)
        xf32 = xf.astype(np.float32)
        xf64 = xf.astype(np.float64)
        xbool = np.arange(len(xi16))<(len(xi16)/2.)
        sqlutil.upload(mytab, (xi16, xi32, xi64, xf32, xf64, xbool), 
                       ('xi16',
                        'xi32',
                        'xi64',
                        'xf32',
                        'xf64',
                        'xbool'), **self.kw)
        yi16,yi32,yi64, yf32,yf64,ybool = sqlutil.get(
            '''select xi16,xi32,xi64,xf32,xf64,xbool from %s''' % (mytab), **self.kw)
        try:
            assert((xi16 == yi16).all())
            assert((xi32 == yi32).all())
            assert((xi64 == yi64).all())
            assert(np.allclose(xf32, yf32))
            assert(np.allclose(xf64, yf64))
            assert((ybool==xbool).all())
        finally:
            sqlutil.execute('drop table %s' % mytab, **self.kw)


class TestSQLite:
    def setup(self):
        self.fname = 'sql.db'
        self.kw = dict(db=self.fname, driver='sqlite3')
        conn = sqlite3.dbapi2.Connection(self.fname)
        cur = conn.cursor()
        cur.execute('create table tmp (a int, b double precision)')
        cur.execute('insert into tmp values(1,2), (2,3), (3,4), (4,5);')
        conn.commit()

    def testSimple(self):
        a, b = sqlutil.get('select a,b from tmp', **self.kw)
        assert(len(a) == 4)

    def teardown(self):
        os.unlink(self.fname)
