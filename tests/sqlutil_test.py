import unittest
import os
import psycopg2
import numpy as np
import sqlutilpy as sqlutil
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


class PostgresTest(unittest.TestCase):
    def setUp(self):
        conn = psycopg2.connect('dbname=%s user=%s host=%s' % (
            PG_DB, PG_USER, PG_HOST
        ))
        cur = conn.cursor()
        cur.execute(
            '''
create table sqlutil_test (sicol smallint, intcol int, bigicol bigint,
        realcol real, dpcol double precision, timecol timestamp, 
textcol varchar);
insert into sqlutil_test (sicol, intcol, bigicol, realcol, dpcol, timecol,
textcol)
        values( 1,2,3,4.,5.,'2018-01-01 10:00:00','tester1');
insert into sqlutil_test (sicol, intcol, bigicol, realcol, dpcol, timecol,
textcol)
        values( 11,12,13,14.,15.,'2018-02-01 10:00:00','tester2');
        ''')
        conn.commit()
        self.conn = conn
        self.kw = {'host': PG_HOST, 'user': PG_USER, 'db': PG_DB}

    def tearDown(self):
        self.conn.cursor().execute('drop table sqlutil_test;')
        self.conn.commit()

    def test_getConn(self):
        conn = sqlutil.getConnection(host=PG_HOST, user=PG_USER, db=PG_DB,
                                     driver='psycopg2')
        conn.close()
        pass

    def test_get(self):
        a, b, c, d, e, f = sqlutil.get(
            'select sicol,intcol,bigicol,realcol,dpcol,textcol from sqlutil_test order by sicol', **self.kw)
        self.assertTrue((a == np.array([1, 11])).all())
        self.assertTrue((b == np.array([2, 12])).all())
        self.assertTrue((c == np.array([3, 13])).all())
        self.assertTrue((d == np.array([4, 14])).all())
        self.assertTrue((e == np.array([5, 15])).all())
        self.assertTrue((f == np.array(['tester1', 'tester2'])).all())

    def test_Null(self):
        a, b = sqlutil.get('values ( 1, 2.) , ( NULL, NULL) ', **self.kw)
        self.assertTrue(np.isnan(b[1]))
        self.assertTrue(a[1] == -9999)
        a, b = sqlutil.get('values (NULL,NULL), ( 1, 2.)', **self.kw)
        self.assertTrue(np.isnan(b[0]))
        self.assertTrue(a[0] == -9999)

    def test_get_dict(self):
        R0 = sqlutil.get(
            'select sicol,intcol,bigicol,realcol,dpcol,textcol from sqlutil_test order by sicol', asDict=True, **self.kw)
        Rd = sqlutil.get(
            'select sicol,intcol,bigicol,realcol,dpcol,textcol from sqlutil_test order by sicol', asDict=True, **self.kw)
        for i, k in enumerate('sicol,intcol,bigicol,realcol,dpcol,textcol'.split(',')):
            self.assertTrue((Rd[k] == R0[i]).all())

    def test_upload(self):
        mytab = 'sqlutil_test_tab'
        xi = np.arange(10)
        xf = getrand(10, True)
        sqlutil.upload(mytab, (xi, xf), ('xcol', 'ycol'), **self.kw)
        yi, yf = sqlutil.get('select xcol,ycol from %s' % (mytab), **self.kw)
        self.assertTrue((xi == yi).all())
        self.assertTrue((xi == yi).all())
        sqlutil.execute('drop table %s' % mytab, **self.kw)


if __name__ == '__main__':
    unittest.main()
