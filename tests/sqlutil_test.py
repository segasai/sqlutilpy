import unittest
import os
import psycopg2
import sqlutilpy as sqlutil
PG_DB = os.environ['SQLUTIL_TEST_PG_DB']
PG_HOST = os.environ['SQLUTIL_TEST_PG_HOST']
PG_USER = os.environ['SQLUTIL_TEST_PG_USER']

class PostgresTest(unittest.TestCase):
    def setUp(self):
        conn = psycopg2.connect('dbname=%s user=%s host=%s'%(
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
        values( 1,2,3,4.,5.,'2018-01-01 10:00:00','tester');
insert into sqlutil_test (sicol, intcol, bigicol, realcol, dpcol, timecol,
textcol)
        values( 11,12,13,14.,15.,'2018-02-01 10:00:00','tester');
        ''')
        conn.commit()
        self.conn = conn;

    def tearDown(self):
        self.conn.cursor().execute('drop table sqlutil_test;')
        self.conn.commit()

    def test_getConn(self):
        conn = sqlutil.getConnection(host=PG_HOST, user=PG_USER, db=PG_DB,
                                     driver='psycopg2')
        conn.close()
        pass

    def test_get(self):
        kw = {'host':PG_HOST,'user':PG_USER,'db':PG_DB}
        a,b,c,d = sqlutil.get('select sicol,intcol,realcol,dpcol from sqlutil_test',**kw)
        

if __name__=='__main__':
    unittest.main()
