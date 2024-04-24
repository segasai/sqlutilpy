import pytest
import os
import sqlutilpy as sqlutil
import sqlite3


@pytest.fixture()
def setup():
    fname = 'sql.db'
    kw = dict(db=fname, driver='sqlite3')
    conn = sqlite3.dbapi2.Connection(fname)
    cur = conn.cursor()
    cur.execute('create table tmp (a int, b double precision)')
    cur.execute('insert into tmp values(1,2), (2,3), (3,4), (4,5);')
    conn.commit()
    yield kw
    os.unlink(fname)


def test_simple(setup):
    a, b = sqlutil.get('select a,b from tmp', **setup)
    assert (len(a) == 4)


def test_preamb(setup):
    a, b = sqlutil.get('select a,b from tmp', preamb='select 1', **setup)
    assert (len(a) == 4)


def test_empty(setup):
    a, b = sqlutil.get('select a,b from tmp where a<0', **setup)
    assert (len(a) == 0)
