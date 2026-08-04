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


def test_no_result_set(setup):
    # statements that do not return a result set should not crash
    res = sqlutil.get('create table tmp2 (a int)', **setup)
    assert res == []


def test_intnull_none(setup):
    conn = sqlite3.connect('sql.db')
    cur = conn.cursor()
    cur.execute('insert into tmp values (5, NULL), (NULL, 7)')
    conn.commit()
    conn.close()
    a, b = sqlutil.get('select a, b from tmp where a is NULL or b is NULL',
                       intNullVal=None,
                       **setup)
    assert a.dtype.kind == 'O'
    assert a[0] == 5
    assert a[1] is None
