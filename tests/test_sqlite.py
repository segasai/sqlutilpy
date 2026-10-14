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


def test_unknown_driver(setup):
    # an unknown driver must be rejected before any connection attempt,
    # both with and without a user supplied connection
    import numpy as np
    conn = sqlite3.connect('sql.db')
    for kw in [dict(db='sql.db'), dict(conn=conn)]:
        with pytest.raises(sqlutil.SqlUtilException):
            sqlutil.get('select 1', driver='nosuchdriver', **kw)
        with pytest.raises(sqlutil.SqlUtilException):
            sqlutil.execute('select 1', driver='nosuchdriver', **kw)
        with pytest.raises(sqlutil.SqlUtilException):
            sqlutil.upload('tmp3', (np.arange(2), ), ('a', ),
                           driver='nosuchdriver',
                           **kw)
    with pytest.raises(sqlutil.SqlUtilException):
        sqlutil.getCursor(conn, driver='nosuchdriver')
    with pytest.raises(sqlutil.SqlUtilException):
        sqlutil.getConnection(db='sql.db', driver='nosuchdriver')
    conn.close()


def test_notNamed_deprecated(setup):
    # notNamed is a deprecated alias of batched=not notNamed
    with pytest.warns(DeprecationWarning, match='notNamed'):
        a, b = sqlutil.get('select a,b from tmp', notNamed=True, **setup)
    assert (len(a) == 4)
    conn = sqlite3.connect('sql.db')
    with pytest.warns(DeprecationWarning, match='notNamed'):
        sqlutil.getCursor(conn, driver='sqlite3', notNamed=True)
    conn.close()


def test_keyword_only(setup):
    # the options can only be given as keywords
    import numpy as np
    with pytest.raises(TypeError):
        sqlutil.get('select a,b from tmp', None, 'sql.db', 'sqlite3')
    with pytest.raises(TypeError):
        sqlutil.execute('select 1', None, 'sql.db', 'sqlite3')
    with pytest.raises(TypeError):
        sqlutil.upload('tmp4', (np.arange(2), ), ('a', ), 'sql.db',
                       'sqlite3')
    with pytest.raises(TypeError):
        sqlutil.local_join('select 1', 'tmp5', (np.arange(2), ), ('a', ),
                           'sql.db', 'sqlite3')
    with pytest.raises(TypeError):
        sqlutil.getConnection('sql.db', 'sqlite3')
    conn = sqlite3.connect('sql.db')
    with pytest.raises(TypeError):
        sqlutil.getCursor(conn, 'sqlite3')
    conn.close()
