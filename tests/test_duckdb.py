import pytest
import os
import numpy as np
import sqlutilpy as sqlutil
import duckdb


@pytest.fixture()
def setup():
    fname = 'duck.db'
    kw = dict(db=fname, driver='duckdb')
    conn = duckdb.connect(fname)
    cur = conn.cursor()
    cur.execute('create table tmp (a int, b double precision)')
    cur.execute('insert into tmp values(1,2), (2,3), (3,4), (4,5);')
    conn.commit()
    yield kw
    os.unlink(fname)


@pytest.fixture()
def setup_types():
    conn = duckdb.connect(':memory:')
    cur = conn.cursor()
    cur.execute('''create table tmp_types (
        i int, i_n int, f double, f_n double, s varchar, s_n varchar,
        b bool, ts timestamp, ts_n timestamp, d decimal(10,3),
        arr int[], allnull int)''')
    cur.execute('''insert into tmp_types values
        (1, 10, 1.5, 2.5, 'aa', 'bb', true,
         '2020-01-01', '2020-01-01', 1.25, [1,2], NULL),
        (2, NULL, 2.5, NULL, 'cc', NULL, false,
         '2020-01-02', NULL, 2.5, [3], NULL)''')
    cur.close()
    yield conn
    conn.close()


def test_simple(setup):
    a, b = sqlutil.get('select a,b from tmp', **setup)
    assert (len(a) == 4)


def test_preamb(setup):
    a, b = sqlutil.get('select a,b from tmp', preamb='select 1', **setup)
    assert (len(a) == 4)


def test_empty(setup):
    a, b = sqlutil.get('select a,b from tmp where a<0', **setup)
    assert (len(a) == 0)


def test_types_and_nulls(setup_types):
    conn = setup_types
    res = sqlutil.get('select * from tmp_types order by i',
                      conn=conn,
                      driver='duckdb',
                      asDict=True)

    assert res['i'].dtype.kind == 'i'
    assert (res['i'] == np.array([1, 2])).all()

    # integer nulls replaced by intNullVal
    assert res['i_n'].dtype.kind == 'i'
    assert (res['i_n'] == np.array([10, -9999])).all()

    # float nulls replaced by nan
    assert res['f'].dtype.kind == 'f'
    assert np.isclose(res['f_n'][0], 2.5)
    assert np.isnan(res['f_n'][1])

    # strings converted to fixed width arrays,
    # nulls replaced by strNullVal
    assert res['s'].dtype.kind == 'U'
    assert (res['s'] == np.array(['aa', 'cc'])).all()
    assert res['s_n'].dtype.kind == 'U'
    assert (res['s_n'] == np.array(['bb', 'None'])).all()

    assert res['b'].dtype.kind == 'b'
    assert (res['b'] == np.array([True, False])).all()

    # timestamps
    assert res['ts'].dtype.kind == 'M'
    assert res['ts'][0] == np.datetime64('2020-01-01')
    assert res['ts_n'].dtype.kind == 'M'
    assert np.isnat(res['ts_n'][1])

    # decimals as floats
    assert res['d'].dtype.kind == 'f'
    assert np.isclose(res['d'][0], 1.25)

    # arrays of different lengths
    assert (res['arr'][0] == np.array([1, 2])).all()
    assert (res['arr'][1] == np.array([3])).all()

    # all-null integer column
    assert (res['allnull'] == np.array([-9999, -9999])).all()


def test_intnull_none(setup_types):
    # if intNullVal is None, integer columns with nulls
    # are returned as object columns with Nones
    conn = setup_types
    i_n, = sqlutil.get('select i_n from tmp_types order by i',
                       conn=conn,
                       driver='duckdb',
                       intNullVal=None)
    assert i_n.dtype.kind == 'O'
    assert i_n[0] == 10
    assert i_n[1] is None


def test_custom_nullvals(setup_types):
    conn = setup_types
    i_n, s_n = sqlutil.get('select i_n, s_n from tmp_types order by i',
                           conn=conn,
                           driver='duckdb',
                           intNullVal=-77,
                           strNullVal='MISSING')
    assert (i_n == np.array([10, -77])).all()
    assert (s_n == np.array(['bb', 'MISSING'])).all()


def test_repeated_columns(setup_types):
    conn = setup_types
    with pytest.warns(UserWarning):
        res = sqlutil.get('select i, i from tmp_types',
                          conn=conn,
                          driver='duckdb',
                          asDict=True)
    assert len(res) == 2
    for v in res.values():
        assert (np.sort(v) == np.array([1, 2])).all()


def test_empty_typed(setup_types):
    conn = setup_types
    i, s = sqlutil.get('select i, s from tmp_types where i < 0',
                       conn=conn,
                       driver='duckdb')
    assert isinstance(i, np.ndarray)
    assert isinstance(s, np.ndarray)
    assert len(i) == 0
    assert len(s) == 0
    assert i.dtype.kind == 'i'
