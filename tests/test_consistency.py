import pytest
import os
import sqlutilpy as sqlutil
import numpy as np

# PostgreSQL config
PG_DB = os.environ.get('SQLUTIL_TEST_PG_DB')
PG_HOST = os.environ.get('SQLUTIL_TEST_PG_HOST')
PG_USER = os.environ.get('SQLUTIL_TEST_PG_USER')
PG_PASS = os.environ.get('SQLUTIL_TEST_PG_PASS', "")


@pytest.fixture(scope="module")
def pg_conn():
    if not PG_DB:
        pytest.skip("PostgreSQL environment variables not set")
    conn = sqlutil.getConnection(
        host=PG_HOST,
        user=PG_USER,
        db=PG_DB,
        password=PG_PASS,
        driver='psycopg')
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def sqlite_conn():
    fname = 'test_consistency.db'
    conn = sqlutil.getConnection(db=fname, driver='sqlite3')
    yield conn
    conn.close()
    if os.path.exists(fname):
        os.unlink(fname)


@pytest.fixture(scope="module")
def duckdb_conn():
    conn = sqlutil.getConnection(db=':memory:', driver='duckdb')
    yield conn
    conn.close()


def create_and_populate(conn, driver, table_name, data):
    # data is a list of tuples: (col_name, col_type_pg, col_type_sqlite,
    # col_type_duckdb, values)

    cols = []
    vals = []

    for col_def in data:
        name = col_def[0]
        if driver == 'psycopg':
            type_ = col_def[1]
        elif driver == 'sqlite3':
            type_ = col_def[2]
        elif driver == 'duckdb':
            type_ = col_def[3]
        cols.append(f"{name} {type_}")
        vals.append(col_def[4])

    create_sql = f"create table {table_name} ({', '.join(cols)})"

    # Handle drop if exists
    try:
        cur = conn.cursor()
        cur.execute(f"drop table if exists {table_name}")
        cur.execute(create_sql)

        # Transpose vals to get rows
        rows = list(zip(*vals))

        placeholders = ",".join(
            ["?" if driver != 'psycopg' else "%s"] * len(cols))
        insert_sql = f"insert into {table_name} values ({placeholders})"

        for row in rows:
            cur.execute(insert_sql, row)

        conn.commit()
    finally:
        cur.close()


def test_consistency_types_and_nulls(pg_conn, sqlite_conn, duckdb_conn):
    table_name = "test_consistency"

    # Define data
    # (name, pg_type, sqlite_type, duckdb_type, [values])
    data = [
        ("id", "int", "integer", "integer", [1, 2, 3, 4]),
        ("val_int", "int", "integer", "integer", [10, None, 30, 40]),
        ("val_float", "double precision", "real",
         "double", [1.1, 2.2, None, 4.4]),
        ("val_str", "varchar(20)", "text", "varchar", ["a", "b", "c", None]),
        ("val_long_str", "varchar(50)", "text", "varchar", [
         "short", "very_long_string_exceeding_default_maybe", "medium", None]),
    ]

    # Populate PG
    create_and_populate(pg_conn, 'psycopg', table_name, data)

    # Populate SQLite
    create_and_populate(sqlite_conn, 'sqlite3', table_name, data)

    # Populate DuckDB
    create_and_populate(duckdb_conn, 'duckdb', table_name, data)

    # Test with default strLength (20)
    query = f"select * from {table_name} order by id"

    # Fetch results
    pg_res = sqlutil.get(query, conn=pg_conn, driver='psycopg', strLength=10)
    sqlite_res = sqlutil.get(
        query,
        conn=sqlite_conn,
        driver='sqlite3',
        strLength=10)
    duckdb_res = sqlutil.get(
        query,
        conn=duckdb_conn,
        driver='duckdb',
        strLength=10)

    # Compare column by column
    col_names = [d[0] for d in data]

    for i, name in enumerate(col_names):
        pg_col = pg_res[i]
        sl_col = sqlite_res[i]
        dd_col = duckdb_res[i]

        print(f"\nComparing column: {name}")
        # print(f"PG: dtype={pg_col.dtype}")
        # print(f"SL: dtype={sl_col.dtype}")
        # print(f"DD: dtype={dd_col.dtype}")

        print(f"PG values: {pg_col}")
        print(f"SL values: {sl_col}")
        print(f"DD values: {dd_col}")

        if name == 'val_long_str':
            # Expect truncation to 10 chars
            # PG does it.
            # SL/DD currently don't.
            assert np.array_equal(
                pg_col, sl_col), f"SQLite mismatch for {name}"
            assert np.array_equal(
                pg_col, dd_col), f"DuckDB mismatch for {name}"
        elif name == 'val_int':
            assert np.array_equal(
                pg_col, sl_col), f"SQLite mismatch for {name}"
            assert np.array_equal(
                pg_col, dd_col), f"DuckDB mismatch for {name}"
        elif name == 'val_float':
            np.testing.assert_array_equal(
                pg_col, sl_col, err_msg=f"SQLite mismatch for {name}")
            np.testing.assert_array_equal(
                pg_col, dd_col, err_msg=f"DuckDB mismatch for {name}")
        elif name == 'val_str':
            assert np.array_equal(
                pg_col, sl_col), f"SQLite mismatch for {name}"
            assert np.array_equal(
                pg_col, dd_col), f"DuckDB mismatch for {name}"
        else:
            assert np.array_equal(
                pg_col, sl_col), f"SQLite mismatch for {name}"
            assert np.array_equal(
                pg_col, dd_col), f"DuckDB mismatch for {name}"


def test_consistency_empty_result(pg_conn, sqlite_conn, duckdb_conn):
    table_name = "test_consistency_empty"
    # Create tables
    cols = "id int, val_str varchar(20)"
    
    # PG
    with pg_conn.cursor() as cur:
        cur.execute(f"create temp table {table_name} ({cols})")
        
    # SQLite
    cur = sqlite_conn.cursor()
    cur.execute(f"create table {table_name} (id integer, val_str text)")
    cur.close()
        
    # DuckDB
    cur = duckdb_conn.cursor()
    cur.execute(f"create table {table_name} (id integer, val_str varchar)")
    cur.close()
        
    query = f"select * from {table_name}"
    
    pg_res = sqlutil.get(query, conn=pg_conn, driver='psycopg')
    sqlite_res = sqlutil.get(query, conn=sqlite_conn, driver='sqlite3')
    duckdb_res = sqlutil.get(query, conn=duckdb_conn, driver='duckdb')
    
    # Check structure
    assert len(pg_res) == 2
    assert len(sqlite_res) == 2
    assert len(duckdb_res) == 2
    
    # Check types
    # PG returns numpy arrays
    assert isinstance(pg_res[0], np.ndarray)
    
    # SQLite/DuckDB currently return lists -> this should fail
    # or if we fix it, they should return numpy arrays
    assert isinstance(sqlite_res[0], np.ndarray), "SQLite returned list instead of numpy array"
    assert isinstance(duckdb_res[0], np.ndarray), "DuckDB returned list instead of numpy array"
    
    # Check dictionary keys
    pg_dict = sqlutil.get(query, conn=pg_conn, driver='psycopg', asDict=True)
    sqlite_dict = sqlutil.get(query, conn=sqlite_conn, driver='sqlite3', asDict=True)
    duckdb_dict = sqlutil.get(query, conn=duckdb_conn, driver='duckdb', asDict=True)
    
    assert set(pg_dict.keys()) == {'id', 'val_str'}
    assert set(sqlite_dict.keys()) == {'id', 'val_str'}
    assert set(duckdb_dict.keys()) == {'id', 'val_str'}
