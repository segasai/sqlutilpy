import numpy as np
from sqlutilpy.sqlutil import __fix_numpy_record_array


def test_fix_numpy_integers():
    # Case: Integers with None -> Object array
    # Should convert to int array with sentinel
    data = [(1,), (None,), (2,)]
    arr = np.rec.array(data, names=['a'])
    # arr['a'] is object type here

    fixed = __fix_numpy_record_array(arr, intNullVal=-9999, strNullVal='None')

    assert fixed['a'].dtype.kind == 'i'
    assert fixed['a'][0] == 1
    assert fixed['a'][1] == -9999
    assert fixed['a'][2] == 2


def test_fix_numpy_floats():
    # Case: Floats with None -> Object array
    # Should convert to float array with NaN
    data = [(1.1,), (None,), (2.2,)]
    arr = np.rec.array(data, names=['a'])

    fixed = __fix_numpy_record_array(arr, intNullVal=-9999, strNullVal='None')

    assert fixed['a'].dtype.kind == 'f'
    assert np.isclose(fixed['a'][0], 1.1)
    assert np.isnan(fixed['a'][1])
    assert np.isclose(fixed['a'][2], 2.2)


def test_fix_numpy_strings():
    # Case: Strings with None -> Object array
    # Should convert to string array with sentinel
    data = [('foo',), (None,), ('bar',)]
    arr = np.rec.array(data, names=['a'])

    fixed = __fix_numpy_record_array(
        arr, intNullVal=-9999, strNullVal='MISSING')

    assert fixed['a'].dtype.kind in ('U', 'S')
    assert fixed['a'][0] == 'foo'
    assert fixed['a'][1] == 'MISSING'
    assert fixed['a'][2] == 'bar'


def test_fix_numpy_mixed_remain_object():
    # Case: Mixed types (int and string) -> Object array
    # Should remain Object
    data = [(1,), ('a',)]
    # Force object dtype to simulate what happens when automatic inference fails to find a common primitive type
    # or creates an object array
    arr = np.rec.array(data, names=['a'], dtype=[('a', 'O')])

    fixed = __fix_numpy_record_array(arr, intNullVal=-9999, strNullVal='None')

    assert fixed['a'].dtype.kind == 'O'
    assert fixed['a'][0] == 1
    assert fixed['a'][1] == 'a'


def test_fix_numpy_all_none():
    # Case: All None
    # Cannot infer type, should remain Object
    data = [(None,), (None,)]
    arr = np.rec.array(data, names=['a'], dtype=[('a', 'O')])

    fixed = __fix_numpy_record_array(arr, intNullVal=-9999, strNullVal='None')

    assert fixed['a'].dtype.kind == 'O'
    assert fixed['a'][0] is None


def test_fix_numpy_empty():
    # Case: Empty array
    # We need to provide a dtype or formats to create an empty recarray from
    # empty list
    arr = np.array([], dtype=[('a', 'i4')]).view(np.recarray)

    fixed = __fix_numpy_record_array(arr, intNullVal=-9999, strNullVal='None')

    assert len(fixed) == 0


def test_fix_numpy_already_typed():
    # Case: Already typed (no Nones involved during creation)
    data = [(1,), (2,)]
    arr = np.rec.array(data, dtype=[('a', int)])

    fixed = __fix_numpy_record_array(arr, intNullVal=-9999, strNullVal='None')

    assert fixed['a'].dtype.kind == 'i'
    assert fixed['a'][0] == 1
