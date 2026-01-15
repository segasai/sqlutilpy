To test the postgresql part of the code you can use the environment variables
like this:

env SQLUTIL_TEST_PG_USER=skoposov SQLUTIL_TEST_PG_HOST=localhost SQLUTIL_TEST_PG_DB=test  SQLUTIL_TEST_PG_PASS="" pytest

assuming you have a local postgresql database called test.