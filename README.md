# sqlutilpy
Python module to query SQL databases and return numpy arrays 
By default the module only works for PostgreSQL and sqlite databases

# Querying the database and retrieving the results

> ra,dec = sqlutil.get('select ra,dec from mytable', host='HOST_NAME_OF_MY_PG_SERVER', db='THE_NAME_OF_MY_DB')
