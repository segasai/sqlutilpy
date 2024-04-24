[![Build Status](https://github.com/segasai/sqlutilpy/workflows/Testing/badge.svg)](https://github.com/segasai/sqlutilpy/actions)
[![Documentation Status](https://readthedocs.org/projects/sqlutilpy/badge/?version=latest)](http://sqlutilpy.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/segasai/sqlutilpy/badge.svg?branch=master)](https://coveralls.io/github/segasai/sqlutilpy?branch=master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6867957.svg)](https://doi.org/10.5281/zenodo.6867957)

# sqlutilpy
Python module to query SQL databases and return numpy arrays, upload
tables and run join queries involving local arrays and the tables in the DB.
This module is optimized to be able to deal efficiently with query results with millions of rows.
The module only works with PostgreSQL, SQLite and DuckDB databases.

The full documentation is available [here](http://sqlutilpy.readthedocs.io/en/latest/)

Author: Sergey Koposov (Uni of Cambridge/CMU/Uni of Edinburgh)

## Installation
To install the package you just need to do pip install. 

```
pip install sqlutilpy
```
## Authentication

Throughout this readme, I will assume that the .pgpass file ( https://www.postgresql.org/docs/11/libpq-pgpass.html ) 
has been created with your login/password details for Postgresql. If that is not the case, all of the 
commands given below will also need user='....' and password='...' options

## Connection information

Most of the sqlutilpy commands require hostname, database name, user etc. 
If you don't want to always type it, you can use standard PostgreSQL environment variables
like PGPORT, PGDATABASE, PGUSER, PGHOST for the port, database name, user name and hostname
of the connection. 


## Querying the database and retrieving the results

This command will run the query and put the columns into variables ra,dec
```python
import sqlutilpy
ra,dec = squtilpy.get('select ra,dec from mytable', 
                 host='HOST_NAME_OF_MY_PG_SERVER', 
                 db='THE_NAME_OF_MY_DB')
```

By default sqlutilpy.get executes the query and returns the tuple with 
results. You can return the results as dictionary using asDict option.

## Uploading your arrays as column in a table

```python
x = np.arange(10)                                                   
y = x**.5                                                           
sqlutilpy.upload('mytable',(x,y),('xcol','ycol'))    
``` 
This will create a table called mytable with columns xcol and ycol 

## Join query involving your local data and the database table

Imagine you have arrays myid and y and you want to to extract all the 
information from somebigtable for objects with id=myid. In principle
you could upload the arrays in the DB and run a query, but local_join function does that for you.

```python
myid = np.arange(10)
y = np.random.uniform(size=10)

R=sqlutilpy.local_join('''select * from mytmptable as m, 
           somebigtable as s where s.id=m.myid order by m.myid''',                                              
           'mytmptable',(myid, y),('myid','ycol'))
```

It executes a query as if you arrays were in mytmptable. (behind the scenes
it uploads the data to the db and runs a query)

## Keeping the connection open. 

Often it is beneficial to preserve an open connection to the database. You can do that if you first 
obtain the connection using sqlutilpy.getConnection() and then provide it directly
to sqlutil.get() and friends using conn=conn argument
```python
conn = sqlutilpy.getConnection(db='mydb', user='meuser', password='something', host='hostname')
R= sqlutilpy.get('select 1', conn=conn)
R1= sqlutilpy.get('select 1', conn=conn)
```

# How to cite the software

If you use this package, please cite it through zenodo https://doi.org/10.5281/zenodo.6867957

