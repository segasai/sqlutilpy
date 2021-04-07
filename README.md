[![Build Status](https://github.com/segasai/sqlutilpy/workflows/Testing/badge.svg)](https://github.com/segasai/sqlutilpy/actions)
[![Documentation Status](https://readthedocs.org/projects/sqlutilpy/badge/?version=latest)](http://sqlutilpy.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/segasai/sqlutilpy/badge.svg?branch=master)](https://coveralls.io/github/segasai/sqlutilpy?branch=master)

# sqlutilpy
Python module to query SQL databases and return numpy arrays, upload
tables and run join queries involving local arrays and the tables in the DB.
The module only works with PostgreSQL and SQLite databases.

The full documentation is available [here](http://sqlutilpy.readthedocs.io/en/latest/)

Author: Sergey Koposov (Uni of Cambridge/CMU/Uni of Edinburgh)

## Installation
To install the package you just need to do pip install. 

```
pip install sqlutilpy
```
## Authentification
Throughout this readme, I'll assume that the .pgpass file ( https://www.postgresql.org/docs/11/libpq-pgpass.html ) 
has been created with the login/password details for Postgresql. If that is not the case, all of the 
commands given above will also need user='....' and password='...' options

## Querying the database and retrieving the results
```
import sqlutilpy
ra,dec = squtilpy.get('select ra,dec from mytable', host='HOST_NAME_OF_MY_PG_SERVER', db='THE_NAME_OF_MY_DB')
```

By default sqlutilpy.get executes the result and returns the tuple of 
results. But you can return the results as dictionary using asDict option.

## Uploading your arrays as column in a table
```
x = np.arange(10)                                                   
y = x**.5                                                           
sqlutilpy.upload('mytable',(x,y),('xcol','ycol'))    
``` 
This will create a table called mytable with columns xcol and ycol 

## Join query involving your local data and the database table

Imagine you have arrays myid and y and you want to to extract all the 
information from somebigtable for objects with id=myid. In principle
you could upload the arrays in the DB and run a query, but local_join function does that for you.

```
myid = np.arange(10)
y = x**.5
R=sqlutilpy.local_join('select * from mytmptable as m, somebigtable as s where s.id=m.myid order by m.myid',                                                                            
'mytmptable',(x,y),('myid','ycol'))
```
It executes a query as if you arrays where in a mytmptable. ( behind the scenes
it uploads the data to the db and runs a query)


## Keeping the connection open. 

Often it's benefitial to preserve an open connection. You can do that if you first 
obtain the connection using sqlutilpy.getConnection() and then provide it directly
to sqlutil.get() and friends using conn=conn argument


