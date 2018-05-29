[![Build Status](https://travis-ci.org/segasai/sqlutilpy.svg?branch=master)](https://travis-ci.org/segasai/sqlutilpy)
[![Documentation Status](https://readthedocs.org/projects/sqlutilpy/badge/?version=latest)](http://sqlutilpy.readthedocs.io/en/latest/?badge=latest)


# sqlutilpy
Python module to query SQL databases and return numpy arrays, upload
tables and run join queries involving local arrays and the tables in the DB.
The module only works for PostgreSQL and sqlite databases.

The full documentation is available [here](http://sqlutilpy.readthedocs.io/en/latest/)

## Installation
To install the package you just need to do pip install. 

```
pip install sqlutilpy
```


## Querying the database and retrieving the results
```
ra,dec = sqlutil.get('select ra,dec from mytable', host='HOST_NAME_OF_MY_PG_SERVER', db='THE_NAME_OF_MY_DB')
```

By default sqlutil.get executes the result and returns the tuple of 
results

## Uploading your arrays as column in a table
```
x = np.arange(10)                                                   
y = x**.5                                                           
sqlutil.upload('mytable',(x,y),('xcol','ycol'))    
``` 
This will create a table called mytable with columns xcol and ycol 

## Join query involving your local data and the database table

Imagine you have arrays myid and y and you want to to extract all the 
information from somebigtable for objects with id=myid. In principle
you could upload the arrays in the DB and run a query, but local_join function does that for you.

```
myid = np.arange(10)
y = x**.5
sqlutilpy.local_join('select * from mytmptable as m, somebigtable as s where s.id=m.myid order by m.myid',                                                                           'mytmptable',(x,y),('myid','ycol'))
```
It executes a query as if you arrays where in a mytmptable. ( behind the scenes
it uploads the data to the db and runs a query)

