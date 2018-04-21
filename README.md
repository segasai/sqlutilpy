[![Build Status](https://travis-ci.org/segasai/sqlutilpy.svg?branch=master)](https://travis-ci.org/segasai/sqlutilpy)

# sqlutilpy
Python module to query SQL databases and return numpy arrays 
By default the module only works for PostgreSQL and sqlite databases

# Querying the database and retrieving the results

> ra,dec = sqlutil.get('select ra,dec from mytable', host='HOST_NAME_OF_MY_PG_SERVER', db='THE_NAME_OF_MY_DB')

# Uploading your arrays as column in a table
   > x = np.arange(10)                                                   
   > y = x**.5                                                           
   > sqlutil.upload('mytable',(x,y),('xcol','ycol'))    
   This will create a table called mytable with columns xcol and ycol 
  

