import sys
def getmod_path():
    import sqlutilpy
    path = '/'.join(sqlutilpy.__file__.split('/')[:-1]) + '/'
    return path

with open('.coveragerc', 'w') as fd:
    path = getmod_path()
    print ('''[run]
    source='''+path,file=fd)


