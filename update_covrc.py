import sys
import importlib


def getmod_path(modname):
    mod = importlib.import_module(modname)
    path = '/'.join(mod.__path__[0].split('/')[:-1]) + '/'
    # path = '/'.join(path.split('/')[:-1]) + '/'
    return path


with open('.coveragerc', 'w') as fd:
    path = getmod_path('sqlutilpy')
    print(f'''[run]
source={path}
''', file=fd)
