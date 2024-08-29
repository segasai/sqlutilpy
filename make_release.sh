#!/bin/bash 
set -o nounset -o errexit
VERSION=$1
if  [ `git status --porcelain=v1 | grep -v '^??'|wc -l ` -eq 0 ] ; then echo 'Good'; else {
    echo "Uncommitted changes found";
    exit 1;
} ; fi 
echo "__version__ = '$VERSION'" > src/sqlutilpy/version.py
git commit -m "New version $VERSION" -v  src/sqlutilpy/version.py
git tag $VERSION
rm -rf dist/
rm -rf build/
rm -rf src/sqlutilpy.egg-info
TMPDIR=`mktemp -d`
cp -r * .??* $TMPDIR
cd $TMPDIR
python -m build --sdist --wheel
twine check dist/*
echo "Now execute twine upload $TMPDIR/dist/*"
echo "rm -rf $TMPDIR"
