#!/bin/bash
set -o nounset -o errexit -o pipefail
VERSION=$1
# untracked files count too: setuptools-git-versioning marks the build
# as dirty if any are present, and PyPI rejects such versions
if [ -n "$(git status --porcelain=v1)" ]; then
    echo "Uncommitted or untracked changes found:"
    git status --short
    exit 1
fi
if git rev-parse -q --verify "refs/tags/$VERSION" >/dev/null; then
    echo "Tag $VERSION already exists"
    exit 1
fi
echo "__version__ = '$VERSION'" > src/sqlutilpy/version.py
git commit -m "New version $VERSION" src/sqlutilpy/version.py
git tag $VERSION
# build from a fresh clone so that only committed files end up in the package
TMPDIR=$(mktemp -d)
git clone -q . $TMPDIR/sqlutilpy
cd $TMPDIR/sqlutilpy
python -m build --sdist --wheel
twine check dist/*
echo "Now execute twine upload $TMPDIR/sqlutilpy/dist/*"
echo "then git push && git push origin $VERSION"
echo "rm -rf $TMPDIR"
