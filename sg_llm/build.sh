#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p build
pushd build
cmake .. -DCMAKE_PREFIX_PATH=~/.local/lib/python3.8/site-packages/pybind11 -DCMAKE_INSTALL_PREFIX=${DIR}/python
cmake --build . --target install
popd