#!/bin/sh
# Reproduced from https://github.com/bmcage/odes/
# Copyright (C) 2011-12  Pavol Kišon
# Copyright (C) 2011-12  Benny Malengier
# All rights reserved.
# Reproduced under the new-BSD licence
set -ex

SUNDIALS=sundials-"${SUNDIALS_VERSION:-5.1.0}"
SUNDIALS_FILE=$SUNDIALS.tar.gz
SUNDIALS_URL=https://github.com/LLNL/sundials/releases/download/v5.1.0/sundials-5.1.0.tar.gz
PRECISION="${SUNDIALS_PRECISION:-double}"
INDEX_SIZE="${SUNDIALS_INDEX_SIZE:-64}"

wget "$SUNDIALS_URL"

tar -xzvf "$SUNDIALS_FILE"

mkdir sundials_build

if [ "$PRECISION" = "extended" ]; then
    cd sundials_build &&
        cmake -DCMAKE_INSTALL_PREFIX=$SUNDIALS_DIR -DLAPACK_ENABLE=OFF -DSUNDIALS_INDEX_SIZE="$INDEX_SIZE" -DSUNDIALS_PRECISION="$PRECISION" -DEXAMPLES_INSTALL:BOOL=OFF ../$SUNDIALS &&
        make && make install
else
    cd sundials_build &&
        cmake -DCMAKE_INSTALL_PREFIX=$SUNDIALS_DIR -DLAPACK_ENABLE=ON -DSUNDIALS_INDEX_SIZE="$INDEX_SIZE" -DSUNDIALS_PRECISION="$PRECISION" -DEXAMPLES_INSTALL:BOOL=OFF ../$SUNDIALS &&
        make && make install
fi