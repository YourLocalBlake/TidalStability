#!/bin/sh

# Reproduced from https://github.com/bmcage/odes/
# Copyright (C) 2011-12  Pavol Kišon
# Copyright (C) 2011-12  Benny Malengier
# All rights reserved.
# Reproduced under the new-BSD licence and slightly modified.

SUNDIALS_DEFAULT_VERSION='5.1.0'
SUNDIALS_DEFAULT_PRECISION='double'
SUNDIALS_DEFAULT_INDEX_SIZE='64'

export SUNDIALS_DIR=$HOME/sundials/"${SUNDIALS_VERSION:-$SUNDIALS_DEFAULT_VERSION}"/"${SUNDIALS_PRECISION:-$SUNDIALS_DEFAULT_PRECISION}"/"${SUNDIALS_INDEX_SIZE:-$SUNDIALS_DEFAULT_INDEX_SIZE}"
SUNDIALS_LIBDIR=$SUNDIALS_DIR/lib
SUNDIALS_INCLUDEDIR=$SUNDIALS_DIR/include

if [ ! -d "$SUNDIALS_LIBDIR" ]; then
    mkdir -p $SUNDIALS_DIR
    echo "Installing sundials ${SUNDIALS_VERSION:-$SUNDIALS_DEFAULT_VERSION}"
    ./ci/install_sundials.sh
else
    echo "Using cached sundials in $SUNDIALS_LIBDIR"
fi

export LD_LIBRARY_PATH=$SUNDIALS_LIBDIR:$LD_LIBRARY_PATH
export LIBRARY_PATH=$SUNDIALS_LIBDIR:$LIBRARY_PATH
export CPATH=$SUNDIALS_INCLUDEDIR:$CPATH