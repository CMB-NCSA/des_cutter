#!/usr/bin/env bash

PRODUCT_DIR="$1"

if [ $PRODUCT_DIR = "." ]; then
    PRODUCT_DIR=$PWD
fi

echo "Adding: $PRODUCT_DIR to paths"
export PYTHONPATH=${PRODUCT_DIR}/python:${PYTHONPATH}
export PATH=${PRODUCT_DIR}/bin:${PATH}
export DES_CUTTER_DIR=$PRODUCT_DIR
echo "Added DES_CUTTER_DIR:" $DES_CUTTER_DIR
