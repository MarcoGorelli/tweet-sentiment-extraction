#!/bin/bash

SLUG=$1;

rm $SLUG.ipynb;
ipynb-py-convert $SLUG.py $SLUG.ipynb
