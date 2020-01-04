#!/bin/sh
rm package.tgz
find . -name '._*' -exec rm {} \;
tar --exclude '__pycache__' --exclude 'saved_models' --exclude 'images' --exclude './data' --exclude './implementations/cyclegan/core' -zcvf package.tgz ./*
