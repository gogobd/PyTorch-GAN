#!/bin/sh
rm package.tgz
tar --exclude './data' -zcvf package.tgz ./*
