#!/bin/sh
rm package.tgz
tar --exclude './data' --exclude './implementations/context_encoder/checkpoints' -zcvf package.tgz ./*
