#!/usr/bin/env bash

git clone git@github.com:d-li14/mobilenetv3.pytorch.git ./tmp

mkdir pretrained
cp ./tmp/pretrained/* ./pretrained/
rm -rf ./tmp