#!/bin/bash

set -xe

echo "Creating Dataset"
tar -xf train.tar

echo "copying tflite model"
cp ../mobilenetv2_int8_ptq.tflite ./