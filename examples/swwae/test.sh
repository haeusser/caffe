#!/usr/bin/env sh

# put this in your ~/.bashrc:
# export CAFFE_ROOT=/home/haeusser/libs/caffe
# or similar path


TOOLS=$CAFFE_ROOT/build/tools
LOG_DIR=$CAFFE_ROOT/examples/swwae/log

MODELFILE=$1

echo "testing $MODELFILE"

mkdir -p $CAFFE_ROOT/examples/swwae/modelfiles
mkdir -p $LOG_DIR
 
$TOOLS/caffe test \
    -model=examples/swwae/train.prototxt \
    -weights=$MODELFILE \
    2>&1 | tee $LOG_DIR/test.log