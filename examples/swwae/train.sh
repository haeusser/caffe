#!/usr/bin/env sh

# put this in your ~/.bashrc:
# export CAFFE_ROOT=/home/haeusser/libs/caffe
# or similar path

TOOLS=$CAFFE_ROOT/build/tools
LOG_DIR=$CAFFE_ROOT/examples/swwae/log
mkdir -p $CAFFE_ROOT/examples/swwae/modelfiles
mkdir -p $LOG_DIR
MODELFILE=$(ls $CAFFE_ROOT/examples/swwae/modelfiles -t | head -1)

$TOOLS/caffe train \
    --solver=$CAFFE_ROOT/examples/swwae/solver.prototxt \
    --snapshot=$CAFFE_ROOT/examples/swwae/modelfiles/$MODELFILE \
    2>&1 | tee $LOG_DIR/log.log
