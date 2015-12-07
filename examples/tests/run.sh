 
#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  -solver=examples/tests/pynet_solver.prototxt \
  -gpu 0,1
