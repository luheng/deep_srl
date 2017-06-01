SRC_DIR=./bazel-neural_srl/external/dynet/proto
DST_DIR=./python/neural_srl/shared

SRC_DIR2=./proto/

#protoc -I=$SRC_DIR --python_out=$DST_DIR "$SRC_DIR/tensor.proto"

protoc -I="${SRC_DIR2}" --python_out=$DST_DIR "${SRC_DIR2}/scores.proto"
