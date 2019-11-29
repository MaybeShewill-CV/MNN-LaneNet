#!/usr/bin/env bash

set -eux

PYTHONPATH=$(pwd)

input_para_nums=$#
parameters=1
para0=$0

function usage() {
    echo "usage: $para0 mnn_convert_tool_path"
    echo "       mnn_convert_tool_path  Your local compiled mnn model converter tool path"
    echo "examples: "
    echo "       $para0 MNNProject_ROOT/tools/converter/build/MNNConverter"
    exit 1
}

if [ ${input_para_nums} != ${parameters} ];
then
    usage
    exit
else
    para1=$1
fi

function convert_model() {
    # freeze tf model
    python ./tools/freeze_lanenet_model.py --model_meta_path ./checkpoint/tusimple_lanenet.ckpt.meta --save_path ./checkpoint/tusimple_lanenet.pb
    # convert mnn model
    ${para1} -f TF --modelFile ./checkpoint/tusimple_lanenet.pb --MNNModel ./checkpoint/lanenet_model.mnn --bizCode MNN
}

function main() {
if [ ${input_para_nums} != ${parameters} ];
then
    usage
else
    convert_model
fi
}

main