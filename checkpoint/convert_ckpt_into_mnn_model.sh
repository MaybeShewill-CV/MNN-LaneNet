#!/usr/bin/env bash

function print_usage() {
    echo "usage: bash model_converter_script mnn_convert_tool_path"
    echo "       mnn_convert_tool_path  Your local compiled mnn model converter tool path"
    echo "examples: "
    echo "       bash ./checkpoint/convert_ckpt_into_mnn_model.sh MNNProject_ROOT/tools/converter/build/MNNConverter"
    exit 1
}

# if less than two arguments supplied, display usage
if [  $# != 1 ];
then
  print_usage
  exit 1
fi

if [[ ( $1 == "--help") ||  $1 == "-h" ]];
then
  print_usage
  exit
fi

para1=$1

function convert_model() {
    # freeze tf model
    python ./tools/freeze_lanenet_model.py --model_meta_path ./checkpoint/tusimple_lanenet.ckpt.meta --save_path ./checkpoint/tusimple_lanenet.pb
    # convert mnn model
    ${para1} -f TF --modelFile ./checkpoint/tusimple_lanenet.pb --MNNModel ./checkpoint/lanenet_model.mnn --bizCode MNN
}

convert_model
