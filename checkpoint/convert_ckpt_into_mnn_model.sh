#!/usr/bin/env bash

set -eux

PYTHONPATH=$(pwd) python ./tools/freeze_lanenet_model --weights_path ./checkpoint/tusimple_lanenet_vgg.ckpt --save_path ./checkpoint/tusimple_lanenet.pb

MNNConverter -f TF --modelFile ./checkpoint/tusimple_lanenet.pb --MNNModel ./checkpoint/lanenet_model.mnn --bizCode MNN