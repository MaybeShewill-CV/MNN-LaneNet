#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/5 下午4:53
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : freeze_lanenet_model.py.py
# @IDE: PyCharm
"""
Freeze Lanenet model into frozen pb file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

MODEL_WEIGHTS_FILE_PATH = './checkpoint/tusimple_lanenet_vgg.ckpt'
MODEL_META_FILE_PATH = './checkpoint/tusimple_lanenet_vgg.ckpt.meta'
OUTPUT_PB_FILE_PATH = './checkpoint/tusimple_lanenet.pb'


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--model_meta_path', default=MODEL_META_FILE_PATH)
    parser.add_argument('-s', '--save_path', default=OUTPUT_PB_FILE_PATH)

    return parser.parse_args()


def convert_ckpt_into_pb_file(model_meta_file_path, pb_file_path):
    """

    :param model_meta_file_path:
    :param pb_file_path:
    :return:
    """
    # create a session
    saver = tf.train.import_meta_graph(model_meta_file_path)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.85
    sess_config.gpu_options.allow_growth = False
    sess_config.gpu_options.allocator_type = 'BFC'
    sess_config.allow_soft_placement = True
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=tf.train.latest_checkpoint('./checkpoint'))

        graph = sess.graph
        binary_seg_ret = graph.get_tensor_by_name("lanenet_model/vgg_backend/binary_seg/ArgMax:0")
        instance_seg_ret = graph.get_tensor_by_name(
            "lanenet_model/vgg_backend/instance_seg/pix_embedding_conv/pix_embedding_conv:0")

        with tf.variable_scope('lanenet/'):
            binary_seg_ret = tf.cast(binary_seg_ret, dtype=tf.float32)
            binary_seg_ret = tf.squeeze(binary_seg_ret, axis=0, name='final_binary_output')
            instance_seg_ret = tf.squeeze(instance_seg_ret, axis=0, name='final_pixel_embedding_output')

            converted_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                input_graph_def=sess.graph.as_graph_def(),
                output_node_names=[
                    'lanenet/input_tensor',
                    'lanenet/final_binary_output',
                    'lanenet/final_pixel_embedding_output'
                ]
            )

            with tf.gfile.GFile(pb_file_path, "wb") as f:
                f.write(converted_graph_def.SerializeToString())


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()

    convert_ckpt_into_pb_file(
        model_meta_file_path=args.model_meta_path,
        pb_file_path=args.save_path
    )
