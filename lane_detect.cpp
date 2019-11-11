// Author: MaybeShewill-CV
// File: lane_detect.cpp
// Date: 2019/11/11 上午11:08

// Lane Detection Test Project

#include <string>
#include <memory>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "lanenet/lanenet_model.h"
#include "common/config_parse_utils/config_parser.h"

using lanenet::lane_detection::LaneNet;
using common::config_parse_utils::ConfigParser;

int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetStderrLogging(google::GLOG_INFO);

    if (argc != 3) {
        LOG(INFO) << "Usage: ";
        LOG(INFO) << "lane_detector.out config.ini image_file_path";
        return -1;
    }

    std::string config_file_path = argv[1];
    ConfigParser config(config_file_path);

    auto image_path = argv[2];
    auto input_image = cv::imread(image_path, cv::IMREAD_UNCHANGED);

    auto detector = std::unique_ptr<LaneNet>(new LaneNet(config));

    cv::Mat binary_mask;
    cv::Mat instance_mask;
    detector->detect(input_image, binary_mask, instance_mask);
    cv::imwrite("binary_ret.png", binary_mask);
    cv::imwrite("instance_ret.png", instance_mask);

    return 0;
}