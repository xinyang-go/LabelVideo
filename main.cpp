/**
 * @file main.cpp
 * @brief 标注视频数据
 * @author xinyang
 * @date 2020-10-23
 */

#include "label.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, const char *argv[]) {
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " <video> <yaml>" << std::endl;
        std::cout << std::endl;
        std::cout << "--     video: the path to the video." << std::endl;
        std::cout << "--     yaml: the path to the yaml label file." << std::endl;
        return -1;
    }
    std::string video_file(argv[1]);
    std::string yaml_file(argv[2]);

    if (!fs::exists(video_file)) {
        std::cout << "[ERROR]: not such file: " << video_file << std::endl;
        return -1;
    }

    Labeler labeler(video_file, yaml_file);
    labeler.start();
    labeler.save_yaml(yaml_file);

    return 0;
}
