//
// Created by xinyang on 2020/10/23.
//

#define CVUI_IMPLEMENTATION

#include "cvui.h"
#include "label.hpp"
#include <opencv2/tracking.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <fmt/format.h>

namespace fs = std::filesystem;

constexpr const char *WINDOW_NAME = "LabelVideo";
constexpr int IMAGE_SIZE = 960;
constexpr int PANEL_SIZE = 400;

Labeler::Labeler(const std::string &video, int jump)
        : cap(video), track_id_factory(0), current_frame(0), jump_frame(jump),
          total_frame(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT))),
          wr(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH))),
          hr(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))),
          ws(wr > hr ? IMAGE_SIZE : IMAGE_SIZE * wr / hr),
          hs(hr > wr ? IMAGE_SIZE : IMAGE_SIZE * hr / wr),
          roi((IMAGE_SIZE - ws) / 2, (IMAGE_SIZE - hs) / 2, ws, hs),
          focused_id(-1) {
    std::cout << fmt::format("[INFO]: image size = [{}x{}]", wr, hr) << std::endl;
}

Labeler::Labeler(const std::string &video, const std::string &yaml, int jump)
        : cap(video), track_id_factory(0), current_frame(0), jump_frame(jump),
          total_frame(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT))),
          wr(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH))),
          hr(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))),
          ws(wr > hr ? IMAGE_SIZE : IMAGE_SIZE * wr / hr),
          hs(hr > wr ? IMAGE_SIZE : IMAGE_SIZE * hr / wr),
          roi((IMAGE_SIZE - ws) / 2, (IMAGE_SIZE - hs) / 2, ws, hs),
          focused_id(-1) {
    load_yaml(yaml);
    std::cout << fmt::format("[INFO]: image size = [{}x{}]", wr, hr) << std::endl;
}

void Labeler::save_yaml(const std::string &file) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    for (const auto &[num, labels] :data) {
        out << YAML::Key << num;
        out << YAML::Value << YAML::Flow << YAML::Node(labels);
    }
    out << YAML::EndMap;
    std::ofstream ofs(file);
    ofs << out.c_str();
    std::cout << "[INFO]: yaml file save succeed." << std::endl;
}

void Labeler::load_yaml(const std::string &file) {
    if (fs::exists(file)) {
        std::cout << "[INFO]: load exist label data." << std::endl;
        auto node = YAML::LoadFile(file);
        data = node.as<LabelsMap>();
    } else {
        std::cout << "[WARNING]: no exist label data." << std::endl;
    }
}

void Labeler::start() {
    cvui::init(WINDOW_NAME, 0);
    bool running = true;
    while (running) {
        cv::Mat src = load_image(current_frame);
        cv::Mat im2show;
        cv::resize(src, im2show, {ws, hs});
        while (true) {
            cv::Mat frame = draw_current_frame(im2show);
            cvui::imshow(WINDOW_NAME, frame);
            int k = cv::waitKey(20);
            if (k == 'z') {
                running = false;
                break;
            } else if (k == 'q') {
                last_frame();
                break;
            } else if (k == 'e') {
                next_frame();
                track_target();
                break;
            } else if (k == 'a') {
                add_target(im2show);
            } else if (k == 'd') {
                remove_target();
            }
        }

    }
}

cv::Mat Labeler::draw_current_frame(const cv::Mat &im2show) {
    cv::Mat frame(IMAGE_SIZE, IMAGE_SIZE + PANEL_SIZE, CV_8UC3);
    frame.setTo(0);
    im2show.copyTo(frame(roi));
    for (int i = 0; i < data[current_frame].size(); i++) {
        const auto &l = data[current_frame][i];
        cv::Scalar color{64, 255, 64};
        if (l.is_track) color = {255, 64, 64};
        if (i == focused_id) color = {64, 64, 255};
        double x1 = l.x - l.w / 2;
        double y1 = l.y - l.h / 2;
        double x2 = x1 + l.w;
        double y2 = y1 + l.h;
        x1 = x1 * ws / wr + roi.x;
        y1 = y1 * hs / hr + roi.y;
        x2 = x2 * ws / wr + roi.x;
        y2 = y2 * hs / hr + roi.y;
        std::string str = l.name + ":" + std::to_string(l.id);
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
        if (cvui::iarea(x1, y1, x2 - x1, y2 - y1) == cvui::CLICK) {
            focused_id = i;
        }
        auto ts = cv::getTextSize(str, 0, 0.5, 1, nullptr);
        x2 = x1 + ts.width;
        y2 = y1 - ts.height - 3;
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, -1);
        cv::putText(frame, str, cv::Point(x1, y1 - 2), 0, 0.5, {0, 0, 0}, 1);
    }
    cv::rectangle(frame, {0, 0, IMAGE_SIZE, IMAGE_SIZE}, {255, 255, 255}, 1);
    cvui::text(frame, IMAGE_SIZE + 10, 5, fmt::format("HELP:", current_frame, total_frame));
    cvui::text(frame, IMAGE_SIZE + 20, 20, fmt::format("z: exit labeling", current_frame, total_frame));
    cvui::text(frame, IMAGE_SIZE + 20, 35, fmt::format("q: last frame", current_frame, total_frame));
    cvui::text(frame, IMAGE_SIZE + 20, 50, fmt::format("e: next frame", current_frame, total_frame));
    cvui::text(frame, IMAGE_SIZE + 20, 65, fmt::format("a: add new roi", current_frame, total_frame));
    cvui::text(frame, IMAGE_SIZE + 20, 80, fmt::format("d: remove selected roi", current_frame, total_frame));
    cvui::text(frame, IMAGE_SIZE + 10, 100, fmt::format("frame: [{}/{}]", current_frame, total_frame));
    return frame;
}

cv::Mat Labeler::load_image(int frame, bool cache) {
    auto node = image_cache.find(frame);
    if (node == image_cache.end()) {
        cap.set(cv::CAP_PROP_POS_FRAMES, current_frame);
        cv::Mat img;
        cap >> img;
        if (cache) image_cache[current_frame] = img;
        return img.clone();
    } else {
        return node->second.clone();
    }
}

void Labeler::next_frame() {
    for (auto &l: data[current_frame]) {
        l.is_track = false;
    }
    if (current_frame + jump_frame >= total_frame) {
        std::cout << "[WARNING]: already the final frame." << std::endl;
    } else {
        current_frame += jump_frame;
    }
    focused_id = -1;
}

void Labeler::last_frame() {
    for (auto &l: data[current_frame]) {
        l.is_track = false;
    }
    if (current_frame - jump_frame < 0) {
        std::cout << "[WARNING]: already the first frame." << std::endl;
    } else {
        current_frame -= jump_frame;
    }
    focused_id = -1;
}

void Labeler::add_target(const cv::Mat &im2show) {
    cv::Mat frame = draw_current_frame(im2show);
    cvui::text(frame, IMAGE_SIZE + 10, 120, "Select a ROI and then press SPACE or ENTER button!");
    cvui::text(frame, IMAGE_SIZE + 10, 140, "Cancel the selection process by pressing c button!");
    auto rect = cv::selectROI("roi", im2show);
    cv::destroyWindow("roi");
    std::string name;
    // get target name
    do {
        frame = draw_current_frame(im2show);
        cvui::text(frame, IMAGE_SIZE + 10, 120, fmt::format("Input target name: {}", name));
        cvui::text(frame, IMAGE_SIZE + 10, 140, fmt::format("Press ENTER to finish!", name));
        cvui::text(frame, IMAGE_SIZE + 10, 140, fmt::format("Press ESC to cancel!", name));
        cvui::imshow(WINDOW_NAME, frame);
        int k = cv::waitKey(20);
        if (('A' <= k && k <= 'Z') || ('a' <= k && k <= 'z')) {
            name.push_back(k);
        } else if (k == 13) {
            break;
        } else if (k == 27) {
            break;
        } else if (k == 8) {
            if (!name.empty()) name.pop_back();
        }
    } while (true);

    // get target id
    int id = track_id_factory++;
    do {
        frame = draw_current_frame(im2show);
        cvui::text(frame, IMAGE_SIZE + 10, 120, fmt::format("Input target id: {}", id));
        cvui::text(frame, IMAGE_SIZE + 10, 140, fmt::format("Press ENTER to finish!", name));
        cvui::text(frame, IMAGE_SIZE + 10, 140, fmt::format("Press ESC to cancel!", name));
        cvui::imshow(WINDOW_NAME, frame);
        int k = cv::waitKey(20);
        if ('0' <= k && k <= '9') {
            id = id * 10 + k - '0';
        } else if (k == 13) {
            double x = (rect.x + rect.width / 2.) * wr / ws;
            double y = (rect.y + rect.height / 2.) * hr / hs;
            double w = (double) rect.width * wr / ws;
            double h = (double) rect.height * hr / hs;
            data[current_frame].emplace_back(Label{name, id, x, y, w, h});
            break;
        } else if (k == 27) {
            break;
        } else if (k == 8) {
            id /= 10;
        }
    } while (true);
}

void Labeler::remove_target() {
    if (focused_id < 0 || focused_id >= data[current_frame].size()) return;
    data[current_frame].erase(data[current_frame].begin() + focused_id);
    focused_id = -1;
}

void Labeler::track_target() {
    if (current_frame - jump_frame < 0 || data[current_frame - jump_frame].empty()) return;
    for (const auto &l: data[current_frame - jump_frame]) {
        auto tracker = cv::TrackerMedianFlow::create();
        cv::Mat init = load_image(current_frame - jump_frame);
        tracker->init(init, {l.x - l.w / 2, l.y - l.h / 2, l.w, l.h});
        cv::Rect2d rect;
        bool miss = false;
        for (int f = current_frame - jump_frame + 1; f <= current_frame; f++) {
            cv::Mat img = load_image(f);
            if (!tracker->update(img, rect)) {
                miss = true;
                break;
            }
        }
        if (!miss && std::find_if(data[current_frame].begin(), data[current_frame].end(),
                                  [&](auto o) { return o.id == l.id; }) == data[current_frame].end()) {
            double x = rect.x + rect.width / 2.;
            double y = rect.y + rect.height / 2.;
            double w = rect.width;
            double h = rect.height;
            data[current_frame].emplace_back(Label{l.name, l.id, x, y, w, h, true});
        }
    }
}
