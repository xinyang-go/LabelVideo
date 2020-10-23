//
// Created by xinyang on 2020/10/23.
//

#ifndef _LABEL_HPP_
#define _LABEL_HPP_

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

/**
 * @brief 标签类
 */
struct Label {
    /// 目标名称，即目标类别
    std::string name;
    /// 标签id:在目标追踪过程中，相同id的标签表示同一个目标
    int id;
    /// 目标中心点x坐标
    double x;
    /// 目标中心点y坐标
    double y;
    /// 目标宽度
    double w;
    /// 目标高度
    double h;
    /// 标志当前标签是新跟踪得到的
    bool is_track{false};
};

using Labels = std::vector<Label>;
using LabelsMap = std::map<int, Labels>;

namespace YAML {
    /**
     * @brief YAML自定义用户类型
     */
    template<>
    struct convert<Label> {
        static Node encode(const Label &obj) {
            Node node;
            node["name"] = obj.name;
            node["id"] = obj.id;
            node["x"] = obj.x;
            node["y"] = obj.y;
            node["w"] = obj.w;
            node["h"] = obj.h;
            return node;
        }

        static bool decode(const Node &n, Label &obj) {
            if (!n.IsMap() || n.size() != 6) {
                return false;
            }
            if (!n["name"]) return false;
            obj.name = n["name"].as<std::string>();
            if (!n["id"]) return false;
            obj.id = n["id"].as<int>();
            if (!n["x"]) return false;
            obj.x = n["x"].as<double>();
            if (!n["y"]) return false;
            obj.y = n["y"].as<double>();
            if (!n["w"]) return false;
            obj.w = n["w"].as<double>();
            if (!n["h"]) return false;
            obj.h = n["h"].as<double>();
            return true;
        }
    };
}


class Labeler {
public:
    /**
     * @brief 创建一个标注器
     * @param video 待标注视频路径
     * @param jump 每隔几帧标注一张图片
     */
    explicit Labeler(const std::string &video, int jump = 1);

    /**
     * @brief 创建一个标注器，并读取已有的标注数据
     * @param video 待标注视频路径
     * @param yaml 已标注的yaml数据文件路径
     * @param jump 每隔几帧标注一张图片
     */
    Labeler(const std::string &video, const std::string &yaml, int jump = 1);

    /**
     * @brief 将当前标注数据保存到yaml文件
     * @param file yaml文件路径
     */
    void save_yaml(const std::string &file);

    /**
     * @brief 从已标注好的yaml文件读取标注数据
     * @param file yaml文件路径
     */
    void load_yaml(const std::string &file);

    /**
     * @brief 开始进行标注
     */
    void start();

private:
    /**
     * @brief 绘制当前帧
     * @param im2show 需要显示的图像
     * @return 当前帧
     */
    cv::Mat draw_current_frame(const cv::Mat &im2show);

    /**
     * @brief 进入下一帧
     */
    void next_frame();

    /**
     * @brief 回到上一帧
     */
    void last_frame();

    /**
     * @brief 添加一个目标
     */
    void add_target(const cv::Mat &frame);

    /**
     * @brief 删除一个目标
     */
    void remove_target();

    /**
     * @brief 从上一帧追踪当前帧目标
     */
    void track_target();

    /**
     * @brief 加载当前帧的图像
     * @param frame 加载第几帧的图像
     * @param cache 是否将该图像加入缓存
     * @return 加载的图像
     */
    cv::Mat load_image(int frame, bool cache = true);

    /// 视频流
    cv::VideoCapture cap;
    /// 标注数据
    LabelsMap data;
    /// 图像缓存
    std::map<int, cv::Mat> image_cache;
    /// 追踪ID
    int track_id_factory;
    /// 每隔几帧进行一次标注
    const int jump_frame;
    /// 当前帧数
    int current_frame;
    /// 总帧数
    const int total_frame;
    /// 原始图像宽度
    const int wr;
    /// 原始图像高度
    const int hr;
    /// 图像显示宽度
    const int ws;
    /// 图像显示高度
    const int hs;
    /// 图像显示区域
    const cv::Rect roi;
    /// 当前选中的框
    int focused_id;
};

#endif /* _LABEL_HPP */
