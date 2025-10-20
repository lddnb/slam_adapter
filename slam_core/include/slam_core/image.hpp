# pragma once
#include <opencv2/core/core.hpp>

namespace ms_slam::slam_core
{
class Image
{
public:
    Image() = default;

    Image(const cv::Mat& mat, double timestamp)
    : data_(mat)
    , timestamp_(timestamp)
    {
    }

    [[nodiscard]] int width() const noexcept { return data_.cols; }
    [[nodiscard]] int height() const noexcept { return data_.rows; }
    [[nodiscard]] int type() const noexcept { return data_.type(); }

    [[nodiscard]] cv::Mat& data() noexcept { return data_; }
    [[nodiscard]] const cv::Mat& data() const noexcept { return data_; }
    [[nodiscard]] double timestamp() const noexcept { return timestamp_; }
private:
    cv::Mat data_;
    double timestamp_;
};
} // namespace ms_slam::slam_core