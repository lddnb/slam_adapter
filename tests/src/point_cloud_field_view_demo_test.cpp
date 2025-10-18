#include "slam_core/point_cloud.hpp"
#include "slam_core/point_types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

using namespace ms_slam::slam_core;
using Catch::Approx;

TEST_CASE("PointCloud field_view exposes scalar spans", "[PointCloud][Ranges]")
{
    PointCloud<PointXYZIDescriptor> cloud;
    cloud.resize(3);

    auto intensities = cloud.field_view<IntensityTag>();

    static_assert(std::ranges::contiguous_range<decltype(intensities)>);
    static_assert(std::is_same_v<std::ranges::range_value_t<decltype(intensities)>, float>);

    // field_view<IntensityTag>() 返回的是直接指向底层强度数组的 std::span，算法可以原地写入无需拷贝
    std::iota(intensities.begin(), intensities.end(), 1.0f);

    auto doubled = intensities | std::views::transform([](float value) { return value * 2.0f; });

    std::vector<float> result;
    result.reserve(intensities.size());
    std::ranges::copy(doubled, std::back_inserter(result));

    REQUIRE(result.size() == 3);
    REQUIRE(result[0] == Approx(2.0f));
    REQUIRE(result[1] == Approx(4.0f));
    REQUIRE(result[2] == Approx(6.0f));

    REQUIRE(cloud.intensity(0) == Approx(1.0f));
    REQUIRE(cloud.intensity(1) == Approx(2.0f));
    REQUIRE(cloud.intensity(2) == Approx(3.0f));
}

TEST_CASE("PointCloud field_view exposes span-per-point ranges", "[PointCloud][Ranges]")
{
    PointCloud<PointXYZINormalDescriptor> cloud;
    cloud.push_back(PointXYZINormal{1.0f, 0.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.0f});
    cloud.push_back(PointXYZINormal{0.0f, 1.0f, 0.0f, 1.5f, 0.0f, 1.0f, 0.0f});
    cloud.push_back(PointXYZINormal{0.0f, 0.0f, 1.0f, 2.5f, 0.0f, 0.0f, 1.0f});

    auto normals = cloud.field_view<NormalTag>();

    static_assert(std::ranges::random_access_range<decltype(normals)>);
    static_assert(!std::ranges::contiguous_range<decltype(normals)>);
    static_assert(std::is_same_v<std::ranges::range_value_t<decltype(normals)>, std::span<float>>);

    // 每次解引用返回一个 span<float>，表示单个法向量，允许像访问二维数组一样定位到分量
    normals[1][2] = 5.0f;

    auto z_components = normals | std::views::transform([](std::span<float> normal) { return normal[2]; });

    std::vector<float> z_values;
    z_values.reserve(cloud.size());
    std::ranges::copy(z_components, std::back_inserter(z_values));

    REQUIRE(z_values.size() == 3);
    REQUIRE(z_values[0] == Approx(0.0f));
    REQUIRE(z_values[1] == Approx(5.0f));
    REQUIRE(z_values[2] == Approx(1.0f));

    auto second_normal = cloud.field_point<NormalTag>(1);
    REQUIRE(second_normal(2) == Approx(5.0f));
}

TEST_CASE("PointCloud const field_view composes with views", "[PointCloud][Ranges]")
{
    PointCloud<PointXYZINormalCurvatureDescriptor> cloud;
    cloud.resize(4);

    auto curvature = cloud.field_view<CurvatureTag>();
    // span 提供直接写入底层缓冲区的能力，这里使用预设数组填充曲率数据，避免额外复制
    const std::array<float, 4> values{0.25f, 0.35f, 0.45f, 0.55f};
    std::ranges::copy(values, curvature.begin());

    const auto& const_cloud = cloud;
    auto const_curvature = const_cloud.field_view<CurvatureTag>();

    static_assert(std::ranges::view<decltype(const_curvature)>);
    static_assert(std::is_same_v<std::ranges::range_value_t<decltype(const_curvature)>, float>);

    // const 版本的 field_view 依旧是一个 view，可以与标准视图组合生成惰性流水线

    auto filtered = const_curvature | std::views::filter([](float value) { return value > 0.4f; })
        | std::views::transform([](float value) { return value * 10.0f; });

    std::vector<float> scaled;
    // 结合 filter / transform 视图演示惰性求值：只会在 copy 时真正遍历并生成结果
    std::ranges::copy(filtered, std::back_inserter(scaled));

    REQUIRE(scaled.size() == 2);
    REQUIRE(scaled[0] == Approx(4.5f));
    REQUIRE(scaled[1] == Approx(5.5f));

    auto sum = std::accumulate(const_curvature.begin(), const_curvature.end(), 0.0f);
    REQUIRE(sum == Approx(0.25f + 0.35f + 0.45f + 0.55f));
}
