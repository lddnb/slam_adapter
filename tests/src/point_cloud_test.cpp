#include "slam_core/point_cloud.hpp"
#include "slam_core/point_types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <iterator>
#include <ranges>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ms_slam::slam_core;
using Catch::Approx;

TEST_CASE("PointCloud DefaultConstructor", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud;
    REQUIRE(cloud.size() == 0);
    REQUIRE(cloud.empty());
    REQUIRE(cloud.capacity() == 0);
}

TEST_CASE("PointCloud SizedConstructor", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud(10);
    REQUIRE(cloud.size() == 10);
    REQUIRE_FALSE(cloud.empty());
    REQUIRE(cloud.capacity() >= 10);
}

TEST_CASE("PointCloud Resize", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud;
    cloud.resize(5);
    REQUIRE(cloud.size() == 5);
    cloud.resize(10);
    REQUIRE(cloud.size() == 10);
    cloud.resize(0);
    REQUIRE(cloud.size() == 0);
    REQUIRE(cloud.empty());
}

TEST_CASE("PointCloud Reserve", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud;
    cloud.reserve(100);
    REQUIRE(cloud.size() == 0);
    REQUIRE(cloud.capacity() >= 100);
}

TEST_CASE("PointCloud Clear", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud(10);
    cloud.clear();
    REQUIRE(cloud.size() == 0);
    REQUIRE(cloud.empty());
    REQUIRE(cloud.capacity() >= 10);
}

TEST_CASE("PointCloud Release", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud(10);
    cloud.release();
    REQUIRE(cloud.size() == 0);
    REQUIRE(cloud.capacity() == 0);
}

TEST_CASE("PointCloud PushBack", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud;
    cloud.push_back(1.0f, 2.0f, 3.0f);
    REQUIRE(cloud.size() == 1);
    auto pos = cloud.position(0);
    REQUIRE(pos.x() == Approx(1.0f));
    REQUIRE(pos.y() == Approx(2.0f));
    REQUIRE(pos.z() == Approx(3.0f));

    Eigen::Vector3f vec(4.0f, 5.0f, 6.0f);
    cloud.push_back(vec);
    REQUIRE(cloud.size() == 2);
    auto pos2 = cloud.position(1);
    REQUIRE(pos2.x() == Approx(4.0f));
    REQUIRE(pos2.y() == Approx(5.0f));
    REQUIRE(pos2.z() == Approx(6.0f));

    PointXYZ point{7.0f, 8.0f, 9.0f};
    point.x = 7.5f;
    point.y = 8.5f;
    point.z = 9.5f;
    cloud.push_back(point);
    auto pos3 = cloud.position(2);
    REQUIRE(pos3.x() == Approx(7.5f));
    REQUIRE(pos3.y() == Approx(8.5f));
    REQUIRE(pos3.z() == Approx(9.5f));
}

TEST_CASE("PointCloud PositionsMatrix", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud;
    cloud.push_back(1.0f, 2.0f, 3.0f);
    cloud.push_back(4.0f, 5.0f, 6.0f);

    auto matrix_map = cloud.positions_matrix();
    REQUIRE(matrix_map.rows() == 3);
    REQUIRE(matrix_map.cols() == 2);
    REQUIRE(matrix_map(0, 0) == Approx(1.0f));
    REQUIRE(matrix_map(1, 1) == Approx(5.0f));
}

TEST_CASE("PointCloud Transform", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud;
    cloud.push_back(1.0f, 0.0f, 0.0f);
    cloud.push_back(0.0f, 1.0f, 0.0f);

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << 10.0f, 20.0f, 30.0f;
    transform.rotate(Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitZ()));

    cloud.transform(transform);

    auto pos1 = cloud.position(0);
    REQUIRE(pos1.x() == Approx(10.0f).margin(1e-6));
    REQUIRE(pos1.y() == Approx(21.0f).margin(1e-6));
    REQUIRE(pos1.z() == Approx(30.0f).margin(1e-6));

    auto pos2 = cloud.position(1);
    REQUIRE(pos2.x() == Approx(9.0f).margin(1e-6));
    REQUIRE(pos2.y() == Approx(20.0f).margin(1e-6));
    REQUIRE(pos2.z() == Approx(30.0f).margin(1e-6));
}

TEST_CASE("PointCloud FieldAccess", "[PointCloud]")
{
    PointCloud<PointXYZIDescriptor> cloud;
    cloud.resize(1);

    auto intensity_span = cloud.field_span<IntensityTag>();
    intensity_span[0] = 42.0f;

    auto intensity_block = cloud.field_block<IntensityTag>(0);
    REQUIRE(intensity_block.size() == 1);
    REQUIRE(intensity_block[0] == Approx(42.0f));

    REQUIRE(cloud.intensity(0) == Approx(42.0f));
    cloud.intensity(0) = 24.0f;
    REQUIRE(cloud.intensity(0) == Approx(24.0f));
}

TEST_CASE("PointCloud FieldEigenMaps", "[PointCloud]")
{
    PointCloud<PointXYZINormalDescriptor> cloud;
    cloud.push_back(PointXYZINormal{1.0f, 2.0f, 3.0f, 0.5f, 0.0f, 0.0f, 1.0f});
    cloud.push_back(PointXYZINormal{4.0f, 5.0f, 6.0f, 1.5f, 0.0f, 1.0f, 0.0f});

    auto position_1 = cloud.position(1);
    position_1.x() = 10.0f;
    position_1.y() = 20.0f;
    position_1.z() = 30.0f;

    auto normals_map = cloud.field_matrix<NormalTag>();
    REQUIRE(normals_map.rows() == 3);
    REQUIRE(normals_map.cols() == 2);
    REQUIRE(normals_map(0, 0) == Approx(0.0f));
    REQUIRE(normals_map(2, 0) == Approx(1.0f));
    REQUIRE(normals_map(1, 1) == Approx(1.0f));

    auto intensity_vec = cloud.field_vector<IntensityTag>();
    REQUIRE(intensity_vec.size() == 2);
    intensity_vec(0) = 2.5f;
    intensity_vec(1) = 3.5f;
    auto const& const_cloud = cloud;
    auto const_normals = const_cloud.field_matrix<NormalTag>();
    auto const_intensity = const_cloud.field_vector<IntensityTag>();
    auto const_position = const_cloud.positions_matrix();
    REQUIRE(const_normals(2, 0) == Approx(1.0f));
    REQUIRE(const_intensity(0) == Approx(2.5f));
    REQUIRE(const_intensity(1) == Approx(3.5f));
    REQUIRE(const_position(0, 1) == Approx(10.0f));
    REQUIRE(const_position(1, 1) == Approx(20.0f));
    REQUIRE(const_position(2, 1) == Approx(30.0f));

    auto normal0 = cloud.normal(0);
    REQUIRE(normal0.rows() == 3);
    REQUIRE(normal0(2) == Approx(1.0f));

    auto normal1 = const_cloud.normal(1);
    REQUIRE(normal1(1) == Approx(1.0f));
}

TEST_CASE("PointCloud FieldViewSupportsRanges", "[PointCloud]")
{
    PointCloud<PointXYZINormalDescriptor> cloud;
    cloud.push_back(PointXYZINormal{1.0f, 2.0f, 3.0f, 0.5f, 0.0f, 1.0f, 0.0f});
    cloud.push_back(PointXYZINormal{4.0f, 5.0f, 6.0f, 0.8f, 1.0f, 0.0f, 0.0f});

    auto intensity_view = cloud.field_view<IntensityTag>();
    static_assert(std::ranges::random_access_range<decltype(intensity_view)>);
    std::vector<float> intensities;
    std::ranges::copy(intensity_view, std::back_inserter(intensities));
    REQUIRE(intensities.size() == 2);
    REQUIRE(intensities[0] == Approx(0.5f));
    REQUIRE(intensities[1] == Approx(0.8f));

    intensity_view[1] = 1.5f;
    REQUIRE(cloud.intensity(1) == Approx(1.5f));

    auto normal_view = cloud.field_view<NormalTag>();
    static_assert(std::ranges::random_access_range<decltype(normal_view)>);
    REQUIRE(normal_view.size() == cloud.size());
    auto first_normal = normal_view[0];
    REQUIRE(first_normal.size() == 3);
    first_normal[2] = 2.0f;
    REQUIRE(cloud.normal(0)(2) == Approx(2.0f));

    auto x_components_view = normal_view | std::views::transform([](auto normal) { return normal[0]; });
    std::vector<float> x_components;
    x_components.reserve(cloud.size());
    for (float value : x_components_view) {
        x_components.push_back(value);
    }
    REQUIRE(x_components.size() == 2);
    REQUIRE(x_components[0] == Approx(0.0f));
    REQUIRE(x_components[1] == Approx(1.0f));

    const auto& const_cloud = cloud;
    auto const_normal_view = const_cloud.field_view<NormalTag>();
    static_assert(std::ranges::random_access_range<decltype(const_normal_view)>);
    REQUIRE(const_normal_view.size() == const_cloud.size());
    REQUIRE(const_normal_view[0][2] == Approx(2.0f));
}

TEST_CASE("PointCloud SinglePointAccessors", "[PointCloud]")
{
    PointCloud<PointXYZINormalDescriptor> cloud;
    cloud.push_back(PointXYZINormal{0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.3f, 0.9f});
    cloud.push_back(PointXYZINormal{1.0f, 1.0f, 1.0f, 2.0f, 0.5f, 0.4f, 0.7f});

    auto normal0 = cloud.normal(0);
    REQUIRE(normal0.rows() == 3);
    normal0(0) = 1.0f;
    normal0(1) = 0.0f;
    normal0(2) = 0.0f;
    REQUIRE(cloud.normal(0)(0) == Approx(1.0f));

    cloud.intensity(1) = 3.5f;

    const auto& const_cloud = cloud;
    auto const_normal1 = const_cloud.normal(1);
    REQUIRE(const_normal1(0) == Approx(0.5f));
    REQUIRE(const_normal1(1) == Approx(0.4f));
    REQUIRE(const_normal1(2) == Approx(0.7f));

    const auto& intensity_ref = const_cloud.intensity(1);
    REQUIRE(intensity_ref == Approx(3.5f));
}

TEST_CASE("PointCloud RGBAccess", "[PointCloud]")
{
    PointCloud<PointXYZIRGBDescriptor> cloud;
    cloud.push_back(PointXYZIRGB{1.0f, 2.0f, 3.0f, 0.8f, 10, 20, 30});

    auto rgb_map = cloud.rgb(0);
    REQUIRE(rgb_map.rows() == 3);
    REQUIRE(rgb_map(0) == 10);
    REQUIRE(rgb_map(1) == 20);
    REQUIRE(rgb_map(2) == 30);

    rgb_map(1) = static_cast<std::uint8_t>(200);
    REQUIRE(cloud.rgb(0)(1) == static_cast<std::uint8_t>(200));

    const auto& const_cloud_rgb = cloud;
    auto const_rgb_map = const_cloud_rgb.rgb(0);
    REQUIRE(const_rgb_map(1) == static_cast<std::uint8_t>(200));
}

TEST_CASE("PointCloud CurvatureAccess", "[PointCloud]")
{
    PointCloud<PointXYZINormalCurvatureDescriptor> cloud;
    cloud.resize(1);
    cloud.curvature(0) = 0.75f;
    REQUIRE(cloud.curvature(0) == Approx(0.75f));

    const auto& const_cloud_curv = cloud;
    REQUIRE(const_cloud_curv.curvature(0) == Approx(0.75f));
}

TEST_CASE("PointCloud TimestampAccess", "[PointCloud]")
{
    PointCloud<PointXYZITDescriptor> cloud;
    cloud.push_back(PointXYZIT{1.0f, 2.0f, 3.0f, 4.0f, 123ULL});

    cloud.timestamp(0) = 456ULL;
    REQUIRE(cloud.timestamp(0) == 456ULL);

    const auto& const_cloud = cloud;
    REQUIRE(const_cloud.timestamp(0) == 456ULL);
}

struct CustomPoint {
    Eigen::Vector3f position;
    float intensity;
};

TEST_CASE("PointCloud PushBackCustomPoint", "[PointCloud]")
{
    PointCloud<PointXYZIDescriptor> cloud;
    CustomPoint p{ {1.0f, 2.0f, 3.0f}, 42.0f };
    cloud.push_back(p);

    REQUIRE(cloud.size() == 1);
    auto pos = cloud.position(0);
    REQUIRE(pos.x() == Approx(1.0f));
    REQUIRE(pos.y() == Approx(2.0f));
    REQUIRE(pos.z() == Approx(3.0f));

    auto intensity = cloud.field_block<IntensityTag>(0);
    REQUIRE(intensity[0] == Approx(42.0f));
    REQUIRE(cloud.intensity(0) == Approx(42.0f));
}

TEST_CASE("PointCloud AppendRange", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud;

    std::vector<PointXYZ> points = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    cloud.append(points);
    REQUIRE(cloud.size() == 2);
    REQUIRE(cloud.position(0).x() == Approx(1.0f));
    REQUIRE(cloud.position(1).z() == Approx(6.0f));

    std::vector<Eigen::Vector3f> eigen_points;
    eigen_points.emplace_back(7.0f, 8.0f, 9.0f);
    eigen_points.emplace_back(10.0f, 11.0f, 12.0f);
    cloud.append(eigen_points);
    REQUIRE(cloud.size() == 4);
    REQUIRE(cloud.position(3).x() == Approx(10.0f));
    REQUIRE(cloud.position(3).y() == Approx(11.0f));
    REQUIRE(cloud.position(3).z() == Approx(12.0f));

    cloud.append({PointXYZ{13.0f, 14.0f, 15.0f}});
    REQUIRE(cloud.size() == 5);
    auto last = cloud.position(4);
    REQUIRE(last.x() == Approx(13.0f));
    REQUIRE(last.y() == Approx(14.0f));
    REQUIRE(last.z() == Approx(15.0f));
}

TEST_CASE("PointCloud ParallelTransform", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud;
    for (int i = 0; i < 1000; ++i) {
        cloud.push_back(static_cast<float>(i), static_cast<float>(i * 2), static_cast<float>(i * 3));
    }

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << 10.0f, 20.0f, 30.0f;
    transform.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()));

    PointCloud<PointXYZDescriptor> cloud_parallel = cloud;
    cloud.transform(transform);
    cloud_parallel.transform_parallel(transform);

    for (size_t i = 0; i < cloud.size(); ++i) {
        auto pos1 = cloud.position(i);
        auto pos2 = cloud_parallel.position(i);
        REQUIRE(pos1.x() == Approx(pos2.x()).margin(1e-5));
        REQUIRE(pos1.y() == Approx(pos2.y()).margin(1e-5));
        REQUIRE(pos1.z() == Approx(pos2.z()).margin(1e-5));
    }
}

TEST_CASE("PointCloud CopyConstructorAndAssignment", "[PointCloud]")
{
    PointCloud<PointXYZIDescriptor> cloud1;
    cloud1.push_back(1.0f, 2.0f, 3.0f);
    auto intensity_span1 = cloud1.field_span<IntensityTag>();
    intensity_span1[0] = 100.0f;

    PointCloud<PointXYZIDescriptor> cloud2(cloud1);
    REQUIRE(cloud2.size() == 1);
    REQUIRE(cloud2.position(0).x() == Approx(1.0f));
    REQUIRE(cloud2.field_span<IntensityTag>()[0] == Approx(100.0f));

    PointCloud<PointXYZIDescriptor> cloud3;
    cloud3 = cloud1;
    REQUIRE(cloud3.size() == 1);
    REQUIRE(cloud3.position(0).y() == Approx(2.0f));
    REQUIRE(cloud3.field_span<IntensityTag>()[0] == Approx(100.0f));
}

TEST_CASE("PointCloud ComplexPointType", "[PointCloud]")
{
    PointCloud<PointXYZINormalDescriptor> cloud;
    cloud.resize(2);

    auto positions = cloud.positions_matrix();
    positions.col(0) << 1.0f, 2.0f, 3.0f;
    positions.col(1) << 4.0f, 5.0f, 6.0f;

    auto intensities = cloud.field_span<IntensityTag>();
    intensities[0] = 10.0f;
    intensities[1] = 20.0f;

    auto normals = cloud.field_span<NormalTag>();
    Eigen::Map<Eigen::Matrix<float, 3, 2>> normal_map(normals.data());
    normal_map.col(0) << 0.0f, 1.0f, 0.0f;
    normal_map.col(1) << 1.0f, 0.0f, 0.0f;

    REQUIRE(cloud.position(1).x() == Approx(4.0f));
    REQUIRE(cloud.field_block<IntensityTag>(0)[0] == Approx(10.0f));
    REQUIRE(cloud.field_block<NormalTag>(1)[0] == Approx(1.0f));
}

TEST_CASE("PointCloud ForEachPosition", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud;
    cloud.push_back(1.0f, 2.0f, 3.0f);
    cloud.push_back(4.0f, 5.0f, 6.0f);

    float sum_x = 0.0f;
    cloud.for_each_position([&](auto point) {
        sum_x += point.x();
        point.x() *= 2.0;
    });

    REQUIRE(sum_x == Approx(5.0f));
    REQUIRE(cloud.position(0).x() == Approx(2.0f));
    REQUIRE(cloud.position(1).x() == Approx(8.0f));
}

TEST_CASE("PointCloud Transformed", "[PointCloud]")
{
    PointCloud<PointXYZDescriptor> cloud;
    cloud.push_back(1.0f, 0.0f, 0.0f);

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << 5.0f, 0.0f, 0.0f;

    PointCloud<PointXYZDescriptor> transformed_cloud = cloud.transformed(transform);

    REQUIRE(cloud.size() == 1);
    REQUIRE(cloud.position(0).x() == Approx(1.0f));

    REQUIRE(transformed_cloud.size() == 1);
    REQUIRE(transformed_cloud.position(0).x() == Approx(6.0f));
}

TEST_CASE("PointCloud ExtractSubset", "[PointCloud]")
{
    PointCloud<PointXYZINormalDescriptor> cloud;
    cloud.push_back(PointXYZINormal{1.0f, 2.0f, 3.0f, 0.5f, 1.0f, 0.0f, 0.0f});
    cloud.push_back(PointXYZINormal{4.0f, 5.0f, 6.0f, 1.5f, 0.0f, 1.0f, 0.0f});
    cloud.push_back(PointXYZINormal{7.0f, 8.0f, 9.0f, 2.5f, 0.0f, 0.0f, 1.0f});
    cloud.push_back(PointXYZINormal{10.0f, 11.0f, 12.0f, 3.5f, 1.0f, 1.0f, 0.0f});

    auto subset = cloud.extract(std::vector<std::size_t>{2, 0});
    REQUIRE(subset.size() == 2);
    auto first = subset.position(0);
    REQUIRE(first.x() == Approx(7.0f));
    REQUIRE(first.y() == Approx(8.0f));
    REQUIRE(first.z() == Approx(9.0f));
    REQUIRE(subset.intensity(0) == Approx(2.5f));
    REQUIRE(subset.normal(0)(2) == Approx(1.0f));

    auto view = std::views::iota(std::size_t{0}, cloud.size()) | std::views::filter([](std::size_t idx) { return idx % 2 == 0; });
    auto even_subset = cloud.extract(view);
    REQUIRE(even_subset.size() == 2);
    REQUIRE(even_subset.position(0).x() == Approx(1.0f));
    REQUIRE(even_subset.position(1).x() == Approx(7.0f));
    REQUIRE(even_subset.intensity(0) == Approx(0.5f));
    REQUIRE(even_subset.intensity(1) == Approx(2.5f));

    REQUIRE_THROWS_AS(cloud.extract(std::vector<std::size_t>{4}), std::out_of_range);
}
