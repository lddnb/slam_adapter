#include "slam_core/point_cloud.hpp"
#include "slam_core/point_types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>
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