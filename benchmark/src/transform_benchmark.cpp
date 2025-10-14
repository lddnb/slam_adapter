#include <benchmark/benchmark.h>

#include <vector>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <Eigen/Core>
#include <Eigen/Geometry>

// slam_core
#include <slam_core/point_cloud.hpp>

// small_gicp
#include <small_gicp/points/point_cloud.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

// ============================================================================
// Shared Test Data (generated once, used by all benchmarks)
// ============================================================================

struct TestData {
    std::vector<Eigen::Vector3f> points;
    Eigen::Isometry3f transform;
    Eigen::Isometry3f inv_transform;
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    Eigen::Matrix3f R_inv;
    Eigen::Vector3f t_inv;
    Eigen::Matrix4f T_4x4;
    Eigen::Matrix4f T_4x4_inv;

    TestData(size_t num_points, unsigned int seed = 42) {
        // Use fixed seed for reproducibility
        srand(seed);

        // Generate random points
        points.reserve(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            points.push_back(Eigen::Vector3f::Random() * 100.0f);
        }

        // Generate random transform
        transform = Eigen::Isometry3f::Identity();
        Eigen::Vector3f axis = Eigen::Vector3f::Random().normalized();
        float angle = M_PI / 6.0f;
        transform.rotate(Eigen::AngleAxisf(angle, axis));
        transform.pretranslate(Eigen::Vector3f::Random() * 10.0f);

        // Precompute inverse and matrix forms
        inv_transform = transform.inverse();
        R = transform.rotation();
        t = transform.translation();
        R_inv = R.transpose();
        t_inv = -R_inv * t;
        T_4x4 = transform.matrix();
        T_4x4_inv = T_4x4.inverse();
    }
};

// Global test data for different point cloud sizes
std::unique_ptr<TestData> g_test_data_1k;
std::unique_ptr<TestData> g_test_data_10k;
std::unique_ptr<TestData> g_test_data_100k;
std::unique_ptr<TestData> g_test_data_1m;

// Get test data for given size
const TestData& get_test_data(size_t num_points) {
    if (num_points == 1000) {
        if (!g_test_data_1k) g_test_data_1k = std::make_unique<TestData>(1000);
        return *g_test_data_1k;
    } else if (num_points == 10000) {
        if (!g_test_data_10k) g_test_data_10k = std::make_unique<TestData>(10000);
        return *g_test_data_10k;
    } else if (num_points == 100000) {
        if (!g_test_data_100k) g_test_data_100k = std::make_unique<TestData>(100000);
        return *g_test_data_100k;
    } else if (num_points == 1000000) {
        if (!g_test_data_1m) g_test_data_1m = std::make_unique<TestData>(1000000);
        return *g_test_data_1m;
    }
    throw std::runtime_error("Unsupported point cloud size");
}

// ============================================================================
// Benchmark 1: slam_core PointCloud (SoA)
// ============================================================================

static void BM_SlamCore_Transform(benchmark::State& state) {
    using namespace ms_slam::slam_core;

    const size_t num_points = state.range(0);
    const auto& test_data = get_test_data(num_points);

    // Pre-build source point cloud once
    PointCloudXYZ source_cloud(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        source_cloud.push_back(test_data.points[i]);
    }
    PointCloudXYZ cloud(num_points);

    // Benchmark loop
    for (auto _ : state) {
        state.PauseTiming();
        cloud = source_cloud;
        state.ResumeTiming();

        cloud.transform(test_data.transform);
        benchmark::DoNotOptimize(cloud.positions().data());
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * 3 * sizeof(float));
}

// ============================================================================
// Benchmark 2: small_gicp PointCloud (AoS with 4D homogeneous)
// ============================================================================

static void BM_SmallGICP_Transform(benchmark::State& state) {
    const size_t num_points = state.range(0);
    const auto& test_data = get_test_data(num_points);

    // Convert to double precision for small_gicp
    Eigen::Matrix4d T = test_data.T_4x4.cast<double>();

    // Pre-build source point cloud once
    small_gicp::PointCloud source_cloud;
    source_cloud.resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        source_cloud.point(i) << test_data.points[i].cast<double>().eval(), 1.0;
    }
    small_gicp::PointCloud cloud = source_cloud;

    // Benchmark loop
    for (auto _ : state) {
        state.PauseTiming();
        cloud = source_cloud;
        state.ResumeTiming();

        for (size_t i = 0; i < cloud.size(); ++i) {
            cloud.point(i) = T * cloud.point(i);
        }
        benchmark::DoNotOptimize(cloud.points.data());
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * 4 * sizeof(double));
}

// ============================================================================
// Benchmark 3: PCL PointCloud
// ============================================================================

static void BM_PCL_Transform(benchmark::State& state) {
    const size_t num_points = state.range(0);
    const auto& test_data = get_test_data(num_points);

    // Pre-build point cloud once using shared test data
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->points.reserve(num_points);
    for (const auto& pt : test_data.points) {
        cloud->points.emplace_back(pt.x(), pt.y(), pt.z());
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    // Pre-allocate the transformed cloud outside the loop
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZ>);
    transformed->points.reserve(num_points);

    // Benchmark loop
    for (auto _ : state) {
        // This is not an in-place transform, so we don't need to reset the source `cloud`.
        // We just overwrite the `transformed` cloud's contents each time.
        pcl::transformPointCloud(*cloud, *transformed, test_data.T_4x4);
        benchmark::DoNotOptimize(transformed->points.data());
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * 3 * sizeof(float));
}

// ============================================================================
// Benchmark 4: Naive Eigen Loop (baseline)
// ============================================================================

static void BM_NaiveEigenLoop_Transform(benchmark::State& state) {
    const size_t num_points = state.range(0);
    const auto& test_data = get_test_data(num_points);

    // Prepare a vector for transformation
    std::vector<Eigen::Vector3f> transformed(num_points);

    // Benchmark loop
    for (auto _ : state) {
        state.PauseTiming();
        // Reset data from original source for each iteration
        std::copy(test_data.points.begin(), test_data.points.end(), transformed.begin());
        state.ResumeTiming();

        for (auto& pt : transformed) {
            pt = test_data.R * pt + test_data.t;
        }
        benchmark::DoNotOptimize(transformed.data());
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * 3 * sizeof(float));
}

// ============================================================================
// Benchmark 5: Eigen Matrix (batch copy)
// ============================================================================

static void BM_EigenBatch_Transform(benchmark::State& state) {
    const size_t num_points = state.range(0);
    const auto& test_data = get_test_data(num_points);

    // Pre-allocate source matrix
    Eigen::Matrix3Xf source_mat(3, num_points);
    for (size_t i = 0; i < num_points; ++i) {
        source_mat.col(i) = test_data.points[i];
    }
    Eigen::Matrix3Xf mat(3, num_points);

    // Benchmark loop
    for (auto _ : state) {
        state.PauseTiming();
        mat = source_mat;
        state.ResumeTiming();

        mat = test_data.R * mat;
        mat.colwise() += test_data.t;
        benchmark::DoNotOptimize(mat.data());
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * 3 * sizeof(float));
}

// ============================================================================
// Benchmark 6: Eigen Map (zero-copy, ideal case)
// ============================================================================

static void BM_EigenMap_Transform(benchmark::State& state) {
    const size_t num_points = state.range(0);
    const auto& test_data = get_test_data(num_points);

    // Pre-allocate column-major source buffer
    std::vector<float> source_buffer(num_points * 3);
    Eigen::Map<Eigen::Matrix3Xf> source_map(source_buffer.data(), 3, num_points);
    for (size_t i = 0; i < num_points; ++i) {
        source_map.col(i) = test_data.points[i];
    }
    std::vector<float> buffer(num_points * 3);

    // Benchmark loop
    for (auto _ : state) {
        state.PauseTiming();
        buffer = source_buffer;
        state.ResumeTiming();

        Eigen::Map<Eigen::Matrix3Xf> mat(buffer.data(), 3, num_points);
        mat = test_data.R * mat;
        mat.colwise() += test_data.t;
        benchmark::DoNotOptimize(buffer.data());
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * 3 * sizeof(float));
}

// ============================================================================
// Register Benchmarks
// ============================================================================

// Test point cloud sizes: 1K, 10K, 100K, 1M
BENCHMARK(BM_SlamCore_Transform)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SmallGICP_Transform)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_PCL_Transform)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_NaiveEigenLoop_Transform)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_EigenBatch_Transform)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_EigenMap_Transform)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMillisecond);

// Run the benchmarks
BENCHMARK_MAIN();
