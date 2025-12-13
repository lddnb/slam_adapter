#pragma once

#include "slam_core/sensor/point_cloud.hpp"
#define TINYPLY_IMPLEMENTATION
#include "slam_core/utils/tinyply.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <spdlog/spdlog.h>

namespace ms_slam::slam_core
{
namespace ply_io
{
namespace detail
{

/**
 * @brief 判断Ply元素是否包含指定属性
 * @param element tinyply元素对象
 * @param name 属性名称
 * @return 是否找到对应属性
 */
inline bool ElementHasProperty(const tinyply::PlyElement& element, const std::string& name)
{
    return std::any_of(element.properties.begin(), element.properties.end(), [&](const tinyply::PlyProperty& prop) { return prop.name == name; });
}

/**
 * @brief 从Ply数据缓冲区读取指定类型的值
 * @tparam Target 需要转换的目标类型
 * @param data tinyply数据引用
 * @param index 数据索引
 * @return 转换后的目标类型数值
 * @throws std::out_of_range 当索引越界时抛出异常
 * @throws std::runtime_error 当遇到不支持的Ply类型时抛出异常
 */
template <typename Target>
Target ReadValue(const tinyply::PlyData& data, std::size_t index)
{
    const auto property_it = tinyply::PropertyTable.find(data.t);
    if (property_it == tinyply::PropertyTable.end() || property_it->second.stride == 0) {
        throw std::runtime_error("unsupported ply property type");
    }

    const auto total_entries = data.buffer.size_bytes() / static_cast<std::size_t>(property_it->second.stride);
    if (index >= total_entries) {
        throw std::out_of_range("ply data index out of range");
    }

    const auto* buffer = data.buffer.get_const();
    switch (data.t) {
        case tinyply::Type::INT8:
            return static_cast<Target>(reinterpret_cast<const int8_t*>(buffer)[index]);
        case tinyply::Type::UINT8:
            return static_cast<Target>(reinterpret_cast<const uint8_t*>(buffer)[index]);
        case tinyply::Type::INT16:
            return static_cast<Target>(reinterpret_cast<const int16_t*>(buffer)[index]);
        case tinyply::Type::UINT16:
            return static_cast<Target>(reinterpret_cast<const uint16_t*>(buffer)[index]);
        case tinyply::Type::INT32:
            return static_cast<Target>(reinterpret_cast<const int32_t*>(buffer)[index]);
        case tinyply::Type::UINT32:
            return static_cast<Target>(reinterpret_cast<const uint32_t*>(buffer)[index]);
        case tinyply::Type::FLOAT32:
            return static_cast<Target>(reinterpret_cast<const float*>(buffer)[index]);
        case tinyply::Type::FLOAT64:
            return static_cast<Target>(reinterpret_cast<const double*>(buffer)[index]);
        default:
            throw std::runtime_error("unsupported ply property type");
    }
}

/**
 * @brief 将模板类型映射为对应的Ply类型枚举
 * @tparam T 输入的标量类型
 * @param 无
 * @return tinyply::Type 对应的Ply标量类型
 */
template <typename T>
constexpr tinyply::Type PlyTypeFor()
{
    if constexpr (std::is_same_v<T, float>) {
        return tinyply::Type::FLOAT32;
    } else if constexpr (std::is_same_v<T, double>) {
        return tinyply::Type::FLOAT64;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        return tinyply::Type::UINT8;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        return tinyply::Type::INT8;
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
        return tinyply::Type::UINT16;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        return tinyply::Type::INT16;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return tinyply::Type::UINT32;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return tinyply::Type::INT32;
    } else {
        static_assert(::ms_slam::slam_core::detail::dependent_false<T>::value, "unsupported scalar type for ply io");
    }
}

/**
 * @brief 为点云描述符填充可选的标量字段
 * @tparam Descriptor 点云描述符类型
 * @param cloud 点云对象
 * @param index 点索引
 * @param data tinyply数据指针
 * @param accessor 字段访问器
 * @return 无
 */
template <typename Descriptor>
void AssignOptionalScalarField(PointCloud<Descriptor>& cloud, std::size_t index, const std::shared_ptr<tinyply::PlyData>& data, auto&& accessor)
{
    using Scalar = std::remove_reference_t<decltype(accessor(cloud, index))>;
    if (!data) {
        accessor(cloud, index) = Scalar{};
        return;
    }
    accessor(cloud, index) = detail::ReadValue<Scalar>(*data, index);
}

/**
 * @brief 为点云描述符填充可选的向量字段
 * @tparam Descriptor 点云描述符类型
 * @param cloud 点云对象
 * @param index 点索引
 * @param data tinyply数据指针
 * @param accessor 字段访问器
 * @return 无
 */
template <typename Descriptor>
void AssignOptionalVectorField(PointCloud<Descriptor>& cloud, std::size_t index, const std::shared_ptr<tinyply::PlyData>& data, auto&& accessor)
{
    auto field = accessor(cloud, index);
    using Scalar = std::remove_reference_t<decltype(field(0))>;
    if (!data) {
        // 当文件中缺失该字段时填充零值，保证下游流程拥有确定性输入
        for (Eigen::Index i = 0; i < field.size(); ++i) {
            field(i) = Scalar{};
        }
        return;
    }

    const std::size_t dims = static_cast<std::size_t>(field.size());
    for (std::size_t d = 0; d < dims; ++d) {
        // tinyply向量字段展平存储，需按维度索引恢复
        field(static_cast<Eigen::Index>(d)) = detail::ReadValue<Scalar>(*data, index * dims + d);
    }
}

}  // namespace detail

/**
 * @brief 从输入流读取Ply点云数据
 * @tparam Descriptor 点云描述符类型
 * @param input 输入流引用
 * @return 智能指针形式的点云对象
 * @throws std::runtime_error 当Ply头解析失败或关键字段缺失时抛出异常
 */
template <typename Descriptor>
typename PointCloud<Descriptor>::Ptr ReadPointCloud(std::istream& input)
{
    tinyply::PlyFile file;
    if (!file.parse_header(input)) {
        throw std::runtime_error("failed to parse ply header");
    }

    const auto elements = file.get_elements();
    const auto vertex_it =
        std::find_if(elements.begin(), elements.end(), [](const tinyply::PlyElement& element) { return element.name == "vertex"; });
    if (vertex_it == elements.end()) {
        throw std::runtime_error("ply file is missing vertex element");
    }

    auto has_property = [&](const std::string& name) { return detail::ElementHasProperty(*vertex_it, name); };

    if (!has_property("x") || !has_property("y") || !has_property("z")) {
        throw std::runtime_error("ply file is missing xyz coordinates");
    }

    std::shared_ptr<tinyply::PlyData> xyz_data = file.request_properties_from_element("vertex", {"x", "y", "z"});
    std::shared_ptr<tinyply::PlyData> intensity_data;
    std::shared_ptr<tinyply::PlyData> timestamp_data;
    std::shared_ptr<tinyply::PlyData> curvature_data;
    std::shared_ptr<tinyply::PlyData> rgb_data;
    std::shared_ptr<tinyply::PlyData> normal_data;

    if constexpr (has_field_v<IntensityTag, Descriptor>) {
        if (has_property("intensity")) {
            intensity_data = file.request_properties_from_element("vertex", {"intensity"});
        }
    }

    if constexpr (has_field_v<TimestampTag, Descriptor>) {
        if (has_property("timestamp")) {
            timestamp_data = file.request_properties_from_element("vertex", {"timestamp"});
        }
    }

    if constexpr (has_field_v<CurvatureTag, Descriptor>) {
        if (has_property("curvature")) {
            curvature_data = file.request_properties_from_element("vertex", {"curvature"});
        }
    }

    if constexpr (has_field_v<RGBTag, Descriptor>) {
        if (has_property("red") && has_property("green") && has_property("blue")) {
            rgb_data = file.request_properties_from_element("vertex", {"red", "green", "blue"});
        }
    }

    if constexpr (has_field_v<NormalTag, Descriptor>) {
        if (has_property("nx") && has_property("ny") && has_property("nz")) {
            normal_data = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
        }
    }

    file.read(input);

    const std::size_t vertex_count = vertex_it->size;
    if (!xyz_data) {
        throw std::runtime_error("unexpected vertex coordinate layout in ply file");
    }
    const auto xyz_property_it = tinyply::PropertyTable.find(xyz_data->t);
    if (xyz_property_it == tinyply::PropertyTable.end() || xyz_property_it->second.stride == 0) {
        throw std::runtime_error("unsupported vertex coordinate type in ply file");
    }
    const auto xyz_entry_count =
        xyz_data->buffer.size_bytes() / static_cast<std::size_t>(xyz_property_it->second.stride);
    if (xyz_entry_count != vertex_count * 3) {
        throw std::runtime_error("unexpected vertex coordinate layout in ply file");
    }

    auto cloud = std::make_shared<PointCloud<Descriptor>>(vertex_count);
    cloud->resize(vertex_count);

    using PositionScalar = typename PointCloud<Descriptor>::scalar_type;
    for (std::size_t i = 0; i < vertex_count; ++i) {
        // 将展平坐标写入点云内的向量表示
        auto position = cloud->position(i);
        position(0) = detail::ReadValue<PositionScalar>(*xyz_data, i * 3 + 0);
        position(1) = detail::ReadValue<PositionScalar>(*xyz_data, i * 3 + 1);
        position(2) = detail::ReadValue<PositionScalar>(*xyz_data, i * 3 + 2);

        if constexpr (has_field_v<IntensityTag, Descriptor>) {
            detail::AssignOptionalScalarField(*cloud, i, intensity_data, [](auto& c, std::size_t idx) -> auto& { return c.intensity(idx); });
        }

        if constexpr (has_field_v<TimestampTag, Descriptor>) {
            detail::AssignOptionalScalarField(*cloud, i, timestamp_data, [](auto& c, std::size_t idx) -> auto& { return c.timestamp(idx); });
        }

        if constexpr (has_field_v<CurvatureTag, Descriptor>) {
            detail::AssignOptionalScalarField(*cloud, i, curvature_data, [](auto& c, std::size_t idx) -> auto& { return c.curvature(idx); });
        }

        if constexpr (has_field_v<RGBTag, Descriptor>) {
            detail::AssignOptionalVectorField(*cloud, i, rgb_data, [](auto& c, std::size_t idx) { return c.rgb(idx); });
        }

        if constexpr (has_field_v<NormalTag, Descriptor>) {
            detail::AssignOptionalVectorField(*cloud, i, normal_data, [](auto& c, std::size_t idx) { return c.normal(idx); });
        }
    }

    return cloud;
}

/**
 * @brief 从文件路径加载Ply点云
 * @tparam Descriptor 点云描述符类型
 * @param filename Ply文件路径
 * @return 智能指针形式的点云对象
 * @throws std::runtime_error 当文件无法打开时抛出异常
 */
template <typename Descriptor>
typename PointCloud<Descriptor>::Ptr LoadPointCloud(const std::string& filename)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open ply file: " + filename);
    }
    return ReadPointCloud<Descriptor>(input);
}

/**
 * @brief 将点云写入Ply输出流
 * @tparam Descriptor 点云描述符类型
 * @param cloud 点云对象
 * @param output 输出流引用
 * @param binary 是否使用二进制格式
 * @throws std::runtime_error 当点云布局与期望不符时抛出异常
 * @return 无
 */
template <typename Descriptor>
void WritePointCloud(const PointCloud<Descriptor>& cloud, std::ostream& output, bool binary = true)
{
    tinyply::PlyFile file;
    const std::size_t vertex_count = cloud.size();

    const auto positions = cloud.positions();
    if (positions.size() != vertex_count * 3) {
        throw std::runtime_error("point cloud position layout mismatch");
    }

    using PositionScalar = typename PointCloud<Descriptor>::scalar_type;
    file.add_properties_to_element(
        "vertex",
        {"x", "y", "z"},
        detail::PlyTypeFor<PositionScalar>(),
        vertex_count,
        reinterpret_cast<const uint8_t*>(positions.data()),
        tinyply::Type::INVALID,
        0);

    if constexpr (has_field_v<IntensityTag, Descriptor>) {
        const auto intensity = cloud.template field_vector<IntensityTag>();
        file.add_properties_to_element(
            "vertex",
            {"intensity"},
            detail::PlyTypeFor<typename PointCloud<Descriptor>::template FieldScalarT<IntensityTag>>(),
            vertex_count,
            reinterpret_cast<const uint8_t*>(intensity.data()),
            tinyply::Type::INVALID,
            0);
    }

    if constexpr (has_field_v<TimestampTag, Descriptor>) {
        const auto timestamp = cloud.template field_vector<TimestampTag>();
        file.add_properties_to_element(
            "vertex",
            {"timestamp"},
            detail::PlyTypeFor<typename PointCloud<Descriptor>::template FieldScalarT<TimestampTag>>(),
            vertex_count,
            reinterpret_cast<const uint8_t*>(timestamp.data()),
            tinyply::Type::INVALID,
            0);
    }

    if constexpr (has_field_v<CurvatureTag, Descriptor>) {
        const auto curvature = cloud.template field_vector<CurvatureTag>();
        file.add_properties_to_element(
            "vertex",
            {"curvature"},
            detail::PlyTypeFor<typename PointCloud<Descriptor>::template FieldScalarT<CurvatureTag>>(),
            vertex_count,
            reinterpret_cast<const uint8_t*>(curvature.data()),
            tinyply::Type::INVALID,
            0);
    }

    if constexpr (has_field_v<RGBTag, Descriptor>) {
        const auto rgb = cloud.template field_matrix<RGBTag>();
        file.add_properties_to_element(
            "vertex",
            {"red", "green", "blue"},
            detail::PlyTypeFor<typename PointCloud<Descriptor>::template FieldScalarT<RGBTag>>(),
            vertex_count,
            reinterpret_cast<const uint8_t*>(rgb.data()),
            tinyply::Type::INVALID,
            0);
    }

    if constexpr (has_field_v<NormalTag, Descriptor>) {
        const auto normal = cloud.template field_matrix<NormalTag>();
        file.add_properties_to_element(
            "vertex",
            {"nx", "ny", "nz"},
            detail::PlyTypeFor<typename PointCloud<Descriptor>::template FieldScalarT<NormalTag>>(),
            vertex_count,
            reinterpret_cast<const uint8_t*>(normal.data()),
            tinyply::Type::INVALID,
            0);
    }

    file.write(output, binary);
}

/**
 * @brief 将点云保存为指定路径下的Ply文件
 * @tparam Descriptor 点云描述符类型
 * @param cloud 点云对象
 * @param filename 输出文件路径
 * @param binary 是否使用二进制格式
 * @throws std::runtime_error 当文件无法打开写入时抛出异常
 * @return 无
 */
template <typename Descriptor>
void SavePointCloud(const PointCloud<Descriptor>& cloud, const std::string& filename, bool binary = true)
{
    std::ofstream output(filename, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("failed to open ply file for writing: " + filename);
    }
    WritePointCloud(cloud, output, binary);
}

}  // namespace ply_io
}  // namespace ms_slam::slam_core
