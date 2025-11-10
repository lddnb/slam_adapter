#pragma once

#ifdef USE_PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

struct EIGEN_ALIGN16 PointT {
    PCL_ADD_POINT4D;
    float intensity;
    union {
        double timestamp;
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(PointT, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(double, timestamp, timestamp))

typedef pcl::PointCloud<PointT> PointCloudT;
#endif