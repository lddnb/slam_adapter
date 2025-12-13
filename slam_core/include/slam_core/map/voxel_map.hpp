#pragma once

#include <cstdint>

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <manif/functions.h>

namespace ms_slam::slam_core
{

constexpr int64_t kVoxelHashP = 116101;
constexpr int64_t kVoxelHashMax = 10000000000;

/**
 * @brief 三维体素索引
 */
struct VoxelLoc {
    int64_t x = 0;  ///< x轴索引
    int64_t y = 0;  ///< y轴索引
    int64_t z = 0;  ///< z轴索引

    bool operator==(const VoxelLoc& other) const = default;
};

/**
 * @brief 体素哈希函数
 */
struct VoxelLocHash {
    std::size_t operator()(const VoxelLoc& loc) const
    {
        return static_cast<std::size_t>((((loc.z * kVoxelHashP) % kVoxelHashMax + loc.y) * kVoxelHashP) % kVoxelHashMax + loc.x);
    }
};

/**
 * @brief 带有协方差的点
 */
struct PointWithCov {
    Eigen::Vector3f body_point = Eigen::Vector3f::Zero();   ///< LiDAR坐标系下的点
    Eigen::Vector3f world_point = Eigen::Vector3f::Zero();  ///< 世界坐标系下的点
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();          ///< 世界系协方差
    Eigen::Matrix3d cov_lidar = Eigen::Matrix3d::Zero();    ///< LiDAR系协方差
};

/**
 * @brief 点到平面的匹配结果
 */
struct PointToPlaneResidual {
    Eigen::Vector3f body_point = Eigen::Vector3f::Zero();   ///< LiDAR坐标系下的点
    Eigen::Vector3f world_point = Eigen::Vector3f::Zero();  ///< 世界坐标系下的点
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();       ///< 平面法向量
    Eigen::Vector3f center = Eigen::Vector3f::Zero();       ///< 平面中心
    Eigen::Matrix<double, 6, 6> plane_cov = Eigen::Matrix<double, 6, 6>::Zero();  ///< 平面协方差
    double d = 0.0;      ///< 平面距离项
    int layer = 0;       ///< 所在层
    Eigen::Matrix3d cov_lidar = Eigen::Matrix3d::Zero();  ///< LiDAR系协方差
};

/**
 * @brief 平面描述结构体
 */
struct Plane {
    Eigen::Vector3f center = Eigen::Vector3f::Zero();  ///< 平面中心
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();  ///< 主法向
    Eigen::Vector3f y_normal = Eigen::Vector3f::Zero();  ///< 次法向
    Eigen::Vector3f x_normal = Eigen::Vector3f::Zero();  ///< 第三特征向量
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();  ///< 点协方差
    Eigen::Matrix<double, 6, 6> plane_cov = Eigen::Matrix<double, 6, 6>::Zero();  ///< 平面协方差
    float radius = 0.0F;  ///< 半径
    float min_eigen_value = 1.0F;  ///< 最小特征值
    float mid_eigen_value = 1.0F;  ///< 中间特征值
    float max_eigen_value = 1.0F;  ///< 最大特征值
    double d = 0.0;  ///< 平面距离项
    int points_size = 0;  ///< 点数量
    bool is_plane = false;  ///< 是否为平面
    bool is_init = false;   ///< 是否初始化
    int id = 0;             ///< 平面编号
    bool is_update = false;  ///< 更新标志
    int last_update_points_size = 0;  ///< 上次更新点数
    bool update_enable = true;        ///< 是否允许更新
};

/**
 * @brief 体素地图参数
 */
struct VoxelMapConfig {
    float voxel_size = 0.5F;                     ///< 体素尺寸
    int max_layer = 4;                           ///< 最大层数
    std::vector<int> layer_point_size = {5, 5, 5, 5, 5};  ///< 各层点数阈值
    int max_points_size = 1000;                  ///< 最大点数
    int max_cov_points_size = 1000;              ///< 允许累计协方差点数
    float planer_threshold = 0.01F;              ///< 平面阈值
    double sigma_num = 3.0;                      ///< 匹配容忍倍数
};

/**
 * @brief 协方差对比函数
 * @param x 点A
 * @param y 点B
 * @return 是否A协方差更小
 */
bool VarContrast(const PointWithCov& x, const PointWithCov& y);

class OctoTree;

/**
 * @brief 体素地图
 */
class VoxelMap
{
  public:
    /**
     * @brief 构造函数
     * @param config 地图参数
     */
    explicit VoxelMap(const VoxelMapConfig& config = VoxelMapConfig());

    /**
     * @brief 初始构建体素地图
     * @param input_points 输入点集
     */
    void Build(const std::vector<PointWithCov>& input_points);

    /**
     * @brief 单线程更新体素地图
     * @param input_points 新增点集
     */
    void Update(const std::vector<PointWithCov>& input_points);

    /**
     * @brief 并行更新体素地图
     * @param input_points 新增点集
     */
    void UpdateParallel(const std::vector<PointWithCov>& input_points);

    /**
     * @brief 构建匹配残差列表
     * @param pv_list 待匹配点
     * @param ptpl_list 匹配结果
     * @param non_match 未匹配点
     */
    void BuildResidualList(
        const std::vector<PointWithCov>& pv_list,
        std::vector<PointToPlaneResidual>& ptpl_list,
        std::vector<Eigen::Vector3d>& non_match) const;

    /**
     * @brief 构建匹配残差（不使用并行版本）
     * @param pv_list 待匹配点
     * @param ptpl_list 匹配结果
     * @param non_match 未匹配点
     */
    void BuildResidualListNormal(
        const std::vector<PointWithCov>& pv_list,
        std::vector<PointToPlaneResidual>& ptpl_list,
        std::vector<Eigen::Vector3d>& non_match) const;

    /**
     * @brief 获取更新的平面
     * @param pub_max_voxel_layer 发布层数
     * @param plane_list 输出平面
     */
    void GetUpdatedPlanes(int pub_max_voxel_layer, std::vector<Plane>& plane_list);

    /**
     * @brief 获取底层八叉树映射
     * @return 八叉树容器
     */
    [[nodiscard]] const std::unordered_map<VoxelLoc, std::unique_ptr<OctoTree>, VoxelLocHash>& Data() const { return voxel_map_; }

  private:
    /**
     * @brief 计算体素索引
     * @param point 输入点
     * @return 体素索引
     */
    VoxelLoc ComputeVoxelLoc(const Eigen::Vector3f& point) const;

    /**
     * @brief 构建单个残差
     * @param pv 输入点
     * @param current_octo 当前八叉树
     * @param current_layer 当前层级
     * @param is_success 匹配是否成功
     * @param prob 匹配概率
     * @param single_ptpl 输出残差
     */
    void BuildSingleResidual(
        const PointWithCov& pv,
        const OctoTree* current_octo,
        int current_layer,
        bool& is_success,
        double& prob,
        PointToPlaneResidual& single_ptpl) const;

    /**
     * @brief 尝试构建单点残差
     * @param pv 输入点
     * @param single_ptpl 输出残差
     * @return 匹配是否成功
     */
    bool TryBuildResidualForPoint(const PointWithCov& pv, PointToPlaneResidual& single_ptpl) const;

    /**
     * @brief 递归收集更新平面
     * @param current_octo 当前八叉树
     * @param pub_max_voxel_layer 发布层数
     * @param plane_list 平面集合
     */
    void CollectUpdatedPlanes(OctoTree* current_octo, int pub_max_voxel_layer, std::vector<Plane>& plane_list);

    /**
     * @brief 创建新的八叉树节点
     * @param position 体素索引
     * @return 新节点
     */
    std::unique_ptr<OctoTree> CreateOctoTree(const VoxelLoc& position);

    /**
     * @brief 获取或创建八叉树节点
     * @param position 体素索引
     * @return 节点指针
     */
    OctoTree* GetOrCreateTree(const VoxelLoc& position);

    VoxelMapConfig config_;
    std::unordered_map<VoxelLoc, std::unique_ptr<OctoTree>, VoxelLocHash> voxel_map_;
};

/**
 * @brief 单个八叉树节点
 */
class OctoTree
{
  public:
    /**
     * @brief 构造函数
     * @param max_layer 最大层数
     * @param layer 当前层
     * @param layer_point_size 分层点数阈值
     * @param max_point_size 最大点数
     * @param max_cov_points_size 最大协方差点数
     * @param planer_threshold 平面判定阈值
     */
    OctoTree(int max_layer, int layer, const std::vector<int>& layer_point_size, int max_point_size, int max_cov_points_size, float planer_threshold);

    /**
     * @brief 配置体素中心
     * @param center 中心坐标
     * @param quarter_length 四分之一体素边长
     */
    void ConfigureCenter(const Eigen::Vector3f& center, float quarter_length);

    /**
     * @brief 添加构图阶段点
     * @param point 输入点
     */
    void AddInitialPoint(const PointWithCov& point);

    /**
     * @brief 初始化八叉树
     */
   void Initialize();

    /**
     * @brief 更新八叉树
     * @param point 新点
     */
    void Update(const PointWithCov& point);

    /**
     * @brief 是否初始化完成
     * @return 初始化结果
     */
    [[nodiscard]] bool IsInitialized() const { return init_octo_; }

    /**
     * @brief 当前节点是否包含平面
     * @return 平面判定
     */
    [[nodiscard]] bool HasPlane() const { return plane_.is_plane; }

    /**
     * @brief 获取平面数据
     * @return 平面引用
     */
    [[nodiscard]] const Plane& PlaneData() const { return plane_; }

    /**
     * @brief 获取平面数据指针
     * @return 平面指针
     */
    Plane* MutablePlane() { return &plane_; }

    /**
     * @brief 获取四分之一体素长度
     * @return 四分之一长度
     */
    [[nodiscard]] float QuarterLength() const { return quater_length_; }

    /**
     * @brief 获取体素中心
     * @return 体素中心
     */
    [[nodiscard]] const Eigen::Vector3f& Center() const { return voxel_center_; }

    /**
     * @brief 获取当前层级
     * @return 层级
     */
    [[nodiscard]] int Layer() const { return layer_; }

    /**
     * @brief 获取最大层级
     * @return 最大层级
     */
    [[nodiscard]] int MaxLayer() const { return max_layer_; }

    /**
     * @brief 获取子节点
     * @param index 子节点索引
     * @return 子节点指针
     */
    [[nodiscard]] const OctoTree* GetChild(std::size_t index) const { return leaves_[index].get(); }

    /**
     * @brief 获取可写子节点
     * @param index 子节点索引
     * @return 子节点指针
     */
    OctoTree* MutableChild(std::size_t index) { return leaves_[index].get(); }

  private:
    /**
     * @brief 初始化平面
     * @param points 参与拟合的点
     */
    void InitPlane(const std::vector<PointWithCov>& points);

    /**
     * @brief 增量更新平面
     * @param points 新点集合
     */
    void UpdatePlane(const std::vector<PointWithCov>& points);

    /**
     * @brief 递归切分八叉树
     */
    void CutTree();

    /**
     * @brief 处理平面节点的增量更新
     * @param point 新点
     */
    void UpdatePlaneNode(const PointWithCov& point);

    /**
     * @brief 处理分支节点的递归更新
     * @param point 新点
     */
    void UpdateBranchNode(const PointWithCov& point);

    /**
     * @brief 计算子节点索引
     * @param point 点坐标
     * @return 子节点编号
     */
    std::size_t ComputeChildIndex(const Eigen::Vector3f& point) const;

    /**
     * @brief 计算子节点中心
     * @param child_index 子节点编号
     * @return 子节点中心
     */
    Eigen::Vector3f ChildCenter(std::size_t child_index) const;

    /**
     * @brief 确保子节点存在
     * @param child_index 子节点编号
     * @return 子节点指针
     */
    OctoTree* EnsureChild(std::size_t child_index);

    std::vector<PointWithCov> temp_points_;
    std::vector<PointWithCov> new_points_;
    Plane plane_;
    int max_layer_;
    int layer_;
    std::array<std::unique_ptr<OctoTree>, 8> leaves_;
    Eigen::Vector3f voxel_center_ = Eigen::Vector3f::Zero();
    std::vector<int> layer_point_size_;
    float quater_length_ = 0.0F;
    float planer_threshold_;
    int max_plane_update_threshold_;
    int update_size_threshold_;
    int all_points_num_;
    int new_points_num_;
    int max_points_size_;
    int max_cov_points_size_;
    bool init_octo_;
    bool update_cov_enable_;
    bool update_enable_;
};

/**
 * @brief 计算LiDAR点在机体系下的协方差
 * @param pb LiDAR点
 * @param range_inc 距离噪声
 * @param degree_inc 角度噪声
 * @return 协方差矩阵
 */
Eigen::Matrix3d CalcBodyCov(const Eigen::Vector3f& pb, float range_inc, float degree_inc);

template <typename State>
/**
 * @brief 将LiDAR协方差转换至世界系
 * @param p_lidar LiDAR坐标
 * @param T_i_l 外参
 * @param state 当前状态
 * @param COV_lidar LiDAR系协方差
 * @return 世界系协方差
 */
Eigen::Matrix3d TransformLiDARCovToWorld(const Eigen::Vector3f& p_lidar, const Eigen::Isometry3d& T_i_l, const State& state, const Eigen::Matrix3d& COV_lidar)
{
    Eigen::Matrix3d point_crossmat;
    point_crossmat << manif::skew(p_lidar.cast<double>());

    // lidar到body的方差传播
    // 注意外参的var是先rot 后pos
    Eigen::Matrix3d il_rot_var = Eigen::Matrix3d::Zero();
    il_rot_var.diagonal() << 0.00001, 0.00001, 0.00001;
    Eigen::Matrix3d il_t_var = Eigen::Matrix3d::Zero();
    il_t_var.diagonal() << 0.00001, 0.00001, 0.00001;

    Eigen::Matrix3d COV_body =
            T_i_l.rotation() * COV_lidar * T_i_l.rotation().transpose()
            + T_i_l.rotation() * (-point_crossmat) * il_rot_var * (-point_crossmat).transpose() * T_i_l.rotation().transpose()
            + il_t_var;

    // body的坐标
    Eigen::Vector3d p_body = T_i_l.rotation() * p_lidar.cast<double>() + T_i_l.translation();
    point_crossmat << manif::skew(p_body);
    Eigen::Matrix3d rot_var = state.cov().template block<3, 3>(3, 3);
    Eigen::Matrix3d t_var = state.cov().template block<3, 3>(0, 0);

    Eigen::Matrix3d cov_world = state.R() * COV_body * state.R().transpose() +
                                state.R() * (-point_crossmat) * rot_var * (-point_crossmat).transpose() * state.R().transpose() + t_var;

    return cov_world;
}

}  // namespace ms_slam::slam_core
