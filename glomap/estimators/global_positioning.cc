#include "glomap/estimators/global_positioning.h"

#include "glomap/estimators/cost_function.h"

#include <fstream>
#include <sstream>
#include <iostream>

namespace glomap {
namespace {

Eigen::Vector3d RandVector3d(std::mt19937& random_generator,
                             double low,
                             double high) {
  std::uniform_real_distribution<double> distribution(low, high);
  return Eigen::Vector3d(distribution(random_generator),
                         distribution(random_generator),
                         distribution(random_generator));
}

}  // namespace

// L2 正则化的残差项
struct PositionRegularization {
    PositionRegularization(const Eigen::Vector3d& initial_position)
        : initial_position_(initial_position) {}

    template <typename T>
    bool operator()(const T* const position, T* residual) const {
        // 计算当前位置和初始位置的差异 (L2 正则化)
        residual[0] = position[0] - T(initial_position_[0]);
        residual[1] = position[1] - T(initial_position_[1]);
        residual[2] = position[2] - T(initial_position_[2]);
        return true;
    }

    const Eigen::Vector3d initial_position_;  // 初始相机位置
};

GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options)
    : options_(options) {
  random_generator_.seed(options_.seed);
}

bool GlobalPositioner::Solve(const ViewGraph& view_graph,
                             std::unordered_map<camera_t, Camera>& cameras,
                             std::unordered_map<image_t, Image>& images,
                             std::unordered_map<track_t, Track>& tracks) {
  if (images.empty()) {
    LOG(ERROR) << "Number of images = " << images.size();
    return false;
  }
  if (view_graph.image_pairs.empty() &&
      options_.constraint_type != GlobalPositionerOptions::ONLY_POINTS) {
    LOG(ERROR) << "Number of image_pairs = " << view_graph.image_pairs.size();
    return false;
  }
  if (tracks.empty() &&
      options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    LOG(ERROR) << "Number of tracks = " << tracks.size();
    return false;
  }

  LOG(INFO) << "Setting up the global positioner problem";

  // Setup the problem.
  SetupProblem(view_graph, tracks);

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(view_graph, images, tracks);

  // Add the camera to camera constraints to the problem.
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_POINTS) {
    AddCameraToCameraConstraints(view_graph, images);
  }

  // Add the point to camera constraints to the problem.
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    AddPointToCameraConstraints(cameras, images, tracks);
  }

  AddCamerasAndPointsToParameterGroups(images, tracks);

  // // 为相机位置添加 L2 正则化，将相机位置保持在初始值附近
  // for (auto& [image_id, image] : images) {
  //     Eigen::Vector3d initial_position = image.cam_from_world.translation;

  //     // 设置正则化项的权重
  //     double regularization_weight = 1;
      
  //     // 使用 ScaledLoss 来设置权重
  //     ceres::LossFunction* loss_function = new ceres::ScaledLoss(nullptr, regularization_weight, ceres::TAKE_OWNERSHIP);

  //     // 创建 L2 正则化残差块
  //     ceres::CostFunction* position_regularizer = 
  //         new ceres::AutoDiffCostFunction<PositionRegularization, 3, 3>(
  //             new PositionRegularization(initial_position));
      
  //     // 将残差块添加到问题中，并应用带权重的损失函数
  //     problem_->AddResidualBlock(position_regularizer, loss_function, image.cam_from_world.translation.data());
  // }

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(images, tracks);

  LOG(INFO) << "Solving the global positioner problem";

  ceres::Solver::Summary summary;
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  ConvertResults(images);
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(
    const ViewGraph& view_graph,
    const std::unordered_map<track_t, Track>& tracks) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  // Allocate enough memory for the scales. One for each residual.
  // Due to possibly invalid image pairs or tracks, the actual number of
  // residuals may be smaller.
  scales_.clear();
  scales_.reserve(
      view_graph.image_pairs.size() +
      std::accumulate(tracks.begin(),
                      tracks.end(),
                      0,
                      [](int sum, const std::pair<track_t, Track>& track) {
                        return sum + track.second.observations.size();
                      }));
}

// 根据 image_id (即行号) 从 txt 文件中获取对应行的 translation 值
Eigen::Vector3d GetTranslationFromTxt(const std::string& filename, int image_id) {
    std::ifstream infile(filename);
    std::string line;
    int current_line = 0;

    // 遍历文件中的每一行
    while (std::getline(infile, line)) {
        current_line++;

        // 当行号等于 image_id 时，处理该行
        if (current_line == image_id) {
            std::istringstream iss(line);
            std::string dummy_column;  // 用来忽略第一列
            double x, y, z;

            // 读取第1列并跳过，接着读取第2、3、4列
            if (std::getline(iss, dummy_column, ',') && iss >> x && iss.get() && iss >> y && iss.get() && iss >> z) {
                return Eigen::Vector3d(x, y, z);  // 返回读取到的 translation
            }
        }
    }
    // 如果没有找到对应的行，返回一个默认的零向量
    return Eigen::Vector3d::Zero();
}

void GlobalPositioner::InitializeRandomPositions(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {

 std::string filename = "/home/hjl/data/translation.txt";

  std::unordered_set<image_t> constrained_positions;
  constrained_positions.reserve(images.size());
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;

    constrained_positions.insert(image_pair.image_id1);
    constrained_positions.insert(image_pair.image_id2);
  }

  if (options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    for (const auto& [track_id, track] : tracks) {
      if (track.observations.size() < options_.min_num_view_per_track) continue;
      for (const auto& observation : tracks[track_id].observations) {
        if (images.find(observation.first) == images.end()) continue;
        Image& image = images[observation.first];
        if (!image.is_registered) continue;
        constrained_positions.insert(observation.first);
      }
    }
  }

  if (!options_.generate_random_positions || !options_.optimize_positions) {
    for (auto& [image_id, image] : images) {
      // image.cam_from_world.translation = image.Center();
      Eigen::Vector3d translation_from_file = GetTranslationFromTxt(filename, image_id); // 根据 image_id 从文件中获取 translation 值
      if (translation_from_file != Eigen::Vector3d::Zero()) {
              image.cam_from_world.translation = translation_from_file;
          } else {
              // 如果文件中没有找到，使用随机位置
              image.cam_from_world.translation = image.Center();
              std::cout << "image_id: " << image_id << ", translation: " << image.cam_from_world.translation.transpose() << std::endl;     ////////
          }
    }
    return;
  }

  // Generate random positions for the cameras centers.
  for (auto& [image_id, image] : images) {
    Eigen::Vector3d translation_from_file = GetTranslationFromTxt(filename, image_id); // 根据 image_id 从文件中获取 translation 值
    // Only set the cameras to be random if they are needed to be optimized
    if (translation_from_file != Eigen::Vector3d::Zero()) {
            image.cam_from_world.translation = translation_from_file;
        } else {
            // 如果文件中没有找到，使用随机位置
            image.cam_from_world.translation = 100.0 * RandVector3d(random_generator_, -1, 1);
            std::cout << "image_id: " << image_id << ", translation: " << image.cam_from_world.translation.transpose() << std::endl;     ////////
        }
    // if (constrained_positions.find(image_id) != constrained_positions.end())
    //   image.cam_from_world.translation =
    //       100.0 * RandVector3d(random_generator_, -1, 1);
    // else
    //   image.cam_from_world.translation = image.Center();
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddCameraToCameraConstraints(
    const ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;

    const image_t image_id1 = image_pair.image_id1;
    const image_t image_id2 = image_pair.image_id2;
    if (images.find(image_id1) == images.end() ||
        images.find(image_id2) == images.end()) {
      continue;
    }

    CHECK_GT(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& scale = scales_.emplace_back(1);

    const Eigen::Vector3d translation =
        -(images[image_id2].cam_from_world.rotation.inverse() *
          image_pair.cam2_from_cam1.translation);
    ceres::CostFunction* cost_function =
        BATAPairwiseDirectionError::Create(translation);
    problem_->AddResidualBlock(
        cost_function,
        options_.loss_function.get(),
        images[image_id1].cam_from_world.translation.data(),
        images[image_id2].cam_from_world.translation.data(),
        &scale);

    problem_->SetParameterLowerBound(&scale, 0, 1e-5);
  }

  VLOG(2) << problem_->NumResidualBlocks()
          << " camera to camera constraints were added to the position "
             "estimation problem.";
}

void GlobalPositioner::AddPointToCameraConstraints(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // The number of camera-to-camera constraints coming from the relative poses

  const size_t num_cam_to_cam = problem_->NumResidualBlocks();
  // Find the tracks that are relevant to the current set of cameras
  const size_t num_pt_to_cam = tracks.size();

  VLOG(2) << num_pt_to_cam
          << " point to camera constriants were added to the position "
             "estimation problem.";

  if (num_pt_to_cam == 0) return;

  double weight_scale_pt = 1.0;
  // Set the relative weight of the point to camera constraints based on
  // the number of camera to camera constraints.
  if (num_cam_to_cam > 0 &&
      options_.constraint_type ==
          GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED) {
    weight_scale_pt = options_.constraint_reweight_scale *
                      static_cast<double>(num_cam_to_cam) /
                      static_cast<double>(num_pt_to_cam);
  }
  VLOG(2) << "Point to camera weight scaled: " << weight_scale_pt;

  if (loss_function_ptcam_uncalibrated_ == nullptr) {
    loss_function_ptcam_uncalibrated_ =
        std::make_shared<ceres::ScaledLoss>(options_.loss_function.get(),
                                            0.5 * weight_scale_pt,
                                            ceres::DO_NOT_TAKE_OWNERSHIP);
  }

  if (options_.constraint_type ==
      GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED) {
    loss_function_ptcam_calibrated_ =
        std::make_shared<ceres::ScaledLoss>(options_.loss_function.get(),
                                            weight_scale_pt,
                                            ceres::DO_NOT_TAKE_OWNERSHIP);
  } else {
    loss_function_ptcam_calibrated_ = options_.loss_function;
  }

  for (auto& [track_id, track] : tracks) {
    if (track.observations.size() < options_.min_num_view_per_track) continue;

    // Only set the points to be random if they are needed to be optimized
    if (options_.optimize_points && options_.generate_random_points) {
      track.xyz = 100.0 * RandVector3d(random_generator_, -1, 1);
      track.is_initialized = true;
    }

    AddTrackToProblem(track_id, cameras, images, tracks);
  }
}

void GlobalPositioner::AddTrackToProblem(
    track_t track_id,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // For each view in the track add the point to camera correspondences.
  for (const auto& observation : tracks[track_id].observations) {
    if (images.find(observation.first) == images.end()) continue;

    Image& image = images[observation.first];
    if (!image.is_registered) continue;

    const Eigen::Vector3d& feature_undist =
        image.features_undist[observation.second];
    if (feature_undist.array().isNaN().any()) {
      LOG(WARNING)
          << "Ignoring feature because it failed to undistort: track_id="
          << track_id << ", image_id=" << observation.first
          << ", feature_id=" << observation.second;
      continue;
    }

    const Eigen::Vector3d translation =
        image.cam_from_world.rotation.inverse() *
        image.features_undist[observation.second];
    ceres::CostFunction* cost_function =
        BATAPairwiseDirectionError::Create(translation);

    CHECK_GT(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& scale = scales_.emplace_back(1);
    if (!options_.generate_scales && tracks[track_id].is_initialized) {
      const Eigen::Vector3d trans_calc =
          tracks[track_id].xyz - image.cam_from_world.translation;
      scale = std::max(1e-5,
                       translation.dot(trans_calc) / trans_calc.squaredNorm());
    }

    // For calibrated and uncalibrated cameras, use different loss functions
    // Down weight the uncalibrated cameras
    if (cameras[image.camera_id].has_prior_focal_length) {
      problem_->AddResidualBlock(cost_function,
                                 loss_function_ptcam_calibrated_.get(),
                                 image.cam_from_world.translation.data(),
                                 tracks[track_id].xyz.data(),
                                 &scale);
    } else {
      problem_->AddResidualBlock(cost_function,
                                 loss_function_ptcam_uncalibrated_.get(),
                                 image.cam_from_world.translation.data(),
                                 tracks[track_id].xyz.data(),
                                 &scale);
    }

    problem_->SetParameterLowerBound(&scale, 0, 1e-5);
  }
}

void GlobalPositioner::AddCamerasAndPointsToParameterGroups(
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

  // Add scale parameters to group 0 (large and independent)
  for (double& scale : scales_) {
    parameter_ordering->AddElementToGroup(&scale, 0);
  }

  // Add point parameters to group 1.
  int group_id = 1;
  if (tracks.size() > 0) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data()))
        parameter_ordering->AddElementToGroup(track.xyz.data(), group_id);
    }
    group_id++;
  }

  // Add camera parameters to group 2 if there are tracks, otherwise group 1.
  for (auto& [image_id, image] : images) {
    if (problem_->HasParameterBlock(image.cam_from_world.translation.data())) {
      parameter_ordering->AddElementToGroup(
          image.cam_from_world.translation.data(), group_id);
    }
  }
}

void GlobalPositioner::ParameterizeVariables(
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // For the global positioning, do not set any camera to be constant for easier
  // convergence

  // If do not optimize the positions, set the camera positions to be constant
  if (!options_.optimize_positions) {
    for (auto& [image_id, image] : images)
      if (problem_->HasParameterBlock(image.cam_from_world.translation.data()))
        problem_->SetParameterBlockConstant(
            image.cam_from_world.translation.data());
  }

  // If do not optimize the rotations, set the camera rotations to be constant
  if (!options_.optimize_points) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data())) {
        problem_->SetParameterBlockConstant(track.xyz.data());
      }
    }
  }

  // If do not optimize the scales, set the scales to be constant
  if (!options_.optimize_scales) {
    for (double& scale : scales_) {
      problem_->SetParameterBlockConstant(&scale);
    }
  }

  // Set up the options for the solver
  // Do not use iterative solvers, for its suboptimal performance.
  if (tracks.size() > 0) {
    options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
  } else {
    options_.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.solver_options.preconditioner_type = ceres::JACOBI;
  }
}

void GlobalPositioner::ConvertResults(
    std::unordered_map<image_t, Image>& images) {
  // translation now stores the camera position, needs to convert back to
  // translation
  for (auto& [image_id, image] : images) {
    image.cam_from_world.translation =
        -(image.cam_from_world.rotation * image.cam_from_world.translation);
  }
}

}  // namespace glomap
