#include "glomap/estimators/relpose_estimation.h"

#include <PoseLib/robust.h>

#include <fstream>
#include <sstream>
#include <iostream>

namespace glomap {

extern Eigen::Matrix3d GetRotationFromTxt(const std::string& rotation_file, int image_id);
extern Eigen::Vector3d GetTranslationFromTxt(const std::string& translation_file, int image_id);

void EstimateRelativePoses(ViewGraph& view_graph,
                           std::unordered_map<camera_t, Camera>& cameras,
                           std::unordered_map<image_t, Image>& images,
                           const RelativePoseEstimationOptions& options) {
  std::vector<image_pair_t> valid_pair_ids;
  for (auto& [image_pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    valid_pair_ids.push_back(image_pair_id);
  }

  // Define outside loop to reuse memory and avoid reallocation.
  std::vector<Eigen::Vector2d> points2D_1, points2D_2;
  std::vector<char> inliers;

  const int64_t num_image_pairs = valid_pair_ids.size();
  const int64_t kNumChunks = 10;
  const int64_t interval =
      std::ceil(static_cast<double>(num_image_pairs) / kNumChunks);
  LOG(INFO) << "Estimating relative pose for " << num_image_pairs << " pairs";
  for (int64_t chunk_id = 0; chunk_id < kNumChunks; chunk_id++) {
    std::cout << "\r Estimating relative pose: " << chunk_id * kNumChunks << "%"
              << std::flush;
    const int64_t start = chunk_id * interval;
    const int64_t end =
        std::min<int64_t>((chunk_id + 1) * interval, num_image_pairs);

#pragma omp parallel for schedule(dynamic) private( \
    points2D_1, points2D_2, inliers)
    for (int64_t pair_idx = start; pair_idx < end; pair_idx++) {
      ImagePair& image_pair = view_graph.image_pairs[valid_pair_ids[pair_idx]];
      const Image& image1 = images[image_pair.image_id1];
      const Image& image2 = images[image_pair.image_id2];
      const Eigen::MatrixXi& matches = image_pair.matches;

      // Collect the original 2D points
      points2D_1.clear();
      points2D_2.clear();
      for (size_t idx = 0; idx < matches.rows(); idx++) {
        points2D_1.push_back(image1.features[matches(idx, 0)]);
        points2D_2.push_back(image2.features[matches(idx, 1)]);
      }

      inliers.clear();

      // Step 1: Read the initial absolute rotation and translation for both images
      Eigen::Matrix3d rotation_matrix1 = GetRotationFromTxt("/home/hjl/data/rotation.txt", image_pair.image_id1);
      Eigen::Vector3d translation1 = GetTranslationFromTxt("/home/hjl/data/translation.txt", image_pair.image_id1);
      Eigen::Matrix3d rotation_matrix2 = GetRotationFromTxt("/home/hjl/data/rotation.txt", image_pair.image_id2);
      Eigen::Vector3d translation2 = GetTranslationFromTxt("/home/hjl/data/translation.txt", image_pair.image_id2);

      // Check if the rotations and translations from the file are valid
      bool use_file_data = !(rotation_matrix1.isIdentity() && translation1.isZero()) &&
                          !(rotation_matrix2.isIdentity() && translation2.isZero());

      if (use_file_data) {
          // If both rotations and translations are valid, use them to compute the relative pose
          // Step 2: Compute relative pose based on the absolute poses
          Eigen::Matrix3d relative_rotation = rotation_matrix2.transpose() * rotation_matrix1;
          Eigen::Vector3d relative_translation = translation2 - relative_rotation * translation1;
          
        #pragma omp critical
        {
            std::cout << "Relative Rotation Matrix:\n" << relative_rotation << std::endl;
            std::cout << "Relative Translation Vector:\n" << relative_translation.transpose() << std::endl;
        }

          // Step 3: Convert the relative pose to the glomap format (rotation and translation)
          Eigen::Quaterniond relative_rotation_quat(relative_rotation);

          // Store the relative pose in the image pair (rotation and translation)
          for (int i = 0; i < 4; i++) {
              image_pair.cam2_from_cam1.rotation.coeffs()[i] = relative_rotation_quat.coeffs()[(i + 1) % 4];
          }
          image_pair.cam2_from_cam1.translation = relative_translation;

      } else {
          // If the data is not available or invalid, use the original logic for relative pose estimation
          std::cout << "Using original pose estimation logic for image pair: " << image_pair.image_id1 << " and " << image_pair.image_id2 << std::endl;

          poselib::CameraPose pose_rel_calc;
          try {
              poselib::estimate_relative_pose(
                  points2D_1,
                  points2D_2,
                  ColmapCameraToPoseLibCamera(cameras[image1.camera_id]),
                  ColmapCameraToPoseLibCamera(cameras[image2.camera_id]),
                  options.ransac_options,
                  options.bundle_options,
                  &pose_rel_calc,
                  &inliers);
          } catch (const std::exception& e) {
              LOG(ERROR) << "Error in relative pose estimation: " << e.what();
              image_pair.is_valid = false;
              continue;
          }

          // Convert the relative pose to the glomap format
          for (int i = 0; i < 4; i++) {
              image_pair.cam2_from_cam1.rotation.coeffs()[i] = pose_rel_calc.q[(i + 1) % 4];
          }
          image_pair.cam2_from_cam1.translation = pose_rel_calc.t;
      }

      // poselib::CameraPose pose_rel_calc;
      // std::cout << "Estimating relative pose for image pair: "
      //     << image_pair.image_id1 << " and " << image_pair.image_id2 << std::endl;
      // try {
      //   poselib::estimate_relative_pose(
      //       points2D_1,
      //       points2D_2,
      //       ColmapCameraToPoseLibCamera(cameras[image1.camera_id]),
      //       ColmapCameraToPoseLibCamera(cameras[image2.camera_id]),
      //       options.ransac_options,
      //       options.bundle_options,
      //       &pose_rel_calc,
      //       &inliers);
      // } catch (const std::exception& e) {
      //   LOG(ERROR) << "Error in relative pose estimation: " << e.what();
      //   image_pair.is_valid = false;
      //   continue;
      // }

      // // Convert the relative pose to the glomap format
      // for (int i = 0; i < 4; i++) {
      //   image_pair.cam2_from_cam1.rotation.coeffs()[i] =
      //       pose_rel_calc.q[(i + 1) % 4];
      // }
      // image_pair.cam2_from_cam1.translation = pose_rel_calc.t;
    }
  }

  std::cout << "\r Estimating relative pose: 100%" << std::endl;
  LOG(INFO) << "Estimating relative pose done";
}

}  // namespace glomap
