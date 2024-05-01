/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Jan 10, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file implementation/Estimator.hpp
 * @brief Header implementation file for the Estimator class.
 * @author Stefan Leutenegger
 */

/// \brief okvis Main namespace of this package.
namespace okvis {

// Add an observation to a landmark.
/**
 * @brief Add an observation to a landmark.
 * \tparam GEOMETRY_TYPE The camera geometry type for this observation.
 * @param landmarkId ID of landmark.
 * @param poseId ID of pose where the landmark was observed.
 * @param camIdx ID of camera frame where the landmark was observed.
 * @param keypointIdx ID of keypoint corresponding to the landmark.
 * @return Residual block ID for that observation.
 */
 //输入的模板类型GEOMETRY_TYPE = camera_geometry_t
 //输入的参数分别是 地图点id 双目帧id 左目/右目id 特征点在这幅图像上的id
 //向ceres中添加BA的残差方程，更新landmarksMap_的信息,让地图点看到这个特征点
template<class GEOMETRY_TYPE>
::ceres::ResidualBlockId Estimator::addObservation(uint64_t landmarkId,
                                                   uint64_t poseId,
                                                   size_t camIdx,
                                                   size_t keypointIdx) 
{
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),
                        "landmark not added");

  // avoid double observations
  okvis::KeypointIdentifier kid(poseId, camIdx, keypointIdx);
  //首先判断这个地图点是否和这个特征点已经对应了?如果对应了则进入条件直接返回
  if (landmarksMap_.at(landmarkId).observations.find(kid)!= landmarksMap_.at(landmarkId).observations.end())
  {
    return NULL;
  }

  // get the keypoint measurement
  okvis::MultiFramePtr multiFramePtr = multiFramePtrMap_.at(poseId);
  Eigen::Vector2d measurement;
  multiFramePtr->getKeypoint(camIdx, keypointIdx, measurement);//获得这个特征点在序号为camIdx的帧下的测量值 measurement
  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  double size = 1.0;
  multiFramePtr->getKeypointSize(camIdx, keypointIdx, size);//得到这个特征点的邻域大小 size
  information *= 64.0 / (size * size);

  // create error term
  //表示重头影误差函数 详见算法实现文档
  std::shared_ptr <ceres::ReprojectionError< GEOMETRY_TYPE>> reprojectionError( new ceres::ReprojectionError<GEOMETRY_TYPE>(multiFramePtr->template geometryAs<GEOMETRY_TYPE>(camIdx),
                 																											 camIdx, measurement, information));
  //将重投影误差加入到ceres优化中
  //搜索 ::ceres::ResidualBlockId Map::addResidualBlock(
  ::ceres::ResidualBlockId retVal = mapPtr_->addResidualBlock(
      reprojectionError,
      cauchyLossFunctionPtr_ ? cauchyLossFunctionPtr_.get() : NULL,//这里默认是使用的cauchy函数作为loss function 设置的参数为1:cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      mapPtr_->parameterBlockPtr(poseId),
      mapPtr_->parameterBlockPtr(landmarkId),
      mapPtr_->parameterBlockPtr(statesMap_.at(poseId).sensors.at(SensorStates::Camera).at(camIdx).at(CameraSensorStates::T_SCi).id));

  // remember
  // 定义 okvis::PointMap landmarksMap_; 存储所有的地图点，元素=(地图点id,地图点数据类型)
  //向这个地图点插入信息，看到了这个特征点
  landmarksMap_.at(landmarkId).observations.insert( std::pair<okvis::KeypointIdentifier, uint64_t>(kid, reinterpret_cast<uint64_t>(retVal)));//reinterpret_cast是强制类型转换符

  return retVal;
}

}  // namespace okvis
