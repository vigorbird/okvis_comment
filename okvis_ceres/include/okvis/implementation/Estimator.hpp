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
 //�����ģ������GEOMETRY_TYPE = camera_geometry_t
 //����Ĳ����ֱ��� ��ͼ��id ˫Ŀ֡id ��Ŀ/��Ŀid �����������ͼ���ϵ�id
 //��ceres�����BA�Ĳв�̣�����landmarksMap_����Ϣ,�õ�ͼ�㿴�����������
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
  //�����ж������ͼ���Ƿ������������Ѿ���Ӧ��?�����Ӧ�����������ֱ�ӷ���
  if (landmarksMap_.at(landmarkId).observations.find(kid)!= landmarksMap_.at(landmarkId).observations.end())
  {
    return NULL;
  }

  // get the keypoint measurement
  okvis::MultiFramePtr multiFramePtr = multiFramePtrMap_.at(poseId);
  Eigen::Vector2d measurement;
  multiFramePtr->getKeypoint(camIdx, keypointIdx, measurement);//�����������������ΪcamIdx��֡�µĲ���ֵ measurement
  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  double size = 1.0;
  multiFramePtr->getKeypointSize(camIdx, keypointIdx, size);//�õ����������������С size
  information *= 64.0 / (size * size);

  // create error term
  //��ʾ��ͷӰ���� ����㷨ʵ���ĵ�
  std::shared_ptr <ceres::ReprojectionError< GEOMETRY_TYPE>> reprojectionError( new ceres::ReprojectionError<GEOMETRY_TYPE>(multiFramePtr->template geometryAs<GEOMETRY_TYPE>(camIdx),
                 																											 camIdx, measurement, information));
  //����ͶӰ�����뵽ceres�Ż���
  //���� ::ceres::ResidualBlockId Map::addResidualBlock(
  ::ceres::ResidualBlockId retVal = mapPtr_->addResidualBlock(
      reprojectionError,
      cauchyLossFunctionPtr_ ? cauchyLossFunctionPtr_.get() : NULL,//����Ĭ����ʹ�õ�cauchy������Ϊloss function ���õĲ���Ϊ1:cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      mapPtr_->parameterBlockPtr(poseId),
      mapPtr_->parameterBlockPtr(landmarkId),
      mapPtr_->parameterBlockPtr(statesMap_.at(poseId).sensors.at(SensorStates::Camera).at(camIdx).at(CameraSensorStates::T_SCi).id));

  // remember
  // ���� okvis::PointMap landmarksMap_; �洢���еĵ�ͼ�㣬Ԫ��=(��ͼ��id,��ͼ����������)
  //�������ͼ�������Ϣ�����������������
  landmarksMap_.at(landmarkId).observations.insert( std::pair<okvis::KeypointIdentifier, uint64_t>(kid, reinterpret_cast<uint64_t>(retVal)));//reinterpret_cast��ǿ������ת����

  return retVal;
}

}  // namespace okvis
