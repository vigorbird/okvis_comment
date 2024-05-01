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
 *  Created on: Oct 17, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file VioKeyframeWindowMatchingAlgorithm.cpp
 * @brief Source file for the VioKeyframeWindowMatchingAlgorithm class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/cameras/CameraBase.hpp>
#include <okvis/MultiFrame.hpp>

// cameras and distortions
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

#include <opencv2/features2d/features2d.hpp> // for cv::KeyPoint

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor.
template<class CAMERA_GEOMETRY_T>
VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::VioKeyframeWindowMatchingAlgorithm(okvis::Estimator& estimator, int matchingType, float distanceThreshold,bool usePoseUncertainty) 
{
  matchingType_ = matchingType;
  distanceThreshold_ = distanceThreshold;
  estimator_ = &estimator;
  usePoseUncertainty_ = usePoseUncertainty;
}

template<class CAMERA_GEOMETRY_T>
VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::~VioKeyframeWindowMatchingAlgorithm() {

}

// Set which frames to match.
/**
   * @brief Set which frames to match.
   * @param mfIdA   The multiframe ID to match against.
   * @param mfIdB   The new multiframe ID.
   * @param camIdA  ID of the frame inside multiframe A to match.
   * @param camIdB  ID of the frame inside multiframe B to match.
   */
//�������˫Ŀƥ�����������mfIdA=mfIdB��camIdA=0��camIdB=1
//����Ĳ����ֱ��� ˫Ŀ֡A��id ˫Ŀ֡B��id ˫Ŀ֡A����ͼ��or��ͼ�� ˫Ŀ֡B����ͼ��or��ͼ��
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setFrames(uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB) 
{

  OKVIS_ASSERT_TRUE(Exception, !(mfIdA == mfIdB && camIdA == camIdB), "trying to match identical frames.");

  // remember indices
  mfIdA_ = mfIdA;//˫Ŀ֡A��id
  mfIdB_ = mfIdB;//˫Ŀ֡B��id
  camIdA_ = camIdA;//˫Ŀ֡A����������������
  camIdB_ = camIdB;//˫Ŀ֡B����������������
  // frames and related information
  frameA_ = estimator_->multiFrame(mfIdA_);//�õ���Ӧ��ͼ��
  frameB_ = estimator_->multiFrame(mfIdB_);//�õ���Ӧ��ͼ��

  // focal length
  fA_ = frameA_->geometryAs<CAMERA_GEOMETRY_T>(camIdA_)->focalLengthU();
  fB_ = frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->focalLengthU();

  // calculate the relative transformations and uncertainties
  // TODO donno, if and what we need here - I'll see
  estimator_->getCameraSensorStates(mfIdA_, camIdA, T_SaCa_);//����������ϵ��imu����ϵ�ı任���� ���ֵ�ǹ̶�����
  estimator_->getCameraSensorStates(mfIdB_, camIdB, T_SbCb_);//����������ϵ��imu����ϵ�ı任���� ���ֵ�ǹ̶�����
  estimator_->get_T_WS(mfIdA_, T_WSa_);//�õ�A�����imu����������ϵ�µ�λ��
  estimator_->get_T_WS(mfIdB_, T_WSb_);//�õ�B�����imu����������ϵ�µ�λ��
  T_SaW_ = T_WSa_.inverse();
  T_SbW_ = T_WSb_.inverse();
  T_WCa_ = T_WSa_ * T_SaCa_;//�õ�A�������������ϵ�µ�λ��
  T_WCb_ = T_WSb_ * T_SbCb_;//�õ�B�������������ϵ�µ�λ��
  T_CaW_ = T_WCa_.inverse();
  T_CbW_ = T_WCb_.inverse();
  T_CaCb_ = T_WCa_.inverse() * T_WCb_;//�õ��������֮��ı任����
  T_CbCa_ = T_CaCb_.inverse();

  validRelativeUncertainty_ = false;
}

// Set the matching type.
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setMatchingType(
    int matchingType) {
  matchingType_ = matchingType;
}

//����㷨ʵ���ĵ�
// This will be called exactly once for each call to DenseMatcher::match().
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::doSetup() {

  // setup stereo triangulator
  // first, let's get the relative uncertainty.
  okvis::kinematics::Transformation T_CaCb;
  Eigen::Matrix<double, 6, 6> UOplus = Eigen::Matrix<double, 6, 6>::Zero();
  if (usePoseUncertainty_) 
  {
    OKVIS_THROW(Exception, "No pose uncertainty use currently supported");
  } else //Ĭ�Ͻ����������
  {
    UOplus.setIdentity();
    UOplus.bottomRightCorner<3, 3>() *= 1e-8;
    uint64_t currentId = estimator_->currentFrameId();//�ȼ��� return statesMap_.rbegin()->first;
    //���� bool Estimator::isInImuWindow(
    //��ǰ֡���ٶȺ�bias���Ż�������
    //����Ҫƥ�������ͼ����������һ��˫Ŀ֡
    if (estimator_->isInImuWindow(currentId) && (mfIdA_ != mfIdB_))
	{
      okvis::SpeedAndBias speedAndBias;
      estimator_->getSpeedAndBias(currentId, 0, speedAndBias);//��ȡ��ǰ֡��Ӧ���ٶȺ�bias
      double scale = std::max(1.0, speedAndBias.head<3>().norm());
      UOplus.topLeftCorner<3, 3>() *= (scale * scale) * 1.0e-2;
    } else {
      UOplus.topLeftCorner<3, 3>() *= 4e-8;
    }
  }

  // now set the frames and uncertainty
  //���� void ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::resetFrames(
  probabilisticStereoTriangulator_.resetFrames(frameA_, frameB_, camIdA_,camIdB_, T_CaCb_, UOplus);//�Ƚ���Ҫ�ĺ���!!!!!!!!!!!!!!!!����㷨ʵ���ĵ�

  // reset the match counter
  numMatches_ = 0;
  numUncertainMatches_ = 0;

  //1���ȶ�Aͼ����д���
  const size_t numA = frameA_->numKeypoints(camIdA_);//A֡ӵ�е����������
  skipA_.clear();
  skipA_.resize(numA, false);
  raySigmasA_.resize(numA);
  // calculate projections only once
  if (matchingType_ == Match3D2D) 
  {
    // allocate a matrix to store projections
    projectionsIntoB_ = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sizeA(),2);
    projectionsIntoBUncertainties_ = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sizeA() * 2, 2);

    // do the projections for each keypoint, if applicable
    for (size_t k = 0; k < numA; ++k) //����Aͼ���������
	{
      uint64_t lm_id = frameA_->landmarkId(camIdA_, k);//�õ��������Ӧ�ĵ�ͼ��id
      //���� bool isLandmarkAdded(uint64_t landmarkId) const 
      //landmarksMap_û�ж�Ӧ�ĵ�ͼ��
      if (lm_id == 0 || !estimator_->isLandmarkAdded(lm_id)) //��ʾ���������û�ж�Ӧ�ĵ�ͼ��
	  {
        // this can happen, if you called the 2D-2D version just before,
        // without inserting the landmark into the graph
        skipA_[k] = true;
        continue;
      }

      okvis::MapPoint landmark;
      estimator_->getLandmark(lm_id, landmark);//�õ���ͼ��
      Eigen::Vector4d hp_W = landmark.point;

     
      //��ʾ�����ͼ��û�б����뵽ceres�Ĳ�������
      if (!estimator_->isLandmarkInitialized(lm_id)) //���� bool Estimator::isLandmarkInitialized(
	  {
        skipA_[k] = true;
        continue;
      }

      // project (distorted)
      Eigen::Vector2d kptB;
      const Eigen::Vector4d hp_Cb = T_CbW_ * hp_W;//���ռ��ͶӰ��B�������ϵ��
      //���ڵ�ǰ֡�������ϵ�µ������ʹ�û���ģ��ͶӰ��ͼ���� �õ���������kptB��ִ����һ�����ж�ͶӰ�Ƿ�ɹ�
      if (frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->projectHomogeneous(hp_Cb, &kptB)!= okvis::cameras::CameraBase::ProjectionStatus::Successful) 
	  {
        skipA_[k] = true;
        continue;
      }

      if (landmark.observations.size() < 2) 
	  {
        estimator_->setLandmarkInitialized(lm_id, false);//���� void Estimator::setLandmarkInitialized(
        skipA_[k] = true;
        continue;
      }

      // project and get uncertainty
      Eigen::Matrix<double, 2, 4> jacobian;
      Eigen::Matrix4d P_C = Eigen::Matrix4d::Zero();
      P_C.topLeftCorner<3, 3>() = UOplus.topLeftCorner<3, 3>();  // get from before -- velocity scaled
      frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->projectHomogeneous(hp_Cb, &kptB, &jacobian);//���ڵ�ǰ֡�������ϵ�µ������ʹ�û���ģ��ͶӰ��ͼ���� �õ���������kptB��
      //����VioKeyframeWindowMatchingAlgorithm���еı��� projectionsIntoBUncertainties_
      projectionsIntoBUncertainties_.block<2, 2>(2 * k, 0) = jacobian * P_C * jacobian.transpose();
      //���� VioKeyframeWindowMatchingAlgorithm���еı��� projectionsIntoB_
	  projectionsIntoB_.row(k) = kptB;

      // precalculate ray uncertainties
      double keypointAStdDev;//���������ֱ����С
      frameA_->getKeypointSize(camIdA_, k, keypointAStdDev);
      keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
	  //����VioKeyframeWindowMatchingAlgorithm���еı��� raySigmasA_
      raySigmasA_[k] = sqrt(sqrt(2)) * keypointAStdDev / fA_;  // (sqrt(MeasurementCovariance.norm()) / _fA)
    }
  } else {//���ƥ��������2D-2D
    for (size_t k = 0; k < numA; ++k)//�����ؼ�֡��������֡��������
	{
      double keypointAStdDev;
      frameA_->getKeypointSize(camIdA_, k, keypointAStdDev);//�õ������������ֱ��
      keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
	  //����VioKeyframeWindowMatchingAlgorithm���еı��� raySigmasA_
      raySigmasA_[k] = sqrt(sqrt(2)) * keypointAStdDev / fA_;
      if (frameA_->landmarkId(camIdA_, k) == 0)//��ʾ��ǰ֡���������û�е�ͼ������Ӧ
	  {
        continue;
      }
	  //���� bool isLandmarkAdded(uint64_t landmarkId) const 
	  //��landmarksMap_��Ѱ���Ƿ��������ͼ��
      if (estimator_->isLandmarkAdded(frameA_->landmarkId(camIdA_, k))) 
	  {
	    //���� bool Estimator::isLandmarkInitialized(uint64_t landmarkId) const 
	    //��ʼ����ʾ�Ƿ��Ѿ��������ͼ���Ӧ�Ĳ�����ѹ�뵽��ceres��
        if (estimator_->isLandmarkInitialized(frameA_->landmarkId(camIdA_, k)))
		{
		 //��ʾ��������ڵر���ҵر���Ѿ�ѹ�뵽��ceres�������У���ƥ�����ֱ��������������
          skipA_[k] = true;
        }
      }
    }
  }

  //2.�ٶ�Bͼ����д���
  const size_t numB = frameB_->numKeypoints(camIdB_);
  skipB_.clear();
  skipB_.reserve(numB);
  raySigmasB_.resize(numB);
  // do the projections for each keypoint, if applicable
  if (matchingType_ == Match3D2D)
  {
    for (size_t k = 0; k < numB; ++k)//������ǰ֡��������
	{
	      okvis::MapPoint landmark;
		  //���Bͼ���д����������Ӧ�ĵر����landmarksMap_�ṹ���е�ͼ��
	      if (frameB_->landmarkId(camIdB_, k) != 0 && estimator_->isLandmarkAdded(frameB_->landmarkId(camIdB_, k)))//���� bool isLandmarkAdded(uint64_t landmarkId) const 
		  {
	        estimator_->getLandmark(frameB_->landmarkId(camIdB_, k), landmark);
			//�������ĵ�ʽ��� ����ΪBͼ���е���������㲻�ý���ƥ����
			//��������ͼ��۲⵽�����������
	        skipB_.push_back( landmark.observations.find(okvis::KeypointIdentifier(mfIdB_, camIdB_, k))!= landmark.observations.end());
	      } else {
	        skipB_.push_back(false);
	      }
	      double keypointBStdDev;
	      frameB_->getKeypointSize(camIdB_, k, keypointBStdDev);
	      keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
	      raySigmasB_[k] = sqrt(sqrt(2)) * keypointBStdDev / fB_;matchBody
    }
  } else //���ƥ��������2D-2D
  {
    for (size_t k = 0; k < numB; ++k) 
	{
	      double keypointBStdDev;
	      frameB_->getKeypointSize(camIdB_, k, keypointBStdDev);
	      keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
	      raySigmasB_[k] = sqrt(sqrt(2)) * keypointBStdDev / fB_;

	      if (frameB_->landmarkId(camIdB_, k) == 0) 
		  {
	        skipB_.push_back(false);
	        continue;
	      }
		  //�����landmarksMap_�ṹ���е�ͼ�㱻���֡�۲⵽�˲��������ͼ���Ѿ�ѹ�뵽��ceres�Ĳ������У�����Ϊ����������������Ϊ����������
	      if (estimator_->isLandmarkAdded(frameB_->landmarkId(camIdB_, k))) 
		  {
	        skipB_.push_back(estimator_->isLandmarkInitialized(frameB_->landmarkId(camIdB_, k)));  // old: isSet - check.
	      } else {
	        skipB_.push_back(false);
	      }
    }
  }

}

// What is the size of list A?
template<class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::sizeA() const {
  return frameA_->numKeypoints(camIdA_);
}
// What is the size of list B?
template<class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::sizeB() const {
  return frameB_->numKeypoints(camIdB_);
}

// Set the distance threshold for which matches exceeding it will not be returned as matches.
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setDistanceThreshold(
    float distanceThreshold) {
  distanceThreshold_ = distanceThreshold;
}

// Get the distance threshold for which matches exceeding it will not be returned as matches.
template<class CAMERA_GEOMETRY_T>
float VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::distanceThreshold() const {
  return distanceThreshold_;
}

// Geometric verification of a match.
///�ü���ģ�Ͷ�ƥ���Խ���ɸѡ
template<class CAMERA_GEOMETRY_T>
bool VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::verifyMatch(size_t indexA, size_t indexB) const 
{

  if (matchingType_ == Match2D2D) 
  {

    // potential 2d2d match - verify by triangulation
    Eigen::Vector4d hP;
    bool isParallel;
	//���� bool ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::stereoTriangulate(
	//��Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!
	//hP, isParallel���������
    bool valid = probabilisticStereoTriangulator_.stereoTriangulate( indexA, indexB, hP, isParallel, std::max(raySigmasA_[indexA], raySigmasB_[indexB]));
    if (valid) {
      return true;
    }
  } else //3D-2D�����
 {
    // get projection into B
    Eigen::Vector2d kptB = projectionsIntoB_.row(indexA);

    // uncertainty
    double keypointBStdDev;
    frameB_->getKeypointSize(camIdB_, indexB, keypointBStdDev);
    keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
	////U��ʾB֡���������Э����+�����ٶ������ͶӰ��B�������һ����Ϣ����
    Eigen::Matrix2d U = Eigen::Matrix2d::Identity() * keypointBStdDev* keypointBStdDev+ projectionsIntoBUncertainties_.block<2, 2>(2 * indexA, 0);

    Eigen::Vector2d keypointBMeasurement;
    frameB_->getKeypoint(camIdB_, indexB, keypointBMeasurement);
    Eigen::Vector2d err = kptB - keypointBMeasurement;//����A֡�������㵽B֡ͶӰ��B֡��ԭʼ���ƫ��
    const int chi2 = err.transpose() * U.inverse() * err;

    if (chi2 < 4.0) 
	{
      return true;
    }
  }
  return false;
}

// A function that tells you how many times setMatching() will be called.
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::reserveMatches(
    size_t /*numMatches*/) {
  //_triangulatedPoints.clear();
}

// Get the number of matches.
template<class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::numMatches() {
  return numMatches_;
}

// Get the number of uncertain matches.
template<class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::numUncertainMatches() {
  return numUncertainMatches_;
}

// At the end of the matching step, this function is called once
// for each pair of matches discovered.
//һ��Ҫע���� �������뷶Χ��ֻ�������������addObservation������
//�� addObservation�ǽ���ͶӰ�������뵽ceres��
//�� setBestMatch����ֻ��matchBody�б����ù�
//��Ҫ����������ceres��Ӳв������ceres����ӵ�ͼ��Ĳ����飬����ceres�в�����Ķ���
//����㷨ʵ���ĵ�
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setBestMatch(size_t indexA, size_t indexB, double /*distance*/) 
{

  // assign correspondences
  uint64_t lmIdA = frameA_->landmarkId(camIdA_, indexA);//Aͼ���������Ӧ�ĵ�ͼ�����
  uint64_t lmIdB = frameB_->landmarkId(camIdB_, indexB);//Bͼ���������Ӧ�ĵ�ͼ�����

  if (matchingType_ == Match2D2D)
  {

    // check that not both are set
    // ��ʾ���������㶼�Ѿ��˳�ʼ��,����Ҫ�ٽ���ƥ��
    if (lmIdA != 0 && lmIdB != 0) {
      return;
    }

    // re-triangulate...
    // potential 2d2d match - verify by triangulation
    Eigen::Vector4d hP_Ca;
    bool canBeInitialized;

    //���ֵ canBeInitialized=false ��ʾ��������ƽ��
    //���ֵ�� hP_Ca = ���A����ϵ�µ���ά�� �ǵ�λ�����������
    //���� bool ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::stereoTriangulate(
    //��Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    bool valid = probabilisticStereoTriangulator_.stereoTriangulate(  indexA, indexB, hP_Ca, canBeInitialized, std::max(raySigmasA_[indexA], raySigmasB_[indexB]));
    if (!valid) 
	{
      return;
    }

    // get the uncertainty
    //���� void ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::getUncertainty(
    //���ֵ�� pointUOplus_A, canBeInitialized
    //��Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (canBeInitialized) //�������������ʾ�������߲�ƽ��
	{  // know more exactly
      Eigen::Matrix3d pointUOplus_A;//һ��ע�����Ǹ��ֲ����� ����û���õ�
	  //���� void ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::getUncertainty(
	  //���ֵpointUOplus_A = 
	  //���������о����ﱻ������
      probabilisticStereoTriangulator_.getUncertainty(indexA, indexB, hP_Ca, pointUOplus_A, canBeInitialized);
    }

    // check and adapt landmark status
    bool insertA = lmIdA == 0;
    bool insertB = lmIdB == 0;
    bool insertHomogeneousPointParameterBlock = false;
    uint64_t lmId = 0;  // 0 just to avoid warning
    //�������������㶼û�е�ͼ������Ӧ
    if (insertA && insertB) 
	{
      // ok, we need to assign a new Id...
      lmId = okvis::IdProvider::instance().newId();//�½�һ����ͼ��id
      //������������е���Ϣ����ʾ������֡�������������ͼ��
      frameA_->setLandmarkId(camIdA_, indexA, lmId);
      frameB_->setLandmarkId(camIdB_, indexB, lmId);
      lmIdA = lmId;
      lmIdB = lmId;
      // and add it to the graph
      insertHomogeneousPointParameterBlock = true;
    } else {//�����㶼��һ���Ѿ������˳�ʼ��
       //��ʾA֡�������Ѿ��ж�Ӧ�ĵر��
      if (!insertA) 
	  {
        lmId = lmIdA;
		//����  bool isLandmarkAdded(uint64_t landmarkId) const 
        if (!estimator_->isLandmarkAdded(lmId)) //landmarksMap_״̬�ռ������δ���A֡�������Ӧ�ĵ�ͼ��
		{
          // add landmark and observation to the graph
          insertHomogeneousPointParameterBlock = true;
          insertA = true;
        }
      }
      if (!insertB) //��ʾB֡�������Ѿ��ж�Ӧ�ĵر��
	  {
        lmId = lmIdB;
        if (!estimator_->isLandmarkAdded(lmId))//landmarksMap_״̬�ռ������δ���B֡�������Ӧ�ĵ�ͼ��
		{
          // add landmark and observation to the graph
          insertHomogeneousPointParameterBlock = true;
          insertB = true;
        }
      }
    }
	
    // add landmark to graph if necessary
    //���������������������������µĵ�ͼ��Ĳ�����
    if (insertHomogeneousPointParameterBlock) 
	{
	  //addLandmark��������: 1.��ceres����������ͼ�������.2.����landmarksMap_�ṹ�е���Ϣ���¼���һ����ͼ��
	  //���� bool Estimator::addLandmark(
      estimator_->addLandmark(lmId, T_WCa_ * hP_Ca);
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmId), lmId<<" not added, bug");
	  //���� void Estimator::setLandmarkInitialized(
	  //������������������� �Ѿ�����ʼ������
      estimator_->setLandmarkInitialized(lmId, canBeInitialized);
    } else 
    { //���A��Bͼ���е��������е�ͼ���Ӧ���������ͼ�㶼�Ѿ�������landmarksMap_״̬�ռ�����
          //���ǲ���ceres����������ˣ������޸�����ֵ��
      // update initialization status, set better estimate, if possible
      if (canBeInitialized) 
	  {
	    //���� void Estimator::setLandmarkInitialized(uint64_t landmarkId,bool initialized) 
        estimator_->setLandmarkInitialized(lmId, true);
		//���� bool Estimator::setLandmark(uint64_t landmarkId, const Eigen::Vector4d & landmark)
		//1.��mapPtr_����ȡ����ͼ���Ӧ�Ĳ�����ָ�룬���޸���ֵ
		//2.�޸�landmarksMap_�е�ͼ���ֵ
        estimator_->setLandmark(lmId, T_WCa_ * hP_Ca);
      }
    }

    // in image A
    okvis::MapPoint landmark;
	//A֡�������㻹û�ж�Ӧ�ĵر�㣬�ҵ�ͼ����A֡��û�ж�Ӧ������Ĺ۲�
    if (insertA  && landmark.observations.find(okvis::KeypointIdentifier(mfIdA_, camIdA_, indexA))== landmark.observations.end()) 
	{  // ensure no double observations...
            // TODO hp_Sa NOT USED!
      Eigen::Vector4d hp_Sa(T_SaCa_ * hP_Ca);//ת����A�����IMU����ϵ��
      hp_Sa.normalize();//û���õ��������
      frameA_->setLandmarkId(camIdA_, indexA, lmId);//�������A�����������ͼ��
      lmIdA = lmId;//������º���Ҳû���õ�lmIdA�������ѽ
      // initialize in graph
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmId),
                        "landmark id=" << lmId<<" not added");
	  //lmId�������ͼ���Ӧ��id mfIdA_��Aͼ���˫Ŀ֡�ṹ��id��camIdA_��Aͼ���Ӧ���Ǹ�˫Ŀ֡����ͼ������ͼ��
	  //indexA = ��������Aͼ���е����
	  //addObservation��������: ��ceres�����BA�Ĳв�̣�����landmarksMap_����Ϣ,�õ�ͼ�㿴�����������
	  //���� ::ceres::ResidualBlockId Estimator::addObservation(
      estimator_->addObservation<camera_geometry_t>(lmId, mfIdA_, camIdA_,indexA);
    }

    // in image B
    ///B֡�и������㻹û�ж�Ӧ�ر�㣬�Ҹõر����B֡��û�ж�Ӧ������Ĺ۲�
    if (insertB&& landmark.observations.find(okvis::KeypointIdentifier(mfIdB_, camIdB_, indexB))== landmark.observations.end()) 
	{  // ensure no double observations...
      Eigen::Vector4d hp_Sb(T_SbCb_ * T_CbCa_ * hP_Ca);
      hp_Sb.normalize();//û���õ��������
      frameB_->setLandmarkId(camIdB_, indexB, lmId);
      lmIdB = lmId;//������º���Ҳû���õ�lmIdA�������ѽ
      // initialize in graph
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmId),"landmark " << lmId << " not added");
	  //addObservation��������: ��ceres�����BA�Ĳв�̣�����landmarksMap_����Ϣ���õ�ͼ�㿴�����������
      estimator_->addObservation<camera_geometry_t>(lmId, mfIdB_, camIdB_,indexB);
    }

    // let's check for consistency with other observations:
    okvis::ceres::HomogeneousPointParameterBlock point(T_WCa_ * hP_Ca, 0);
    if(canBeInitialized)
    {
      //��mapPtr_����ȡ����ͼ���Ӧ�Ĳ�����ָ�룬���޸���ֵ
      //�޸�landmarksMap_�е�ͼ���ֵ
      //���� bool Estimator::setLandmark(uint64_t landmarkId, const Eigen::Vector4d & landmark)
      estimator_->setLandmark(lmId, point.estimate());
    }

  } else //�����3d-2dƥ��
 {
    OKVIS_ASSERT_TRUE_DBG(Exception,lmIdB==0,"bug. Id in frame B already set.");

    // get projection into B
    Eigen::Vector2d kptB = projectionsIntoB_.row(indexA);//��A��������ƥ���Bͼ���е�����
    Eigen::Vector2d keypointBMeasurement;
    frameB_->getKeypoint(camIdB_, indexB, keypointBMeasurement);//B֡��������Ĺ۲�ֵ����ʵֵ��

    Eigen::Vector2d err = kptB - keypointBMeasurement;
    double keypointBStdDev;
    frameB_->getKeypointSize(camIdB_, indexB, keypointBStdDev);//�õ�B֡��������ĳߴ�
    keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
    Eigen::Matrix2d U_tot = Eigen::Matrix2d::Identity() * keypointBStdDev * keypointBStdDev + projectionsIntoBUncertainties_.block<2, 2>(2 * indexA, 0);

    const double chi2 = err.transpose().eval() * U_tot.inverse() * err;

    if (chi2 > 4.0) {
      return;
    }

    // saturate allowed image uncertainty
     // Э��������ģֵ��������Ϊ��ȷ��
    if (U_tot.norm() > 25.0 / (keypointBStdDev * keypointBStdDev * sqrt(2))) 
	{
      numUncertainMatches_++;
      //return;
    }
    //���� bool Frame::setLandmarkId(size_t keypointIdx, uint64_t landmarkId)
    frameB_->setLandmarkId(camIdB_, indexB, lmIdA);//B֡�����������ͼ��
    lmIdB = lmIdA;
    okvis::MapPoint landmark;
    estimator_->getLandmark(lmIdA, landmark);//�õ��ر�����Ϣlandmark

    // initialize in graph
    //�������������ʾ֮ǰ�����ͼ��û�б�B֡�۲⵽
    if (landmark.observations.find(okvis::KeypointIdentifier(mfIdB_, camIdB_, indexB))== landmark.observations.end()) 
	{  // ensure no double observations...
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmIdB), "not added");
	  //���� ::ceres::ResidualBlockId Estimator::addObservation(
	  //����Ĳ����ֱ��� ��ͼ��id ˫Ŀ֡id ��Ŀ/��Ŀid �����������ͼ���ϵ�id
	  //addObservation��������: ��ceres�����BA�Ĳв�̣�����landmarksMap_����Ϣ���õ�ͼ�㿴�����������
      estimator_->addObservation<camera_geometry_t>(lmIdB, mfIdB_, camIdB_,indexB);//�Ըõر�����B֡��indexB��������Ĺ۲�
    }

  }
  numMatches_++;
}

template class VioKeyframeWindowMatchingAlgorithm<
    okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > ;

template class VioKeyframeWindowMatchingAlgorithm<
    okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> > ;

template class VioKeyframeWindowMatchingAlgorithm<
    okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion8> > ;

}
