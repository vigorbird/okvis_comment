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
 *  Created on: Mar 27, 2015
 *      Author: Andreas Forster (an.forster@gmail.com)
 *    Modified: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file Frontend.cpp
 * @brief Source file for the Frontend class.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 */

#include <okvis/Frontend.hpp>

#include <brisk/brisk.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <glog/logging.h>

#include <okvis/ceres/ImuError.hpp>
#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
#include <okvis/IdProvider.hpp>

// cameras and distortions
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

// Kneip RANSAC
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor.
Frontend::Frontend(size_t numCameras)
    : isInitialized_(false),
      numCameras_(numCameras),
      briskDetectionOctaves_(0),
      briskDetectionThreshold_(50.0),
      briskDetectionAbsoluteThreshold_(800.0),
      briskDetectionMaximumKeypoints_(450),
      briskDescriptionRotationInvariance_(true),
      briskDescriptionScaleInvariance_(false),
      briskMatchingThreshold_(60.0),
      matcher_(
          std::unique_ptr<okvis::DenseMatcher>(new okvis::DenseMatcher(4))),
      keyframeInsertionOverlapThreshold_(0.6),
      keyframeInsertionMatchingRatioThreshold_(0.2) {
  // create mutexes for feature detectors and descriptor extractors
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectorMutexes_.push_back(
        std::unique_ptr<std::mutex>(new std::mutex()));
  }
  initialiseBriskFeatureDetectors();
}

// Detection and descriptor extraction on a per image basis.
//T_WC=����imu�õ��������λ�ˣ����������
bool Frontend::detectAndDescribe(size_t cameraIndex,
                                 std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint> * keypoints) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < numCameras_, "Camera index exceeds number of cameras.");
  std::lock_guard<std::mutex> lock(*featureDetectorMutexes_[cameraIndex]);

  // check there are no keypoints here
  OKVIS_ASSERT_TRUE(Exception, keypoints == nullptr, "external keypoints currently not supported")

  frameOut->setDetector(cameraIndex, featureDetectors_[cameraIndex]);//��������ʹ�õ���brisk��������ȡ����
  frameOut->setExtractor(cameraIndex, descriptorExtractors_[cameraIndex]);//��������ʹ�õ���brisk�ṩ��������

  frameOut->detect(cameraIndex);//��ȡ�����㣬���� int Frame::detect()

  // ExtractionDirection == gravity direction in camera frame
  Eigen::Vector3d g_in_W(0, 0, -1);
  Eigen::Vector3d extractionDirection = T_WC.inverse().C() * g_in_W;
  frameOut->describe(cameraIndex, extractionDirection);//��ȡ�����ӣ����� int Frame::describe(const Eigen::Vector3d & extractionDirection)

  // set detector/extractor to nullpointer? TODO
  return true;
}

// Matching as well as initialization of landmarks and state.
//Transformation�����Ż��õ��ĵ�ǰ֡��λ��
//framesInOut�Ѿ���ȡ��������������ӵ�˫Ŀͼ�� ��������Ҳ�����
//asKeyframe �����������һ֡�Ƿ�Ϊ�ؼ�֡
bool Frontend::dataAssociationAndInitialization(
    okvis::Estimator& estimator,
    okvis::kinematics::Transformation& /*T_WS_propagated*/, // TODO sleutenegger: why is this not used here?
    const okvis::VioParameters &params,
    const std::shared_ptr<okvis::MapPointVector> /*map*/, // TODO sleutenegger: why is this not used here?
    std::shared_ptr<okvis::MultiFrame> framesInOut,
    bool *asKeyframe) 
 {
  // match new keypoints to existing landmarks/keypoints
  // initialise new landmarks (states)
  // outlier rejection by consistency check
  // RANSAC (2D2D / 3D2D)
  // decide keyframe
  // left-right stereo match & init

  // find distortion type
  okvis::cameras::NCameraSystem::DistortionType distortionType = params.nCameraSystem.distortionType(0);//euroc������ radialtangential����
  for (size_t i = 1; i < params.nCameraSystem.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception,
                      distortionType == params.nCameraSystem.distortionType(i),
                      "mixed frame types are not supported yet");
  }
  int num3dMatches = 0;

  // 1.first frame? (did do addStates before, so 1 frame minimum in estimator)
  //1.�ж��Ƿ�Ϊ��һ֡ͼ��
  if (estimator.numFrames() > 1) //���ǳ�ʼ��״̬
  {

    int requiredMatches = 5;//��Ҫ����Сƥ�����

    double uncertainMatchFraction = 0;
    bool rotationOnly = false;

    // match to last keyframe
    TimerSwitchable matchKeyframesTimer("2.4.1 matchToKeyframes");//ʲô��û��
    switch (distortionType) 
	{
	  //1.1��ǰ֡�͹ؼ�֡����ƥ��!!!!!!!!!!!!!!!
      case okvis::cameras::NCameraSystem::RadialTangential: { num3dMatches = matchToKeyframes<VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > >(estimator, params, framesInOut->id(), rotationOnly, false,&uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::EquidistantDistortion> > >(
            estimator, params, framesInOut->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, framesInOut->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    //matchKeyframesTimer.stop();//ʲô��û�ɣ�ע�ͷ��㿴����
    if (!isInitialized_) 
	{
      if (!rotationOnly) 
	  {
        isInitialized_ = true;//���������о���isInitialized_����ֵ��
        LOG(INFO) << "Initialized!";
      }
    }

    if (num3dMatches <= requiredMatches) 
	{
      LOG(WARNING) << "Tracking failure. Number of 3d2d-matches: " << num3dMatches;
    }

    // keyframe decision, at the moment only landmarks that match with keyframe are initialised
    //1.2�жϵ�ǰ֡�Ƿ�����Ϊ�ؼ�֡
    *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, framesInOut);//�Ƚ���Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!

    // match to last frame
    TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrame");//ʲô��û��
    switch (distortionType) {
	  //1.3����һ֡����ƥ��
      case okvis::cameras::NCameraSystem::RadialTangential: {matchToLastFrame<VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > >(estimator, params, framesInOut->id(),false);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::EquidistantDistortion> > >(
            estimator, params, framesInOut->id(),
            false);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, framesInOut->id(),
            false);

        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    //matchToLastFrameTimer.stop();//ʲô��û�ɣ�����ע�͵����㿴����
  } else//��ʼ��״̬������ƥ�� ֱ�ӽ���һ֡����Ϊ�ؼ�֡
    *asKeyframe = true;  // first frame needs to be keyframe


  // 2.do stereo match to get new landmarks
  TimerSwitchable matchStereoTimer("2.4.3 matchStereo");//ʲô��û��
  switch (distortionType) 
  {
  	//����˫Ŀƥ��
    case okvis::cameras::NCameraSystem::RadialTangential: 
	{
		//���������о����������matchStereo����
		matchStereo<VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > >(estimator,framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::Equidistant: {
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::EquidistantDistortion> > >(estimator,
                                                             framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::RadialTangential8: {
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion8> > >(estimator,
                                                                   framesInOut);
      break;
    }
    default:
      OKVIS_THROW(Exception, "Unsupported distortion type.")
      break;
  }
  //matchStereoTimer.stop();//ʲô��û�� ע�͵����㿴����

  return true;
}

// Propagates pose, speeds and biases with given IMU measurements.
bool Frontend::propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                           const okvis::ImuParameters & imuParams,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBias & speedAndBiases,
                           const okvis::Time& t_start, const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const {
  if (imuMeasurements.size() < 2) {
    LOG(WARNING)
        << "- Skipping propagation as only one IMU measurement has been given to frontend."
        << " Normal when starting up.";
    return 0;
  }
  int measurements_propagated = okvis::ceres::ImuError::propagation(
      imuMeasurements, imuParams, T_WS_propagated, speedAndBiases, t_start,
      t_end, covariance, jacobian);

  return measurements_propagated > 0;
}

// Decision whether a new frame should be keyframe or not.
bool Frontend::doWeNeedANewKeyframe(const okvis::Estimator& estimator,std::shared_ptr<okvis::MultiFrame> currentFrame) 
{

  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  if (!isInitialized_)
    return false;

  double overlap = 0.0;
  double ratio = 0.0;

  // go through all the frames and try to match the initialized keypoints
  for (size_t im = 0; im < currentFrame->numFrames(); ++im) //�����������
  {

    // get the hull of all keypoints in current frame
    std::vector<cv::Point2f> frameBPoints, frameBHull;
    std::vector<cv::Point2f> frameBMatches, frameBMatchesHull;
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > frameBLandmarks;

    const size_t numB = currentFrame->numKeypoints(im);//��ȡ����������������������������
    frameBPoints.reserve(numB);
    frameBLandmarks.reserve(numB);
    Eigen::Vector2d keypoint;
    for (size_t k = 0; k < numB; ++k) //������������������������������
	{
      currentFrame->getKeypoint(im, k, keypoint);
      // insert it
      frameBPoints.push_back(cv::Point2f(keypoint[0], keypoint[1]));//�洢��������������
      // also remember matches
      if (currentFrame->landmarkId(im, k) != 0) 
	  {
        frameBMatches.push_back(cv::Point2f(keypoint[0], keypoint[1]));//�洢�����Ѿ��͵�ͼƥ���������
      }
    }

    if (frameBPoints.size() < 3)
      continue;
    cv::convexHull(frameBPoints, frameBHull);//Ѱ�����������㹹���͹���㣬frameBHull
    if (frameBMatches.size() < 3)
      continue;
    cv::convexHull(frameBMatches, frameBMatchesHull);//Ѱ���Ѿ��͵�ͼ��ƥ����������������͹���㣬frameBMatchesHull

    // areas
    double frameBArea = cv::contourArea(frameBHull);//����������͹�������
    double frameBMatchesArea = cv::contourArea(frameBMatchesHull);//�Ѿ�ƥ���͹�������

    // overlap area
    double overlapArea = frameBMatchesArea / frameBArea;//�Ѿ�ƥ���͹�������/�����������͹�����
    // matching ratio inside overlap area: count
    int pointsInFrameBMatchesArea = 0;
    if (frameBMatchesHull.size() > 2) 
	{
      for (size_t k = 0; k < frameBPoints.size(); ++k) //��������������
	  {
	    //��������������Ƿ���ƥ���͹���������
        if (cv::pointPolygonTest(frameBMatchesHull, frameBPoints[k], false)> 0) //�����Ƿ��������ڣ�false��ʾ���صĽ�������֣�+1��ʾ��������
		{
          pointsInFrameBMatchesArea++;
        }
      }
    }
    double matchingRatio = double(frameBMatches.size())/ double(pointsInFrameBMatchesArea);//�Ѿ�ƥ��͵�ͼ��ƥ������������/��͹���ڵ����������

    // calculate overlap score
    overlap = std::max(overlapArea, overlap);//����������д洢��������ͼ����ߵı���
    ratio = std::max(matchingRatio, ratio);
  }

  // take a decision
  ////keyframeInsertionOverlapThreshold_=0.6
  //keyframeInsertionMatchingRatioThreshold_= 0.2
  //������������������������Żᴴ���ؼ�֡
  if (overlap > keyframeInsertionOverlapThreshold_ && ratio > keyframeInsertionMatchingRatioThreshold_)
    return false;
  else
    return true;
}

// Match a new multiframe to existing keyframes
/**
  * @brief Match a new multiframe to existing keyframes
  * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks
  * @warning As this function uses the estimator it is not threadsafe
  * @param		estimator			   Estimator.
  * @param[in]	params				   Parameter struct.
  * @param[in]	currentFrameId		   ID of the current frame that should be matched against keyframes.
  * @param[out] rotationOnly 		   Was the rotation only RANSAC motion model good enough to
  * 								   explain the motion between the new frame and the keyframes?
  * @param[in]	usePoseUncertainty 	   Use the pose uncertainty for the matching.
  * @param[out] uncertainMatchFraction  Return the fraction of uncertain matches. Set to nullptr if not interested.
  * @param[in]	removeOutliers		   Remove outliers during RANSAC.
  * @return The number of matches in total.
    template<class MATCHING_ALGORITHM>
  int matchToKeyframes(okvis::Estimator& estimator,
                       const okvis::VioParameters& params,
                       const uint64_t currentFrameId, bool& rotationOnly,
                       bool usePoseUncertainty = true,
                       double* uncertainMatchFraction = 0,
                       bool removeOutliers = true);  // for wide-baseline matches (good initial guess)
  */
//�����ģ��MATCHING_ALGORITHM = VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > 
//����Ĳ���Ϊ rotationOnly=false usePoseUncertainty=false uncertainMatchFraction=0
template<class MATCHING_ALGORITHM>
int Frontend::matchToKeyframes(okvis::Estimator& estimator,
                               const okvis::VioParameters & params,
                               const uint64_t currentFrameId,
                               bool& rotationOnly,
                               bool usePoseUncertainty,
                               double* uncertainMatchFraction,
                               bool removeOutliers) {
  rotationOnly = true;
  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return 0;
  }

  int retCtr = 0;
  int numUncertainMatches = 0;

  // go through all the frames and try to match the initialized keypoints��
  //��˫Ŀ�����ͼ�������������ؼ�֡����3D-2Dƥ��
  size_t kfcounter = 0;
  for (size_t age = 1; age < estimator.numFrames(); ++age) 
  {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId))
      continue;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im)//��������ͼ��
	{
	  //ƥ���㷨��ʼ��
	  VioKeyframeWindowMatchingAlgorithm
	  //���� VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::VioKeyframeWindowMatchingAlgorithm
      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match3D2D,
                                           briskMatchingThreshold_,//Ĭ��60
                                           usePoseUncertainty);//����Ϊfalse
      //���� void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setFrames
      //olderFrameId����ǰ�ؼ�֡��˫Ŀid currentFrameId�ǵ�ǰ˫Ŀ֡��id im����������������
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 3D-2D
      //��ʼ����ƥ�䣬match�����������������������
      //���� std::unique_ptr<okvis::DenseMatcher> matcher_; 
      //���� void DenseMatcher::match(MATCHING_ALGORITHM_T & matchingAlgorithm) 
     	 //���� void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::doSetup() {
     	 //���� void DenseMatcher::matchBody( void������ doWorkLinearMatching
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      retCtr += matchingAlgorithm.numMatches();
      numUncertainMatches += matchingAlgorithm.numUncertainMatches();

    }
    kfcounter++;
    if (kfcounter > 2)
      break;
  }

  //������������ؼ�֡����2D-2Dƥ��
  //����Ѿ���ʼ���������ô����ransace3d-2dƥ��
  //���ϵͳ��ʼ��û�������ô����ransac2d-2dƥ�䣬�������ƥ������Ƿ��Ѿ���ʼ�����
  kfcounter = 0;
  bool firstFrame = true;
  for (size_t age = 1; age < estimator.numFrames(); ++age) 
  {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId))
      continue;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) 
	{
	  //ƥ���㷨��ʼ��
      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match2D2D,
                                           briskMatchingThreshold_,//Ĭ��60
                                           usePoseUncertainty);//����Ϊfalse
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 2D-2D for initialization of new (mono-)correspondences
      //���濪ʼƥ��
      //���� DenseMatcher::match
     	 //���� void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::doSetup() {
     	 //���� void DenseMatcher::matchBody( void������ doWorkLinearMatching
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      retCtr += matchingAlgorithm.numMatches();
      numUncertainMatches += matchingAlgorithm.numUncertainMatches();
    }

    // remove outliers
    // only do RANSAC 3D2D with most recent KF
    if (kfcounter == 0 && isInitialized_)//����һ��Ҫע���� ֻ�г�ʼ����ɺ�Ž���ransac3d2d 
   	{
   	   //��ǰ�������Ѿ�ʹ��doSetup����ǰ֡��������͵�ͼ�������ƥ�䣬������������ʹ��PnP��ransac�㷨ȥ����Щoutlier,������Щoutlier��map�е�ceres�Ż�ɾ��
       runRansac3d2d(estimator, params.nCameraSystem,estimator.multiFrame(currentFrameId), removeOutliers);//���� int Frontend::runRansac3d2d(
    }

    bool rotationOnly_tmp = false;
    // do RANSAC 2D2D for initialization only
    if (!isInitialized_) //����һ��Ҫע��ֻ����ϵͳû�г�ʼ��ʱ�Ž���ransac2d2d
	{
      runRansac2d2d(estimator, params, currentFrameId, olderFrameId, true, removeOutliers, rotationOnly_tmp);//���� int Frontend::runRansac2d2d(
    }
    if (firstFrame) 
	{
      rotationOnly = rotationOnly_tmp;
      firstFrame = false;
    }

    kfcounter++;
    if (kfcounter > 1)
      break;
  }

  // calculate fraction of safe matches
  if (uncertainMatchFraction)
  {
    *uncertainMatchFraction = double(numUncertainMatches) / double(retCtr);
  }

  return retCtr;
}

// Match a new multiframe to the last frame.
template<class MATCHING_ALGORITHM>
int Frontend::matchToLastFrame(okvis::Estimator& estimator,
                               const okvis::VioParameters& params,
                               const uint64_t currentFrameId,
                               bool usePoseUncertainty,
                               bool removeOutliers) {

  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return 0;
  }

  uint64_t lastFrameId = estimator.frameIdByAge(1);//�͵�ǰ֡�����֡

  if (estimator.isKeyframe(lastFrameId)) //��������һ֡�ǹؼ�֡��ֱ�ӷ���
  {
    // already done
    return 0;
  }

  int retCtr = 0;

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im)//�����������
  {
    MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                         MATCHING_ALGORITHM::Match3D2D,
                                         briskMatchingThreshold_,
                                         usePoseUncertainty);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 3D-2D
    matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    retCtr += matchingAlgorithm.numMatches();
  }

  runRansac3d2d(estimator, params.nCameraSystem,estimator.multiFrame(currentFrameId), removeOutliers);

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im)
  {
    MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                         MATCHING_ALGORITHM::Match2D2D,
                                         briskMatchingThreshold_,
                                         usePoseUncertainty);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 2D-2D for initialization of new (mono-)correspondences
    matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    retCtr += matchingAlgorithm.numMatches();
  }

  // remove outliers
  bool rotationOnly = false;
  if (!isInitialized_)
    runRansac2d2d(estimator, params, currentFrameId, lastFrameId, false, removeOutliers, rotationOnly);

  return retCtr;
}

// Match the frames inside the multiframe to each other to initialise new landmarks.
//�����ģ������Ϊ VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> >
template<class MATCHING_ALGORITHM>
void Frontend::matchStereo(okvis::Estimator& estimator,std::shared_ptr<okvis::MultiFrame> multiFrame)
{

  const size_t camNumber = multiFrame->numFrames();//�������е�������� ������ݷ���ֻ����1����ΪmatchStereo����ֻ���ڵ�һ֡ʱ�Ż�����������
  const uint64_t mfId = multiFrame->id();//Ҫƥ���֡��id

  
  for (size_t im0 = 0; im0 < camNumber; im0++)//����Ҫ���Ƶ����
  {
    
    for (size_t im1 = im0 + 1; im1 < camNumber; im1++) 
	{
      // first, check the possibility for overlap
      // FIXME: implement this in the Multiframe...!!

      // check overlap ��֮֡���Ƿ����ص��Ĳ���
      if(!multiFrame->hasOverlap(im0, im1))//���� NCameraSystem::hasOverlap(
	  {
        continue;
      }

      //�������캯�� VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::VioKeyframeWindowMatchingAlgorithm(
      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match2D2D,
                                           briskMatchingThreshold_,//Ĭ��60
                                           false);  // usePoseUncertainty=false
      //���� void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setFrames
      //����Ĳ����ֱ��� ˫Ŀ֡A��id ˫Ŀ֡B��id ˫Ŀ֡A����ͼ��or��ͼ�� ˫Ŀ֡B����ͼ��or��ͼ��
      matchingAlgorithm.setFrames(mfId, mfId, im0, im1);  // newest frame
      // match 2D-2D
      //���� void DenseMatcher::match(MATCHING_ALGORITHM_T & matchingAlgorithm) 
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);

      // match 3D-2D
      matchingAlgorithm.setMatchingType(MATCHING_ALGORITHM::Match3D2D);
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);

      // match 2D-3D
      matchingAlgorithm.setFrames(mfId, mfId, im1, im0);  // newest frame
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    }
  }

  // TODO: for more than 2 cameras check that there were no duplications!

  // TODO: ensure 1-1 matching.

  // TODO: no RANSAC ?

  for (size_t im = 0; im < camNumber; im++)
  {
    const size_t ksize = multiFrame->numKeypoints(im);
    for (size_t k = 0; k < ksize; ++k) 
	{
      if (multiFrame->landmarkId(im, k) != 0) //��ʾ�Ѿ��͵�ͼ�����Ӧ
	  {
        continue;  // already identified correspondence
      }
	  //���� bool MultiFrame::setLandmarkId
	  //im��ʾ����ͼ������ͼ��k��ʾ��������ͼ���е����
      multiFrame->setLandmarkId(im, k, okvis::IdProvider::instance().newId());//�������˫Ŀ֡�������Ӧ�ĵ�ͼ��
    }
  }
}

// Perform 3D/2D RANSAC.
/**
 * @brief Perform 3D/2D RANSAC.
 * @warning As this function uses the estimator it is not threadsafe.
 * @param estimator 	  Estimator.
 * @param nCameraSystem   Camera configuration and parameters.
 * @param currentFrame	  Frame with the new potential matches.
 * @param removeOutliers  Remove observation of outliers in estimator.
 * @return Number of inliers.
 */
 //����Ĳ���removeOutliers=true
int Frontend::runRansac3d2d(okvis::Estimator& estimator,
                            const okvis::cameras::NCameraSystem& nCameraSystem,
                            std::shared_ptr<okvis::MultiFrame> currentFrame,
                            bool removeOutliers) {
  if (estimator.numFrames() < 2) {
    // nothing to match against, we are just starting up.
    return 1;
  }

  /////////////////////
  //   KNEIP RANSAC
  /////////////////////
  int numInliers = 0;

  // absolute pose adapter for Kneip toolchain
  //���� opengv::absolute_pose::FrameNoncentralAbsoluteAdapter::FrameNoncentralAbsoluteAdapter(
  opengv::absolute_pose::FrameNoncentralAbsoluteAdapter adapter(estimator,
                                                                nCameraSystem,
                                                                currentFrame);

  size_t numCorrespondences = adapter.getNumberCorrespondences();//��õ������
  if (numCorrespondences < 5)
    return numCorrespondences;

  // create a RelativePoseSac problem and RANSAC
  //����ʹ�õľ���opengv�Դ��Ķ�����
  //FrameAbsolutePoseSacProblem���Լ������ Ӧ���Ǿ���ִ��һ��pnp-ransac
  opengv::sac::Ransac<opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem> ransac;
  std::shared_ptr<
      opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem> absposeproblem_ptr(new opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem(
          adapter, opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem::Algorithm::GP3P));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 9;
  ransac.max_iterations_ = 50;
  // initial guess not needed...
  // run the ransac
  ransac.computeModel(0);//��ʼ����ransac

  // assign transformation
  numInliers = ransac.inliers_.size();//ransac������ �õ��ڵ���
  if (numInliers >= 10) 
  {

    // kick out outliers:
    std::vector<bool> inliers(numCorrespondences, false); 
    for (size_t k = 0; k < ransac.inliers_.size(); ++k) 
	{
      inliers.at(ransac.inliers_.at(k)) = true;
    }

    for (size_t k = 0; k < numCorrespondences; ++k) 
	{
      if (!inliers[k]) 
	  {
        // get the landmark id:
        size_t camIdx = adapter.camIndex(k);
        size_t keypointIdx = adapter.keypointIndex(k);
        uint64_t lmId = currentFrame->landmarkId(camIdx, keypointIdx);//�õ�����������Ӧ�ĵ�ͼ���id

        // reset ID:
        //�������֡��������û�е�ͼ������Ӧ
        currentFrame->setLandmarkId(camIdx, keypointIdx, 0);

        // remove observation
        if (removeOutliers)//Ĭ��������true
		{
		     //1.����landmarksMap_���ݣ������ͼ��û�й۲⵽�����������
		 	//2.��ceres��ɾ������в�飬��������ر���
          estimator.removeObservation(lmId, currentFrame->id(), camIdx,keypointIdx);
        }
      }
    }
  }
  return numInliers;
}

// Perform 2D/2D RANSAC.
/**
 * @brief Perform 2D/2D RANSAC.
 * @warning As this function uses the estimator it is not threadsafe.
 * @param estimator 		Estimator.
 * @param params			Parameter struct.
 * @param currentFrameId	ID of the new multiframe containing matches with the frame with ID olderFrameId.
 * @param olderFrameId		ID of the multiframe to which the current frame has been matched against.
 * @param initializePose	If the pose has not yet been initialised should the function try to initialise it.
 * @param removeOutliers	Remove observation of outliers in estimator.
 * @param[out] rotationOnly Was the rotation only RANSAC model enough to explain the matches.
 * @return Number of inliers.
 */
 //removeOutliers=true initializePose=true
 //���rotationOnly=true ���ʾ��ɳ�ʼ����
int Frontend::runRansac2d2d(okvis::Estimator& estimator,
                            const okvis::VioParameters& params,
                            uint64_t currentFrameId, uint64_t olderFrameId,
                            bool initializePose,
                            bool removeOutliers,
                            bool& rotationOnly) {
  // match 2d2d
  rotationOnly = false;
  const size_t numCameras = params.nCameraSystem.numCameras();

  size_t totalInlierNumber = 0;
  bool rotation_only_success = false;
  bool rel_pose_success = false;

  // run relative RANSAC
  //�����������
  for (size_t im = 0; im < numCameras; ++im) 
  {

    // relative pose adapter for Kneip toolchain
    //�Ƚ���Ҫ�ĺ���
    //���� opengv::relative_pose::FrameRelativeAdapter::FrameRelativeAdapter( 
    opengv::relative_pose::FrameRelativeAdapter adapter(estimator,
                                                        params.nCameraSystem,
                                                        olderFrameId, im,
                                                        currentFrameId, im);

    size_t numCorrespondences = adapter.getNumberCorrespondences();

    if (numCorrespondences < 10)
      continue;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!

    // try both the rotation-only RANSAC and the relative one:

    // create a RelativePoseSac problem and RANSAC
    //FrameRotationOnlySacProblem���Լ������
    typedef opengv::sac_problems::relative_pose::FrameRotationOnlySacProblem FrameRotationOnlySacProblem;
    opengv::sac::Ransac<FrameRotationOnlySacProblem> rotation_only_ransac;
    std::shared_ptr<FrameRotationOnlySacProblem> rotation_only_problem_ptr(new FrameRotationOnlySacProblem(adapter));
    rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
    rotation_only_ransac.threshold_ = 9;
    rotation_only_ransac.max_iterations_ = 50;

    // run the ransac
    rotation_only_ransac.computeModel(0);//��ɴ���ת��2d-2d ransac

    // get quality
    int rotation_only_inliers = rotation_only_ransac.inliers_.size();
    float rotation_only_ratio = float(rotation_only_inliers)/ float(numCorrespondences);

    // now the rel_pose one:
    //FrameRelativePoseSacProblem���Լ������
    typedef opengv::sac_problems::relative_pose::FrameRelativePoseSacProblem FrameRelativePoseSacProblem;
    opengv::sac::Ransac<FrameRelativePoseSacProblem> rel_pose_ransac;
    std::shared_ptr<FrameRelativePoseSacProblem> rel_pose_problem_ptr(new FrameRelativePoseSacProblem(adapter, FrameRelativePoseSacProblem::STEWENIUS));
    rel_pose_ransac.sac_model_ = rel_pose_problem_ptr;
    rel_pose_ransac.threshold_ = 9;     //(1.0 - cos(0.5/600));
    rel_pose_ransac.max_iterations_ = 50;

    // run the ransac
    rel_pose_ransac.computeModel(0);//������λ�Ƶ�2d-2d ransac

    // assess success
    int rel_pose_inliers = rel_pose_ransac.inliers_.size();
    float rel_pose_ratio = float(rel_pose_inliers) / float(numCorrespondences);

    // decide on success and fill inliers
    std::vector<bool> inliers(numCorrespondences, false);
    if (rotation_only_ratio > rel_pose_ratio || rotation_only_ratio > 0.8) //���������ʾ������ֻ��������ת
	{
      if (rotation_only_inliers > 10) 
	  {
        rotation_only_success = true;
      }
      rotationOnly = true;//�������ת��ransace���õ�������λ�Ƶ�ransac
      totalInlierNumber += rotation_only_inliers;
      for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k) 
	  {
        inliers.at(rotation_only_ransac.inliers_.at(k)) = true;
      }
    }else //���������ʾ�����˽�����λ�ƺ���ת
    {
      if (rel_pose_inliers > 10) 
	  {
        rel_pose_success = true;//
      }
      totalInlierNumber += rel_pose_inliers;
      for (size_t k = 0; k < rel_pose_ransac.inliers_.size(); ++k) 
	  {
        inliers.at(rel_pose_ransac.inliers_.at(k)) = true;
      }
    }

    // failure?
    if (!rotation_only_success && !rel_pose_success) //��ʾ����ת��λ�Ƶ�ransac inliner��̫�٣�ֱ��������
	{
      continue;
    }

    // otherwise: kick out outliers!
    //�Ӹ��µ�״̬��ɾ��outlier
    std::shared_ptr<okvis::MultiFrame> multiFrame = estimator.multiFrame(currentFrameId);

    for (size_t k = 0; k < numCorrespondences; ++k) 
	{
      size_t idxB = adapter.getMatchKeypointIdxB(k);
      if (!inliers[k]) 
	  {
        uint64_t lmId = multiFrame->landmarkId(im, k);
        // reset ID:
        multiFrame->setLandmarkId(im, k, 0);
        // remove observation
        if (removeOutliers) 
		{
          if (lmId != 0 && estimator.isLandmarkAdded(lmId))
		  {
            estimator.removeObservation(lmId, currentFrameId, im, idxB);
          }
        }
      }
    }

    // initialize pose if necessary
    //�����ؾ���������ϵ�µ�λ��
    if (initializePose && !isInitialized_) //��ϵͳ��û�г�ʼ��ʱ
	{
      if (rel_pose_success)
        LOG(INFO)
            << "Initializing pose from 2D-2D RANSAC";
      else
        LOG(INFO)
            << "Initializing pose from 2D-2D RANSAC: orientation only";

      Eigen::Matrix4d T_C1C2_mat = Eigen::Matrix4d::Identity();//��֡ͼ�����̬��

      okvis::kinematics::Transformation T_SCA, T_WSA, T_SC0, T_WS0;
      uint64_t idA = olderFrameId;
      uint64_t id0 = currentFrameId;
      estimator.getCameraSensorStates(idA, im, T_SCA);//�õ�֮ǰ֡�������imu�任�����
      estimator.get_T_WS(idA, T_WSA);//�õ�֮ǰ֡����̬
      estimator.getCameraSensorStates(id0, im, T_SC0);//�õ���ǰ֡�������imu�任�����
      estimator.get_T_WS(id0, T_WS0);//�õ���ǰ֡����̬ Ӧ������imu���ֵõ���
      
      //�������if��Ϊ�˼������ǰ֡��֮ǰ֡�������̬�任
      if (rel_pose_success)//������������true���ʾ���������λ�Ƶ��ƶ�����ֻ�ǽ�������̬�ı任
	  {
        // update pose
        // if the IMU is used, this will be quickly optimized to the correct scale. Hopefully.
        T_C1C2_mat.topLeftCorner<3, 4>() = rel_pose_ransac.model_coefficients_;//����ͼ��֮��������̬�任��ͨ��2d-2dransac����õ���

        //initialize with projected length according to motion prior.

        okvis::kinematics::Transformation T_C1C2 = T_SCA.inverse()* T_WSA.inverse() * T_WS0 * T_SC0;//ͨ��imu���ֵõ��������������̬��
        //???????????????Ϊʲô�߶�����ô����
        //������õ������λ��*�߶�=imu�߶��µ����λ��
        T_C1C2_mat.topRightCorner<3, 1>() = T_C1C2_mat.topRightCorner<3, 1>()* std::max(0.0,double(T_C1C2_mat.topRightCorner<3, 1>().transpose()* T_C1C2.r()));
      } else //ֻ��������ת
      {
        // rotation only assigned...
        T_C1C2_mat.topLeftCorner<3, 3>() = rotation_only_ransac.model_coefficients_;
      }

      // set.���õ�ǰ֡����������ϵ�µ�����
      estimator.set_T_WS(id0,T_WSA * T_SCA * okvis::kinematics::Transformation(T_C1C2_mat)* T_SC0.inverse());
    }
  }//���������������

  if (rel_pose_success || rotation_only_success)
    return totalInlierNumber;
  else 
  {
    rotationOnly = true;  // hack...
    return -1;
  }

  return 0;
}

// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or at startup.
void Frontend::initialiseBriskFeatureDetectors() {
  for (auto it = featureDetectorMutexes_.begin();
      it != featureDetectorMutexes_.end(); ++it) {
    (*it)->lock();
  }
  featureDetectors_.clear();
  descriptorExtractors_.clear();
  for (size_t i = 0; i < numCameras_; ++i) //�������������������������ȡ�����������ӷ���
  {
    featureDetectors_.push_back( std::shared_ptr<cv::FeatureDetector>(
//�������ǰѲ���Ҫ��ע�͵���������Ķ�
/*
#ifdef __ARM_NEON__
            new cv::GridAdaptedFeatureDetector( 
            new cv::FastFeatureDetector(briskDetectionThreshold_),
                briskDetectionMaximumKeypoints_, 7, 4 ))); // from config file, except the 7x4...
#else*/
	        //���漸���������������ļ����ɵ�
            new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(briskDetectionThreshold_, briskDetectionOctaves_,briskDetectionAbsoluteThreshold_,briskDetectionMaximumKeypoints_)));
//#endif
	//���漸��������Ĭ�����õ�
    descriptorExtractors_.push_back(std::shared_ptr<cv::DescriptorExtractor>( new brisk::BriskDescriptorExtractor(briskDescriptionRotationInvariance_,briskDescriptionScaleInvariance_)));
  }
  for (auto it = featureDetectorMutexes_.begin();it != featureDetectorMutexes_.end(); ++it) 
  {
    (*it)->unlock();
  }
}

}  // namespace okvis
