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
//T_WC=根据imu得到的相机的位姿，是输入变量
bool Frontend::detectAndDescribe(size_t cameraIndex,
                                 std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint> * keypoints) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < numCameras_, "Camera index exceeds number of cameras.");
  std::lock_guard<std::mutex> lock(*featureDetectorMutexes_[cameraIndex]);

  // check there are no keypoints here
  OKVIS_ASSERT_TRUE(Exception, keypoints == nullptr, "external keypoints currently not supported")

  frameOut->setDetector(cameraIndex, featureDetectors_[cameraIndex]);//作者这里使用的是brisk特征点提取方法
  frameOut->setExtractor(cameraIndex, descriptorExtractors_[cameraIndex]);//作者这里使用的是brisk提供的描述子

  frameOut->detect(cameraIndex);//提取特征点，搜索 int Frame::detect()

  // ExtractionDirection == gravity direction in camera frame
  Eigen::Vector3d g_in_W(0, 0, -1);
  Eigen::Vector3d extractionDirection = T_WC.inverse().C() * g_in_W;
  frameOut->describe(cameraIndex, extractionDirection);//提取描述子，搜索 int Frame::describe(const Eigen::Vector3d & extractionDirection)

  // set detector/extractor to nullpointer? TODO
  return true;
}

// Matching as well as initialization of landmarks and state.
//Transformation根据优化得到的当前帧的位姿
//framesInOut已经提取过特征点和描述子的双目图像 既是输入也是输出
//asKeyframe 输出变量，这一帧是否为关键帧
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
  okvis::cameras::NCameraSystem::DistortionType distortionType = params.nCameraSystem.distortionType(0);//euroc配置是 radialtangential类型
  for (size_t i = 1; i < params.nCameraSystem.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception,
                      distortionType == params.nCameraSystem.distortionType(i),
                      "mixed frame types are not supported yet");
  }
  int num3dMatches = 0;

  // 1.first frame? (did do addStates before, so 1 frame minimum in estimator)
  //1.判断是否为第一帧图像
  if (estimator.numFrames() > 1) //不是初始化状态
  {

    int requiredMatches = 5;//需要的最小匹配对数

    double uncertainMatchFraction = 0;
    bool rotationOnly = false;

    // match to last keyframe
    TimerSwitchable matchKeyframesTimer("2.4.1 matchToKeyframes");//什么都没干
    switch (distortionType) 
	{
	  //1.1当前帧和关键帧进行匹配!!!!!!!!!!!!!!!
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
    //matchKeyframesTimer.stop();//什么都没干，注释方便看代码
    if (!isInitialized_) 
	{
      if (!rotationOnly) 
	  {
        isInitialized_ = true;//整个代码中就里isInitialized_被赋值了
        LOG(INFO) << "Initialized!";
      }
    }

    if (num3dMatches <= requiredMatches) 
	{
      LOG(WARNING) << "Tracking failure. Number of 3d2d-matches: " << num3dMatches;
    }

    // keyframe decision, at the moment only landmarks that match with keyframe are initialised
    //1.2判断当前帧是否设置为关键帧
    *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, framesInOut);//比较重要的函数!!!!!!!!!!!!!!!!!!!!

    // match to last frame
    TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrame");//什么都没干
    switch (distortionType) {
	  //1.3和上一帧进行匹配
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
    //matchToLastFrameTimer.stop();//什么都没干，这里注释掉方便看代码
  } else//初始化状态不进行匹配 直接将第一帧设置为关键帧
    *asKeyframe = true;  // first frame needs to be keyframe


  // 2.do stereo match to get new landmarks
  TimerSwitchable matchStereoTimer("2.4.3 matchStereo");//什么都没干
  switch (distortionType) 
  {
  	//进行双目匹配
    case okvis::cameras::NCameraSystem::RadialTangential: 
	{
		//整个代码中就这里调用了matchStereo函数
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
  //matchStereoTimer.stop();//什么都没干 注释掉方便看代码

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
  for (size_t im = 0; im < currentFrame->numFrames(); ++im) //遍历左右相机
  {

    // get the hull of all keypoints in current frame
    std::vector<cv::Point2f> frameBPoints, frameBHull;
    std::vector<cv::Point2f> frameBMatches, frameBMatchesHull;
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > frameBLandmarks;

    const size_t numB = currentFrame->numKeypoints(im);//提取出左相机或者右相机的所有特征点
    frameBPoints.reserve(numB);
    frameBLandmarks.reserve(numB);
    Eigen::Vector2d keypoint;
    for (size_t k = 0; k < numB; ++k) //遍历左相机或者右相机的所有特征点
	{
      currentFrame->getKeypoint(im, k, keypoint);
      // insert it
      frameBPoints.push_back(cv::Point2f(keypoint[0], keypoint[1]));//存储的是所有特征点
      // also remember matches
      if (currentFrame->landmarkId(im, k) != 0) 
	  {
        frameBMatches.push_back(cv::Point2f(keypoint[0], keypoint[1]));//存储的是已经和地图匹配的特征点
      }
    }

    if (frameBPoints.size() < 3)
      continue;
    cv::convexHull(frameBPoints, frameBHull);//寻找所有特征点构造的凸包点，frameBHull
    if (frameBMatches.size() < 3)
      continue;
    cv::convexHull(frameBMatches, frameBMatchesHull);//寻找已经和地图点匹配的特征点所构造的凸包点，frameBMatchesHull

    // areas
    double frameBArea = cv::contourArea(frameBHull);//所有特征点凸包的面积
    double frameBMatchesArea = cv::contourArea(frameBMatchesHull);//已经匹配的凸包的面积

    // overlap area
    double overlapArea = frameBMatchesArea / frameBArea;//已经匹配的凸包的面积/所有特征点的凸包面积
    // matching ratio inside overlap area: count
    int pointsInFrameBMatchesArea = 0;
    if (frameBMatchesHull.size() > 2) 
	{
      for (size_t k = 0; k < frameBPoints.size(); ++k) //遍历所有特征点
	  {
	    //检测所有特征点是否在匹配的凸包的面积内
        if (cv::pointPolygonTest(frameBMatchesHull, frameBPoints[k], false)> 0) //检测点是否在轮廓内，false表示返回的结果是数字，+1表示在轮廓内
		{
          pointsInFrameBMatchesArea++;
        }
      }
    }
    double matchingRatio = double(frameBMatches.size())/ double(pointsInFrameBMatchesArea);//已经匹配和地图点匹配的特征点个数/在凸包内的特征点个数

    // calculate overlap score
    overlap = std::max(overlapArea, overlap);//这另个变量中存储的是左右图像最高的比例
    ratio = std::max(matchingRatio, ratio);
  }

  // take a decision
  ////keyframeInsertionOverlapThreshold_=0.6
  //keyframeInsertionMatchingRatioThreshold_= 0.2
  //必须如下两个条件都不满足才会创建关键帧
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
//输入的模板MATCHING_ALGORITHM = VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > 
//输入的参数为 rotationOnly=false usePoseUncertainty=false uncertainMatchFraction=0
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

  // go through all the frames and try to match the initialized keypoints、
  //将双目相机的图像和最近的三个关键帧进行3D-2D匹配
  size_t kfcounter = 0;
  for (size_t age = 1; age < estimator.numFrames(); ++age) 
  {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId))
      continue;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im)//遍历左右图像
	{
	  //匹配算法初始化
	  VioKeyframeWindowMatchingAlgorithm
	  //搜索 VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::VioKeyframeWindowMatchingAlgorithm
      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match3D2D,
                                           briskMatchingThreshold_,//默认60
                                           usePoseUncertainty);//输入为false
      //搜索 void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setFrames
      //olderFrameId是以前关键帧的双目id currentFrameId是当前双目帧的id im是左相机还是右相机
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 3D-2D
      //开始进行匹配，match函数调用了下面的两个函数
      //定义 std::unique_ptr<okvis::DenseMatcher> matcher_; 
      //搜索 void DenseMatcher::match(MATCHING_ALGORITHM_T & matchingAlgorithm) 
     	 //搜索 void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::doSetup() {
     	 //搜索 void DenseMatcher::matchBody( void，搜索 doWorkLinearMatching
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      retCtr += matchingAlgorithm.numMatches();
      numUncertainMatches += matchingAlgorithm.numUncertainMatches();

    }
    kfcounter++;
    if (kfcounter > 2)
      break;
  }

  //与最近的两个关键帧进行2D-2D匹配
  //如果已经初始化完成了那么进行ransace3d-2d匹配
  //如果系统初始化没有完成那么记性ransac2d-2d匹配，并且这个匹配决定是否已经初始化完成
  kfcounter = 0;
  bool firstFrame = true;
  for (size_t age = 1; age < estimator.numFrames(); ++age) 
  {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId))
      continue;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) 
	{
	  //匹配算法初始化
      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match2D2D,
                                           briskMatchingThreshold_,//默认60
                                           usePoseUncertainty);//输入为false
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 2D-2D for initialization of new (mono-)correspondences
      //下面开始匹配
      //搜索 DenseMatcher::match
     	 //搜索 void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::doSetup() {
     	 //搜索 void DenseMatcher::matchBody( void，搜索 doWorkLinearMatching
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      retCtr += matchingAlgorithm.numMatches();
      numUncertainMatches += matchingAlgorithm.numUncertainMatches();
    }

    // remove outliers
    // only do RANSAC 3D2D with most recent KF
    if (kfcounter == 0 && isInitialized_)//这里一定要注意了 只有初始化完成后才进行ransac3d2d 
   	{
   	   //在前面我们已经使用doSetup将当前帧的特征点和地图点进行了匹配，现在这里我们使用PnP的ransac算法去除那些outlier,并将这些outlier从map中的ceres优化删除
       runRansac3d2d(estimator, params.nCameraSystem,estimator.multiFrame(currentFrameId), removeOutliers);//搜索 int Frontend::runRansac3d2d(
    }

    bool rotationOnly_tmp = false;
    // do RANSAC 2D2D for initialization only
    if (!isInitialized_) //这里一定要注意只有在系统没有初始化时才进行ransac2d2d
	{
      runRansac2d2d(estimator, params, currentFrameId, olderFrameId, true, removeOutliers, rotationOnly_tmp);//搜索 int Frontend::runRansac2d2d(
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

  uint64_t lastFrameId = estimator.frameIdByAge(1);//和当前帧最近的帧

  if (estimator.isKeyframe(lastFrameId)) //如果最近的一帧是关键帧则直接返回
  {
    // already done
    return 0;
  }

  int retCtr = 0;

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im)//遍历左右相机
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
//输入的模板类型为 VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> >
template<class MATCHING_ALGORITHM>
void Frontend::matchStereo(okvis::Estimator& estimator,std::shared_ptr<okvis::MultiFrame> multiFrame)
{

  const size_t camNumber = multiFrame->numFrames();//估计器中的相机数量 这里根据分析只能是1，因为matchStereo函数只有在第一帧时才会调用这个函数
  const uint64_t mfId = multiFrame->id();//要匹配的帧的id

  
  for (size_t im0 = 0; im0 < camNumber; im0++)//遍历要估计的相机
  {
    
    for (size_t im1 = im0 + 1; im1 < camNumber; im1++) 
	{
      // first, check the possibility for overlap
      // FIXME: implement this in the Multiframe...!!

      // check overlap 两帧之间是否有重叠的部分
      if(!multiFrame->hasOverlap(im0, im1))//搜索 NCameraSystem::hasOverlap(
	  {
        continue;
      }

      //搜索构造函数 VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::VioKeyframeWindowMatchingAlgorithm(
      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match2D2D,
                                           briskMatchingThreshold_,//默认60
                                           false);  // usePoseUncertainty=false
      //搜索 void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setFrames
      //输入的参数分别是 双目帧A的id 双目帧B的id 双目帧A是左图像or右图像 双目帧B是左图像or右图像
      matchingAlgorithm.setFrames(mfId, mfId, im0, im1);  // newest frame
      // match 2D-2D
      //搜索 void DenseMatcher::match(MATCHING_ALGORITHM_T & matchingAlgorithm) 
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
      if (multiFrame->landmarkId(im, k) != 0) //表示已经和地图点相对应
	  {
        continue;  // already identified correspondence
      }
	  //搜索 bool MultiFrame::setLandmarkId
	  //im表示是左图像还是右图像，k表示特征点在图像中的序号
      multiFrame->setLandmarkId(im, k, okvis::IdProvider::instance().newId());//设置这个双目帧特征点对应的地图点
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
 //输入的参数removeOutliers=true
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
  //搜索 opengv::absolute_pose::FrameNoncentralAbsoluteAdapter::FrameNoncentralAbsoluteAdapter(
  opengv::absolute_pose::FrameNoncentralAbsoluteAdapter adapter(estimator,
                                                                nCameraSystem,
                                                                currentFrame);

  size_t numCorrespondences = adapter.getNumberCorrespondences();//获得点的数量
  if (numCorrespondences < 5)
    return numCorrespondences;

  // create a RelativePoseSac problem and RANSAC
  //下面使用的就是opengv自带的东西了
  //FrameAbsolutePoseSacProblem是自己定义的 应该是就是执行一个pnp-ransac
  opengv::sac::Ransac<opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem> ransac;
  std::shared_ptr<
      opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem> absposeproblem_ptr(new opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem(
          adapter, opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem::Algorithm::GP3P));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 9;
  ransac.max_iterations_ = 50;
  // initial guess not needed...
  // run the ransac
  ransac.computeModel(0);//开始进行ransac

  // assign transformation
  numInliers = ransac.inliers_.size();//ransac结束后 得到内点数
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
        uint64_t lmId = currentFrame->landmarkId(camIdx, keypointIdx);//得到这个特征点对应的地图点的id

        // reset ID:
        //设置这个帧的特征点没有地图点和其对应
        currentFrame->setLandmarkId(camIdx, keypointIdx, 0);

        // remove observation
        if (removeOutliers)//默认设置是true
		{
		     //1.更新landmarksMap_数据，这个地图点没有观测到过这个特征点
		 	//2.从ceres中删除这个残差块，并更新相关变量
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
 //如果rotationOnly=true 则表示完成初始化了
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
  //遍历左右相机
  for (size_t im = 0; im < numCameras; ++im) 
  {

    // relative pose adapter for Kneip toolchain
    //比较重要的函数
    //搜索 opengv::relative_pose::FrameRelativeAdapter::FrameRelativeAdapter( 
    opengv::relative_pose::FrameRelativeAdapter adapter(estimator,
                                                        params.nCameraSystem,
                                                        olderFrameId, im,
                                                        currentFrameId, im);

    size_t numCorrespondences = adapter.getNumberCorrespondences();

    if (numCorrespondences < 10)
      continue;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!

    // try both the rotation-only RANSAC and the relative one:

    // create a RelativePoseSac problem and RANSAC
    //FrameRotationOnlySacProblem是自己定义的
    typedef opengv::sac_problems::relative_pose::FrameRotationOnlySacProblem FrameRotationOnlySacProblem;
    opengv::sac::Ransac<FrameRotationOnlySacProblem> rotation_only_ransac;
    std::shared_ptr<FrameRotationOnlySacProblem> rotation_only_problem_ptr(new FrameRotationOnlySacProblem(adapter));
    rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
    rotation_only_ransac.threshold_ = 9;
    rotation_only_ransac.max_iterations_ = 50;

    // run the ransac
    rotation_only_ransac.computeModel(0);//完成纯旋转的2d-2d ransac

    // get quality
    int rotation_only_inliers = rotation_only_ransac.inliers_.size();
    float rotation_only_ratio = float(rotation_only_inliers)/ float(numCorrespondences);

    // now the rel_pose one:
    //FrameRelativePoseSacProblem是自己定义的
    typedef opengv::sac_problems::relative_pose::FrameRelativePoseSacProblem FrameRelativePoseSacProblem;
    opengv::sac::Ransac<FrameRelativePoseSacProblem> rel_pose_ransac;
    std::shared_ptr<FrameRelativePoseSacProblem> rel_pose_problem_ptr(new FrameRelativePoseSacProblem(adapter, FrameRelativePoseSacProblem::STEWENIUS));
    rel_pose_ransac.sac_model_ = rel_pose_problem_ptr;
    rel_pose_ransac.threshold_ = 9;     //(1.0 - cos(0.5/600));
    rel_pose_ransac.max_iterations_ = 50;

    // run the ransac
    rel_pose_ransac.computeModel(0);//完成相对位移的2d-2d ransac

    // assess success
    int rel_pose_inliers = rel_pose_ransac.inliers_.size();
    float rel_pose_ratio = float(rel_pose_inliers) / float(numCorrespondences);

    // decide on success and fill inliers
    std::vector<bool> inliers(numCorrespondences, false);
    if (rotation_only_ratio > rel_pose_ratio || rotation_only_ratio > 0.8) //这个条件表示机器人只进行了旋转
	{
      if (rotation_only_inliers > 10) 
	  {
        rotation_only_success = true;
      }
      rotationOnly = true;//如果纯旋转的ransace内置点多于相对位移的ransac
      totalInlierNumber += rotation_only_inliers;
      for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k) 
	  {
        inliers.at(rotation_only_ransac.inliers_.at(k)) = true;
      }
    }else //这个条件表示机器人进行了位移和旋转
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
    if (!rotation_only_success && !rel_pose_success) //表示纯旋转和位移的ransac inliner都太少，直接跳过。
	{
      continue;
    }

    // otherwise: kick out outliers!
    //从更新的状态中删除outlier
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
    //更新载具世界坐标系下的位姿
    if (initializePose && !isInitialized_) //在系统还没有初始化时
	{
      if (rel_pose_success)
        LOG(INFO)
            << "Initializing pose from 2D-2D RANSAC";
      else
        LOG(INFO)
            << "Initializing pose from 2D-2D RANSAC: orientation only";

      Eigen::Matrix4d T_C1C2_mat = Eigen::Matrix4d::Identity();//两帧图像的姿态差

      okvis::kinematics::Transformation T_SCA, T_WSA, T_SC0, T_WS0;
      uint64_t idA = olderFrameId;
      uint64_t id0 = currentFrameId;
      estimator.getCameraSensorStates(idA, im, T_SCA);//得到之前帧的相机到imu变换的外参
      estimator.get_T_WS(idA, T_WSA);//得到之前帧的姿态
      estimator.getCameraSensorStates(id0, im, T_SC0);//得到当前帧的相机到imu变换的外参
      estimator.get_T_WS(id0, T_WS0);//得到当前帧的姿态 应该是由imu积分得到的
      
      //下面这个if是为了计算出当前帧和之前帧的相对姿态变换
      if (rel_pose_success)//如果这个变量是true则表示相机进行了位移的移动而不只是进行了姿态的变换
	  {
        // update pose
        // if the IMU is used, this will be quickly optimized to the correct scale. Hopefully.
        T_C1C2_mat.topLeftCorner<3, 4>() = rel_pose_ransac.model_coefficients_;//两个图像之间的相对姿态变换，通过2d-2dransac计算得到的

        //initialize with projected length according to motion prior.

        okvis::kinematics::Transformation T_C1C2 = T_SCA.inverse()* T_WSA.inverse() * T_WS0 * T_SC0;//通过imu积分得到的两个相机的姿态差
        //???????????????为什么尺度是这么求解的
        //用相机得到的相对位姿*尺度=imu尺度下的相对位姿
        T_C1C2_mat.topRightCorner<3, 1>() = T_C1C2_mat.topRightCorner<3, 1>()* std::max(0.0,double(T_C1C2_mat.topRightCorner<3, 1>().transpose()* T_C1C2.r()));
      } else //只进行了旋转
      {
        // rotation only assigned...
        T_C1C2_mat.topLeftCorner<3, 3>() = rotation_only_ransac.model_coefficients_;
      }

      // set.设置当前帧在世界坐标系下的坐标
      estimator.set_T_WS(id0,T_WSA * T_SCA * okvis::kinematics::Transformation(T_C1C2_mat)* T_SC0.inverse());
    }
  }//遍历左右相机结束

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
  for (size_t i = 0; i < numCameras_; ++i) //设置左右两个相机的特征点提取方法和描述子方法
  {
    featureDetectors_.push_back( std::shared_ptr<cv::FeatureDetector>(
//这里我们把不需要的注释掉方便代码阅读
/*
#ifdef __ARM_NEON__
            new cv::GridAdaptedFeatureDetector( 
            new cv::FastFeatureDetector(briskDetectionThreshold_),
                briskDetectionMaximumKeypoints_, 7, 4 ))); // from config file, except the 7x4...
#else*/
	        //下面几个参数是由配置文件生成的
            new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(briskDetectionThreshold_, briskDetectionOctaves_,briskDetectionAbsoluteThreshold_,briskDetectionMaximumKeypoints_)));
//#endif
	//下面几个参数是默认配置的
    descriptorExtractors_.push_back(std::shared_ptr<cv::DescriptorExtractor>( new brisk::BriskDescriptorExtractor(briskDescriptionRotationInvariance_,briskDescriptionScaleInvariance_)));
  }
  for (auto it = featureDetectorMutexes_.begin();it != featureDetectorMutexes_.end(); ++it) 
  {
    (*it)->unlock();
  }
}

}  // namespace okvis
