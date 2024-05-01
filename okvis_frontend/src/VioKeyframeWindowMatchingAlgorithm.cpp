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
//如果对于双目匹配则输入参数mfIdA=mfIdB，camIdA=0，camIdB=1
//输入的参数分别是 双目帧A的id 双目帧B的id 双目帧A是左图像or右图像 双目帧B是左图像or右图像
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setFrames(uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB) 
{

  OKVIS_ASSERT_TRUE(Exception, !(mfIdA == mfIdB && camIdA == camIdB), "trying to match identical frames.");

  // remember indices
  mfIdA_ = mfIdA;//双目帧A的id
  mfIdB_ = mfIdB;//双目帧B的id
  camIdA_ = camIdA;//双目帧A是左相机还是右相机
  camIdB_ = camIdB;//双目帧B是左相机还是右相机
  // frames and related information
  frameA_ = estimator_->multiFrame(mfIdA_);//得到对应的图像
  frameB_ = estimator_->multiFrame(mfIdB_);//得到对应的图像

  // focal length
  fA_ = frameA_->geometryAs<CAMERA_GEOMETRY_T>(camIdA_)->focalLengthU();
  fB_ = frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->focalLengthU();

  // calculate the relative transformations and uncertainties
  // TODO donno, if and what we need here - I'll see
  estimator_->getCameraSensorStates(mfIdA_, camIdA, T_SaCa_);//获得相机坐标系到imu坐标系的变换矩阵 这个值是固定死的
  estimator_->getCameraSensorStates(mfIdB_, camIdB, T_SbCb_);//获得相机坐标系到imu坐标系的变换矩阵 这个值是固定死的
  estimator_->get_T_WS(mfIdA_, T_WSa_);//得到A相机的imu在世界坐标系下的位姿
  estimator_->get_T_WS(mfIdB_, T_WSb_);//得到B相机的imu在世界坐标系下的位姿
  T_SaW_ = T_WSa_.inverse();
  T_SbW_ = T_WSb_.inverse();
  T_WCa_ = T_WSa_ * T_SaCa_;//得到A相机在世界坐标系下的位姿
  T_WCb_ = T_WSb_ * T_SbCb_;//得到B相机在世界坐标系下的位姿
  T_CaW_ = T_WCa_.inverse();
  T_CbW_ = T_WCb_.inverse();
  T_CaCb_ = T_WCa_.inverse() * T_WCb_;//得到两个相机之间的变换矩阵
  T_CbCa_ = T_CaCb_.inverse();

  validRelativeUncertainty_ = false;
}

// Set the matching type.
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setMatchingType(
    int matchingType) {
  matchingType_ = matchingType;
}

//详见算法实现文档
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
  } else //默认进入这个条件
  {
    UOplus.setIdentity();
    UOplus.bottomRightCorner<3, 3>() *= 1e-8;
    uint64_t currentId = estimator_->currentFrameId();//等价于 return statesMap_.rbegin()->first;
    //搜索 bool Estimator::isInImuWindow(
    //当前帧的速度和bias在优化窗口中
    //并且要匹配的两个图像不是来自于一个双目帧
    if (estimator_->isInImuWindow(currentId) && (mfIdA_ != mfIdB_))
	{
      okvis::SpeedAndBias speedAndBias;
      estimator_->getSpeedAndBias(currentId, 0, speedAndBias);//提取当前帧对应的速度和bias
      double scale = std::max(1.0, speedAndBias.head<3>().norm());
      UOplus.topLeftCorner<3, 3>() *= (scale * scale) * 1.0e-2;
    } else {
      UOplus.topLeftCorner<3, 3>() *= 4e-8;
    }
  }

  // now set the frames and uncertainty
  //搜索 void ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::resetFrames(
  probabilisticStereoTriangulator_.resetFrames(frameA_, frameB_, camIdA_,camIdB_, T_CaCb_, UOplus);//比较重要的函数!!!!!!!!!!!!!!!!详见算法实现文档

  // reset the match counter
  numMatches_ = 0;
  numUncertainMatches_ = 0;

  //1、先对A图像进行处理
  const size_t numA = frameA_->numKeypoints(camIdA_);//A帧拥有的特征点个数
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
    for (size_t k = 0; k < numA; ++k) //遍历A图像的特征点
	{
      uint64_t lm_id = frameA_->landmarkId(camIdA_, k);//得到特征点对应的地图点id
      //搜索 bool isLandmarkAdded(uint64_t landmarkId) const 
      //landmarksMap_没有对应的地图点
      if (lm_id == 0 || !estimator_->isLandmarkAdded(lm_id)) //表示这个特征点没有对应的地图点
	  {
        // this can happen, if you called the 2D-2D version just before,
        // without inserting the landmark into the graph
        skipA_[k] = true;
        continue;
      }

      okvis::MapPoint landmark;
      estimator_->getLandmark(lm_id, landmark);//得到地图点
      Eigen::Vector4d hp_W = landmark.point;

     
      //表示这个地图点没有被加入到ceres的参数块中
      if (!estimator_->isLandmarkInitialized(lm_id)) //搜索 bool Estimator::isLandmarkInitialized(
	  {
        skipA_[k] = true;
        continue;
      }

      // project (distorted)
      Eigen::Vector2d kptB;
      const Eigen::Vector4d hp_Cb = T_CbW_ * hp_W;//将空间点投影到B相机坐标系下
      //将在当前帧相机坐标系下的坐标点使用畸变模型投影到图像中 得到像素坐标kptB，执行这一步是判断投影是否成功
      if (frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->projectHomogeneous(hp_Cb, &kptB)!= okvis::cameras::CameraBase::ProjectionStatus::Successful) 
	  {
        skipA_[k] = true;
        continue;
      }

      if (landmark.observations.size() < 2) 
	  {
        estimator_->setLandmarkInitialized(lm_id, false);//搜索 void Estimator::setLandmarkInitialized(
        skipA_[k] = true;
        continue;
      }

      // project and get uncertainty
      Eigen::Matrix<double, 2, 4> jacobian;
      Eigen::Matrix4d P_C = Eigen::Matrix4d::Zero();
      P_C.topLeftCorner<3, 3>() = UOplus.topLeftCorner<3, 3>();  // get from before -- velocity scaled
      frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->projectHomogeneous(hp_Cb, &kptB, &jacobian);//将在当前帧相机坐标系下的坐标点使用畸变模型投影到图像中 得到像素坐标kptB，
      //更新VioKeyframeWindowMatchingAlgorithm类中的变量 projectionsIntoBUncertainties_
      projectionsIntoBUncertainties_.block<2, 2>(2 * k, 0) = jacobian * P_C * jacobian.transpose();
      //更新 VioKeyframeWindowMatchingAlgorithm类中的变量 projectionsIntoB_
	  projectionsIntoB_.row(k) = kptB;

      // precalculate ray uncertainties
      double keypointAStdDev;//该特征点的直径大小
      frameA_->getKeypointSize(camIdA_, k, keypointAStdDev);
      keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
	  //更新VioKeyframeWindowMatchingAlgorithm类中的变量 raySigmasA_
      raySigmasA_[k] = sqrt(sqrt(2)) * keypointAStdDev / fA_;  // (sqrt(MeasurementCovariance.norm()) / _fA)
    }
  } else {//如果匹配类型是2D-2D
    for (size_t k = 0; k < numA; ++k)//遍历关键帧或者相邻帧的特征点
	{
      double keypointAStdDev;
      frameA_->getKeypointSize(camIdA_, k, keypointAStdDev);//得到特征点的邻域直径
      keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
	  //更新VioKeyframeWindowMatchingAlgorithm类中的变量 raySigmasA_
      raySigmasA_[k] = sqrt(sqrt(2)) * keypointAStdDev / fA_;
      if (frameA_->landmarkId(camIdA_, k) == 0)//表示当前帧这个特征点没有地图点和其对应
	  {
        continue;
      }
	  //搜索 bool isLandmarkAdded(uint64_t landmarkId) const 
	  //从landmarksMap_中寻找是否有这个地图点
      if (estimator_->isLandmarkAdded(frameA_->landmarkId(camIdA_, k))) 
	  {
	    //搜索 bool Estimator::isLandmarkInitialized(uint64_t landmarkId) const 
	    //初始化表示是否已经将这个地图点对应的参数块压入到了ceres中
        if (estimator_->isLandmarkInitialized(frameA_->landmarkId(camIdA_, k)))
		{
		 //表示特征点存在地标点且地标点已经压入到了ceres参数块中，则匹配过程直接跳过该特征点
          skipA_[k] = true;
        }
      }
    }
  }

  //2.再对B图像进行处理
  const size_t numB = frameB_->numKeypoints(camIdB_);
  skipB_.clear();
  skipB_.reserve(numB);
  raySigmasB_.resize(numB);
  // do the projections for each keypoint, if applicable
  if (matchingType_ == Match3D2D)
  {
    for (size_t k = 0; k < numB; ++k)//遍历当前帧的特征点
	{
	      okvis::MapPoint landmark;
		  //如果B图像中存在特征点对应的地标点且landmarksMap_结构中有地图点
	      if (frameB_->landmarkId(camIdB_, k) != 0 && estimator_->isLandmarkAdded(frameB_->landmarkId(camIdB_, k)))//搜素 bool isLandmarkAdded(uint64_t landmarkId) const 
		  {
	        estimator_->getLandmark(frameB_->landmarkId(camIdB_, k), landmark);
			//如果下面的等式相等 则认为B图像中的这个特征点不用进行匹配了
			//如果这个地图点观测到了这个特征点
	        skipB_.push_back( landmark.observations.find(okvis::KeypointIdentifier(mfIdB_, camIdB_, k))!= landmark.observations.end());
	      } else {
	        skipB_.push_back(false);
	      }
	      double keypointBStdDev;
	      frameB_->getKeypointSize(camIdB_, k, keypointBStdDev);
	      keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
	      raySigmasB_[k] = sqrt(sqrt(2)) * keypointBStdDev / fB_;matchBody
    }
  } else //如果匹配类型是2D-2D
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
		  //如果在landmarksMap_结构中有地图点被这个帧观测到了并且这个地图点已经压入到了ceres的参数块中，则认为可以跳过。否则认为不能跳过。
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
///用几何模型对匹配点对进行筛选
template<class CAMERA_GEOMETRY_T>
bool VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::verifyMatch(size_t indexA, size_t indexB) const 
{

  if (matchingType_ == Match2D2D) 
  {

    // potential 2d2d match - verify by triangulation
    Eigen::Vector4d hP;
    bool isParallel;
	//搜素 bool ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::stereoTriangulate(
	//重要的函数!!!!!!!!!!!!!!!!!!!!!
	//hP, isParallel是输出变量
    bool valid = probabilisticStereoTriangulator_.stereoTriangulate( indexA, indexB, hP, isParallel, std::max(raySigmasA_[indexA], raySigmasB_[indexB]));
    if (valid) {
      return true;
    }
  } else //3D-2D的情况
 {
    // get projection into B
    Eigen::Vector2d kptB = projectionsIntoB_.row(indexA);

    // uncertainty
    double keypointBStdDev;
    frameB_->getKeypointSize(camIdB_, indexB, keypointBStdDev);
    keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
	////U表示B帧中特征点的协方差+由于速度引起的投影误差（B特征点的一个信息矩阵）
    Eigen::Matrix2d U = Eigen::Matrix2d::Identity() * keypointBStdDev* keypointBStdDev+ projectionsIntoBUncertainties_.block<2, 2>(2 * indexA, 0);

    Eigen::Vector2d keypointBMeasurement;
    frameB_->getKeypoint(camIdB_, indexB, keypointBMeasurement);
    Eigen::Vector2d err = kptB - keypointBMeasurement;//衡量A帧中特征点到B帧投影与B帧中原始点的偏差
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
//一定要注意了 整个代码范围内只有在这个函数中addObservation被调用
//而 addObservation是将重投影误差函数加入到ceres中
//而 setBestMatch函数只在matchBody中被调用过
//主要的作用是向ceres添加残差函数，向ceres中添加地图点的参数块，更改ceres中参数块的读数
//详见算法实现文档
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setBestMatch(size_t indexA, size_t indexB, double /*distance*/) 
{

  // assign correspondences
  uint64_t lmIdA = frameA_->landmarkId(camIdA_, indexA);//A图像特征点对应的地图点序号
  uint64_t lmIdB = frameB_->landmarkId(camIdB_, indexB);//B图像特征点对应的地图点序号

  if (matchingType_ == Match2D2D)
  {

    // check that not both are set
    // 表示两个特征点都已经了初始化,不需要再进行匹配
    if (lmIdA != 0 && lmIdB != 0) {
      return;
    }

    // re-triangulate...
    // potential 2d2d match - verify by triangulation
    Eigen::Vector4d hP_Ca;
    bool canBeInitialized;

    //输出值 canBeInitialized=false 表示两个射线平行
    //输出值是 hP_Ca = 相机A坐标系下的三维点 是单位化的齐次坐标
    //搜索 bool ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::stereoTriangulate(
    //重要的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    bool valid = probabilisticStereoTriangulator_.stereoTriangulate(  indexA, indexB, hP_Ca, canBeInitialized, std::max(raySigmasA_[indexA], raySigmasB_[indexB]));
    if (!valid) 
	{
      return;
    }

    // get the uncertainty
    //搜索 void ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::getUncertainty(
    //输出值是 pointUOplus_A, canBeInitialized
    //重要的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (canBeInitialized) //进入这个条件表示两个射线不平行
	{  // know more exactly
      Eigen::Matrix3d pointUOplus_A;//一定注意这是个局部变量 后面没有用到
	  //搜索 void ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::getUncertainty(
	  //输出值pointUOplus_A = 
	  //整个代码中就这里被调用了
      probabilisticStereoTriangulator_.getUncertainty(indexA, indexB, hP_Ca, pointUOplus_A, canBeInitialized);
    }

    // check and adapt landmark status
    bool insertA = lmIdA == 0;
    bool insertB = lmIdB == 0;
    bool insertHomogeneousPointParameterBlock = false;
    uint64_t lmId = 0;  // 0 just to avoid warning
    //左右两个特征点都没有地图点和其对应
    if (insertA && insertB) 
	{
      // ok, we need to assign a new Id...
      lmId = okvis::IdProvider::instance().newId();//新建一个地图点id
      //更新两个相机中的信息，表示这两个帧都看到了这个地图点
      frameA_->setLandmarkId(camIdA_, indexA, lmId);
      frameB_->setLandmarkId(camIdB_, indexB, lmId);
      lmIdA = lmId;
      lmIdB = lmId;
      // and add it to the graph
      insertHomogeneousPointParameterBlock = true;
    } else {//两个点都有一个已经进行了初始化
       //表示A帧特征点已经有对应的地标点
      if (!insertA) 
	  {
        lmId = lmIdA;
		//搜索  bool isLandmarkAdded(uint64_t landmarkId) const 
        if (!estimator_->isLandmarkAdded(lmId)) //landmarksMap_状态空间中如果未添加A帧特征点对应的地图点
		{
          // add landmark and observation to the graph
          insertHomogeneousPointParameterBlock = true;
          insertA = true;
        }
      }
      if (!insertB) //表示B帧特征点已经有对应的地标点
	  {
        lmId = lmIdB;
        if (!estimator_->isLandmarkAdded(lmId))//landmarksMap_状态空间中如果未添加B帧特征点对应的地图点
		{
          // add landmark and observation to the graph
          insertHomogeneousPointParameterBlock = true;
          insertB = true;
        }
      }
    }
	
    // add landmark to graph if necessary
    //如果这个条件成立则向参数块插入新的地图点的参数块
    if (insertHomogeneousPointParameterBlock) 
	{
	  //addLandmark函数作用: 1.向ceres中添加这个地图点参数块.2.更新landmarksMap_结构中的信息即新加入一个地图点
	  //搜索 bool Estimator::addLandmark(
      estimator_->addLandmark(lmId, T_WCa_ * hP_Ca);
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmId), lmId<<" not added, bug");
	  //搜索 void Estimator::setLandmarkInitialized(
	  //设置这个参数化的数据 已经被初始化过了
      estimator_->setLandmarkInitialized(lmId, canBeInitialized);
    } else 
    { //如果A或B图像中的特征点有地图点对应并且这个地图点都已经存在于landmarksMap_状态空间中了
          //我们不向ceres插入参数块了，我们修改它的值。
      // update initialization status, set better estimate, if possible
      if (canBeInitialized) 
	  {
	    //搜索 void Estimator::setLandmarkInitialized(uint64_t landmarkId,bool initialized) 
        estimator_->setLandmarkInitialized(lmId, true);
		//搜索 bool Estimator::setLandmark(uint64_t landmarkId, const Eigen::Vector4d & landmark)
		//1.从mapPtr_中提取出地图点对应的参数块指针，并修改其值
		//2.修改landmarksMap_中地图点的值
        estimator_->setLandmark(lmId, T_WCa_ * hP_Ca);
      }
    }

    // in image A
    okvis::MapPoint landmark;
	//A帧的特征点还没有对应的地标点，且地图点在A帧中没有对应特征点的观测
    if (insertA  && landmark.observations.find(okvis::KeypointIdentifier(mfIdA_, camIdA_, indexA))== landmark.observations.end()) 
	{  // ensure no double observations...
            // TODO hp_Sa NOT USED!
      Eigen::Vector4d hp_Sa(T_SaCa_ * hP_Ca);//转换到A相机的IMU坐标系中
      hp_Sa.normalize();//没有用到这个变量
      frameA_->setLandmarkId(camIdA_, indexA, lmId);//设置相机A看到了这个地图点
      lmIdA = lmId;//这个更新后面也没有用到lmIdA这个变量呀
      // initialize in graph
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmId),
                        "landmark id=" << lmId<<" not added");
	  //lmId是这个地图点对应的id mfIdA_是A图像的双目帧结构的id。camIdA_是A图像对应的那个双目帧的左图像还是右图像
	  //indexA = 特征点在A图像中的序号
	  //addObservation函数作用: 向ceres中添加BA的残差方程，更新landmarksMap_的信息,让地图点看到这个特征点
	  //搜索 ::ceres::ResidualBlockId Estimator::addObservation(
      estimator_->addObservation<camera_geometry_t>(lmId, mfIdA_, camIdA_,indexA);
    }

    // in image B
    ///B帧中该特征点还没有对应地标点，且该地标点在B帧中没有对应特征点的观测
    if (insertB&& landmark.observations.find(okvis::KeypointIdentifier(mfIdB_, camIdB_, indexB))== landmark.observations.end()) 
	{  // ensure no double observations...
      Eigen::Vector4d hp_Sb(T_SbCb_ * T_CbCa_ * hP_Ca);
      hp_Sb.normalize();//没有用到这个变量
      frameB_->setLandmarkId(camIdB_, indexB, lmId);
      lmIdB = lmId;//这个更新后面也没有用到lmIdA这个变量呀
      // initialize in graph
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmId),"landmark " << lmId << " not added");
	  //addObservation函数作用: 向ceres中添加BA的残差方程，更新landmarksMap_的信息，让地图点看到这个特征点
      estimator_->addObservation<camera_geometry_t>(lmId, mfIdB_, camIdB_,indexB);
    }

    // let's check for consistency with other observations:
    okvis::ceres::HomogeneousPointParameterBlock point(T_WCa_ * hP_Ca, 0);
    if(canBeInitialized)
    {
      //从mapPtr_中提取出地图点对应的参数块指针，并修改其值
      //修改landmarksMap_中地图点的值
      //搜索 bool Estimator::setLandmark(uint64_t landmarkId, const Eigen::Vector4d & landmark)
      estimator_->setLandmark(lmId, point.estimate());
    }

  } else //如果是3d-2d匹配
 {
    OKVIS_ASSERT_TRUE_DBG(Exception,lmIdB==0,"bug. Id in frame B already set.");

    // get projection into B
    Eigen::Vector2d kptB = projectionsIntoB_.row(indexA);//与A中特征点匹配的B图像中的坐标
    Eigen::Vector2d keypointBMeasurement;
    frameB_->getKeypoint(camIdB_, indexB, keypointBMeasurement);//B帧中特征点的观测值（真实值）

    Eigen::Vector2d err = kptB - keypointBMeasurement;
    double keypointBStdDev;
    frameB_->getKeypointSize(camIdB_, indexB, keypointBStdDev);//得到B帧中特征点的尺寸
    keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
    Eigen::Matrix2d U_tot = Eigen::Matrix2d::Identity() * keypointBStdDev * keypointBStdDev + projectionsIntoBUncertainties_.block<2, 2>(2 * indexA, 0);

    const double chi2 = err.transpose().eval() * U_tot.inverse() * err;

    if (chi2 > 4.0) {
      return;
    }

    // saturate allowed image uncertainty
     // 协方差矩阵的模值过大，则认为不确定
    if (U_tot.norm() > 25.0 / (keypointBStdDev * keypointBStdDev * sqrt(2))) 
	{
      numUncertainMatches_++;
      //return;
    }
    //搜索 bool Frame::setLandmarkId(size_t keypointIdx, uint64_t landmarkId)
    frameB_->setLandmarkId(camIdB_, indexB, lmIdA);//B帧看到了这个地图点
    lmIdB = lmIdA;
    okvis::MapPoint landmark;
    estimator_->getLandmark(lmIdA, landmark);//得到地标点的信息landmark

    // initialize in graph
    //进入这个条件表示之前这个地图点没有被B帧观测到
    if (landmark.observations.find(okvis::KeypointIdentifier(mfIdB_, camIdB_, indexB))== landmark.observations.end()) 
	{  // ensure no double observations...
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmIdB), "not added");
	  //搜索 ::ceres::ResidualBlockId Estimator::addObservation(
	  //输入的参数分别是 地图点id 双目帧id 左目/右目id 特征点在这幅图像上的id
	  //addObservation函数作用: 向ceres中添加BA的残差方程，更新landmarksMap_的信息即让地图点看到这个特征点
      estimator_->addObservation<camera_geometry_t>(lmIdB, mfIdB_, camIdB_,indexB);//对该地标点添加B帧第indexB个特征点的观测
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
