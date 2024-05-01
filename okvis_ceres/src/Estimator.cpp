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
 *  Created on: Dec 30, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file Estimator.cpp
 * @brief Source file for the Estimator class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>
#include <okvis/Estimator.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor if a ceres map is already available.
Estimator::Estimator(
    std::shared_ptr<okvis::ceres::Map> mapPtr)
    : mapPtr_(mapPtr),
      referencePoseId_(0),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0)
{
}

// The default constructor.
Estimator::Estimator()
    : mapPtr_(new okvis::ceres::Map()),
      referencePoseId_(0),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0)
{
}

Estimator::~Estimator()
{
}

// Add a camera to the configuration. Sensors can only be added and never removed.
int Estimator::addCamera(const ExtrinsicsEstimationParameters & extrinsicsEstimationParameters)
{
  extrinsicsEstimationParametersVec_.push_back(extrinsicsEstimationParameters);
  return extrinsicsEstimationParametersVec_.size() - 1;
}

// Add an IMU to the configuration.
int Estimator::addImu(const ImuParameters & imuParameters)
{
  if(imuParametersVec_.size()>1){
    LOG(ERROR) << "only one IMU currently supported";
    return -1;
  }
  imuParametersVec_.push_back(imuParameters);
  return imuParametersVec_.size() - 1;
}

// Remove all cameras from the configuration
void Estimator::clearCameras(){
  extrinsicsEstimationParametersVec_.clear();
}

// Remove all IMUs from the configuration.
void Estimator::clearImus(){
  imuParametersVec_.clear();
}

// Add a pose to the state.
/*
一、添加参数块：
1、获得当前帧后Tws将其加入到ceres的参数块中
构造参数块：std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock( new okvis::ceres::PoseParameterBlock(T_WS, states.id,multiFrame->timestamp()));
加入到ceres中：mapPtr_->addParameterBlock(poseParameterBlock,ceres::Map::Pose6d))
2、将camera到imu的姿态加入到ceres的参数快中(在优化时设置成了常量不进行优化)
构造参数块：std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlockPtr(new okvis::ceres::PoseParameterBlock(T_SC, id,multiFrame->timestamp()));
加入到ceres中：mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr,ceres::Map::Pose6d))
3、得到当前帧的速度和bias，将这个状态加入到ceres的参数中
构造参数块：std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock(new okvis::ceres::SpeedAndBiasParameterBlock(speedAndBias, id, multiFrame->timestamp()));
加入到ceres中：mapPtr_->addParameterBlock(speedAndBiasParameterBlock))

二、构造误差函数：
1.1初始时：
构造误差函数：std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS, information));
误差函数加入到ceres中：mapPtr_->addResidualBlock(poseError,NULL,poseParameterBlock);

构造误差函数：std::shared_ptr<ceres::SpeedAndBiasError > speedAndBiasError(  new ceres::SpeedAndBiasError(speedAndBias, 1.0, sigma_bg*sigma_bg, sigma_ba*sigma_ba));
误差函数加入到ceres中：mapPtr_->addResidualBlock(  speedAndBiasError,
											    	NULL,
													mapPtr_->parameterBlockPtr(
													states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
1.2非初始时：
构造误差函数：std::shared_ptr<ceres::ImuError> imuError( new ceres::ImuError(imuMeasurements, imuParametersVec_.at(i),
                            					 lastElementIterator->second.timestamp,
                            					 states.timestamp));
误差函数加入到ceres中：mapPtr_->addResidualBlock(
          imuError,
          NULL,
          mapPtr_->parameterBlockPtr(lastElementIterator->second.id),//搜索 std::shared_ptr<okvis::ceres::ParameterBlock> Map::parameterBlockPtr(
          mapPtr_->parameterBlockPtr(lastElementIterator->second.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id),
          mapPtr_->parameterBlockPtr(states.id), 
          mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
*/
//asKeyframe插入的这一帧是否为关键帧,常年输入的都是false
//这个函数是在matchLoop被调用过
bool Estimator::addStates(okvis::MultiFramePtr multiFrame,const okvis::ImuMeasurementDeque & imuMeasurements, bool asKeyframe)
{
  // note: this is before matching...
  // TODO !!
  //1.首先根据imu的测量值得到当前帧的位姿T_WS
  okvis::kinematics::Transformation T_WS;
  okvis::SpeedAndBias speedAndBias;
  if (statesMap_.empty())//初始化状态
  {
    // in case this is the first frame ever, let's initialize the pose:
    bool success0 = initPoseFromImu(imuMeasurements, T_WS);
    OKVIS_ASSERT_TRUE_DBG(Exception, success0,
        "pose could not be initialized from imu measurements.");
    if (!success0)
      return false;
    speedAndBias.setZero();//将第一帧相机的速度和bias设置为0
    speedAndBias.segment<3>(6) = imuParametersVec_.at(0).a0;
  } else 
  {
    // get the previous states1.首先获得上一个时刻的位姿和速度与bias
    //上一帧的位姿的参数块id
    uint64_t T_WS_id = statesMap_.rbegin()->second.id;//rbegin指向容器的最后一个元素
    //搜索  enum ImuSensorStates
    //上一帧的速度和bias的参数块id
    uint64_t speedAndBias_id = statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).at(0).at(ImuSensorStates::SpeedAndBias).id;
    OKVIS_ASSERT_TRUE_DBG(Exception, mapPtr_->parameterBlockExists(T_WS_id),
                       "this is an okvis bug. previous pose does not exist.");
	 //static_pointer_cast指针转换函数，将ParameterBlock转换为PoseParameterBlock，这样estimate（）输出的就是位姿信息
	 //从参数块中提取出数据
    T_WS = std::static_pointer_cast<ceres::PoseParameterBlock>(mapPtr_->parameterBlockPtr(T_WS_id))->estimate();//搜索 std::shared_ptr<okvis::ceres::ParameterBlock> Map::parameterBlockPtr(
    //OKVIS_ASSERT_TRUE_DBG(
    //    Exception, speedAndBias_id,
    //    "this is an okvis bug. previous speedAndBias does not exist.");
    speedAndBias = std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>( mapPtr_->parameterBlockPtr(speedAndBias_id))->estimate();

    // propagate pose and speedAndBias
    //1.2.根据上一时刻的状态和imu的测量值得到这一时刻的状态。没有计算协方差和雅克比
    int numUsedImuMeasurements = ceres::ImuError::propagation(imuMeasurements, imuParametersVec_.at(0), T_WS, speedAndBias,statesMap_.rbegin()->second.timestamp, multiFrame->timestamp());
    OKVIS_ASSERT_TRUE_DBG(Exception, numUsedImuMeasurements > 1,
                       "propagation failed");
    if (numUsedImuMeasurements < 1){
      LOG(INFO) << "numUsedImuMeasurements=" << numUsedImuMeasurements;
      return false;
    }
  }


  // create a states object:
  //2.根据当前帧的位姿 构建一个states状态，并更新这个状态的一些信息，并将当前帧的姿态Tws加入ceres的参数块中addParameterBlock。
  //状态的id就是双目帧的id
  States states(asKeyframe, multiFrame->id(), multiFrame->timestamp());

  // check if id was used before
  OKVIS_ASSERT_TRUE_DBG(Exception,statesMap_.find(states.id)==statesMap_.end(),"pose ID" <<states.id<<" was used before!");

  // create global states
  //搜索 PoseParameterBlock::PoseParameterBlock( const 
  //构造位姿参数块
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock( new okvis::ceres::PoseParameterBlock(T_WS, states.id,multiFrame->timestamp()));
  // 状态中的global是一个6维的数组，每一维元素都是一个stateinfo
  states.global.at(GlobalStates::T_WS).exists = true;
  states.global.at(GlobalStates::T_WS).id = states.id;//搜索 enum GlobalStates

  if(statesMap_.empty())
  {
    referencePoseId_ = states.id; // set this as reference pose
    if (!mapPtr_->addParameterBlock(poseParameterBlock,ceres::Map::Pose6d)) //搜索 bool Map::addParameterBlock( 
	{
      return false;
    }
  } else 
  {
    //向ceres中加入当前imu的位姿的参数块!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (!mapPtr_->addParameterBlock(poseParameterBlock,ceres::Map::Pose6d)) //搜索 bool Map::addParameterBlock(
	{
      return false;
    }
  }

  // add to buffer
  //将states加入到statesMap_，更新statesMap_，将camera到imu的姿态加入到ceres的参数快中，将速度和bias加入到ceres的参数块中。
  //如果是第一帧图像则将相机坐标系到imu坐标系的变换加入到ceres参数块中
  statesMap_.insert(std::pair<uint64_t, States>(states.id, states));//将状态加入状态空间，整个代码中就这里向statesMap_中添加数据了
  multiFramePtrMap_.insert(std::pair<uint64_t, okvis::MultiFramePtr>(states.id, multiFrame));//将帧加入帧类地图，整个代码中就这里向multiFramePtrMap_插入值

  // the following will point to the last states:
  std::map<uint64_t, States>::reverse_iterator lastElementIterator = statesMap_.rbegin();
  lastElementIterator++;//迭代器前移一位,指向前一个状态,即新加入状态的上一个状态

  // initialize new sensor states
  // cameras:
  //extrinsicsEstimationParametersVec_中储存了左右两个相机的外参
  for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) 
  {

    SpecificSensorStatesContainer cameraInfos(2);
    cameraInfos.at(CameraSensorStates::T_SCi).exists=true;//搜索 enum CameraSensorStates
    cameraInfos.at(CameraSensorStates::Intrinsics).exists=false;
	//第i个相机的相对位移和相对旋转的标准差小于阈值
    if(((extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation<1e-12)||(extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation<1e-12))&& (statesMap_.size() > 1))
	{//永远进入这个条件
      // use the same block...
      //将状态地图倒数第二个状态的相机传感器数组中第i个相机的T_SCi位姿对应元素的id 赋值给相机信息容器
      cameraInfos.at(CameraSensorStates::T_SCi).id = lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id;
    } else 
    {//初始化时进入这个条件
      const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlockPtr(new okvis::ceres::PoseParameterBlock(T_SC, id,multiFrame->timestamp()));
      if(!mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr,ceres::Map::Pose6d))//向地图中添加位姿参数块!!!!!!!!!!!!!!!!!!!!!!
	  {
        return false;
      }
      cameraInfos.at(CameraSensorStates::T_SCi).id = id;
    }
    // update the states info
    //将相机信息容器添加到新添加状态的相机传感器容器中
    statesMap_.rbegin()->second.sensors.at(SensorStates::Camera).push_back(cameraInfos);
    states.sensors.at(SensorStates::Camera).push_back(cameraInfos);
  }

  // IMU states are automatically propagated.
  // IMU状态的更新 imuParametersVec_存储的是imu的信息，如果你只使用一个imu只循环一次
  for (size_t i=0; i<imuParametersVec_.size(); ++i)
  {
    SpecificSensorStatesContainer imuInfo(2);
    imuInfo.at(ImuSensorStates::SpeedAndBias).exists = true;
    uint64_t id = IdProvider::instance().newId();
	//非常重要的函数 ，构造速度和bias的参数块
	//搜索 SpeedAndBiasParameterBlock::SpeedAndBiasParameterBlock(
    std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock(new okvis::ceres::SpeedAndBiasParameterBlock(speedAndBias, id, multiFrame->timestamp()));

    if(!mapPtr_->addParameterBlock(speedAndBiasParameterBlock))//向ceres中添加当前帧的速度和bias参数块!!!!!!!!!!!!!!!!!!!!!
	{
      return false;
    }
    imuInfo.at(ImuSensorStates::SpeedAndBias).id = id;
    statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).push_back(imuInfo);
    states.sensors.at(SensorStates::Imu).push_back(imuInfo);
  }


  //3.构造误差函数
  // depending on whether or not this is the very beginning, we will add priors or relative terms to the last state:
  // //表示在最开始,刚刚只添加了一维状态
  if (statesMap_.size() == 1) 
  {
    // let's add a prior
    Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Zero();
    information(5,5) = 1.0e8; information(0,0) = 1.0e8; information(1,1) = 1.0e8; information(2,2) = 1.0e8;
    //位姿的误差函数还在applyMarginalization函数中被调用了 搜索 std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS_0, information));
    std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS, information));//非常重要位姿的误差函数!!!!!!!!!!!，这里输入的当前帧的姿态竟然是测量值
    /*auto id2= */ mapPtr_->addResidualBlock(poseError,NULL,poseParameterBlock);//将姿态的误差函数加入到ceres中，搜索 ::ceres::ResidualBlockId Map::addResidualBlock(
    //mapPtr_->isJacobianCorrect(id2,1.0e-6);

    // sensor states对于双目相机 就只轮询两遍
    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) 
	{
      double translationStdev = extrinsicsEstimationParametersVec_.at(i).sigma_absolute_translation;
      double translationVariance = translationStdev*translationStdev;
      double rotationStdev = extrinsicsEstimationParametersVec_.at(i).sigma_absolute_orientation;
      double rotationVariance = rotationStdev*rotationStdev;
      if(translationVariance>1.0e-16 && rotationVariance>1.0e-16)
	  {
        const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
        std::shared_ptr<ceres::PoseError > cameraPoseError( new ceres::PoseError(T_SC, translationVariance, rotationVariance));
        // add to map
        mapPtr_->addResidualBlock(
            cameraPoseError,
            NULL,
            mapPtr_->parameterBlockPtr(
                states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id));
        //mapPtr_->isJacobianCorrect(id,1.0e-6);
      }
      else //默认进入这个条件
	  {
	  //搜索 bool Map::setParameterBlockConstant(
        mapPtr_->setParameterBlockConstant(states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id);
      }
    }
    for (size_t i = 0; i < imuParametersVec_.size(); ++i)//只轮询一遍
	{
      Eigen::Matrix<double,6,1> variances;
      // get these from parameter file
      const double sigma_bg = imuParametersVec_.at(0).sigma_bg;
      const double sigma_ba = imuParametersVec_.at(0).sigma_ba;
	  //非常重要的函数，速度和bias的误差函数
	  //1表示的是速度的协方差
	  //整个代码就这里调用了速度和bias误差函数
      std::shared_ptr<ceres::SpeedAndBiasError > speedAndBiasError(  new ceres::SpeedAndBiasError(speedAndBias, 1.0, sigma_bg*sigma_bg, sigma_ba*sigma_ba));
      // add to map 将速度和bias的cost function 加入到ceres中
      mapPtr_->addResidualBlock(
          speedAndBiasError,
          NULL,
          mapPtr_->parameterBlockPtr(
              states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
      //mapPtr_->isJacobianCorrect(id,1.0e-6);
    }
  }
  else{//不是初始化
    // add IMU error terms
      // 添加IMU的误差
    for (size_t i = 0; i < imuParametersVec_.size(); ++i)//只遍历一次对于一个imu来说
	{
	  //这个是实现imu残差的主要函数!!!!!!!!
      std::shared_ptr<ceres::ImuError> imuError( new ceres::ImuError(imuMeasurements, imuParametersVec_.at(i),
                            					 lastElementIterator->second.timestamp,
                            					 states.timestamp));
	  /// IMU的残差块
      /// NULL
      /// 地图中倒数第二个状态的(位姿)ID---->数据块
      /// 地图中倒数第二个状态的第i个IMU传感器的速度与偏置部分对应的ID----->数据块
      /// 地图中最新一个状态的(位姿)ID----->数据块
      /// 地图中最新一个状态的第i个IMU传感器的速度与偏置部分对应的ID------->数据块
      //搜索 ::ceres::ResidualBlockId Map::addResidualBlock(
      /*::ceres::ResidualBlockId id = */mapPtr_->addResidualBlock(
          imuError,
          NULL,
          mapPtr_->parameterBlockPtr(lastElementIterator->second.id),//搜索 std::shared_ptr<okvis::ceres::ParameterBlock> Map::parameterBlockPtr(
          mapPtr_->parameterBlockPtr(lastElementIterator->second.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id),
          mapPtr_->parameterBlockPtr(states.id), 
          mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
      //imuError->setRecomputeInformation(false);
      //mapPtr_->isJacobianCorrect(id,1.0e-9);
      //imuError->setRecomputeInformation(true);
    }

    // add relative sensor state errors
    //这个循环在euroc的配置情况下是不会进入的
    //进入这个条件表示需要优化相机坐标系到imu坐标系的位姿变化
    //euroc设置是默认这个参数不变的
    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) 
	{
	 //首先需要判断上一帧状态的外参ID和当前帧状态的外参ID是否相等,相等即表示相机的外参未进行更新
	 //如果不相等则进入这个条件
      if(lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id != states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id)
	  {
        // i.e. they are different estimated variables, so link them with a temporal error term
        double dt = (states.timestamp - lastElementIterator->second.timestamp).toSec();
        double translationSigmaC = extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation;
        double translationVariance = translationSigmaC * translationSigmaC * dt;
        double rotationSigmaC = extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation;
        double rotationVariance = rotationSigmaC * rotationSigmaC * dt;
		//这是重要的相对的外参误差!!!!!!!!!!!!!!!!!!
        std::shared_ptr<ceres::RelativePoseError> relativeExtrinsicsError( new ceres::RelativePoseError(translationVariance, rotationVariance));
		//地图中添加相对外参误差误差块
		///形参
		/// 相对外参误差
		/// NULL
		/// 上一帧第i个相机对应T_SC的序号---->数据块
		/// 当前帧第i个相机对应T_SC的序号---->数据块
		//搜索 std::shared_ptr<okvis::ceres::ParameterBlock> Map::parameterBlockPtr(
		mapPtr_->addResidualBlock(
            relativeExtrinsicsError,
            NULL,
            mapPtr_->parameterBlockPtr(lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id),
            mapPtr_->parameterBlockPtr( states.sensors.at(SensorStates::Camera).at(i).at( CameraSensorStates::T_SCi).id));
        //mapPtr_->isJacobianCorrect(id,1.0e-6);
      }
    }
    // only camera. this is slightly inconsistent, since the IMU error term contains both
    // a term for global states as well as for the sensor-internal ones (i.e. biases).
    // TODO: magnetometer, pressure, ...
  }

  return true;
}//addStates函数结束----------------------------------------------------------------------

// Add a landmark.
//向ceres中添加这个地图点参数块
//更新landmarksMap_结构中的信息，新加入一个地图点
bool Estimator::addLandmark(uint64_t landmarkId,const Eigen::Vector4d & landmark) 
{
  //新建一个地图点的参数块，这个参数块的id就是地图点的id
  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> pointParameterBlock( new okvis::ceres::HomogeneousPointParameterBlock(landmark, landmarkId));

  //向ceres中添加参数块
  //搜索 bool Map::addParameterBlock( 
  //只有状态空间中已经有了这个地图点时才会返回false
  if (!mapPtr_->addParameterBlock(pointParameterBlock,okvis::ceres::Map::HomogeneousPoint)) 
  {
    return false;
  }

  // remember
  double dist = std::numeric_limits<double>::max();
  if(fabs(landmark[3])>1.0e-8)
  {
    dist = (landmark/landmark[3]).head<3>().norm(); // euclidean distance
  }
  //向存储所有地图点的结构landmarksMap_中，插入这个新的地图点
  landmarksMap_.insert( std::pair<uint64_t, MapPoint>(landmarkId, MapPoint(landmarkId, landmark, 0.0, dist)));
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "bug: inconsistend landmarkdMap_ with mapPtr_.");
  return true;
}

// Remove an observation from a landmark.
//更新mapPtr_中的地图数据
//从ceres中删除这个残差块
bool Estimator::removeObservation(::ceres::ResidualBlockId residualBlockId) 
{
  const ceres::Map::ParameterBlockCollection parameters = mapPtr_->parameters(residualBlockId);
  const uint64_t landmarkId = parameters.at(1).first;
  // remove in landmarksMap
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);//元素=(地图点id,地图点数据类型)
  for(std::map<okvis::KeypointIdentifier, uint64_t >::iterator it = mapPoint.observations.begin();it!= mapPoint.observations.end(); )
  {
    if(it->second == uint64_t(residualBlockId))
	{
      it = mapPoint.observations.erase(it);
    } else {
      it++;
    }
  }
  // remove residual block
  //从ceres中删除这个残差块
  //搜索 bool Map::removeResidualBlock(::ceres::ResidualBlockId residualBlockId) 
  mapPtr_->removeResidualBlock(residualBlockId);
  return true;
}

// Remove an observation from a landmark, if available.
//1.更新landmarksMap_数据，这个地图点没有观测到过这个特征点
//2.从ceres中删除这个残差块，并更新相关变量
bool Estimator::removeObservation(uint64_t landmarkId, uint64_t poseId,size_t camIdx, size_t keypointIdx) 
{
  if(landmarksMap_.find(landmarkId) == landmarksMap_.end())
  {
    for (PointMap::iterator it = landmarksMap_.begin(); it!= landmarksMap_.end(); ++it) {
      LOG(INFO) << it->first<<", no. obs = "<<it->second.observations.size();
    }
    LOG(INFO) << landmarksMap_.at(landmarkId).id;
  }
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),
                     "landmark not added");

  okvis::KeypointIdentifier kid(poseId,camIdx,keypointIdx);//构建特征点索引号
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  std::map<okvis::KeypointIdentifier, uint64_t >::iterator it = mapPoint.observations.find(kid);
  if(it == landmarksMap_.at(landmarkId).observations.end())
  {
    return false; // observation not present
  }

  // remove residual block
  //reinterpret_cast是强制类型转换符
  //ResidualBlockId的用处 当我们使用ceres插入AddResidualBlock时，会在ceres内部对每一个插入的残差有一个id而这个id的类型就是ResidualBlockId
  mapPtr_->removeResidualBlock(reinterpret_cast< ::ceres::ResidualBlockId>(it->second));//搜索 bool Map::removeResidualBlock(::ceres::ResidualBlockId residualBlockId) 

  // remove also in local map
  mapPoint.observations.erase(it);

  return true;
}

/**
 * @brief Does a vector contain a certain element.
 * @tparam Class of a vector element.
 * @param vector Vector to search element in.
 * @param query Element to search for.
 * @return True if query is an element of vector.
 */
template<class T>
bool vectorContains(const std::vector<T> & vector, const T & query)
{
  for(size_t i=0; i<vector.size(); ++i)
  {
    if(vector[i] == query)
	{
      return true;
    }
  }
  return false;
}

// Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
// The new number of frames in the window will be numKeyframes+numImuFrames.
/**
 * @brief Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
 *		  The new number of frames in the window will be numKeyframes+numImuFrames.
 * @param numKeyframes Number of keyframes.
 * @param numImuFrames Number of frames in IMU window.
 * @param removedLandmarks Get the landmarks that were removed by this operation.-这其实是一个输出值
 * @return True if successful.
 */
 //euroc设置 numKeyframes=5  numImuFrames=3
 //这是整个okvis中最最重要的函数
bool Estimator::applyMarginalizationStrategy(size_t numKeyframes, size_t numImuFrames,okvis::MapPointVector& removedLandmarks)
{
  // keep the newest numImuFrames
  //保证状态空间中的状态数目大于numImuFrames=3，否则直接返回
  //map是新加入的元素为最后的元素
  //1. 如果imu窗口中载具姿态数量小于三则直接返回
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();//它指向容器c的最后一个元素
  for(size_t k=0; k<numImuFrames; k++)
  {
    rit++;
    if(rit==statesMap_.rend())//它指向容器c的第一个元素前面的位置
	{
      // nothing to do.
      return true;
    }
  }

  // remove linear marginalizationError, if existing
  // 移除残差块，如果需要的话
  //2.从ceres中删除边缘化残差块，并更新相关变量
  if (marginalizationErrorPtr_ && marginalizationResidualId_) 
  {
    //搜索 Map::removeResidualBlock(::ceres::ResidualBlockId residualBlockId) 
    //输入的marginalizationResidualId_是需要被边缘化的参数块id
    //a.从ceres中删除这个残差块
    //b.从id2ResidualBlock_Multimap_删除残差块
    //c.更新residualBlockId2ParameterBlockCollection_Map_
    //d.更新residualBlockId2ResidualBlockSpec_Map_
    bool success = mapPtr_->removeResidualBlock(marginalizationResidualId_);//marginalizationResidualId_=边缘化残差块的ceres id
    OKVIS_ASSERT_TRUE_DBG(Exception, success,"could not remove marginalization error");
    marginalizationResidualId_ = 0;
    if (!success)
      return false;
  }

  // these will keep track of what we want to marginalize out.
  std::vector<uint64_t> paremeterBlocksToBeMarginalized;//为需要边缘化的数据块id的集合
  std::vector<bool> keepParameterBlocks;//维数与paremeterBlocksToBeMarginalized相等的容器（false）

  if (!marginalizationErrorPtr_) //应该是初始化时才会进入一次
  {
    //搜索 MarginalizationError::MarginalizationError
    //*mapPtr_.get()返回值类型okvis::ceres::Map
    marginalizationErrorPtr_.reset( new ceres::MarginalizationError(*mapPtr_.get()));//表示对指针赋值
  }

  // distinguish if we marginalize everything or everything but pose
  std::vector<uint64_t> removeFrames;
  std::vector<uint64_t> removeAllButPose;
  std::vector<uint64_t> allLinearizedFrames;
  size_t countedKeyframes = 0;
  //3.获取要被剔除的状态帧
  ///removeAllButPose表示窗口（3帧）之前的所有未剔除的状态（帧），这些帧（包括关键帧）除了位姿之外的所有数据块（速度/偏置）都被进行边缘化
  /// removeFrames表示窗口之前的所有未剔除的状态，但是不包括临近窗口的5个关键帧，这些帧的（位姿，外参）数据块都被进行边缘化
  /// allLinearizedFrames表示窗口（3帧）之前的所有未剔除的状态（帧）
  //千万千万要注意这里的rit不是结束位置而是在倒数第四个位置上
  //经过我们测试发现 removeFrames的数量一直维持在1个，而removeAllButPose数量一直维持在6个
  while (rit != statesMap_.rend())//一个逆序迭代器，容器c的第一个元素前面的位置
  {
    if (!rit->second.isKeyframe || countedKeyframes >= numKeyframes) 
	{
      removeFrames.push_back(rit->second.id);//就只在这里更新了
    } else 
   {
      countedKeyframes++;
    }
    removeAllButPose.push_back(rit->second.id);//就只在这里更新了
    allLinearizedFrames.push_back(rit->second.id);
    ++rit;// check the next frame
  }

  
  //下面的函数由两个for循环构成
  //一个是遍历removeFrames
  //一个是遍历removeAllButPose  
  // marginalize everything but pose:
   // 4.遍历边缘化除了imu窗口帧(最新的连续的三个帧)之外的所有帧的速度和bias
  for(size_t k = 0; k<removeAllButPose.size(); ++k)
  {
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeAllButPose[k]);

    ///在OKVIS中只有状态的global变量只有GlobalStates::T_WS位置有值
    //其实这个for循环是什么都没做
    for (size_t i = 0; i < it->second.global.size(); ++i) 
	{
      if (i == GlobalStates::T_WS) //好像只有这个global类型啊 。搜索 enum GlobalStates
	  {
        continue; // we do not remove the pose here.
      }//因为我们这里只用到了GlobalStates::T_WS类型所以下面的代码永远不会被执行到
	  /*
      if (!it->second.global[i].exists)//不能存在直接跳过 
	  {
        continue; // if it doesn't exist, we don't do anything.
      }
	   //设置为TWS数据块为固定，直接跳过
      //global的id和状态的id相等，而状态的id和位姿（TWS）数据块的id相等
      if (mapPtr_->parameterBlockPtr(it->second.global[i].id)->fixed())
	  {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      // 与下一个状态id保持一致，则直接跳过
      if(checkit->second.global[i].exists && checkit->second.global[i].id == it->second.global[i].id)
	  {
        continue;
      }
	  ///状态的存在性定义为false
      it->second.global[i].exists = false; // remember we removed
       ///添加状态第i个全局量（TSW，等等）id（即是数据块的id）到容器中
      paremeterBlocksToBeMarginalized.push_back(it->second.global[i].id);
      keepParameterBlocks.push_back(false);
	  //ResidualBlockCollection元素 = 残差的信息=残差在ceres中的id+loss function函数指针+误差函数指针
	  //搜索 Map::ResidualBlockCollection Map::residuals(
	  //这个函数的作用是从id2ResidualBlock_Multimap_结构中寻找与输入的参数快id相关的残差块，并将这些残差块返回
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.global[i].id);
      for (size_t r = 0; r < residuals.size(); ++r)//遍历与这个参数块相关的残差块
	  {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =  std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>( residuals[r].errorInterfacePtr);
        if(!reprojectionError) //没有重投影误差
		{   // we make sure no reprojection errors are yet included.
		   //搜索 bool MarginalizationError::addResidualBlock(
		   //非常重要的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);//直接添加残差到误差集合中
        }
      }*/
    }


    /// 需要边缘化帧的传感器类别
    /// 第一行：遍历所有传感器
    /// 第二行：遍历该传感器的数量
    /// 第三行：第i种第j个传感器的状态数量
    // 这个for循环的作用是边缘化除了imu窗口帧(最新的连续的三个帧)之外的所有帧的速度和bias
    //这个函数的作用主要是将速度和bias对应的残差块添加到marginalizationErrorPtr_中去
    // add all error terms of the sensor states.
    for (size_t i = 0; i < it->second.sensors.size(); ++i) {
      for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
        for (size_t k = 0; k < it->second.sensors[i][j].size(); ++k) 
		{
		  //搜索 enum SensorStates，只用到了 Camera = 0,  Imu = 1,
		  //搜索 enum CameraSensorStates，T_SCi = 0,  CameraSensorStates::Intrinsics = 1,
          if (i == SensorStates::Camera && k == CameraSensorStates::T_SCi) 
		  {
            continue; // we do not remove the extrinsics pose here.
          }
		  //该状态不存在的直接跳过
		  //根据我们分析可知intrinstrics=false，
          if (!it->second.sensors[i][j][k].exists) {
            continue;
          }

		  
		  //只边缘化imu的速度和bias
		  //如果imu的速度和bias设置为固定值则也直接跳过
          if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)->fixed()) 
		  {
            continue;  // we never eliminate fixed blocks.
          }

		  //
          std::map<uint64_t, States>::iterator checkit = it;
          checkit++;
          // only get rid of it, if it's different
          //相比下一个状态量，未发生变化，直接跳过
          if(checkit->second.sensors[i][j][k].exists && checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id)
		  {
            continue;
          }
          it->second.sensors[i][j][k].exists = false; // remember we removed
          //需要被边缘化的传感器状态（外参或速度偏置数据块）的序号
          paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
          keepParameterBlocks.push_back(false);
		  //搜索 Map::ResidualBlockCollection Map::residuals(
	     //这个函数的作用是从id2ResidualBlock_Multimap_结构中寻找与输入的参数快id相关的残差块，并将这些残差块返回
          ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.sensors[i][j][k].id);
          for (size_t r = 0; r < residuals.size(); ++r) //遍历与这个速度和bias相关的残差块
		  {
		     //ResidualBlockCollection元素 = 残差的信息=残差在ceres中的id+loss function函数指针+误差函数指针
            std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError = std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
            if(!reprojectionError) //不是重投影误差
			{   // we make sure no reprojection errors are yet included.
			  //搜索 bool MarginalizationError::addResidualBlock(
		      //非常重要的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		      //这里是将速度和bias对应的残差块添加到marginalizationErrorPtr_中了
              marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);//添加残差到系统
            }
          }
        }
      }
    }
  }// for(size_t k = 0; k<removeAllButPose.size(); ++k)结束

  
  // marginalize ONLY pose now:
  ///5.除去imu窗口帧和最近的五个关键帧对剩下的帧进行边缘化
  bool reDoFixation = false;
  for(size_t k = 0; k<removeFrames.size(); ++k)
  {
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeFrames[k]);

    // schedule removal - but always keep the very first frame.
    //if(it != statesMap_.begin()){
    if(true)
	{ /////DEBUG
      it->second.global[GlobalStates::T_WS].exists = false; // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[GlobalStates::T_WS].id);
      keepParameterBlocks.push_back(false);
    }
    //5.1将imu误差相关的要被边缘化的残差块加入到边缘化模块中
    // add remaing error terms
     //这个函数的作用是从id2ResidualBlock_Multimap_结构中寻找与输入的机器人姿态参数快id相关的残差块，并将这些残差块返回
    ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.global[GlobalStates::T_WS].id);

    for (size_t r = 0; r < residuals.size(); ++r)//遍历位姿相关的残差块
	{
      if(std::dynamic_pointer_cast<ceres::PoseError>(residuals[r].errorInterfacePtr)) //如果是位姿偏差
	  {                  // avoids linearising initial pose error
	                     //搜索 bool Map::removeResidualBlock(::ceres::ResidualBlockId residualBlockId) 
	                    //1.从ceres中删除这个残差块
				//2. 从id2ResidualBlock_Multimap_删除残差块
				//3.更新residualBlockId2ParameterBlockCollection_Map_
				//4.更新residualBlockId2ResidualBlockSpec_Map_
				mapPtr_->removeResidualBlock(residuals[r].residualBlockId);
				reDoFixation = true;
        continue; 
      }
      std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>( residuals[r].errorInterfacePtr);
      if(!reprojectionError)
	  {   // we make sure no reprojection errors are yet included.
	 	 //搜索 bool MarginalizationError::addResidualBlock(
        marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);//将机器人姿态相关并与重投影无关的残差块加入到marginalizationErrorPtr_中
      }
    }

    // add remaining error terms of the sensor states.
    /*我们这里把实际情况中没有运行的代码注释掉 方便观看
    size_t i = SensorStates::Camera;
    for (size_t j = 0; j < it->second.sensors[i].size(); ++j) //左右两个相机
	{
      size_t k = CameraSensorStates::T_SCi;
      if (!it->second.sensors[i][j][k].exists) 
	  {
        continue;
      }
      if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)->fixed()) 
	  {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if(checkit->second.sensors[i][j][k].exists && checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id)
	  {
        continue;
      }
      it->second.sensors[i][j][k].exists = false; // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.sensors[i][j][k].id);//找到和相机外参相关的残差块
      for (size_t r = 0; r < residuals.size(); ++r) 
	  {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError = std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>( residuals[r].errorInterfacePtr);
        if(!reprojectionError)
		{   // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);//相机外参相关的残差块加入到marginalizationErrorPtr_
        }
      }
    }*/

    // now finally we treat all the observations.
    //5.2遍历landmarksMap_结构中的地图点
    OKVIS_ASSERT_TRUE_DBG(Exception, allLinearizedFrames.size()>0, "bug");
    uint64_t currentKfId = allLinearizedFrames.at(0);//距离imu窗口帧最近的那个帧

    {
      for(PointMap::iterator pit = landmarksMap_.begin();pit != landmarksMap_.end(); )//遍历地图中地图点
	  {
        //元素 = 残差的信息 = 残差在ceres中的id+loss function函数指针+误差函数指针
        //找到与找个地图点参数块相关的所有残差块
        ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(pit->first);

        // first check if we can skip
        bool skipLandmark = true;
        bool hasNewObservations = false;
        bool justDelete = false;
        bool marginalize = true;
        bool errorTermAdded = false;
        std::map<uint64_t,bool> visibleInFrame;
        size_t obsCount = 0;//被非imu窗口帧观测到的次数
        for (size_t r = 0; r < residuals.size(); ++r) //遍历所有与这个地图点有关的残差块
		{
	          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =  std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>( residuals[r].errorInterfacePtr);
	          if (reprojectionError) 
			  {
			    //搜索 Map::ParameterBlockCollection Map::parameters
			    //parameters作用是找到与输入的BA残差块id相关的所有参数块,其中包括位姿和地图点的id
			    //parameters函数返回的数据结构中元素  =(参数块id,参数块指针构成)
			    //找到与输入的BA残差块id相关的位姿的参数块。
	            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
	            // since we have implemented the linearisation to account for robustification,
	            // we don't kick out bad measurements here any more like
	            // if(vectorContains(allLinearizedFrames,poseId)){ ...
	            //   if (error.transpose() * error > 6.0) { ... removeObservation ... }
	            // }
	            if(vectorContains(removeFrames,poseId))//表示这个地图点在imu窗口之前的普通帧中被观测到了
				{
	              skipLandmark = false;//表示需要对该地图点进行边缘化操作
	            }
	            if(poseId>=currentKfId) //imu窗口中的帧看到了这个地图点
				{
	              marginalize = false;
	              hasNewObservations = true;
	            }
				//imu窗口之外的帧看到了这个地图点
	            if(vectorContains(allLinearizedFrames, poseId))
				{
	              visibleInFrame.insert(std::pair<uint64_t,bool>(poseId,true));
	              obsCount++;
	            }
	          }
        }//遍历地图点的残差块结束
	 //a.如果无该特征点有关的残差项--------------------------------------------------
        if(residuals.size()==0)
		{
		  //搜索 bool Map::removeParameterBlock(uint64_t parameterBlockId) 
          mapPtr_->removeParameterBlock(pit->first);//从ceres中删除这个地图点对应的残差块，删除这个地图点的参数块
          removedLandmarks.push_back(pit->second);//更新移除地标点的集合
          pit = landmarksMap_.erase(pit);//从地图中删除这个地图点
          continue;
        }
	 //b.表示对该地图点不进行边缘化操作-----------------------------------------------
        if(skipLandmark) 
		{
          pit++;//移动到下一个地图点
          continue;
        }

        // so, we need to consider it.
        //再次遍历和这个地图点相关的的残差块
        //判断与该地标点关联的残差是不是重投影误差
        for (size_t r = 0; r < residuals.size(); ++r)
		{
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError = std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
          if (reprojectionError) 
		  {
		    //找到与输入的BA残差块id相关的位姿的参数块。
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
			///（imu窗口之外的普通帧看到了这个地图点且imu窗口中有帧可以观测到该地标点）或
            ///（非imu窗口帧看不到这个地图点且最新的三帧也不能观测到该地标点）,比如论文中地标点2,3
            //经过测试发现第二个条件永远不会满足!!!!!!!
            if((vectorContains(removeFrames,poseId) && hasNewObservations) ||(!vectorContains(allLinearizedFrames,poseId) && marginalize))
            {
              // ok, let's ignore the observation.
              //c.-----------------------------------------------------------------
              removeObservation(residuals[r].residualBlockId);//从ceres中删除这个残差块
              residuals.erase(residuals.begin() + r);
              r--;
            } else if(marginalize && vectorContains(allLinearizedFrames,poseId)) 
            {///imu窗口中不能观测到该地标点且非imu窗口帧能看到这个地图点,比如论文中地标点1
             
              // TODO: consider only the sensible ones for marginalization
              //d------------------------------------------------------------------------
              if(obsCount<2) //被imu窗口之外的所有帧观测到的次数小于2
			  { //visibleInFrame.size()
                removeObservation(residuals[r].residualBlockId);//从ceres中删除这个残差块
                residuals.erase(residuals.begin() + r);
                r--;
              } else 
              {
                // add information to be considered in marginalization later.
                // e.添加该观测进入残差空间中--------------------------------------------------------
                errorTermAdded = true;
                marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId, false);//将这个地图点的残差块添加到marginalizationErrorPtr_中
              }
            }
            // check anything left
             //检查该地标点的二次投影误差是否被剔除完了
             //如果剔除完了说明imu窗口中没有这个地图点的BA约束
            if (residuals.size() == 0) 
			{
              justDelete = true;
              marginalize = false;
            }
          }
        }//二次遍历和这个地图点相关的的残差块
        
        //地标点的观测全部被删除完了，即imu窗口中没有这个地图点的BA约束
        if(justDelete)
		{
		  //从ceres中删除与这个地图点参数块相关的残差块；从ceres中删除这个参数块。
          mapPtr_->removeParameterBlock(pit->first);//搜索 bool Map::removeParameterBlock(uint64_t parameterBlockId) 
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);//从地图中删除这个地图点
          continue;
        }
		
		//imu窗口中不能观测到该地标点且被imu窗口之外的帧观测到的次数大于2 
        if(marginalize&&errorTermAdded)
		{
          paremeterBlocksToBeMarginalized.push_back(pit->first);//表示这个地图点需要被边缘化
          keepParameterBlocks.push_back(false);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);//这里千万要注意一个细节 ，因为这里作者删除了landmarksMap_一个值那么指针pit自动指向了下一个元素
          continue;
        }

        pit++;//移动到下个地图点
      }//for(PointMap::iterator pit = landmarksMap_.begin();pit != landmarksMap_.end(); )地图中地图点循环结束
    }

    // update book-keeping and go to the next frame
    //if(it != statesMap_.begin()){ // let's remember that we kept the very first pose
    if(true) 
	{ ///// DEBUG
      multiFramePtrMap_.erase(it->second.id);
      statesMap_.erase(it->second.id);//整个代码中就这里删除了statesMap_的内容,输入的是要被边缘化的帧的id
    }
  }//遍历removeframes结束

  // now apply the actual marginalization
  if(paremeterBlocksToBeMarginalized.size()>0)//要被边缘化的参数块个数大于0
  {
    std::vector< ::ceres::ResidualBlockId> addedPriors;
	//搜索 bool MarginalizationError::marginalizeOut(
	//非常重要的函数!!!!!!!!!!!!!!!!!!!!!!!! 开始执行边缘化
	/// 1&2通过shur消元更新H_和b_矩阵
    //3.重新设置ceres优化的残差维度。ceres中更新costfunction中的参数块的大小。删除ceres中的参数块。
    marginalizationErrorPtr_->marginalizeOut(paremeterBlocksToBeMarginalized, keepParameterBlocks);
  }

  // update error computation
  if(paremeterBlocksToBeMarginalized.size()>0)//为需要边缘化的数据块id的集合
  {
    //非常重要的函数 void MarginalizationError::updateErrorComputation() 
    //进行状态的更新
    //非常重要的函数!!!!!!!!!!!!!!!!!!!!!!!
    //我们已经得到了H矩阵 我们对H矩阵进行svd分解然后计算得到J矩阵
     //并进一步得到-J.transpose().inverse()*b=e0
    marginalizationErrorPtr_->updateErrorComputation();
  }

  // add the marginalization term again
  if(marginalizationErrorPtr_->num_residuals()==0)
  {
    marginalizationErrorPtr_.reset();//这个清零 意味着要对ceres的边缘化残差块清空了
  }
  //向ceres中添加误差函数
  if (marginalizationErrorPtr_) 
  {
	  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock> > parameterBlockPtrs;
	  //搜索 void MarginalizationError::getParameterBlockPtrs(
	  //得到未被边缘化掉的所有数据块
	  marginalizationErrorPtr_->getParameterBlockPtrs(parameterBlockPtrs);
	  //搜索 ::ceres::ResidualBlockId Map::addResidualBlock(
	  //整个代码范围内就只在这里将边缘化的误差函数加入到了ceres中，没有loss函数
	  marginalizationResidualId_ = mapPtr_->addResidualBlock( marginalizationErrorPtr_, NULL, parameterBlockPtrs);//将未被边缘化的误差函数加入优化地图中
	  OKVIS_ASSERT_TRUE_DBG(Exception, marginalizationResidualId_, "could not add marginalization error");
	  if (!marginalizationResidualId_)
	    return false;
  }
	
	if(reDoFixation)//如果删除了位姿误差，初始化的时候会发生
	{
	  // finally fix the first pose properly
		//mapPtr_->resetParameterization(statesMap_.begin()->first, ceres::Map::Pose3d);
	  okvis::kinematics::Transformation T_WS_0;
	  get_T_WS(statesMap_.begin()->first, T_WS_0);//获取imu窗口的距离当前帧最远的帧
	  Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Zero();
	  information(5,5) = 1.0e14; information(0,0) = 1.0e14; 
	  information(1,1) = 1.0e14; information(2,2) = 1.0e14;
	  std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS_0, information));
	  //搜索::ceres::ResidualBlockId Map::addResidualBlock(
	  //向ceres中增加误差函数
	  //重新补一个初始位姿误差
	  mapPtr_->addResidualBlock(poseError,NULL,mapPtr_->parameterBlockPtr(statesMap_.begin()->first));
	}

  return true;
}
//applyMarginalizationStrategy函数结束--------------------------------------------------------

// Prints state information to buffer.
void Estimator::printStates(uint64_t poseId, std::ostream & buffer) const {
  buffer << "GLOBAL: ";
  for(size_t i = 0; i<statesMap_.at(poseId).global.size(); ++i){
    if(statesMap_.at(poseId).global.at(i).exists) {
      uint64_t id = statesMap_.at(poseId).global.at(i).id;
      if(mapPtr_->parameterBlockPtr(id)->fixed())
        buffer << "(";
      buffer << "id="<<id<<":";
      buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
      if(mapPtr_->parameterBlockPtr(id)->fixed())
        buffer << ")";
      buffer <<", ";
    }
  }
  buffer << "SENSOR: ";
  for(size_t i = 0; i<statesMap_.at(poseId).sensors.size(); ++i){
    for(size_t j = 0; j<statesMap_.at(poseId).sensors.at(i).size(); ++j){
      for(size_t k = 0; k<statesMap_.at(poseId).sensors.at(i).at(j).size(); ++k){
        if(statesMap_.at(poseId).sensors.at(i).at(j).at(k).exists) {
          uint64_t id = statesMap_.at(poseId).sensors.at(i).at(j).at(k).id;
          if(mapPtr_->parameterBlockPtr(id)->fixed())
            buffer << "(";
          buffer << "id="<<id<<":";
          buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
          if(mapPtr_->parameterBlockPtr(id)->fixed())
            buffer << ")";
          buffer <<", ";
        }
      }
    }
  }
  buffer << std::endl;
}

// Initialise pose from IMU measurements. For convenience as static.
//输入的数据是imu测量值和上一个图像的位姿T_WS，将得到的位姿更新到T_WS变量中
//初始时相机的位置=0，主要是使用加速度测量值对姿态进行求解，初始姿态是在地球坐标系下的
//详见算法实现文档
bool Estimator::initPoseFromImu(
    const okvis::ImuMeasurementDeque & imuMeasurements,
    okvis::kinematics::Transformation & T_WS)
{
  // set translation to zero, unit rotation
  T_WS.setIdentity();

  if (imuMeasurements.size() == 0)
    return false;

  // acceleration vector
  Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();it < imuMeasurements.end(); ++it) 
  {
    acc_B += it->measurement.accelerometers;
  }
  acc_B /= double(imuMeasurements.size());//计算加速度的平均值
  Eigen::Vector3d e_acc = acc_B.normalized();

  // align with ez_W:
  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
  Eigen::Matrix<double, 6, 1> poseIncrement;//前三维是位移t 后三维是轴角
  poseIncrement.head<3>() = Eigen::Vector3d::Zero();
  poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();//计算得到轴
  double angle = std::acos(ez_W.transpose() * e_acc);
  poseIncrement.tail<3>() *= angle;
  T_WS.oplus(-poseIncrement);//poseIncrement表示轴角

  return true;
}

// Start ceres optimization.
#ifdef USE_OPENMP
void Estimator::optimize(size_t numIter, size_t numThreads,
                                 bool verbose)
#else//默认进入这个条件
//默认verbose=false
//这个函数就是开始非线性优化，使用的是dogleg方法，线性求解器使用的是SPARSE_SCHUR
//最大优化步长=10
//并更新地图点的数据
void Estimator::optimize(size_t numIter, size_t /*numThreads*/,
                                 bool verbose) // avoid warning since numThreads unused
#warning openmp not detected, your system may be slower than expected
#endif

{
  // assemble options
  mapPtr_->options.linear_solver_type = ::ceres::SPARSE_SCHUR;
  //mapPtr_->options.initial_trust_region_radius = 1.0e4;
  //mapPtr_->options.initial_trust_region_radius = 2.0e6;
  //mapPtr_->options.preconditioner_type = ::ceres::IDENTITY;
  mapPtr_->options.trust_region_strategy_type = ::ceres::DOGLEG;
  //mapPtr_->options.trust_region_strategy_type = ::ceres::LEVENBERG_MARQUARDT;
  //mapPtr_->options.use_nonmonotonic_steps = true;
  //mapPtr_->options.max_consecutive_nonmonotonic_steps = 10;
  //mapPtr_->options.function_tolerance = 1e-12;
  //mapPtr_->options.gradient_tolerance = 1e-12;
  //mapPtr_->options.jacobi_scaling = false;
#ifdef USE_OPENMP
    mapPtr_->options.num_threads = numThreads;
#endif//默认进入这个条件
  mapPtr_->options.max_num_iterations = numIter;

  if (verbose) //默认false
  {
    mapPtr_->options.minimizer_progress_to_stdout = true;
  } else {
    mapPtr_->options.minimizer_progress_to_stdout = false;
  }

  // call solver
  //搜索 void solve() {
  mapPtr_->solve();//调用了ceres::solve函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 //优化结束后....
  // update landmarks
  {
    for(auto it = landmarksMap_.begin(); it!=landmarksMap_.end(); ++it)//遍历所有的地图点
	{
      Eigen::MatrixXd H(3,3);
	  //获得这个地图点对应的Hessian矩阵 
	  //搜索 void Map::getLhs(
	  //整个代码中就这里调用了getLhs函数
	  //输入的是参数块id，提取出和这个参数块相关的残差块，并计算这个残差块对这个参数块的雅克比J
      //然后我们将JT*J叠加到一起作为最终的值返回H
      mapPtr_->getLhs(it->first,H);
      //计算得到这个hessian矩阵的最小和最大特征值
      Eigen::SelfAdjointEigenSolver< Eigen::Matrix3d > saes(H);
      Eigen::Vector3d eigenvalues = saes.eigenvalues();
      const double smallest = (eigenvalues[0]);
      const double largest = (eigenvalues[2]);
	  //3.2更新质量
      if(smallest<1.0e-12)
	  {
        // this means, it has a non-observable depth
        it->second.quality = 0.0;
      } else {
        // OK, well constrained
        it->second.quality = sqrt(smallest)/sqrt(largest);//好像quality这个参数并没有用到
      }

      // update coordinates
      //3.3更新坐标
      it->second.point = std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
          mapPtr_->parameterBlockPtr(it->first))->estimate();//根据优化后的地图点坐标，重新赋值landmarksMap_结构中地图点坐标
    }
  }

  // summary output
  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
}

// Set a time limit for the optimization process.
/* timeLimit Time limit in seconds. If timeLimit < 0 the time limit is removed.
   * @param[in] minIterations minimum iterations the optimization process should do
   *            disregarding the time limit.*/
//fyy setOptimizationTimeLimit
bool Estimator::setOptimizationTimeLimit(double timeLimit, int minIterations) 
{
  if(ceresCallback_ != nullptr)
  {
    if(timeLimit < 0.0) {
      // no time limit => set minimum iterations to maximum iterations
      ceresCallback_->setMinimumIterations(mapPtr_->options.max_num_iterations);
      return true;
    }
    ceresCallback_->setTimeLimit(timeLimit);
    ceresCallback_->setMinimumIterations(minIterations);
    return true;
  }
  else if(timeLimit >= 0.0) {
    ceresCallback_ = std::unique_ptr<okvis::ceres::CeresIterationCallback>(new okvis::ceres::CeresIterationCallback(timeLimit,minIterations));
	//  options类型 ceres::Solver::Options options;
    mapPtr_->options.callbacks.push_back(ceresCallback_.get());//设置ceres相关
    return true;
  }
  // no callback yet registered with ceres.
  // but given time limit is lower than 0, so no callback needed
  return true;
}

// getters
// Get a specific landmark.
bool Estimator::getLandmark(uint64_t landmarkId,
                                    MapPoint& mapPoint) const
{
  std::lock_guard<std::mutex> l(statesMutex_);
  if (landmarksMap_.find(landmarkId) == landmarksMap_.end()) {
    OKVIS_THROW_DBG(Exception,"landmark with id = "<<landmarkId<<" does not exist.")
    return false;
  }
  mapPoint = landmarksMap_.at(landmarkId);
  return true;
}

// Checks whether the landmark is initialized.
//ceres的参数块中没有这个地图点的参数
bool Estimator::isLandmarkInitialized(uint64_t landmarkId) const 
{
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),"landmark not added");
  return std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(mapPtr_->parameterBlockPtr(landmarkId))->initialized();
}

// Get a copy of all the landmarks as a PointMap.
size_t Estimator::getLandmarks(PointMap & landmarks) const
{
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks = landmarksMap_;
  return landmarksMap_.size();
}

// Get a copy of all the landmark in a MapPointVector. This is for legacy support.
// Use getLandmarks(okvis::PointMap&) if possible.
size_t Estimator::getLandmarks(MapPointVector & landmarks) const
{
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks.clear();
  landmarks.reserve(landmarksMap_.size());
  for(PointMap::const_iterator it=landmarksMap_.begin(); it!=landmarksMap_.end(); ++it
  {
    landmarks.push_back(it->second);
  }
  return landmarksMap_.size();
}

// Get pose for a given pose ID.
bool Estimator::get_T_WS(uint64_t poseId,
                                 okvis::kinematics::Transformation & T_WS) const
{
  if (!getGlobalStateEstimateAs<ceres::PoseParameterBlock>(poseId,  GlobalStates::T_WS, T_WS)) 
  {
    return false;
  }

  return true;
}

// Feel free to implement caching for them...
// Get speeds and IMU biases for a given pose ID.
bool Estimator::getSpeedAndBias(uint64_t poseId, uint64_t imuIdx,
                                okvis::SpeedAndBias & speedAndBias) const
{
  if (!getSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
      poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias,
      speedAndBias)) {
    return false;
  }
  return true;
}

// Get camera states for a given pose ID.
bool Estimator::getCameraSensorStates( uint64_t poseId, size_t cameraIdx,okvis::kinematics::Transformation & T_SCi) const
{
  return getSensorStateEstimateAs<ceres::PoseParameterBlock>(poseId, cameraIdx, SensorStates::Camera, CameraSensorStates::T_SCi, T_SCi);
}

// Get the ID of the current keyframe.
uint64_t Estimator::currentKeyframeId() const {
  for (std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin();
      rit != statesMap_.rend(); ++rit) {
    if (rit->second.isKeyframe) {
      return rit->first;
    }
  }
  OKVIS_THROW_DBG(Exception, "no keyframes existing...");
  return 0;
}

// Get the ID of an older frame.
uint64_t Estimator::frameIdByAge(size_t age) const 
{
  std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin();
  for(size_t i=0; i<age; ++i){
    ++rit;
    OKVIS_ASSERT_TRUE_DBG(Exception, rit != statesMap_.rend(),
                       "requested age " << age << " out of range.");
  }
  return rit->first;
}

// Get the ID of the newest frame added to the state.
uint64_t Estimator::currentFrameId() const {
  OKVIS_ASSERT_TRUE_DBG(Exception, statesMap_.size()>0, "no frames added yet.")
  return statesMap_.rbegin()->first;
}

// Checks if a particular frame is still in the IMU window
bool Estimator::isInImuWindow(uint64_t frameId) const 
{
  if(statesMap_.at(frameId).sensors.at(SensorStates::Imu).size()==0)
  {
    return false; // no IMU added
  }
  return statesMap_.at(frameId).sensors.at(SensorStates::Imu).at(0).at(ImuSensorStates::SpeedAndBias).exists;
}

// Set pose for a given pose ID.
bool Estimator::set_T_WS(uint64_t poseId,
                                 const okvis::kinematics::Transformation & T_WS)
{
  if (!setGlobalStateEstimateAs<ceres::PoseParameterBlock>(poseId,GlobalStates::T_WS,T_WS)) 
  {
    return false;
  }

  return true;
}

// Set the speeds and IMU biases for a given pose ID.
bool Estimator::setSpeedAndBias(uint64_t poseId, size_t imuIdx, const okvis::SpeedAndBias & speedAndBias)
{
  return setSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
      poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias, speedAndBias);
}

// Set the transformation from sensor to camera frame for a given pose ID.
bool Estimator::setCameraSensorStates(
    uint64_t poseId, size_t cameraIdx,
    const okvis::kinematics::Transformation & T_SCi)
{
  return setSensorStateEstimateAs<ceres::PoseParameterBlock>(
      poseId, cameraIdx, SensorStates::Camera, CameraSensorStates::T_SCi, T_SCi);
}

// Set the homogeneous coordinates for a landmark.
//1.从mapPtr_中提取出地图点对应的参数块指针，并修改其值
//2.修改landmarksMap_中地图点的值
bool Estimator::setLandmark(uint64_t landmarkId, const Eigen::Vector4d & landmark)
{
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(landmarkId);
#ifndef NDEBUG //默认进入这个条件
  
  std::shared_ptr<ceres::HomogeneousPointParameterBlock> derivedParameterBlockPtr =
 			 std::dynamic_pointer_cast<ceres::HomogeneousPointParameterBlock>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception,"wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(landmark);//搜索 void HomogeneousPointParameterBlock::setEstimate(
#else
  std::static_pointer_cast<ceres::HomogeneousPointParameterBlock>(
      parameterBlockPtr)->setEstimate(landmark);
#endif

  // also update in map
  landmarksMap_.at(landmarkId).point = landmark;
  return true;
}

// Set the landmark initialization state.
void Estimator::setLandmarkInitialized(uint64_t landmarkId,bool initialized) 
{
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),"landmark not added");
  //参数块中的信息 已经被初始化了
  std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(mapPtr_->parameterBlockPtr(landmarkId))->setInitialized(initialized);
}

// private stuff
// getters
bool Estimator::getGlobalStateParameterBlockPtr(
    uint64_t poseId, int stateType,
    std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const
{
  // check existence in states set
  // std::map<uint64_t, States> statesMap_; 
  if (statesMap_.find(poseId) == statesMap_.end()) 
  {
    OKVIS_THROW(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) 
  {
    OKVIS_THROW(Exception,"pose with id = "<<id<<" does not exist.")
    return false;
  }

  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}


template<class PARAMETER_BLOCK_T>
bool Estimator::getGlobalStateParameterBlockAs(
    uint64_t poseId, int stateType,
    PARAMETER_BLOCK_T & stateParameterBlock) const
{
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  //getGlobalStateParameterBlockPtr 这个函数就在上面
  if (!getGlobalStateParameterBlockPtr(poseId, stateType, parameterBlockPtr))
  {
    return false;
  }
#ifndef NDEBUG //默认进入这个条件
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr = std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
    LOG(INFO) << "--"<<parameterBlockPtr->typeInfo();
    std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
    OKVIS_THROW_DBG(Exception,"wrong pointer type requested: requested "
                 <<info->typeInfo()<<" but is of type"
                 <<parameterBlockPtr->typeInfo())
    return false;
  }
  stateParameterBlock = *derivedParameterBlockPtr;
#else
  stateParameterBlock = *std::static_pointer_cast<PARAMETER_BLOCK_T>(
      parameterBlockPtr);
#endif
  return true;
}
template<class PARAMETER_BLOCK_T>
bool Estimator::getGlobalStateEstimateAs(uint64_t poseId, int stateType,typename PARAMETER_BLOCK_T::estimate_t & state) const
{
  PARAMETER_BLOCK_T stateParameterBlock;
  if (!getGlobalStateParameterBlockAs(poseId, stateType, stateParameterBlock))//搜索 bool Estimator::getGlobalStateParameterBlockAs(
  {
    return false;
  }
  state = stateParameterBlock.estimate();
  return true;
}

bool Estimator::getSensorStateParameterBlockPtr(
    uint64_t poseId, int sensorIdx, int sensorType, int stateType,
    std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const
{
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).sensors.at(sensorType).at(sensorIdx).at(
      stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }
  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}
template<class PARAMETER_BLOCK_T>
bool Estimator::getSensorStateParameterBlockAs(
    uint64_t poseId, int sensorIdx, int sensorType, int stateType,
    PARAMETER_BLOCK_T & stateParameterBlock) const
{
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  if (!getSensorStateParameterBlockPtr(poseId, sensorIdx, sensorType, stateType,
                                       parameterBlockPtr)) {
    return false;
  }
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
  std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
    std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
    OKVIS_THROW_DBG(Exception,"wrong pointer type requested: requested "
                     <<info->typeInfo()<<" but is of type"
                     <<parameterBlockPtr->typeInfo())
    return false;
  }
  stateParameterBlock = *derivedParameterBlockPtr;
#else
  stateParameterBlock = *std::static_pointer_cast<PARAMETER_BLOCK_T>(
      parameterBlockPtr);
#endif
  return true;
}
template<class PARAMETER_BLOCK_T>
bool Estimator::getSensorStateEstimateAs(
    uint64_t poseId, int sensorIdx, int sensorType, int stateType,
    typename PARAMETER_BLOCK_T::estimate_t & state) const
{
  PARAMETER_BLOCK_T stateParameterBlock;
  if (!getSensorStateParameterBlockAs(poseId, sensorIdx, sensorType, stateType,
                                      stateParameterBlock)) {
    return false;
  }
  state = stateParameterBlock.estimate();
  return true;
}

//
template<class PARAMETER_BLOCK_T>
bool Estimator::setGlobalStateEstimateAs(
    uint64_t poseId, int stateType,
    const typename PARAMETER_BLOCK_T::estimate_t & state)
{
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) 
  {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) 
  {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(id);
#ifndef NDEBUG//默认进入这个条件
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) 
  {
    OKVIS_THROW_DBG(Exception,"wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(state);
#else
  std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)->setEstimate(state);
#endif
  return true;
}

template<class PARAMETER_BLOCK_T>
bool Estimator::setSensorStateEstimateAs(
    uint64_t poseId, int sensorIdx, int sensorType, int stateType,
    const typename PARAMETER_BLOCK_T::estimate_t & state)
{
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).sensors.at(sensorType).at(sensorIdx).at(
      stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_
      ->parameterBlockPtr(id);
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
  std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception,"wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(state);
#else
  std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)->setEstimate(
      state);
#endif
  return true;
}

}  // namespace okvis


