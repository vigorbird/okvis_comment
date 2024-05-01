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
 *  Created on: Aug 21, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file ThreadedKFVio.cpp
 * @brief Source file for the ThreadedKFVio class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <map>

#include <glog/logging.h>

#include <okvis/ThreadedKFVio.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

static const int max_camera_input_queue_size = 10;
static const okvis::Duration temporal_imu_data_overlap(0.02);  // overlap of imu data before and after two consecutive frames [seconds]

#ifdef USE_MOCK
// Constructor for gmock.
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters, okvis::MockVioBackendInterface& estimator,
    okvis::MockVioFrontendInterface& frontend)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      estimator_(estimator),
      frontend_(frontend),
      parameters_(parameters),
      maxImuInputQueueSize_(60) {
  init();
}
#else
// Constructor. ，默认是这个条件 ThreadedKFVio构造函数
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      estimator_(),
      frontend_(parameters.nCameraSystem.numCameras()),
      parameters_(parameters),
      maxImuInputQueueSize_(
          2 * max_camera_input_queue_size * parameters.imu.rate
              / parameters.sensors_information.cameraRate) 
{
  setBlocking(false);
  init();
}
#endif

// Initialises settings and calls startThreads().
void ThreadedKFVio::init() 
{
  assert(parameters_.nCameraSystem.numCameras() > 0);
  numCameras_ = parameters_.nCameraSystem.numCameras();
  numCameraPairs_ = 1;

  frontend_.setBriskDetectionOctaves(parameters_.optimization.detectionOctaves);//euroc设置=0
  frontend_.setBriskDetectionThreshold(parameters_.optimization.detectionThreshold);//euroc设置=40
  frontend_.setBriskDetectionMaximumKeypoints(parameters_.optimization.maxNoKeypoints);//euroc设置=400 每张图像特征点的最大数量

  lastOptimizedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)
  lastAddedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)

  //更新imu的参数
  //整个程序中就调用过一次这个函数
  estimator_.addImu(parameters_.imu);//搜索 int Estimator::addImu(const ImuParameters & imuParameters)
  for (size_t i = 0; i < numCameras_; ++i) //遍历左右图像
  {
    // parameters_.camera_extrinsics is never set (default 0's)...
    // do they ever change?
    //更新左/右相机的参数
    //整个程序就这里调用过一次
    estimator_.addCamera(parameters_.camera_extrinsics);//搜索 int Estimator::addCamera(const ExtrinsicsEstimationParameters & extrinsicsEstimationParameters)

    //产生相机接收队列的实例
	cameraMeasurementsReceived_.emplace_back(std::shared_ptr<threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > >(new threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> >()));
  }
  
  // set up windows so things don't crash on Mac OS
  if(parameters_.visualization.displayImages)//euroc设置为true
  {
    for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) 
	{
      std::stringstream windowname;
      windowname << "OKVIS camera " << im;
  	  cv::namedWindow(windowname.str());
    }
  }
  
  startThreads();//非常重要的函数！！！！
}

// Start all threads.
void ThreadedKFVio::startThreads() 
{

  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.emplace_back(&ThreadedKFVio::frameConsumerLoop, this, i);//这个应该也是开启线程 左右图像各一个线程  ？？？
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.emplace_back(&ThreadedKFVio::matchingLoop, this);//这个应该也是开启线程 双目默认只开启一个
  }
  imuConsumerThread_ = std::thread(&ThreadedKFVio::imuConsumerLoop, this);//开启了一个线程 搜索ThreadedKFVio::imuConsumerLoop
  positionConsumerThread_ = std::thread(&ThreadedKFVio::positionConsumerLoop,this);//开启了一个线程 搜索void ThreadedKFVio::positionConsumerLoop()
  gpsConsumerThread_ = std::thread(&ThreadedKFVio::gpsConsumerLoop, this);//不用管 是个空线程
  magnetometerConsumerThread_ = std::thread( &ThreadedKFVio::magnetometerConsumerLoop, this);//不用管 是个空线程
  differentialConsumerThread_ = std::thread( &ThreadedKFVio::differentialConsumerLoop, this);//不用管 是个空线程

  // algorithm threads
  visualizationThread_ = std::thread(&ThreadedKFVio::visualizationLoop, this);//开启了一个线程 搜索 ThreadedKFVio::visualizationLoop()
  optimizationThread_ = std::thread(&ThreadedKFVio::optimizationLoop, this);//开启了一个线程 搜索 ThreadedKFVio::optimizationLoop
  publisherThread_ = std::thread(&ThreadedKFVio::publisherLoop, this);//开启了一个线程 搜索 ThreadedKFVio::publisherLoop() 
}

// Destructor. This calls Shutdown() for all threadsafe queues and joins all threads.
ThreadedKFVio::~ThreadedKFVio() {
  for (size_t i = 0; i < numCameras_; ++i) {
    cameraMeasurementsReceived_.at(i)->Shutdown();
  }
  keypointMeasurements_.Shutdown();
  matchedFrames_.Shutdown();
  imuMeasurementsReceived_.Shutdown();
  optimizationResults_.Shutdown();
  visualizationData_.Shutdown();
  imuFrameSynchronizer_.shutdown();
  positionMeasurementsReceived_.Shutdown();

  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.at(i).join();//函数可以在当前线程等待线程运行结束。
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.at(i).join();
  }
  imuConsumerThread_.join();
  positionConsumerThread_.join();
  gpsConsumerThread_.join();
  magnetometerConsumerThread_.join();
  differentialConsumerThread_.join();
  visualizationThread_.join();
  optimizationThread_.join();
  publisherThread_.join();

  /*okvis::kinematics::Transformation endPosition;
  estimator_.get_T_WS(estimator_.currentFrameId(), endPosition);
  std::stringstream s;
  s << endPosition.r();
  LOG(INFO) << "Sensor end position:\n" << s.str();
  LOG(INFO) << "Distance to origin: " << endPosition.r().norm();*/
#ifndef DEACTIVATE_TIMERS
  LOG(INFO) << okvis::timing::Timing::print();
#endif
}

// Add a new image.
//keypoints默认值= 0
bool ThreadedKFVio::addImage(const okvis::Time & stamp, size_t cameraIndex,
                             const cv::Mat & image,
                             const std::vector<cv::KeyPoint> * keypoints,
                             bool* /*asKeyframe*/) {
  assert(cameraIndex<numCameras_);

  //
  if (lastAddedImageTimestamp_ > stamp && fabs((lastAddedImageTimestamp_ - stamp).toSec())> parameters_.sensors_information.frameTimestampTolerance) 
  {
    LOG(ERROR)
        << "Received image from the past. Dropping the image.";
    return false;
  }
  lastAddedImageTimestamp_ = stamp;

  std::shared_ptr<okvis::CameraMeasurement> frame = std::make_shared<okvis::CameraMeasurement>();
  frame->measurement.image = image;
  frame->timeStamp = stamp;
  frame->sensorId = cameraIndex;

  if (keypoints != nullptr) 
  {
    frame->measurement.deliversKeypoints = true;
    frame->measurement.keypoints = *keypoints;
  } else 
  {
    frame->measurement.deliversKeypoints = false;
  }

  if (blocking_) //默认进入这个条件
  {
    //定义: std::vector<std::shared_ptr< okvis::threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > > > cameraMeasurementsReceived_;
    //搜索 bool PushBlockingIfFull(const QueueType& value, size_t max_queue_size) 
    cameraMeasurementsReceived_[cameraIndex]->PushBlockingIfFull(frame, 1);//向左/右相机队列中插入测量得到的图像，1表示如果这个队列中有图像则等待
    return true;
  } else 
  {
    cameraMeasurementsReceived_[cameraIndex]->PushNonBlockingDroppingIfFull(frame, max_camera_input_queue_size);
    return cameraMeasurementsReceived_[cameraIndex]->Size() == 1;
  }
}

// Add an abstracted image observation.
bool ThreadedKFVio::addKeypoints(
    const okvis::Time & /*stamp*/, size_t /*cameraIndex*/,
    const std::vector<cv::KeyPoint> & /*keypoints*/,
    const std::vector<uint64_t> & /*landmarkIds*/,
    const cv::Mat & /*descriptors*/,
    bool* /*asKeyframe*/) {
  OKVIS_THROW(
      Exception,
      "ThreadedKFVio::addKeypoints() not implemented anymore since changes to _keypointMeasurements queue.");
  return false;
}

// Add an IMU measurement.
bool ThreadedKFVio::addImuMeasurement(const okvis::Time & stamp,
                                      const Eigen::Vector3d & alpha,
                                      const Eigen::Vector3d & omega) 
{

  okvis::ImuMeasurement imu_measurement;
  imu_measurement.measurement.accelerometers = alpha;
  imu_measurement.measurement.gyroscopes = omega;
  imu_measurement.timeStamp = stamp;

  if (blocking_)//默认是进入这个条件的
  {
    //定义 okvis::threadsafe::ThreadSafeQueue<okvis::ImuMeasurement> imuMeasurementsReceived_;
    //搜索 bool PushBlockingIfFull(const QueueType& value, size_t max_queue_size) {
    //如果imu测量值队列中测量值数量大于1则等待，直到可以插入队列
    imuMeasurementsReceived_.PushBlockingIfFull(imu_measurement, 1);//
    return true;
  } else 
  {
    imuMeasurementsReceived_.PushNonBlockingDroppingIfFull(imu_measurement, maxImuInputQueueSize_);
    return imuMeasurementsReceived_.Size() == 1;
  }
}

// Add a position measurement.
void ThreadedKFVio::addPositionMeasurement(const okvis::Time & stamp,
                                           const Eigen::Vector3d & position,
                                           const Eigen::Vector3d & positionOffset,
                                           const Eigen::Matrix3d & positionCovariance) {
  okvis::PositionMeasurement position_measurement;
  position_measurement.measurement.position = position;
  position_measurement.measurement.positionOffset = positionOffset;
  position_measurement.measurement.positionCovariance = positionCovariance;
  position_measurement.timeStamp = stamp;

  if (blocking_) {
    positionMeasurementsReceived_.PushBlockingIfFull(position_measurement, 1);
    return;
  } else {
    positionMeasurementsReceived_.PushNonBlockingDroppingIfFull(
        position_measurement, maxPositionInputQueueSize_);
    return;
  }
}

// Add a GPS measurement.
void ThreadedKFVio::addGpsMeasurement(const okvis::Time &, double, double,
                                      double, const Eigen::Vector3d &,
                                      const Eigen::Matrix3d &) {
  OKVIS_THROW(Exception, "GPS measurements not supported")
}

// Add a magnetometer measurement.
void ThreadedKFVio::addMagnetometerMeasurement(const okvis::Time &,
                                               const Eigen::Vector3d &, double) {
  OKVIS_THROW(Exception, "Magnetometer measurements not supported")
}

// Add a static pressure measurement.
void ThreadedKFVio::addBarometerMeasurement(const okvis::Time &, double, double) {

  OKVIS_THROW(Exception, "Barometer measurements not supported")
}

// Add a differential pressure measurement.
void ThreadedKFVio::addDifferentialPressureMeasurement(const okvis::Time &,
                                                       double, double) {

  OKVIS_THROW(Exception, "Differential pressure measurements not supported")
}

// Set the blocking variable that indicates whether the addMeasurement() functions
// should return immediately (blocking=false), or only when the processing is complete.
void ThreadedKFVio::setBlocking(bool blocking) //在main函数中被设置
{
  blocking_ = blocking;//
  // disable time limit for optimization
  if(blocking_) 
  {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
	//全局搜索 fyy Estimator::setOptimizationTimeLimit
    estimator_.setOptimizationTimeLimit(-1.0,parameters_.optimization.max_iterations);
  }
}

// Loop to process frames from camera with index cameraIndex
//cameraIndex表示是左相机还是右相机
void ThreadedKFVio::frameConsumerLoop(size_t cameraIndex)
{
  std::shared_ptr<okvis::CameraMeasurement> frame;
  std::shared_ptr<okvis::MultiFrame> multiFrame;
  //声明这四个变量完全是空的什么都不做
  TimerSwitchable beforeDetectTimer("1.1 frameLoopBeforeDetect"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer("1.1.1 waitForFrameSynchronizerMutex"+std::to_string(cameraIndex),true);
  TimerSwitchable addNewFrameToSynchronizerTimer("1.1.2 addNewFrameToSynchronizer"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForStateVariablesMutexTimer("1.1.3 waitForStateVariablesMutex"+std::to_string(cameraIndex),true);
  TimerSwitchable propagationTimer("1.1.4 propagationTimer"+std::to_string(cameraIndex),true);
  TimerSwitchable detectTimer("1.2 detectAndDescribe"+std::to_string(cameraIndex),true);
  TimerSwitchable afterDetectTimer("1.3 afterDetect"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer2("1.3.1 waitForFrameSynchronizerMutex2"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForMatchingThreadTimer("1.4 waitForMatchingThread"+std::to_string(cameraIndex),true);


  for (;;) 
  {
    // get data and check for termination request
    //定义 std::vector<std::shared_ptr< okvis::threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > > >  cameraMeasurementsReceived_;
    //从图像队列中提取出图像并保存在frame中
     if (cameraMeasurementsReceived_[cameraIndex]->PopBlocking(&frame) == false)//搜索 bool PopBlocking(QueueType* value) 
	{
      return;
    }
    //beforeDetectTimer.start();//什么都没做，为了方便看代码我们把这个注释掉
    {  // lock the frame synchronizer
      //waitForFrameSynchronizerMutexTimer.start();//什么都没做，为了方便看代码我们把这个注释掉
      
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);//frameSynchronizer_mutex_整个变量就只在这个函数中被使用了
      
      //waitForFrameSynchronizerMutexTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
      // add new frame to frame synchronizer and get the MultiFrame containing it
      //addNewFrameToSynchronizerTimer.start();//什么都没做，为了方便看代码我们把这个注释掉
      
      multiFrame = frameSynchronizer_.addNewFrame(frame);//搜索 FrameSynchronizer::addNewFrame
      
      //addNewFrameToSynchronizerTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
    }  // unlock frameSynchronizer only now as we can be sure that not two states are added for the same timestamp
    okvis::kinematics::Transformation T_WS;
    okvis::Time lastTimestamp;
    okvis::SpeedAndBias speedAndBiases;
    // copy last state variables
    {
      //waitForStateVariablesMutexTimer.start();//什么都没做，为了方便看代码我们把这个注释掉
      std::lock_guard<std::mutex> lock(lastState_mutex_);
      //waitForStateVariablesMutexTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
      T_WS = lastOptimized_T_WS_;//最后一次优化后的位姿结果
      speedAndBiases = lastOptimizedSpeedAndBiases_;//最后一次优化后的速度和偏差
      lastTimestamp = lastOptimizedStateTimestamp_;//最后一次优化后的时间戳
    }

    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime = multiFrame->timestamp()+ temporal_imu_data_overlap;//temporal_imu_data_overlap默认设置=0.02秒，euroc使用的imu频率是0.05秒
    okvis::Time imuDataBeginTime = lastTimestamp - temporal_imu_data_overlap;

    OKVIS_ASSERT_TRUE_DBG(Exception,imuDataBeginTime < imuDataEndTime,"imu data end time is smaller than begin time.");

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(okvis::Time(imuDataEndTime)) == false)  
	{
      return;
    }
    OKVIS_ASSERT_TRUE_DBG(Exception,
                          imuDataEndTime < imuMeasurements_.back().timeStamp,
                          "Waiting for up to date imu data seems to have failed!");
    //搜索 okvis::ImuMeasurementDeque ThreadedKFVio::getImuMeasurments(
    //从imuMeasurements_中截取数据
    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,imuDataEndTime);//获得图像之间的imu测量值

    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0) {
      //beforeDetectTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
      continue;
    }

    if (imuData.front().timeStamp > frame->timeStamp) {
      LOG(WARNING) << "Frame is newer than oldest IMU measurement. Dropping it.";
      //beforeDetectTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
      continue;
    }

    // get T_WC(camIndx) for detectAndDescribe()
    if (estimator_.numFrames() == 0) //状态求解器中目前还没有相机的的状态，那么刚开始到第一帧相机的姿态是由imu得到的
	{
      // first frame ever
      //主要是为了计算T_WS中的初始姿态
      bool success = okvis::Estimator::initPoseFromImu(imuData,T_WS);//比较重要的函数!!!!!!!!!!!!
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        lastOptimized_T_WS_ = T_WS;//非常重要 设置了初始第一帧的相机在世界坐标系下的姿态
        lastOptimizedSpeedAndBiases_.setZero();//上一时刻的bias设置为0
        lastOptimizedSpeedAndBiases_.segment<3>(6) = imu_params_.a0;//imu_params_.a0是加速度的平均bias，euroc设置是0,0,0
        lastOptimizedStateTimestamp_ = multiFrame->timestamp();
      }
      OKVIS_ASSERT_TRUE_DBG(Exception, success,"pose could not be initialized from imu measurements.");
      if (!success) {
        //beforeDetectTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
        continue;
      }
    } 
	else 
    {
      // get old T_WS
      //propagationTimer.start();//什么都没做，为了方便看代码我们把这个注释掉
      //非常重要的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //为了根据imu测量值和上一时刻的位姿计算得到 这一时刻的T_WS和speedAndBiases
      okvis::ceres::ImuError::propagation(imuData, parameters_.imu, T_WS,speedAndBiases, lastTimestamp,multiFrame->timestamp());
      //propagationTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
    }
	//根据imu坐标系到世界坐标系的变换求取相机坐标系到世界坐标系的转换
    okvis::kinematics::Transformation T_WC = T_WS * (*parameters_.nCameraSystem.T_SC(frame->sensorId));
    //beforeDetectTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
    //detectTimer.start();//什么都没做，为了方便看代码我们把这个注释掉
    //比较重要的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    frontend_.detectAndDescribe(frame->sensorId, multiFrame, T_WC, nullptr);//特征提取左/右图像特征 搜索 Frontend::detectAndDescribe
    //detectTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
    //afterDetectTimer.start();//什么都没做，为了方便看代码我们把这个注释掉

    bool push = false;
    {  // we now tell frame synchronizer that detectAndDescribe is done for MF with our timestamp
      //waitForFrameSynchronizerMutexTimer2.start();//什么都没做，为了方便看代码我们把这个注释掉
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      //waitForFrameSynchronizerMutexTimer2.stop();//什么都没做，为了方便看代码我们把这个注释掉
      frameSynchronizer_.detectionEndedForMultiFrame(multiFrame->id());//搜索 FrameSynchronizer::detectionEndedForMultiFrame(

      if (frameSynchronizer_.detectionCompletedForAllCameras( multiFrame->id()))
	  {
//        LOG(INFO) << "detection completed for multiframe with id "<< multi_frame->id();
        push = true;
      }
    }  // unlocking frame synchronizer
    //afterDetectTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
	//push为真，表示左右目图像都完成了特征提取
    if (push)
	{
      // use queue size 1 to propagate a congestion to the _cameraMeasurementsReceived queue
      // and check for termination request
      //waitForMatchingThreadTimer.start();//什么都没做，为了方便看代码我们把这个注释掉
      
	   //将multiFrame添加到keypointMeasurements_的队列中
      ///ThreadSafeQueue<std::shared_ptr<okvis::MultiFrame> > keypointMeasurements_；特征点队列
      //keypointMeasurements_在整个代码范围内 只有在这里被更新了
      if (keypointMeasurements_.PushBlockingIfFull(multiFrame, 1) == false) //更新队列信息!!!!!!!!!!!!!!!!!!!!!!!!!!
	  {
        return;
      }
      //waitForMatchingThreadTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
    }
  }//大的for循环结束
}

// Loop that matches frames with existing frames.
// 帧间匹配线程
void ThreadedKFVio::matchingLoop() {

	//声明这四个变量完全是空的什么都不做
  TimerSwitchable prepareToAddStateTimer("2.1 prepareToAddState",true);
  TimerSwitchable waitForOptimizationTimer("2.2 waitForOptimization",true);
  TimerSwitchable addStateTimer("2.3 addState",true);
  TimerSwitchable matchingTimer("2.4 matching",true);

  for (;;) 
  {
    // get new frame
    std::shared_ptr<okvis::MultiFrame> frame;

    // get data and check for termination request
    //从特征点队列中弹出数据，特征点队列中的数据是已经提取过特征点和描述子的双目相机结构
    if (keypointMeasurements_.PopBlocking(&frame) == false)
      return;

    //prepareToAddStateTimer.start();//什么都没做，为了方便看代码我们把这个注释掉
    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime = frame->timestamp() + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastAddedStateTimestamp_- temporal_imu_data_overlap;

    OKVIS_ASSERT_TRUE_DBG(Exception,imuDataBeginTime < imuDataEndTime,
        "imu data end time is smaller than begin time." <<
        "current frametimestamp " << frame->timestamp() << " (id: " << frame->id() <<
        "last timestamp         " << lastAddedStateTimestamp_ << " (id: " << estimator_.currentFrameId());

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(okvis::Time(imuDataEndTime)) == false)
      return; OKVIS_ASSERT_TRUE_DBG(Exception,
        imuDataEndTime < imuMeasurements_.back().timeStamp,
        "Waiting for up to date imu data seems to have failed!");
    //1.获取当前帧到这一帧的imu测量数据
    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,
                                                           imuDataEndTime);

    //prepareToAddStateTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0)
      continue;

    // make sure that optimization of last frame is over.
    // TODO If we didn't actually 'pop' the _matchedFrames queue until after optimization this would not be necessary
    {
      //waitForOptimizationTimer.start();//什么都没做，为了方便看代码我们把这个注释掉
      std::unique_lock<std::mutex> l(estimator_mutex_);
	  //2.判断ceres优化是否完成 如果完成继续执行
      while (!optimizationDone_)
        optimizationNotification_.wait(l);
      //waitForOptimizationTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
      //addStateTimer.start();//什么都没做，为了方便看代码我们把这个注释掉
      okvis::Time t0Matching = okvis::Time::now();
      bool asKeyframe = false;
	  //3.向ceres添加参数-这一帧的相机和imu测量值
      if (estimator_.addStates(frame, imuData, asKeyframe))//重要的一步!!!!!!!!向ceres添加参数，搜索 bool Estimator::addStates(
	  {
        lastAddedStateTimestamp_ = frame->timestamp();
        //addStateTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
      } else {
        LOG(ERROR) << "Failed to add state! will drop multiframe.";
        //addStateTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
        continue;
      }

      // -- matching keypoints, initialising landmarks etc.
      //4.从ceres优化的参数中得到T_WS
      okvis::kinematics::Transformation T_WS;
      estimator_.get_T_WS(frame->id(), T_WS);//这个应该是从优化的变量中提取出现在的位姿
      //matchingTimer.start();//什么都没做，为了方便看代码我们把这个注释掉
      //匹配函数+地标点初始化创建函数+RANSAC函数+判断关键帧函数
      //搜索 bool Frontend::dataAssociationAndInitialization(
      //5.进行匹配 ,同时给出判断这一帧是否为关键帧
      frontend_.dataAssociationAndInitialization(estimator_, T_WS, parameters_,map_, frame, &asKeyframe);//非常重要的函数!!!!!!!!!!!!!!!!!!!!!!
      //matchingTimer.stop();//什么都没做，为了方便看代码我们把这个注释掉
      //6.设置关键帧
      if (asKeyframe)
        estimator_.setKeyframe(frame->id(), asKeyframe);//向地图中插入关键帧只是设置了一个标志位，搜索 void setKeyframe(uint64_t frameId, bool isKeyframe)
      if(!blocking_)//默认进入这个条件
	  {
	     //euroc timeLimitForMatchingAndOptimization=0.035秒
        double timeLimit = parameters_.optimization.timeLimitForMatchingAndOptimization-(okvis::Time::now()-t0Matching).toSec();
		//搜索bool Estimator::setOptimizationTimeLimit(double timeLimit, int minIterations) {
        estimator_.setOptimizationTimeLimit(std::max<double>(0.0, timeLimit),parameters_.optimization.min_iterations);//euroc 设置min_iterations=3
      }
      optimizationDone_ = false;
    }  // unlock estimator_mutex_

    // use queue size 1 to propagate a congestion to the _matchedFrames queue
    //6.将得到的匹配过的双目帧结构压入到匹配队列中
    if (matchedFrames_.PushBlockingIfFull(frame, 1) == false)
      return;
  }
}

// Loop to process IMU measurements.
//根据imu的测量值更新当前状态然后将这个状态压入到optimizationResults_队列中
void ThreadedKFVio::imuConsumerLoop() 
{
  okvis::ImuMeasurement data;
  TimerSwitchable processImuTimer("0 processImuMeasurements",true);//这个什么都没干
  for (;;) 
  {
    // get data and check for termination request
    if (imuMeasurementsReceived_.PopBlocking(&data) == false)//从队列中弹出测量值
      return;
    //processImuTimer.start();//什么都没干 这里注释掉方便看代码
    okvis::Time start;
    const okvis::Time* end;  // do not need to copy end timestamp
    {
      std::lock_guard<std::mutex> imuLock(imuMeasurements_mutex_);
      OKVIS_ASSERT_TRUE(Exception,
                        imuMeasurements_.empty()
                        || imuMeasurements_.back().timeStamp < data.timeStamp,
                        "IMU measurement from the past received");

      if (parameters_.publishing.publishImuPropagatedState)//这个参数默认是true
	  {
        if (!repropagationNeeded_ && imuMeasurements_.size() > 0) //repropagationNeeded_ 这个变量在优化线程中被赋值为true了，边缘化之后这变量会被赋值为true
		{
          start = imuMeasurements_.back().timeStamp;
        } else if (repropagationNeeded_)//进入这个条件表示需要重新propagation一遍
        {
          std::lock_guard<std::mutex> lastStateLock(lastState_mutex_);
          start = lastOptimizedStateTimestamp_;
          T_WS_propagated_ = lastOptimized_T_WS_;
          speedAndBiases_propagated_ = lastOptimizedSpeedAndBiases_;
          repropagationNeeded_ = false;
        } else
          start = okvis::Time(0, 0);
		
        end = &data.timeStamp;
      }
      imuMeasurements_.push_back(data);//在优化的线程中会清空imuMeasurements_
    }  // unlock _imuMeasurements_mutex

    // notify other threads that imu data with timeStamp is here.
    imuFrameSynchronizer_.gotImuData(data.timeStamp);//告诉系统得到了一个数据

    if (parameters_.publishing.publishImuPropagatedState)//这个参数默认是true
	{
      Eigen::Matrix<double, 15, 15> covariance;
      Eigen::Matrix<double, 15, 15> jacobian;

	  //搜索 Frontend::propagation
	  //其实这个函数本质上还是调用的ImuError::propagation函数
	  //注意了 这里的雅克比和协方差都没有用到
      frontend_.propagation(imuMeasurements_, imu_params_, T_WS_propagated_,
                            speedAndBiases_propagated_, start, *end, &covariance,
                            &jacobian);
      OptimizationResults result;
      result.stamp = *end;
      result.T_WS = T_WS_propagated_;
      result.speedAndBiases = speedAndBiases_propagated_;
      result.omega_S = imuMeasurements_.back().measurement.gyroscopes- speedAndBiases_propagated_.segment<3>(3);
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) //遍历左右相机
	  {
        result.vector_of_T_SCi.push_back(okvis::kinematics::Transformation(*parameters_.nCameraSystem.T_SC(i)));//相机到imu的固定位姿变化
      }
      result.onlyPublishLandmarks = false;
      optimizationResults_.PushNonBlockingDroppingIfFull(result,1);
    }
    //processImuTimer.stop();//什么都没干 这里注释掉方便看代码
  }
}

// Loop to process position measurements.
void ThreadedKFVio::positionConsumerLoop()
{
  okvis::PositionMeasurement data;
  for (;;) 
  {
    // get data and check for termination request
    if (positionMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    // collect
    {
      std::lock_guard<std::mutex> positionLock(positionMeasurements_mutex_);
      positionMeasurements_.push_back(data);
    }
  }
}

// Loop to process GPS measurements.
void ThreadedKFVio::gpsConsumerLoop() {
}

// Loop to process magnetometer measurements.
void ThreadedKFVio::magnetometerConsumerLoop() {
}

// Loop to process differential pressure measurements.
void ThreadedKFVio::differentialConsumerLoop() {
}

// Loop that visualizes completed frames.
void ThreadedKFVio::visualizationLoop() 
{
  okvis::VioVisualizer visualizer_(parameters_);
  for (;;) 
  {
    VioVisualizer::VisualizationData::Ptr new_data;
    if (visualizationData_.PopBlocking(&new_data) == false)
      return;
    //visualizer_.showDebugImages(new_data);
    std::vector<cv::Mat> out_images(parameters_.nCameraSystem.numCameras());//存储的是左右两个图像
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) 
	{
      out_images[i] = visualizer_.drawMatches(new_data, i);//搜索 cv::Mat VioVisualizer::drawMatches
    }
	displayImages_.PushNonBlockingDroppingIfFull(out_images,1);//线程队列相关的操作
  }
}

// trigger display (needed because OSX won't allow threaded display)
void ThreadedKFVio::display() 
{
  std::vector<cv::Mat> out_images;
  if (displayImages_.Size() == 0)
	return;
  if (displayImages_.PopBlocking(&out_images) == false)
    return;
  // draw
  for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) 
  {
    std::stringstream windowname;
    windowname << "OKVIS camera " << im;
    cv::imshow(windowname.str(), out_images[im]);
  }
  cv::waitKey(1);
}

// Get a subset of the recorded IMU measurements.
okvis::ImuMeasurementDeque ThreadedKFVio::getImuMeasurments(kvis::Time& imuDataBeginTime, okvis::Time& imuDataEndTime) 
{
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest imu time, return empty queue.
  if (imuDataEndTime < imuDataBeginTime || imuDataBeginTime > imuMeasurements_.back().timeStamp)
    return okvis::ImuMeasurementDeque();

  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  // get iterator to imu data before previous frame
  okvis::ImuMeasurementDeque::iterator first_imu_package = imuMeasurements_.begin();
  okvis::ImuMeasurementDeque::iterator last_imu_package =  imuMeasurements_.end();
  // TODO go backwards through queue. Is probably faster.
  for (auto iter = imuMeasurements_.begin(); iter != imuMeasurements_.end();++iter) 
  {
    // move first_imu_package iterator back until iter->timeStamp is higher than requested begintime
    if (iter->timeStamp <= imuDataBeginTime)
      first_imu_package = iter;

    // set last_imu_package iterator as soon as we hit first timeStamp higher than requested endtime & break
    if (iter->timeStamp >= imuDataEndTime) {
      last_imu_package = iter;
      // since we want to include this last imu measurement in returned Deque we
      // increase last_imu_package iterator once.
      ++last_imu_package;
      break;
    }
  }

  // create copy of imu buffer
  return okvis::ImuMeasurementDeque(first_imu_package, last_imu_package);
}

// Remove IMU measurements from the internal buffer.
int ThreadedKFVio::deleteImuMeasurements(const okvis::Time& eraseUntil) 
{
  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  if (imuMeasurements_.front().timeStamp > eraseUntil)
    return 0;

  okvis::ImuMeasurementDeque::iterator eraseEnd;
  int removed = 0;
  for (auto it = imuMeasurements_.begin(); it != imuMeasurements_.end(); ++it) {
    eraseEnd = it;
    if (it->timeStamp >= eraseUntil)
      break;
    ++removed;
  }

  imuMeasurements_.erase(imuMeasurements_.begin(), eraseEnd);

  return removed;
}

// Loop that performs the optimization and marginalisation.
//优化和边缘化线程
void ThreadedKFVio::optimizationLoop() 
{
  TimerSwitchable optimizationTimer("3.1 optimization",true);//是个空函数什么都没做
  TimerSwitchable marginalizationTimer("3.2 marginalization",true);//是个空函数什么都没做
  TimerSwitchable afterOptimizationTimer("3.3 afterOptimization",true);

  for (;;)
  {
    std::shared_ptr<okvis::MultiFrame> frame_pairs;
    VioVisualizer::VisualizationData::Ptr visualizationDataPtr;
    okvis::Time deleteImuMeasurementsUntil(0, 0);
	//1.从队列中弹出已经匹配的帧，这里我们成为当前帧（经过了match线程） 类型okvis::MultiFrame
    if (matchedFrames_.PopBlocking(&frame_pairs) == false)
      return;
    OptimizationResults result;//结构类型 搜索  struct OptimizationResults {
    {
      std::lock_guard<std::mutex> l(estimator_mutex_);
      //optimizationTimer.start();//什么都没做 方便看代码
      //if(frontend_.isInitialized()){
      //搜索 void Estimator::optimize(
      //重要的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       //2.开始优化 并更新地图点的信息
        estimator_.optimize(parameters_.optimization.max_iterations, 2, false);//euroc设置max_iterations=10
      //}
      /*if (estimator_.numFrames() > 0 && !frontend_.isInitialized()){
        // undo translation
        for(size_t n=0; n<estimator_.numFrames(); ++n){
          okvis::kinematics::Transformation T_WS_0;
          estimator_.get_T_WS(estimator_.frameIdByAge(n),T_WS_0);
          Eigen::Matrix4d T_WS_0_mat = T_WS_0.T();
          T_WS_0_mat.topRightCorner<3,1>().setZero();
          estimator_.set_T_WS(estimator_.frameIdByAge(n),okvis::kinematics::Transformation(T_WS_0_mat));
          okvis::SpeedAndBias sb_0 = okvis::SpeedAndBias::Zero();
          if(estimator_.getSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0)){
            sb_0.head<3>().setZero();
            estimator_.setSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0);
          }
        }
      }*/

      //optimizationTimer.stop();//什么都没做 方便看代码

      // get timestamp of last frame in IMU window. Need to do this before marginalization as it will be removed there (if not keyframe)
      if (estimator_.numFrames()> size_t(parameters_.optimization.numImuFrames))//numImuFrames euroc设置=3
	  {
	    //默认temporal_imu_data_overlap默认设置=0.02秒，euroc使用的imu频率是0.05秒
        deleteImuMeasurementsUntil = estimator_.multiFrame(estimator_.frameIdByAge(parameters_.optimization.numImuFrames))->timestamp() - temporal_imu_data_overlap;
      }

      //marginalizationTimer.start();//什么都没做 方便看代码
      //numKeyframes=默认设置5
      //重要的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //搜索 Estimator::applyMarginalizationStrategy(
      estimator_.applyMarginalizationStrategy(
          parameters_.optimization.numKeyframes,
          parameters_.optimization.numImuFrames, result.transferredLandmarks);//transferredLandmarks=得到要被边缘化的地图点
      //marginalizationTimer.stop();//什么都没做 方便看代码
      //afterOptimizationTimer.start();//什么都没做 方便看代码

      // now actually remove measurements
      //从 imuMeasurements_中删除deleteImuMeasurementsUntil时间戳之前的imu的观测值
      deleteImuMeasurements(deleteImuMeasurementsUntil);
      // saving optimized state and saving it in OptimizationResults struct
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        estimator_.get_T_WS(frame_pairs->id(), lastOptimized_T_WS_);//得到优化之后的当前帧的姿态 存储到lastOptimized_T_WS_
        estimator_.getSpeedAndBias(frame_pairs->id(), 0,lastOptimizedSpeedAndBiases_);//得到优化之后的当前帧的速度和bias 存储到lastOptimizedSpeedAndBiases_
        lastOptimizedStateTimestamp_ = frame_pairs->timestamp();

        // if we publish the state after each IMU propagation we do not need to publish it here.
        if (!parameters_.publishing.publishImuPropagatedState) //publishImuPropagatedState=默认为true
		{
          result.T_WS = lastOptimized_T_WS_;
          result.speedAndBiases = lastOptimizedSpeedAndBiases_;
          result.stamp = lastOptimizedStateTimestamp_;
          result.onlyPublishLandmarks = false;
        }
        else //默认进入这个条件
          result.onlyPublishLandmarks = true;

		//得到landmarksMap_结构中所有的地图点 并保存在result.landmarksVector结构中
        estimator_.getLandmarks(result.landmarksVector);//搜索 Estimator::getLandmarks(MapPointVector & landmarks) const

        repropagationNeeded_ = true;//因为状态被更新了因此这里我们设置标志位 意思是需要对imu重新进行propagation
      }

	  //我们把可视化的部分注释掉 方便看代码
      if (parameters_.visualization.displayImages) //euroc设置为true
	  {
        // fill in information that requires access to estimator.
        visualizationDataPtr = VioVisualizer::VisualizationData::Ptr(
            new VioVisualizer::VisualizationData());
        visualizationDataPtr->observations.resize(frame_pairs->numKeypoints());
        okvis::MapPoint landmark;
        okvis::ObservationVector::iterator it = visualizationDataPtr
            ->observations.begin();
        for (size_t camIndex = 0; camIndex < frame_pairs->numFrames();
            ++camIndex) {
          for (size_t k = 0; k < frame_pairs->numKeypoints(camIndex); ++k) {
            OKVIS_ASSERT_TRUE_DBG(Exception,it != visualizationDataPtr->observations.end(),"Observation-vector not big enough");
            it->keypointIdx = k;
            frame_pairs->getKeypoint(camIndex, k, it->keypointMeasurement);
            frame_pairs->getKeypointSize(camIndex, k, it->keypointSize);
            it->cameraIdx = camIndex;
            it->frameId = frame_pairs->id();
            it->landmarkId = frame_pairs->landmarkId(camIndex, k);
            if (estimator_.isLandmarkAdded(it->landmarkId)) {
              estimator_.getLandmark(it->landmarkId, landmark);
              it->landmark_W = landmark.point;
              if (estimator_.isLandmarkInitialized(it->landmarkId))
                it->isInitialized = true;
              else
                it->isInitialized = false;
            } else {
              it->landmark_W = Eigen::Vector4d(0, 0, 0, 0);  // set to infinity to tell visualizer that landmark is not added
            }
            ++it;
          }
        }
        visualizationDataPtr->keyFrames = estimator_.multiFrame(
            estimator_.currentKeyframeId());
        estimator_.get_T_WS(estimator_.currentKeyframeId(),
                            visualizationDataPtr->T_WS_keyFrame);
      }
      optimizationDone_ = true;
    }  // unlock mutex
    optimizationNotification_.notify_all();//std::condition_variable optimizationNotification_;

	/*注释掉不会进入的条件，方便看代码
    if (!parameters_.publishing.publishImuPropagatedState) //默认不进入这个条件，publishImuPropagatedState=默认为true
	{
      // adding further elements to result that do not access estimator.
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        result.vector_of_T_SCi.push_back( okvis::kinematics::Transformation(*parameters_.nCameraSystem.T_SC(i)));
      }
    }*/
    optimizationResults_.Push(result);//将结果压入到优化队列中

    // adding further elements to visualization data that do not access estimator
    if (parameters_.visualization.displayImages)//euroc设置为true
	{
      visualizationDataPtr->currentFrames = frame_pairs;
      visualizationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr, 1);
    }
    //afterOptimizationTimer.stop();//什么都没做 方便看代码
  }
}

// Loop that publishes the newest state and landmarks.
void ThreadedKFVio::publisherLoop() {
  for (;;) 
  {
    // get the result data
    OptimizationResults result;
    if (optimizationResults_.PopBlocking(&result) == false)//如果优化队列没有输出信息则直接返回
      return;

    // call all user callbacks
    //我们把作者没有定义的函数注释掉 方便看代码
    /*
    if (stateCallback_ && !result.onlyPublishLandmarks)
      stateCallback_(result.stamp, result.T_WS);
    */
    
    if (fullStateCallback_ && !result.onlyPublishLandmarks)
    {
       //定义 PoseViewer::publishFullStateAsCallback = fullStateCallback_
       //我们可以看到优化队列中的值由如下构成:时间戳，机器人在世界坐标系下的位置，速度和角速度的bias和加速度的bias，角速度
      fullStateCallback_(result.stamp, result.T_WS, result.speedAndBiases,result.omega_S);
    }
    /*
	if (fullStateCallbackWithExtrinsics_ && !result.onlyPublishLandmarks)
      fullStateCallbackWithExtrinsics_(result.stamp, result.T_WS,result.speedAndBiases, result.omega_S,result.vector_of_T_SCi);
    if (landmarksCallback_ && !result.landmarksVector.empty())
      landmarksCallback_(result.stamp, result.landmarksVector,result.transferredLandmarks);  //TODO(gohlp): why two maps?
      */
  }
}

}  // namespace okvis
