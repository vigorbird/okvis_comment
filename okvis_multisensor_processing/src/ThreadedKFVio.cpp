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
// Constructor. ��Ĭ����������� ThreadedKFVio���캯��
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

  frontend_.setBriskDetectionOctaves(parameters_.optimization.detectionOctaves);//euroc����=0
  frontend_.setBriskDetectionThreshold(parameters_.optimization.detectionThreshold);//euroc����=40
  frontend_.setBriskDetectionMaximumKeypoints(parameters_.optimization.maxNoKeypoints);//euroc����=400 ÿ��ͼ����������������

  lastOptimizedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)
  lastAddedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)

  //����imu�Ĳ���
  //���������о͵��ù�һ���������
  estimator_.addImu(parameters_.imu);//���� int Estimator::addImu(const ImuParameters & imuParameters)
  for (size_t i = 0; i < numCameras_; ++i) //��������ͼ��
  {
    // parameters_.camera_extrinsics is never set (default 0's)...
    // do they ever change?
    //������/������Ĳ���
    //���������������ù�һ��
    estimator_.addCamera(parameters_.camera_extrinsics);//���� int Estimator::addCamera(const ExtrinsicsEstimationParameters & extrinsicsEstimationParameters)

    //����������ն��е�ʵ��
	cameraMeasurementsReceived_.emplace_back(std::shared_ptr<threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > >(new threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> >()));
  }
  
  // set up windows so things don't crash on Mac OS
  if(parameters_.visualization.displayImages)//euroc����Ϊtrue
  {
    for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) 
	{
      std::stringstream windowname;
      windowname << "OKVIS camera " << im;
  	  cv::namedWindow(windowname.str());
    }
  }
  
  startThreads();//�ǳ���Ҫ�ĺ�����������
}

// Start all threads.
void ThreadedKFVio::startThreads() 
{

  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.emplace_back(&ThreadedKFVio::frameConsumerLoop, this, i);//���Ӧ��Ҳ�ǿ����߳� ����ͼ���һ���߳�  ������
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.emplace_back(&ThreadedKFVio::matchingLoop, this);//���Ӧ��Ҳ�ǿ����߳� ˫ĿĬ��ֻ����һ��
  }
  imuConsumerThread_ = std::thread(&ThreadedKFVio::imuConsumerLoop, this);//������һ���߳� ����ThreadedKFVio::imuConsumerLoop
  positionConsumerThread_ = std::thread(&ThreadedKFVio::positionConsumerLoop,this);//������һ���߳� ����void ThreadedKFVio::positionConsumerLoop()
  gpsConsumerThread_ = std::thread(&ThreadedKFVio::gpsConsumerLoop, this);//���ù� �Ǹ����߳�
  magnetometerConsumerThread_ = std::thread( &ThreadedKFVio::magnetometerConsumerLoop, this);//���ù� �Ǹ����߳�
  differentialConsumerThread_ = std::thread( &ThreadedKFVio::differentialConsumerLoop, this);//���ù� �Ǹ����߳�

  // algorithm threads
  visualizationThread_ = std::thread(&ThreadedKFVio::visualizationLoop, this);//������һ���߳� ���� ThreadedKFVio::visualizationLoop()
  optimizationThread_ = std::thread(&ThreadedKFVio::optimizationLoop, this);//������һ���߳� ���� ThreadedKFVio::optimizationLoop
  publisherThread_ = std::thread(&ThreadedKFVio::publisherLoop, this);//������һ���߳� ���� ThreadedKFVio::publisherLoop() 
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
    frameConsumerThreads_.at(i).join();//���������ڵ�ǰ�̵߳ȴ��߳����н�����
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
//keypointsĬ��ֵ= 0
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

  if (blocking_) //Ĭ�Ͻ����������
  {
    //����: std::vector<std::shared_ptr< okvis::threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > > > cameraMeasurementsReceived_;
    //���� bool PushBlockingIfFull(const QueueType& value, size_t max_queue_size) 
    cameraMeasurementsReceived_[cameraIndex]->PushBlockingIfFull(frame, 1);//����/����������в�������õ���ͼ��1��ʾ��������������ͼ����ȴ�
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

  if (blocking_)//Ĭ���ǽ������������
  {
    //���� okvis::threadsafe::ThreadSafeQueue<okvis::ImuMeasurement> imuMeasurementsReceived_;
    //���� bool PushBlockingIfFull(const QueueType& value, size_t max_queue_size) {
    //���imu����ֵ�����в���ֵ��������1��ȴ���ֱ�����Բ������
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
void ThreadedKFVio::setBlocking(bool blocking) //��main�����б�����
{
  blocking_ = blocking;//
  // disable time limit for optimization
  if(blocking_) 
  {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
	//ȫ������ fyy Estimator::setOptimizationTimeLimit
    estimator_.setOptimizationTimeLimit(-1.0,parameters_.optimization.max_iterations);
  }
}

// Loop to process frames from camera with index cameraIndex
//cameraIndex��ʾ����������������
void ThreadedKFVio::frameConsumerLoop(size_t cameraIndex)
{
  std::shared_ptr<okvis::CameraMeasurement> frame;
  std::shared_ptr<okvis::MultiFrame> multiFrame;
  //�������ĸ�������ȫ�ǿյ�ʲô������
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
    //���� std::vector<std::shared_ptr< okvis::threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > > >  cameraMeasurementsReceived_;
    //��ͼ���������ȡ��ͼ�񲢱�����frame��
     if (cameraMeasurementsReceived_[cameraIndex]->PopBlocking(&frame) == false)//���� bool PopBlocking(QueueType* value) 
	{
      return;
    }
    //beforeDetectTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
    {  // lock the frame synchronizer
      //waitForFrameSynchronizerMutexTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);//frameSynchronizer_mutex_����������ֻ����������б�ʹ����
      
      //waitForFrameSynchronizerMutexTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      // add new frame to frame synchronizer and get the MultiFrame containing it
      //addNewFrameToSynchronizerTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      
      multiFrame = frameSynchronizer_.addNewFrame(frame);//���� FrameSynchronizer::addNewFrame
      
      //addNewFrameToSynchronizerTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
    }  // unlock frameSynchronizer only now as we can be sure that not two states are added for the same timestamp
    okvis::kinematics::Transformation T_WS;
    okvis::Time lastTimestamp;
    okvis::SpeedAndBias speedAndBiases;
    // copy last state variables
    {
      //waitForStateVariablesMutexTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      std::lock_guard<std::mutex> lock(lastState_mutex_);
      //waitForStateVariablesMutexTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      T_WS = lastOptimized_T_WS_;//���һ���Ż����λ�˽��
      speedAndBiases = lastOptimizedSpeedAndBiases_;//���һ���Ż�����ٶȺ�ƫ��
      lastTimestamp = lastOptimizedStateTimestamp_;//���һ���Ż����ʱ���
    }

    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime = multiFrame->timestamp()+ temporal_imu_data_overlap;//temporal_imu_data_overlapĬ������=0.02�룬eurocʹ�õ�imuƵ����0.05��
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
    //���� okvis::ImuMeasurementDeque ThreadedKFVio::getImuMeasurments(
    //��imuMeasurements_�н�ȡ����
    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,imuDataEndTime);//���ͼ��֮���imu����ֵ

    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0) {
      //beforeDetectTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      continue;
    }

    if (imuData.front().timeStamp > frame->timeStamp) {
      LOG(WARNING) << "Frame is newer than oldest IMU measurement. Dropping it.";
      //beforeDetectTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      continue;
    }

    // get T_WC(camIndx) for detectAndDescribe()
    if (estimator_.numFrames() == 0) //״̬�������Ŀǰ��û������ĵ�״̬����ô�տ�ʼ����һ֡�������̬����imu�õ���
	{
      // first frame ever
      //��Ҫ��Ϊ�˼���T_WS�еĳ�ʼ��̬
      bool success = okvis::Estimator::initPoseFromImu(imuData,T_WS);//�Ƚ���Ҫ�ĺ���!!!!!!!!!!!!
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        lastOptimized_T_WS_ = T_WS;//�ǳ���Ҫ �����˳�ʼ��һ֡���������������ϵ�µ���̬
        lastOptimizedSpeedAndBiases_.setZero();//��һʱ�̵�bias����Ϊ0
        lastOptimizedSpeedAndBiases_.segment<3>(6) = imu_params_.a0;//imu_params_.a0�Ǽ��ٶȵ�ƽ��bias��euroc������0,0,0
        lastOptimizedStateTimestamp_ = multiFrame->timestamp();
      }
      OKVIS_ASSERT_TRUE_DBG(Exception, success,"pose could not be initialized from imu measurements.");
      if (!success) {
        //beforeDetectTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
        continue;
      }
    } 
	else 
    {
      // get old T_WS
      //propagationTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      //�ǳ���Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //Ϊ�˸���imu����ֵ����һʱ�̵�λ�˼���õ� ��һʱ�̵�T_WS��speedAndBiases
      okvis::ceres::ImuError::propagation(imuData, parameters_.imu, T_WS,speedAndBiases, lastTimestamp,multiFrame->timestamp());
      //propagationTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
    }
	//����imu����ϵ����������ϵ�ı任��ȡ�������ϵ����������ϵ��ת��
    okvis::kinematics::Transformation T_WC = T_WS * (*parameters_.nCameraSystem.T_SC(frame->sensorId));
    //beforeDetectTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
    //detectTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
    //�Ƚ���Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    frontend_.detectAndDescribe(frame->sensorId, multiFrame, T_WC, nullptr);//������ȡ��/��ͼ������ ���� Frontend::detectAndDescribe
    //detectTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
    //afterDetectTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�

    bool push = false;
    {  // we now tell frame synchronizer that detectAndDescribe is done for MF with our timestamp
      //waitForFrameSynchronizerMutexTimer2.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      //waitForFrameSynchronizerMutexTimer2.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      frameSynchronizer_.detectionEndedForMultiFrame(multiFrame->id());//���� FrameSynchronizer::detectionEndedForMultiFrame(

      if (frameSynchronizer_.detectionCompletedForAllCameras( multiFrame->id()))
	  {
//        LOG(INFO) << "detection completed for multiframe with id "<< multi_frame->id();
        push = true;
      }
    }  // unlocking frame synchronizer
    //afterDetectTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
	//pushΪ�棬��ʾ����Ŀͼ�������������ȡ
    if (push)
	{
      // use queue size 1 to propagate a congestion to the _cameraMeasurementsReceived queue
      // and check for termination request
      //waitForMatchingThreadTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      
	   //��multiFrame��ӵ�keypointMeasurements_�Ķ�����
      ///ThreadSafeQueue<std::shared_ptr<okvis::MultiFrame> > keypointMeasurements_�����������
      //keypointMeasurements_���������뷶Χ�� ֻ�������ﱻ������
      if (keypointMeasurements_.PushBlockingIfFull(multiFrame, 1) == false) //���¶�����Ϣ!!!!!!!!!!!!!!!!!!!!!!!!!!
	  {
        return;
      }
      //waitForMatchingThreadTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
    }
  }//���forѭ������
}

// Loop that matches frames with existing frames.
// ֡��ƥ���߳�
void ThreadedKFVio::matchingLoop() {

	//�������ĸ�������ȫ�ǿյ�ʲô������
  TimerSwitchable prepareToAddStateTimer("2.1 prepareToAddState",true);
  TimerSwitchable waitForOptimizationTimer("2.2 waitForOptimization",true);
  TimerSwitchable addStateTimer("2.3 addState",true);
  TimerSwitchable matchingTimer("2.4 matching",true);

  for (;;) 
  {
    // get new frame
    std::shared_ptr<okvis::MultiFrame> frame;

    // get data and check for termination request
    //������������е������ݣ�����������е��������Ѿ���ȡ��������������ӵ�˫Ŀ����ṹ
    if (keypointMeasurements_.PopBlocking(&frame) == false)
      return;

    //prepareToAddStateTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
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
    //1.��ȡ��ǰ֡����һ֡��imu��������
    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,
                                                           imuDataEndTime);

    //prepareToAddStateTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0)
      continue;

    // make sure that optimization of last frame is over.
    // TODO If we didn't actually 'pop' the _matchedFrames queue until after optimization this would not be necessary
    {
      //waitForOptimizationTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      std::unique_lock<std::mutex> l(estimator_mutex_);
	  //2.�ж�ceres�Ż��Ƿ���� �����ɼ���ִ��
      while (!optimizationDone_)
        optimizationNotification_.wait(l);
      //waitForOptimizationTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      //addStateTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      okvis::Time t0Matching = okvis::Time::now();
      bool asKeyframe = false;
	  //3.��ceres��Ӳ���-��һ֡�������imu����ֵ
      if (estimator_.addStates(frame, imuData, asKeyframe))//��Ҫ��һ��!!!!!!!!��ceres��Ӳ��������� bool Estimator::addStates(
	  {
        lastAddedStateTimestamp_ = frame->timestamp();
        //addStateTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      } else {
        LOG(ERROR) << "Failed to add state! will drop multiframe.";
        //addStateTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
        continue;
      }

      // -- matching keypoints, initialising landmarks etc.
      //4.��ceres�Ż��Ĳ����еõ�T_WS
      okvis::kinematics::Transformation T_WS;
      estimator_.get_T_WS(frame->id(), T_WS);//���Ӧ���Ǵ��Ż��ı�������ȡ�����ڵ�λ��
      //matchingTimer.start();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      //ƥ�亯��+�ر���ʼ����������+RANSAC����+�жϹؼ�֡����
      //���� bool Frontend::dataAssociationAndInitialization(
      //5.����ƥ�� ,ͬʱ�����ж���һ֡�Ƿ�Ϊ�ؼ�֡
      frontend_.dataAssociationAndInitialization(estimator_, T_WS, parameters_,map_, frame, &asKeyframe);//�ǳ���Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!
      //matchingTimer.stop();//ʲô��û����Ϊ�˷��㿴�������ǰ����ע�͵�
      //6.���ùؼ�֡
      if (asKeyframe)
        estimator_.setKeyframe(frame->id(), asKeyframe);//���ͼ�в���ؼ�ֻ֡��������һ����־λ������ void setKeyframe(uint64_t frameId, bool isKeyframe)
      if(!blocking_)//Ĭ�Ͻ����������
	  {
	     //euroc timeLimitForMatchingAndOptimization=0.035��
        double timeLimit = parameters_.optimization.timeLimitForMatchingAndOptimization-(okvis::Time::now()-t0Matching).toSec();
		//����bool Estimator::setOptimizationTimeLimit(double timeLimit, int minIterations) {
        estimator_.setOptimizationTimeLimit(std::max<double>(0.0, timeLimit),parameters_.optimization.min_iterations);//euroc ����min_iterations=3
      }
      optimizationDone_ = false;
    }  // unlock estimator_mutex_

    // use queue size 1 to propagate a congestion to the _matchedFrames queue
    //6.���õ���ƥ�����˫Ŀ֡�ṹѹ�뵽ƥ�������
    if (matchedFrames_.PushBlockingIfFull(frame, 1) == false)
      return;
  }
}

// Loop to process IMU measurements.
//����imu�Ĳ���ֵ���µ�ǰ״̬Ȼ�����״̬ѹ�뵽optimizationResults_������
void ThreadedKFVio::imuConsumerLoop() 
{
  okvis::ImuMeasurement data;
  TimerSwitchable processImuTimer("0 processImuMeasurements",true);//���ʲô��û��
  for (;;) 
  {
    // get data and check for termination request
    if (imuMeasurementsReceived_.PopBlocking(&data) == false)//�Ӷ����е�������ֵ
      return;
    //processImuTimer.start();//ʲô��û�� ����ע�͵����㿴����
    okvis::Time start;
    const okvis::Time* end;  // do not need to copy end timestamp
    {
      std::lock_guard<std::mutex> imuLock(imuMeasurements_mutex_);
      OKVIS_ASSERT_TRUE(Exception,
                        imuMeasurements_.empty()
                        || imuMeasurements_.back().timeStamp < data.timeStamp,
                        "IMU measurement from the past received");

      if (parameters_.publishing.publishImuPropagatedState)//�������Ĭ����true
	  {
        if (!repropagationNeeded_ && imuMeasurements_.size() > 0) //repropagationNeeded_ ����������Ż��߳��б���ֵΪtrue�ˣ���Ե��֮��������ᱻ��ֵΪtrue
		{
          start = imuMeasurements_.back().timeStamp;
        } else if (repropagationNeeded_)//�������������ʾ��Ҫ����propagationһ��
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
      imuMeasurements_.push_back(data);//���Ż����߳��л����imuMeasurements_
    }  // unlock _imuMeasurements_mutex

    // notify other threads that imu data with timeStamp is here.
    imuFrameSynchronizer_.gotImuData(data.timeStamp);//����ϵͳ�õ���һ������

    if (parameters_.publishing.publishImuPropagatedState)//�������Ĭ����true
	{
      Eigen::Matrix<double, 15, 15> covariance;
      Eigen::Matrix<double, 15, 15> jacobian;

	  //���� Frontend::propagation
	  //��ʵ������������ϻ��ǵ��õ�ImuError::propagation����
	  //ע���� ������ſ˱Ⱥ�Э���û���õ�
      frontend_.propagation(imuMeasurements_, imu_params_, T_WS_propagated_,
                            speedAndBiases_propagated_, start, *end, &covariance,
                            &jacobian);
      OptimizationResults result;
      result.stamp = *end;
      result.T_WS = T_WS_propagated_;
      result.speedAndBiases = speedAndBiases_propagated_;
      result.omega_S = imuMeasurements_.back().measurement.gyroscopes- speedAndBiases_propagated_.segment<3>(3);
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) //�����������
	  {
        result.vector_of_T_SCi.push_back(okvis::kinematics::Transformation(*parameters_.nCameraSystem.T_SC(i)));//�����imu�Ĺ̶�λ�˱仯
      }
      result.onlyPublishLandmarks = false;
      optimizationResults_.PushNonBlockingDroppingIfFull(result,1);
    }
    //processImuTimer.stop();//ʲô��û�� ����ע�͵����㿴����
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
    std::vector<cv::Mat> out_images(parameters_.nCameraSystem.numCameras());//�洢������������ͼ��
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) 
	{
      out_images[i] = visualizer_.drawMatches(new_data, i);//���� cv::Mat VioVisualizer::drawMatches
    }
	displayImages_.PushNonBlockingDroppingIfFull(out_images,1);//�̶߳�����صĲ���
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
//�Ż��ͱ�Ե���߳�
void ThreadedKFVio::optimizationLoop() 
{
  TimerSwitchable optimizationTimer("3.1 optimization",true);//�Ǹ��պ���ʲô��û��
  TimerSwitchable marginalizationTimer("3.2 marginalization",true);//�Ǹ��պ���ʲô��û��
  TimerSwitchable afterOptimizationTimer("3.3 afterOptimization",true);

  for (;;)
  {
    std::shared_ptr<okvis::MultiFrame> frame_pairs;
    VioVisualizer::VisualizationData::Ptr visualizationDataPtr;
    okvis::Time deleteImuMeasurementsUntil(0, 0);
	//1.�Ӷ����е����Ѿ�ƥ���֡���������ǳ�Ϊ��ǰ֡��������match�̣߳� ����okvis::MultiFrame
    if (matchedFrames_.PopBlocking(&frame_pairs) == false)
      return;
    OptimizationResults result;//�ṹ���� ����  struct OptimizationResults {
    {
      std::lock_guard<std::mutex> l(estimator_mutex_);
      //optimizationTimer.start();//ʲô��û�� ���㿴����
      //if(frontend_.isInitialized()){
      //���� void Estimator::optimize(
      //��Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       //2.��ʼ�Ż� �����µ�ͼ�����Ϣ
        estimator_.optimize(parameters_.optimization.max_iterations, 2, false);//euroc����max_iterations=10
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

      //optimizationTimer.stop();//ʲô��û�� ���㿴����

      // get timestamp of last frame in IMU window. Need to do this before marginalization as it will be removed there (if not keyframe)
      if (estimator_.numFrames()> size_t(parameters_.optimization.numImuFrames))//numImuFrames euroc����=3
	  {
	    //Ĭ��temporal_imu_data_overlapĬ������=0.02�룬eurocʹ�õ�imuƵ����0.05��
        deleteImuMeasurementsUntil = estimator_.multiFrame(estimator_.frameIdByAge(parameters_.optimization.numImuFrames))->timestamp() - temporal_imu_data_overlap;
      }

      //marginalizationTimer.start();//ʲô��û�� ���㿴����
      //numKeyframes=Ĭ������5
      //��Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //���� Estimator::applyMarginalizationStrategy(
      estimator_.applyMarginalizationStrategy(
          parameters_.optimization.numKeyframes,
          parameters_.optimization.numImuFrames, result.transferredLandmarks);//transferredLandmarks=�õ�Ҫ����Ե���ĵ�ͼ��
      //marginalizationTimer.stop();//ʲô��û�� ���㿴����
      //afterOptimizationTimer.start();//ʲô��û�� ���㿴����

      // now actually remove measurements
      //�� imuMeasurements_��ɾ��deleteImuMeasurementsUntilʱ���֮ǰ��imu�Ĺ۲�ֵ
      deleteImuMeasurements(deleteImuMeasurementsUntil);
      // saving optimized state and saving it in OptimizationResults struct
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        estimator_.get_T_WS(frame_pairs->id(), lastOptimized_T_WS_);//�õ��Ż�֮��ĵ�ǰ֡����̬ �洢��lastOptimized_T_WS_
        estimator_.getSpeedAndBias(frame_pairs->id(), 0,lastOptimizedSpeedAndBiases_);//�õ��Ż�֮��ĵ�ǰ֡���ٶȺ�bias �洢��lastOptimizedSpeedAndBiases_
        lastOptimizedStateTimestamp_ = frame_pairs->timestamp();

        // if we publish the state after each IMU propagation we do not need to publish it here.
        if (!parameters_.publishing.publishImuPropagatedState) //publishImuPropagatedState=Ĭ��Ϊtrue
		{
          result.T_WS = lastOptimized_T_WS_;
          result.speedAndBiases = lastOptimizedSpeedAndBiases_;
          result.stamp = lastOptimizedStateTimestamp_;
          result.onlyPublishLandmarks = false;
        }
        else //Ĭ�Ͻ����������
          result.onlyPublishLandmarks = true;

		//�õ�landmarksMap_�ṹ�����еĵ�ͼ�� ��������result.landmarksVector�ṹ��
        estimator_.getLandmarks(result.landmarksVector);//���� Estimator::getLandmarks(MapPointVector & landmarks) const

        repropagationNeeded_ = true;//��Ϊ״̬����������������������ñ�־λ ��˼����Ҫ��imu���½���propagation
      }

	  //���ǰѿ��ӻ��Ĳ���ע�͵� ���㿴����
      if (parameters_.visualization.displayImages) //euroc����Ϊtrue
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

	/*ע�͵������������������㿴����
    if (!parameters_.publishing.publishImuPropagatedState) //Ĭ�ϲ��������������publishImuPropagatedState=Ĭ��Ϊtrue
	{
      // adding further elements to result that do not access estimator.
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        result.vector_of_T_SCi.push_back( okvis::kinematics::Transformation(*parameters_.nCameraSystem.T_SC(i)));
      }
    }*/
    optimizationResults_.Push(result);//�����ѹ�뵽�Ż�������

    // adding further elements to visualization data that do not access estimator
    if (parameters_.visualization.displayImages)//euroc����Ϊtrue
	{
      visualizationDataPtr->currentFrames = frame_pairs;
      visualizationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr, 1);
    }
    //afterOptimizationTimer.stop();//ʲô��û�� ���㿴����
  }
}

// Loop that publishes the newest state and landmarks.
void ThreadedKFVio::publisherLoop() {
  for (;;) 
  {
    // get the result data
    OptimizationResults result;
    if (optimizationResults_.PopBlocking(&result) == false)//����Ż�����û�������Ϣ��ֱ�ӷ���
      return;

    // call all user callbacks
    //���ǰ�����û�ж���ĺ���ע�͵� ���㿴����
    /*
    if (stateCallback_ && !result.onlyPublishLandmarks)
      stateCallback_(result.stamp, result.T_WS);
    */
    
    if (fullStateCallback_ && !result.onlyPublishLandmarks)
    {
       //���� PoseViewer::publishFullStateAsCallback = fullStateCallback_
       //���ǿ��Կ����Ż������е�ֵ�����¹���:ʱ���������������������ϵ�µ�λ�ã��ٶȺͽ��ٶȵ�bias�ͼ��ٶȵ�bias�����ٶ�
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
