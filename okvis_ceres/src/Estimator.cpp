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
һ����Ӳ����飺
1����õ�ǰ֡��Tws������뵽ceres�Ĳ�������
��������飺std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock( new okvis::ceres::PoseParameterBlock(T_WS, states.id,multiFrame->timestamp()));
���뵽ceres�У�mapPtr_->addParameterBlock(poseParameterBlock,ceres::Map::Pose6d))
2����camera��imu����̬���뵽ceres�Ĳ�������(���Ż�ʱ���ó��˳����������Ż�)
��������飺std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlockPtr(new okvis::ceres::PoseParameterBlock(T_SC, id,multiFrame->timestamp()));
���뵽ceres�У�mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr,ceres::Map::Pose6d))
3���õ���ǰ֡���ٶȺ�bias�������״̬���뵽ceres�Ĳ�����
��������飺std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock(new okvis::ceres::SpeedAndBiasParameterBlock(speedAndBias, id, multiFrame->timestamp()));
���뵽ceres�У�mapPtr_->addParameterBlock(speedAndBiasParameterBlock))

��������������
1.1��ʼʱ��
����������std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS, information));
�������뵽ceres�У�mapPtr_->addResidualBlock(poseError,NULL,poseParameterBlock);

����������std::shared_ptr<ceres::SpeedAndBiasError > speedAndBiasError(  new ceres::SpeedAndBiasError(speedAndBias, 1.0, sigma_bg*sigma_bg, sigma_ba*sigma_ba));
�������뵽ceres�У�mapPtr_->addResidualBlock(  speedAndBiasError,
											    	NULL,
													mapPtr_->parameterBlockPtr(
													states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
1.2�ǳ�ʼʱ��
����������std::shared_ptr<ceres::ImuError> imuError( new ceres::ImuError(imuMeasurements, imuParametersVec_.at(i),
                            					 lastElementIterator->second.timestamp,
                            					 states.timestamp));
�������뵽ceres�У�mapPtr_->addResidualBlock(
          imuError,
          NULL,
          mapPtr_->parameterBlockPtr(lastElementIterator->second.id),//���� std::shared_ptr<okvis::ceres::ParameterBlock> Map::parameterBlockPtr(
          mapPtr_->parameterBlockPtr(lastElementIterator->second.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id),
          mapPtr_->parameterBlockPtr(states.id), 
          mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
*/
//asKeyframe�������һ֡�Ƿ�Ϊ�ؼ�֡,��������Ķ���false
//�����������matchLoop�����ù�
bool Estimator::addStates(okvis::MultiFramePtr multiFrame,const okvis::ImuMeasurementDeque & imuMeasurements, bool asKeyframe)
{
  // note: this is before matching...
  // TODO !!
  //1.���ȸ���imu�Ĳ���ֵ�õ���ǰ֡��λ��T_WS
  okvis::kinematics::Transformation T_WS;
  okvis::SpeedAndBias speedAndBias;
  if (statesMap_.empty())//��ʼ��״̬
  {
    // in case this is the first frame ever, let's initialize the pose:
    bool success0 = initPoseFromImu(imuMeasurements, T_WS);
    OKVIS_ASSERT_TRUE_DBG(Exception, success0,
        "pose could not be initialized from imu measurements.");
    if (!success0)
      return false;
    speedAndBias.setZero();//����һ֡������ٶȺ�bias����Ϊ0
    speedAndBias.segment<3>(6) = imuParametersVec_.at(0).a0;
  } else 
  {
    // get the previous states1.���Ȼ����һ��ʱ�̵�λ�˺��ٶ���bias
    //��һ֡��λ�˵Ĳ�����id
    uint64_t T_WS_id = statesMap_.rbegin()->second.id;//rbeginָ�����������һ��Ԫ��
    //����  enum ImuSensorStates
    //��һ֡���ٶȺ�bias�Ĳ�����id
    uint64_t speedAndBias_id = statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).at(0).at(ImuSensorStates::SpeedAndBias).id;
    OKVIS_ASSERT_TRUE_DBG(Exception, mapPtr_->parameterBlockExists(T_WS_id),
                       "this is an okvis bug. previous pose does not exist.");
	 //static_pointer_castָ��ת����������ParameterBlockת��ΪPoseParameterBlock������estimate��������ľ���λ����Ϣ
	 //�Ӳ���������ȡ������
    T_WS = std::static_pointer_cast<ceres::PoseParameterBlock>(mapPtr_->parameterBlockPtr(T_WS_id))->estimate();//���� std::shared_ptr<okvis::ceres::ParameterBlock> Map::parameterBlockPtr(
    //OKVIS_ASSERT_TRUE_DBG(
    //    Exception, speedAndBias_id,
    //    "this is an okvis bug. previous speedAndBias does not exist.");
    speedAndBias = std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>( mapPtr_->parameterBlockPtr(speedAndBias_id))->estimate();

    // propagate pose and speedAndBias
    //1.2.������һʱ�̵�״̬��imu�Ĳ���ֵ�õ���һʱ�̵�״̬��û�м���Э������ſ˱�
    int numUsedImuMeasurements = ceres::ImuError::propagation(imuMeasurements, imuParametersVec_.at(0), T_WS, speedAndBias,statesMap_.rbegin()->second.timestamp, multiFrame->timestamp());
    OKVIS_ASSERT_TRUE_DBG(Exception, numUsedImuMeasurements > 1,
                       "propagation failed");
    if (numUsedImuMeasurements < 1){
      LOG(INFO) << "numUsedImuMeasurements=" << numUsedImuMeasurements;
      return false;
    }
  }


  // create a states object:
  //2.���ݵ�ǰ֡��λ�� ����һ��states״̬�����������״̬��һЩ��Ϣ��������ǰ֡����̬Tws����ceres�Ĳ�������addParameterBlock��
  //״̬��id����˫Ŀ֡��id
  States states(asKeyframe, multiFrame->id(), multiFrame->timestamp());

  // check if id was used before
  OKVIS_ASSERT_TRUE_DBG(Exception,statesMap_.find(states.id)==statesMap_.end(),"pose ID" <<states.id<<" was used before!");

  // create global states
  //���� PoseParameterBlock::PoseParameterBlock( const 
  //����λ�˲�����
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock( new okvis::ceres::PoseParameterBlock(T_WS, states.id,multiFrame->timestamp()));
  // ״̬�е�global��һ��6ά�����飬ÿһάԪ�ض���һ��stateinfo
  states.global.at(GlobalStates::T_WS).exists = true;
  states.global.at(GlobalStates::T_WS).id = states.id;//���� enum GlobalStates

  if(statesMap_.empty())
  {
    referencePoseId_ = states.id; // set this as reference pose
    if (!mapPtr_->addParameterBlock(poseParameterBlock,ceres::Map::Pose6d)) //���� bool Map::addParameterBlock( 
	{
      return false;
    }
  } else 
  {
    //��ceres�м��뵱ǰimu��λ�˵Ĳ�����!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (!mapPtr_->addParameterBlock(poseParameterBlock,ceres::Map::Pose6d)) //���� bool Map::addParameterBlock(
	{
      return false;
    }
  }

  // add to buffer
  //��states���뵽statesMap_������statesMap_����camera��imu����̬���뵽ceres�Ĳ������У����ٶȺ�bias���뵽ceres�Ĳ������С�
  //����ǵ�һ֡ͼ�����������ϵ��imu����ϵ�ı任���뵽ceres��������
  statesMap_.insert(std::pair<uint64_t, States>(states.id, states));//��״̬����״̬�ռ䣬���������о�������statesMap_�����������
  multiFramePtrMap_.insert(std::pair<uint64_t, okvis::MultiFramePtr>(states.id, multiFrame));//��֡����֡���ͼ�����������о�������multiFramePtrMap_����ֵ

  // the following will point to the last states:
  std::map<uint64_t, States>::reverse_iterator lastElementIterator = statesMap_.rbegin();
  lastElementIterator++;//������ǰ��һλ,ָ��ǰһ��״̬,���¼���״̬����һ��״̬

  // initialize new sensor states
  // cameras:
  //extrinsicsEstimationParametersVec_�д���������������������
  for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) 
  {

    SpecificSensorStatesContainer cameraInfos(2);
    cameraInfos.at(CameraSensorStates::T_SCi).exists=true;//���� enum CameraSensorStates
    cameraInfos.at(CameraSensorStates::Intrinsics).exists=false;
	//��i����������λ�ƺ������ת�ı�׼��С����ֵ
    if(((extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation<1e-12)||(extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation<1e-12))&& (statesMap_.size() > 1))
	{//��Զ�����������
      // use the same block...
      //��״̬��ͼ�����ڶ���״̬����������������е�i�������T_SCiλ�˶�ӦԪ�ص�id ��ֵ�������Ϣ����
      cameraInfos.at(CameraSensorStates::T_SCi).id = lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id;
    } else 
    {//��ʼ��ʱ�����������
      const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlockPtr(new okvis::ceres::PoseParameterBlock(T_SC, id,multiFrame->timestamp()));
      if(!mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr,ceres::Map::Pose6d))//���ͼ�����λ�˲�����!!!!!!!!!!!!!!!!!!!!!!
	  {
        return false;
      }
      cameraInfos.at(CameraSensorStates::T_SCi).id = id;
    }
    // update the states info
    //�������Ϣ������ӵ������״̬�����������������
    statesMap_.rbegin()->second.sensors.at(SensorStates::Camera).push_back(cameraInfos);
    states.sensors.at(SensorStates::Camera).push_back(cameraInfos);
  }

  // IMU states are automatically propagated.
  // IMU״̬�ĸ��� imuParametersVec_�洢����imu����Ϣ�������ֻʹ��һ��imuֻѭ��һ��
  for (size_t i=0; i<imuParametersVec_.size(); ++i)
  {
    SpecificSensorStatesContainer imuInfo(2);
    imuInfo.at(ImuSensorStates::SpeedAndBias).exists = true;
    uint64_t id = IdProvider::instance().newId();
	//�ǳ���Ҫ�ĺ��� �������ٶȺ�bias�Ĳ�����
	//���� SpeedAndBiasParameterBlock::SpeedAndBiasParameterBlock(
    std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock(new okvis::ceres::SpeedAndBiasParameterBlock(speedAndBias, id, multiFrame->timestamp()));

    if(!mapPtr_->addParameterBlock(speedAndBiasParameterBlock))//��ceres����ӵ�ǰ֡���ٶȺ�bias������!!!!!!!!!!!!!!!!!!!!!
	{
      return false;
    }
    imuInfo.at(ImuSensorStates::SpeedAndBias).id = id;
    statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).push_back(imuInfo);
    states.sensors.at(SensorStates::Imu).push_back(imuInfo);
  }


  //3.��������
  // depending on whether or not this is the very beginning, we will add priors or relative terms to the last state:
  // //��ʾ���ʼ,�ո�ֻ�����һά״̬
  if (statesMap_.size() == 1) 
  {
    // let's add a prior
    Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Zero();
    information(5,5) = 1.0e8; information(0,0) = 1.0e8; information(1,1) = 1.0e8; information(2,2) = 1.0e8;
    //λ�˵���������applyMarginalization�����б������� ���� std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS_0, information));
    std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS, information));//�ǳ���Ҫλ�˵�����!!!!!!!!!!!����������ĵ�ǰ֡����̬��Ȼ�ǲ���ֵ
    /*auto id2= */ mapPtr_->addResidualBlock(poseError,NULL,poseParameterBlock);//����̬���������뵽ceres�У����� ::ceres::ResidualBlockId Map::addResidualBlock(
    //mapPtr_->isJacobianCorrect(id2,1.0e-6);

    // sensor states����˫Ŀ��� ��ֻ��ѯ����
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
      else //Ĭ�Ͻ����������
	  {
	  //���� bool Map::setParameterBlockConstant(
        mapPtr_->setParameterBlockConstant(states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id);
      }
    }
    for (size_t i = 0; i < imuParametersVec_.size(); ++i)//ֻ��ѯһ��
	{
      Eigen::Matrix<double,6,1> variances;
      // get these from parameter file
      const double sigma_bg = imuParametersVec_.at(0).sigma_bg;
      const double sigma_ba = imuParametersVec_.at(0).sigma_ba;
	  //�ǳ���Ҫ�ĺ������ٶȺ�bias������
	  //1��ʾ�����ٶȵ�Э����
	  //�������������������ٶȺ�bias����
      std::shared_ptr<ceres::SpeedAndBiasError > speedAndBiasError(  new ceres::SpeedAndBiasError(speedAndBias, 1.0, sigma_bg*sigma_bg, sigma_ba*sigma_ba));
      // add to map ���ٶȺ�bias��cost function ���뵽ceres��
      mapPtr_->addResidualBlock(
          speedAndBiasError,
          NULL,
          mapPtr_->parameterBlockPtr(
              states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
      //mapPtr_->isJacobianCorrect(id,1.0e-6);
    }
  }
  else{//���ǳ�ʼ��
    // add IMU error terms
      // ���IMU�����
    for (size_t i = 0; i < imuParametersVec_.size(); ++i)//ֻ����һ�ζ���һ��imu��˵
	{
	  //�����ʵ��imu�в����Ҫ����!!!!!!!!
      std::shared_ptr<ceres::ImuError> imuError( new ceres::ImuError(imuMeasurements, imuParametersVec_.at(i),
                            					 lastElementIterator->second.timestamp,
                            					 states.timestamp));
	  /// IMU�Ĳв��
      /// NULL
      /// ��ͼ�е����ڶ���״̬��(λ��)ID---->���ݿ�
      /// ��ͼ�е����ڶ���״̬�ĵ�i��IMU���������ٶ���ƫ�ò��ֶ�Ӧ��ID----->���ݿ�
      /// ��ͼ������һ��״̬��(λ��)ID----->���ݿ�
      /// ��ͼ������һ��״̬�ĵ�i��IMU���������ٶ���ƫ�ò��ֶ�Ӧ��ID------->���ݿ�
      //���� ::ceres::ResidualBlockId Map::addResidualBlock(
      /*::ceres::ResidualBlockId id = */mapPtr_->addResidualBlock(
          imuError,
          NULL,
          mapPtr_->parameterBlockPtr(lastElementIterator->second.id),//���� std::shared_ptr<okvis::ceres::ParameterBlock> Map::parameterBlockPtr(
          mapPtr_->parameterBlockPtr(lastElementIterator->second.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id),
          mapPtr_->parameterBlockPtr(states.id), 
          mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
      //imuError->setRecomputeInformation(false);
      //mapPtr_->isJacobianCorrect(id,1.0e-9);
      //imuError->setRecomputeInformation(true);
    }

    // add relative sensor state errors
    //���ѭ����euroc������������ǲ�������
    //�������������ʾ��Ҫ�Ż��������ϵ��imu����ϵ��λ�˱仯
    //euroc������Ĭ��������������
    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) 
	{
	 //������Ҫ�ж���һ֡״̬�����ID�͵�ǰ֡״̬�����ID�Ƿ����,��ȼ���ʾ��������δ���и���
	 //��������������������
      if(lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id != states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id)
	  {
        // i.e. they are different estimated variables, so link them with a temporal error term
        double dt = (states.timestamp - lastElementIterator->second.timestamp).toSec();
        double translationSigmaC = extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation;
        double translationVariance = translationSigmaC * translationSigmaC * dt;
        double rotationSigmaC = extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation;
        double rotationVariance = rotationSigmaC * rotationSigmaC * dt;
		//������Ҫ����Ե�������!!!!!!!!!!!!!!!!!!
        std::shared_ptr<ceres::RelativePoseError> relativeExtrinsicsError( new ceres::RelativePoseError(translationVariance, rotationVariance));
		//��ͼ������������������
		///�β�
		/// ���������
		/// NULL
		/// ��һ֡��i�������ӦT_SC�����---->���ݿ�
		/// ��ǰ֡��i�������ӦT_SC�����---->���ݿ�
		//���� std::shared_ptr<okvis::ceres::ParameterBlock> Map::parameterBlockPtr(
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
}//addStates��������----------------------------------------------------------------------

// Add a landmark.
//��ceres����������ͼ�������
//����landmarksMap_�ṹ�е���Ϣ���¼���һ����ͼ��
bool Estimator::addLandmark(uint64_t landmarkId,const Eigen::Vector4d & landmark) 
{
  //�½�һ����ͼ��Ĳ����飬����������id���ǵ�ͼ���id
  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> pointParameterBlock( new okvis::ceres::HomogeneousPointParameterBlock(landmark, landmarkId));

  //��ceres����Ӳ�����
  //���� bool Map::addParameterBlock( 
  //ֻ��״̬�ռ����Ѿ����������ͼ��ʱ�Ż᷵��false
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
  //��洢���е�ͼ��ĽṹlandmarksMap_�У���������µĵ�ͼ��
  landmarksMap_.insert( std::pair<uint64_t, MapPoint>(landmarkId, MapPoint(landmarkId, landmark, 0.0, dist)));
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "bug: inconsistend landmarkdMap_ with mapPtr_.");
  return true;
}

// Remove an observation from a landmark.
//����mapPtr_�еĵ�ͼ����
//��ceres��ɾ������в��
bool Estimator::removeObservation(::ceres::ResidualBlockId residualBlockId) 
{
  const ceres::Map::ParameterBlockCollection parameters = mapPtr_->parameters(residualBlockId);
  const uint64_t landmarkId = parameters.at(1).first;
  // remove in landmarksMap
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);//Ԫ��=(��ͼ��id,��ͼ����������)
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
  //��ceres��ɾ������в��
  //���� bool Map::removeResidualBlock(::ceres::ResidualBlockId residualBlockId) 
  mapPtr_->removeResidualBlock(residualBlockId);
  return true;
}

// Remove an observation from a landmark, if available.
//1.����landmarksMap_���ݣ������ͼ��û�й۲⵽�����������
//2.��ceres��ɾ������в�飬��������ر���
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

  okvis::KeypointIdentifier kid(poseId,camIdx,keypointIdx);//����������������
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  std::map<okvis::KeypointIdentifier, uint64_t >::iterator it = mapPoint.observations.find(kid);
  if(it == landmarksMap_.at(landmarkId).observations.end())
  {
    return false; // observation not present
  }

  // remove residual block
  //reinterpret_cast��ǿ������ת����
  //ResidualBlockId���ô� ������ʹ��ceres����AddResidualBlockʱ������ceres�ڲ���ÿһ������Ĳв���һ��id�����id�����;���ResidualBlockId
  mapPtr_->removeResidualBlock(reinterpret_cast< ::ceres::ResidualBlockId>(it->second));//���� bool Map::removeResidualBlock(::ceres::ResidualBlockId residualBlockId) 

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
 * @param removedLandmarks Get the landmarks that were removed by this operation.-����ʵ��һ�����ֵ
 * @return True if successful.
 */
 //euroc���� numKeyframes=5  numImuFrames=3
 //��������okvis��������Ҫ�ĺ���
bool Estimator::applyMarginalizationStrategy(size_t numKeyframes, size_t numImuFrames,okvis::MapPointVector& removedLandmarks)
{
  // keep the newest numImuFrames
  //��֤״̬�ռ��е�״̬��Ŀ����numImuFrames=3������ֱ�ӷ���
  //map���¼����Ԫ��Ϊ����Ԫ��
  //1. ���imu�������ؾ���̬����С������ֱ�ӷ���
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();//��ָ������c�����һ��Ԫ��
  for(size_t k=0; k<numImuFrames; k++)
  {
    rit++;
    if(rit==statesMap_.rend())//��ָ������c�ĵ�һ��Ԫ��ǰ���λ��
	{
      // nothing to do.
      return true;
    }
  }

  // remove linear marginalizationError, if existing
  // �Ƴ��в�飬�����Ҫ�Ļ�
  //2.��ceres��ɾ����Ե���в�飬��������ر���
  if (marginalizationErrorPtr_ && marginalizationResidualId_) 
  {
    //���� Map::removeResidualBlock(::ceres::ResidualBlockId residualBlockId) 
    //�����marginalizationResidualId_����Ҫ����Ե���Ĳ�����id
    //a.��ceres��ɾ������в��
    //b.��id2ResidualBlock_Multimap_ɾ���в��
    //c.����residualBlockId2ParameterBlockCollection_Map_
    //d.����residualBlockId2ResidualBlockSpec_Map_
    bool success = mapPtr_->removeResidualBlock(marginalizationResidualId_);//marginalizationResidualId_=��Ե���в���ceres id
    OKVIS_ASSERT_TRUE_DBG(Exception, success,"could not remove marginalization error");
    marginalizationResidualId_ = 0;
    if (!success)
      return false;
  }

  // these will keep track of what we want to marginalize out.
  std::vector<uint64_t> paremeterBlocksToBeMarginalized;//Ϊ��Ҫ��Ե�������ݿ�id�ļ���
  std::vector<bool> keepParameterBlocks;//ά����paremeterBlocksToBeMarginalized��ȵ�������false��

  if (!marginalizationErrorPtr_) //Ӧ���ǳ�ʼ��ʱ�Ż����һ��
  {
    //���� MarginalizationError::MarginalizationError
    //*mapPtr_.get()����ֵ����okvis::ceres::Map
    marginalizationErrorPtr_.reset( new ceres::MarginalizationError(*mapPtr_.get()));//��ʾ��ָ�븳ֵ
  }

  // distinguish if we marginalize everything or everything but pose
  std::vector<uint64_t> removeFrames;
  std::vector<uint64_t> removeAllButPose;
  std::vector<uint64_t> allLinearizedFrames;
  size_t countedKeyframes = 0;
  //3.��ȡҪ���޳���״̬֡
  ///removeAllButPose��ʾ���ڣ�3֡��֮ǰ������δ�޳���״̬��֡������Щ֡�������ؼ�֡������λ��֮����������ݿ飨�ٶ�/ƫ�ã��������б�Ե��
  /// removeFrames��ʾ����֮ǰ������δ�޳���״̬�����ǲ������ٽ����ڵ�5���ؼ�֡����Щ֡�ģ�λ�ˣ���Σ����ݿ鶼�����б�Ե��
  /// allLinearizedFrames��ʾ���ڣ�3֡��֮ǰ������δ�޳���״̬��֡��
  //ǧ��ǧ��Ҫע�������rit���ǽ���λ�ö����ڵ������ĸ�λ����
  //�������ǲ��Է��� removeFrames������һֱά����1������removeAllButPose����һֱά����6��
  while (rit != statesMap_.rend())//һ�����������������c�ĵ�һ��Ԫ��ǰ���λ��
  {
    if (!rit->second.isKeyframe || countedKeyframes >= numKeyframes) 
	{
      removeFrames.push_back(rit->second.id);//��ֻ�����������
    } else 
   {
      countedKeyframes++;
    }
    removeAllButPose.push_back(rit->second.id);//��ֻ�����������
    allLinearizedFrames.push_back(rit->second.id);
    ++rit;// check the next frame
  }

  
  //����ĺ���������forѭ������
  //һ���Ǳ���removeFrames
  //һ���Ǳ���removeAllButPose  
  // marginalize everything but pose:
   // 4.������Ե������imu����֡(���µ�����������֡)֮�������֡���ٶȺ�bias
  for(size_t k = 0; k<removeAllButPose.size(); ++k)
  {
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeAllButPose[k]);

    ///��OKVIS��ֻ��״̬��global����ֻ��GlobalStates::T_WSλ����ֵ
    //��ʵ���forѭ����ʲô��û��
    for (size_t i = 0; i < it->second.global.size(); ++i) 
	{
      if (i == GlobalStates::T_WS) //����ֻ�����global���Ͱ� ������ enum GlobalStates
	  {
        continue; // we do not remove the pose here.
      }//��Ϊ��������ֻ�õ���GlobalStates::T_WS������������Ĵ�����Զ���ᱻִ�е�
	  /*
      if (!it->second.global[i].exists)//���ܴ���ֱ������ 
	  {
        continue; // if it doesn't exist, we don't do anything.
      }
	   //����ΪTWS���ݿ�Ϊ�̶���ֱ������
      //global��id��״̬��id��ȣ���״̬��id��λ�ˣ�TWS�����ݿ��id���
      if (mapPtr_->parameterBlockPtr(it->second.global[i].id)->fixed())
	  {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      // ����һ��״̬id����һ�£���ֱ������
      if(checkit->second.global[i].exists && checkit->second.global[i].id == it->second.global[i].id)
	  {
        continue;
      }
	  ///״̬�Ĵ����Զ���Ϊfalse
      it->second.global[i].exists = false; // remember we removed
       ///���״̬��i��ȫ������TSW���ȵȣ�id���������ݿ��id����������
      paremeterBlocksToBeMarginalized.push_back(it->second.global[i].id);
      keepParameterBlocks.push_back(false);
	  //ResidualBlockCollectionԪ�� = �в����Ϣ=�в���ceres�е�id+loss function����ָ��+����ָ��
	  //���� Map::ResidualBlockCollection Map::residuals(
	  //��������������Ǵ�id2ResidualBlock_Multimap_�ṹ��Ѱ��������Ĳ�����id��صĲв�飬������Щ�в�鷵��
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.global[i].id);
      for (size_t r = 0; r < residuals.size(); ++r)//�����������������صĲв��
	  {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =  std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>( residuals[r].errorInterfacePtr);
        if(!reprojectionError) //û����ͶӰ���
		{   // we make sure no reprojection errors are yet included.
		   //���� bool MarginalizationError::addResidualBlock(
		   //�ǳ���Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);//ֱ����Ӳв������
        }
      }*/
    }


    /// ��Ҫ��Ե��֡�Ĵ��������
    /// ��һ�У��������д�����
    /// �ڶ��У������ô�����������
    /// �����У���i�ֵ�j����������״̬����
    // ���forѭ���������Ǳ�Ե������imu����֡(���µ�����������֡)֮�������֡���ٶȺ�bias
    //���������������Ҫ�ǽ��ٶȺ�bias��Ӧ�Ĳв����ӵ�marginalizationErrorPtr_��ȥ
    // add all error terms of the sensor states.
    for (size_t i = 0; i < it->second.sensors.size(); ++i) {
      for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
        for (size_t k = 0; k < it->second.sensors[i][j].size(); ++k) 
		{
		  //���� enum SensorStates��ֻ�õ��� Camera = 0,  Imu = 1,
		  //���� enum CameraSensorStates��T_SCi = 0,  CameraSensorStates::Intrinsics = 1,
          if (i == SensorStates::Camera && k == CameraSensorStates::T_SCi) 
		  {
            continue; // we do not remove the extrinsics pose here.
          }
		  //��״̬�����ڵ�ֱ������
		  //�������Ƿ�����֪intrinstrics=false��
          if (!it->second.sensors[i][j][k].exists) {
            continue;
          }

		  
		  //ֻ��Ե��imu���ٶȺ�bias
		  //���imu���ٶȺ�bias����Ϊ�̶�ֵ��Ҳֱ������
          if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)->fixed()) 
		  {
            continue;  // we never eliminate fixed blocks.
          }

		  //
          std::map<uint64_t, States>::iterator checkit = it;
          checkit++;
          // only get rid of it, if it's different
          //�����һ��״̬����δ�����仯��ֱ������
          if(checkit->second.sensors[i][j][k].exists && checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id)
		  {
            continue;
          }
          it->second.sensors[i][j][k].exists = false; // remember we removed
          //��Ҫ����Ե���Ĵ�����״̬����λ��ٶ�ƫ�����ݿ飩�����
          paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
          keepParameterBlocks.push_back(false);
		  //���� Map::ResidualBlockCollection Map::residuals(
	     //��������������Ǵ�id2ResidualBlock_Multimap_�ṹ��Ѱ��������Ĳ�����id��صĲв�飬������Щ�в�鷵��
          ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.sensors[i][j][k].id);
          for (size_t r = 0; r < residuals.size(); ++r) //����������ٶȺ�bias��صĲв��
		  {
		     //ResidualBlockCollectionԪ�� = �в����Ϣ=�в���ceres�е�id+loss function����ָ��+����ָ��
            std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError = std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
            if(!reprojectionError) //������ͶӰ���
			{   // we make sure no reprojection errors are yet included.
			  //���� bool MarginalizationError::addResidualBlock(
		      //�ǳ���Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		      //�����ǽ��ٶȺ�bias��Ӧ�Ĳв����ӵ�marginalizationErrorPtr_����
              marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);//��Ӳвϵͳ
            }
          }
        }
      }
    }
  }// for(size_t k = 0; k<removeAllButPose.size(); ++k)����

  
  // marginalize ONLY pose now:
  ///5.��ȥimu����֡�����������ؼ�֡��ʣ�µ�֡���б�Ե��
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
    //5.1��imu�����ص�Ҫ����Ե���Ĳв����뵽��Ե��ģ����
    // add remaing error terms
     //��������������Ǵ�id2ResidualBlock_Multimap_�ṹ��Ѱ��������Ļ�������̬������id��صĲв�飬������Щ�в�鷵��
    ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.global[GlobalStates::T_WS].id);

    for (size_t r = 0; r < residuals.size(); ++r)//����λ����صĲв��
	{
      if(std::dynamic_pointer_cast<ceres::PoseError>(residuals[r].errorInterfacePtr)) //�����λ��ƫ��
	  {                  // avoids linearising initial pose error
	                     //���� bool Map::removeResidualBlock(::ceres::ResidualBlockId residualBlockId) 
	                    //1.��ceres��ɾ������в��
				//2. ��id2ResidualBlock_Multimap_ɾ���в��
				//3.����residualBlockId2ParameterBlockCollection_Map_
				//4.����residualBlockId2ResidualBlockSpec_Map_
				mapPtr_->removeResidualBlock(residuals[r].residualBlockId);
				reDoFixation = true;
        continue; 
      }
      std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>( residuals[r].errorInterfacePtr);
      if(!reprojectionError)
	  {   // we make sure no reprojection errors are yet included.
	 	 //���� bool MarginalizationError::addResidualBlock(
        marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);//����������̬��ز�����ͶӰ�޹صĲв����뵽marginalizationErrorPtr_��
      }
    }

    // add remaining error terms of the sensor states.
    /*���������ʵ�������û�����еĴ���ע�͵� ����ۿ�
    size_t i = SensorStates::Camera;
    for (size_t j = 0; j < it->second.sensors[i].size(); ++j) //�����������
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
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.sensors[i][j][k].id);//�ҵ�����������صĲв��
      for (size_t r = 0; r < residuals.size(); ++r) 
	  {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError = std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>( residuals[r].errorInterfacePtr);
        if(!reprojectionError)
		{   // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);//��������صĲв����뵽marginalizationErrorPtr_
        }
      }
    }*/

    // now finally we treat all the observations.
    //5.2����landmarksMap_�ṹ�еĵ�ͼ��
    OKVIS_ASSERT_TRUE_DBG(Exception, allLinearizedFrames.size()>0, "bug");
    uint64_t currentKfId = allLinearizedFrames.at(0);//����imu����֡������Ǹ�֡

    {
      for(PointMap::iterator pit = landmarksMap_.begin();pit != landmarksMap_.end(); )//������ͼ�е�ͼ��
	  {
        //Ԫ�� = �в����Ϣ = �в���ceres�е�id+loss function����ָ��+����ָ��
        //�ҵ����Ҹ���ͼ���������ص����вв��
        ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(pit->first);

        // first check if we can skip
        bool skipLandmark = true;
        bool hasNewObservations = false;
        bool justDelete = false;
        bool marginalize = true;
        bool errorTermAdded = false;
        std::map<uint64_t,bool> visibleInFrame;
        size_t obsCount = 0;//����imu����֡�۲⵽�Ĵ���
        for (size_t r = 0; r < residuals.size(); ++r) //���������������ͼ���йصĲв��
		{
	          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =  std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>( residuals[r].errorInterfacePtr);
	          if (reprojectionError) 
			  {
			    //���� Map::ParameterBlockCollection Map::parameters
			    //parameters�������ҵ��������BA�в��id��ص����в�����,���а���λ�˺͵�ͼ���id
			    //parameters�������ص����ݽṹ��Ԫ��  =(������id,������ָ�빹��)
			    //�ҵ��������BA�в��id��ص�λ�˵Ĳ����顣
	            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
	            // since we have implemented the linearisation to account for robustification,
	            // we don't kick out bad measurements here any more like
	            // if(vectorContains(allLinearizedFrames,poseId)){ ...
	            //   if (error.transpose() * error > 6.0) { ... removeObservation ... }
	            // }
	            if(vectorContains(removeFrames,poseId))//��ʾ�����ͼ����imu����֮ǰ����ͨ֡�б��۲⵽��
				{
	              skipLandmark = false;//��ʾ��Ҫ�Ըõ�ͼ����б�Ե������
	            }
	            if(poseId>=currentKfId) //imu�����е�֡�����������ͼ��
				{
	              marginalize = false;
	              hasNewObservations = true;
	            }
				//imu����֮���֡�����������ͼ��
	            if(vectorContains(allLinearizedFrames, poseId))
				{
	              visibleInFrame.insert(std::pair<uint64_t,bool>(poseId,true));
	              obsCount++;
	            }
	          }
        }//������ͼ��Ĳв�����
	 //a.����޸��������йصĲв���--------------------------------------------------
        if(residuals.size()==0)
		{
		  //���� bool Map::removeParameterBlock(uint64_t parameterBlockId) 
          mapPtr_->removeParameterBlock(pit->first);//��ceres��ɾ�������ͼ���Ӧ�Ĳв�飬ɾ�������ͼ��Ĳ�����
          removedLandmarks.push_back(pit->second);//�����Ƴ��ر��ļ���
          pit = landmarksMap_.erase(pit);//�ӵ�ͼ��ɾ�������ͼ��
          continue;
        }
	 //b.��ʾ�Ըõ�ͼ�㲻���б�Ե������-----------------------------------------------
        if(skipLandmark) 
		{
          pit++;//�ƶ�����һ����ͼ��
          continue;
        }

        // so, we need to consider it.
        //�ٴα����������ͼ����صĵĲв��
        //�ж���õر������Ĳв��ǲ�����ͶӰ���
        for (size_t r = 0; r < residuals.size(); ++r)
		{
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError = std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
          if (reprojectionError) 
		  {
		    //�ҵ��������BA�в��id��ص�λ�˵Ĳ����顣
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
			///��imu����֮�����ͨ֡�����������ͼ����imu��������֡���Թ۲⵽�õر�㣩��
            ///����imu����֡�����������ͼ�������µ���֡Ҳ���ܹ۲⵽�õر�㣩,���������еر��2,3
            //�������Է��ֵڶ���������Զ��������!!!!!!!
            if((vectorContains(removeFrames,poseId) && hasNewObservations) ||(!vectorContains(allLinearizedFrames,poseId) && marginalize))
            {
              // ok, let's ignore the observation.
              //c.-----------------------------------------------------------------
              removeObservation(residuals[r].residualBlockId);//��ceres��ɾ������в��
              residuals.erase(residuals.begin() + r);
              r--;
            } else if(marginalize && vectorContains(allLinearizedFrames,poseId)) 
            {///imu�����в��ܹ۲⵽�õر���ҷ�imu����֡�ܿ��������ͼ��,���������еر��1
             
              // TODO: consider only the sensible ones for marginalization
              //d------------------------------------------------------------------------
              if(obsCount<2) //��imu����֮�������֡�۲⵽�Ĵ���С��2
			  { //visibleInFrame.size()
                removeObservation(residuals[r].residualBlockId);//��ceres��ɾ������в��
                residuals.erase(residuals.begin() + r);
                r--;
              } else 
              {
                // add information to be considered in marginalization later.
                // e.��Ӹù۲����в�ռ���--------------------------------------------------------
                errorTermAdded = true;
                marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId, false);//�������ͼ��Ĳв����ӵ�marginalizationErrorPtr_��
              }
            }
            // check anything left
             //���õر��Ķ���ͶӰ����Ƿ��޳�����
             //����޳�����˵��imu������û�������ͼ���BAԼ��
            if (residuals.size() == 0) 
			{
              justDelete = true;
              marginalize = false;
            }
          }
        }//���α����������ͼ����صĵĲв��
        
        //�ر��Ĺ۲�ȫ����ɾ�����ˣ���imu������û�������ͼ���BAԼ��
        if(justDelete)
		{
		  //��ceres��ɾ���������ͼ���������صĲв�飻��ceres��ɾ����������顣
          mapPtr_->removeParameterBlock(pit->first);//���� bool Map::removeParameterBlock(uint64_t parameterBlockId) 
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);//�ӵ�ͼ��ɾ�������ͼ��
          continue;
        }
		
		//imu�����в��ܹ۲⵽�õر���ұ�imu����֮���֡�۲⵽�Ĵ�������2 
        if(marginalize&&errorTermAdded)
		{
          paremeterBlocksToBeMarginalized.push_back(pit->first);//��ʾ�����ͼ����Ҫ����Ե��
          keepParameterBlocks.push_back(false);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);//����ǧ��Ҫע��һ��ϸ�� ����Ϊ��������ɾ����landmarksMap_һ��ֵ��ôָ��pit�Զ�ָ������һ��Ԫ��
          continue;
        }

        pit++;//�ƶ����¸���ͼ��
      }//for(PointMap::iterator pit = landmarksMap_.begin();pit != landmarksMap_.end(); )��ͼ�е�ͼ��ѭ������
    }

    // update book-keeping and go to the next frame
    //if(it != statesMap_.begin()){ // let's remember that we kept the very first pose
    if(true) 
	{ ///// DEBUG
      multiFramePtrMap_.erase(it->second.id);
      statesMap_.erase(it->second.id);//���������о�����ɾ����statesMap_������,�������Ҫ����Ե����֡��id
    }
  }//����removeframes����

  // now apply the actual marginalization
  if(paremeterBlocksToBeMarginalized.size()>0)//Ҫ����Ե���Ĳ������������0
  {
    std::vector< ::ceres::ResidualBlockId> addedPriors;
	//���� bool MarginalizationError::marginalizeOut(
	//�ǳ���Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!!! ��ʼִ�б�Ե��
	/// 1&2ͨ��shur��Ԫ����H_��b_����
    //3.��������ceres�Ż��Ĳв�ά�ȡ�ceres�и���costfunction�еĲ�����Ĵ�С��ɾ��ceres�еĲ����顣
    marginalizationErrorPtr_->marginalizeOut(paremeterBlocksToBeMarginalized, keepParameterBlocks);
  }

  // update error computation
  if(paremeterBlocksToBeMarginalized.size()>0)//Ϊ��Ҫ��Ե�������ݿ�id�ļ���
  {
    //�ǳ���Ҫ�ĺ��� void MarginalizationError::updateErrorComputation() 
    //����״̬�ĸ���
    //�ǳ���Ҫ�ĺ���!!!!!!!!!!!!!!!!!!!!!!!
    //�����Ѿ��õ���H���� ���Ƕ�H�������svd�ֽ�Ȼ�����õ�J����
     //����һ���õ�-J.transpose().inverse()*b=e0
    marginalizationErrorPtr_->updateErrorComputation();
  }

  // add the marginalization term again
  if(marginalizationErrorPtr_->num_residuals()==0)
  {
    marginalizationErrorPtr_.reset();//������� ��ζ��Ҫ��ceres�ı�Ե���в�������
  }
  //��ceres���������
  if (marginalizationErrorPtr_) 
  {
	  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock> > parameterBlockPtrs;
	  //���� void MarginalizationError::getParameterBlockPtrs(
	  //�õ�δ����Ե�������������ݿ�
	  marginalizationErrorPtr_->getParameterBlockPtrs(parameterBlockPtrs);
	  //���� ::ceres::ResidualBlockId Map::addResidualBlock(
	  //�������뷶Χ�ھ�ֻ�����ｫ��Ե�����������뵽��ceres�У�û��loss����
	  marginalizationResidualId_ = mapPtr_->addResidualBlock( marginalizationErrorPtr_, NULL, parameterBlockPtrs);//��δ����Ե�������������Ż���ͼ��
	  OKVIS_ASSERT_TRUE_DBG(Exception, marginalizationResidualId_, "could not add marginalization error");
	  if (!marginalizationResidualId_)
	    return false;
  }
	
	if(reDoFixation)//���ɾ����λ������ʼ����ʱ��ᷢ��
	{
	  // finally fix the first pose properly
		//mapPtr_->resetParameterization(statesMap_.begin()->first, ceres::Map::Pose3d);
	  okvis::kinematics::Transformation T_WS_0;
	  get_T_WS(statesMap_.begin()->first, T_WS_0);//��ȡimu���ڵľ��뵱ǰ֡��Զ��֡
	  Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Zero();
	  information(5,5) = 1.0e14; information(0,0) = 1.0e14; 
	  information(1,1) = 1.0e14; information(2,2) = 1.0e14;
	  std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS_0, information));
	  //����::ceres::ResidualBlockId Map::addResidualBlock(
	  //��ceres����������
	  //���²�һ����ʼλ�����
	  mapPtr_->addResidualBlock(poseError,NULL,mapPtr_->parameterBlockPtr(statesMap_.begin()->first));
	}

  return true;
}
//applyMarginalizationStrategy��������--------------------------------------------------------

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
//�����������imu����ֵ����һ��ͼ���λ��T_WS�����õ���λ�˸��µ�T_WS������
//��ʼʱ�����λ��=0����Ҫ��ʹ�ü��ٶȲ���ֵ����̬������⣬��ʼ��̬���ڵ�������ϵ�µ�
//����㷨ʵ���ĵ�
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
  acc_B /= double(imuMeasurements.size());//������ٶȵ�ƽ��ֵ
  Eigen::Vector3d e_acc = acc_B.normalized();

  // align with ez_W:
  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
  Eigen::Matrix<double, 6, 1> poseIncrement;//ǰ��ά��λ��t ����ά�����
  poseIncrement.head<3>() = Eigen::Vector3d::Zero();
  poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();//����õ���
  double angle = std::acos(ez_W.transpose() * e_acc);
  poseIncrement.tail<3>() *= angle;
  T_WS.oplus(-poseIncrement);//poseIncrement��ʾ���

  return true;
}

// Start ceres optimization.
#ifdef USE_OPENMP
void Estimator::optimize(size_t numIter, size_t numThreads,
                                 bool verbose)
#else//Ĭ�Ͻ����������
//Ĭ��verbose=false
//����������ǿ�ʼ�������Ż���ʹ�õ���dogleg���������������ʹ�õ���SPARSE_SCHUR
//����Ż�����=10
//�����µ�ͼ�������
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
#endif//Ĭ�Ͻ����������
  mapPtr_->options.max_num_iterations = numIter;

  if (verbose) //Ĭ��false
  {
    mapPtr_->options.minimizer_progress_to_stdout = true;
  } else {
    mapPtr_->options.minimizer_progress_to_stdout = false;
  }

  // call solver
  //���� void solve() {
  mapPtr_->solve();//������ceres::solve����!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 //�Ż�������....
  // update landmarks
  {
    for(auto it = landmarksMap_.begin(); it!=landmarksMap_.end(); ++it)//�������еĵ�ͼ��
	{
      Eigen::MatrixXd H(3,3);
	  //��������ͼ���Ӧ��Hessian���� 
	  //���� void Map::getLhs(
	  //���������о����������getLhs����
	  //������ǲ�����id����ȡ���������������صĲв�飬����������в��������������ſ˱�J
      //Ȼ�����ǽ�JT*J���ӵ�һ����Ϊ���յ�ֵ����H
      mapPtr_->getLhs(it->first,H);
      //����õ����hessian�������С���������ֵ
      Eigen::SelfAdjointEigenSolver< Eigen::Matrix3d > saes(H);
      Eigen::Vector3d eigenvalues = saes.eigenvalues();
      const double smallest = (eigenvalues[0]);
      const double largest = (eigenvalues[2]);
	  //3.2��������
      if(smallest<1.0e-12)
	  {
        // this means, it has a non-observable depth
        it->second.quality = 0.0;
      } else {
        // OK, well constrained
        it->second.quality = sqrt(smallest)/sqrt(largest);//����quality���������û���õ�
      }

      // update coordinates
      //3.3��������
      it->second.point = std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
          mapPtr_->parameterBlockPtr(it->first))->estimate();//�����Ż���ĵ�ͼ�����꣬���¸�ֵlandmarksMap_�ṹ�е�ͼ������
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
	//  options���� ceres::Solver::Options options;
    mapPtr_->options.callbacks.push_back(ceresCallback_.get());//����ceres���
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
//ceres�Ĳ�������û�������ͼ��Ĳ���
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
//1.��mapPtr_����ȡ����ͼ���Ӧ�Ĳ�����ָ�룬���޸���ֵ
//2.�޸�landmarksMap_�е�ͼ���ֵ
bool Estimator::setLandmark(uint64_t landmarkId, const Eigen::Vector4d & landmark)
{
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(landmarkId);
#ifndef NDEBUG //Ĭ�Ͻ����������
  
  std::shared_ptr<ceres::HomogeneousPointParameterBlock> derivedParameterBlockPtr =
 			 std::dynamic_pointer_cast<ceres::HomogeneousPointParameterBlock>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception,"wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(landmark);//���� void HomogeneousPointParameterBlock::setEstimate(
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
  //�������е���Ϣ �Ѿ�����ʼ����
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
  //getGlobalStateParameterBlockPtr ���������������
  if (!getGlobalStateParameterBlockPtr(poseId, stateType, parameterBlockPtr))
  {
    return false;
  }
#ifndef NDEBUG //Ĭ�Ͻ����������
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
  if (!getGlobalStateParameterBlockAs(poseId, stateType, stateParameterBlock))//���� bool Estimator::getGlobalStateParameterBlockAs(
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
#ifndef NDEBUG//Ĭ�Ͻ����������
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


