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
 *  Created on: Sep 12, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file MarginalizationError.cpp
 * @brief Source file for the MarginalizationError class.
 * @author Stefan Leutenegger
 */

#include <functional>

#include <okvis/ceres/MarginalizationError.hpp>
#include <okvis/ceres/LocalParamizationAdditionalInterfaces.hpp>
#include <okvis/assert_macros.hpp>

/*
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>

*/

//#define USE_NEW_LINEARIZATION_POINT

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

//�������ľ�����һ��3*3���� rows=5,cols=5;
//���½�һ��5*5�������Ͻ�3*3�����Ծ�������ľ���
inline void conservativeResize(Eigen::MatrixXd& matrixXd, int rows, int cols) 
{
  Eigen::MatrixXd tmp(rows, cols);
  const int common_rows = std::min(rows, (int) matrixXd.rows());
  const int common_cols = std::min(cols, (int) matrixXd.cols());
  tmp.topLeftCorner(common_rows, common_cols) = matrixXd.topLeftCorner(common_rows, common_cols);
  matrixXd.swap(tmp);
}

inline void conservativeResize(Eigen::VectorXd& vectorXd, int size) 
{
  if (vectorXd.rows() == 1) 
  {
    Eigen::VectorXd tmp(size);  //Eigen::VectorXd tmp = Eigen::VectorXd::Zero(size,Eigen::RowMajor);
    const int common_size = std::min((int) vectorXd.cols(), size);
    tmp.head(common_size) = vectorXd.head(common_size);
    vectorXd.swap(tmp);
  } else {
    Eigen::VectorXd tmp(size);  //Eigen::VectorXd tmp = Eigen::VectorXd::Zero(size);
    const int common_size = std::min((int) vectorXd.rows(), size);
    tmp.head(common_size) = vectorXd.head(common_size);
    vectorXd.swap(tmp);
  }
}

// Default constructor. Initialises a new okvis::ceres::Map.
MarginalizationError::MarginalizationError() {
  mapPtr_ = 0;
  denseIndices_ = 0;
  residualBlockId_ = 0;
  errorComputationValid_ = false;
}

// Default constructor from okvis::ceres::Map.
MarginalizationError::MarginalizationError(Map& map) {
  setMap(map);
  denseIndices_ = 0;
  residualBlockId_ = 0;
  errorComputationValid_ = false;
}

MarginalizationError::MarginalizationError(
    Map& map, std::vector< ::ceres::ResidualBlockId> & residualBlockIds) {
  setMap(map);
  denseIndices_ = 0;
  residualBlockId_ = 0;
  errorComputationValid_ = false;
  bool success = addResidualBlocks(residualBlockIds);
  OKVIS_ASSERT_TRUE(
      Exception,
      success,
      "residual blocks supplied or their connected parameter blocks were not properly added to the map");
}

// Set the underlying okvis::ceres::Map.
void MarginalizationError::setMap(Map& map) {
  mapPtr_ = &map;
  residualBlockId_ = 0;  // reset.
}

// Add some residuals to this marginalisation error. This means, they will get linearised.
bool MarginalizationError::addResidualBlocks(
    const std::vector< ::ceres::ResidualBlockId> & residualBlockIds,
    const std::vector<bool> & keepResidualBlocks) {
  // add one block after the other
  for (size_t i = 0; i < residualBlockIds.size(); ++i) {
    bool keep = false;
    if (keepResidualBlocks.size() == residualBlockIds.size()) {
      keep = keepResidualBlocks[i];
    }
    if (!addResidualBlock(residualBlockIds[i], keep))
      return false;
  }
  return true;
}

// Add some residuals to this marginalisation error. This means, they will get linearised.
/// \brief Add one residual to this marginalisation error. This means, it will get linearised.
 /// \warning Note that once added here, it will be removed from the okvis::ceres::Map and stay linerised
 ///		  at exactly the point passed here.
 /// @param[in] residualBlockId Residual block id, the corresponding term of which will be added.
 /// @param[in] keepResidualBlock Currently not in use.
//keepĬ��=flase ���keepΪfalse,����Ż���ͼ���Ƴ��òв�
//��������������Ǹ����˱�Ե����H�����Ҵ�ceres��ɾ��Ҫ����Ե���Ĳв��
bool MarginalizationError::addResidualBlock( ::ceres::ResidualBlockId residualBlockId, bool keep) 
{

  // get the residual block & check
  std::shared_ptr<ErrorInterface> errorInterfacePtr =  mapPtr_->errorInterfacePtr(residualBlockId);//��ȡ��residualBlockId���в��Ĳв�
  OKVIS_ASSERT_TRUE_DBG(Exception, errorInterfacePtr, "residual block id does not exist.");
  if (errorInterfacePtr == 0) 
  {
    return false;
  }

  errorComputationValid_ = false;  // flag that the error computation is invalid

  // get the parameter blocks
  //�������ҵ�������Ĳв��id��ص����в�����
  //�����imuerror��������Ӧ������һʱ�̵���̬+��һʱ�̵��ٶȺ�bias+��һʱ�̵���̬+��һʱ�̵��ٶȺ�bias
  //�����ba��������Ӧ����  ��������ϵ��imu����ϵ�ı任(7��ά��)+��ͼ������������ϵ�µĵ�(��Ϊ����ε����꣬����ά�ȵ���4)+imu����ϵ���������ϵ�ı任=ͨ����Ϊ�ǳ����������Ż�(7��ά��)
  //������ٶȺ�bias��Ӧ�Ĳ��������ٶ�+���ٶȺͼ��ٶȵ�bias
  //�����pose�в�����Ӧ�Ĳ�������λ��
  //a.��������Ĳв��õ�����в���Ӧ�Ĳ�����
  Map::ParameterBlockCollection parameters = mapPtr_->parameters(residualBlockId);

  // insert into parameter block ordering book-keeping
  //b.���������飺
  for (size_t i = 0; i < parameters.size(); ++i)
  {
    //�洢���ǲ�����id+������ָ��
    
    Map::ParameterBlockSpec parameterBlockSpec = parameters[i];//��Ӧ��ĳ��������

    // does it already exist as a parameter block connected?
    //����һ���µ����ݽṹ
    ParameterBlockInfo info;
	//parameterBlockId2parameterBlockInfoIdx_Ԫ��= [������id,�������е�λ��]
    std::map<uint64_t, size_t>::iterator it =  parameterBlockId2parameterBlockInfoIdx_.find(parameterBlockSpec.first);
	
    if (it == parameterBlockId2parameterBlockInfoIdx_.end())//�����parameterBlockId2parameterBlockInfoIdx_��û�����������ݿ�
	{  // not found. add it.
        // let's see, if it is actually a landmark, because then it will go into the sparse part
      bool isLandmark = false;
	  //�ж����ݿ��ǲ��ǵر��
      if (std::dynamic_pointer_cast<HomogeneousPointParameterBlock>(parameterBlockSpec.second) != 0) 
	  {
        isLandmark = true;
      }

      // resize equation system
      const size_t origSize = H_.cols();//ԭ��H����Ĵ�С
      size_t additionalSize = 0;
	   //�жϲ������Ƿ�̶�
      if (!parameterBlockSpec.second->fixed()) ////////DEBUG
        additionalSize = parameterBlockSpec.second->minimalDimension();//�¼ӵĲ��������Сά��
      size_t denseSize = 0;
       //parameterBlockInfos_Ϊ���ݿ���Ϣ�ļ���
      //orderingIdx��ʾһ�����ݿ��ں�ɪ��������ʼλ�õ���(��)
      //denseIndices_��ʾһ�����ݿ�����Ϣ�����е�λ��
      if (denseIndices_ > 0)
        denseSize = parameterBlockInfos_.at(denseIndices_ - 1).orderingIdx+ parameterBlockInfos_.at(denseIndices_ - 1).minimalDimension;//H������dense������Ľ���λ��


      //����H��ά���Ͷ�H�������ά�͸���
      if(additionalSize>0) 
	  {
        if (!isLandmark) //��������ݿ��Ҳ��ǵر��
		{
          // insert
          // lhs
          Eigen::MatrixXd H01 = H_.topRightCorner(denseSize,origSize - denseSize);
          Eigen::MatrixXd H10 = H_.bottomLeftCorner(origSize - denseSize, denseSize);
          Eigen::MatrixXd H11 = H_.bottomRightCorner(origSize - denseSize,origSize - denseSize);
          // rhs
          Eigen::VectorXd b1 = b0_.tail(origSize - denseSize);
          //��ʾ������H_,H_��ά��Ϊ(origSize + additionalSize,origSize + additionalSize),��H_�������Ͻǵ�ֵ
          conservativeResize(H_, origSize + additionalSize, origSize + additionalSize);  // lhs
          //����b0_��ά��(origSize + additionalSize),�ұ���b0_�����ֵ
          conservativeResize(b0_, origSize + additionalSize);  // rhs

          H_.topRightCorner(denseSize, origSize - denseSize) = H01;
          H_.bottomLeftCorner(origSize - denseSize, denseSize) = H10;
          H_.bottomRightCorner(origSize - denseSize, origSize - denseSize) = H11;
          H_.block(0, denseSize, H_.rows(), additionalSize).setZero();
          H_.block(denseSize, 0, additionalSize, H_.rows()).setZero();

          b0_.tail(origSize - denseSize) = b1;
          b0_.segment(denseSize, additionalSize).setZero();
        } else //�����ӵ��ǵ�ͼ��
        {
          conservativeResize(H_, origSize + additionalSize, origSize + additionalSize);  // lhs
          conservativeResize(b0_, origSize + additionalSize);  // rhs
          // just append
          b0_.tail(additionalSize).setZero();
          H_.bottomRightCorner(H_.rows(), additionalSize).setZero();
          H_.bottomRightCorner(additionalSize, H_.rows()).setZero();
        }
      }

      // update book-keeping
      if (!isLandmark) //��������鲻�ǵ�ͼ��
	  {
	    //ParameterBlockInfo���캯�� 
	    //���ṹ�е�linearizationPoint����ֵ��
	    //����ĵ�һ�������ǲ������id
	    //�ڶ��������ǲ������ָ��
	    //�����������������������H�����е�λ��
	    //���ĸ�������ʾ����������Ƿ�Ϊ��ͼ��
        info = ParameterBlockInfo(parameterBlockSpec.first, parameterBlockSpec.second, denseSize,isLandmark);
        parameterBlockInfos_.insert(parameterBlockInfos_.begin() + denseIndices_, info);

        parameterBlockId2parameterBlockInfoIdx_.insert(std::pair<uint64_t, size_t>(parameterBlockSpec.first,denseIndices_));

        //  update base_t book-keeping
        //�����ceres�ĺ���
        //����  typedef ::ceres::CostFunction base_t;
        base_t::mutable_parameter_block_sizes()->insert( base_t::mutable_parameter_block_sizes()->begin() + denseIndices_,info.dimension);

        denseIndices_++;  // remember we increased the dense part of the problem

        // also increase the rest
        for (size_t j = denseIndices_; j < parameterBlockInfos_.size(); ++j) 
	 {
		 //parameterBlockInfos_����������е�λ�ã������Ǹ������������Ϣ
          parameterBlockInfos_.at(j).orderingIdx += additionalSize;
          parameterBlockId2parameterBlockInfoIdx_[parameterBlockInfos_.at(j).parameterBlockPtr->id()] += 1;
        }
      } else //����������ǵ�ͼ��
      {
        // just add at the end
        info = ParameterBlockInfo(parameterBlockSpec.first,
        						  parameterBlockSpec.second,
        						  parameterBlockInfos_.back().orderingIdx+ parameterBlockInfos_.back().minimalDimension,
            					  isLandmark);
        parameterBlockInfos_.push_back(info);
        parameterBlockId2parameterBlockInfoIdx_.insert(std::pair<uint64_t, size_t>(parameterBlockSpec.first,parameterBlockInfos_.size() - 1));

        //  update base_t book-keeping
        //�����ceres�ĺ���!!!!!!!!!!!!!!!!�޸�ceres�в�����Ĵ�С
        // typedef ::ceres::CostFunction base_t;
        //number and sizes of input parameter blocks
        base_t::mutable_parameter_block_sizes()->push_back(info.dimension);
      }
      assert(
          parameterBlockInfos_[parameterBlockId2parameterBlockInfoIdx_[parameterBlockSpec.first]].parameterBlockId == parameterBlockSpec.first);
    } else //��ʾparameterBlockId2parameterBlockInfoIdx_�������������
    {

#ifdef USE_NEW_LINEARIZATION_POINT
      // switch linearization point - easy to do on the linearized part...
      size_t i = it->second;
      Eigen::VectorXd Delta_Chi_i(parameterBlockInfos_[i].minimalDimension);
      parameterBlockInfos_[i].parameterBlockPtr->minus(parameterBlockInfos_[i].linearizationPoint.get(),parameterBlockInfos_[i].parameterBlockPtr->parameters(),Delta_Chi_i.data());
      b0_ -= H_.block(0,parameterBlockInfos_[i].orderingIdx,H_.rows(),parameterBlockInfos_[i].minimalDimension)* Delta_Chi_i;
      parameterBlockInfos_[i].resetLinearizationPoint( parameterBlockInfos_[i].parameterBlockPtr);
#endif//Ĭ�Ͻ����������
      info = parameterBlockInfos_.at(it->second);//��ʵԼ����ʲô��û��
    }
  }//�������������

  // update base_t book-keeping on residuals
  // ���� typedef ::ceres::CostFunction base_t;
  //c������ceres���ۺ����Ĳв��ά��
  base_t::set_num_residuals(H_.cols());//�����ceres���������òв��ά�� �������Է���H_������к��е�ά������ͬ��
  double** parametersRaw = new double*[parameters.size()];
  Eigen::VectorXd residualsEigen(errorInterfacePtr->residualDim());//����һ���в�ά��������
  double* residualsRaw = residualsEigen.data();

  double** jacobiansRaw = new double*[parameters.size()];
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::aligned_allocator< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > jacobiansEigen(parameters.size());

  double** jacobiansMinimalRaw = new double*[parameters.size()];
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > jacobiansMinimalEigen(parameters.size());

  for (size_t i = 0; i < parameters.size(); ++i)//���������飬��ÿ����������ſ˱Ⱥͽ���������Ӧ��ϵ
  {
    OKVIS_ASSERT_TRUE_DBG(Exception, isParameterBlockConnected(parameters[i].first),
        "okvis bug: no linearization point, since not connected.");
	//���ݿ������(ԭʼ)����
	//parametersԪ�� = (������id,������ָ�빹��)
    parametersRaw[i] =parameterBlockInfos_[parameterBlockId2parameterBlockInfoIdx_[parameters[i].first]].linearizationPoint.get();  // first estimate Jacobian!!

    //��i�����ݿ���Ÿ��Ⱦ���,�������Ϊ�в��ά��,�������Ϊ���ݿ��ά��
    jacobiansEigen[i].resize(errorInterfacePtr->residualDim(), parameters[i].second->dimension());
    jacobiansRaw[i] = jacobiansEigen[i].data();
	//��i�����ݿ����С�Ÿ��Ⱦ���,�������Ϊ�в��ά��,�������Ϊ���ݿ����Сά��
    jacobiansMinimalEigen[i].resize(errorInterfacePtr->residualDim(),arameters[i].second->minimalDimension());
    jacobiansMinimalRaw[i] = jacobiansMinimalEigen[i].data();
  }

  // evaluate residual block
  //d.����õ�����в����ÿ����������ſ˱Ⱦ���Ͳв�
  errorInterfacePtr->EvaluateWithMinimalJacobians(parametersRaw, residualsRaw,
                                                  jacobiansRaw,//��ʵ������û���õ�
                                                  jacobiansMinimalRaw);


  // correct for loss function if applicable
  //e.�������в��ʹ����loss function��ô�����ſ˱ȺͲв�
  //residualBlockId2ResidualBlockSpecMap�洢���Ǵ��в�����Ϣ,Ԫ�� = (�в���ceres�е�id���в���ceres�е�id+loss function����ָ��+����ָ��)
  //std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError = std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
  ::ceres::LossFunction* lossFunction = mapPtr_->residualBlockId2ResidualBlockSpecMap().find(residualBlockId)->second.lossFunctionPtr;
  if (lossFunction) 
  {
    //����ʵ�ʲ��Է��ֻ�����������!!!!!!!ֻ��BA���Ż��������������Ϊ��
    OKVIS_ASSERT_TRUE_DBG(
        Exception,
        mapPtr_->residualBlockId2ResidualBlockSpecMap().find(residualBlockId)
        != mapPtr_->residualBlockId2ResidualBlockSpecMap().end(),
        "???");

    // following ceres in internal/ceres/corrector.cc
    const double sq_norm = residualsEigen.transpose() * residualsEigen;
    double rho[3];
	//�����ceres�ĺ���!!!!!!!!!!!!!��ceres������lossFunction�����ҵ����������
	//rho�д洢����loss functin������ֵ��һ�׵�����ֵ�����׵�����ֵ
    lossFunction->Evaluate(sq_norm, rho);
    const double sqrt_rho1 = sqrt(rho[1]);
    double residual_scaling;
    double alpha_sq_norm;
    if ((sq_norm == 0.0) || (rho[2] <= 0.0))
	{
      residual_scaling = sqrt_rho1;
      alpha_sq_norm = 0.0;

    } else 
    {
      // Calculate the smaller of the two solutions to the equation
      //
      // 0.5 *  alpha^2 - alpha - rho'' / rho' *  z'z = 0.
      //
      // Start by calculating the discriminant D.
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];

      // Since both rho[1] and rho[2] are guaranteed to be positive at
      // this point, we know that D > 1.0.

      const double alpha = 1.0 - sqrt(D);
      OKVIS_ASSERT_TRUE_DBG(Exception, !std::isnan(alpha), "??");

      // Calculate the constants needed by the correction routines.
      residual_scaling = sqrt_rho1 / (1 - alpha);
      alpha_sq_norm = alpha / sq_norm;
    }

    // correct Jacobians (Equation 11 in BANS)
    //�����ſ˱ȺͲв�
    for (size_t i = 0; i < parameters.size(); ++i) 
	{
      jacobiansMinimalEigen[i] = sqrt_rho1* (jacobiansMinimalEigen[i]- alpha_sq_norm * residualsEigen * (residualsEigen.transpose() * jacobiansMinimalEigen[i]));
    }

    // correct residuals (caution: must be after "correct Jacobians"):
    residualsEigen *= residual_scaling;
  }//lossfunction�жϽ���

  // add blocks to lhs and rhs
  // f.�����Ǽ�������ӵ� H_new �� b_new ��
  for (size_t i = 0; i < parameters.size(); ++i)//����������
  {
    Map::ParameterBlockSpec parameterBlockSpec = parameters[i];//ĳ��������
    //�õ�������������Ϣ
    ParameterBlockInfo parameterBlockInfo_i = parameterBlockInfos_.at(parameterBlockId2parameterBlockInfoIdx_[parameters[i].first]);

    OKVIS_ASSERT_TRUE_DBG(
        Exception,
        parameterBlockInfo_i.parameterBlockId == parameters[i].second->id(),
        "okvis bug: inconstistent okvis ordering");

    if (parameterBlockInfo_i.minimalDimension == 0)
      continue;

    OKVIS_ASSERT_TRUE_DBG(Exception,H_.allFinite(),"WTF1");
	//����H�����еĶԽ�Ԫ��
    //H�е�i�����ݿ������ĺ�ɪ����(i���ݿ�~i���ݿ�)
    H_.block(parameterBlockInfo_i.orderingIdx, parameterBlockInfo_i.orderingIdx,
    		 parameterBlockInfo_i.minimalDimension,parameterBlockInfo_i.minimalDimension) += jacobiansMinimalEigen.at(i).transpose().eval() * jacobiansMinimalEigen.at(i);
    //b�е�i�����ݿ��ֵ
	b0_.segment(parameterBlockInfo_i.orderingIdx,parameterBlockInfo_i.minimalDimension) -= jacobiansMinimalEigen.at(i).transpose().eval() * residualsEigen;

    OKVIS_ASSERT_TRUE_DBG(Exception,H_.allFinite(),
                      "WTF2 " <<jacobiansMinimalEigen.at(i).transpose().eval() * jacobiansMinimalEigen.at(i));
	
	//��i�����ݿ�֮ǰ�����ݿ�j
	//����H�����еķǶԽ�Ԫ��
	 // ���ݿ�(i,j)��Ӧ�ĺ�ɪ�����
    for (size_t j = 0; j < i; ++j) 
	{
      ParameterBlockInfo parameterBlockInfo_j = parameterBlockInfos_.at(parameterBlockId2parameterBlockInfoIdx_[parameters[j].first]);

      OKVIS_ASSERT_TRUE_DBG(
          Exception,
          parameterBlockInfo_j.parameterBlockId == parameters[j].second->id(),
          "okvis bug: inconstistent okvis ordering");

      if (parameterBlockInfo_j.minimalDimension == 0)
        continue;

      // upper triangular:
      H_.block(parameterBlockInfo_i.orderingIdx,parameterBlockInfo_j.orderingIdx,
               parameterBlockInfo_i.minimalDimension,parameterBlockInfo_j.minimalDimension) += jacobiansMinimalEigen.at(i).transpose().eval() * jacobiansMinimalEigen.at(j);
      // lower triangular:
      H_.block(parameterBlockInfo_j.orderingIdx, parameterBlockInfo_i.orderingIdx,
               parameterBlockInfo_j.minimalDimension, parameterBlockInfo_i.minimalDimension) += jacobiansMinimalEigen.at(j).transpose().eval() * jacobiansMinimalEigen.at(i);
    }
  }

  // finally, we also have to delete the nonlinear residual block from the map:
  if (!keep) //�ӵ�ͼ��ɾ������в�飬һ��������������
  {
    //g.��ceres��ɾ������в��
    mapPtr_->removeResidualBlock(residualBlockId);
  }

  // cleanup temporarily allocated stuff
  delete[] parametersRaw;
  delete[] jacobiansRaw;
  delete[] jacobiansMinimalRaw;

  check();/// ������ݽṹ

  return true;
}

// Info: is this parameter block connected to this marginalization error?
bool MarginalizationError::isParameterBlockConnected(
    uint64_t parameterBlockId) {
  OKVIS_ASSERT_TRUE_DBG(Exception, mapPtr_->parameterBlockExists(parameterBlockId),
      "this parameter block does not even exist in the map...");
  std::map<uint64_t, size_t>::iterator it =
      parameterBlockId2parameterBlockInfoIdx_.find(parameterBlockId);
  if (it == parameterBlockId2parameterBlockInfoIdx_.end())
    return false;
  else
    return true;
}

// Checks the internal datastructure (debug)
/// ������ݽṹ
void MarginalizationError::check() {
// check basic sizes
  OKVIS_ASSERT_TRUE_DBG(
      Exception,
      base_t::parameter_block_sizes().size()==parameterBlockInfos_.size(),
      "check failed"); OKVIS_ASSERT_TRUE_DBG(
      Exception,
      parameterBlockId2parameterBlockInfoIdx_.size()==parameterBlockInfos_.size(),
      "check failed"); OKVIS_ASSERT_TRUE_DBG(Exception, base_t::num_residuals()==H_.cols(), "check failed"); OKVIS_ASSERT_TRUE_DBG(Exception, base_t::num_residuals()==H_.rows(), "check failed"); OKVIS_ASSERT_TRUE_DBG(Exception, base_t::num_residuals()==b0_.rows(),
      "check failed"); OKVIS_ASSERT_TRUE_DBG(Exception, parameterBlockInfos_.size()>=denseIndices_,
      "check failed");
  int totalsize = 0;
  // check parameter block sizes
  for (size_t i = 0; i < parameterBlockInfos_.size(); ++i) 
  {
    totalsize += parameterBlockInfos_[i].minimalDimension;
    OKVIS_ASSERT_TRUE_DBG(
        Exception,
        parameterBlockInfos_[i].dimension==size_t(base_t::parameter_block_sizes()[i]),
        "check failed"); OKVIS_ASSERT_TRUE_DBG(
        Exception,
        mapPtr_->parameterBlockExists(parameterBlockInfos_[i].parameterBlockId),
        "check failed"); OKVIS_ASSERT_TRUE_DBG(
        Exception,
        parameterBlockId2parameterBlockInfoIdx_[parameterBlockInfos_[i].parameterBlockId]==i,
        "check failed");
    if (i < denseIndices_) 
	{
      OKVIS_ASSERT_TRUE_DBG(Exception, !parameterBlockInfos_[i].isLandmark,
          "check failed");
    } else {
      OKVIS_ASSERT_TRUE_DBG(Exception, parameterBlockInfos_[i].isLandmark,
                        "check failed");
    }

  }
  // check contiguous
  for (size_t i = 1; i < parameterBlockInfos_.size(); ++i) 
  {
    OKVIS_ASSERT_TRUE_DBG(
        Exception,
        parameterBlockInfos_[i-1].orderingIdx+parameterBlockInfos_[i-1].minimalDimension==parameterBlockInfos_[i].orderingIdx,
        "check failed "<<parameterBlockInfos_[i-1].orderingIdx<<"+"<<parameterBlockInfos_[i-1].minimalDimension<<"=="<<parameterBlockInfos_[i].orderingIdx);
  }
// check dimension again
  OKVIS_ASSERT_TRUE_DBG(Exception, base_t::num_residuals()==totalsize, "check failed");
}

// Call this in order to (re-)add this error term after whenever it had been modified.
void MarginalizationError::getParameterBlockPtrs(  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock> >& parameterBlockPtrs) 
{
//OKVIS_ASSERT_TRUE_DBG(Exception,_errorComputationValid,"Call updateErrorComputation() before addToMap!");
  OKVIS_ASSERT_TRUE_DBG(Exception, mapPtr_!=0, "no Map object passed ever!");
  for (size_t i = 0; i < parameterBlockInfos_.size(); ++i) 
  {
    parameterBlockPtrs.push_back(parameterBlockInfos_[i].parameterBlockPtr);
  }
}

// Marginalise out a set of parameter blocks.
/// ��Ե�����ݿ�
/// 1&2ͨ��shur��Ԫ����H_��b_����
//3.��������ceres�Ż��Ĳв�ά�ȡ�ceres�и���costfunction�еĲ�����Ĵ�С��ɾ��ceres�еĲ����顣
//parameterBlockIds Ԫ�� = Ҫ��Ե���Ĳ������id
bool MarginalizationError::marginalizeOut(const std::vector<uint64_t>& parameterBlockIds,const std::vector<bool> & keepParameterBlocks) 
{
  if (parameterBlockIds.size() == 0) {
    return false;
  }

  // copy so we can manipulate
  //parameterBlockIdsCopy�д洢�Ĳ����� û���ظ���id������ᴦ��
  std::vector<uint64_t> parameterBlockIdsCopy = parameterBlockIds;
  if (parameterBlockIds.size() != keepParameterBlocks.size()) //�����ϲ��ᷢ��
  {
    OKVIS_ASSERT_TRUE_DBG(
        Exception,
        keepParameterBlocks.size() == 0,
        "input vectors must either be of same size or omit optional parameter keepParameterBlocks: "<<
        parameterBlockIds.size()<<" vs "<<keepParameterBlocks.size());
  }
  std::map<uint64_t, bool> keepParameterBlocksCopy;//��ӣ����ݿ���š�bool����keepParameterBlocksCopy��
  for (size_t i = 0; i < parameterBlockIdsCopy.size(); ++i) 
  {
    bool keep = false;
    if (i < keepParameterBlocks.size()) 
	{
      keep = keepParameterBlocks.at(i);//�����϶���false
    }
    keepParameterBlocksCopy.insert(std::pair<uint64_t, bool>(parameterBlockIdsCopy.at(i), keep));
  }

  /* figure out which blocks need to be marginalized out */
  std::vector<std::pair<int, int> > marginalizationStartIdxAndLengthPairslandmarks;//Ԫ�� = (��ͼ���������H�����е�λ�ã���Сά��)
  std::vector<std::pair<int, int> > marginalizationStartIdxAndLengthPairsDense;//Ԫ�� = (�������зǵ�ͼ����H�����е�λ�ã���Сά��)
  size_t marginalizationParametersLandmarks = 0;//��¼Ҫ����Ե���Ĳ������е�ͼ�����ά��
  size_t marginalizationParametersDense = 0;//��¼Ҫ����Ե���Ĳ������зǵ�ͼ�����ά��

  // make sure no duplications...
  //ɾ��parameterBlockIdsCopy���ظ��Ĳ�����
  std::sort(parameterBlockIdsCopy.begin(), parameterBlockIdsCopy.end());//�������򣩣����ݿ���ŵĴ�С��������
  for (size_t i = 1; i < parameterBlockIdsCopy.size(); ++i) 
  {
    if (parameterBlockIdsCopy[i] == parameterBlockIdsCopy[i - 1]) 
    {
      parameterBlockIdsCopy.erase(parameterBlockIdsCopy.begin() + i);//ɾ����ͬ�����ݿ�
      --i;
    }
  }
  
  for (size_t i = 0; i < parameterBlockIdsCopy.size(); ++i) 
  {
    //���ݲ�����id 
    std::map<uint64_t, size_t>::iterator it = parameterBlockId2parameterBlockInfoIdx_.find(parameterBlockIdsCopy[i]);

    // sanity check - are we trying to marginalize stuff that is not connected to this error term?
    OKVIS_ASSERT_TRUE(
        Exception,
        it != parameterBlockId2parameterBlockInfoIdx_.end(),
        "trying to marginalize out unconnected parameter block id = "<<parameterBlockIdsCopy[i])
    if (it == parameterBlockId2parameterBlockInfoIdx_.end())
      return false;

    // distinguish dense and landmark (sparse) part for more efficient pseudo-inversion later on
    size_t startIdx = parameterBlockInfos_.at(it->second).orderingIdx;//�ҵ������������H�����е�λ��
    size_t minDim = parameterBlockInfos_.at(it->second).minimalDimension;//������������Сά��
    if (parameterBlockInfos_.at(it->second).isLandmark) 
   {
      marginalizationStartIdxAndLengthPairslandmarks.push_back(std::pair<int, int>(startIdx, minDim));
      marginalizationParametersLandmarks += minDim;//���±���Ե���Ĳ������е�ͼ�����ά��
    } else {
      marginalizationStartIdxAndLengthPairsDense.push_back(std::pair<int, int>(startIdx, minDim));
      marginalizationParametersDense += minDim;//���±���Ե���Ĳ������зǵ�ͼ�����ά��
    }
  }

  // make sure the marginalization pairs are ordered
   // �����������ݣ��ر�����ݿ���H�����ʼλ�ã���������
  std::sort(marginalizationStartIdxAndLengthPairslandmarks.begin(),
            marginalizationStartIdxAndLengthPairslandmarks.end(),
            [](std::pair<int,int> left, std::pair<int,int> right) {
              return left.first < right.first;
            });
  //�����������ݣ��ǵر�����ݿ���H�����ʼλ�ã���������
  std::sort(marginalizationStartIdxAndLengthPairsDense.begin(),
            marginalizationStartIdxAndLengthPairsDense.end(),
            [](std::pair<int,int> left, std::pair<int,int> right) {
              return left.first < right.first;
            });

  // unify contiguous marginalization requests
  //�������Ĳ����������һ��
  for (size_t m = 1; m < marginalizationStartIdxAndLengthPairslandmarks.size(); ++m) 
  {
     //�����һ�����ݿ飨����Ե���ģ���H������+��һ�����ݿ飨����Ե��������Сά��=��ǰ���ݿ飨����Ե��������㣨������
    //ע�����Ե���ĵر�㲻һ����������
    if (marginalizationStartIdxAndLengthPairslandmarks.at(m - 1).first + marginalizationStartIdxAndLengthPairslandmarks.at(m - 1).second == marginalizationStartIdxAndLengthPairslandmarks.at(m).first) 
   {
      marginalizationStartIdxAndLengthPairslandmarks.at(m - 1).second +=  marginalizationStartIdxAndLengthPairslandmarks.at(m).second;
      marginalizationStartIdxAndLengthPairslandmarks.erase( marginalizationStartIdxAndLengthPairslandmarks.begin() + m);
      --m;
    }
  }
  for (size_t m = 1; m < marginalizationStartIdxAndLengthPairsDense.size(); ++m) 
  {
    //�����һ�����ݿ飨����Ե���ģ���H������+��һ�����ݿ飨����Ե��������Сά��=��ǰ���ݿ飨����Ե��������㣨������
    //ע�����Ե���ķǵر�����ݿ鲻һ����������
    if (marginalizationStartIdxAndLengthPairsDense.at(m - 1).first+ marginalizationStartIdxAndLengthPairsDense.at(m - 1).second == marginalizationStartIdxAndLengthPairsDense.at(m).first) 
   {
      marginalizationStartIdxAndLengthPairsDense.at(m - 1).second += marginalizationStartIdxAndLengthPairsDense.at(m).second;
      marginalizationStartIdxAndLengthPairsDense.erase( marginalizationStartIdxAndLengthPairsDense.begin() + m);
      --m;
    }
  }

  errorComputationValid_ = false;  // flag that the error computation is invalid

  // include in the fix rhs part deviations from linearization point of the parameter blocks to be marginalized
  // corrected: this is not necessary, will cancel itself

  /* landmark part (if existing) */
  //1.��ʼ����Ҫ����Ե���ĵ�ͼ��
  if (marginalizationStartIdxAndLengthPairslandmarks.size() > 0) 
  {

    // preconditioner
     //��ȡH_����ĶԽ�����Ԫ��,����Ԫ���������1.0e-9,ѡ�� p=H_����Խ�����Ԫ�ص�ƽ����,�������ֵ��ֵΪ10^-3
    Eigen::VectorXd p = (H_.diagonal().array() > 1.0e-9).select(H_.diagonal().cwiseSqrt(),1.0e-3);
    Eigen::VectorXd p_inv = p.cwiseInverse();//���Ԫ������

    // scale H and b
    //����H_�ĳ߶� Ŀ���ǲ�����ֵ̫С
    //asDiagonal��������ɶԽǾ���
    H_ = p_inv.asDiagonal() * H_ * p_inv.asDiagonal();
    b0_ = p_inv.asDiagonal() * b0_;

    Eigen::MatrixXd U(H_.rows() - marginalizationParametersLandmarks, H_.rows() - marginalizationParametersLandmarks);
    Eigen::MatrixXd V(marginalizationParametersLandmarks, marginalizationParametersLandmarks);
    Eigen::MatrixXd W(H_.rows() - marginalizationParametersLandmarks, marginalizationParametersLandmarks);
    Eigen::VectorXd b_a(H_.rows() - marginalizationParametersLandmarks);
    Eigen::VectorXd b_b(marginalizationParametersLandmarks);

    // split preconditioner
    Eigen::VectorXd p_a(H_.rows() - marginalizationParametersLandmarks);//���ٵ�ͼ����ά��
    Eigen::VectorXd p_b(marginalizationParametersLandmarks);//����Ե���ر���ά��
    //����pΪp_a,p_b��p_aΪ���漰��Ե����p��Ԫ��, p_bΪ��Ҫ��Ե����p��Ԫ��
    splitVector(marginalizationStartIdxAndLengthPairslandmarks, p, p_a, p_b);  // output

    // split lhs
    //����H_����,����U�ǲ��漰��Ե����H�е�Ԫ��,W�Ǵ���Ե�����ݿ�ͱ������ݿ�Ľ��溣ɪ����,VΪ����Ե�����ݿ�ĺ�ɪ����
    splitSymmetricMatrix(marginalizationStartIdxAndLengthPairslandmarks, H_, U,W, V);  // output

    // split rhs
    //����b����,b_aΪ���漰��Ե����b��Ԫ��,b_bΪ����Ե����b��Ԫ��
    splitVector(marginalizationStartIdxAndLengthPairslandmarks, b0_, b_a, b_b);  // output


    // invert the marginalization block
    static const int sdim =::okvis::ceres::HomogeneousPointParameterBlock::MinimalDimension;//Ϊ3
    
    b0_.resize(b_a.rows());
    b0_ = b_a;
    H_.resize(U.rows(), U.cols());
    H_ = U;
    const size_t numBlocks = V.cols() / sdim;//Ҫ����Ե���ĵ�ͼ��ĸ���
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> delta_H( numBlocks);
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> delta_b( numBlocks);
    Eigen::MatrixXd M1(W.rows(), W.cols());//M1 ���� H12*H22.inverse��
    size_t idx = 0;
    for (size_t i = 0; int(i) < V.cols(); i += sdim) 
   {
      Eigen::Matrix<double, sdim, sdim> V_inv_sqrt;
      Eigen::Matrix<double, sdim, sdim> V1 = V.block(i, i, sdim, sdim);
      MarginalizationError::pseudoInverseSymmSqrt(V1, V_inv_sqrt);//V_inv_sqrtΪV1�Ĺ����濪����
      Eigen::MatrixXd M = W.block(0, i, W.rows(), sdim) * V_inv_sqrt;//M ���� H12H22^0.5
      Eigen::MatrixXd M1 = W.block(0, i, W.rows(), sdim) * V_inv_sqrt* V_inv_sqrt.transpose();
      // accumulate
      delta_H.at(idx).resize(U.rows(), U.cols());
      delta_b.at(idx).resize(b_a.rows());
      if (i == 0) 
	{
        delta_H.at(idx) = M * M.transpose();
        delta_b.at(idx) = M1 * b_b.segment<sdim>(i);
      } else {
        delta_H.at(idx) = delta_H.at(idx - 1) + M * M.transpose();
        delta_b.at(idx) = delta_b.at(idx - 1) + M1 * b_b.segment<sdim>(i);
      }
      ++idx;
    }
    // Schur
    // Schur��Ԫ,�൱��ͨ����˹��Ԫ,������Ե���ĵر����������Է��������޳�
    b0_ -= delta_b.at(idx - 1);
    H_ -= delta_H.at(idx - 1);

    // unscale
    // ��ȥ��һ����Ӱ��
    H_ = p_a.asDiagonal() * H_ * p_a.asDiagonal();
    b0_ = p_a.asDiagonal() * b0_;
  }//����Ե���ĵ�ͼ�㴦�����

  /* dense part (if existing) */
  //2.��ʼ�����Ե���ķǵ�ͼ���״̬-ע��û��ʹ��ѭ��
  if (marginalizationStartIdxAndLengthPairsDense.size() > 0) 
  {

    // preconditioner
      //��ȡH_����ĶԽ�����Ԫ��,����Ԫ���������1.0e-9,ѡ�� p=H_����Խ�����Ԫ�ص�ƽ����
    Eigen::VectorXd p = (H_.diagonal().array() > 1.0e-9).select(H_.diagonal().cwiseSqrt(),1.0e-3);
    Eigen::VectorXd p_inv = p.cwiseInverse();

    // scale H and b
    ///asDiagonal��������ɶԽǾ���
    H_ = p_inv.asDiagonal() * H_ * p_inv.asDiagonal();
    b0_ = p_inv.asDiagonal() * b0_;

    Eigen::MatrixXd U(H_.rows() - marginalizationParametersDense,
                      H_.rows() - marginalizationParametersDense);
    Eigen::MatrixXd V(marginalizationParametersDense,
                      marginalizationParametersDense);
    Eigen::MatrixXd W(H_.rows() - marginalizationParametersDense,
                      marginalizationParametersDense);
    Eigen::VectorXd b_a(H_.rows() - marginalizationParametersDense);
    Eigen::VectorXd b_b(marginalizationParametersDense);

    // split preconditioner
    Eigen::VectorXd p_a(H_.rows() - marginalizationParametersDense);
    Eigen::VectorXd p_b(marginalizationParametersDense);
    splitVector(marginalizationStartIdxAndLengthPairsDense, p, p_a, p_b);  // output

    // split lhs
    splitSymmetricMatrix(marginalizationStartIdxAndLengthPairsDense, H_, U, W, V);  // output

    // split rhs
    splitVector(marginalizationStartIdxAndLengthPairsDense, b0_, b_a, b_b);  // output �����ôû���õ�B_b?

    // invert the marginalization block
    Eigen::MatrixXd V_inverse_sqrt(V.rows(), V.cols());
    Eigen::MatrixXd V1 = 0.5 * (V + V.transpose());//���������ʹ�õ���V��Vת�õĶ���֮һ��ֱ����V����ô?
    pseudoInverseSymmSqrt(V1, V_inverse_sqrt);

    // Schur
    Eigen::MatrixXd M = W * V_inverse_sqrt;
    // rhs
    b0_.resize(b_a.rows());
    b0_ = (b_a - M * V_inverse_sqrt.transpose() * b_b);
    // lh* b_bs
    H_.resize(U.rows(), U.cols());

    H_ = (U - M * M.transpose());

    // unscale
    H_ = p_a.asDiagonal() * H_ * p_a.asDiagonal();
    b0_ = p_a.asDiagonal() * b0_;
  }//�������Ե���ķǵ�ͼ���״̬

 //���ϵ�����������Ҫ�Ǹ�����schur����H��b����
  //3.����ceres�вв��ά��=ԭ���в�ά��-��Ե����ͼ��ά��-��Ե���ǵ�ͼ��ά�ȡ�
  //ceres�и���costfunction�еĲ�����Ĵ�С=ɾ�������顣ɾ��ceres�еĲ����顣��ceres��ɾ������в�飬
  // also adapt the ceres-internal size information
  //���� typedef ::ceres::CostFunction base_t;
  //����ceres�в��ά�� base_t::num_residuals()=���ڲв��ά��
  //����һ��ceres�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  base_t::set_num_residuals(base_t::num_residuals() - marginalizationParametersDense - marginalizationParametersLandmarks);

  /* delete all the book-keeping */
  for (size_t i = 0; i < parameterBlockIdsCopy.size(); ++i)//����Ҫ��Ե���Ĳ����� 
  {
    size_t idx = parameterBlockId2parameterBlockInfoIdx_.find( parameterBlockIdsCopy[i])->second;//�õ������ݿ��Ӧ����Ϣλ��(��parameterBlockInfos_��)
    int margSize = parameterBlockInfos_.at(idx).minimalDimension;//��ȡ���ݿ��ά��
    parameterBlockInfos_.erase(parameterBlockInfos_.begin() + idx);//�����ݿ��Ӧ����Ϣ����Ϣ������ɾ��

    for (size_t j = idx; j < parameterBlockInfos_.size(); ++j) 
	{
      parameterBlockInfos_.at(j).orderingIdx -= margSize;
      parameterBlockId2parameterBlockInfoIdx_.at( parameterBlockInfos_.at(j).parameterBlockId) -= 1;
    }

    parameterBlockId2parameterBlockInfoIdx_.erase(parameterBlockIdsCopy[i]);//ɾ�������ݿ���parameterBlockId2parameterBlockInfoIdx_�ж�Ӧ��ֵ

    // also adapt the ceres-internal book-keepin
     //mutable_parameter_block_sizes����������������Ϊvector<int32>,�洢����number and sizes of input parameter blocks
     //ceres�и���costfunction�еĲ�����Ĵ�С
     //����һ��ceres�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    base_t::mutable_parameter_block_sizes()->erase( mutable_parameter_block_sizes()->begin() + idx);//����һ��ceres�ĺ���
  }

  /* assume everything got dense */
  // this is a conservative assumption, but true in particular when marginalizing
  // poses w/o landmarks
   ///�������е�δ��ɾ�������ݿ鶼���ǵر��
  /// parameterBlockInfos_��ά��Ӧ�ñ�parameterBlockIdsCopy�Դ�
  /// ���������еĵر��1���Ա�1,2,3�ؼ�֡����,�������Ӧ��������ͶӰ���,�ֱ��������ͶӰ���ʱ,���parameterBlockInfos_����߸����ݿ�,��:-------
  /// ----------------1,2,3�ؼ�֡��λ��,���,�Լ��ر��1������
  /// ��parameterBlockIdsCopyֻ�еر��1�͹ؼ�֡1��λ�˺����
  denseIndices_ = parameterBlockInfos_.size();
  for (size_t i = 0; i < parameterBlockInfos_.size(); ++i) 
  {
    if (parameterBlockInfos_.at(i).isLandmark) 
	{
      parameterBlockInfos_.at(i).isLandmark = false;
    }
  }

   /*�������ǰѺʹ��밲ȫ����ص�ע�͵��ˣ����㿴����
  // check if the removal is safe
  for (size_t i = 0; i < parameterBlockIdsCopy.size(); ++i) 
  {
    Map::ResidualBlockCollection residuals = mapPtr_->residuals(parameterBlockIdsCopy[i]);//���ݲ������ҵ����еĲв��
    if (residuals.size() != 0 && keepParameterBlocksCopy.at(parameterBlockIdsCopy[i]) == false)
      mapPtr_->printParameterBlockInfo(parameterBlockIdsCopy[i]);
    OKVIS_ASSERT_TRUE_DBG(
        Exception,
        residuals.size()==0 || keepParameterBlocksCopy.at(parameterBlockIdsCopy[i]) == true,
        "trying to marginalize out a parameterBlock that is still connected to other error terms."
        <<" keep = "<<int(keepParameterBlocksCopy.at(parameterBlockIdsCopy[i])));
  }
  */
    for (size_t i = 0; i < parameterBlockIdsCopy.size(); ++i) 
  {
    if (keepParameterBlocksCopy.at(parameterBlockIdsCopy[i])) 
	{
      OKVIS_THROW(Exception,"unmarginalizeLandmark(parameterBlockIdsCopy[i]) not implemented.")
    } else//һ�㶼������������
    {
      //��ceres��ɾ���������������صĲв�飻��ceres��ɾ����������顣
      //������ceres�ĺ���!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      mapPtr_->removeParameterBlock(parameterBlockIdsCopy[i]);
    }
  }

  check();

  return true;
}

// This must be called before optimization after adding residual blocks and/or marginalizing,
// since it performs all the lhs and rhs computations on from a given _H and _b.
/// ����H��b�����Ż�֮ǰ����Ӳв��ͱ�Ե��֮��
//�����Ѿ��õ���H���� ���Ƕ�H�������svd�ֽ�Ȼ�����õ�J����
//����һ���õ�-J.transpose().inverse()*b=e0
void MarginalizationError::updateErrorComputation() 
{
  if (errorComputationValid_)
    return;  // already done.

  // now we also know the error dimension:
  //0.����ceres���ۺ����Ĳв�ά��
  base_t::set_num_residuals(H_.cols());//ceres�ĺ���

  // preconditioner
  Eigen::VectorXd p = (H_.diagonal().array() > 1.0e-9).select(H_.diagonal().cwiseSqrt(),1.0e-3);
  Eigen::VectorXd p_inv = p.cwiseInverse();

  // lhs SVD: _H = J^T*J = _U*S*_U^T
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes( 0.5  * p_inv.asDiagonal() * (H_ + H_.transpose())  * p_inv.asDiagonal() );

  static const double epsilon = std::numeric_limits<double>::epsilon();//���б������ļ��������ʶ�����С���㸡������
  double tolerance = epsilon * H_.cols()* saes.eigenvalues().array().maxCoeff();//��С���㸡����*H_��ά��*�������ֵ
  S_ = Eigen::VectorXd(  (saes.eigenvalues().array() > tolerance).select( saes.eigenvalues().array(), 0));//����ֵ�Խ���
  S_pinv_ = Eigen::VectorXd( (saes.eigenvalues().array() > tolerance).select( saes.eigenvalues().array().inverse(), 0));//����ֵ�����ĶԽ���

  S_sqrt_ = S_.cwiseSqrt();//S_ÿ��Ԫ�ص�ƽ����
  S_pinv_sqrt_ = S_pinv_.cwiseSqrt();//S_pinv_ÿ��Ԫ�ص�ƽ����

  // assign Jacobian
   /// ͨ��SVD����sqrt(H),��J.
  J_ = (p.asDiagonal() * saes.eigenvectors() * (S_sqrt_.asDiagonal())).transpose();

  // constant error (residual) _e0 := (-pinv(J^T) * _b):
  Eigen::MatrixXd J_pinv_T = (S_pinv_sqrt_.asDiagonal())* saes.eigenvectors().transpose()  *p_inv.asDiagonal() ;
  e0_ = (-J_pinv_T * b0_);//�õ�-J.transpose().inverse()*b

  // reconstruct. TODO: check if this really improves quality --- doesn't seem so...
  //H_ = J_.transpose() * J_;
  //b0_ = -J_.transpose() * e0_;
  errorComputationValid_ = true;
}

// Computes the linearized deviation from the references (linearization points)
bool MarginalizationError::computeDeltaChi(Eigen::VectorXd& DeltaChi) const {
  DeltaChi.resize(H_.rows());
  for (size_t i = 0; i < parameterBlockInfos_.size(); ++i) {
    // stack Delta_Chi vector
    if (!parameterBlockInfos_[i].parameterBlockPtr->fixed()) {
      Eigen::VectorXd Delta_Chi_i(parameterBlockInfos_[i].minimalDimension);
      parameterBlockInfos_[i].parameterBlockPtr->minus(
          parameterBlockInfos_[i].linearizationPoint.get(),
          parameterBlockInfos_[i].parameterBlockPtr->parameters(),
          Delta_Chi_i.data());
      DeltaChi.segment(parameterBlockInfos_[i].orderingIdx,
                       parameterBlockInfos_[i].minimalDimension) = Delta_Chi_i;
			}
  }
  return true;
}

// Computes the linearized deviation from the references (linearization points)
//ʹ�õ�ǰ��״̬���Ե��֮ǰ��״̬����õ�Delta_Chi
bool MarginalizationError::computeDeltaChi(double const* const * parameters,
                                           Eigen::VectorXd& DeltaChi) const 
{
  DeltaChi.resize(H_.rows());
  //parameterBlockInfos_����ǲ�������H�����е�λ�� �����Ǹ������������Ϣ
  for (size_t i = 0; i < parameterBlockInfos_.size(); ++i) 
  {
	    // stack Delta_Chi vector
	    if (!parameterBlockInfos_[i].parameterBlockPtr->fixed()) 
	    {
	      Eigen::VectorXd Delta_Chi_i(parameterBlockInfos_[i].minimalDimension);
		  //��һ���͵ڶ�������������Ĳ�����������������ǰ��������������õ���Delta_Chi_i
		  //�����minus�Ǹ���״̬�Լ������
	      parameterBlockInfos_[i].parameterBlockPtr->minus(parameterBlockInfos_[i].linearizationPoint.get(), parameters[i],Delta_Chi_i.data());
	      DeltaChi.segment(parameterBlockInfos_[i].orderingIdx, parameterBlockInfos_[i].minimalDimension) = Delta_Chi_i;
	    }
  }
  return true;
}

//This evaluates the error term and additionally computes the Jacobians.
//�ſ˱Ƚ����� jacobian analytic
bool MarginalizationError::Evaluate(double const* const * parameters,
                                    double* residuals,
                                    double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool MarginalizationError::EvaluateWithMinimalJacobians(double const* const * parameters, double* residuals, double** jacobians, double** jacobiansMinimal) const 
{
  OKVIS_ASSERT_TRUE_DBG(
      Exception,
      errorComputationValid_,
      "trying to opmimize, but updateErrorComputation() was not called after adding residual blocks/marginalizing");

  Eigen::VectorXd Delta_Chi;
  computeDeltaChi(parameters, Delta_Chi);//ʹ�õ�ǰ��״̬���Ե��֮ǰ��״̬����õ�Delta_Chi

  for (size_t i = 0; i < parameterBlockInfos_.size(); ++i) 
  {

    // decompose the jacobians: minimal ones are easy
    if (jacobiansMinimal != NULL) 
   {
      if (jacobiansMinimal[i] != NULL) 
      {
        Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > Jmin_i(jacobiansMinimal[i], e0_.rows(), parameterBlockInfos_[i].minimalDimension);
        Jmin_i = J_.block(0, parameterBlockInfos_[i].orderingIdx, e0_.rows(),parameterBlockInfos_[i].minimalDimension);
      }
    }

    // hallucinate the non-minimal Jacobians
    if (jacobians != NULL) 
   {
      if (jacobians[i] != NULL) 
	  {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > J_i(jacobians[i], e0_.rows(),parameterBlockInfos_[i].dimension);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Jmin_i =J_.block(0, parameterBlockInfos_[i].orderingIdx, e0_.rows(),parameterBlockInfos_[i].minimalDimension);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> J_lift(parameterBlockInfos_[i].parameterBlockPtr->minimalDimension(), parameterBlockInfos_[i].parameterBlockPtr->dimension());
        parameterBlockInfos_[i].parameterBlockPtr->liftJacobian(parameterBlockInfos_[i].linearizationPoint.get(), J_lift.data());

        J_i = Jmin_i * J_lift;
      }
    }
  }

  // finally the error (residual) e = (-pinv(J^T) * _b + _J*Delta_Chi):
  Eigen::Map<Eigen::VectorXd> e(residuals, e0_.rows());
  e = e0_ + J_ * Delta_Chi;//����õ��в� e0��J_�����Ե��֮������ڲ����

  return true;
}

}  // namespace ceres
}  // namespace okvis

