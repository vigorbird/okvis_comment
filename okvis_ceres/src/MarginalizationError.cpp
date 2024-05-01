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

//如果输入的矩阵是一个3*3矩阵 rows=5,cols=5;
//则新建一个5*5矩阵，左上角3*3矩阵仍旧是输入的矩阵
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
//keep默认=flase 如果keep为false,则从优化地图中移除该残差
//这个函数的作用是更新了边缘化的H矩阵并且从ceres中删除要被边缘化的残差块
bool MarginalizationError::addResidualBlock( ::ceres::ResidualBlockId residualBlockId, bool keep) 
{

  // get the residual block & check
  std::shared_ptr<ErrorInterface> errorInterfacePtr =  mapPtr_->errorInterfacePtr(residualBlockId);//提取第residualBlockId个残差块的残差
  OKVIS_ASSERT_TRUE_DBG(Exception, errorInterfacePtr, "residual block id does not exist.");
  if (errorInterfacePtr == 0) 
  {
    return false;
  }

  errorComputationValid_ = false;  // flag that the error computation is invalid

  // get the parameter blocks
  //作用是找到与输入的残差块id相关的所有参数块
  //如果是imuerror则参数块对应的是上一时刻的姿态+上一时刻的速度和bias+这一时刻的姿态+这一时刻的速度和bias
  //如果是ba则参数块对应的是  世界坐标系到imu坐标系的变换(7个维度)+地图点在世界坐标系下的点(因为是齐次点坐标，所以维度等于4)+imu坐标系到相机坐标系的变换=通常认为是常量不进行优化(7个维度)
  //如果是速度和bias对应的参数块是速度+角速度和加速度的bias
  //如果是pose残差块则对应的参数块是位姿
  //a.根据输入的残差块得到这个残差块对应的参数块
  Map::ParameterBlockCollection parameters = mapPtr_->parameters(residualBlockId);

  // insert into parameter block ordering book-keeping
  //b.遍历参数块：
  for (size_t i = 0; i < parameters.size(); ++i)
  {
    //存储的是参数块id+参数块指针
    
    Map::ParameterBlockSpec parameterBlockSpec = parameters[i];//对应的某个参数块

    // does it already exist as a parameter block connected?
    //这是一个新的数据结构
    ParameterBlockInfo info;
	//parameterBlockId2parameterBlockInfoIdx_元素= [参数块id,在容器中的位置]
    std::map<uint64_t, size_t>::iterator it =  parameterBlockId2parameterBlockInfoIdx_.find(parameterBlockSpec.first);
	
    if (it == parameterBlockId2parameterBlockInfoIdx_.end())//如果在parameterBlockId2parameterBlockInfoIdx_中没有搜索到数据块
	{  // not found. add it.
        // let's see, if it is actually a landmark, because then it will go into the sparse part
      bool isLandmark = false;
	  //判断数据块是不是地标点
      if (std::dynamic_pointer_cast<HomogeneousPointParameterBlock>(parameterBlockSpec.second) != 0) 
	  {
        isLandmark = true;
      }

      // resize equation system
      const size_t origSize = H_.cols();//原来H矩阵的大小
      size_t additionalSize = 0;
	   //判断参数块是否固定
      if (!parameterBlockSpec.second->fixed()) ////////DEBUG
        additionalSize = parameterBlockSpec.second->minimalDimension();//新加的参数块的最小维度
      size_t denseSize = 0;
       //parameterBlockInfos_为数据块信息的集合
      //orderingIdx表示一个数据块在海瑟矩阵中起始位置的行(列)
      //denseIndices_表示一个数据块在信息容器中的位置
      if (denseIndices_ > 0)
        denseSize = parameterBlockInfos_.at(denseIndices_ - 1).orderingIdx+ parameterBlockInfos_.at(denseIndices_ - 1).minimalDimension;//H矩阵中dense参数快的结束位置


      //计算H的维数和对H阵进行扩维和更新
      if(additionalSize>0) 
	  {
        if (!isLandmark) //添加了数据块且不是地标点
		{
          // insert
          // lhs
          Eigen::MatrixXd H01 = H_.topRightCorner(denseSize,origSize - denseSize);
          Eigen::MatrixXd H10 = H_.bottomLeftCorner(origSize - denseSize, denseSize);
          Eigen::MatrixXd H11 = H_.bottomRightCorner(origSize - denseSize,origSize - denseSize);
          // rhs
          Eigen::VectorXd b1 = b0_.tail(origSize - denseSize);
          //表示：重置H_,H_的维数为(origSize + additionalSize,origSize + additionalSize),且H_保留左上角的值
          conservativeResize(H_, origSize + additionalSize, origSize + additionalSize);  // lhs
          //重置b0_的维数(origSize + additionalSize),且保留b0_上面的值
          conservativeResize(b0_, origSize + additionalSize);  // rhs

          H_.topRightCorner(denseSize, origSize - denseSize) = H01;
          H_.bottomLeftCorner(origSize - denseSize, denseSize) = H10;
          H_.bottomRightCorner(origSize - denseSize, origSize - denseSize) = H11;
          H_.block(0, denseSize, H_.rows(), additionalSize).setZero();
          H_.block(denseSize, 0, additionalSize, H_.rows()).setZero();

          b0_.tail(origSize - denseSize) = b1;
          b0_.segment(denseSize, additionalSize).setZero();
        } else //如果添加的是地图点
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
      if (!isLandmark) //这个参数块不是地图点
	  {
	    //ParameterBlockInfo构造函数 
	    //给结构中的linearizationPoint赋初值了
	    //输入的第一个参数是参数块的id
	    //第二个参数是参数块的指针
	    //第三个参数是这个参数块在H矩阵中的位置
	    //第四个参数表示这个参数快是否为地图点
        info = ParameterBlockInfo(parameterBlockSpec.first, parameterBlockSpec.second, denseSize,isLandmark);
        parameterBlockInfos_.insert(parameterBlockInfos_.begin() + denseIndices_, info);

        parameterBlockId2parameterBlockInfoIdx_.insert(std::pair<uint64_t, size_t>(parameterBlockSpec.first,denseIndices_));

        //  update base_t book-keeping
        //这个是ceres的函数
        //定义  typedef ::ceres::CostFunction base_t;
        base_t::mutable_parameter_block_sizes()->insert( base_t::mutable_parameter_block_sizes()->begin() + denseIndices_,info.dimension);

        denseIndices_++;  // remember we increased the dense part of the problem

        // also increase the rest
        for (size_t j = denseIndices_; j < parameterBlockInfos_.size(); ++j) 
	 {
		 //parameterBlockInfos_序号是容器中的位置，内容是各个参数块的信息
          parameterBlockInfos_.at(j).orderingIdx += additionalSize;
          parameterBlockId2parameterBlockInfoIdx_[parameterBlockInfos_.at(j).parameterBlockPtr->id()] += 1;
        }
      } else //这个参数块是地图点
      {
        // just add at the end
        info = ParameterBlockInfo(parameterBlockSpec.first,
        						  parameterBlockSpec.second,
        						  parameterBlockInfos_.back().orderingIdx+ parameterBlockInfos_.back().minimalDimension,
            					  isLandmark);
        parameterBlockInfos_.push_back(info);
        parameterBlockId2parameterBlockInfoIdx_.insert(std::pair<uint64_t, size_t>(parameterBlockSpec.first,parameterBlockInfos_.size() - 1));

        //  update base_t book-keeping
        //这个是ceres的函数!!!!!!!!!!!!!!!!修改ceres中参数块的大小
        // typedef ::ceres::CostFunction base_t;
        //number and sizes of input parameter blocks
        base_t::mutable_parameter_block_sizes()->push_back(info.dimension);
      }
      assert(
          parameterBlockInfos_[parameterBlockId2parameterBlockInfoIdx_[parameterBlockSpec.first]].parameterBlockId == parameterBlockSpec.first);
    } else //表示parameterBlockId2parameterBlockInfoIdx_中有这个参数块
    {

#ifdef USE_NEW_LINEARIZATION_POINT
      // switch linearization point - easy to do on the linearized part...
      size_t i = it->second;
      Eigen::VectorXd Delta_Chi_i(parameterBlockInfos_[i].minimalDimension);
      parameterBlockInfos_[i].parameterBlockPtr->minus(parameterBlockInfos_[i].linearizationPoint.get(),parameterBlockInfos_[i].parameterBlockPtr->parameters(),Delta_Chi_i.data());
      b0_ -= H_.block(0,parameterBlockInfos_[i].orderingIdx,H_.rows(),parameterBlockInfos_[i].minimalDimension)* Delta_Chi_i;
      parameterBlockInfos_[i].resetLinearizationPoint( parameterBlockInfos_[i].parameterBlockPtr);
#endif//默认进入这个条件
      info = parameterBlockInfos_.at(it->second);//其实约等于什么都没干
    }
  }//遍历参数块结束

  // update base_t book-keeping on residuals
  // 定义 typedef ::ceres::CostFunction base_t;
  //c．设置ceres代价函数的残差的维度
  base_t::set_num_residuals(H_.cols());//这个是ceres函数，设置残差的维度 经过测试发现H_矩阵的行和列的维度是相同的
  double** parametersRaw = new double*[parameters.size()];
  Eigen::VectorXd residualsEigen(errorInterfacePtr->residualDim());//定义一个残差维数的向量
  double* residualsRaw = residualsEigen.data();

  double** jacobiansRaw = new double*[parameters.size()];
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::aligned_allocator< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > jacobiansEigen(parameters.size());

  double** jacobiansMinimalRaw = new double*[parameters.size()];
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > jacobiansMinimalEigen(parameters.size());

  for (size_t i = 0; i < parameters.size(); ++i)//遍历参数块，将每个参数块的雅克比和结果建立其对应关系
  {
    OKVIS_ASSERT_TRUE_DBG(Exception, isParameterBlockConnected(parameters[i].first),
        "okvis bug: no linearization point, since not connected.");
	//数据块的线性(原始)数据
	//parameters元素 = (参数块id,参数块指针构成)
    parametersRaw[i] =parameterBlockInfos_[parameterBlockId2parameterBlockInfoIdx_[parameters[i].first]].linearizationPoint.get();  // first estimate Jacobian!!

    //第i个数据块的雅各比矩阵,矩阵的行为残差的维度,矩阵的列为数据块的维度
    jacobiansEigen[i].resize(errorInterfacePtr->residualDim(), parameters[i].second->dimension());
    jacobiansRaw[i] = jacobiansEigen[i].data();
	//第i个数据块的最小雅各比矩阵,矩阵的行为残差的维度,矩阵的列为数据块的最小维度
    jacobiansMinimalEigen[i].resize(errorInterfacePtr->residualDim(),arameters[i].second->minimalDimension());
    jacobiansMinimalRaw[i] = jacobiansMinimalEigen[i].data();
  }

  // evaluate residual block
  //d.计算得到这个残差对于每个参数块的雅克比矩阵和残差
  errorInterfacePtr->EvaluateWithMinimalJacobians(parametersRaw, residualsRaw,
                                                  jacobiansRaw,//其实这个结果没有用到
                                                  jacobiansMinimalRaw);


  // correct for loss function if applicable
  //e.如果这个残差块使用了loss function那么更新雅克比和残差
  //residualBlockId2ResidualBlockSpecMap存储的是纯残差块的信息,元素 = (残差在ceres中的id，残差在ceres中的id+loss function函数指针+误差函数指针)
  //std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError = std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
  ::ceres::LossFunction* lossFunction = mapPtr_->residualBlockId2ResidualBlockSpecMap().find(residualBlockId)->second.lossFunctionPtr;
  if (lossFunction) 
  {
    //经过实际测试发现会进入这个条件!!!!!!!只有BA误差才会进入这条件，因为在
    OKVIS_ASSERT_TRUE_DBG(
        Exception,
        mapPtr_->residualBlockId2ResidualBlockSpecMap().find(residualBlockId)
        != mapPtr_->residualBlockId2ResidualBlockSpecMap().end(),
        "???");

    // following ceres in internal/ceres/corrector.cc
    const double sq_norm = residualsEigen.transpose() * residualsEigen;
    double rho[3];
	//这个是ceres的函数!!!!!!!!!!!!!在ceres中搜索lossFunction可以找到这个含函数
	//rho中存储的是loss functin函数的值，一阶导数的值，二阶导数的值
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
    //更新雅克比和残差
    for (size_t i = 0; i < parameters.size(); ++i) 
	{
      jacobiansMinimalEigen[i] = sqrt_rho1* (jacobiansMinimalEigen[i]- alpha_sq_norm * residualsEigen * (residualsEigen.transpose() * jacobiansMinimalEigen[i]));
    }

    // correct residuals (caution: must be after "correct Jacobians"):
    residualsEigen *= residual_scaling;
  }//lossfunction判断结束

  // add blocks to lhs and rhs
  // f.最后就是计算新添加的 H_new 和 b_new 了
  for (size_t i = 0; i < parameters.size(); ++i)//遍历参数块
  {
    Map::ParameterBlockSpec parameterBlockSpec = parameters[i];//某个参数块
    //得到这个参数块的信息
    ParameterBlockInfo parameterBlockInfo_i = parameterBlockInfos_.at(parameterBlockId2parameterBlockInfoIdx_[parameters[i].first]);

    OKVIS_ASSERT_TRUE_DBG(
        Exception,
        parameterBlockInfo_i.parameterBlockId == parameters[i].second->id(),
        "okvis bug: inconstistent okvis ordering");

    if (parameterBlockInfo_i.minimalDimension == 0)
      continue;

    OKVIS_ASSERT_TRUE_DBG(Exception,H_.allFinite(),"WTF1");
	//更新H矩阵中的对角元素
    //H中第i个数据块和自身的海瑟矩阵(i数据块~i数据块)
    H_.block(parameterBlockInfo_i.orderingIdx, parameterBlockInfo_i.orderingIdx,
    		 parameterBlockInfo_i.minimalDimension,parameterBlockInfo_i.minimalDimension) += jacobiansMinimalEigen.at(i).transpose().eval() * jacobiansMinimalEigen.at(i);
    //b中第i个数据块的值
	b0_.segment(parameterBlockInfo_i.orderingIdx,parameterBlockInfo_i.minimalDimension) -= jacobiansMinimalEigen.at(i).transpose().eval() * residualsEigen;

    OKVIS_ASSERT_TRUE_DBG(Exception,H_.allFinite(),
                      "WTF2 " <<jacobiansMinimalEigen.at(i).transpose().eval() * jacobiansMinimalEigen.at(i));
	
	//第i个数据块之前的数据块j
	//更新H矩阵中的非对角元素
	 // 数据块(i,j)对应的海瑟矩阵块
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
  if (!keep) //从地图中删除这个残差块，一定会进入这个条件
  {
    //g.从ceres中删除这个残差块
    mapPtr_->removeResidualBlock(residualBlockId);
  }

  // cleanup temporarily allocated stuff
  delete[] parametersRaw;
  delete[] jacobiansRaw;
  delete[] jacobiansMinimalRaw;

  check();/// 检查数据结构

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
/// 检查数据结构
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
/// 边缘化数据块
/// 1&2通过shur消元更新H_和b_矩阵
//3.重新设置ceres优化的残差维度。ceres中更新costfunction中的参数块的大小。删除ceres中的参数块。
//parameterBlockIds 元素 = 要边缘化的参数块的id
bool MarginalizationError::marginalizeOut(const std::vector<uint64_t>& parameterBlockIds,const std::vector<bool> & keepParameterBlocks) 
{
  if (parameterBlockIds.size() == 0) {
    return false;
  }

  // copy so we can manipulate
  //parameterBlockIdsCopy中存储的参数块 没有重复的id，后面会处理
  std::vector<uint64_t> parameterBlockIdsCopy = parameterBlockIds;
  if (parameterBlockIds.size() != keepParameterBlocks.size()) //基本上不会发生
  {
    OKVIS_ASSERT_TRUE_DBG(
        Exception,
        keepParameterBlocks.size() == 0,
        "input vectors must either be of same size or omit optional parameter keepParameterBlocks: "<<
        parameterBlockIds.size()<<" vs "<<keepParameterBlocks.size());
  }
  std::map<uint64_t, bool> keepParameterBlocksCopy;//添加（数据块序号、bool）到keepParameterBlocksCopy中
  for (size_t i = 0; i < parameterBlockIdsCopy.size(); ++i) 
  {
    bool keep = false;
    if (i < keepParameterBlocks.size()) 
	{
      keep = keepParameterBlocks.at(i);//基本上都是false
    }
    keepParameterBlocksCopy.insert(std::pair<uint64_t, bool>(parameterBlockIdsCopy.at(i), keep));
  }

  /* figure out which blocks need to be marginalized out */
  std::vector<std::pair<int, int> > marginalizationStartIdxAndLengthPairslandmarks;//元素 = (地图点参数块在H矩阵中的位置，最小维度)
  std::vector<std::pair<int, int> > marginalizationStartIdxAndLengthPairsDense;//元素 = (参数块中非地图点在H矩阵中的位置，最小维度)
  size_t marginalizationParametersLandmarks = 0;//记录要被边缘化的参数块中地图点的总维度
  size_t marginalizationParametersDense = 0;//记录要被边缘化的参数块中非地图点的总维度

  // make sure no duplications...
  //删除parameterBlockIdsCopy中重复的参数块
  std::sort(parameterBlockIdsCopy.begin(), parameterBlockIdsCopy.end());//排序（升序），数据块序号的从小到大排序
  for (size_t i = 1; i < parameterBlockIdsCopy.size(); ++i) 
  {
    if (parameterBlockIdsCopy[i] == parameterBlockIdsCopy[i - 1]) 
    {
      parameterBlockIdsCopy.erase(parameterBlockIdsCopy.begin() + i);//删除相同的数据块
      --i;
    }
  }
  
  for (size_t i = 0; i < parameterBlockIdsCopy.size(); ++i) 
  {
    //根据参数块id 
    std::map<uint64_t, size_t>::iterator it = parameterBlockId2parameterBlockInfoIdx_.find(parameterBlockIdsCopy[i]);

    // sanity check - are we trying to marginalize stuff that is not connected to this error term?
    OKVIS_ASSERT_TRUE(
        Exception,
        it != parameterBlockId2parameterBlockInfoIdx_.end(),
        "trying to marginalize out unconnected parameter block id = "<<parameterBlockIdsCopy[i])
    if (it == parameterBlockId2parameterBlockInfoIdx_.end())
      return false;

    // distinguish dense and landmark (sparse) part for more efficient pseudo-inversion later on
    size_t startIdx = parameterBlockInfos_.at(it->second).orderingIdx;//找到这个参数块在H矩阵中的位置
    size_t minDim = parameterBlockInfos_.at(it->second).minimalDimension;//这个参数块的最小维度
    if (parameterBlockInfos_.at(it->second).isLandmark) 
   {
      marginalizationStartIdxAndLengthPairslandmarks.push_back(std::pair<int, int>(startIdx, minDim));
      marginalizationParametersLandmarks += minDim;//更新被边缘化的参数块中地图点的总维度
    } else {
      marginalizationStartIdxAndLengthPairsDense.push_back(std::pair<int, int>(startIdx, minDim));
      marginalizationParametersDense += minDim;//更新被边缘化的参数块中非地图点的总维度
    }
  }

  // make sure the marginalization pairs are ordered
   // 将集合中数据（地标点数据块在H阵的起始位置）升序排序
  std::sort(marginalizationStartIdxAndLengthPairslandmarks.begin(),
            marginalizationStartIdxAndLengthPairslandmarks.end(),
            [](std::pair<int,int> left, std::pair<int,int> right) {
              return left.first < right.first;
            });
  //将集合中数据（非地标点数据块在H阵的起始位置）升序排序
  std::sort(marginalizationStartIdxAndLengthPairsDense.begin(),
            marginalizationStartIdxAndLengthPairsDense.end(),
            [](std::pair<int,int> left, std::pair<int,int> right) {
              return left.first < right.first;
            });

  // unify contiguous marginalization requests
  //将连续的参数块组合在一起
  for (size_t m = 1; m < marginalizationStartIdxAndLengthPairslandmarks.size(); ++m) 
  {
     //如果上一个数据块（待边缘化的）在H阵的起点+上一个数据块（待边缘化）的最小维数=当前数据块（带边缘化）的起点（连续）
    //注意待边缘化的地标点不一定是连续的
    if (marginalizationStartIdxAndLengthPairslandmarks.at(m - 1).first + marginalizationStartIdxAndLengthPairslandmarks.at(m - 1).second == marginalizationStartIdxAndLengthPairslandmarks.at(m).first) 
   {
      marginalizationStartIdxAndLengthPairslandmarks.at(m - 1).second +=  marginalizationStartIdxAndLengthPairslandmarks.at(m).second;
      marginalizationStartIdxAndLengthPairslandmarks.erase( marginalizationStartIdxAndLengthPairslandmarks.begin() + m);
      --m;
    }
  }
  for (size_t m = 1; m < marginalizationStartIdxAndLengthPairsDense.size(); ++m) 
  {
    //如果上一个数据块（待边缘化的）在H阵的起点+上一个数据块（待边缘化）的最小维数=当前数据块（带边缘化）的起点（连续）
    //注意待边缘化的非地标点数据块不一定是连续的
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
  //1.开始处理要被边缘化的地图点
  if (marginalizationStartIdxAndLengthPairslandmarks.size() > 0) 
  {

    // preconditioner
     //提取H_矩阵的对角线上元素,并且元素如果大于1.0e-9,选择 p=H_矩阵对角线上元素的平方根,否则这个值赋值为10^-3
    Eigen::VectorXd p = (H_.diagonal().array() > 1.0e-9).select(H_.diagonal().cwiseSqrt(),1.0e-3);
    Eigen::VectorXd p_inv = p.cwiseInverse();//逐个元素求倒数

    // scale H and b
    //调整H_的尺度 目的是不让其值太小
    //asDiagonal将向量变成对角矩阵
    H_ = p_inv.asDiagonal() * H_ * p_inv.asDiagonal();
    b0_ = p_inv.asDiagonal() * b0_;

    Eigen::MatrixXd U(H_.rows() - marginalizationParametersLandmarks, H_.rows() - marginalizationParametersLandmarks);
    Eigen::MatrixXd V(marginalizationParametersLandmarks, marginalizationParametersLandmarks);
    Eigen::MatrixXd W(H_.rows() - marginalizationParametersLandmarks, marginalizationParametersLandmarks);
    Eigen::VectorXd b_a(H_.rows() - marginalizationParametersLandmarks);
    Eigen::VectorXd b_b(marginalizationParametersLandmarks);

    // split preconditioner
    Eigen::VectorXd p_a(H_.rows() - marginalizationParametersLandmarks);//减少地图点后的维度
    Eigen::VectorXd p_b(marginalizationParametersLandmarks);//待边缘化地标点的维度
    //分裂p为p_a,p_b。p_a为不涉及边缘化的p中元素, p_b为需要边缘化的p中元素
    splitVector(marginalizationStartIdxAndLengthPairslandmarks, p, p_a, p_b);  // output

    // split lhs
    //分裂H_矩阵,其中U是不涉及边缘化的H中的元素,W是待边缘化数据块和保留数据块的交叉海瑟矩阵,V为待边缘化数据块的海瑟矩阵
    splitSymmetricMatrix(marginalizationStartIdxAndLengthPairslandmarks, H_, U,W, V);  // output

    // split rhs
    //分裂b向量,b_a为不涉及边缘化的b中元素,b_b为待边缘化的b中元素
    splitVector(marginalizationStartIdxAndLengthPairslandmarks, b0_, b_a, b_b);  // output


    // invert the marginalization block
    static const int sdim =::okvis::ceres::HomogeneousPointParameterBlock::MinimalDimension;//为3
    
    b0_.resize(b_a.rows());
    b0_ = b_a;
    H_.resize(U.rows(), U.cols());
    H_ = U;
    const size_t numBlocks = V.cols() / sdim;//要被边缘化的地图点的个数
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> delta_H( numBlocks);
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> delta_b( numBlocks);
    Eigen::MatrixXd M1(W.rows(), W.cols());//M1 就是 H12*H22.inverse。
    size_t idx = 0;
    for (size_t i = 0; int(i) < V.cols(); i += sdim) 
   {
      Eigen::Matrix<double, sdim, sdim> V_inv_sqrt;
      Eigen::Matrix<double, sdim, sdim> V1 = V.block(i, i, sdim, sdim);
      MarginalizationError::pseudoInverseSymmSqrt(V1, V_inv_sqrt);//V_inv_sqrt为V1的广义逆开根号
      Eigen::MatrixXd M = W.block(0, i, W.rows(), sdim) * V_inv_sqrt;//M 就是 H12H22^0.5
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
    // Schur消元,相当于通过高斯消元,将待边缘化的地标点变量从线性方程组中剔除
    b0_ -= delta_b.at(idx - 1);
    H_ -= delta_H.at(idx - 1);

    // unscale
    // 除去归一化的影响
    H_ = p_a.asDiagonal() * H_ * p_a.asDiagonal();
    b0_ = p_a.asDiagonal() * b0_;
  }//待边缘化的地图点处理完毕

  /* dense part (if existing) */
  //2.开始处理边缘化的非地图点的状态-注意没有使用循环
  if (marginalizationStartIdxAndLengthPairsDense.size() > 0) 
  {

    // preconditioner
      //提取H_矩阵的对角线上元素,并且元素如果大于1.0e-9,选择 p=H_矩阵对角线上元素的平方根
    Eigen::VectorXd p = (H_.diagonal().array() > 1.0e-9).select(H_.diagonal().cwiseSqrt(),1.0e-3);
    Eigen::VectorXd p_inv = p.cwiseInverse();

    // scale H and b
    ///asDiagonal将向量变成对角矩阵
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
    splitVector(marginalizationStartIdxAndLengthPairsDense, b0_, b_a, b_b);  // output 奇怪怎么没有用到B_b?

    // invert the marginalization block
    Eigen::MatrixXd V_inverse_sqrt(V.rows(), V.cols());
    Eigen::MatrixXd V1 = 0.5 * (V + V.transpose());//好奇怪这里使用的是V和V转置的二分之一，直接用V不行么?
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
  }//处理完边缘化的非地图点的状态

 //以上的两个函数主要是更新了schur补的H和b矩阵
  //3.设置ceres中残差的维度=原来残差维度-边缘化地图点维度-边缘化非地图点维度。
  //ceres中更新costfunction中的参数块的大小=删除参数块。删除ceres中的参数块。从ceres中删除这个残差块，
  // also adapt the ceres-internal size information
  //定义 typedef ::ceres::CostFunction base_t;
  //设置ceres残差的维度 base_t::num_residuals()=现在残差的维度
  //这是一个ceres的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  base_t::set_num_residuals(base_t::num_residuals() - marginalizationParametersDense - marginalizationParametersLandmarks);

  /* delete all the book-keeping */
  for (size_t i = 0; i < parameterBlockIdsCopy.size(); ++i)//遍历要边缘化的参数块 
  {
    size_t idx = parameterBlockId2parameterBlockInfoIdx_.find( parameterBlockIdsCopy[i])->second;//得到该数据块对应的信息位置(在parameterBlockInfos_中)
    int margSize = parameterBlockInfos_.at(idx).minimalDimension;//提取数据块的维数
    parameterBlockInfos_.erase(parameterBlockInfos_.begin() + idx);//将数据块对应的信息从信息容器中删除

    for (size_t j = idx; j < parameterBlockInfos_.size(); ++j) 
	{
      parameterBlockInfos_.at(j).orderingIdx -= margSize;
      parameterBlockId2parameterBlockInfoIdx_.at( parameterBlockInfos_.at(j).parameterBlockId) -= 1;
    }

    parameterBlockId2parameterBlockInfoIdx_.erase(parameterBlockIdsCopy[i]);//删除该数据块在parameterBlockId2parameterBlockInfoIdx_中对应的值

    // also adapt the ceres-internal book-keepin
     //mutable_parameter_block_sizes函数返回数据类型为vector<int32>,存储的是number and sizes of input parameter blocks
     //ceres中更新costfunction中的参数块的大小
     //这是一个ceres的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    base_t::mutable_parameter_block_sizes()->erase( mutable_parameter_block_sizes()->begin() + idx);//这是一个ceres的函数
  }

  /* assume everything got dense */
  // this is a conservative assumption, but true in particular when marginalizing
  // poses w/o landmarks
   ///假设所有的未被删除的数据块都不是地标点
  /// parameterBlockInfos_的维数应该比parameterBlockIdsCopy略大
  /// 比如论文中的地标点1可以被1,2,3关键帧看到,所以其对应三个二次投影误差,分别添加三个投影误差时,会给parameterBlockInfos_添加七个数据块,如:-------
  /// ----------------1,2,3关键帧的位姿,外参,以及地标点1的坐标
  /// 而parameterBlockIdsCopy只有地标点1和关键帧1的位姿和外参
  denseIndices_ = parameterBlockInfos_.size();
  for (size_t i = 0; i < parameterBlockInfos_.size(); ++i) 
  {
    if (parameterBlockInfos_.at(i).isLandmark) 
	{
      parameterBlockInfos_.at(i).isLandmark = false;
    }
  }

   /*这里我们把和代码安全性相关的注释掉了，方便看代码
  // check if the removal is safe
  for (size_t i = 0; i < parameterBlockIdsCopy.size(); ++i) 
  {
    Map::ResidualBlockCollection residuals = mapPtr_->residuals(parameterBlockIdsCopy[i]);//根据参数块找到所有的残差块
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
    } else//一般都会进入这个条件
    {
      //从ceres中删除与这个参数块相关的残差块；从ceres中删除这个参数块。
      //调用了ceres的函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      mapPtr_->removeParameterBlock(parameterBlockIdsCopy[i]);
    }
  }

  check();

  return true;
}

// This must be called before optimization after adding residual blocks and/or marginalizing,
// since it performs all the lhs and rhs computations on from a given _H and _b.
/// 更新H和b，在优化之前，添加残差块和边缘化之后
//我们已经得到了H矩阵 我们对H矩阵进行svd分解然后计算得到J矩阵
//并进一步得到-J.transpose().inverse()*b=e0
void MarginalizationError::updateErrorComputation() 
{
  if (errorComputationValid_)
    return;  // already done.

  // now we also know the error dimension:
  //0.设置ceres代价函数的残差维度
  base_t::set_num_residuals(H_.cols());//ceres的函数

  // preconditioner
  Eigen::VectorXd p = (H_.diagonal().array() > 1.0e-9).select(H_.diagonal().cwiseSqrt(),1.0e-3);
  Eigen::VectorXd p_inv = p.cwiseInverse();

  // lhs SVD: _H = J^T*J = _U*S*_U^T
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes( 0.5  * p_inv.asDiagonal() * (H_ + H_.transpose())  * p_inv.asDiagonal() );

  static const double epsilon = std::numeric_limits<double>::epsilon();//运行编译程序的计算机所能识别的最小非零浮点数。
  double tolerance = epsilon * H_.cols()* saes.eigenvalues().array().maxCoeff();//最小非零浮点数*H_的维度*最大特征值
  S_ = Eigen::VectorXd(  (saes.eigenvalues().array() > tolerance).select( saes.eigenvalues().array(), 0));//特征值对角阵
  S_pinv_ = Eigen::VectorXd( (saes.eigenvalues().array() > tolerance).select( saes.eigenvalues().array().inverse(), 0));//特征值倒数的对角阵

  S_sqrt_ = S_.cwiseSqrt();//S_每个元素的平方根
  S_pinv_sqrt_ = S_pinv_.cwiseSqrt();//S_pinv_每个元素的平方根

  // assign Jacobian
   /// 通过SVD计算sqrt(H),即J.
  J_ = (p.asDiagonal() * saes.eigenvectors() * (S_sqrt_.asDiagonal())).transpose();

  // constant error (residual) _e0 := (-pinv(J^T) * _b):
  Eigen::MatrixXd J_pinv_T = (S_pinv_sqrt_.asDiagonal())* saes.eigenvectors().transpose()  *p_inv.asDiagonal() ;
  e0_ = (-J_pinv_T * b0_);//得到-J.transpose().inverse()*b

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
//使用当前的状态与边缘化之前的状态相减得到Delta_Chi
bool MarginalizationError::computeDeltaChi(double const* const * parameters,
                                           Eigen::VectorXd& DeltaChi) const 
{
  DeltaChi.resize(H_.rows());
  //parameterBlockInfos_序号是参数块在H矩阵中的位置 内容是各个参数块的信息
  for (size_t i = 0; i < parameterBlockInfos_.size(); ++i) 
  {
	    // stack Delta_Chi vector
	    if (!parameterBlockInfos_[i].parameterBlockPtr->fixed()) 
	    {
	      Eigen::VectorXd Delta_Chi_i(parameterBlockInfos_[i].minimalDimension);
		  //第一个和第二个参数是输入的参数，第三个参数是前面两个参数相减得到的Delta_Chi_i
		  //这里的minus是各个状态自己定义的
	      parameterBlockInfos_[i].parameterBlockPtr->minus(parameterBlockInfos_[i].linearizationPoint.get(), parameters[i],Delta_Chi_i.data());
	      DeltaChi.segment(parameterBlockInfos_[i].orderingIdx, parameterBlockInfos_[i].minimalDimension) = Delta_Chi_i;
	    }
  }
  return true;
}

//This evaluates the error term and additionally computes the Jacobians.
//雅克比解析解 jacobian analytic
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
  computeDeltaChi(parameters, Delta_Chi);//使用当前的状态与边缘化之前的状态相减得到Delta_Chi

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
  e = e0_ + J_ * Delta_Chi;//计算得到残差 e0和J_在这边缘化之后就用于不会变

  return true;
}

}  // namespace ceres
}  // namespace okvis

