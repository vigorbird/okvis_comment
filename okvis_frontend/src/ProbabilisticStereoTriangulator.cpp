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
 *  Created on: Oct 18, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file ProbabilisticStereoTriangulator.cpp
 * @brief Source file for the ProbabilisticStereoTriangulator class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <okvis/triangulation/stereo_triangulation.hpp>
#include <okvis/triangulation/ProbabilisticStereoTriangulator.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>

// cameras and distortions
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
namespace triangulation {

// Default constructor; make sure to call resetFrames before triangulation!
template<class CAMERA_GEOMETRY_T>
ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::ProbabilisticStereoTriangulator(
    double pixelSigma)
    : camIdA_(-1),
      camIdB_(-1) {
  // relative transformation - have a local copy
  T_AB_.setIdentity();
  // relative uncertainty - have a local copy
  UOplus_.setZero();

  sigmaRay_ = pixelSigma / 300.0;
}

// Constructor to set frames and relative transformation.
template<class CAMERA_GEOMETRY_T>
ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::ProbabilisticStereoTriangulator(
    std::shared_ptr<okvis::MultiFrame> frameA_ptr,
    std::shared_ptr<okvis::MultiFrame> frameB_ptr, size_t camIdA, size_t camIdB,
    const okvis::kinematics::Transformation& T_AB,
    const Eigen::Matrix<double, 6, 6>& UOplus, double pixelSigma)
    : frameA_(frameA_ptr),
      frameB_(frameB_ptr),
      camIdA_(camIdA),
      camIdB_(camIdB),
      T_AB_(T_AB),
      UOplus_(UOplus) {
  T_BA_ = T_AB_.inverse();
  // also do all backprojections
//	_frameA_ptr->computeAllBackProjections(false);
//	_frameB_ptr->computeAllBackProjections(false);
  // prepare the pose prior, since this will not change.
  ::okvis::ceres::PoseError poseError(T_AB_, UOplus_.inverse());
  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> J_minimal;  // Jacobian
  Eigen::Matrix<double, 7, 7, Eigen::RowMajor> J;  // Jacobian
  poseA_ = ::okvis::ceres::PoseParameterBlock(
      okvis::kinematics::Transformation(), 0, okvis::Time(0));
  poseB_ = ::okvis::ceres::PoseParameterBlock(T_AB_, 0, okvis::Time(0));
  extrinsics_ = ::okvis::ceres::PoseParameterBlock(
      okvis::kinematics::Transformation(), 0, okvis::Time(0));
  double residuals[6];
  // evaluate to get the jacobian
  double* parameters = poseB_.parameters();
  double* jacobians = J.data();
  double* jacobians_minimal = J_minimal.data();
  poseError.EvaluateWithMinimalJacobians(&parameters, &residuals[0], &jacobians,
                                         &jacobians_minimal);
  // prepare lhs of Gauss-Newton:
  H_.setZero();
  H_.topLeftCorner<6, 6>() = J_minimal.transpose() * J_minimal;

  sigmaRay_ = pixelSigma
      / std::min(
          frameA_->geometryAs<CAMERA_GEOMETRY_T>(camIdA_)->focalLengthU(),
          frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->focalLengthU());
}

// Reset frames and relative transformation.
/**
   *  \brief Reset frames and relative transformation.
   *  \param frameA_ptr: First multiframe.
   *  \param frameB_ptr: Second multiframe.
   *  \param camIdA: Camera ID for first frame.这个参数表示是左相机还是右相机
   *  \param camIdB: Camera ID for second frame.
   *  \param T_AB: Relative transformation from frameA to frameB.
   *  \param UOplus Oplus-type uncertainty of T_AB.
   */.
template<class CAMERA_GEOMETRY_T>
void ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::resetFrames(std::shared_ptr<okvis::MultiFrame> frameA_ptr,std::shared_ptr<okvis::MultiFrame> frameB_ptr, 
    																	size_t camIdA, size_t camIdB,
    																	const okvis::kinematics::Transformation& T_AB,
    																	const Eigen::Matrix<double, 6, 6>& UOplus) 
 {
  T_AB_ = T_AB;
  T_BA_ = T_AB_.inverse();

  frameA_ = frameA_ptr;
  frameB_ = frameB_ptr;
  camIdA_ = camIdA;
  camIdB_ = camIdB;

  UOplus_ = UOplus;
  // also do all backprojections
//	_frameA_ptr->computeAllBackProjections(false);
//	_frameB_ptr->computeAllBackProjections(false);
  // prepare the pose prior, since this will not change.
  ::okvis::ceres::PoseError poseError(T_AB_, UOplus_.inverse());//这个是设置ceres误差函数，在这里的主要作用是为了计算雅克比
  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> J_minimal;  // Jacobian
  Eigen::Matrix<double, 7, 7, Eigen::RowMajor> J;  // Jacobian
  //poseA_的位姿为单位位姿，没有用到
  poseA_ = ::okvis::ceres::PoseParameterBlock( okvis::kinematics::Transformation(), 0, okvis::Time(0));//PoseParameterBlock::PoseParameterBlock(
  poseB_ = ::okvis::ceres::PoseParameterBlock(T_AB_, 0, okvis::Time(0));
  extrinsics_ = ::okvis::ceres::PoseParameterBlock(okvis::kinematics::Transformation(), 0, okvis::Time(0));
  double residuals[6];
  // evaluate to get the jacobian
  double* parameters = poseB_.parameters();//参数的顺序是前三维为位置t 最后一维是四元数的w
  double* jacobians = J.data();//感觉这里计算jacobians好像没有用，只用到了jacobians_minimal
  double* jacobians_minimal = J_minimal.data();
  //重要的函数!!!!!!!!!!!!!!!!!!!!!!!搜索 bool PoseError::EvaluateWithMinimalJacobians(
  poseError.EvaluateWithMinimalJacobians(&parameters, &residuals[0], &jacobians,&jacobians_minimal);
  
  //Information matrix of pose and landmark.
  //更新了ProbabilisticStereoTriangulator类中的变量 Eigen::MatrixXd H_;
  H_.setZero();
  H_.topLeftCorner<6, 6>() = J_minimal.transpose() * J_minimal;

  //更新 ProbabilisticStereoTriangulator类中的 sigmaRay_变量
  sigmaRay_ = 0.5/ std::min( frameA_->geometryAs<CAMERA_GEOMETRY_T>(camIdA_)->focalLengthU(), frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->focalLengthU());

}

// Default destructor.
template<class CAMERA_GEOMETRY_T>
ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::~ProbabilisticStereoTriangulator() {
}


/**
  *  \brief Triangulation.
  *  \param[in]  keypointIdxA: Index of keypoint to be triangulated in frame A.
  *  \param[in]  keypointIdxB: Index of keypoint to be triangulated in frame B.
  *  \param[out] outHomogeneousPoint_A: Output triangulation in A-coordinates.
  *  \param[out] outCanBeInitializedInaccuarate This will be set to false,
  * 			 if it can certainly not be initialised (conservative, but inaccurate guess)
  *  \param[in]  sigmaRay Ray uncertainty.
  * \return 3-sigma consistency check result, in A-coordinates.
  */
// Triangulation.
//作者使用的是这个三角化函数
//outHomogeneousPoint_A是三角化后的A相机坐标系下的三维点 是单位化的齐次坐标
//sigmaRay这个参数好像没有用到呀!!!!!
//如果两个射线接近于平行，那么 outCanBeInitializedInaccuarate=false
template<class CAMERA_GEOMETRY_T>
bool ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::stereoTriangulate(
    size_t keypointIdxA, size_t keypointIdxB,
    Eigen::Vector4d& outHomogeneousPoint_A,
    bool & outCanBeInitializedInaccuarate,
    double sigmaRay) const {

  OKVIS_ASSERT_TRUE_DBG(Exception,frameA_&&frameB_,"initialize with frames before use!");

  // chose the source of uncertainty
  double sigmaR = sigmaRay;
  if (sigmaR == -1.0)
    sigmaR = sigmaRay_;

  // call triangulation
  bool isValid;
  bool isParallel;
  Eigen::Vector2d keypointCoordinatesA, keypointCoordinatesB;
  Eigen::Vector3d backProjectionDirectionA_inA, backProjectionDirectionB_inA;

  frameA_->getKeypoint(camIdA_, keypointIdxA, keypointCoordinatesA);//得到A图像的特征点坐标 keypointCoordinatesA
  frameB_->getKeypoint(camIdB_, keypointIdxB, keypointCoordinatesB);//得到B图像的特征点坐标 keypointCoordinatesB

  //搜索 bool PinholeCamera<DISTORTION_T>::backProject
  frameA_->geometryAs<CAMERA_GEOMETRY_T>(camIdA_)->backProject( keypointCoordinatesA, &backProjectionDirectionA_inA);//得到A图像不带畸变的归一化的平面坐标    backProjectionDirectionA_inA
  frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->backProject( keypointCoordinatesB, &backProjectionDirectionB_inA);// 得到B图像不带畸变的归一化的平面坐标 backProjectionDirectionB_inA
  backProjectionDirectionB_inA = T_AB_.C() * backProjectionDirectionB_inA;//将B相机坐标系下的射线转换到A相机坐标下

  ///三角化，形参
  /// 第一个参数 表示A帧中A系的原点
  /// 第二个参数 表示A相机特征点所对应的射线在A坐标系下的坐标
  /// 第三个参数 表示B帧原点在A系中的坐标
  /// 第四个参数 表示B相机特征点所对应的射线在A坐标系下的坐标
  /// sigma为特征点的不确定度，isValid和isParallel是输出的变量
  //得到的hpA为三角化后的坐标点在A相机坐标系下的坐标-一定要注意了返回的hpA是单位齐次坐标，我们将三维点投影到相机像素坐标的过程中其实我们只需要的是齐次坐标就够了。
  Eigen::Vector4d hpA = triangulateFast(
      Eigen::Vector3d(0, 0, 0),  backProjectionDirectionA_inA.normalized(),  
      T_AB_.r(),  backProjectionDirectionB_inA.normalized(), 
      sigmaR, isValid, isParallel);
  outCanBeInitializedInaccuarate = !isParallel;

  if (!isValid) {
    return false;
  }

  // check reprojection:
  double errA, errB;
  isValid = computeReprojectionError4(frameA_, camIdA_, keypointIdxA, hpA, errA);//将三角化的点hpA投影到A相机图像中得到误差errA
  if (!isValid) {
    return false;
  }
  Eigen::Vector4d outHomogeneousPoint_B = T_BA_ * Eigen::Vector4d(hpA);
  if (!computeReprojectionError4(frameB_, camIdB_, keypointIdxB,outHomogeneousPoint_B, errB)) 
  {
    isValid = false;
    return false;
  }
  if (errA > 4.0 || errB > 4.0) 
  {
    isValid = false;
  }

  // assign output
  outHomogeneousPoint_A = Eigen::Vector4d(hpA);

  return isValid;
}

// Triangulation.
/**
 *	\brief Triangulation.
 *	\param[in]	keypointIdxA: Index of keypoint to be triangulated in frame A.
 *	\param[in]	keypointIdxB: Index of keypoint to be triangulated in frame B.
 *	\param[out] outHomogeneousPoint_A: Output triangulation in A-coordinates.
 *	\param[out] outPointUOplus_A: Output uncertainty, represented w.r.t. ceres disturbance, in A-coordinates.
 *	\param[out] outCanBeInitialized Whether or not the triangulation can be considered initialized.
 *	\param[in]	sigmaRay Ray uncertainty.
 *	\return 3-sigma consistency check result.
 */
template<class CAMERA_GEOMETRY_T>
bool ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::stereoTriangulate(
    size_t keypointIdxA, size_t keypointIdxB,
    Eigen::Vector4d& outHomogeneousPoint_A, Eigen::Matrix3d& outPointUOplus_A,
    bool& outCanBeInitialized, double sigmaRay) const 
{
  OKVIS_ASSERT_TRUE_DBG(Exception,frameA_&&frameB_,"initialize with frames before use!");

  // get the triangulation
  bool canBeInitialized;
  if (!stereoTriangulate(keypointIdxA, keypointIdxB, outHomogeneousPoint_A, canBeInitialized,sigmaRay)){
    return false;
  }

  // and get the uncertainty /
  getUncertainty(keypointIdxA, keypointIdxB, outHomogeneousPoint_A,outPointUOplus_A, outCanBeInitialized);
  outCanBeInitialized &= canBeInitialized; // be conservative -- if the initial one failed, the 2nd should, too...
  return true;
}

// Get triangulation uncertainty.
/**
  *  \brief Get triangulation uncertainty.
  *  \param[in] keypointIdxA: Index of keypoint to be triangulated in frame A.
  *  \param[in] keypointIdxB: Index of keypoint to be triangulated in frame B.
  *  \param[in] homogeneousPoint_A: Input triangulated point in A-coordinates.
  *  \param[out] outPointUOplus_A: Output uncertainty, represented w.r.t. ceres disturbance, in A-coordinates.
  *  \param[out] outCanBeInitialized Whether or not the triangulation can be considered initialized.
  */
  //在实际的测量的过程中我们其实只是用了outCanBeInitialized这个输出变量，没有使用outPointUOplus_A
  //同时更新了ProbabilisticStereoTriangulator类中的H_矩阵
  //详见算法实现文档
template<class CAMERA_GEOMETRY_T>
void ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::getUncertainty(
    size_t keypointIdxA, size_t keypointIdxB,
    const Eigen::Vector4d& homogeneousPoint_A,
    Eigen::Matrix3d& outPointUOplus_A, bool& outCanBeInitialized) const 
{
  OKVIS_ASSERT_TRUE_DBG(Exception,frameA_&&frameB_,"initialize with frames before use!");

  // also get the point in the other coordinate representation
  //Eigen::Vector4d& homogeneousPoint_B=_T_BA*homogeneousPoint_A;
  Eigen::Vector4d hPA = homogeneousPoint_A;//A相机坐标系下的三角化的坐标点

  // calculate point uncertainty by constructing the lhs of the Gauss-Newton equation system.
  // note: the transformation T_WA is assumed constant and identity w.l.o.g.
  Eigen::Matrix<double, 9, 9> H = H_;

  //	keypointA_t& kptA = _frameA_ptr->keypoint(keypointIdxA);
  //	keypointB_t& kptB = _frameB_ptr->keypoint(keypointIdxB);
  Eigen::Vector2d kptA, kptB;
  frameA_->getKeypoint(camIdA_, keypointIdxA, kptA);//得到A相机的特征点kptA
  frameB_->getKeypoint(camIdB_, keypointIdxB, kptB);//得到B相机的特征点kptB

  // assemble the stuff from the reprojection errors
  //1.将三角化的点投影到A相机坐标系下计算其雅克比J_hpA J_hpA_min
  double keypointStdDev;
  frameA_->getKeypointSize(camIdA_, keypointIdxA, keypointStdDev);//A相机特征点的方差
  keypointStdDev = 0.8 * keypointStdDev / 12.0;
  Eigen::Matrix2d inverseMeasurementCovariance = Eigen::Matrix2d::Identity() * (1.0 / (keypointStdDev * keypointStdDev));
  ::okvis::ceres::ReprojectionError<CAMERA_GEOMETRY_T> reprojectionErrorA(frameA_->geometryAs<CAMERA_GEOMETRY_T>(camIdA_), 0, kptA,inverseMeasurementCovariance);
  //typename keypointA_t::measurement_t residualA;
  Eigen::Matrix<double, 2, 1> residualA;
  Eigen::Matrix<double, 2, 4, Eigen::RowMajor> J_hpA;//重投影误差中对地图点的雅克比
  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_hpA_min;
  double* jacobiansA[3];
  jacobiansA[0] = 0;  // do not calculate, T_WA is fixed identity transform
  jacobiansA[1] = J_hpA.data();
  jacobiansA[2] = 0;  // fixed extrinsics
  double* jacobiansA_min[3];
  jacobiansA_min[0] = 0;  // do not calculate, T_WA is fixed identity transform
  jacobiansA_min[1] = J_hpA_min.data();
  jacobiansA_min[2] = 0;  // fixed extrinsics
  const double* parametersA[3];
  //const double* test = _poseA.parameters();
  parametersA[0] = poseA_.parameters();//相机A 世界坐标系到imu坐标系的变换
  parametersA[1] = hPA.data();
  parametersA[2] = extrinsics_.parameters();
  reprojectionErrorA.EvaluateWithMinimalJacobians(parametersA, residualA.data(),jacobiansA, jacobiansA_min);

  //2.将三角化的点投影B相机坐标系下计算其雅克比J_hpA J_hpA_min
  inverseMeasurementCovariance.setIdentity();
  frameB_->getKeypointSize(camIdB_, keypointIdxB, keypointStdDev);//B相机特征点的方差
  keypointStdDev = 0.8 * keypointStdDev / 12.0;
  inverseMeasurementCovariance *= 1.0 / (keypointStdDev * keypointStdDev);

  ::okvis::ceres::ReprojectionError<CAMERA_GEOMETRY_T> reprojectionErrorB(frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_), 0, kptB,inverseMeasurementCovariance);
  Eigen::Matrix<double, 2, 1> residualB;
  Eigen::Matrix<double, 2, 7, Eigen::RowMajor> J_TB;
  Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J_TB_min;
  Eigen::Matrix<double, 2, 4, Eigen::RowMajor> J_hpB;
  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_hpB_min;
  double* jacobiansB[3];
  jacobiansB[0] = J_TB.data();
  jacobiansB[1] = J_hpB.data();
  jacobiansB[2] = 0;  // fixed extrinsics
  double* jacobiansB_min[3];
  jacobiansB_min[0] = J_TB_min.data();
  jacobiansB_min[1] = J_hpB_min.data();
  jacobiansB_min[2] = 0;  // fixed extrinsics
  const double* parametersB[3];
  parametersB[0] = poseB_.parameters();//B相对A的位姿
  parametersB[1] = hPA.data();
  parametersB[2] = extrinsics_.parameters();
  reprojectionErrorB.EvaluateWithMinimalJacobians(parametersB, residualB.data(),jacobiansB, jacobiansB_min);

  // evaluate again closer:
  // /将hPA点沿着AB连线的中线向下移动了0.8左右，不懂这里的目的??????????
  hPA.head<3>() = 0.8 * (hPA.head<3>() - T_AB_.r() / 2.0 * hPA[3]) + T_AB_.r() / 2.0 * hPA[3];
  reprojectionErrorB.EvaluateWithMinimalJacobians(parametersB, residualB.data(),jacobiansB, jacobiansB_min);
  if (residualB.transpose() * residualB < 4.0)
    outCanBeInitialized = false;
  else
    outCanBeInitialized = true;

  // now add to H:
  //H是一个9*9的矩阵
  H.bottomRightCorner<3, 3>() += J_hpA_min.transpose() * J_hpA_min;
  H.topLeftCorner<6, 6>() += J_TB_min.transpose() * J_TB_min;
  H.topRightCorner<6, 3>() += J_TB_min.transpose() * J_hpB_min;
  H.bottomLeftCorner<3, 6>() += J_hpB_min.transpose() * J_TB_min;
  H.bottomRightCorner<3, 3>() += J_hpB_min.transpose() * J_hpB_min;

  // invert (if invertible) to get covariance:
  Eigen::Matrix<double, 9, 9> cov;
  if (H.colPivHouseholderQr().rank() < 9) //H帧不满秩
  {
    outCanBeInitialized = false;
    return;
  }
  cov = H.inverse();  // FIXME: use the QR decomposition for this...
  outPointUOplus_A = cov.bottomRightCorner<3, 3>();//与残差关于特征点在A帧相机坐标系位置的海瑟矩阵有关
}

// Compute the reprojection error.
/**
 * @brief Compute the reprojection error.
 * @param[in] frame 			Multiframe.
 * @param[in] camId 			Camera ID.
 * @param[in] keypointId		ID of keypoint to calculate error for.
 * @param[in] homogeneousPoint	Homogeneous coordinates of point to calculate error for
 * @param[out] outError 		Reprojection error.
 * @return True if reprojection was successful.
 */
 //详见算实现文档
 /*
 此时我们已经得到了三角化的点在A相机坐标系下的坐标，现在我们需要计算误差。
 我们将这个三角化的点重新投影回相机A然后得到一个坐标p, 此时这个特征点有自己的二维像素坐标pm,
 则我们可以计算的到误差=outError
 */
template<class CAMERA_GEOMETRY_T>
bool ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>::computeReprojectionError4(
    const std::shared_ptr<okvis::MultiFrame>& frame, size_t camId,
    size_t keypointId, const Eigen::Vector4d& homogeneousPoint,
    double& outError) const {

  OKVIS_ASSERT_LT_DBG(Exception, keypointId, frame->numKeypoints(camId),
      "Index out of bounds");
  Eigen::Vector2d y;
  //将三角化后的点投影到相机图像中
  //搜索 CameraBase::ProjectionStatus PinholeCamera<DISTORTION_T>::projectHomogeneous(
  okvis::cameras::CameraBase::ProjectionStatus status = frame->geometryAs<CAMERA_GEOMETRY_T>(camId)->projectHomogeneous(homogeneousPoint, &y);
  if (status == okvis::cameras::CameraBase::ProjectionStatus::Successful) 
  {
    Eigen::Vector2d k;
    Eigen::Matrix2d inverseCov = Eigen::Matrix2d::Identity();
    double keypointStdDev;
    frame->getKeypoint(camId, keypointId, k);//得到特征点的坐标 k
    frame->getKeypointSize(camId, keypointId, keypointStdDev);//得到特征点的方差
    keypointStdDev = 0.8 * keypointStdDev / 12.0;
    inverseCov *= 1.0 / (keypointStdDev * keypointStdDev);

    y -= k;
    outError = y.dot(inverseCov * y);
    return true;
  } else
    return false;
}

template class ProbabilisticStereoTriangulator<
    okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> > ;
template class ProbabilisticStereoTriangulator<
    okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > ;
template class ProbabilisticStereoTriangulator<
    okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion8> > ;

}  // namespace triangulation
}  // namespace okvis
