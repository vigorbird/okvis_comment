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
 *  Created on: Jun 26, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file okvis_app_synchronous.cpp
 * @brief This file processes a dataset.
 
 This node goes through a dataset in order and waits until all processing is done
 before adding a new message to algorithm

 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <memory>
#include <functional>
#include <atomic>

#include <Eigen/Core>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop
#include <okvis/VioParametersReader.hpp>
#include <okvis/ThreadedKFVio.hpp>

#include <boost/filesystem.hpp>

class PoseViewer
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  constexpr static const double imageSize = 500.0;
  PoseViewer()
  {
    cv::namedWindow("OKVIS Top View");
    _image.create(imageSize, imageSize, CV_8UC3);
    drawing_ = false;
    showing_ = false;
  }
  // this we can register as a callback
  void publishFullStateAsCallback(
      const okvis::Time & /*t*/, const okvis::kinematics::Transformation & T_WS,
      const Eigen::Matrix<double, 9, 1> & speedAndBiases,
      const Eigen::Matrix<double, 3, 1> & /*omega_S*/)
  {

    // just append the path
    Eigen::Vector3d r = T_WS.r();
    Eigen::Matrix3d C = T_WS.C();
    _path.push_back(cv::Point2d(r[0], r[1]));
    _heights.push_back(r[2]);
    // maintain scaling
    if (r[0] - _frameScale < _min_x)
      _min_x = r[0] - _frameScale;
    if (r[1] - _frameScale < _min_y)
      _min_y = r[1] - _frameScale;
    if (r[2] < _min_z)
      _min_z = r[2];
    if (r[0] + _frameScale > _max_x)
      _max_x = r[0] + _frameScale;
    if (r[1] + _frameScale > _max_y)
      _max_y = r[1] + _frameScale;
    if (r[2] > _max_z)
      _max_z = r[2];
    _scale = std::min(imageSize / (_max_x - _min_x), imageSize / (_max_y - _min_y));

    // draw it
    while (showing_) {
    }
    drawing_ = true;
    // erase
    _image.setTo(cv::Scalar(10, 10, 10));
    drawPath();
    // draw axes
    Eigen::Vector3d e_x = C.col(0);
    Eigen::Vector3d e_y = C.col(1);
    Eigen::Vector3d e_z = C.col(2);
    cv::line(
        _image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(
            _path.back() + cv::Point2d(e_x[0], e_x[1]) * _frameScale),
        cv::Scalar(0, 0, 255), 1, CV_AA);
    cv::line(
        _image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(
            _path.back() + cv::Point2d(e_y[0], e_y[1]) * _frameScale),
        cv::Scalar(0, 255, 0), 1, CV_AA);
    cv::line(
        _image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(
            _path.back() + cv::Point2d(e_z[0], e_z[1]) * _frameScale),
        cv::Scalar(255, 0, 0), 1, CV_AA);

    // some text:
    std::stringstream postext;
    postext << "position = [" << r[0] << ", " << r[1] << ", " << r[2] << "]";
    cv::putText(_image, postext.str(), cv::Point(15,15),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    std::stringstream veltext;
    veltext << "velocity = [" << speedAndBiases[0] << ", " << speedAndBiases[1] << ", " << speedAndBiases[2] << "]";
    cv::putText(_image, veltext.str(), cv::Point(15,35),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

    drawing_ = false; // notify
  }
  
  void display()
  {
    while (drawing_) {
    }
    showing_ = true;
    cv::imshow("OKVIS Top View", _image);
    showing_ = false;
    cv::waitKey(1);
  }
  
 private:
  cv::Point2d convertToImageCoordinates(const cv::Point2d & pointInMeters) const
  {
    cv::Point2d pt = (pointInMeters - cv::Point2d(_min_x, _min_y)) * _scale;
    return cv::Point2d(pt.x, imageSize - pt.y); // reverse y for more intuitive top-down plot
  }
  void drawPath()
  {
    for (size_t i = 0; i + 1 < _path.size(); ) {
      cv::Point2d p0 = convertToImageCoordinates(_path[i]);
      cv::Point2d p1 = convertToImageCoordinates(_path[i + 1]);
      cv::Point2d diff = p1-p0;
      if(diff.dot(diff)<2.0){
        _path.erase(_path.begin() + i + 1);  // clean short segment
        _heights.erase(_heights.begin() + i + 1);
        continue;
      }
      double rel_height = (_heights[i] - _min_z + _heights[i + 1] - _min_z)
                      * 0.5 / (_max_z - _min_z);
      cv::line(
          _image,
          p0,
          p1,
          rel_height * cv::Scalar(255, 0, 0)
              + (1.0 - rel_height) * cv::Scalar(0, 0, 255),
          1, CV_AA);
      i++;
    }
  }
  cv::Mat _image;
  std::vector<cv::Point2d> _path;
  std::vector<double> _heights;
  double _scale = 1.0;
  double _min_x = -0.5;
  double _min_y = -0.5;
  double _min_z = -0.5;
  double _max_x = 0.5;
  double _max_y = 0.5;
  double _max_z = 0.5;
  const double _frameScale = 0.2;  // [m]
  std::atomic_bool drawing_;//原子操作 避免多线程对同一资源的操作
  std::atomic_bool showing_;
};

// this is just a workbench. most of the stuff here will go into the Frontend class.
//输入的参数命令 = .okvis_app_synchronous path/to/okvis/config/config_fpga_p2_euroc.yaml path/to/MH_01_easy/mav0/
int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  if (argc != 3 && argc != 4) 
  {
    LOG(ERROR)<<
    "Usage: ./" << argv[0] << " configuration-yaml-file dataset-folder [skip-first-seconds]";
    return -1;
  }

  okvis::Duration deltaT(0.0);
  if (argc == 4) 
  {
    deltaT = okvis::Duration(atof(argv[3]));
  }

  // read configuration file
  std::string configFilename(argv[1]);//配置文件名

  okvis::VioParametersReader vio_parameters_reader(configFilename);
  okvis::VioParameters parameters;
  vio_parameters_reader.getParameters(parameters);

  //搜索  ThreadedKFVio构造函数
  okvis::ThreadedKFVio okvis_estimator(parameters);//超级重要的函数 里面开启了线程!!!!!!!!!!!!!!!!!!!!!! 并设置了阻塞的状态为false

  PoseViewer poseViewer;
  //PoseViewer::publishFullStateAsCallback这个函数在publisherLoop中会被调用一次
  //bind函数绑定带参数的某个类的成员函数时，第二个输入的参数应该是这个类的引用
  //设定了publisherLoop线程中的回调函数
  okvis_estimator.setFullStateCallback(  std::bind(&PoseViewer::publishFullStateAsCallback, 
  													&poseViewer, 
  													std::placeholders::_1, 
  													std::placeholders::_2,
  													std::placeholders::_3, 
  													std::placeholders::_4));

  okvis_estimator.setBlocking(true);//设置了阻塞的状态为 true

  // the folder path
  //数据集文件夹所在位置
  std::string path(argv[2]);

  const unsigned int numCameras = parameters.nCameraSystem.numCameras();//如果是双目的话这个参数等于2

  // open the IMU file
  std::string line;
  std::ifstream imu_file(path + "/imu0/data.csv");
  /*这里我们把保证代码稳定性的代码注释掉 方便看
  if (!imu_file.good()) {
    LOG(ERROR)<< "no imu file found at " << path+"/imu0/data.csv";
    return -1;
  }*/
  int number_of_lines = 0;
  while (std::getline(imu_file, line))
    ++number_of_lines;
  /*这里我们把保证代码稳定性的代码注释掉 方便看
  LOG(INFO)<< "No. IMU measurements: " << number_of_lines-1;
  if (number_of_lines - 1 <= 0) {
    LOG(ERROR)<< "no imu messages present in " << path+"/imu0/data.csv";
    return -1;
  }*/
  // set reading position to second line
  imu_file.clear();
  imu_file.seekg(0, std::ios::beg);
  std::getline(imu_file, line);//为了读取第一行，为了方便后面的直接读取数据

  std::vector<okvis::Time> times;
  okvis::Time latest(0);
  int num_camera_images = 0;
  std::vector < std::vector < std::string >> image_names(numCameras);//存储的是左右相机的图像名称。这个变量存储的顺序是左图像按照时间排序的名称，右图像按照时间排序的名称
  for (size_t i = 0; i < numCameras; ++i) //遍历左右图像
  {
    num_camera_images = 0;
    std::string folder(path + "/cam" + std::to_string(i) + "/data");

    //遍历某个相机的所有图像文件
    for (auto it = boost::filesystem::directory_iterator(folder); it != boost::filesystem::directory_iterator(); it++) 
	{
      if (!boost::filesystem::is_directory(it->path())) {  //we eliminate directories
        num_camera_images++;
        image_names.at(i).push_back(it->path().filename().string());
      } else {
        continue;
      }
    }

    if (num_camera_images == 0) {
      LOG(ERROR)<< "no images at " << folder;
      return 1;
    }

    LOG(INFO)<< "No. cam " << i << " images: " << num_camera_images;
    // the filenames are not going to be sorted. So do this here
    std::sort(image_names.at(i).begin(), image_names.at(i).end());
  }

  std::vector < std::vector <std::string>::iterator > cam_iterators(numCameras);
  for (size_t i = 0; i < numCameras; ++i) 
  {
    cam_iterators.at(i) = image_names.at(i).begin();
  }

  //下面开始主循环了!!!!!!!!!!!!!!!!!!!!!!!!!
  int counter = 0;
  okvis::Time start(0.0);//存储的是第一张左图像的时间戳
  //遍历所有的图像
  while (true) 
  {
    okvis_estimator.display();//
    poseViewer.display();

    // check if at the end
    for (size_t i = 0; i < numCameras; ++i) 
	{
      if (cam_iterators[i] == image_names[i].end()) 
	  {
        std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
        cv::waitKey();
        return 0;
      }
    }

    /// add images
    okvis::Time t;

    //将左右图像和imu测量值加入到okvis_estimator结构中
    for (size_t i = 0; i < numCameras; ++i) //遍历左右图像
	{
      cv::Mat filtered = cv::imread(path + "/cam" + std::to_string(i) + "/data/" + *cam_iterators.at(i),cv::IMREAD_GRAYSCALE);
      std::string nanoseconds = cam_iterators.at(i)->substr(cam_iterators.at(i)->size() - 13, 9);
	  std::string seconds = cam_iterators.at(i)->substr(0, cam_iterators.at(i)->size() - 13);
      t = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));//表示相机的时间戳
      if (start == okvis::Time(0.0)) //只有第一个左相机时才会进入这个条件，而且只进入一次
	  {
        start = t;
      }

      // get all IMU measurements till then
      //读取imu的数据直到图像的时间戳
      okvis::Time t_imu = start;
      do {
		        if (!std::getline(imu_file, line)) 
				{
		          std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
		          cv::waitKey();
		          return 0;
		        }

		        std::stringstream stream(line);
		        std::string s;
		        std::getline(stream, s, ',');
		        std::string nanoseconds = s.substr(s.size() - 9, 9);
		        std::string seconds = s.substr(0, s.size() - 9);

		        Eigen::Vector3d gyr;//得到IMU角度速度的读数  单位是 rad/s
		        for (int j = 0; j < 3; ++j) 
				{
		          std::getline(stream, s, ',');
		          gyr[j] = std::stof(s);
		        }

		        Eigen::Vector3d acc;//得到IMU加速度的读数 单位是 m/s2
		        for (int j = 0; j < 3; ++j) 
				{
		          std::getline(stream, s, ',');
		          acc[j] = std::stof(s);
		        }

		        t_imu = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));//得到imu的时间戳
		        // add the IMU measurement for (blocking) processing
		        if (t_imu - start + okvis::Duration(1.0) > deltaT)//只保留第一帧前面1s之内的数据， 
				{
					//向imu队列中插入imu的测量值 搜索 bool ThreadedKFVio::addImuMeasurement(
		          okvis_estimator.addImuMeasurement(t_imu, acc, gyr);//比较重要的函数!!!!!!!!!!!!!!!!!!1
		        }

      } while (t_imu <= t);//do while先执行后判断

      // add the image to the frontend for (blocking) processing
      if (t - start > deltaT) //deltaT默认=0
	  {
	    //搜索 bool ThreadedKFVio::addImage(
        okvis_estimator.addImage(t, i, filtered);//filtered=读取的灰度图像 i表示是左图像还是右图像，加入到图像队列中
      }

      cam_iterators[i]++;
    }//表示遍历完左右图像了
	
    ++counter;

    // display progress 显示处理的进度
    if (counter % 20 == 0) 
	{
      std::cout << "\rProgress: "<< int(double(counter) / double(num_camera_images) * 100) << "%  "<< std::flush;
    }

  }//结束最大的while循环了

  
  std::cout << std::endl << std::flush;
  return 0;
}
