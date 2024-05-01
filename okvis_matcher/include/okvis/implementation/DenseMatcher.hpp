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
 *  Created on: 2013
 *      Author: Simon Lynen
 *    Modified: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file implementation/DenseMatcher.hpp
 * @brief Header implementation file for the DenseMatcher class.
 * @author Simon Lynen
 * @author Stefan Leutenegger
 */

#include <map>

/// \brief okvis Main namespace of this package.
namespace okvis {

// This function creates all the matching threads and assigns the best matches afterwards.
//输入的模板类型MATCHING_ALGORITHM_T = VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::VioKeyframeWindowMatchingAlgorithm(
//输入的void (DenseMatcher::*doWorkPtr)(MatchJob&, MATCHING_ALGORITHM_T*) = &DenseMatcher::template doWorkLinearMatching<matching_algorithm_t>
template<typename MATCHING_ALGORITHM_T>
void DenseMatcher::matchBody( void (DenseMatcher::*doWorkPtr)(MatchJob&, MATCHING_ALGORITHM_T*),MATCHING_ALGORITHM_T& matchingAlgorithm) 
{
  // create lock list
  std::mutex* locks = new std::mutex[matchingAlgorithm.sizeB()];

  //the pairing list
  //定义 typedef std::vector<pairing_t> pairing_list_t;
  pairing_list_t vpairs;//元素 = (图像A的id,距离)，元素的序号=图像B的特征点序号
  // a list with best matches for each "A" point
  //pairing_t是一个结构体 存储的是序号和距离
  std::vector<std::vector<pairing_t> > vMyBest;

  vMyBest.resize(matchingAlgorithm.sizeA());//A相机的特征点个数

  // this point is not paired so far, score max
  vpairs.resize(matchingAlgorithm.sizeB(),pairing_t(-1, std::numeric_limits<distance_t>::max()));

  // prepare the jobs for the threads
  std::vector<MatchJob> jobs(numMatcherThreads_);//numMatcherThreads_作者设置的是4，MatchJob是一个结构体 搜索 struct MatchJob {
  for (int i = 0; i < numMatcherThreads_; ++i) 
  {
    jobs[i].iThreadID = i;
    jobs[i].vpairs = &vpairs;
    jobs[i].vMyBest = &vMyBest;
    jobs[i].mutexes = locks;
  }

  //create all threads
  //  boost::thread_group matchers;
  for (int i = 0; i < numMatcherThreads_; ++i) //numMatcherThreads_作者设置的是4
  {
    //std::unique_ptr<okvis::ThreadPool> matcherThreadPool_;
    //定义 std::unique_ptr<okvis::ThreadPool> matcherThreadPool_;  
    //搜索 ThreadPool::enqueue(
    //主要是这个函数进行了匹配的工作
    //这里doWorkPtr(本质上是一个函数) = doWorkLinearMatching，搜索void DenseMatcher::doWorkLinearMatching(，其实这个函数就在这个文档的下面
    matcherThreadPool_->enqueue(doWorkPtr, this, jobs[i], &matchingAlgorithm);//创建一个线程，调用函数doWorkLinearMatching进行匹配，这个函数的输入参数为jobs[i], &matchingAlgorithm
    //    matchers.create_thread(boost::bind(doWorkPtr, this, jobs[i], &matchingAlgorithm));
  }

  //  matchers.join_all();
   //协调所有的线程
  matcherThreadPool_->waitForEmptyQueue();

  // Looks like running this in one thread is faster than creating 30+ new threads for every image.
  //TODO(gohlp): distribute this to n threads.

  //  for (int i = 0; i < _numMatcherThreads; ++i)
  //  {
  //	  (this->*doWorkPtr)(jobs[i], &matchingAlgorithm);
  //  }

  matchingAlgorithm.reserveMatches(vpairs.size());

  // assemble the pairs and return
  //主要是进行状态的更新
  const distance_t& const_distratiothres = matchingAlgorithm.distanceRatioThreshold();
  const distance_t& const_distthres = matchingAlgorithm.distanceThreshold();//距离的阈值设置为60
  for (size_t i = 0; i < vpairs.size(); ++i) //遍历已经匹配的特征点
  {
      //useDistanceRatioThreshold_=false,表示使用最优的绝对距离，而不是最优和次优的相对距离
    if (useDistanceRatioThreshold_ && vpairs[i].distance < const_distthres) 
	{
	      const std::vector<pairing_t>& best_matches_list = vMyBest[vpairs[i].indexA];
	      OKVIS_ASSERT_TRUE_DBG(Exception, best_matches_list[0].indexA != -1, "assertion failed");

	      if (best_matches_list[1].indexA != -1) 
		  {
	        const distance_t& best_match_distance = best_matches_list[0].distance;
	        const distance_t& second_best_match_distance = best_matches_list[1].distance;
	        // Only assign if the distance ratio better than the threshold.
	        if (best_match_distance == 0 || second_best_match_distance / best_match_distance> const_distratiothres) 
			{
		     
	          matchingAlgorithm.setBestMatch(vpairs[i].indexA, i,vpairs[i].distance);
	        }
	      } else 
	     {
	        // If there is only one matching feature, we assign it.
	        matchingAlgorithm.setBestMatch(vpairs[i].indexA, i, vpairs[i].distance);
	      }
    } else if (vpairs[i].distance < const_distthres) //默认进入这个条件
    {
    	  //搜索 void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setBestMatch(
		  //这个函数向ceres加入了残差矩阵
      	  matchingAlgorithm.setBestMatch(vpairs[i].indexA, i, vpairs[i].distance);//图像A中的特征点序号，图像B的特征点序号，两个特征点间的距离
    }
  }

  delete[] locks;
}

// Execute a matching algorithm. This is the fast, templated version. Use this.
//这里输入模板类型为 VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::VioKeyframeWindowMatchingAlgorithm(
template<typename MATCHING_ALGORITHM_T>
void DenseMatcher::match(MATCHING_ALGORITHM_T & matchingAlgorithm) 
{
  typedef MATCHING_ALGORITHM_T matching_algorithm_t;
  //搜索 VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::doSetup()
  matchingAlgorithm.doSetup();

  //搜索 void DenseMatcher::matchBody( void (DenseMatcher::*doWorkPtr)(MatchJob&, MATCHING_ALGORITHM_T*),MATCHING_ALGORITHM_T& matchingAlgorithm) 
  //搜索 void DenseMatcher::doWorkLinearMatching(
  // call the matching body with the linear matching function pointer
  matchBody(&DenseMatcher::template doWorkLinearMatching<matching_algorithm_t>,
  			matchingAlgorithm);
}

// Execute a matching algorithm implementing image space matching.
template<typename MATCHING_ALGORITHM_T>
void DenseMatcher::matchInImageSpace(MATCHING_ALGORITHM_T & matchingAlgorithm) {
  typedef MATCHING_ALGORITHM_T matching_algorithm_t;
  matchingAlgorithm.doSetup();

  // call the matching body with the image space matching function pointer
  matchBody(
      &DenseMatcher::template doWorkImageSpaceMatching<matching_algorithm_t>,
      matchingAlgorithm);
}

// This calculates the distance between to keypoint descriptors. If it is better than the /e numBest_
// found so far, it is included in the aiBest list.
/**
 * @brief This calculates the distance between to keypoint descriptors. If it is better than the /e numBest_
 *		  found so far, it is included in the aiBest list.
 * @tparam MATCHING_ALGORITHM_T The algorithm to use. E.g. a class derived from MatchingAlgorithm.
 * @param matchingAlgorithm The matching algorithm to use.
 * @param[inout] aiBest The \e numBest_ pairings found so far.
 * @param[in] shortindexA Keypoint index in frame A.
 * @param[in] i Keypoint index in frame B.
 */
//这里输入模板类型为 VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::VioKeyframeWindowMatchingAlgorithm(
//计算特征点对之间的距离的函数
template<typename MATCHING_ALGORITHM_T>
inline void DenseMatcher::listBIteration(  MATCHING_ALGORITHM_T* matchingAlgorithm, 
											std::vector<pairing_t>& aiBest,
    										size_t shortindexA, size_t i) 
{
  OKVIS_ASSERT_TRUE(std::runtime_error, matchingAlgorithm != NULL,
                    "matching algorithm is NULL");
  typename DenseMatcher::distance_t tmpdist;

  // is this better than worst found so far?
  //这是一个非常重要的函数
  tmpdist = matchingAlgorithm->distance(shortindexA, i);//计算匹配距离,搜索 virtual float distance(size_t indexA, size_t indexB) const 
  if (tmpdist < aiBest[numBest_ - 1].distance) //出现更小的距离
  {
    pairing_t tmp(static_cast<int>(i), tmpdist);
    typename std::vector<pairing_t>::iterator lb = std::lower_bound( aiBest.begin(), aiBest.end(), tmp);  //get position for insertion
    typename std::vector<pairing_t>::iterator it, it_next;
    it = it_next = aiBest.end();

    --it;
    --it_next;
    // Insert the new match value into the list
    while (it_next != lb) 
	{
      --it;
      *it_next = *it;  //move value one position to the back
      --it_next;
    }
    *lb = tmp;  //insert both index and score to the correct position to keep strict weak->strong ordering
  }
}

// The threading worker. This matches a keypoint with every other keypoint to find the best match.
//这里输入模板类型为 VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::VioKeyframeWindowMatchingAlgorithm(
template<typename MATCHING_ALGORITHM_T>
void DenseMatcher::doWorkLinearMatching( MatchJob & my_job, MATCHING_ALGORITHM_T * matchingAlgorithm) 
{
  OKVIS_ASSERT_TRUE(std::runtime_error, matchingAlgorithm != NULL,
                    "matching algorithm is NULL");
  try {
    int start = my_job.iThreadID;//当前线程的id
    distance_t const_distthres = matchingAlgorithm->distanceThreshold();//距离阈值为60
    if (useDistanceRatioThreshold_) //默认不进入这个条件
	{
      // When using the distance ratio threshold, we want to build a list of good matches
      // independent of the threshold first and then later threshold on the ratio.
      const_distthres = std::numeric_limits<distance_t>::max();
    }

    size_t sizeA = matchingAlgorithm->sizeA();//A帧A相机的特征点数
    //遍历A图像的特征点
    for (size_t shortindexA = start; shortindexA < sizeA; shortindexA += numMatcherThreads_) //步长为线程的数量，如果只有一个线程的话这里就是对A图像的特征点进行遍历
	{
      if (matchingAlgorithm->skipA(shortindexA))//如果特征点满足直接跳过的条件，由doSetup设置
        continue;

      //typename DenseMatcher::distance_t tmpdist;
      std::vector<pairing_t> & aiBest = (*my_job.vMyBest)[shortindexA];//aiBest的元素= (id，距离)

      // initialize the best match to be -1 (no match) and set the score to be the distance threshold
      // No matches worse than the distance threshold will get through.
      aiBest.resize(numBest_, pairing_t(-1, const_distthres));  //the best x matches for this feature from the long list

      size_t numElementsInListB = matchingAlgorithm->sizeB();//B帧B相机特征点的数量
      for (size_t i = 0; i < numElementsInListB; ++i) //遍历B图像的特征点
	  {
        if (matchingAlgorithm->skipB(i)) 
		{
          continue;
        }
        //匹配函数，其中
        ///matchingAlgorithm为匹配变量（储存两帧的信息），
        /// aiBest为A相机第shortindexA个特征点对应的的容器，从小到大排序
        /// A相机第shortindexA个特征点，i表示B相机第i个特征点
        //这个函数就在这个文件前面，aiBest既是输入也是输出
        //搜索 inline void DenseMatcher::listBIteration
        listBIteration(matchingAlgorithm, aiBest, shortindexA, i);

      }
	  //搜索 void DenseMatcher::assignbest(
	  //*(my_job.vpairs)和*(my_job.vMyBest)是输出变量
	  //this call assigns the match and reassigns losing matches recursively
      assignbest(static_cast<int>(shortindexA), *(my_job.vpairs), *(my_job.vMyBest), my_job.mutexes, 0); 
	  
    }
  } catch (const std::exception & e) {
    // \todo Install an error handler in the matching algorithm?
    std::cout << "\033[31mException in matching thread:\033[0m " << e.what();
  }
}

// The threading worker. This matches a keypoint with only a subset of the other keypoints
// to find the best match. (From matchingAlgorithm->getListBStartIterator() to
// MatchingAlgorithm->getListBEndIterator().
template<typename MATCHING_ALGORITHM_T>
void DenseMatcher::doWorkImageSpaceMatching(
    MatchJob & my_job, MATCHING_ALGORITHM_T* matchingAlgorithm) {
  OKVIS_ASSERT_TRUE(std::runtime_error, matchingAlgorithm != NULL,
                    "matching algorithm is NULL");
  try {
    int start = my_job.iThreadID;

    size_t numElementsInListB = matchingAlgorithm->sizeB();
    size_t numElementsInListA = matchingAlgorithm->sizeA();

    distance_t const_distthres = matchingAlgorithm->distanceThreshold();
    if (useDistanceRatioThreshold_) {
      // When using the distance ratio threshold, we want to build a list of good matches
      // independent of the threshold first and then later threshold on the ratio.
      const_distthres = std::numeric_limits<distance_t>::max();
    }

    for (size_t shortindexA = start; shortindexA < matchingAlgorithm->sizeA();
        shortindexA += numMatcherThreads_) {
      if (matchingAlgorithm->skipA(shortindexA))
        continue;

      typename DenseMatcher::distance_t tmpdist;
      std::vector<pairing_t>& aiBest = (*my_job.vMyBest)[shortindexA];

      // initialize the best match to be -1 (no match) and set the score to be the distance threshold
      // No matches worse than the distance threshold will get through.
      aiBest.resize(numBest_, pairing_t(-1, const_distthres));  //the best x matches for this feature from the long list

      typename MATCHING_ALGORITHM_T::listB_tree_structure_t::iterator itBegin =
          matchingAlgorithm->getListBStartIterator(shortindexA);
      typename MATCHING_ALGORITHM_T::listB_tree_structure_t::iterator itEnd =
          matchingAlgorithm->getListBEndIterator(shortindexA);
      //check all features from the long list
      for (typename MATCHING_ALGORITHM_T::listB_tree_structure_t::iterator it =
          itBegin; it != itEnd; ++it) {
        size_t i = it->second;

        if (matchingAlgorithm->skipB(i)) {
          continue;
        }

        listBIteration(matchingAlgorithm, aiBest, shortindexA, i);

      }

      assignbest(static_cast<int>(shortindexA), *(my_job.vpairs),
                 *(my_job.vMyBest), my_job.mutexes, 0);  //this call assigns the match and reassigns losing matches recursively
    }

  } catch (const std::exception & e) {
    // \todo Install an error handler in the matching algorithm?
    std::cout << "\033[31mException in matching thread:\033[0m " << e.what();
  }
}

}  // namespace okvis
