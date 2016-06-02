#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <deque>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

#include <Eigen/StdVector>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "kitti.h"
#include "velo.h"

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "Usage: velo kittidatasetnumber. e.g. velo 00" << std::endl;
        return 1;
    }
    std::string dataset = argv[1];
    loadCalibration(dataset);
    loadTimes(dataset);

    std::cerr << velo_to_cam << std::endl;

    std::cerr << times.size() << std::endl;

    cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create();
    cv::Ptr<cv::GFTTDetector> gftt = cv::GFTTDetector::create(corner_count, 0.01, 3);

    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;

    char video[] = "video";
    cvNamedWindow(video);

    for(int frame = 0, _frame = times.size(); frame < _frame; frame++) {
        std::cerr << "Frame: " << frame << std::endl;
        cv::Mat img = loadImage(dataset, frame);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
                new pcl::PointCloud<pcl::PointXYZ>);
        loadPoints(cloud, dataset, frame);

        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scans;
        segmentPoints(cloud, scans);

        detectFeatures(keypoints,
                descriptors,
                gftt,
                freak,
                img,
                frame);

        cv::Mat draw;
        cv::drawKeypoints(img, keypoints[frame], draw);
        cv::imshow(video, draw);
        cvWaitKey(1);
    }
    cvWaitKey();
    return 0;
}
