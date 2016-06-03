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

    for(int i=0; i<num_cams; i++) {
        std::cerr << cam_mat[i] << std::endl;
    }

    std::cerr << velo_to_cam << std::endl;

    std::cerr << times.size() << std::endl;

    cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create();
    cv::Ptr<cv::GFTTDetector> gftt = cv::GFTTDetector::create(corner_count, 0.01, 3);

    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;

    char video[] = "video";
    cvNamedWindow(video);

    for(int frame = 0, _frame = times.size(); frame < _frame; frame++) {
        //std::cerr << "Frame: " << frame << std::endl;
        cv::Mat img = loadImage(dataset, 0, frame);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
                new pcl::PointCloud<pcl::PointXYZ>);
        loadPoints(cloud, dataset, frame);

        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scans;
        segmentPoints(cloud, scans);
        if(scans.size() != 64) {
            std::cerr << "Scan " << frame << " has " << scans.size() << std::endl;
        }

        detectFeatures(keypoints,
                descriptors,
                gftt,
                freak,
                img,
                frame);

        std::vector<std::vector<cv::Point2d>> projection;
        projectLidarToCamera(scans, projection, 0);

        cv::Mat draw;
        cvtColor(img, draw, cv::COLOR_GRAY2BGR);
        //cv::drawKeypoints(img, keypoints[frame], draw);
        for(auto p : keypoints[frame]) {
            cv::circle(draw, p.pt, 2, cv::Scalar(0, 0, 255), -1, 8, 0);
        }
        for(int s=0, _s = projection.size(); s<_s; s++) {
            auto P = projection[s];
            for(auto p : P) {
                cv::circle(draw, p, 2, 
                        cv::Scalar(255 * (s%2), 255 * ((s+1)%2), 0), -1, 8, 0);
            }
        }
        cv::imshow(video, draw);
        cvWaitKey(1);
    }
    cvWaitKey();
    return 0;
}
