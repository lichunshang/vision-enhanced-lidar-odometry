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

#include "utility.h"
#include "kitti.h"
#include "costfunctions.h"
#include "velo.h"

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "Usage: velo kittidatasetnumber. e.g. velo 00" << std::endl;
        return 1;
    }
    std::string dataset = argv[1];
    std::ofstream output;
    output.open(("results/" + std::string(argv[1]) + ".txt").c_str());
    loadCalibration(dataset);
    loadTimes(dataset);

    for(int i=0; i<num_cams; i++) {
        std::cerr << cam_mat[i] << std::endl;
    }

    //std::cerr << velo_to_cam << std::endl;

    //std::cerr << times.size() << std::endl;

    int num_frames = times.size();

    cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create();
    cv::Ptr<cv::GFTTDetector> gftt = cv::GFTTDetector::create(corner_count, 0.01, 3);

    std::vector<std::vector<std::vector<cv::KeyPoint>>> keypoints(num_cams,
            std::vector<std::vector<cv::KeyPoint>>(num_frames));
    std::vector<std::vector<cv::Mat>> descriptors(num_cams,
            std::vector<cv::Mat>(num_frames));
    std::vector<std::vector<std::vector<int>>> has_depth(num_cams,
            std::vector<std::vector<int>>(num_frames));
    std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> kp_with_depth(
            num_cams,
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>(num_frames));

    char video[] = "video";
    cvNamedWindow(video);

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

    for(int frame = 0; frame < num_frames; frame++) {
        std::cerr << "Frame (" << dataset << "): " 
            << frame << "/" << num_frames << std::endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
                new pcl::PointCloud<pcl::PointXYZ>);
        loadPoints(cloud, dataset, frame);

        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scans;
        segmentPoints(cloud, scans);
        if(scans.size() != 64) {
            std::cerr << "Scan " << frame << " has " << scans.size() << std::endl;
        }
        for(int cam = 0; cam<num_cams; cam++) {
            cv::Mat img = loadImage(dataset, cam, frame);

            auto a = clock()/double(CLOCKS_PER_SEC);
            detectFeatures(keypoints[cam][frame],
                    descriptors[cam][frame],
                    gftt,
                    freak,
                    img);

            auto b = clock()/double(CLOCKS_PER_SEC);
            std::vector<std::vector<cv::Point2f>> projection;
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scans_valid;
            projectLidarToCamera(scans, projection, scans_valid, cam);

            auto c = clock()/double(CLOCKS_PER_SEC);
            kp_with_depth[cam][frame] = 
                pcl::PointCloud<pcl::PointXYZ>::Ptr(
                        new pcl::PointCloud<pcl::PointXYZ>);
            has_depth[cam][frame] = 
                featureDepthAssociation(scans_valid,
                    projection,
                    keypoints[cam][frame],
                    kp_with_depth[cam][frame]);
            auto d = clock()/double(CLOCKS_PER_SEC);
            std::cerr << "clock (" << cam << "): " << a << " " << b << " " << c << " " << d << std::endl;
            if(cam == 0) {
                cv::Mat draw;
                cvtColor(img, draw, cv::COLOR_GRAY2BGR);
                //cv::drawKeypoints(img, keypoints[frame], draw);
                for(int k=0; k<keypoints[cam][frame].size(); k++) {
                    auto p = keypoints[cam][frame][k];
                    if(has_depth[cam][frame][k] != -1) {
                        cv::circle(draw, p.pt, 3, cv::Scalar(0, 0, 255), -1, 8, 0);
                    } else {
                        cv::circle(draw, p.pt, 3, cv::Scalar(255, 200, 0), -1, 8, 0);
                    }
                }
                for(int s=0, _s = projection.size(); s<_s; s++) {
                    auto P = projection[s];
                    /*
                    if(P.size()) {
                        std::cerr << s << ": " 
                            << P[0] << " " 
                            << P[P.size()-1] << " " 
                            << P.size() << std::endl;
                    }
                    */
                    for(auto p : P) {
                        cv::circle(draw, p, 1, 
                                cv::Scalar(0, 128, 0), -1, 8, 0);
                    }
                }
                cv::imshow(video, draw);
                cvWaitKey(1);
                //cvWaitKey();
            }
        }

        if(frame > 0) {
            pose *= frameToFrame(
                    descriptors,
                    keypoints,
                    kp_with_depth,
                    has_depth,
                    frame,
                    frame-1);
        }
        output_line(pose, output);

    }
    cvWaitKey();
    return 0;
}
