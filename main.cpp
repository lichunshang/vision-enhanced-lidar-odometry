#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <deque>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <list>
#include <unordered_map>
#include <random>

#include <Eigen/StdVector>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include <isam/isam.h>
#include <isam/robust.h>

#define VISUALIZE
//#define ENABLE_ICP
#define ENABLE_2D2D
//#define ENABLE_3D2D

#include "utility.h"
#include "kitti.h"
#include "costfunctions.h"
#include "velo.h"
#include "lru.h"


int main(int argc, char** argv) {
    cv::setUseOptimized(true);

    cv::cuda::setDevice(1);
    //cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
    if(argc < 2) {
        std::cout << "Usage: velo kittidatasetnumber. e.g. velo 00" << std::endl;
        return 1;
    }
    std::string dataset = argv[1];
    std::ofstream output;
    output.open(("results/" + std::string(argv[1]) + ".txt").c_str());
    loadImage(dataset, 0, 0); // to set width and height
    loadCalibration(dataset);
    loadTimes(dataset);

    int num_frames = times.size();

    // FREAK feature descriptor
    cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create(
            false, // orientation normalization
            false // scale normalization
            );
    // good features to detect
    cv::Ptr<cv::GFTTDetector> gftt = cv::GFTTDetector::create(
            corner_count,
            quality_level,
            min_distance);

    // tracked keypoints, camera canonical coordinates
    std::vector<std::vector<std::vector<cv::Point2f>>> keypoints(num_cams,
            std::vector<std::vector<cv::Point2f>>(num_frames));
    // tracked keypoints, pixel coordinates
    std::vector<std::vector<std::vector<cv::Point2f>>> keypoints_p(num_cams,
            std::vector<std::vector<cv::Point2f>>(num_frames));
    // IDs of keypoints
    std::vector<std::vector<std::vector<int>>> keypoint_ids(num_cams,
            std::vector<std::vector<int>>(num_frames));
    // FREAK descriptors of each keypoint
    std::vector<std::vector<cv::Mat>> descriptors(num_cams,
            std::vector<cv::Mat>(num_frames));
    // -1 if no depth, index of kp_with_depth otherwise
    std::vector<std::vector<std::vector<int>>> has_depth(num_cams,
            std::vector<std::vector<int>>(num_frames));
    // interpolated lidar point, physical coordinates
    std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> kp_with_depth(
            num_cams,
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>(num_frames));
    // images of current and frame, used for optical flow tracking
    std::vector<cv::Mat> imgs(num_cams);
    std::vector<cv::Mat> img_prevs(num_cams);

    // counters for keypoint id, one for each camera
    int id_counter = 0;

#ifdef VISUALIZE
    char video[] = "video";
    cvNamedWindow(video);
#endif

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    double transform[6] = {0, 0, 0, 0, 0, 1};
    ScansLRU lru;


    for(int frame = 0; frame < num_frames; frame++) {
        auto start = clock()/double(CLOCKS_PER_SEC);

        ScanData *sd = lru.get(dataset, frame);
        const auto &scans = sd->scans;
        for(int cam = 0; cam<num_cams; cam++) {
            imgs[cam] = loadImage(dataset, cam, frame);
        }

        for(int cam = 0; cam<num_cams; cam++) {
            if(frame > 0) {
                //for(int prev_cam = 0; prev_cam < num_cams; prev_cam++) {
                int prev_cam = cam;
                    trackFeatures(
                            keypoints,
                            keypoints_p,
                            keypoint_ids,
                            descriptors,
                            img_prevs[prev_cam],
                            imgs[cam],
                            prev_cam,
                            cam,
                            frame-1,
                            frame);
                //}
            }
            if(frame % detect_every == 0) {
                detectFeatures(
                        keypoints[cam],
                        keypoints_p[cam],
                        keypoint_ids[cam],
                        descriptors[cam],
                        gftt,
                        freak,
                        imgs[cam],
                        id_counter,
                        cam,
                        frame);
            }
            if(cam != 0) {
                trackFeatures(
                        keypoints,
                        keypoints_p,
                        keypoint_ids,
                        descriptors,
                        imgs[0],
                        imgs[cam],
                        0,
                        cam,
                        frame,
                        frame);
            }
            consolidateFeatures(
                    keypoints[cam][frame],
                    keypoints_p[cam][frame],
                    keypoint_ids[cam][frame],
                    descriptors[cam][frame],
                    cam);

            removeTerribleFeatures(
                    keypoints[cam][frame],
                    keypoints_p[cam][frame],
                    keypoint_ids[cam][frame],
                    descriptors[cam][frame],
                    freak,
                    imgs[cam],
                    cam);

            std::vector<std::vector<cv::Point2f>> projection;
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scans_valid;
            projectLidarToCamera(scans, projection, scans_valid, cam);

            kp_with_depth[cam][frame] =
                pcl::PointCloud<pcl::PointXYZ>::Ptr(
                        new pcl::PointCloud<pcl::PointXYZ>);
            has_depth[cam][frame] =
                featureDepthAssociation(scans_valid,
                    projection,
                    keypoints[cam][frame],
                    kp_with_depth[cam][frame]);
            img_prevs[cam] = imgs[cam];
        }

        if(frame > 0) {
            // matches are what's fed into frameToFrame,
            // good matches have outliers removed during optimization
            std::vector<std::vector<std::pair<int, int>>> matches(num_cams);
            std::vector<std::vector<std::pair<int, int>>> good_matches(num_cams);
            std::vector<std::vector<ResidualType>> residual_type(num_cams);
            ScanData *sd_prev = lru.get(dataset, frame);
            const auto &scans_prev = sd_prev->scans;
            const auto &trees = sd_prev->trees;
            matchUsingId(keypoint_ids, frame, frame-1, matches);
            pose *= frameToFrame(
                    matches,
                    keypoints,
                    kp_with_depth,
                    has_depth,
                    scans,
                    scans_prev,
                    trees,
                    frame,
                    frame-1,
                    transform,
                    good_matches,
                    residual_type);
            double end = clock()/double(CLOCKS_PER_SEC);
            std::cerr << "Frame (" << dataset << "):"
                << std::setw(5) << frame+1 << "/" << num_frames << ", "
                << std::fixed << std::setprecision(3) <<  end-start << " |";
            for(int j=0; j<6; j++) {
                std::cerr << std::setfill(' ') << std::setw(8)
                    << std::fixed << std::setprecision(4)
                    << transform[j];
            }
            std::cerr << std::endl;

#ifdef VISUALIZE
            int cam = 1;
            cv::Mat draw;
            cv::Mat img = loadImage(dataset, cam, frame);
            cvtColor(img, draw, cv::COLOR_GRAY2BGR);
            auto &K = cam_intrinsic[cam];
            //cv::drawKeypoints(img, keypoints[frame], draw);
            /*
            for(int k=0; k<keypoints[cam][frame].size(); k++) {
                auto p = keypoints_p[cam][frame][k];
                if(has_depth[cam][frame][k] != -1) {
                    cv::circle(draw, p, 2, cv::Scalar(0, 0, 255), -1, 8, 0);
                } else {
                    cv::circle(draw, p, 2, cv::Scalar(255, 200, 0), -1, 8, 0);
                }
            }
            for(int s=0, _s = projection.size(); s<_s; s++) {
                auto P = projection[s];
                for(auto p : P) {
                    auto pp = canonical2pixel(p, K);
                    cv::circle(draw, pp, 1,
                            cv::Scalar(0, 128, 0), -1, 8, 0);
                }
            }
            */

            for(auto m : matches[cam]) {
                auto p1 = keypoints_p[cam][frame][m.first];
                auto p2 = keypoints_p[cam][frame-1][m.second];
                cv::arrowedLine(draw, p1, p2,
                        cv::Scalar(0, 255, 255), 2, CV_AA);
            }

            for(int i=0; i<good_matches[cam].size(); i++) {
                auto m = good_matches[cam][i];
                auto p1 = keypoints_p[cam][frame][m.first];
                auto p2 = keypoints_p[cam][frame-1][m.second];
                cv::Scalar color = cv::Scalar(0, 255, 255);
                switch(residual_type[cam][i]) {
                    case RESIDUAL_3D3D:
                        color = cv::Scalar(0, 0, 255);
                        break;
                    case RESIDUAL_3D2D:
                        color = cv::Scalar(0, 250, 0);
                        break;
                    case RESIDUAL_2D2D:
                        color = cv::Scalar(255, 200, 0);
                        break;
                }
                cv::arrowedLine(draw, p1, p2,
                        color, 2, CV_AA);
            }

            /*
            // Draw stereo matches
            std::vector<std::pair<int, int>> intercamera_matches;
            matchUsingId(keypoint_ids, 0, 1, frame, frame,
                    intercamera_matches);
            for(auto m : intercamera_matches) {
                auto p1 = keypoints_p[0][frame][m.first];
                auto p2 = keypoints_p[1][frame][m.second];
                cv::line(draw, p1, p2, cv::Scalar(255, 0, 255), 2, CV_AA);
            }
            */
            cv::imshow(video, draw);
            cvWaitKey(1);
#endif
        }
        output_line(pose, output);
    }
    return 0;
}
