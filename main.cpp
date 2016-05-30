#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <deque>
#include <string>
#include <complex>

#include <Eigen/StdVector>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "Usage: velo kittidatasetnumber. e.g. velo 00" << std::endl;
    }
    std::string dataset = argv[1];
    loadCalibration(dataset);
    loadTimes(dataset);

    std::cerr << velo_to_cam << std::endl;

    std::cerr << times.size() << std::endl;

    /*

    poses.push_back(Eigen::Matrix4d::Identity(4,4));
    while(std::cin >> velopath) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr points(new pcl::PointCloud<pcl::PointXYZ>);
        loadkittipoints(points, velopath);
        pcl::PointCloud<pcl::PointXYZ>::Ptr points_transformed(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*points, *points_transformed, velo_to_cam);
        clouds.push_back(points_transformed);
        pcl::PointCloud<pcl::PointXYZ>::Ptr points_filtered (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
        approximate_voxel_filter.setLeafSize(0.1, 0.1, 0.1);
        approximate_voxel_filter.setInputCloud(points_transformed);
        approximate_voxel_filter.filter(*points_filtered);
        clouds_filtered.push_back(points_filtered);
        for(int cam = 0; cam < num_cams; cam++) {
            std::cin >> imgpath;
            std::cerr << "loading " << imgpath << std::endl;
            cv::Mat img = cv::imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);
            keypoints[cam].push_back(std::vector<cv::KeyPoint>());
            descriptors[cam].push_back(cv::Mat());
            detect_features(keypoints[cam][local_frame], descriptors[cam][local_frame], brisk, img);
        }
        if(frame % bundle_length == 0 && frame != 0) {
            match(keypoints, descriptors, clouds, clouds_filtered, clouds_acc, poses, matcher);
            while(keypoints[0].size() > 1) {
                for(int cam = 0; cam < num_cams; cam++) {
                    keypoints[cam].pop_front();
                    descriptors[cam].pop_front();
                }
                clouds[0].reset();
                clouds.pop_front();
                clouds_filtered[0].reset();
                clouds_filtered.pop_front();
            }
            while(clouds_acc.size() > 1) {
                clouds_acc[0].reset();
                clouds_acc.pop_front();
            }
            local_frame = 0;
        }
        frame++;
        local_frame++;
    }
    */
    return 0;

}
