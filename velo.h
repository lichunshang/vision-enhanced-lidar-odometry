#pragma once

void detectFeatures(
        std::vector<std::vector<cv::KeyPoint>> &keypoints,
        std::vector<cv::Mat> &descriptors,
        cv::Ptr<cv::FeatureDetector> detector,
        cv::Ptr<cv::DescriptorExtractor> extractor,
        cv::Mat &img,
        const int frame
        ) {
    while(keypoints.size() <= frame) {
        keypoints.push_back(std::vector<cv::KeyPoint>());
    }
    while(descriptors.size() <= frame) {
        descriptors.push_back(cv::Mat());
    }
    for(int row=0; row < row_cells; row++) {
        for(int col = 0; col < col_cells; col++) {
            cv::Mat img_cell = img(
                cv::Range(cell_height * row, cell_height * (row + 1)),
                cv::Range(cell_width * col, cell_width * (col + 1))
            );
            std::vector<cv::KeyPoint> temp_keypoints;
            detector->detect(img_cell, temp_keypoints);
            for(std::vector<cv::KeyPoint>::iterator itk = temp_keypoints.begin(); 
                    itk != temp_keypoints.end(); 
                    itk++) {
                itk->pt.x += cell_width * col;
                itk->pt.y += cell_height * row;
                keypoints[frame].push_back(*itk);
            }
        }
    }
    extractor->compute(img, keypoints[frame], descriptors[frame]);
}

void projectLidarToCamera(
        std::vector<pcl::PointCloud<pcl::PointXYZ::Ptr> scans,
        std::vector<cv::Point> projection
        );
