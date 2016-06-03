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
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans,
        std::vector<std::vector<cv::Point2d>> &projection,
        const int which_cam
        ) {
    for(int s=0; s<scans.size(); s++) {
        projection.push_back(std::vector<cv::Point2d>());
        for(int i=0, _i = scans[s]->size(); i<_i; i++) {
            Eigen::Vector4d P = scans[s]->at(i).getVector4fMap().cast<double>();
            Eigen::Vector3d p = cam_mat[which_cam] * P;
            cv::Point2d c(p(0)/p(2), p(1)/p(2));
            if(p(2) > 0 && c.x >= 0 && c.x < img_width
                    && c.y >= 0 && c.y < img_height) {
                projection[s].push_back(c);
            }
        }
    }
}
