#pragma once

void detectFeatures(
        std::vector<cv::KeyPoint> &keypoints,
        cv::Mat &descriptors,
        const cv::Ptr<cv::FeatureDetector> detector,
        const cv::Ptr<cv::DescriptorExtractor> extractor,
        cv::Mat &img
        ) {
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
                keypoints.push_back(*itk);
            }
        }
    }
    extractor->compute(img, keypoints, descriptors);
}

void projectLidarToCamera(
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans,
        std::vector<std::vector<cv::Point2f>> &projection,
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans_valid,
        const int which_cam
        ) {

    int bad = 0;

    for(int s=0; s<scans.size(); s++) {
        pcl::PointCloud<pcl::PointXYZ> projected_points;
        projection.push_back(std::vector<cv::Point2f>());
        scans_valid.push_back(pcl::PointCloud<pcl::PointXYZ>::Ptr(
                    new pcl::PointCloud<pcl::PointXYZ>));
        for(int i=0, _i = scans[s]->size(); i<_i; i++) {
            Eigen::Vector3f p = cam_mat[which_cam] 
                * scans[s]->at(i).getVector4fMap();
            pcl::PointXYZ pp(p(0), p(1), p(2));
            cv::Point2f c(p(0)/p(2), p(1)/p(2));
            if(p(2) > 0 && c.x >= 0 && c.x < img_width
                    && c.y >= 0 && c.y < img_height) {
                // remove points occluded by current point
                while(projection[s].size() > 0
                        && c.x < projection[s].back().x
                        && p(2) < projected_points.back().z) {
                    projection[s].pop_back();
                    projected_points.points.pop_back();
                    scans_valid[s]->points.pop_back();
                    bad++;
                }
                // ignore occluded points
                if(projection[s].size() > 0
                        && c.x < projection[s].back().x
                        && p(2) > projected_points.back().z) {
                    bad++;
                    continue;
                }
                projection[s].push_back(c);
                projected_points.push_back(pp);
                scans_valid[s]->push_back(scans[s]->at(i));
            }
        }
        //std::cerr << s << " " << scans_valid[s]->size()
        //    << " " << scans_valid[s]->points.size() << std::endl;
    }
    // std::cerr << "bad: " << bad << std::endl;
}

std::vector<int> featureDepthAssociation(
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans,
        const std::vector<std::vector<cv::Point2f>> &projection,
        const std::vector<cv::KeyPoint> &keypoints,
        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_with_depth
        ) {
    std::vector<int> has_depth(keypoints.size(), -1);
    std::cerr << "Sizes: " <<  scans.size() << " " 
        << projection.size() << std::endl;
    int has_depth_n = 0;
    for(int k=0; k<keypoints.size(); k++) {
        cv::KeyPoint kp = keypoints[k];
        int last_interp = -1;
        for(int s=0, _s = scans.size(); s<_s; s++) {
            bool found = false;
            if(projection[s].size() == 0) continue;
            int lo = 0, hi = projection[s].size() - 2, mid = 0;
            while(lo <= hi) {
                mid = (lo + hi)/2;
                if(projection[s][mid].x > kp.pt.x) {
                    hi = mid-1;
                } else if(projection[s][mid+1].x <= kp.pt.x) {
                    lo = mid+1;
                } else {
                    found = true;
                    if(last_interp != -1 
                            && (projection[s][mid].y > kp.pt.y) !=
                                (projection[s-1][last_interp].y > kp.pt.y)
                            ) {
                        /*
                         Perform linear interpolation using four points:

                         s-1, last_interp ----- interp2 ----- s, last_interp+1
                                                   |
                                                   |
                                                   kp
                                                   |
                                                   |
                              s, mid ---------- interp1 --------- s, mid+1
                         */
                        pcl::PointXYZ interp1 = util::linterpolate(
                                scans[s]->at(mid),
                                scans[s]->at(mid+1),
                                projection[s][mid].x,
                                projection[s][mid+1].x,
                                kp.pt.x);
                        pcl::PointXYZ interp2 = util::linterpolate(
                                scans[s-1]->at(last_interp),
                                scans[s-1]->at(last_interp+1),
                                projection[s-1][last_interp].x,
                                projection[s-1][last_interp+1].x,
                                kp.pt.x);
                        float i1y = util::linterpolate(
                                projection[s][mid].y,
                                projection[s][mid+1].y,
                                projection[s][mid].x,
                                projection[s][mid+1].x,
                                kp.pt.x);
                        float i2y = util::linterpolate(
                                projection[s-1][last_interp].y,
                                projection[s-1][last_interp+1].y,
                                projection[s-1][last_interp].x,
                                projection[s-1][last_interp+1].x,
                                kp.pt.x);

                        pcl::PointXYZ kpwd = util::linterpolate(
                                interp1,
                                interp2,
                                i1y,
                                i2y,
                                kp.pt.y);

                        keypoints_with_depth->push_back(kpwd);
                        /*
                        std::cerr << "depth association: " << kp.pt
                            << " " << projection[s][mid]
                            << " " << projection[s][mid+1]
                            << " " << projection[s-1][last_interp]
                            << " " << projection[s-1][last_interp+1]
                            << std::endl << "                 "
                            << " " << scans[s]->at(mid)
                            << " " << scans[s]->at(mid+1)
                            << " " << scans[s-1]->at(last_interp)
                            << " " << scans[s-1]->at(last_interp+1)
                            << " " << kpwd
                            << std::endl;
                            */
                        has_depth[k] = has_depth_n;
                        has_depth_n++;
                    }
                    last_interp = mid;
                    break;
                }
            }
            if(!found) last_interp = -1;
            if(has_depth[k] != -1) break;
        }
    }
    std::cerr << "Has depth: " << has_depth_n << "/" << keypoints.size() << std::endl;
    return has_depth;
}

void frameToFrame(
        const std::vector<std::vector<cv::Mat>> &descriptors,
        const std::vector<std::vector<std::vector<cv::KeyPoint>>> &keypoints,
        const std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> &keypoints_with_depth,
        const std::vector<std::vector<std::vector<int>>> has_depth,
        int frame1,
        int frame2
        ) {
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;

    ceres::Problem problem;
    double transform[6] = {0,0,0,0,0,0};

    for(int cam = 0; cam<num_cams; cam++) {
        matcher.match(descriptors[cam][frame1], descriptors[cam][frame2], matches);
        // find minimum matching distance and filter out the ones more than twice as big as it
        double min_dist = 1e9, max_dist = 0;
        for(int i=0; i<matches.size(); i++) {
            if(matches[i].distance < min_dist) min_dist = matches[i].distance;
            if(matches[i].distance > max_dist) max_dist = matches[i].distance;
        }
        std::cerr << "Matches cam " << cam << ": " <<  matches.size() 
            << " " << min_dist << " " << max_dist
            << std::endl;
        for(int i=0; i<matches.size(); i++) {
            if(matches[i].distance > std::max(2*min_dist, match_thresh)) continue;
            int point1 = matches[i].queryIdx,
                point2 = matches[i].trainIdx;
            if(has_depth[cam][frame1][point1] != -1
                    && has_depth[cam][frame2][point2] != -1) {
                // 3D 3D
                std::cerr << "  3D 3D " << point1 << ", " << point2 << std::endl;
                const pcl::PointXYZ pointM = keypoints_with_depth[cam][frame1]->at(has_depth[cam][frame1][point1]);
                const pcl::PointXYZ pointS = keypoints_with_depth[cam][frame2]->at(has_depth[cam][frame2][point2]);
                ceres::CostFunction* cost_function = 
                    new ceres::AutoDiffCostFunction<cost3D3D,3,6>(
                            new cost3D3D(
                                pointM.x,
                                pointM.y,
                                pointM.z,
                                pointS.x,
                                pointS.y,
                                pointS.z,
                                1
                                )
                            );
                problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(cauchy_thresh_3D3D), transform);
            } else if(has_depth[cam][frame1][point1] != -1
                    && has_depth[cam][frame2][point2] == -1) {
                // 3D 2D
                std::cerr << "  3D 2D " << point1 << ", " << point2 << std::endl;
                const pcl::PointXYZ point3D = keypoints_with_depth[cam][frame1]->at(has_depth[cam][frame1][point1]);
                const auto point2D = keypoints[cam][frame2][point2];
                ceres::CostFunction* cost_function = 
                    new ceres::AutoDiffCostFunction<cost3D2D,2,6>(
                            new cost3D2D(
                                point3D.x,
                                point3D.y,
                                point3D.z,
                                point2D.pt.x,
                                point2D.pt.y,
                                cam_mat[cam](0,0),
                                cam_mat[cam](0,1),
                                cam_mat[cam](0,2),
                                cam_mat[cam](0,3),
                                cam_mat[cam](1,0),
                                cam_mat[cam](1,1),
                                cam_mat[cam](1,2),
                                cam_mat[cam](1,3),
                                cam_mat[cam](2,0),
                                cam_mat[cam](2,1),
                                cam_mat[cam](2,2),
                                cam_mat[cam](2,3),
                                0.1
                                )
                            );
                problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(cauchy_thresh_3D2D), transform);
            }
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    for(int i=0; i<6; i++) {
        std::cerr << transform[i] << " ";
    }
    std::cerr << std::endl;
}

