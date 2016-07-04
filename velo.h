#pragma once

enum ResidualType {
    RESIDUAL_3D3D,
    RESIDUAL_3D2D,
    RESIDUAL_2D3D,
    RESIDUAL_2D2D
};

void detectFeatures(
        std::vector<cv::Point2f> &keypoints,
        cv::Mat &descriptors,
        const cv::Ptr<cv::FeatureDetector> detector,
        const cv::Ptr<cv::DescriptorExtractor> extractor,
        cv::Mat &img,
        const int which_cam
        ) {
    Eigen::Matrix3f Kinv = cam_intrinsic[which_cam].inverse();
    std::vector<cv::KeyPoint> cvKP;
    for(int row=0; row < row_cells; row++) {
        for(int col = 0; col < col_cells; col++) {
            cv::Mat img_cell = img(
                cv::Range(cell_height * row, cell_height * (row + 1)),
                cv::Range(cell_width * col, cell_width * (col + 1))
            );
            std::vector<cv::KeyPoint> temp_keypoints;
            detector->detect(img_cell, temp_keypoints);
            for(auto kp : temp_keypoints) {
                kp.pt.x += cell_width * col;
                kp.pt.y += cell_height * row;
                cvKP.push_back(kp);
            }
        }
    }
    // remember! compute MUTATES cvKP
    extractor->compute(img, cvKP, descriptors);
    for(auto kp : cvKP) {
        Eigen::Vector3f p;
        p << kp.pt.x, 
            kp.pt.y, 1;
        p = Kinv * p;
        keypoints.push_back(cv::Point2f(p(0)/p(2), p(1)/p(2)));
    }
}

void projectLidarToCamera(
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans,
        std::vector<std::vector<cv::Point2f>> &projection,
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans_valid,
        const int which_cam
        ) {

    int bad = 0;
    Eigen::Vector3f t = cam_trans[which_cam];

    for(int s=0; s<scans.size(); s++) {
        pcl::PointCloud<pcl::PointXYZ> projected_points;
        projection.push_back(std::vector<cv::Point2f>());
        scans_valid.push_back(pcl::PointCloud<pcl::PointXYZ>::Ptr(
                    new pcl::PointCloud<pcl::PointXYZ>));
        for(int i=0, _i = scans[s]->size(); i<_i; i++) {
            pcl::PointXYZ p = scans[s]->at(i);
            pcl::PointXYZ pp(p.x + t(0), p.y + t(1), p.z + t(2));
            cv::Point2f c(pp.x/pp.z, pp.y/pp.z);
            if(pp.z > 0 && c.x >= min_x[which_cam] && c.x < max_x[which_cam]
                    && c.y >= min_y[which_cam] && c.y < max_y[which_cam]) {
                // remove points occluded by current point
                while(projection[s].size() > 0
                        && c.x < projection[s].back().x
                        && pp.z < projected_points.back().z) {
                    projection[s].pop_back();
                    projected_points.points.pop_back();
                    scans_valid[s]->points.pop_back();
                    bad++;
                }
                // ignore occluded points
                if(projection[s].size() > 0
                        && c.x < projection[s].back().x
                        && pp.z > projected_points.back().z) {
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
    //std::cerr << "Lidar projection bad: " << bad << std::endl;
}

std::vector<int> featureDepthAssociation(
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans,
        const std::vector<std::vector<cv::Point2f>> &projection,
        const std::vector<cv::Point2f> &keypoints,
        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_with_depth
        ) {
    std::vector<int> has_depth(keypoints.size(), -1);
    /*
    std::cerr << "Sizes: " <<  scans.size() << " " 
        << projection.size() << " "
        << keypoints.size() << std::endl;
        */
    int has_depth_n = 0;
    for(int k=0; k<keypoints.size(); k++) {
        cv::Point2f kp = keypoints[k];
        int last_interp = -1;
        for(int s=0, _s = scans.size(); s<_s; s++) {
            bool found = false;
            if(projection[s].size() <= 1) {
                last_interp = -1;
                continue;
            }
            int lo = 0, hi = projection[s].size() - 2, mid = 0;
            while(lo <= hi) {
                mid = (lo + hi)/2;
                if(projection[s][mid].x > kp.x) {
                    hi = mid-1;
                } else if(projection[s][mid+1].x <= kp.x) {
                    lo = mid+1;
                } else {
                    found = true;
                    if(last_interp != -1 
                            && (projection[s][mid].y > kp.y) !=
                                (projection[s-1][last_interp].y > kp.y)
                            && abs(projection[s][mid].x - 
                                projection[s][mid+1].x) 
                                < depth_assoc_thresh
                            && abs(projection[s-1][last_interp].x - 
                                projection[s-1][last_interp+1].x) 
                                < depth_assoc_thresh
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
                        /*
                        std::cerr << "depth association: " << kp
                            << " " << projection[s][mid]
                            << " " << projection[s][mid+1]
                            << " " << projection[s-1][last_interp]
                            << " " << projection[s-1][last_interp+1]
                            << std::endl;
                        std::cerr << "                   "
                            << " " << scans[s]->at(mid)
                            << " " << scans[s]->at(mid+1)
                            << " " << scans[s-1]->at(last_interp)
                            << " " << scans[s-1]->at(last_interp+1)
                            << " ";
                            */
                        pcl::PointXYZ interp1 = util::linterpolate(
                                scans[s]->at(mid),
                                scans[s]->at(mid+1),
                                projection[s][mid].x,
                                projection[s][mid+1].x,
                                kp.x);
                        pcl::PointXYZ interp2 = util::linterpolate(
                                scans[s-1]->at(last_interp),
                                scans[s-1]->at(last_interp+1),
                                projection[s-1][last_interp].x,
                                projection[s-1][last_interp+1].x,
                                kp.x);
                        float i1y = util::linterpolate(
                                projection[s][mid].y,
                                projection[s][mid+1].y,
                                projection[s][mid].x,
                                projection[s][mid+1].x,
                                kp.x);
                        float i2y = util::linterpolate(
                                projection[s-1][last_interp].y,
                                projection[s-1][last_interp+1].y,
                                projection[s-1][last_interp].x,
                                projection[s-1][last_interp+1].x,
                                kp.x);

                        pcl::PointXYZ kpwd = util::linterpolate(
                                interp1,
                                interp2,
                                i1y,
                                i2y,
                                kp.y);

                        //std::cerr << kpwd << std::endl;

                        keypoints_with_depth->push_back(kpwd);
                        has_depth[k] = has_depth_n;
                        has_depth_n++;
                    }
                    last_interp = mid;
                    break;
                }
            }
            if(!found) {
                last_interp = -1;
            }
            if(has_depth[k] != -1) break;
        }
    }
    /*
    std::cerr << "Has depth: " << has_depth_n << "/" << keypoints.size() << std::endl;
    */
    return has_depth;
}
void residualStats(
        ceres::Problem &problem,
        std::vector<std::vector<std::pair<int, int>>> &good_matches,
        std::vector<std::vector<ResidualType>> &residual_type
        );

Eigen::Matrix4d frameToFrame(
        const std::vector<std::vector<cv::Mat>> &descriptors,
        const std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints,
        const std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> &keypoints_with_depth,
        const std::vector<std::vector<std::vector<int>>> has_depth,
        int frame1,
        int frame2,
        double transform[6]
        ) {
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<std::pair<int, int>>> good_matches(num_cams);

    ceres::Problem problem;

    std::vector<std::vector<ResidualType>> residual_type(num_cams);

    for(int cam = 0; cam<num_cams; cam++) {
        //double start = clock()/double(CLOCKS_PER_SEC);
        std::vector<cv::DMatch> matches;
        matcher.match(
                descriptors[cam][frame1],
                descriptors[cam][frame2], matches);
        /*
        cv::Ptr<cv::cuda::DescriptorMatcher> d_matcher = 
            cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        const cv::cuda::GpuMat d_query(descriptors[cam][frame1]);
        const cv::cuda::GpuMat d_train(descriptors[cam][frame2]);
        cv::cuda::GpuMat d_matches;
        d_matcher->matchAsync(d_query, d_train, d_matches);

        d_matcher->matchConvert(d_matches, matches[cam]);
        */

        /*
        double end = clock()/double(CLOCKS_PER_SEC);
        std::cerr << "Matching: " << descriptors[cam][frame1].size()
            << ", " << descriptors[cam][frame2].size() 
            << "; " << end-start << std::endl;
            */
        auto &mc = matches;
        // find minimum matching distance and filter out the ones more than twice as big as it
        double min_dist = 1e9, max_dist = 0;
        for(int i=0; i<mc.size(); i++) {
            if(mc[i].distance < min_dist) min_dist = mc[i].distance;
            if(mc[i].distance > max_dist) max_dist = mc[i].distance;
        }
        //std::cerr << "Matches cam " << cam << ": " <<  mc.size() 
            //<< " " << min_dist << " " << max_dist
            //<< std::endl;
        for(int i=0; i<mc.size(); i++) {
            if(mc[i].distance > std::max(2*min_dist, match_thresh)) continue;
            int point1 = mc[i].queryIdx,
                point2 = mc[i].trainIdx;
            bool block_created = false;
            if(has_depth[cam][frame1][point1] != -1
                    && has_depth[cam][frame2][point2] != -1) {
                // 3D 3D
                const pcl::PointXYZ pointM =
                    keypoints_with_depth[cam][frame1]
                        ->at(has_depth[cam][frame1][point1]);
                const pcl::PointXYZ pointS =
                    keypoints_with_depth[cam][frame2]
                        ->at(has_depth[cam][frame2][point2]);
                auto pointMS = pointM;
                util::subtract_assign(pointMS, pointS);
                if(util::norm2(pointMS) > 10) continue;
                //std::cerr << "  3D 3D " << cam << ": " << pointM<< ", " << pointS << std::endl;
                ceres::CostFunction* cost_function = 
                    new ceres::AutoDiffCostFunction<cost3D3D,3,6>(
                            new cost3D3D(
                                pointM.x,
                                pointM.y,
                                pointM.z,
                                pointS.x,
                                pointS.y,
                                pointS.z
                                )
                            );
                problem.AddResidualBlock(
                        cost_function,
                        new ceres::CauchyLoss(loss_thresh_3D3D),
                        transform);

                residual_type[cam].push_back(RESIDUAL_3D3D);
                block_created = true;
            } else if(has_depth[cam][frame1][point1] == -1
                    && has_depth[cam][frame2][point2] == -1) {
                // 2D 2D
                if(cam == 0) {
                    const auto pointM = keypoints[cam][frame1][point1];
                    const auto pointS = keypoints[cam][frame2][point2];
                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<cost2D2D,1,6>(
                                new cost2D2D(
                                    pointM.x,
                                    pointM.y,
                                    pointS.x,
                                    pointS.y
                                    )
                                );
                    problem.AddResidualBlock(
                            cost_function,
                            new ceres::ScaledLoss(
                                new ceres::ArctanLoss(loss_thresh_2D2D),
                                weight_2D2D,
                                ceres::TAKE_OWNERSHIP),
                            transform);
                    residual_type[cam].push_back(RESIDUAL_2D2D);
                    block_created = true;
                }
            } else if(has_depth[cam][frame1][point1] != -1
                    && has_depth[cam][frame2][point2] == -1) {
                // 3D 2D
                // std::cerr << "  3D 2D " << point1 << ", " << point2 << std::endl;
                const pcl::PointXYZ point3D = keypoints_with_depth[cam][frame1]
                    ->at(has_depth[cam][frame1][point1]);
                const auto point2D = keypoints[cam][frame2][point2];

                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<cost3D2D,2,6>(
                            new cost3D2D(
                                point3D.x,
                                point3D.y,
                                point3D.z,
                                point2D.x,
                                point2D.y,
                                cam_trans[cam](0),
                                cam_trans[cam](1),
                                cam_trans[cam](2)
                                )
                            );
                problem.AddResidualBlock(
                        cost_function,
                        new ceres::ScaledLoss(
                            new ceres::ArctanLoss(loss_thresh_3D2D), 
                            weight_3D2D,
                            ceres::TAKE_OWNERSHIP),
                        transform);

                residual_type[cam].push_back(RESIDUAL_3D2D);
                block_created = true;
            } else if(has_depth[cam][frame1][point1] == -1
                    && has_depth[cam][frame2][point2] != -1) {
                // 2D 3D
                //std::cerr << "  2D 3D " << point1 << ", " << point2 << std::endl;
                const pcl::PointXYZ point3D = keypoints_with_depth[cam][frame2]
                    ->at(has_depth[cam][frame2][point2]);
                const auto point2D = keypoints[cam][frame1][point1];
                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<cost2D3D,2,6>(
                            new cost2D3D(
                                point3D.x,
                                point3D.y,
                                point3D.z,
                                point2D.x,
                                point2D.y,
                                cam_trans[cam](0),
                                cam_trans[cam](1),
                                cam_trans[cam](2)
                                )
                            );
                problem.AddResidualBlock(
                        cost_function,
                        new ceres::ScaledLoss(
                            new ceres::ArctanLoss(loss_thresh_3D2D), 
                            weight_3D2D,
                            ceres::TAKE_OWNERSHIP),
                        transform);

                residual_type[cam].push_back(RESIDUAL_2D3D);
                block_created = true;
            }
            if(block_created)
                good_matches[cam].push_back(std::make_pair(point1, point2));
        }
    }
    //residualStats(problem, good_matches, residual_type);
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //residualStats(problem, good_matches, residual_type);

    /*
    for(int i=0; i<6; i++) {
        std::cerr << transform[i] << " ";
    }
    std::cerr << std::endl;
    */
    return util::pose_mat2vec(transform);
}

void residualStats(
        ceres::Problem &problem,
        std::vector<std::vector<std::pair<int, int>>> &good_matches,
        std::vector<std::vector<ResidualType>> &residual_type
        ) {
    // compute some statistics about residuals
    double cost;
    std::vector<double> residuals;
    ceres::Problem::EvaluateOptions evaluate_options;
    evaluate_options.apply_loss_function = false;
    problem.Evaluate(evaluate_options, &cost, &residuals, NULL, NULL);
    std::vector<double> residuals_3D3D, residuals_3D2D, residuals_2D3D, residuals_2D2D;
    int ri = 0;
    for(int cam = 0; cam < num_cams; cam++) {
        auto &mc = good_matches[cam];
        for(int i=0; i<mc.size(); i++) {
            switch(residual_type[cam][i]) {
                case RESIDUAL_3D3D:
                    residuals_3D3D.push_back(
                            sqrt(
                                residuals[ri]*residuals[ri] + 
                                residuals[ri+1]*residuals[ri+1] + 
                                residuals[ri+2]*residuals[ri+2])
                            );
                    ri += 3;
                    break;
                case RESIDUAL_3D2D:
                    residuals_3D2D.push_back(
                            sqrt(
                                residuals[ri]*residuals[ri] + 
                                residuals[ri+1]*residuals[ri+1]
                                )
                            );
                    ri += 2;
                    break;
                case RESIDUAL_2D3D:
                    residuals_2D3D.push_back(
                            sqrt(
                                residuals[ri]*residuals[ri] + 
                                residuals[ri+1]*residuals[ri+1]
                                )
                            );
                    ri += 2;
                    break;
                case RESIDUAL_2D2D:
                    residuals_2D2D.push_back(abs(residuals[ri]));
                    ri++;
                    break;
                default:
                    break;
            }
        }
    }
    double sum_3D3D = 0, sum_3D2D = 0, sum_2D3D = 0, sum_2D2D = 0;
    for(auto r : residuals_3D3D) {sum_3D3D += r;}
    for(auto r : residuals_3D2D) {sum_3D2D += r;}
    for(auto r : residuals_2D3D) {sum_2D3D += r;}
    for(auto r : residuals_2D2D) {sum_2D2D += r;}
    std::sort(residuals_3D3D.begin(), residuals_3D3D.end());
    std::sort(residuals_3D2D.begin(), residuals_3D2D.end());
    std::sort(residuals_2D3D.begin(), residuals_2D3D.end());
    std::sort(residuals_2D2D.begin(), residuals_2D2D.end());
    std::cerr << "Cost: " << cost
        << " Residual blocks: " << problem.NumResidualBlocks()
        << " Residuals: " << problem.NumResiduals() << " " << ri << " " << residuals.size()
        << std::endl
        << " Total good matches: " 
        << " " << good_matches[0].size()
        << " " << good_matches[1].size()
        << " " << good_matches[2].size()
        << " " << good_matches[3].size()
        << " " << residual_type[0].size()
        << " " << residual_type[1].size()
        << " " << residual_type[2].size()
        << " " << residual_type[3].size()
        << std::endl;
    if(residuals_3D3D.size())
    std::cerr << "Residual 3D3D:"
        << " median " << std::fixed << std::setprecision(10) << residuals_3D3D[residuals_3D3D.size()/2]
        << " mean " << std::fixed << std::setprecision(10) << sum_3D3D/residuals_3D3D.size() 
        << " count " << residuals_3D3D.size() << std::endl;
    if(residuals_3D2D.size())
    std::cerr << "Residual 3D2D:"
        << " median " << std::fixed << std::setprecision(10) << residuals_3D2D[residuals_3D2D.size()/2]
        << " mean " << std::fixed << std::setprecision(10) << sum_3D2D/residuals_3D2D.size() 
        << " count " << residuals_3D2D.size() << std::endl;
    if(residuals_2D3D.size())
    std::cerr << "Residual 2D3D:"
        << " median " << std::fixed << std::setprecision(10) << residuals_2D3D[residuals_2D3D.size()/2]
        << " mean " << std::fixed << std::setprecision(10) << sum_2D3D/residuals_2D3D.size() 
        << " count " << residuals_2D3D.size() << std::endl;
    if(residuals_2D2D.size())
    std::cerr << "Residual 2D2D:"
        << " median " << std::fixed << std::setprecision(10) << residuals_2D2D[residuals_2D2D.size()/2]
        << " mean " << std::fixed << std::setprecision(10) << sum_2D2D/residuals_2D2D.size() 
        << " count " << residuals_2D2D.size() << std::endl;
}
