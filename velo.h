#pragma once

enum ResidualType {
    RESIDUAL_3D3D,
    RESIDUAL_3D2D,
    RESIDUAL_2D3D,
    RESIDUAL_2D2D
};

cv::Point2f pixel2canonical(
        const cv::Point2f &pp,
        const Eigen::Matrix3f &Kinv) {
    Eigen::Vector3f p;
    p << pp.x, pp.y, 1;
    p = Kinv * p;
    return cv::Point2f(p(0)/p(2), p(1)/p(2));
}

cv::Point2f canonical2pixel(
        const cv::Point2f &pp,
        const Eigen::Matrix3f &K) {
    Eigen::Vector3f p;
    p << pp.x, pp.y, 1;
    p = K * p;
    return cv::Point2f(p(0)/p(2), p(1)/p(2));
}

void trackFeatures(
        std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints,
        std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints_p,
        std::vector<std::vector<std::vector<int>>> &keypoint_ids,
        std::vector<std::vector<cv::Mat>> &descriptors,
        const cv::Mat &img1,
        const cv::Mat &img2,
        const int cam1,
        const int cam2,
        const int frame1,
        const int frame2
        ) {
    const Eigen::Matrix3f &Kinv1 = cam_intrinsic_inv[cam1];
    const Eigen::Matrix3f &Kinv2 = cam_intrinsic_inv[cam2];

    int m = keypoints[cam1][frame1].size();
    if(m == 0) {
        std::cerr << "ERROR: No features to track." << std::endl;
    }
    std::vector<cv::Point2f> points1(m), points2(m);
    for(int i=0; i<m; i++) {
        points1[i] = keypoints_p[cam1][frame1][i];
    }
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
            img1,
            img2,
            points1,
            points2,
            status,
            err,
            cv::Size(lkt_window, lkt_window),
            lkt_pyramid,
            cv::TermCriteria(
                CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                30,
                0.01
                ),
            0
            );
    int col_cells = img_width / min_distance + 2,
        row_cells = img_height / min_distance + 2;
    std::vector<std::vector<cv::Point2f>> occupied(col_cells * row_cells);
    for(int i=0; i<m; i++) {
        if(!status[i]) {
            continue;
        }
        if(util::dist2(points1[i], points2[i]) > flow_outlier) {
            continue;
        }
        // somehow points can be tracked to negative x and y
        if(points2[i].x < 0 || points2[i].y < 0 ||
                points2[i].x >= img_width ||
                points2[i].y >= img_height) {
            continue;
        }
        /*
        int col = points2[i].x / min_distance,
            row = points2[i].y / min_distance;
        bool bad = false;
        int col_start = std::max(col-1, 0),
            col_end = std::min(col+1, col_cells-1),
            row_start = std::max(row-1, 0),
            row_end = std::min(row+1, row_cells-1);
        float md2 = min_distance * min_distance;
        for(int c=col_start; c<=col_end && !bad; c++) {
            for(int r=row_start; r<=row_end && !bad; r++) {
                for(auto pp : occupied[c * row_cells + r]) {
                    if(util::dist2(pp, points2[i]) < md2) {
                        bad = true;
                        break;
                    }
                }
            }
        }
        if(bad) continue;
        occupied[col * row_cells + row].push_back(points2[i]);
        */
        keypoints_p[cam2][frame2].push_back(points2[i]);
        keypoints[cam2][frame2].push_back(
                pixel2canonical(points2[i], Kinv2)
                );
        keypoint_ids[cam2][frame2].push_back(
                keypoint_ids[cam1][frame1][i]);
        descriptors[cam2][frame2].push_back(
                descriptors[cam1][frame1].row(i).clone());
    }
}

void detectFeatures(
        std::vector<std::vector<cv::Point2f>> &keypoints,
        std::vector<std::vector<cv::Point2f>> &keypoints_p,
        std::vector<std::vector<int>> &keypoint_ids,
        std::vector<cv::Mat> &descriptors,
        const cv::Ptr<cv::FeatureDetector> detector,
        const cv::Ptr<cv::DescriptorExtractor> extractor,
        const cv::Mat &img,
        int &id_counter,
        const int cam,
        const int frame
        ) {
    const Eigen::Matrix3f &Kinv = cam_intrinsic_inv[cam];

    int col_cells = img_width / min_distance + 2,
        row_cells = img_height / min_distance + 2;
    std::vector<std::vector<cv::Point2f>> occupied(col_cells * row_cells);
    for(cv::Point2f p : keypoints_p[frame]) {
        int col = p.x / min_distance,
            row = p.y / min_distance;
        occupied[col * row_cells + row].push_back(p);
    }
    std::vector<cv::KeyPoint> cvKP;
    detector->detect(img, cvKP);
    cv::Mat tmp_descriptors;
    // remember! compute MUTATES cvKP
    extractor->compute(img, cvKP, tmp_descriptors);
    int detected = 0;
    for(int kp_i = 0; kp_i < cvKP.size(); kp_i ++) {
        auto kp = cvKP[kp_i];
        //std::cerr << kp.size << " " << kp.angle << " " << kp.response << std::endl;
        int col = kp.pt.x / min_distance,
            row = kp.pt.y / min_distance;
        bool bad = false;
        int col_start = std::max(col-1, 0),
            col_end = std::min(col+1, col_cells-1),
            row_start = std::max(row-1, 0),
            row_end = std::min(row+1, row_cells-1);
        float md2 = min_distance * min_distance;
        for(int c=col_start; c<=col_end && !bad; c++) {
            for(int r=row_start; r<=row_end && !bad; r++) {
                for(auto pp : occupied[c * row_cells + r]) {
                    if(util::dist2(pp, kp.pt) < md2) {
                        bad = true;
                        break;
                    }
                }
            }
        }
        if(bad) continue;
        keypoints_p[frame].push_back(kp.pt);
        keypoints[frame].push_back(
                pixel2canonical(kp.pt, Kinv)
                );
        descriptors[frame].push_back(tmp_descriptors.row(kp_i).clone());
        keypoint_ids[frame].push_back(id_counter++);
        detected++;
    }
    //std::cerr << "Detected: " << detected << std::endl;
}

void consolidateFeatures(
        std::vector<cv::Point2f> &keypoints,
        std::vector<cv::Point2f> &keypoints_p,
        std::vector<int> &keypoint_ids,
        cv::Mat &descriptors,
        const int cam
        ) {
    // merges keypoints of the same id using the geometric median
    // geometric median is computed in canonical coordinates
    const Eigen::Matrix3f &K = cam_intrinsic[cam];
    int m = keypoint_ids.size();
    std::map<int, std::vector<int>> keypoints_map;
    for(int i=0; i<m; i++) {
        keypoints_map[keypoint_ids[i]].push_back(i);
    }
    int mm = keypoints_map.size();
    std::vector<cv::Point2f> tmp_keypoints(mm);
    std::vector<cv::Point2f> tmp_keypoints_p(mm);
    std::vector<int> tmp_keypoint_ids(mm);
    cv::Mat tmp_descriptors(mm, descriptors.cols, descriptors.type());
    int mi = 0;
    for(auto kp : keypoints_map) {
        int id = kp.first;
        int n = kp.second.size();

        cv::Point2f gm_keypoint;
        if(n > 2) {
            std::vector<cv::Point2f> tmp_tmp_keypoints(n);
            for(int i=0; i<n; i++) {
                int j = kp.second[i];
                tmp_tmp_keypoints[i] = keypoints[j];
            }
            gm_keypoint = util::geomedian(tmp_tmp_keypoints);
        } else if(n ==2) {
            gm_keypoint = (
                    keypoints[kp.second[0]] + 
                    keypoints[kp.second[1]])/2;
        } else {
            gm_keypoint = keypoints[kp.second[0]];
        }
        tmp_keypoint_ids[mi] = id;
        tmp_keypoints[mi] = gm_keypoint;
        tmp_keypoints_p[mi] = canonical2pixel(gm_keypoint, K);
        descriptors.row(kp.second[0]).copyTo(tmp_descriptors.row(mi));
        mi++;
    }

    keypoints = tmp_keypoints;
    keypoints_p = tmp_keypoints_p;
    keypoint_ids = tmp_keypoint_ids;
    tmp_descriptors.copyTo(descriptors);
}

void removeTerribleFeatures(
        std::vector<cv::Point2f> &keypoints,
        std::vector<cv::Point2f> &keypoints_p,
        std::vector<int> &keypoint_ids,
        cv::Mat &descriptors,
        const cv::Ptr<cv::DescriptorExtractor> extractor,
        const cv::Mat &img,
        const int cam
        ) {
    // remove features if the extracted descriptor doesn't match
    std::vector<cv::KeyPoint> cvKP(keypoints_p.size());
    for(int i=0; i<keypoints_p.size(); i++) {
        cvKP[i].pt = keypoints_p[i];
    }
    std::vector<cv::Point2f> tmp_keypoints;
    std::vector<cv::Point2f> tmp_keypoints_p;
    std::vector<int> tmp_keypoint_ids;
    cv::Mat tmp_descriptors, tmp_tmp_descriptors;

    int i=0;
    extractor->compute(img, cvKP, tmp_tmp_descriptors);
    for(int j=0; j<cvKP.size(); j++) {
        while(cv::norm(keypoints_p[i] - cvKP[j].pt) > kp_EPS) {
            i++;
        }
        if(cv::norm(descriptors.row(i),
                    tmp_tmp_descriptors.row(j),
                    cv::NORM_HAMMING) < match_thresh) {
            tmp_keypoints.push_back(keypoints[i]);
            tmp_keypoints_p.push_back(keypoints_p[i]);
            tmp_keypoint_ids.push_back(keypoint_ids[i]);
            tmp_descriptors.push_back(descriptors.row(i).clone());
        }
    }
    keypoints = tmp_keypoints;
    keypoints_p = tmp_keypoints_p;
    keypoint_ids = tmp_keypoint_ids;
    tmp_descriptors.copyTo(descriptors);
}

void removeSlightlyLessTerribleFeatures(
        std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints,
        std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints_p,
        std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> &kp_with_depth,
        std::vector<std::vector<std::vector<int>>> &keypoint_ids,
        std::vector<std::vector<cv::Mat>> &descriptors,
        std::vector<std::vector<std::vector<int>>> &has_depth,
        const int frame,
        const std::vector<std::vector<std::pair<int, int>>> &good_matches) {
    // remove features not matched in good_matches
    for(int cam=0; cam<num_cams; cam++) {
        // I'm not smart enough to figure how to delete things
        // other than making a new vector, pointcloud, or mat
        // and then copying everything over :(
        // In Python I could have just written a = a[indices]
        std::set<int> good_indices;
        for(auto gm : good_matches[cam]) {
            good_indices.insert(gm.first);
        }
        int m = good_indices.size(), n = keypoints[cam][frame].size();
        std::cerr << "Good matches of " << cam << ": " << m << "/" << n << std::endl;
        std::vector<cv::Point2f> tmp_keypoints(m);
        std::vector<cv::Point2f> tmp_keypoints_p(m);
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_kp_with_depth(
                new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> tmp_keypoint_ids(m);
        cv::Mat tmp_descriptors(
                m, descriptors[cam][frame].cols,
                descriptors[cam][frame].type());
        std::vector<int> tmp_has_depth(m);
        int j = 0, jd = 0;
        for(int i=0; i<n; i++) {
            if(!good_indices.count(i)) continue;
            tmp_keypoints[j] = keypoints[cam][frame][i];
            tmp_keypoints_p[j] = keypoints_p[cam][frame][i];
            tmp_keypoint_ids[j] = keypoint_ids[cam][frame][i];
            descriptors[cam][frame].row(i).copyTo(tmp_descriptors.row(j));
            int d = has_depth[cam][frame][i];
            if(d != -1) {
                tmp_kp_with_depth->push_back(kp_with_depth[cam][frame]->at(d));
                tmp_has_depth[j] = jd++;
            } else {
                tmp_has_depth[j] = -1;
            }
            j++;
        }
        // these are all allocated on the stack or smart pointers
        // so no memory leaks, hopefully
        keypoints[cam][frame] = tmp_keypoints;
        keypoints_p[cam][frame] = tmp_keypoints_p;
        kp_with_depth[cam][frame] = tmp_kp_with_depth;
        keypoint_ids[cam][frame] = tmp_keypoint_ids;
        tmp_descriptors.copyTo(descriptors[cam][frame]);
        has_depth[cam][frame] = tmp_has_depth;
    }
}

void projectLidarToCamera(
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans,
        std::vector<std::vector<cv::Point2f>> &projection,
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans_valid,
        const int cam
        ) {

    int bad = 0;
    Eigen::Vector3f t = cam_trans[cam];

    for(int s=0; s<scans.size(); s++) {
        pcl::PointCloud<pcl::PointXYZ> projected_points;
        projection.push_back(std::vector<cv::Point2f>());
        scans_valid.push_back(pcl::PointCloud<pcl::PointXYZ>::Ptr(
                    new pcl::PointCloud<pcl::PointXYZ>));
        for(int i=0, _i = scans[s]->size(); i<_i; i++) {
            pcl::PointXYZ p = scans[s]->at(i);
            pcl::PointXYZ pp(p.x + t(0), p.y + t(1), p.z + t(2));
            cv::Point2f c(pp.x/pp.z, pp.y/pp.z);
            if(pp.z > 0 && c.x >= min_x[cam] && c.x < max_x[cam]
                    && c.y >= min_y[cam] && c.y < max_y[cam]) {
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
        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_with_depth,
        std::vector<int> &has_depth
        ) {
    has_depth.resize(keypoints.size());
    for(int i=0; i<has_depth.size(); i++) {
        has_depth[i] = -1;
    }
    /*
    std::cerr << "Sizes: " <<  scans.size() << " "
        << projection.size() << " "
        << keypoints.size() << std::endl;
        */
    int has_depth_n = 0;
    for(int k=0; k<keypoints.size(); k++) {
        has_depth[k] = -1;
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
                         * Perform linear interpolation using four points:
                         * s-1, last_interp ----- interp2 ----- s, last_interp+1
                         *                           |
                         *                           kp
                         *                           |
                         *      s, mid ---------- interp1 --------- s, mid+1
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

void matchFeatures(
        const std::vector<std::vector<cv::Mat>> &descriptors,
        const int cam1,
        const int cam2,
        const int frame1,
        const int frame2,
        std::vector<std::pair<int, int>> &matches
        ) {
    double start = clock()/double(CLOCKS_PER_SEC);
    /*
    std::cerr << "Matching: ";
    std::cerr << descriptors[cam1].size()
        << ", " << descriptors[cam2].size()
        << ", " << frame1 << " " << frame2 << "; ";
    std::cerr << descriptors[cam1][frame1].size()
        << ", " << descriptors[cam2][frame2].size();
        */
    std::vector<cv::DMatch> mc;
#ifdef USE_CUDA
    cv::Ptr<cv::cuda::DescriptorMatcher> d_matcher =
        cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    cv::cuda::GpuMat d_query(descriptors[cam1][frame1]);
    cv::cuda::GpuMat d_train(descriptors[cam2][frame2]);
    cv::cuda::GpuMat d_matches;
    d_matcher->matchAsync(d_query, d_train, d_matches);

    d_matcher->matchConvert(d_matches, mc);
#else
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(
            descriptors[cam1][frame1],
            descriptors[cam2][frame2], mc);
#endif

    double end = clock()/double(CLOCKS_PER_SEC);
    //std::cerr << "; " << end-start << std::endl;
    // find minimum matching distance and filter out the ones more than twice as big as it
    double min_dist = 1e9, max_dist = 0;
    for(int i=0; i<mc.size(); i++) {
        if(mc[i].distance < min_dist) min_dist = mc[i].distance;
        if(mc[i].distance > max_dist) max_dist = mc[i].distance;
    }
    /*
    std::cerr << "Matches cam " << cam1 << ": " <<  mc.size()
        << " " << min_dist << " " << max_dist
        << std::endl;
        */
    for(int i=0; i<mc.size(); i++) {
        if(mc[i].distance > std::max(1.5*min_dist, match_thresh)) continue;
        matches.push_back(std::make_pair(mc[i].queryIdx, mc[i].trainIdx));
    }
}
void matchFeatures(
        const std::vector<std::vector<cv::Mat>> &descriptors,
        const int frame1,
        const int frame2,
        std::vector<std::vector<std::pair<int, int>>> &matches
        ) {
    for(int cam=0; cam<num_cams; cam++) {
        matchFeatures(descriptors, cam, cam, frame1, frame2, matches[cam]);
    }
}

void matchUsingId(
        const std::vector<std::vector<std::vector<int>>> &keypoint_ids,
        const int cam1,
        const int cam2,
        const int frame1,
        const int frame2,
        std::vector<std::pair<int, int>> &matches
        ) {
    std::map<int, int> id2ind;
    for(int ind = 0; ind < keypoint_ids[cam1][frame1].size(); ind++) {
        int id = keypoint_ids[cam1][frame1][ind];
        id2ind[id] = ind;
    }
    for(int ind = 0; ind < keypoint_ids[cam2][frame2].size(); ind++) {
        int id = keypoint_ids[cam2][frame2][ind];
        if(id2ind.count(id))
            matches.push_back(std::make_pair(id2ind[id], ind));
    }
}
void matchUsingId(
        const std::vector<std::vector<std::vector<int>>> &keypoint_ids,
        const int frame1,
        const int frame2,
        std::vector<std::vector<std::pair<int, int>>> &matches
        ) {
    for(int cam=0; cam<num_cams; cam++) {
        matchUsingId(keypoint_ids, cam, cam, frame1, frame2, matches[cam]);
    }
}

void residualStats(
        ceres::Problem &problem,
        const std::vector<std::vector<std::pair<int, int>>> &good_matches,
        const std::vector<std::vector<ResidualType>> &residual_type
        );

Eigen::Matrix4d frameToFrame(
        const std::vector<std::vector<std::pair<int, int>>> &matches,
        const std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints,
        const std::vector<std::vector<std::vector<int>>> &keypoint_ids,
        const std::map<int, pcl::PointXYZ> &landmarks_at_frame,
        const std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> &keypoints_with_depth,
        const std::vector<std::vector<std::vector<int>>> &has_depth,
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans_M,
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans_S,
        const std::vector<pcl::KdTreeFLANN<pcl::PointXYZ>> &kd_trees,
        const int frame1,
        const int frame2,
        double transform[6],
        std::vector<std::vector<std::pair<int, int>>> &good_matches,
        std::vector<std::vector<ResidualType>> &residual_type,
        const bool enable_icp
        ) {

    for(int iter = 1; iter <= f2f_iterations; iter++) {
        ceres::Problem::Options problem_options;
        problem_options.enable_fast_removal = true;
        ceres::Problem problem(problem_options);

        // Visual odometry
        for(int cam = 0; cam<num_cams; cam++) {
            //std::cerr << "Matches: " << matches[cam].size() << std::endl;
            good_matches[cam].clear();
            residual_type[cam].clear();
            const std::vector<std::pair<int, int>> &mc = matches[cam];
            for(int i=0; i<mc.size(); i++) {
                int point1 = mc[i].first,
                    point2 = mc[i].second;
                int id = keypoint_ids[cam][frame2][point2];
                bool d1 = has_depth[cam][frame1][point1] != -1,
                     d2 = has_depth[cam][frame2][point2] != -1;
                pcl::PointXYZ point3_2, point3_1;
                if(landmarks_at_frame.count(id)) {
                    point3_2 = landmarks_at_frame.at(id);
                    /*
                    if(d2) {
                        std::cerr << "Using landmark "
                            << id << ": " << point3_2 
                            << " " << keypoints_with_depth[cam][frame2]
                            ->at(has_depth[cam][frame2][point2]) << std::endl;
                    }
                    */
                    d2 = true;
                } else if(d2) {
                    point3_2 = keypoints_with_depth[cam][frame2]
                        ->at(has_depth[cam][frame2][point2]);
                }
                if(d1) {
                    point3_1 = keypoints_with_depth[cam][frame1]
                        ->at(has_depth[cam][frame1][point1]);
                }
                cv::Point2f point2_1 = keypoints[cam][frame1][point1];
                cv::Point2f point2_2 = keypoints[cam][frame2][point2];
                //std::cerr << "has depth: " << has_depth[cam][frame1].size();

                //std::cerr << " " << has_depth[cam][frame1][point1]
                //    << " " << keypoints_with_depth[cam][frame1]->size();
                //std::cerr << " " << has_depth[cam][frame2][point2]
                //    << " " << keypoints_with_depth[cam][frame2]->size();
                //std::cerr << std::endl;
                if(d1 && d2) {
                    // 3D 3D
                    cost3D3D *cost = new cost3D3D(
                            point3_1.x,
                            point3_1.y,
                            point3_1.z,
                            point3_2.x,
                            point3_2.y,
                            point3_2.z
                            );
                    double residual_test[3];
                    (*cost)(transform, residual_test);
                    if(iter > 1 &&
                            residual_test[0] * residual_test[0] +
                            residual_test[1] * residual_test[1] +
                            residual_test[2] * residual_test[2]
                            >
                            loss_thresh_3D3D*outlier_reject/iter *
                            loss_thresh_3D3D*outlier_reject/iter) {
                        continue;
                    }
                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<cost3D3D,3,6>(
                                cost);
                    problem.AddResidualBlock(
                            cost_function,
                            new ceres::ArctanLoss(loss_thresh_3D3D),
                            transform);

                    residual_type[cam].push_back(RESIDUAL_3D3D);
                    good_matches[cam].push_back(std::make_pair(point1, point2));
                }
                if(!d1 && !d2) {
                    // 2D 2D
#ifdef ENABLE_2D2D
                    cost2D2D *cost =
                        new cost2D2D(
                                point2_1.x,
                                point2_1.y,
                                point2_2.x,
                                point2_2.y,
                                cam_trans[cam](0),
                                cam_trans[cam](1),
                                cam_trans[cam](2)
                                );
                    double residual_test[1];
                    (*cost)(transform, residual_test);
                    if(iter > 1 && abs(residual_test[0]) > loss_thresh_2D2D*outlier_reject/iter) continue;
                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<cost2D2D,1,6>(cost);
                    problem.AddResidualBlock(
                            cost_function,
                            new ceres::ScaledLoss(
                                new ceres::ArctanLoss(loss_thresh_2D2D),
                                weight_2D2D,
                                ceres::TAKE_OWNERSHIP),
                            transform);
                    residual_type[cam].push_back(RESIDUAL_2D2D);
                    good_matches[cam].push_back(std::make_pair(point1, point2));
#endif
                }
#ifdef ENABLE_3D2D
                if(d1) {
                    // 3D 2D
                    cost3D2D *cost =
                        new cost3D2D(
                                point3_1.x,
                                point3_1.y,
                                point3_1.z,
                                point2_2.x,
                                point2_2.y,
                                cam_trans[cam](0),
                                cam_trans[cam](1),
                                cam_trans[cam](2)
                                );
                    double residual_test[2];
                    (*cost)(transform, residual_test);
                    if(iter > 1 && residual_test[0] * residual_test[0] +
                            residual_test[1] * residual_test[1]
                            > loss_thresh_3D2D*outlier_reject/iter
                            * loss_thresh_3D2D*outlier_reject/iter) continue;

                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<cost3D2D,2,6>(cost);
                    problem.AddResidualBlock(
                            cost_function,
                            new ceres::ScaledLoss(
                                new ceres::ArctanLoss(loss_thresh_3D2D),
                                weight_3D2D,
                                ceres::TAKE_OWNERSHIP),
                            transform);

                    residual_type[cam].push_back(RESIDUAL_3D2D);
                    good_matches[cam].push_back(std::make_pair(point1, point2));
                }
                if(d2) {
                    // 2D 3D
                    cost2D3D *cost =
                        new cost2D3D(
                                point3_2.x,
                                point3_2.y,
                                point3_2.z,
                                point2_1.x,
                                point2_1.y,
                                cam_trans[cam](0),
                                cam_trans[cam](1),
                                cam_trans[cam](2)
                                );
                    double residual_test[2];
                    (*cost)(transform, residual_test);
                    if(iter > 1 && residual_test[0] * residual_test[0] +
                            residual_test[1] * residual_test[1]
                            > loss_thresh_3D2D*outlier_reject/iter
                            * loss_thresh_3D2D*outlier_reject/iter) continue;

                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<cost2D3D,2,6>(cost);
                    problem.AddResidualBlock(
                            cost_function,
                            new ceres::ScaledLoss(
                                new ceres::ArctanLoss(loss_thresh_3D2D),
                                weight_3D2D,
                                ceres::TAKE_OWNERSHIP),
                            transform);

                    residual_type[cam].push_back(RESIDUAL_2D3D);
                    good_matches[cam].push_back(std::make_pair(point1, point2));
                }
#endif
            }
        }

        // Point set registration
#ifdef ENABLE_ICP
        std::vector<ceres::ResidualBlockId> icp_blocks;

        //std::cerr << "M: " << scans_M.size() << " S: " << scans_S.size() << kd_trees.size() << std::endl;

        for(int icp_iter = 0; icp_iter < icp_iterations; icp_iter++) {
            while(icp_blocks.size() > 0) {
                auto bid = icp_blocks.back();
                icp_blocks.pop_back();
                problem.RemoveResidualBlock(bid);
            }
            for(int sm = 0; sm < scans_M.size() * enable_icp; sm++) {
                for(int smi = 0; smi < scans_M[sm]->size(); smi+= icp_skip) {
                    pcl::PointXYZ pointM = scans_M[sm]->at(smi);
                    pcl::PointXYZ pointM_untransformed = pointM;
                    util::transform_point(pointM, transform);
                    /*
                     * Point-to-plane ICP where plane is defined by
                     * three Nearest Points (np):
                     *            np_i     np_k
                     * np_s_i ..... * ..... * .....
                     *               \     /
                     *                \   /
                     *                 \ /
                     * np_s_j ......... * .......
                     *                 np_j
                     */
                    int np_i = 0, np_j = 0, np_k = 0;
                    int np_s_i = -1, np_s_j = -1;
                    double np_dist_i = INF, np_dist_j = INF;
                    for(int ss = 0; ss < kd_trees.size(); ss++) {
                        std::vector<int> id(1);
                        std::vector<float> dist2(1);
                        if(kd_trees[ss].nearestKSearch(pointM, 1, id, dist2) <= 0 ||
                                dist2[0] > correspondence_thresh_icp/iter/iter/iter/iter) {
                            continue;
                        }
                        pcl::PointXYZ np = scans_S[ss]->at(id[0]);

                        util::subtract_assign(np, pointM);
                        double d = util::norm2(np);
                        if(d < np_dist_i) {
                            np_dist_j = np_dist_i;
                            np_j = np_i;
                            np_s_j = np_s_i;
                            np_dist_i = d;
                            np_i = id[0];
                            np_s_i = ss;
                        } else if(d < np_dist_j) {
                            np_dist_j = d;
                            np_j = id[0];
                            np_s_j = ss;
                        }
                    }
                    if(np_s_i == -1 || np_s_j == -1) {
                        continue;
                    }
                    int np_k_n = scans_S[np_s_i]->size(),
                        np_k_1p = (np_i+1) % np_k_n,
                        np_k_2p = (np_i-1 + np_k_n) % np_k_n;
                    pcl::PointXYZ np_k_1 = scans_S[np_s_i]->at(np_k_1p),
                        np_k_2 = scans_S[np_s_i]->at(np_k_2p);
                    util::subtract_assign(np_k_1, pointM);
                    util::subtract_assign(np_k_2, pointM);
                    if(util::norm2(np_k_1) < util::norm2(np_k_2)) {
                        np_k = np_k_1p;
                    } else {
                        np_k = np_k_2p;
                    }
                    pcl::PointXYZ s0, s1, s2;
                    s0 = scans_S[np_s_i]->at(np_i);
                    s1 = scans_S[np_s_j]->at(np_j);
                    s2 = scans_S[np_s_i]->at(np_k);
                    Eigen::Vector3f
                        v0 = s0.getVector3fMap(),
                           v1 = s1.getVector3fMap(),
                           v2 = s2.getVector3fMap();
                    Eigen::Vector3f N = (v1 - v0).cross(v2 - v0);
                    if(N.norm() < icp_norm_condition) continue;
                    N /= N.norm();
                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<cost3DPD, 1, 6>(
                                new cost3DPD(
                                    pointM_untransformed.x,
                                    pointM_untransformed.y,
                                    pointM_untransformed.z,
                                    N[0], N[1], N[2],
                                    v0[0], v0[1], v0[2]
                                    )
                                );
                    auto bid = problem.AddResidualBlock(
                            cost_function,
                            new ceres::ScaledLoss(
                                new ceres::CauchyLoss(loss_thresh_3DPD),
                                weight_3DPD,
                                ceres::TAKE_OWNERSHIP),
                            transform);
                    icp_blocks.push_back(bid);
                }
            }
#endif
            //residualStats(problem, good_matches, residual_type);
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.minimizer_progress_to_stdout = false;
            //options.num_threads = 8;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            if(f2f_iterations - iter == 0) {
                //residualStats(problem, good_matches, residual_type);
            }
#ifdef ENABLE_ICP
        }
#endif
        residualStats(problem, good_matches, residual_type);
    }

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
        const std::vector<std::vector<std::pair<int, int>>> &good_matches,
        const std::vector<std::vector<ResidualType>> &residual_type
        ) {
    // compute some statistics about residuals
    double cost;
    std::vector<double> residuals;
    ceres::Problem::EvaluateOptions evaluate_options;
    evaluate_options.apply_loss_function = false;
    problem.Evaluate(evaluate_options, &cost, &residuals, NULL, NULL);
    std::vector<double> residuals_3D3D, residuals_3D2D, residuals_2D3D, residuals_2D2D, residuals_3DPD;
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
    for(; ri < residuals.size(); ri++) {
        residuals_3DPD.push_back(abs(residuals[ri]));
    }
    double sum_3D3D = 0, sum_3D2D = 0, sum_2D3D = 0, sum_2D2D = 0, sum_3DPD = 0;
    for(auto r : residuals_3D3D) {sum_3D3D += r;}
    for(auto r : residuals_3D2D) {sum_3D2D += r;}
    for(auto r : residuals_2D3D) {sum_2D3D += r;}
    for(auto r : residuals_2D2D) {sum_2D2D += r;}
    for(auto r : residuals_3DPD) {sum_3DPD += r;}
    std::sort(residuals_3D3D.begin(), residuals_3D3D.end());
    std::sort(residuals_3D2D.begin(), residuals_3D2D.end());
    std::sort(residuals_2D3D.begin(), residuals_2D3D.end());
    std::sort(residuals_2D2D.begin(), residuals_2D2D.end());
    std::sort(residuals_3DPD.begin(), residuals_3DPD.end());
    std::cerr << "Cost: " << cost
        << " Residual blocks: " << problem.NumResidualBlocks()
        << " Residuals: " << problem.NumResiduals() << " " << ri << " " << residuals.size()
        << std::endl
        << " Total good matches: ";
    for(int cam=0; cam<num_cams; cam++) {
        std::cerr << " " << good_matches[cam].size();
    }
    for(int cam=0; cam<num_cams; cam++) {
        std::cerr << " " << residual_type[cam].size();
    }
    std::cerr << std::endl;
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
    if(residuals_3DPD.size())
        std::cerr << "Residual 3DPD:"
            << " median " << std::fixed << std::setprecision(10) << residuals_3DPD[residuals_3DPD.size()/2]
            << " mean " << std::fixed << std::setprecision(10) << sum_3DPD/residuals_3DPD.size()
            << " count " << residuals_3DPD.size() << std::endl;
}

void triangulatePoint(
        const std::vector<std::map<int, cv::Point2f>> &keypoint_obs2,
        const std::vector<std::map<int, pcl::PointXYZ>> &keypoint_obs3,
        const std::vector<double[6]> &camera_poses,
        pcl::PointXYZ &point,
        bool initial_guess
        ) {
    // given the 2D and 3D observations, the goal is to obtain
    // the 3D position of the point
    int initialized = 0;
    // 0: uninitialized
    // 1: one 2d measurement
    // 2: two 2d measurements
    // 3: initialized
    double transform[3] = {0, 0, 10};
    if(initial_guess) {
        transform[0] = point.x;
        transform[1] = point.y;
        transform[2] = point.z;
        initialized = 3;
    }

    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    for(int cam=0; cam<num_cams; cam++) {
        for(auto obs3 : keypoint_obs3[cam]) {
            /*
            std::cerr << "3D observation " <<  obs3.second
                << " at " << obs3.first << ": ";
            for(int i=0; i<6; i++) {
                std::cerr << camera_poses[obs3.first][i] << " ";
            }
            std::cerr << std::endl;
            std::cerr << util::pose_mat2vec(camera_poses[obs3.first]);
            std::cerr << std::endl;
            */
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<triangulation3D, 3, 3>(
                        new triangulation3D(
                            obs3.second.x,
                            obs3.second.y,
                            obs3.second.z,
                            camera_poses[obs3.first][0],
                            camera_poses[obs3.first][1],
                            camera_poses[obs3.first][2],
                            camera_poses[obs3.first][3],
                            camera_poses[obs3.first][4],
                            camera_poses[obs3.first][5]
                            )
                        );
            problem.AddResidualBlock(
                    cost_function,
                    new ceres::TrivialLoss,
                    //new ceres::CauchyLoss(loss_thresh_3D3D),
                    transform);
            if(!initialized) {
                initialized = 3;
                ceres::Solve(options, &problem, &summary);
            }
        }
    }
    for(int cam=0; cam<num_cams; cam++) {
        for(auto obs2 : keypoint_obs2[cam]) {
            /*
            std::cerr << "2D observation " <<  obs2.second << ": ";
            for(int i=0; i<6; i++) {
                std::cerr << camera_poses[obs2.first][i] << " ";
            }
            std::cerr << ", " << cam_trans[cam].transpose();
            std::cerr << std::endl;
            */
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<triangulation2D, 2, 3>(
                        new triangulation2D(
                            obs2.second.x,
                            obs2.second.y,
                            camera_poses[obs2.first][0],
                            camera_poses[obs2.first][1],
                            camera_poses[obs2.first][2],
                            camera_poses[obs2.first][3],
                            camera_poses[obs2.first][4],
                            camera_poses[obs2.first][5],
                            cam_trans[cam](0),
                            cam_trans[cam](1),
                            cam_trans[cam](2)
                            )
                        );
            problem.AddResidualBlock(
                    cost_function,
                    new ceres::ScaledLoss(
                        new ceres::CauchyLoss(loss_thresh_3D2D),
                        weight_3D2D,
                        ceres::TAKE_OWNERSHIP),
                    transform);
        }
    }
    ceres::Solve(options, &problem, &summary);
    point.x = transform[0];
    point.y = transform[1];
    point.z = transform[2];
}

void getLandmarksAtFrame(
        const Eigen::Matrix4d &pose,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr landmarks,
        const std::vector<bool> &keypoint_added,
        const std::vector<std::vector<std::vector<int>>> &keypoint_ids,
        const int frame,
        std::map<int, pcl::PointXYZ> &landmarks_at_frame) {
    Eigen::Matrix4d poseinv = pose.inverse();
    for(int cam = 0; cam < num_cams; cam++) {
        for(int id : keypoint_ids[cam][frame]) {
            if(landmarks_at_frame.count(id)) continue;
            if(!keypoint_added[id]) continue;
            auto point = landmarks->at(id);
            Eigen::Vector4d q;
            q << point.x, point.y, point.z, 1;
            Eigen::Vector4d p = poseinv * q;
            pcl::PointXYZ pp;
            pp.x = p(0)/p(3);
            pp.y = p(1)/p(3);
            pp.z = p(2)/p(3);
            landmarks_at_frame[id] = pp;
            /*
            std::cerr << "Getting landmark " << id << ": "
                << pp << std::endl;
            std::cerr << pose << std::endl;
            */
        }
    }
}
