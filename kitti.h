#pragma once

const int num_cams = 2, // number of cameras we use
    num_cams_actual = 4, // number of cameras actually available in dataset
    lkt_window = 21,
    lkt_pyramid = 4,
    corner_count = 3000, // number of features
    icp_skip = 200,
    f2f_iterations = 2,
    icp_iterations = 3,
    min_matches = 0, // minimum number of feature matches to proceed
    detect_every = 1, // detect new features every this number of frames
    ba_every = 10, // bundle adjust every this number of frames
    ndiagonal = 4;

const double
    flow_outlier = 20000, // pixels^2, squared distance of optical flow
    quality_level = 0.001, // good features to track quality
    min_distance = 12, // pixel distance between nearest features
    weight_3D2D = 10,
    weight_2D2D = 500,
    weight_3DPD = 1, // there are more of them
    loss_thresh_3D2D = 0.01, // reprojection error, canonical camera units
    loss_thresh_2D2D = 0.00002,
    loss_thresh_3DPD = 0.1, // physical distance, meters
    loss_thresh_3D3D = 0.04, // physical distance, meters
    match_thresh = 29, // bits, hamming distance for FREAK features
    depth_assoc_thresh = 0.015, // canonical camera units
    z_weight = 0.6,
    outlier_reject = 5.0,
    correspondence_thresh_icp = 0.5,
    icp_norm_condition = 1e-5,
    agreement_t_thresh = 0.1, // meters
    agreement_r_thresh = 0.05, // radians
    loop_close_thresh = 10; // meters

int img_width = 1226, // kitti data
    img_height = 370;

std::vector<Eigen::Matrix<float, 3, 4>,
    Eigen::aligned_allocator<Eigen::Matrix<float, 3, 4>>> cam_mat;
std::vector<Eigen::Matrix3f,
    Eigen::aligned_allocator<Eigen::Matrix3f>> cam_intrinsic;
std::vector<Eigen::Matrix3f,
    Eigen::aligned_allocator<Eigen::Matrix3f>> cam_intrinsic_inv;
std::vector<Eigen::Vector3f,
    Eigen::aligned_allocator<Eigen::Vector3f>> cam_trans;
std::vector<Eigen::Matrix4f,
    Eigen::aligned_allocator<Eigen::Vector3f>> cam_pose;
Eigen::Matrix4f velo_to_cam, cam_to_velo;
std::vector<double> min_x, max_x, min_y, max_y;

std::ofstream output;

std::vector<double> times;

const std::string kittipath = "/home/dllu/kitti/dataset/sequences/";

void loadCalibration(
        const std::string & dataset
        ) {
    std::string calib_path = kittipath + dataset + "/calib.txt";
    std::ifstream calib_stream(calib_path);
    std::string P;
    velo_to_cam = Eigen::Matrix4f::Identity();
    for(int cam=0; cam<num_cams_actual; cam++) {
        calib_stream >> P;
        cam_mat.push_back(Eigen::Matrix<float, 3, 4>());
        for(int i=0; i<3; i++) {
            for(int j=0; j<4; j++) {
                calib_stream >> cam_mat[cam](i,j);
            }
        }
        Eigen::Matrix3f K = cam_mat[cam].block<3,3>(0,0);
        Eigen::Matrix3f Kinv = K.inverse();
        Eigen::Vector3f Kt = cam_mat[cam].block<3,1>(0,3);
        Eigen::Vector3f t = Kinv * Kt;
        cam_trans.push_back(t);
        cam_intrinsic.push_back(K);
        cam_intrinsic_inv.push_back(K.inverse());

        cam_pose.push_back(Eigen::Matrix4f::Identity());
        cam_pose[cam](0, 3) = t(0);
        cam_pose[cam](1, 3) = t(1);
        cam_pose[cam](2, 3) = t(2);

        Eigen::Vector3f min_pt;
        min_pt << 0, 0, 1;
        min_pt = Kinv * min_pt;
        min_x.push_back(min_pt(0) / min_pt(2));
        min_y.push_back(min_pt(1) / min_pt(2));
        //std::cerr << "min_pt: " << min_pt << std::endl;
        Eigen::Vector3f max_pt;
        max_pt << img_width, img_height, 1;
        max_pt = Kinv * max_pt;
        max_x.push_back(max_pt(0) / max_pt(2));
        max_y.push_back(max_pt(1) / max_pt(2));
        //std::cerr << "max_pt: " << max_pt << std::endl;
    }
    calib_stream >> P;
    for(int i=0; i<3; i++) {
        for(int j=0; j<4; j++) {
            calib_stream >> velo_to_cam(i,j);
        }
    }

    cam_to_velo = velo_to_cam.inverse();
}

void loadTimes(
        const std::string & dataset
        ) {
    std::string time_path = kittipath + dataset + "/times.txt";
    std::ifstream time_stream(time_path);
    double t;
    while(time_stream >> t) {
        times.push_back(t);
    }
}

void loadPoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
        std::string dataset,
        int n
        ) {
    std::stringstream ss;
    ss << kittipath << dataset << "/velodyne/"
        << std::setfill('0') << std::setw(6) << n << ".bin";
    // allocate 40 MB buffer (only ~1300*4*4 KB are needed)
    int32_t num = 10000000;
    float *data = (float*)malloc(num*sizeof(float));

    // pointers
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;

    // load point cloud
    FILE *stream;
    stream = fopen (ss.str().c_str(),"rb");
    num = fread(data,sizeof(float),num,stream)/4;


    for (int32_t i=0; i<num; i++) {
        point_cloud->points.push_back(pcl::PointXYZ(*px,*py,*pz));
        px+=4; py+=4; pz+=4; pr+=4;
    }

    fclose(stream);
    free(data);
}

void segmentPoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans
        ) {
    float prev_y = 0;
    int scan_id = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(
            new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*point_cloud, *cloud_tmp, velo_to_cam);
    std::vector<std::vector<int>> scan_ids;
    for(int i=0, _i = point_cloud->size(); i<_i; i++) {
        pcl::PointXYZ p = point_cloud->at(i);
        if(i > 0 && p.x > 0 && (p.y > 0) != (prev_y > 0)) {
            scan_id++;
        }
        if(scan_id >= scans.size()) {
            scan_ids.push_back(std::vector<int>());
            scans.push_back(pcl::PointCloud<pcl::PointXYZ>::Ptr(
                        new pcl::PointCloud<pcl::PointXYZ>));
        }
        scan_ids[scan_id].push_back(i);
        prev_y = p.y;
    }
    // for some reason, kitti scans are sorted in a strange way
    for(int s=0; s<scan_ids.size(); s++) {
        for(int i = 0, _i = scan_ids[s].size(); i<_i; i++) {
            pcl::PointXYZ q = cloud_tmp->at(scan_ids[s][_i - 1 - (i + _i/2) % _i]);
            scans[s]->push_back(q);
        }
    }
    cloud_tmp.reset();
}

cv::Mat loadImage(
        const std::string & dataset,
        const int cam,
        const int n
        ) {
    std::stringstream ss;
    ss << kittipath << dataset << "/image_" << cam << "/"
        << std::setfill('0') << std::setw(6) << n << ".png";
    cv::Mat I = cv::imread(ss.str(), 0);
    img_width = I.cols;
    img_height = I.rows;
    return I;
}


void output_line(Eigen::Matrix4d result, std::ofstream &output) {
    output<< result(0,0) << " "
        << result(0,1) << " "
        << result(0,2) << " "
        << result(0,3) << " "
        << result(1,0) << " "
        << result(1,1) << " "
        << result(1,2) << " "
        << result(1,3) << " "
        << result(2,0) << " "
        << result(2,1) << " "
        << result(2,2) << " "
        << result(2,3) << " "
        << std::endl;
}
