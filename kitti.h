#pragma once

int num_cams = 1,
    corner_count = 200, // number of features per cell
    row_cells = 6, 
    col_cells = 18,
    img_width = 1226, // kitti data
    img_height = 370,
    cell_width = img_width / col_cells,
    cell_height = img_height / row_cells,
    dist_thresh = 60, // square of min distance between new detected features and existing features
    proj_dist_thresh = 10,
    bundle_length = 5;

const double PI = 3.1415926535897932384626433832795028;
double focal_length = 7.18856e2,
      cx = 6.071928e2,
      cy = 1.852157e2;

Eigen::Matrix4d velo_to_cam, cam_to_velo;

std::ofstream output;

std::vector<double> times;

const std::string kittipath = "/mnt/data/kitti/dataset/sequences/";

void loadCalibration(
        std::string dataset
        ) {
    std::string calib_path = kittipath + dataset + "/calib.txt";
    std::ifstream calib_stream(calib_path);
    std::string P;
    velo_to_cam = Eigen::Matrix4d::Identity();
    calib_stream >> P;
    for(int i=0; i<3; i++) {
        for(int j=0; j<4; j++) {
            calib_stream >> velo_to_cam(i,j);
        }
    }
    focal_length = velo_to_cam(0,0);
    cx = velo_to_cam(0,2);
    cy = velo_to_cam(1,2);

    cam_to_velo = velo_to_cam.inverse();
}

void loadTimes(
        std::string dataset
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
    double prev_azimuth = -1;
    int scan_id = 0;
    for(int i=0, _i = point_cloud->size(); i<_i; i++) {
        pcl::PointXYZ p = point_cloud->at(i);
        double azimuth = std::atan2(p.y, p.x);
        if(azimuth < 0) azimuth += 2*PI;
        if(i > 0 && std::abs(azimuth - prev_azimuth) > 4) {
            scan_id++;
        }
        if(scan_id >= scans.size()) {
            scans.push_back(pcl::PointCloud<pcl::PointXYZ>::Ptr(
                        new pcl::PointCloud<pcl::PointXYZ>));
        }
        scans[scan_id]->push_back(p);
        prev_azimuth = azimuth;
    }
}

cv::Mat loadImage(
        std::string dataset,
        int n
        ) {
    std::stringstream ss;
    ss << kittipath << dataset << "/image_0/" 
        << std::setfill('0') << std::setw(6) << n << ".png";
    return cv::imread(ss.str(), 0);
}
        

void output_line(Eigen::Matrix4d T, std::ofstream &output) {
    Eigen::Matrix4d result = velo_to_cam * T * cam_to_velo;
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
