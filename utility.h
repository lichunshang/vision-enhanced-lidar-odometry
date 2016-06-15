const double INF = 1e18;
class util {
    public:
    static pcl::PointXYZ linterpolate(
            const pcl::PointXYZ p1,
            const pcl::PointXYZ p2,
            const float start,
            const float end,
            const float mid) {
        float a = (mid-start)/(end-start);
        float b = 1 - a;
        float x = p1.x * b + p2.x * a;
        float y = p1.y * b + p2.y * a;
        float z = p1.z * b + p2.z * a;
        return pcl::PointXYZ(x, y, z);
    }
    static float linterpolate(
            const float p1,
            const float p2,
            const float start,
            const float end,
            const float mid) {
        float a = (mid-start)/(end-start);
        float b = 1 - a;
        return p1 * b + p2 * a;
    }
    static inline void add_assign(pcl::PointXYZ &a, const pcl::PointXYZ &b) {
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
    }
    static inline void subtract_assign(pcl::PointXYZ &a, const pcl::PointXYZ &b) {
        a.x -= b.x;
        a.y -= b.y;
        a.z -= b.z;
    }
    static inline pcl::PointXYZ add(const pcl::PointXYZ &a, const pcl::PointXYZ &b) {
        return pcl::PointXYZ(a.x+b.x, a.y+b.y, a.z+b.z);
    }
    static inline void scale(pcl::PointXYZ &p, double s) {
        p.x *= s;
        p.y *= s;
        p.z *= s;
    }
    static inline double norm(const pcl::PointXYZ &p) {
        return std::sqrt(util::norm2(p));
    }
    static inline double norm2(const pcl::PointXYZ &p) {
        return p.x*p.x + p.y*p.y + p.z*p.z;
    }
    static void save_cloud_txt(std::string s, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
        std::ofstream fout;
        fout.open(s.c_str());
        for(int i=0; i<cloud->size(); i++) {
            fout << std::setprecision(9) << cloud->at(i).x << " "
                << cloud->at(i).y << " "
                << cloud->at(i).z << std::endl;
        }
        fout.close();
    }
    static Eigen::Matrix4d pose_mat2vec(const double transform[6]) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity(4,4);
        double R[9];
        double R_angle_axis[3];
        for(int i=0; i<3; i++) R_angle_axis[i] = transform[i];
        ceres::AngleAxisToRotationMatrix<double>(R_angle_axis, R);
        for(int j=0; j<3; j++) {
            for(int i=0; i<3; i++) {
                T(i, j) = R[j*3 + i];
            }
        }
        T(0, 3) = transform[3];
        T(1, 3) = transform[4];
        T(2, 3) = transform[5];
        return T;
    }
    static void pose_vec2mat(const Eigen::Matrix4d T, double transform[6]) {
        double R[9];
        for(int j=0; j<3; j++) {
            for(int i=0; i<3; i++) {
                R[j*3 + i] = T(i, j);
            }
        }
        double R_angle_axis[3];
        ceres::RotationMatrixToAngleAxis<double>(R, R_angle_axis);
        for(int i=0; i<3; i++) transform[i] = R_angle_axis[i];
        transform[3] = T(0, 3);
        transform[4] = T(1, 3);
        transform[5] = T(2, 3);
    }
};
