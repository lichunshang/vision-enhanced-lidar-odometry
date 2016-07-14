const double INF = 1e18;
const double PI = 3.1415926535897932384626433832795028;
const float geomedian_EPS = 1e-6;
const float kp_EPS = 1e-6;
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
    static inline double dist2(const cv::Point2f &a, const cv::Point2f &b) {
        return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
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
    static void transform_point(pcl::PointXYZ &p, const double transform[6]) {
        double x[3] = {p.x, p.y, p.z}, y[3] = {0, 0, 0};
        ceres::AngleAxisRotatePoint(transform, x, y);
        p.x = y[0] + transform[3];
        p.y = y[1] + transform[4];
        p.z = y[2] + transform[5];
    }

    static cv::Point2f geomedian(std::vector<cv::Point2f> P) {
        int m = P.size();
        cv::Point2f y(0,0);
        for(int i=0; i<m; i++) {
            y += P[i];
        }
        y /= (float)m;
        for(int iter = 0; iter < 20; iter++) {
            cv::Point2f yy(0,0);
            float d = 0;
            for(int i=0; i<m; i++) {
                float no = cv::norm(P[i] - y);
                if(no < geomedian_EPS) {
                    return y;
                }
                float nn = 1.0/no;

                yy += P[i]*nn;
                d += nn;
            }
            if(cv::norm(yy/d - y) < geomedian_EPS) {
                return yy/d;
            }
            y = yy/d;
        }
        return y;
    }
};

class UF {
    public:
        UF(int n, int k) {
            _n = n;
            _k = k;
            _p = std::vector<int>(n);
            for(int i=0; i<n; i++) {
                _p[i] = i;
            }
        }
        void Union(int i, int j) {
            _p[Find(i)] = Find(j);
        }
        int Find(int i) {
            if (i > _p.size()) {
                std::cerr << "WTF how can you find something out of bounds" << std::endl;
            }
            return i == _p[i] ? i : _p[i] = Find(_p[i]);
        }
        void aggregate(std::map<int, std::set<int> > &q, int group_size) {
            for(int i=0; i<_n; i++) {
                Find(i);
            }
            for(int i=0; i<_n; i++) {
                _q[Find(i)].insert(i);
            }
            std::cerr << "Distinct classes: " << _q.size() << std::endl;
            int groups = 0;
            for(std::map<int, std::set<int> >::iterator it = _q.begin();
                    it != _q.end();
                    it++) {
                //std::cerr << "Class of " << it->second.size() << std::endl;
                if(it->second.size() == group_size/2) {
                    for(std::set<int>::iterator itt = it->second.begin();
                            itt != it->second.end();
                            itt++) {
                        q[groups].insert(*itt);
                    }
                    groups++;
                }
            }
            std::cerr << "groups: " << groups << std::endl;
        }
    private:
        int _n, _k;
        std::vector<int> _p;
        std::map<int, std::set<int> > _q;
};
