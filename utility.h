class utility {
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
};
