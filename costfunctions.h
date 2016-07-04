#pragma once

struct cost3DPD {
    // 3DPD
    cost3DPD(
            double point_x,
            double point_y,
            double point_z,
            double normal_x,
            double normal_y,
            double normal_z,
            double offset_x,
            double offset_y,
            double offset_z) :
        point_x(point_x),
        point_y(point_y),
        point_z(point_z),
        normal_x(normal_x),
        normal_y(normal_y),
        normal_z(normal_z),
        offset_x(offset_x),
        offset_y(offset_y),
        offset_z(offset_z) {}

    template <typename T>
    bool operator()(const T* x, T* residual) const {
        // x[0], x[1], x[2] are angle-axis rotation
        T M[3], M_original[3];
        M_original[0] = T(point_x);
        M_original[1] = T(point_y);
        M_original[2] = T(point_z);
        ceres::AngleAxisRotatePoint(x, M_original, M);
        M[0] += x[3] - T(offset_x);
        M[1] += x[4] - T(offset_y);
        M[2] += x[5] - T(offset_z);
        residual[0] = M[0] * T(normal_x)
            + M[1] * T(normal_y)
            + M[2] * T(normal_z);
        return true;
    }
    double point_x, point_y, point_z,
           normal_x, normal_y, normal_z,
           offset_x, offset_y, offset_z;
};

struct cost3D3D {
    // 3D point to 3D point distance
    cost3D3D(
            double m_x,
            double m_y,
            double m_z,
            double s_x,
            double s_y,
            double s_z) :
        m_x(m_x),
        m_y(m_y),
        m_z(m_z),
        s_x(s_x),
        s_y(s_y),
        s_z(s_z) {}

    template <typename T>
    bool operator()(const T* x, T* residual) const {
        T M[3], M_original[3];
        M_original[0] = T(m_x);
        M_original[1] = T(m_y);
        M_original[2] = T(m_z);
        ceres::AngleAxisRotatePoint(x, M_original, M);
        residual[0] = M[0] + x[3] - T(s_x);
        residual[1] = M[1] + x[4] - T(s_y);
        residual[2] = M[2] + x[5] - T(s_z);
        return true;
    }
    double m_x, m_y, m_z,
           s_x, s_y, s_z;
};

struct cost3D2D {
    // 3D point to 2D point reprojection distance
    cost3D2D(
            double m_x,
            double m_y,
            double m_z,
            double s_x,
            double s_y,
            double t_x,
            double t_y,
            double t_z) :
        m_x(m_x),
        m_y(m_y),
        m_z(m_z),
        s_x(s_x),
        s_y(s_y),
        t_x(t_x),
        t_y(t_y),
        t_z(t_z) {}
    template <typename T>
    bool operator()(const T* x, T* residual) const {
        T M[3], M_original[3];
        M_original[0] = T(m_x);
        M_original[1] = T(m_y);
        M_original[2] = T(m_z);
        ceres::AngleAxisRotatePoint(x, M_original, M);

        M[0] += x[3] + T(t_x);
        M[1] += x[4] + T(t_y);
        M[2] += x[5] + T(t_z);

        residual[0] = M[0]/M[2] - T(s_x);
        residual[1] = M[1]/M[2] - T(s_y);
        return true;
    }
    double m_x, m_y, m_z,
           s_x, s_y,
           t_x, t_y, t_z;
};

struct cost2D3D {
    // 2D point to 3D point reprojection distance
    cost2D3D(
            double m_x,
            double m_y,
            double m_z,
            double s_x,
            double s_y,
            double t_x,
            double t_y,
            double t_z) :
        m_x(m_x),
        m_y(m_y),
        m_z(m_z),
        s_x(s_x),
        s_y(s_y),
        t_x(t_x),
        t_y(t_y),
        t_z(t_z) {}
    template <typename T>
    bool operator()(const T* x, T* residual) const {
        T M[3], M_original[3], rot[3];
        rot[0] = -x[0];
        rot[1] = -x[1];
        rot[2] = -x[2];
        M_original[0] = T(m_x) - x[3];
        M_original[1] = T(m_y) - x[4];
        M_original[2] = T(m_z) - x[5];
        ceres::AngleAxisRotatePoint(rot, M_original, M);
        M[0] += T(t_x);
        M[1] += T(t_y);
        M[2] += T(t_z);
        residual[0] = M[0]/M[2] - T(s_x);
        residual[1] = M[1]/M[2] - T(s_y);

        return true;
    }
    double m_x, m_y, m_z,
           s_x, s_y,
           t_x, t_y, t_z;
};

struct cost2D2D {
    // 2D point to 2D point epipolar constraint
    cost2D2D(
            double m_x,
            double m_y,
            double s_x,
            double s_y) :
        m_x(m_x),
        m_y(m_y),
        s_x(s_x),
        s_y(s_y) {}

    template <typename T>
    bool operator()(const T* x, T* residual) const {
        T M[3], M_original[3];
        M_original[0] = T(m_x);
        M_original[1] = T(m_y);
        M_original[2] = T(1.0);
        ceres::AngleAxisRotatePoint(x, M_original, M);
        M[0] /= M[2];
        M[1] /= M[2];
        // E = [t]_x R
        // S^T E M = residual
        // S dot (t cross (R M)) = residual
        residual[0] =
                M[0] * (-s_y * x[5] + x[4]) +
                M[1] * ( s_x * x[5] - x[3]) + 
                M[2] * (-s_x * x[4] + s_y * x[3]);
        /*
        residual[0] =
                s_x * (-M[1] * x[5] + x[4]) +
                s_y * ( M[0] * x[5] - x[3]) + 
                s_z * (-M[0] * x[4] + M[1] * x[3]);
                */
        return true;
    }
    double m_x, m_y, m_z,
           s_x, s_y, s_z;
};

struct bundle2D {
    bundle2D(
            double s_x,
            double s_y,
            double t_x,
            double t_y,
            double t_z) :
        s_x(s_x),
        s_y(s_y),
        t_x(t_x),
        t_y(t_y),
        t_z(t_z) {}

    template <typename T>
    bool operator()(const T* point, const T* camera, T* residual) const {
        T M[3], M_original[3], rot[3];
        rot[0] = -camera[0];
        rot[1] = -camera[1];
        rot[2] = -camera[2];
        M_original[0] = T(point[0]) - camera[3];
        M_original[1] = T(point[1]) - camera[4];
        M_original[2] = T(point[2]) - camera[5];
        ceres::AngleAxisRotatePoint(rot, M_original, M);

        M[0] += t_x;
        M[1] += t_y;
        M[2] += t_z;
        residual[0] = M[0]/M[2] - s_x;
        residual[1] = M[1]/M[2] - s_y;
        return true;
    }
    double s_x, s_y,
           t_x, t_y, t_z;
};

struct bundle3D3D {
    // 3D point to 3D point distance
    bundle3D3D(
            double s_x,
            double s_y,
            double s_z,
            double weight) :
        s_x(s_x),
        s_y(s_y),
        s_z(s_z),
        weight(weight) {}

    template <typename T>
    bool operator()(const T* point, const T* camera, T* residual) const {
        T M[3], M_original[3], rot[3];
        rot[0] = -camera[0];
        rot[1] = -camera[1];
        rot[2] = -camera[2];
        M_original[0] = T(point[0]) - camera[3];
        M_original[1] = T(point[1]) - camera[4];
        M_original[2] = T(point[2]) - camera[5];
        ceres::AngleAxisRotatePoint(rot, M_original, M);

        residual[0] = weight * (M[0] - s_x);
        residual[1] = weight * (M[1] - s_y);
        residual[2] = weight * (M[2] - s_z);
        return true;
    }
    double s_x, s_y, s_z,
           weight;
};
