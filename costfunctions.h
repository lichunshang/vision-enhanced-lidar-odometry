#pragma once

struct costPlane {
    costPlane(
            double point_x,
            double point_y,
            double point_z,
            double normal_x,
            double normal_y,
            double normal_z,
            double offset_x,
            double offset_y,
            double offset_z,
            double weight) :
        point_x(point_x),
        point_y(point_y),
        point_z(point_z),
        normal_x(normal_x),
        normal_y(normal_y),
        normal_z(normal_z),
        offset_x(offset_x),
        offset_y(offset_y),
        offset_z(offset_z),
        weight(weight) {}

    template <typename T>
    bool operator()(const T* x, T* residual) const {
        // x[0], x[1], x[2] are angle-axis rotation
        T p[3], p_original[3];
        p_original[0] = T(point_x);
        p_original[1] = T(point_y);
        p_original[2] = T(point_z);
        ceres::AngleAxisRotatePoint(x, p_original, p);
        p[0] += x[3] - offset_x;
        p[1] += x[4] - offset_y;
        p[2] += x[5] - offset_z;
        residual[0] = p[0] * normal_x + p[1] * normal_y + p[2] * normal_z;
        return true;
    }
    double point_x, point_y, point_z,
           normal_x, normal_y, normal_z,
           offset_x, offset_y, offset_z;
};

struct cost3D3D {
    cost3D3D(
            double m_x,
            double m_y,
            double m_z,
            double s_x,
            double s_y,
            double s_z,
            double weight) :
        m_x(m_x),
        m_y(m_y),
        m_z(m_z),
        s_x(s_x),
        s_y(s_y),
        s_z(s_z),
        weight(weight) {}

    template <typename T>
    bool operator()(const T* x, T* residual) const {
        T M[3], M_original[3];
        M_original[0] = T(m_x);
        M_original[1] = T(m_y);
        M_original[2] = T(m_z);
        ceres::AngleAxisRotatePoint(x, M_original, M);
        residual[0] = M[0] - s_x;
        residual[1] = M[1] - s_y;
        residual[2] = M[2] - s_z;
        return true;
    }
    double m_x, m_y, m_z,
           s_x, s_y, s_z;
};
