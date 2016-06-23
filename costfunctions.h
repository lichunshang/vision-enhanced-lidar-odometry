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
        T M[3], M_original[3];
        M_original[0] = T(point_x);
        M_original[1] = T(point_y);
        M_original[2] = T(point_z);
        ceres::AngleAxisRotatePoint(x, M_original, M);
        M[0] += x[3] - offset_x;
        M[1] += x[4] - offset_y;
        M[2] += x[5] - offset_z;
        residual[0] = M[0] * normal_x + M[1] * normal_y + M[2] * normal_z;
        return true;
    }
    double point_x, point_y, point_z,
           normal_x, normal_y, normal_z,
           offset_x, offset_y, offset_z,
           weight;
};

struct cost3D3D {
    // 3D point to 3D point distance
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
        residual[0] = weight * (M[0] + x[3] - s_x);
        residual[1] = weight * (M[1] + x[4] - s_y);
        residual[2] = weight * (M[2] + x[5] - s_z);
        return true;
    }
    double m_x, m_y, m_z,
           s_x, s_y, s_z,
           weight;
};

/*
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
            double t_z,
            double weight) :
        m_x(m_x),
        m_y(m_y),
        m_z(m_z),
        s_x(s_x),
        s_y(s_y),
        t_x(t_x),
        t_y(t_y),
        t_z(t_z),
        weight(weight) {}
    template <typename T>
    bool operator()(const T* x, T* residual) const {
        T M[3], M_original[3];
        M_original[0] = T(m_x);
        M_original[1] = T(m_y);
        M_original[2] = T(m_z);
        ceres::AngleAxisRotatePoint(x, M_original, M);

        M[0] += x[3];
        M[1] += x[4];
        M[2] += x[5];

        T r_0 = P_00 * M[0] + P_01 * M[1] + P_02 * M[2] + P_03;
        T r_1 = P_10 * M[0] + P_11 * M[1] + P_12 * M[2] + P_13;
        T r_2 = P_20 * M[0] + P_21 * M[1] + P_22 * M[2] + P_23;
        residual[0] = weight * (r_0/r_2 - s_x);
        residual[1] = weight * (r_1/r_2 - s_y);
        return true;
    }
    double m_x, m_y, m_z,
           s_x, s_y,
           P_00, P_01, P_02, P_03,
           P_10, P_11, P_12, P_13,
           P_20, P_21, P_22, P_23,
           weight;
};

struct cost2D3D {
    // 2D point to 3D point reprojection distance
    cost2D3D(
            double s_x,
            double s_y,
            double s_z,
            double m_x,
            double m_y,
            double P_00, // P is the 3 by 4 projection matrix double P_01,
            double P_01,
            double P_02,
            double P_03,
            double P_10,
            double P_11,
            double P_12,
            double P_13,
            double P_20,
            double P_21,
            double P_22,
            double P_23,
            double weight) :
        s_x(s_x),
        s_y(s_y),
        s_z(s_z),
        m_x(m_x),
        m_y(m_y),
        P_00(P_00),
        P_01(P_01),
        P_02(P_02),
        P_03(P_03),
        P_10(P_10),
        P_11(P_11),
        P_12(P_12),
        P_13(P_13),
        P_20(P_20),
        P_21(P_21),
        P_22(P_22),
        P_23(P_23),
        weight(weight) {}
    template <typename T>
    bool operator()(const T* x, T* residual) const {
        T S[3], S_original[3], rot[3];
        rot[0] = -x[0];
        rot[1] = -x[1];
        rot[2] = -x[2];
        S_original[0] = T(s_x) - x[3];
        S_original[1] = T(s_y) - x[4];
        S_original[2] = T(s_z) - x[5];
        ceres::AngleAxisRotatePoint(rot, S_original, S);

        T r_0 = P_00 * S[0] + P_01 * S[1] + P_02 * S[2] + P_03;
        T r_1 = P_10 * S[0] + P_11 * S[1] + P_12 * S[2] + P_13;
        T r_2 = P_20 * S[0] + P_21 * S[1] + P_22 * S[2] + P_23;
        residual[0] = weight * (r_0/r_2 - m_x);
        residual[1] = weight * (r_1/r_2 - m_y);
        return true;
    }
    double m_x, m_y,
           s_x, s_y, s_z,
           P_00, P_01, P_02, P_03,
           P_10, P_11, P_12, P_13,
           P_20, P_21, P_22, P_23,
           weight;
};
*/
struct cost2D2D {
    // 2D point to 2D point epipolar constraint
    cost2D2D(
            double m_x,
            double m_y,
            double s_x,
            double s_y,
            double weight) :
        m_x(m_x),
        m_y(m_y),
        s_x(s_x),
        s_y(s_y),
        weight(weight) {}

    template <typename T>
    bool operator()(const T* x, T* residual) const {
        T M[3], M_original[3];
        M_original[0] = T(m_x);
        M_original[1] = T(m_y);
        M_original[2] = T(1);
        ceres::AngleAxisRotatePoint(x, M_original, M);
        // E = [t]_x R
        // S dot E M = residual
        // S dot (t cross (R M)) = residual
        residual[0] = weight * (
                M[0] * (-s_y * x[5] + x[4]) +
                M[1] * ( s_x * x[5] - x[3]) + 
                M[2] * (-s_x * x[4] + s_y * x[3])
                );
        return true;
    }
    double m_x, m_y, m_z,
           s_x, s_y, s_z,
           weight;
};

struct bundle2D {
    bundle2D(
            double s_x,
            double s_y,
            double P_00, // P is the 3 by 4 projection matrix
            double P_01,
            double P_02,
            double P_03,
            double P_10,
            double P_11,
            double P_12,
            double P_13,
            double P_20,
            double P_21,
            double P_22,
            double P_23,
            double weight) :
        s_x(s_x),
        s_y(s_y),
        P_00(P_00),
        P_01(P_01),
        P_02(P_02),
        P_03(P_03),
        P_10(P_10),
        P_11(P_11),
        P_12(P_12),
        P_13(P_13),
        P_20(P_20),
        P_21(P_21),
        P_22(P_22),
        P_23(P_23),
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

        T r_0 = P_00 * M[0] + P_01 * M[1] + P_02 * M[2] + P_03;
        T r_1 = P_10 * M[0] + P_11 * M[1] + P_12 * M[2] + P_13;
        T r_2 = P_20 * M[0] + P_21 * M[1] + P_22 * M[2] + P_23;
        residual[0] = weight * (r_0/r_2 - s_x);
        residual[1] = weight * (r_1/r_2 - s_y);
        return true;
    }
    double s_x, s_y,
           P_00, P_01, P_02, P_03,
           P_10, P_11, P_12, P_13,
           P_20, P_21, P_22, P_23,
           weight;
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
