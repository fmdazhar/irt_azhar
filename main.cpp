#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <eigen3/Eigen/Dense>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <fstream>
#include <string>
#include <filesystem>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <thread>
#include <atomic>

using namespace std;   // Added this
using namespace Eigen; // Added this
std::atomic<bool> tau3_in_changed(false);


// Global variable declaration
// Define constants for number of states and inputs
const size_t number_of_states = 7;
const size_t number_of_inputs = 3;
std::array<double, number_of_states> result;



// Function to capture keyboard input without waiting for Enter key
char my_getch()
{
    char buf = 0;
    struct termios old = {0};
    if (tcgetattr(0, &old) < 0)
    {
        perror("tcsetattr()");
    }
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSANOW, &old) < 0)
    {
        perror("tcsetattr ICANON");
    }
    if (read(0, &buf, 1) < 0)
    {
        perror("read()");
    }
    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(0, TCSADRAIN, &old) < 0)
    {
        perror("tcsetattr ~ICANON");
    }
    return buf;
}
template <typename Type> class Vessel
{
public:
    Type rho_water;
    Type rho_w;
    Type g;
    Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> XFN;
    Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> XBETA;
    Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> YBETA;
    Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> NBETA;
    Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> XGAMMA;
    Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> YGAMMA;
    Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> NGAMMA;
    Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> XFN_FN_SQ;
    Eigen::Matrix<Type, Eigen::Dynamic, 1> alpha_w;
    Eigen::Matrix<Type, Eigen::Dynamic, 1> CFx_w;
    Eigen::Matrix<Type, Eigen::Dynamic, 1> CFy_w;
    Eigen::Matrix<Type, Eigen::Dynamic, 1> CMz_w;
    Type AF_w;
    Type AL_w;
    Type Lpp_w;
    Eigen::Matrix<Type, Eigen::Dynamic, 1> D;
    Type B, T;
    Eigen::Matrix<Type, 3, 1> CoG;
    Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> M, MA, Minv;
    Type ALW;
    Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> MRB;
    Type A11, A22, A26, A62, A66, Loa, Boa, Lpp, c_B, c_T, cb, c_rzz, c_ALW, c_S;
    Eigen::Matrix<Type, 3, 1> c_CoG;
    Eigen::Matrix<Type, Eigen::Dynamic, 1> S;

    Vessel()
    {
        Eigen::VectorXd Scaling(3);
        Scaling << 1.0, 1.0, 1.0;
        initializeVessel(Scaling);
    }
    // Vessel(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &Scaling)
    // {
    //     Eigen::VectorXd Scaling(3);
    //     Scaling << 1.0, 1.0, 1.0;
    //     initializeVessel(Scaling);
    // }

    // Function to load data from file
    Eigen::MatrixXd readMatrixFromFile(const std::string &fileName)
    {
        std::vector<std::vector<double>> data;
        std::ifstream infile(fileName);
        std::string line;

        // Read the data line by line
        while (std::getline(infile, line))
        {
            std::vector<double> rowData;
            std::istringstream iss(line);
            double num;

            // Split each line by whitespace to get the columns
            while (iss >> num)
            {
                rowData.push_back(num);
            }
            data.push_back(rowData);
        }

        // Determine the size of the matrix
        int rows = data.size();
        if (rows == 0) // Return empty matrix if file was empty
            return Eigen::MatrixXd();

        int cols = data[0].size();

        // Create and fill the Eigen matrix
        Eigen::MatrixXd mat(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                mat(i, j) = data[i][j];
            }
        }
        return mat;
    }
    Eigen::MatrixXd readMatrixFromData(const std::string &fileName)
    {
        std::vector<std::vector<double>> data;
        std::ifstream infile(fileName);
        std::string line;

        // Read the data line by line
        while (std::getline(infile, line))
        {
            std::vector<double> rowData;
            std::istringstream iss(line);
            std::string token;

            // Split each line by commas to get the columns
            while (std::getline(iss, token, ','))
            {
                double num = std::stod(token); // Convert string token to double
                rowData.push_back(num);
            }
            data.push_back(rowData);
        }

        // Determine the size of the matrix
        int rows = data.size();
        if (rows == 0) // Return empty matrix if file was empty
            return Eigen::MatrixXd();

        int cols = data[0].size();

        // Create and fill the Eigen matrix
        Eigen::MatrixXd mat(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                mat(i, j) = data[i][j];
            }
        }
        return mat;
    }

    void initializeVessel(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &Scaling)
    {
        // Vessel vessel;
        rho_water = 1.0; // [t/m^3]
        g = 9.81;        // G [m/s^2]

        // Use model identification parameters: 0: Default model; 1: Real ferry data identification trials
        int ident_params = 1;

        std::filesystem::path base_path = std::filesystem::current_path();
        base_path /= "include";

        // Current resistance data
        XFN = readMatrixFromFile(base_path / "XFN.txt");
        XBETA = readMatrixFromFile(base_path / "XBETA.txt");
        YBETA = readMatrixFromFile(base_path / "YBETA.txt");
        NBETA = readMatrixFromFile(base_path / "NBETA.txt");
        XGAMMA = readMatrixFromFile(base_path / "XGAMMA.txt");
        YGAMMA = readMatrixFromFile(base_path / "YGAMMA.txt");
        NGAMMA = readMatrixFromFile(base_path / "NGAMMA.txt");
        XFN_FN_SQ = XFN;
        XFN_FN_SQ.col(1) = XFN_FN_SQ.col(1).array() * XFN_FN_SQ.col(0).array().square();

        // Wind resistance data
        Eigen::MatrixXd wind_resistance_data = readMatrixFromData(base_path / "windcoeff.dat");
        alpha_w = wind_resistance_data.col(0);
        CFx_w = wind_resistance_data.col(1);
        CFy_w = wind_resistance_data.col(2);
        CMz_w = wind_resistance_data.col(3);

        // Wind resistance parameters
        AF_w = 43.7;   // Front area in m^2
        AL_w = 129.6;  // Lateral area in m^2
        Lpp_w = 36.88; // Length of wind attack area in m
        rho_w = 1.226; // Air density in kg/m^3
        S = Scaling;

        switch (ident_params)
        {
        case 0:
            Loa = 57.0;
            Boa = 12.9720;
            Lpp = 36.8;
            c_B = 0.3525;
            c_T = 0.0274;
            cb = 0.5720;
            c_rzz = 0.25;
            c_ALW = 0.8403;
            c_S = 0.8986;
            c_CoG = Eigen::Vector3d(0.0, 0.0, -1.4893);

            A11 = -3.747086e-04;
            A22 = -2.586713e-03;
            A26 = 0.0;
            A62 = 0.0;
            A66 = -8.725378e-05;

            D = Eigen::Vector3d(0.0, 0.0, 0.0);
            break;

        case 1:
            Eigen::VectorXd params1(15);
            params1 << 187100.511775658, 303030.866069127, 57378607.9640273, 198891.964122127,
                771.277192428326, 1775.78135270359, 7746.54537243618, 0.01, 0, 0,
                0, 0, 0.460051055484453, 0.460051055484453, 0.460051055484453; // ... continue initializing params1 ...

            Loa = 57.0;
            Boa = 12.9720;
            Lpp = 36.8;
            c_B = 0.3525;
            c_T = 0.0274;
            cb = 0.5720 * params1(3) / 282210.4130267135;
            c_rzz = sqrt((-8.725378e-05 * 1000 * rho_water / 2 * Lpp * Lpp * Lpp + params1(2)) / (params1(3) * Lpp * Lpp));
            c_ALW = 0.8403;
            c_S = 0.8986;
            c_CoG = Eigen::Vector3d(0.0, 0.0, -1.4893);
            A11 = (params1(3) - params1(0)) / (1000 * rho_water / 2 * Lpp * Lpp * Lpp);
            A22 = (params1(3) - params1(1)) / (1000 * rho_water / 2 * Lpp * Lpp * Lpp);
            A26 = 0.0;
            A62 = 0.0;
            A66 = -8.725378e-05;
            D = Eigen::Vector3d(params1(4) / 1000, params1(5) / 1000, params1(6) / 1000);
            break;
        }
    }
};
struct Results
{
    Eigen::VectorXd tau3_CC, tau3_hull, tau3_wind;
};


// Function to scale input
Eigen::VectorXd scale_input(const Eigen::VectorXd &unscaled_input, const Eigen::VectorXd &scaling)
{
    Eigen::VectorXd typ_U(3);
    typ_U << scaling[2], scaling[2], scaling[2] * scaling[0];

    return unscaled_input.array() / typ_U.array();
}

// Function to scale state
Eigen::VectorXd scale_state(const Eigen::VectorXd &unscaled_state, const Eigen::VectorXd &scaling)
{
    Eigen::VectorXd typ_S(7);
    typ_S << scaling[0], scaling[0], 1, scaling[0] / scaling[1], scaling[0] / scaling[1], 1 / scaling[1], 0;

    return unscaled_state.array() / typ_S.array();
}

// Function to scale state rate
Eigen::VectorXd scale_state_rate(const Eigen::VectorXd &unscaled_state_rate, const Eigen::VectorXd &scaling)
{
    Eigen::VectorXd typ_S_rate(7);
    typ_S_rate << scaling[0] / scaling[1], scaling[0] / scaling[1], 1 / scaling[1], scaling[0] / (scaling[1] * scaling[1]), scaling[0] / (scaling[1] * scaling[1]), 1 / (scaling[1] * scaling[1]), 0;

    return unscaled_state_rate.array() / typ_S_rate.array();
}

// Function to unscale input
Eigen::VectorXd unscale_input(const Eigen::VectorXd &scaled_input, const Eigen::VectorXd &scaling)
{
    Eigen::VectorXd typ_U(3);
    typ_U << scaling[2], scaling[2], scaling[2] * scaling[0];

    return scaled_input.array() * typ_U.array();
}

// Function to unscale state
Eigen::VectorXd unscale_state(const Eigen::VectorXd &scaled_state, const Eigen::VectorXd &scaling)
{
    Eigen::VectorXd typ_S(7);
    typ_S << scaling[0], scaling[0], 1, scaling[0] / scaling[1], scaling[0] / scaling[1], 1 / scaling[1], 0;

    return scaled_state.array() * typ_S.array();
}

double mod_alt(double a, double n)
{
    return a - std::floor(a / n) * n;
}

double ssa(double angle)
{
    const double pi = 3.141592653589793;
    angle = mod_alt(angle + pi, 2 * pi) - pi;
    return angle;
}


double linear_interpolate(const Eigen::VectorXd &x_data, const Eigen::VectorXd &y_data, double x)
{
    // Check if data is valid
    if (x_data.size() != y_data.size() || x_data.size() == 0)
    {
        throw std::invalid_argument("Invalid data provided for interpolation");
    }

    // Search for the interval
    for (size_t i = 0; i < x_data.size() - 1; i++)
    {
        if (x >= x_data[i] && x <= x_data[i + 1])
        {
            // Perform linear interpolation
            return y_data[i] + (x - x_data[i]) * (y_data[i + 1] - y_data[i]) / (x_data[i + 1] - x_data[i]);
        }
    }

    // If we get here, x is out of range
    throw std::out_of_range("x value is out of range for interpolation");
}

Results vessel_forces_ident(const Eigen::VectorXd &x, const Eigen::Vector2d &v_c, const Eigen::Vector2d &v_w, const Eigen::VectorXd &params, const Vessel<double> &stat_opts)
{
    double psi = x(2);
    double u = x(3);
    double v = x(4);
    double r = x(5);
    const double small_val = 0.0001;
    const double factorRad2Deg = 180.0 / M_PI;

    Eigen::Vector3d nu3(u, v, r);

    Eigen::Matrix3d R;
    R << cos(psi), -sin(psi), 0,
        sin(psi), cos(psi), 0,
        0, 0, 1;

    Eigen::Vector3d nu3r = nu3 - R * Eigen::Vector3d(v_c(0), v_c(1), 0);

    double U_c = sqrt(pow(std::abs(nu3r(0)) + small_val, 2) + pow(std::abs(nu3r(1)) + small_val, 2));
    double Fn = nu3r(0) / sqrt(stat_opts.g * stat_opts.Lpp);
    double beta_c = atan2(nu3r(1) + small_val * ((nu3r(1) + small_val > 0) - (nu3r(1) + small_val < 0)), nu3r(0) + small_val);
    double gamma_c = atan2(0.5 * r * stat_opts.Lpp, U_c);

    double dim_res = stat_opts.rho_water / 2 * pow(nu3r(0), 2) * stat_opts.S(0);
    double dim_drift = stat_opts.rho_water / 2 * pow(U_c, 2) * stat_opts.ALW;
    double dim_yaw = stat_opts.rho_water / 2 * (pow(U_c, 2) + pow(r * stat_opts.Lpp / 2, 2)) * stat_opts.ALW;

    double beta_force = -beta_c;
    double XFN_TOT, XHL, YHL, NHL;
    std::function<double(double)> XBETA, XGAMMA, YBETA, YGAMMA, NBETA, NGAMMA, XFN_FN_SQ;

    std::string mode = "lookup";
    if (mode == "lookup")
    {
        boost::math::barycentric_rational<double> br_XBETA(stat_opts.XBETA.col(0).data(), stat_opts.XBETA.col(1).data(), stat_opts.XBETA.rows());
        boost::math::barycentric_rational<double> br_NBETA(stat_opts.NBETA.col(0).data(), stat_opts.NBETA.col(1).data(), stat_opts.NBETA.rows());
        XBETA = [br_XBETA](double x)
        { return br_XBETA(x); };
        NBETA = [br_NBETA](double x)
        { return br_NBETA(x); };
        XGAMMA = [&stat_opts](double x)
        { return linear_interpolate(stat_opts.XGAMMA.col(0), stat_opts.XGAMMA.col(1), x); };
        YBETA = [&stat_opts](double x)
        { return linear_interpolate(stat_opts.YBETA.col(0), stat_opts.YBETA.col(1), x); };
        YGAMMA = [&stat_opts](double x)
        { return linear_interpolate(stat_opts.YGAMMA.col(0), stat_opts.YGAMMA.col(1), x); };
        NGAMMA = [&stat_opts](double x)
        { return linear_interpolate(stat_opts.NGAMMA.col(0), stat_opts.NGAMMA.col(1), x); };
        XFN_FN_SQ = [&stat_opts](double x)
        { return linear_interpolate(stat_opts.XFN_FN_SQ.col(0), stat_opts.XFN_FN_SQ.col(1), x); };
        XFN_TOT = XFN_FN_SQ(Fn) * stat_opts.rho_water / 2 * stat_opts.S(0) * stat_opts.g * stat_opts.Lpp;
    }
    else
    {
        XFN_TOT = stat_opts.rho_water / 2 * stat_opts.S(0) * (nu3r(1) * -0.0003 / (1 / sqrt(stat_opts.g * stat_opts.Lpp)) - 0.015 * pow(nu3r(1), 3) / sqrt(stat_opts.g * stat_opts.Lpp));

        XBETA = [](double x) -> double
        {
            return -(0.017917 + 0.015311) / 2 * (x / 285 * (0.017917 / 0.015311 - 1) + 1) * sin(abs(x * 2 * M_PI / 180));
        };

        XGAMMA = [](double x) -> double
        {
            return (-0.000183693611292432 * (x + 30) + 0.000003210963544062 * pow(x + 30, 2) - 0.000000045562381368 * pow(x + 30, 3) - 0.000000000293220217 * pow(x + 30, 4) + 0.000000000003090797 * pow(x + 30, 5));
        };

        YBETA = [](double x) -> double
        {
            return (0.43887 + 0.37541) / 2 * (x / 90 * (0.43887 / 0.37541 - 1) / 2 + 1) * sin(x * M_PI / 180);
        };

        YGAMMA = [](double x) -> double
        {
            return (-0.000015922796303 * (x + 5) + 0.000000139343598 * pow(x + 5, 2) - 0.000000001338792 * pow(x + 5, 3) - 0.009146608439613 * sin((x + 5) * 2 * M_PI / 180) + 0.000060963424610 * x * sin((x + 5) * 2 * M_PI / 180));
        };

        NBETA = [](double x) -> double
        {
            return (0.030064 + 0.026572) / 2 * (x / 270 * (0.030064 / 0.026572 - 1) / 2 + 1) * sin(x * 2 * M_PI / 180);
        };

        NGAMMA = [](double x) -> double
        {
            return (-0.001219952576676 * x - 0.000000084898394 * pow(x, 2) - 0.000000269361545 * pow(x, 3));
        };
    }

    XHL = XFN_TOT + XBETA(factorRad2Deg * beta_force) * dim_drift + XGAMMA(factorRad2Deg * gamma_c) * dim_yaw;
    YHL = YBETA(factorRad2Deg * beta_force) * dim_drift + YGAMMA(factorRad2Deg * gamma_c) * dim_yaw;
    NHL = NBETA(factorRad2Deg * beta_force) * dim_drift * stat_opts.Lpp + NGAMMA(factorRad2Deg * gamma_c) * dim_yaw * stat_opts.Lpp;
    Eigen::Vector3d tau3_hull = {XHL, YHL, NHL};

    Eigen::Vector3d vw_r = R * Eigen::Vector3d(v_w(0), v_w(1), 0) - nu3;
    double V_w = sqrt(pow(abs(vw_r(0)) + small_val, 2) + pow(abs(vw_r(1)) + small_val, 2));
    double alpha_w = atan2(vw_r(1) + small_val * (vw_r(1) >= 0 ? 1 : -1), vw_r(0) + small_val);

    auto CFx_w = [&stat_opts](double x) -> double
    { return linear_interpolate(stat_opts.alpha_w, stat_opts.CFx_w, x); };
    auto CFy_w = [&stat_opts](double x) -> double
    { return linear_interpolate(stat_opts.alpha_w, stat_opts.CFy_w, x); };
    auto CMz_w = [&stat_opts](double x) -> double
    { return linear_interpolate(stat_opts.alpha_w, stat_opts.CMz_w, x); };

    Eigen::Vector3d tau3_wind;
    tau3_wind << stat_opts.rho_w / 2 * pow(V_w, 2) * stat_opts.AF_w * CFx_w(alpha_w) / 1000,
        stat_opts.rho_w / 2 * pow(V_w, 2) * stat_opts.AL_w * CFy_w(alpha_w) / 1000,
        stat_opts.rho_w / 2 * pow(V_w, 2) * stat_opts.AL_w * stat_opts.Lpp_w * CMz_w(alpha_w) / 1000;

    Eigen::Matrix3d Crb;
    Crb << 0, 0, -params(3) * (stat_opts.CoG(0) * nu3r(2) + nu3(1)),
        0, 0, -params(3) * (stat_opts.CoG(1) * nu3r(2) - nu3(0)),
        0, 0, 0;

    Eigen::Vector3d F_D;
    F_D << params(4) * nu3r(0), params(5) * nu3r(1), params(6) * nu3r(2);

    // Eigen::Matrix3d CA = Eigen::Matrix3d::Zero();
    double ur = nu3r(0);
    double vr = nu3r(1);
    double rr = nu3r(2);

    Eigen::Matrix3d CA;
    CA << 0, 0, 0 * vr + 0 * rr,
        0, 0, 0 * ur,
        0 * vr + 0 * rr, 0, 0;

    Results res;
    res.tau3_CC = -(Crb * nu3 + CA * nu3r + F_D);
    res.tau3_hull = tau3_hull * 1000;
    res.tau3_wind = tau3_wind * 1000;
    return res;
}

Eigen::VectorXd vessel_model_ident(
    const Eigen::VectorXd &x_scaled, const Eigen::VectorXd &u_in_scaled,
    const Vessel<double> &stat_opts, const Eigen::VectorXd &all_other_arguments_vectorized)
{
    // Extracting required values from all_other_arguments_vectorized
    Eigen::Vector2d v_c(all_other_arguments_vectorized[0], all_other_arguments_vectorized[1]);
    Eigen::Vector2d v_w(all_other_arguments_vectorized[2], all_other_arguments_vectorized[3]);
    Eigen::VectorXd F_residual = all_other_arguments_vectorized.segment(4, 3); // Assuming F_residual has 3 components
    Eigen::VectorXd params = all_other_arguments_vectorized.segment(7, 15);    // Adjust as needed
    Eigen::VectorXd scaling = all_other_arguments_vectorized.tail(3);          // Assuming the last 4 values are for scaling

    // Unscaled versions
    Eigen::VectorXd u_in = unscale_input(u_in_scaled, scaling);
    Eigen::VectorXd x = unscale_state(x_scaled, scaling);

    double psi = x[2];
    Eigen::Vector3d nu3(x[3], x[4], x[5]);

    Eigen::Matrix3d R;
    R << cos(psi), -sin(psi), 0,
        sin(psi), cos(psi), 0,
        0, 0, 1;

    Results results = vessel_forces_ident(x, v_c, v_w, params, stat_opts);
    Eigen::Vector3d tau3 = results.tau3_CC + Eigen::Vector3d(params[12] * u_in[0], params[13] * u_in[1], params[14] * u_in[2]) + results.tau3_hull + results.tau3_wind + Eigen::Vector3d(F_residual[0], F_residual[1], F_residual[2]);

    Eigen::Matrix3d M;
    M << params[0], 0, 0,
        0, params[1], 0,
        0, 0, params[2];
    Eigen::Vector3d nu3_dot = M.inverse() * tau3;
    Eigen::Vector3d eta3_dot = R * nu3;

    double data[] = {eta3_dot[0], eta3_dot[1], eta3_dot[2], nu3_dot[0], nu3_dot[1], nu3_dot[2], 0};
    Eigen::VectorXd x_dot = Eigen::VectorXd::Map(data, 7);
    Eigen::VectorXd x_dot_scaled = scale_state_rate(x_dot, scaling);

    return x_dot_scaled;
}

Eigen::VectorXd vessel_model_integrator(
    const Eigen::VectorXd &x_start, const Eigen::VectorXd &u_dem,
    const Vessel<double> &stat_opts, const Eigen::VectorXd &all_other_arguments_vectorized,
    const std::string &integrator_scheme = "heun")
{
    Eigen::VectorXd Scaling = all_other_arguments_vectorized.tail(3); // Assuming the last 3 values are for Scaling
    Eigen::VectorXd dt(1);
    dt(0) = 1 / Scaling[1]; // Assuming scaling is 1 for simplicity. Adjust as needed.

    Eigen::VectorXd k1, k2, k3, k4;
    Eigen::VectorXd x_end;

    if (integrator_scheme == "rk4")
    {
        k1 = vessel_model_ident(x_start, u_dem, stat_opts, all_other_arguments_vectorized);
        Eigen::VectorXd delta = k1 * dt / 2;
        Eigen::VectorXd x_temp = x_start + delta;
        k2 = vessel_model_ident(x_temp, u_dem, stat_opts, all_other_arguments_vectorized);

        x_temp = x_start + k2 * dt / 2;
        k3 = vessel_model_ident(x_temp, u_dem, stat_opts, all_other_arguments_vectorized);

        x_temp = x_start + k3 * dt;
        k4 = vessel_model_ident(x_temp, u_dem, stat_opts, all_other_arguments_vectorized);

        x_end = x_start + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6;
        x_end[2] = ssa(x_end[2]); // Assuming ssa function is defined elsewhere
    }
    else if (integrator_scheme == "euler")
    {
        Eigen::VectorXd x_dot = vessel_model_ident(x_start, u_dem, stat_opts, all_other_arguments_vectorized);

        x_end = x_start + x_dot * dt;
        x_end[2] = ssa(x_end[2]);
    }
    else if (integrator_scheme == "heun")
    {
        Eigen::VectorXd x_dot_start = vessel_model_ident(x_start, u_dem, stat_opts, all_other_arguments_vectorized);
        Eigen::VectorXd x_temp = x_start + x_dot_start * dt;

        Eigen::VectorXd x_dot_end = vessel_model_ident(x_temp, u_dem, stat_opts, all_other_arguments_vectorized);

        x_end = x_start + (x_dot_start + x_dot_end) / 2 * dt;
        x_end[2] = ssa(x_end[2]);
    }

    return x_end;
}



template <typename Type>
std::array<Type, number_of_states> vessel_ode_gen_model(
    const std::array<Type, number_of_states>(&state_vector),
    const std::array<Type, number_of_inputs> &input_array,
    const Vessel<Type> &vessel,
    const std::vector<Type> &all_other_arguments_vectorized)
{
    // Convert state_vector and input_array to Eigen::VectorXd
    Eigen::VectorXd x_start(number_of_states);
    Eigen::VectorXd u_dem(number_of_inputs);

    for (size_t i = 0; i < number_of_states; i++)
    {
        x_start(i) = state_vector[i];
    }

    for (size_t i = 0; i < number_of_inputs; i++)
    {
        u_dem(i) = input_array[i];
    }

    // Convert all_other_arguments_vectorized to Eigen::VectorXd
    Eigen::VectorXd all_other_args_Eigen(all_other_arguments_vectorized.size());
    for (size_t i = 0; i < all_other_arguments_vectorized.size(); i++)
    {
        all_other_args_Eigen(i) = all_other_arguments_vectorized[i];
    }

    // Call vessel_model_integrator
    Eigen::VectorXd x_end;
    x_end = vessel_model_integrator(x_start, u_dem, vessel, all_other_args_Eigen);

    // Convert the result back to an std::array
    // std::array<Type, number_of_states> result;
    for (size_t i = 0; i < number_of_states; i++)
    {
        result[i] = x_end(i);
    }

    return result;
}

template <typename Type>
void keyboard_input_handler(const std::array<Type, number_of_states>(&state_vector),
     std::array<Type, number_of_inputs> &tau3_in,
    const Vessel<Type> &vessel,
    const std::vector<Type> &all_other_arguments_vectorized)
{   
    std::cout << "Keyboard Input Handler Instructions:\n";
    std::cout << " - Press '1', '2', '3' to increment the corresponding element in tau3_in.\n";
    std::cout << " - Press 's' to toggle increment/decrement mode.\n";
    std::cout << " - Press 'a' to cycle through different increment value sets.\n";
    std::cout << " - Press 'q' to quit the handler.\n\n";
    
    // Define three sets of increment values for each element
    const int increment_values[3] = {1000, 500, 200};          // Modify these values as per your needs
    const int alternate_increment_values_1[3] = {100, 50, 20}; // Modify these values as per your needs
    const int alternate_increment_values_2[3] = {10, 5, 2};    // Add a third set of values

    int alt_state = 0; // 0 for default, 1 for first alternate, 2 for second alternate
    bool shift_pressed = false;

    while (true)
    {
        char key = my_getch(); // Read the key

        // Use 's' key to simulate Shift and 'a' key to cycle through Alt states
        if (key == 's')
        {
            shift_pressed = !shift_pressed; // Toggle shift_pressed
        }
        else if (key == 'a')
        {
            alt_state = (alt_state + 1) % 3; // Cycle through alt states
        }
        else
        {
            int increment_index = -1;
            switch (key)
            {
            case '1':
                increment_index = 0;
                break;
            case '2':
                increment_index = 1;
                break;
            case '3':
                increment_index = 2;
                break;
                // Add more cases as needed
            case 'q':
                return;
            }

            if (increment_index != -1)
            {
                int increment_value;
                switch (alt_state)
                {
                case 1:
                    increment_value = alternate_increment_values_1[increment_index];
                    break;
                case 2:
                    increment_value = alternate_increment_values_2[increment_index];
                    break;
                default:
                    increment_value = increment_values[increment_index];
                }
                increment_value *= (shift_pressed ? -1 : 1);
                tau3_in[increment_index] += increment_value;

                // Output status
                std::cout << "Element " << increment_index << " modified by " << increment_value << ". New value: " << tau3_in[increment_index] << std::endl;
                std::cout << "Current values: ";
                for (const auto& value : tau3_in) {
                    std::cout << value << " ";
                }
                result = vessel_ode_gen_model(state_vector, tau3_in, vessel, all_other_arguments_vectorized);

                // Print the results
                std::cout << "Resulting state vector:\n";
                for (const auto &val : result)
                {
                    std::cout << val << " ";
                }
                std::cout << std::endl<< std::endl; 
            }
        }
    }
}
int main()
{
    // ... [Initialize variables as in your provided code] ...
    // Initialize test data for state vector
    std::array<double, number_of_states> state_vector = {300.0, 50.0, 1.0, 1.5, 0.5, 0.05, 0};

    // Initialize test data for input array
    std::array<double, number_of_inputs> input_array = {10000.0, 10000.0, 100000.0};

    // Initialize test data for vessel
    // Eigen::VectorXd Scaling(3);
    // Scaling << 1.0, 1.0, 1.0;
    Vessel<double> vessel;

    // Initialize test data for other arguments
    std::vector<double> all_other_arguments_vectorized = {
        0.0, 0.0,      // v_c
        0.0, 0.0,      // v_w
        0.0, 0.0, 0.0, // F_res
        // Additional parameters, example values
        187100.511775658, 303030.866069127, 57378607.9640273,
        198891.964122127, 771.277192428326, 1775.78135270359,
        7746.54537243618, 0.01, 0, 0, 0, 0,
        0.460051055484453, 0.460051055484453, 0.460051055484453,
        // Scaling
        1.0, 1.0, 1.0};

    // Call the wrapper function
    result = vessel_ode_gen_model(state_vector, input_array, vessel, all_other_arguments_vectorized);

    // Print the results
    std::cout << "Resulting state vector:\n";
    for (const auto &val : result)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl << std::endl;
    keyboard_input_handler(state_vector, input_array, vessel, all_other_arguments_vectorized);

    return 0;
}
