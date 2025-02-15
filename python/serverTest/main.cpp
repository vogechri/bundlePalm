// Original Code from http://kr.zeromq.org/cpp:hwserver
// kongineer.com


#include <zmq.hpp>
#include <string>
#include <iostream>
#include "generated/proto/test.pb.h"
#include <google/protobuf/message_lite.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include<Eigen/Dense>
#include <Eigen/Eigenvalues> 

#include "ceres/ceres.h"
#include "ceres/normal_prior.h"
#include "ceres/rotation.h"

using Eigen::SparseMatrix;
using Eigen::VectorXi;

#ifdef _WIN32
#include<Windows.h>
#elif defined __unix__
#include <unistd.h>
#endif

#define THROW_IF(cond)                                                    \
  do {                                                                    \
    if (cond) [[unlikely]] {                                              \
      std::ostringstream error;                                           \
      error << "Check failed in " << __FILE__ << ":" << __LINE__ << " - " \
            << #cond;                                                     \
      throw std::runtime_error(error.str());                              \
    }                                                                     \
  } while (false)

template <typename Proto>
Proto ParseProto(const std::string& serialized) {
  Proto proto;
  THROW_IF(!proto.ParseFromString(serialized));
  return proto;
}

void delay(int msec)
{
#ifdef _WIN32
    Sleep(msec);
#elif defined __unix__
    sleep(msec);
#endif
}

struct ProxStepPrior {
    ProxStepPrior() {}
    // ProxStepPrior(int block_): block(block_) {}
    // |sqrt(S) * (x-s) |^2 = (x-s)^T S (x-s)
    template <typename T>
    bool operator()(const T *const matBlock,
                    const T *const camera,
                    const T *const camera_s,
                    T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        //   T d[9];
        //   d[0] = camera[0] - camera_s[0];
        //   d[1] = camera[1] - camera_s[1];
        //   d[2] = camera[2] - camera_s[2];
        //   d[3] = camera[3] - camera_s[3];
        //   d[4] = camera[4] - camera_s[4];
        //   d[5] = camera[5] - camera_s[5];
        //   d[6] = camera[6] - camera_s[6];
        //   d[7] = camera[7] - camera_s[7];
        //   d[8] = camera[8] - camera_s[8];

        Eigen::Matrix<T, 9, 1> d = Eigen::Map<const Eigen::Matrix<T, 9, 1>>(camera) - Eigen::Map<const Eigen::Matrix<T, 9, 1>>(camera_s);
        Eigen::Map<Eigen::Matrix<T, 9, 1>> residualsVector(residuals);
        residualsVector = Eigen::Map<const Eigen::Matrix<T, 9, 9>>(matBlock) * d;

        // The error is the difference between the predicted and observed position.
        //   residuals[0] = predicted_x - observed_x;
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create() {
        return (new ceres::AutoDiffCostFunction<ProxStepPrior, 9, 81, 9, 9>(new ProxStepPrior()));
    }
};

struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = -p[0] / p[2]; // that means focal length is flipped.
    T yp = -p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
        new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};

template<int N>
void BlockSqrt(SparseMatrix<double, Eigen::RowMajor>& mat) {
    int numrows = mat.rows();
    THROW_IF(mat.rows() != mat.cols());
    //THROW_IF(mat.);
    double* values = mat.valuePtr();
    //std::cout << "before  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
    for (int i = 0; i < numrows / N; i++) {
        auto mat9x9 = Eigen::Map< Eigen::Matrix<double,N,N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
        //std::cout << "before "<< mat9x9 << " \n";

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,N,N> > eigensolver;
        eigensolver.computeDirect(mat9x9, Eigen::DecompositionOptions::ComputeEigenvectors);
        //VPQ_EXPECT_EQ(eigensolver.info(), Eigen::Success);
  
        // SqrtCovEigenValues are sorted in decreasing order.
        Eigen::Vector<double, N> sqrtEigenValues = eigensolver.eigenvalues().cwiseSqrt();//.cwiseMax(lowerBoundSquared).cwiseSqrt().cwiseInverse();
        mat9x9 = eigensolver.eigenvectors() * sqrtEigenValues.asDiagonal() * eigensolver.eigenvectors().transpose();
        //std::cout << "after  "<< mat9x9.transpose() * mat9x9 << " \n";
    }
    //std::cout << "after  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
}

template<int N>
void BlockInverse(SparseMatrix<double, Eigen::RowMajor>& mat) {
    int numrows = mat.rows();
    THROW_IF(mat.rows() != mat.cols());
    //THROW_IF(mat.);
    double* values = mat.valuePtr();
    //std::cout << "before  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
    for (int i = 0; i < numrows / N; i++) {
        auto mat9x9 = Eigen::Map< Eigen::Matrix<double,N,N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
        std::cout << "before "<< mat9x9 << " \n";
        mat9x9 = mat9x9.inverse();
        std::cout << "after "<< mat9x9 << " \n";        
    }
    //std::cout << "after  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
}

int main() {
    // Initialize the context
    zmq::context_t context(1);

    // Create a socket of type REP (reply)
    zmq::socket_t socket(context, ZMQ_REP);

    // Bind the socket to a TCP address
    std::cout << "Starting the server on port 5555..." << std::endl;
    socket.bind("tcp://*:5555");

    // Those are permanent, variables can change.
    int numCameras = 0;
    int numLandmarks = 0;

    std::vector<double> cameras_s;
    std::vector<double> cameras;
    std::vector<double> landmarks;
    std::vector<double> stepSize;

    ceres::Problem problem;
    ceres::Solver::Options options;

    while (true) {
        zmq::message_t request;

        // Wait for the next request from a client
        socket.recv(&request);
        const std::string received_message(static_cast<char*>(request.data()), request.size());

        request_proto request_p;
        if (!request_p.ParseFromString(received_message)) {
            std::cout << "Received: " << received_message << std::endl;
            // Simulate some work
            // delay(1);
            // Send a reply back to the client
            std::cout << " send back \n";
            std::string reply_message = "Hi from ZeroMQ C++ Server";
            zmq::message_t reply(reply_message.size());
            memcpy(reply.data(), reply_message.data(), reply_message.size());
            socket.send(reply, zmq::send_flags::none);
            continue;
        }

        switch(request_p.options_case()) {
            case request_proto::OptionsCase::kCameras :
            {
                //std::cout << "request_proto::OptionsCase::kCameras" << std::endl;
                camera_proto cams = request_p.cameras(); // we get an update for the cameras only -- update buffer, run its iterations.
                //if (cams.ParseFromString(received_message)) 
                {
                    THROW_IF(cams.cameras_size() != cameras.size());
                    int id=0; // fill existing buffer
                    for(const float& v : cams.cameras()) {
                        cameras[id++] = v;
                    }
                    // solve once more
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    std::cout << summary.FullReport() << "\n";
                    std::cout << "\nMycost: " << summary.final_cost * 2 << "\n";

                    // Send solution back!
                    id = 0;
                    for(const double& v : cameras) {
                        cams.set_cameras(id++, static_cast<float>(v));
                    }

                    // // Send solution back!, actually cameras shoudl be ok?
                    // solution_proto sol;
                    // for(const double& v : cameras) {
                    //     sol.add_cameras(static_cast<float>(v));
                    // }
                    // for(const double& v : landmarks) {
                    //     sol.add_landmarks(static_cast<float>(v));
                    // }
                    // sol.SerializeToString(&encoded_msg);

                    std::string encoded_msg;
                    cams.SerializeToString(&encoded_msg);
                    zmq::message_t reply(encoded_msg.size());
                    // Cast? this is WASTEful
                    memcpy ((void *) reply.data(), encoded_msg.c_str(), encoded_msg.size());
                    // publisher.send(zmq_msg);
                    socket.send(reply, zmq::send_flags::none);
                }
            break;
            }

            case request_proto::OptionsCase::kUpdate :
            {
                //std::cout << "request_proto::OptionsCase::kUpdate" << std::endl;
                prox_cluster_proto update = request_p.update(); // we get an update for the cameras only -- update buffer, run its iterations.
                //if (cams.ParseFromString(received_message)) 
                {
                    THROW_IF(update.cameras_size() != cameras.size());
                    int id=0; // fill existing buffer
                    for(const float& v : update.cameras()) {
                        cameras[id++] = v;
                    }
                    // solve once more
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    std::cout << summary.FullReport() << "\n";
                    std::cout << "\nMycost: " << summary.final_cost * 2 << "\n";

                    // // Send solution back!, actually cameras shoudl be ok?
                    // solution_proto sol;
                    // for(const double& v : cameras) {
                    //     sol.add_cameras(static_cast<float>(v));
                    // }
                    // for(const double& v : landmarks) {
                    //     sol.add_landmarks(static_cast<float>(v));
                    // }
                    // sol.SerializeToString(&encoded_msg);

                    // Send solution back!
                    camera_proto cams;
                    id = 0;
                    for(const double& v : cameras) {
                        cams.set_cameras(id++, static_cast<float>(v));
                    }

                    std::string encoded_msg;
                    cams.SerializeToString(&encoded_msg);
                    zmq::message_t reply(encoded_msg.size());
                    // Cast? this is WASTEful
                    memcpy ((void *) reply.data(), encoded_msg.c_str(), encoded_msg.size());
                    // publisher.send(zmq_msg);
                    socket.send(reply, zmq::send_flags::none);
                }
            break;
            }

            // if we get program we setup new program. if we get cam & prox we update cams (?) and prox term only! do one more it, etc.
            case request_proto::OptionsCase::kProgram : {
            std::cout << "request_proto::OptionsCase::kProgram" << std::endl;
            program_proto pro = request_p.program(); 
            {
                numCameras = pro.cameras_size() / 9;
                numLandmarks = pro.landmarks_size() / 3;
                cameras.clear();
                cameras.reserve(pro.cameras_size());
                for(const float& v : pro.cameras()) {
                    cameras.push_back(v);
                }
                landmarks.clear();
                landmarks.reserve(pro.landmarks_size());
                for(const float& v : pro.landmarks()) {
                    landmarks.push_back(v);
                }
                cameras_s.clear();
                cameras_s.reserve(pro.cameras_size());
                for(const float& v : pro.cameras()) {
                    cameras_s.push_back(v);
                }
                stepSize.clear(); // all 0 to ensure jacobian is reasonable.
                stepSize.resize(9 * pro.cameras_size(), 0);

                // setup problem again.
                problem = ceres::Problem(); // overwrite ..?
                int ceres_id = 0;
                // try to make my life simpler .. make ids match: OK. cameras are 1st landmark second.
                for (int i = 0; i < cameras.size(); i += 9) {
                    problem.AddParameterBlock(&cameras[i], 9);
                }
                for (int i = 0; i < landmarks.size(); i += 3) {
                    problem.AddParameterBlock(&landmarks[i], 3);
                }
                for (int i = 0; i < cameras_s.size(); i += 9) {
                    problem.AddParameterBlock(&cameras_s[i], 9);
                    problem.SetParameterBlockConstant(&cameras_s[i]);
                }
                std::cout << "Parameter blocks added ow step\n";
                for (int i = 0; i < stepSize.size(); i += 81) {
                    problem.AddParameterBlock(&stepSize[i], 81);
                    problem.SetParameterBlockConstant(&stepSize[i]);
                }
                std::cout << "All Parameter blocks added\n";
                // SetParameterBlockVariable -> could set cameras constant, could use for 's'.
                for (int i = 0; i < pro.observations_size() / 2; ++i) {
                    // Each Residual block takes a point and a camera as input and outputs a 2
                    // dimensional residual. Internally, the cost function stores the observed
                    // image location and compares the reprojection against the observation.

                    // *fMessage.mutable_samples() = {fData.begin(), fData.end()}; // copies? OMG

                    // google::protobuf::RepeatedField<float> data(fData.begin(), fData.end());
                    // fMessage.mutable_samples()->Swap(&data);
                    // Parse 1 by 1 -- omg this is bad.
                    ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
                        pro.observations(2 * i + 0), pro.observations(2 * i + 1));
                    problem.AddResidualBlock(cost_function,
                                            nullptr /* squared loss */,
                                            &(cameras[9 * pro.cam_id(i)]),
                                            &(landmarks[3 * pro.lm_id(i)]));
                }
                std::cout << "Added Residual blocks 1\n";

                for (int cam_id = 0 ; cam_id < numCameras; ++cam_id) {
                    //double* values = JpJ.valuePtr();
                    // ceres::Matrix block9x9 = Eigen::Map< Eigen::Matrix<double,9,9> > (&(stepSize[cam_id * 9*9]));
                    // ceres::Vector block9 = Eigen::Map< Eigen::Matrix<double,9,1> > (&(cameras_s[cam_id * 9]));

                    //ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ceres::NormalPrior, 9, 9>(new ceres::NormalPrior(block9x9, block9));
                    ceres::CostFunction* cost_function = ProxStepPrior::Create();
                    //std::cout << "ptr " << &(stepSize [81 * cam_id]) << " " << &(cameras  [9 * cam_id]) << " " << &(cameras_s[9 * cam_id]) << "\n";
                    problem.AddResidualBlock(cost_function,
                        nullptr /* squared loss */,
                        &(stepSize [81 * cam_id]), // it thinks this block size is 9, not 81.
                        &(cameras  [9 * cam_id]),
                        &(cameras_s[9 * cam_id]));
                }
                std::cout << "Added Residual blocks stepsize\n";

            //     Received: Hi from Python ZeroMQ Client
            //     send back
            //    request_proto::OptionsCase::kProgram
            //    Parameter blocks added
            //    Finished eval problem
            //    iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
            //       0  8.509128e+05    0.00e+00    8.57e+06   0.00e+00   0.00e+00  1.00e+04        0    3.18e-01    3.48e-01
            //       1  5.466895e+04    7.96e+05    1.58e+06   0.00e+00   9.81e-01  3.00e+04        1    3.54e-01    7.02e-01
               
            //    Solver Summary (v 2.2.0-eigen-(3.4.0)-lapack-suitesparse-(5.10.1)-eigensparse)
               
            //                                         Original                  Reduced
            //    Parameter blocks                         7874                     7825
            //    Parameters                              24210                    23769
            //    Residual blocks                         31892                    31892
            //    Residuals                               64127                    64127
               
            //    Minimizer                        TRUST_REGION
               
            //    Dense linear algebra library            EIGEN
            //    Trust region strategy     LEVENBERG_MARQUARDT
            //                                            Given                     Used
            //    Linear solver                     DENSE_SCHUR              DENSE_SCHUR
            //    Threads                                     8                        8
            //    Linear solver ordering              AUTOMATIC                  7776,49
            //    Schur structure                         2,3,9                    2,3,9
               
            //    Cost:
            //    Initial                          8.509128e+05
            //    Final                            5.466895e+04
            //    Change                           7.962438e+05
               
            //    Minimizer iterations                        2
            //    Successful steps                            2
            //    Unsuccessful steps                          0
               
            //    Time (in seconds):
            //    Preprocessor                         0.030355
               
            //      Residual only evaluation           0.001627 (1)
            //      Jacobian & residual evaluation     0.648722 (2)
            //      Linear solver                      0.016618 (1)
            //    Minimizer                            0.671797
               
            //    Postprocessor                        0.001850
            //    Total                                0.704002
               
            //    Termination:                   NO_CONVERGENCE (Maximum number of iterations reached. Number of iterations: 1.)
               
               
            //    Mycost: 109338
//                 iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
//    0  5.466894e+04    0.00e+00    1.58e+06   0.00e+00   0.00e+00  1.00e+04        0    3.55e-01    3.98e-01
//    1  4.031896e+04    1.43e+04    2.72e+05   0.00e+00   9.45e-01  3.00e+04        1    3.64e-01    7.63e-01

// Solver Summary (v 2.2.0-eigen-(3.4.0)-lapack-suitesparse-(5.10.1)-eigensparse)

//                                      Original                  Reduced
// Parameter blocks                         7874                     7825
// Parameters                              24210                    23769
// Residual blocks                         31892                    31892
// Residuals                               64127                    64127

// Minimizer                        TRUST_REGION

// Dense linear algebra library            EIGEN
// Trust region strategy     LEVENBERG_MARQUARDT
//                                         Given                     Used
// Linear solver                     DENSE_SCHUR              DENSE_SCHUR
// Threads                                     8                        8
// Linear solver ordering              AUTOMATIC                  7776,49
// Schur structure                         2,3,9                    2,3,9

// Cost:
// Initial                          5.466894e+04
// Final                            4.031896e+04
// Change                           1.434997e+04

// Minimizer iterations                        2
// Successful steps                            2
// Unsuccessful steps                          0

// Time (in seconds):
// Preprocessor                         0.043743

//   Residual only evaluation           0.001643 (1)
//   Jacobian & residual evaluation     0.694730 (2)
//   Linear solver                      0.017152 (1)
// Minimizer                            0.719195

// Postprocessor                        0.001677
// Total                                0.764615
// Mycost: 80637.9
                // 1st get Jacobian(s):
                ceres::Problem::EvaluateOptions evalOptions;
                evalOptions.apply_loss_function = true;
                evalOptions.num_threads = 1;
                ceres::CRSMatrix jacobian;
                std::vector<double> residuals;
                double cost;
                problem.Evaluate(evalOptions, &cost, &residuals, nullptr, &jacobian);
                const size_t numUnknowns = jacobian.num_cols;
                std::cout << "Finished eval problem \n";
                // Now. I need JpTJp, hence.

                const int relevantRows = jacobian.num_rows - 9 * numCameras;

                SparseMatrix<double, Eigen::RowMajor> Jp(relevantRows, 9 * numCameras);
                SparseMatrix<double, Eigen::RowMajor> Jl(relevantRows, 3 * numLandmarks);
                Jp.reserve(VectorXi::Constant(2 * relevantRows, 9));
                Jl.reserve(VectorXi::Constant(2 * relevantRows, 3));
                // JP.setFromTriplets(coefficients.begin(), coefficients.end());
                for (Eigen::Index r = 0; r < relevantRows; ++r) {
                //for (Eigen::Index r = 0; r < 1000; ++r) {
                    int lm_id = pro.lm_id(r/2);
                    int cam_id = pro.cam_id(r/2);
                    //std::cout << r << ":";
                    Eigen::Index idx = jacobian.rows[r];
                    const Eigen::Index c = jacobian.cols[idx]; // index of variable.
                    for (int i = 0; i < 9 && idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];++idx,++i) {
                        // if (9 * cam_id + i != jacobian.cols[idx])
                        //     std::cout << 9 * cam_id + i << " = " << jacobian.cols[idx] << " | ";
                        Jp.insert(r, 9 * cam_id + i) = jacobian.values[idx];
                    }
                    for (int i = 0; i < 3 && idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];++idx, ++i) {
                        // if (cameras.size() + 3 * lm_id + i != jacobian.cols[idx])
                        //     std::cout << 3 * lm_id + i << " = " << jacobian.cols[idx];
                        Jl.insert(r, 3 * lm_id + i) = jacobian.values[idx];
                    }
                }
                Jp.makeCompressed();
                Jl.makeCompressed();

                SparseMatrix<double, Eigen::RowMajor> JpJ(9 * numCameras, 9 * numCameras);
                SparseMatrix<double, Eigen::RowMajor> JlJ(3 * numLandmarks, 3 * numLandmarks);
                JpJ.reserve(VectorXi::Constant(9 * numCameras, 9));
                JlJ.reserve(VectorXi::Constant(3 * numLandmarks, 3));

                JpJ = Jp.transpose() * Jp;
                JlJ = Jl.transpose() * Jl;
                auto JpJ_diag = JpJ.diagonal().array();
                // maybe block diag as well.
                double be = pro.be(); // 1e-4;
                JpJ.diagonal().array() += be * JpJ.diagonal().array();

                BlockSqrt<9>(JpJ);// need templated fct.

                // Todo: I might need to have the (x-s)Step(x-s) cost function in the program already.
                // Update Step is an issue. I would need to remove residual blocks ..
                // So better is:
                // Own normalprior, add residualblocks for JpJ: as 9x9 blocks instead of here.
                // At each step. set S == 0. Compute J. update S, do a step. 

                /////////////////////////////////////////////////////
                // This cannot work at all. The problem: 
                // JpJ and s change at each step.
                // Both MUST be variable blocks and stored in memory.
#ifdef __notHere
                for (int cam_id =0 ; cam_id < numCameras; ++cam_id) {
                    double* values = JpJ.valuePtr();
                    ceres::Matrix block9x9 = Eigen::Map< Eigen::Matrix<double,9,9> > (&(values[cam_id * 9*9]));
                    ceres::Vector block9 = Eigen::Map< Eigen::Matrix<double,9,1> > (&(cameras_s[cam_id * 9]));

                    //ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ceres::NormalPrior, 9, 9>(new ceres::NormalPrior(block9x9, block9));
                    ceres::CostFunction* cost_function = new ceres::ProxStepPrior();
                    problem.AddResidualBlock(cost_function,
                        nullptr /* squared loss */,
                        &(JpJblock [81 * cam_id]),
                        &(cameras  [9 * cam_id]),
                        &(cameras_s[9 * cam_id]));
                    
                    // ceres::CostFunction* cost_function = new ceres::NormalPrior(block9x9, block9);
                    // problem.AddResidualBlock(cost_function,
                    //     nullptr /* squared loss */,
                    //     &(cameras[9 * cam_id]));
                }
#else
// instead reset variable block(s) JpJ and s to sqrt(Stepsize)
                double* values = JpJ.valuePtr();
                for (int id = 0; id < 81 * numCameras; ++id) {
                    stepSize[id] = values[id]; // = 1000 -> diufferent cost: so ok
                }
#endif

                // Solve
                // Make Ceres automatically detect the bundle structure. Note that the
                // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
                // for standard bundle adjustment problems.
                options.linear_solver_type = ceres::DENSE_SCHUR; // SPARSE_SCHUR;// same
                //options.linear_solver_type = ITERATIVE_SCHUR; // same ceres::CGNR;//
                //options.linear_solver_type = ceres::CGNR;
                //options.linear_solver_type = ceres::DENSE_QR; // SHIT
                //options.max_linear_solver_iterations = 0;
                options.num_threads = 8; // ok maybe it is this what makes it slow. Problem: single cpu -> still slow / bottleneck.
                options.minimizer_progress_to_stdout = true;
                options.max_num_iterations = std::max(0, std::min(10, pro.iterations()));
                // options.preconditioner_type = ceres::IDENTITY; // Sucks if CGNR of course.
                //options.preconditioner_type = ceres::JACOBI; // CGNR -> jacobi anyway.

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                std::cout << summary.FullReport() << "\n";
                std::cout << "\nMycost: " << summary.final_cost * 2 << "\n";

                // Send solution back!
                int id = 0;
                //std::cout << pro.cameras_size() << " == " << cameras.size() << std::endl;
                for(const double& v : cameras) {
                    pro.set_cameras(id++, static_cast<float>(v));
                }
                id = 0;
                //std::cout << pro.landmarks_size() << " == " << landmarks.size() << std::endl;
                for(const float& v : landmarks) {
                    pro.set_landmarks(id++, v);
                }

                //*pro.mutable_cameras() = {cameras.begin(), cameras.end()}; // float vs double.           
                // Send Jacobian! back -- lookup how.

#define __write__
    #ifdef __write__
            {
                std::cout << "Write Jac\n";
                ceres::Problem::EvaluateOptions evalOptions;
                evalOptions.apply_loss_function = true;
                evalOptions.num_threads = 1;

                ceres::CRSMatrix jacobian; // likely unordered as shit.
                problem.Evaluate(evalOptions, nullptr, nullptr, nullptr, &jacobian);
                const size_t numUnknowns = jacobian.num_cols;

                // 12 per row, 9 cams, 3 landmark indices. problem: order is as follows by cam_id and lm_id.
                // J_pose is given by going over jac and id. 
                // J_pose is n_res x 9 * # cams
                // J_land is n_res x 3 * # land
                for (Eigen::Index r = 0; r < 100; ++ r) { //jacobian.num_rows; ++r) {
                    std::cout << r << ": ";
                    for (Eigen::Index idx = jacobian.rows[r]; idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];
                        ++idx) {
                    const Eigen::Index c = jacobian.cols[idx];
                    std::cout << c << ", ";// << " = " << jacobian.values[idx] << " | ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
    #endif

                // // Insert the gradient per residual into the dense jacobian matrix.
                // ceres::Matrix denseJacobian(jacobian.num_rows, jacobian.num_cols);
                // denseJacobian.setZero();
                // for (Eigen::Index r = 0; r < jacobian.num_rows; ++r) {
                //     for (Eigen::Index idx = jacobian.rows[r]; idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];
                //         ++idx) {
                //     const Eigen::Index c = jacobian.cols[idx];
                //     denseJacobian(r, c) = jacobian.values[idx];
                //     }
                // }


                // Todo: version setup program, compute Jacobians with 0 step.
                // Extend program with additional cost / modify ceres code?
                // Maybe solve by hand / solve by ceres. compare both.
                // Add 

    #ifdef _send_string_
                std::string reply_message = "C++ Server received program";
                zmq::message_t reply(reply_message.size());
                socket.send(reply, zmq::send_flags::none);
    #else
                // Instead send the result back as programm again.
                std::string encoded_msg;
                pro.SerializeToString(&encoded_msg);
                zmq::message_t reply(encoded_msg.size());
                // Cast? this is WASTEful
                memcpy ((void *) reply.data(), encoded_msg.c_str(), encoded_msg.size());
                // publisher.send(zmq_msg);
                socket.send(reply, zmq::send_flags::none);
    #endif
            }

            break;
            }

            default: {
            break;}
        }

    } // end while 

    return 0;
}

// Example: send c++ proto.
        // std::string encoded_msg;
        // RL::DataSet msg;
        // msg.set_count(i);
        // msg.add_joint_position(1.1);
        // msg.add_joint_position(2.1);
        // msg.add_joint_velocity(-1.1);
        // msg.add_joint_velocity(-2.1);

        // msg.SerializeToString(&encoded_msg);

        // zmq::message_t zmq_msg(encoded_msg.size());
        // memcpy ((void *) zmq_msg.data(), encoded_msg.c_str(),
        //         encoded_msg.size());
        // publisher.send(zmq_msg);