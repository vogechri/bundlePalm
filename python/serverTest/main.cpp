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

    std::vector<double> cameras;
    std::vector<int> cameras_to_ceres; // need to map ceres id to camera id and back
    std::vector<int> landmark_to_ceres;
    std::vector<int> ceres_to_camera_and_landmark; // to both.
    std::vector<double> landmarks;
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

            // if we get program we setup new program. if we get cam & prox we update cams (?) and prox term only! do one more it, etc.
            case request_proto::OptionsCase::kProgram : {
            std::cout << "request_proto::OptionsCase::kProgram" << std::endl;
            program_proto pro = request_p.program(); 
            {
                // pro make & run program
                // for (const auto cam : pro.cameras()) {
                //     std::cout << "cam " << cam << " ";
                // }
                // std::cout << std::endl;
                // execute run 

                numCameras = pro.cameras_size() / 9;
                numLandmarks = pro.landmarks_size() / 3;
                // Copy
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
                cameras_to_ceres.clear();
                cameras_to_ceres.resize(numCameras, -1);
                landmark_to_ceres.clear();
                landmark_to_ceres.resize(numLandmarks, -1);
                ceres_to_camera_and_landmark.clear();
                ceres_to_camera_and_landmark.resize(pro.cameras_size() + pro.landmarks_size(), -1);
       
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

                std::cout << "Parameter block added\n";
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
                    // Likely the map is by 1st occurence. if not present yet, 
                    // std::cout << "ceresId " << ceres_id << " of " << ceres_to_camera_and_landmark.size() 
                    //           << " -> " << pro.cam_id(i) << " of " << cameras_to_ceres.size() << "=" << cameras_to_ceres[pro.cam_id(i)] 
                    //           << " | " << pro.lm_id(i) << " of " << landmark_to_ceres.size() << "="  << landmark_to_ceres[pro.lm_id(i)] << "\n";

                    // Not needed any more.
                    if (cameras_to_ceres[pro.cam_id(i)] == -1) {
                        ceres_to_camera_and_landmark[ceres_id] = pro.cam_id(i);
                        cameras_to_ceres[pro.cam_id(i)] = ceres_id;
                        ceres_id += 9;
                    }
                    if (landmark_to_ceres[pro.lm_id(i)] == -1) {
                        // ceres_to_landmark[ceres_id] = pro.cam_id(i);
                        ceres_to_camera_and_landmark[ceres_id] = pro.lm_id(i);
                        landmark_to_ceres[pro.lm_id(i)] = ceres_id;
                        ceres_id += 3;
                    }
                }
                std::cout << "Finished " << ceres_id << "\n";

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

                if(false) {
                SparseMatrix<double, Eigen::RowMajor> Jp(jacobian.num_rows, 9 * numCameras);
                SparseMatrix<double, Eigen::RowMajor> Jl(jacobian.num_rows, 3 * numLandmarks);
                Jp.reserve(VectorXi::Constant(2 * jacobian.num_rows,9));
                Jl.reserve(VectorXi::Constant(2 * jacobian.num_rows,3));
                // JP.setFromTriplets(coefficients.begin(), coefficients.end());
                for (Eigen::Index r = 0; r < jacobian.num_rows; ++r) {
                //for (Eigen::Index r = 0; r < 1000; ++r) {
                    int lm_id = pro.lm_id(r/2);
                    int cam_id = pro.cam_id(r/2);
                    int ceres_cam_id = cameras_to_ceres[cam_id];
                    int ceres_lm_id = landmark_to_ceres[lm_id];
                    //std::cout << r << ":";
                    Eigen::Index idx = jacobian.rows[r];

                    while (idx < jacobian.rows[r + static_cast<Eigen::Index>(1)]) {
                        const Eigen::Index c = jacobian.cols[idx]; // index of var.

                        // std::cout << "ceres 2 cam/lm " << ceres_to_camera_and_landmark [c] << " = " << cam_id << " / " << lm_id << "\n";
                        // std::cout << "ceres id " << c << " camC:" << ceres_cam_id << " lmC:" << ceres_lm_id << "\n";

                        if (ceres_cam_id == c) { // cam, read 9 values
                            for (int i = 0; i < 9 && idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];++idx,++i) {
                                // read cam values.
                                //denseJacobian(r, c) = jacobian.values[idx];
                                //std::cout << "c" << jacobian.cols[idx] << " ";
                                Jp.insert(r, 9 * cam_id + i) = jacobian.values[idx];
                            }
                        }
                        if (ceres_lm_id == c) { // lm, read 3 values
                            for (int i = 0; i < 3 && idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];++idx, ++i) {
                                // read lm values.
                                //std::cout << "l" << jacobian.cols[idx] << " ";
                                Jl.insert(r, 3 * lm_id + i) = jacobian.values[idx];
                            }
                        }
                    }
                    //std::cout << "\n";
                }
                std::cout << std::endl;
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
                double be = 1e-4;
                JpJ.diagonal().array() += be * JpJ.diagonal().array();
                
                // using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
                // Eigen::Map<SparseMatrix> mapped_jacobian(jacobian.num_rows,
                //                                          jacobian.num_cols,
                //                                          jacobian.values.size(),
                //                                          jacobian.rows.data(),
                //                                          jacobian.cols.data(),
                //                                          jacobian.values.data());

                // 2. add to problem, [p-s]JpJ[p-s] for s fixed.
                // Worst case. read 9x9 block compute eigendecomp, sqrt diagonal.

                // const Eigen::JacobiSVD<ceres::Matrix> svd(denseJacobian,
                // Eigen::ComputeThinU | Eigen::ComputeThinV);
                // const ceres::Vector singularValues = svd.singularValues();

                // problem Jp we do not know the var-indices of the cameras :( in ceres.
            }
            else {
                SparseMatrix<double, Eigen::RowMajor> Jp(jacobian.num_rows, 9 * cameras_to_ceres.size());
                SparseMatrix<double, Eigen::RowMajor> Jl(jacobian.num_rows, 3 * landmark_to_ceres.size());
                Jp.reserve(VectorXi::Constant(2 * jacobian.num_rows,9));
                Jl.reserve(VectorXi::Constant(2 * jacobian.num_rows,3));
                // JP.setFromTriplets(coefficients.begin(), coefficients.end());
                for (Eigen::Index r = 0; r < jacobian.num_rows; ++r) {
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
                double be = 1e-4;
                JpJ.diagonal().array() += be * JpJ.diagonal().array();

                BlockSqrt<9>(JpJ);// need templated fct.

            }
                /////////////////////////////////////////////////////                                

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
                options.max_num_iterations = std::max(0,std::min(10, pro.iterations()));
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
                id=0;
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
                for (Eigen::Index r = 0; r < 1000; ++ r) { //jacobian.num_rows; ++r) {
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