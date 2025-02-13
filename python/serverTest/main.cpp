// Original Code from http://kr.zeromq.org/cpp:hwserver
// kongineer.com


#include <zmq.hpp>
#include <string>
#include <iostream>
#include "generated/proto/test.pb.h"
#include <google/protobuf/message_lite.h>

#include <Eigen/Core>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

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

int main() {
    // Initialize the context
    zmq::context_t context(1);

    // Create a socket of type REP (reply)
    zmq::socket_t socket(context, ZMQ_REP);

    // Bind the socket to a TCP address
    std::cout << "Starting the server on port 5555..." << std::endl;
    socket.bind("tcp://*:5555");

    // Those are permanent, variables can change.
    std::vector<double> cameras;
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
            //std::cout << "request_proto::OptionsCase::kProgram" << std::endl;
            program_proto pro = request_p.program(); 
            {
                // pro make & run program
                // for (const auto cam : pro.cameras()) {
                //     std::cout << "cam " << cam << " ";
                // }
                // std::cout << std::endl;
                // execute run 

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
                problem = ceres::Problem(); // overwrite ..?

                // setup problem again.
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
                options.max_num_iterations = std::max(0,std::min(10,pro.iterations()));
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

    #ifdef __write__
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
                for (Eigen::Index r = 0; r < jacobian.num_rows; ++r) {
                    for (Eigen::Index idx = jacobian.rows[r]; idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];
                        ++idx) {
                    const Eigen::Index c = jacobian.cols[idx];
                    std::cout << r << " " << c << ", ";// << " = " << jacobian.values[idx] << " | ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
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