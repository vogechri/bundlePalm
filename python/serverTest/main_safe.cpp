// Original Code from http://kr.zeromq.org/cpp:hwserver
// kongineer.com


#include <zmq.hpp>
#include <string>
#include <iostream>
#include "generated/proto/test.pb.h"
#include <google/protobuf/message_lite.h>

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

    while (true) {
        zmq::message_t request;

        // Wait for the next request from a client
        socket.recv(&request);
        std::string received_message(static_cast<char*>(request.data()), request.size());

        program pro;
        if (pro.ParseFromString(received_message)) {
            // pro make & run program
            for (const auto cam : pro.cameras()) {
                std::cout << "cam " << cam << " ";
            }
            std::cout << std::endl;
            // execute run 

            // Copy
            std::vector<double> cameras;
            cameras.reserve(pro.cameras_size());
            for(const float& v : pro.cameras()) {
                cameras.push_back(v);
            }
            std::vector<double> landmarks;
            landmarks.reserve(pro.landmarks_size());
            for(const float& v : pro.landmarks()) {
                landmarks.push_back(v);
            }
            ceres::Problem problem;
            // setup
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
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = std::max(0,std::min(10,pro.iterations()));

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << "\n";

            std::string reply_message = "C++ Server received program";
            zmq::message_t reply(reply_message.size());
            socket.send(reply, zmq::send_flags::none);
            continue;
        }

        std::cout << "Received: " << received_message << std::endl;

        ImageVector iv;
        if (iv.ParseFromString(received_message)) {
            for (const auto image : iv.images()) {
                std::cout << "Image " << image.ids(0) << " " << image.data(0) << "\n";
            }
        }
        Image im;
        if (im.ParseFromString(received_message)) {
            for(int i =0;i < im.ids_size(); ++i)
            std::cout << "Image " << im.ids(i) << " " << im.data(i);
        }

        // Simulate some work
        delay(1);

        // Send a reply back to the client
        std::cout << " send back \n";
        std::string reply_message = "Hi from ZeroMQ C++ Server";
        zmq::message_t reply(reply_message.size());
        memcpy(reply.data(), reply_message.data(), reply_message.size());
        socket.send(reply, zmq::send_flags::none);
    }

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