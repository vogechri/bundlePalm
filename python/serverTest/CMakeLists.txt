# CMakeList.txt : CMake project for zmq_cpp_server_ex, include source and define
# kongineer.com
#
# Add this if cmake fails to find the package
add_subdirectory(proto)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

MESSAGE(STATUS CMAKE_MODULE_PATH:${CMAKE_MODULE_PATH})

cmake_minimum_required(VERSION 3.8)
project(zeromq_cpp_server_ex)

# Find ZeroMQ
find_package(ZeroMQ REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen REQUIRED)

message(${ZeroMQ_INCLUDE_DIRS})
message(${ZeroMQ_LIBRARIES})
message(${Protobuf_INCLUDE_DIRS})
message(${Protobuf_LIBRARIES}) # /usr/lib/x86_64-linux-gnu/libprotobuf.so
message(${Protobuf_PROTOC_LIBRARIES}) # not found
message(${Protobuf_LITE_LIBRARIES}) # /usr/lib/x86_64-linux-gnu/libprotobuf-lite.so
#message(${Eigen_INCLUDE_DIRS})
#message(${Ceres_INCLUDE_DIRS})
#message(${Ceres_LIBRARIES})
message(${CMAKE_CURRENT_BINARY_DIR})

# proto lib to link from c++
add_library(proto-objects OBJECT "${CMAKE_CURRENT_LIST_DIR}/proto/test.proto")
target_link_libraries(proto-objects PUBLIC protobuf::libprotobuf)
set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${PROTO_BINARY_DIR}")
#set(PROTO_BINARY_DIR "${CMAKE_CURRENT_LIST_DIR}/generated")
target_include_directories(proto-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")
protobuf_generate(
    TARGET proto-objects
    IMPORT_DIRS "${CMAKE_CURRENT_LIST_DIR}/proto"
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")
protobuf_generate(
    TARGET proto-objects
    LANGUAGE python
    IMPORT_DIRS "${CMAKE_CURRENT_LIST_DIR}/proto"
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

# Add your source files
add_executable(${PROJECT_NAME} main.cpp)

# Include directories
include_directories(${Protobuf_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${ZeroMQ_INCLUDE_DIR}) # ${CMAKE_CURRENT_LIST_DIR}/proto)

# Link the Library
target_link_libraries(${PROJECT_NAME} PRIVATE proto-objects ${ZeroMQ_LIBRARIES} ${Protobuf_LIBRARIES} Ceres::ceres)
