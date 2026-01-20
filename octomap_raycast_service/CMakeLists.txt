cmake_minimum_required(VERSION 3.12)
project(octomap_raycast_service LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O2)

find_package(octomap REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

add_executable(octomap_raycast_service main.cpp)

target_include_directories(octomap_raycast_service PRIVATE
  ${OCTOMAP_INCLUDE_DIRS}
  ${ZMQ_INCLUDE_DIRS}
)

target_link_libraries(octomap_raycast_service
  ${OCTOMAP_LIBRARIES}
  ${ZMQ_LIBRARIES}
)
