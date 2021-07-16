cmake_minimum_required(VERSION 3.19)
project(BoatDetector)
add_subdirectory(hungarian_alg)
add_subdirectory(src)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(X11 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${X11_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${X11_LIBRARIES} Hungarian)

