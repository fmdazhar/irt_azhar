cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(Vessel)


# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set Boost directory (uncomment and modify the path if required) and include 
set(Boost_INCLUDE_DIR "/usr/include/boost/include")
include_directories( ${Boost_INCLUDE_DIR} )

# Find the required packages
find_package(Eigen3 REQUIRED)
find_package(Boost)

# Add an executable for your main file
add_executable(Vessel main.cpp)

# Link the libraries to your executable
target_link_libraries(Vessel PUBLIC Eigen3::Eigen ${Boost_LIBRARIES} stdc++fs)
