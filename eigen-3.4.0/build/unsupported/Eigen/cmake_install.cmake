# Install script for directory: /root/DEEPSDF/eigen-3.4.0/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/AdolcForward"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/AlignedVector3"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/ArpackSupport"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/AutoDiff"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/BVH"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/EulerAngles"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/FFT"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/IterativeSolvers"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/KroneckerProduct"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/LevenbergMarquardt"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/MatrixFunctions"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/MoreVectorization"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/MPRealSupport"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/NonLinearOptimization"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/NumericalDiff"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/OpenGLSupport"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/Polynomials"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/Skyline"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/SparseExtra"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/SpecialFunctions"
    "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/root/DEEPSDF/eigen-3.4.0/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/root/DEEPSDF/eigen-3.4.0/build/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

