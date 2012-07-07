# *********************************************************************
# * Copyrights (c) Marwan Abdellah. All rights reserved.
# * This code is part of my Master's Thesis Project entitled "High
# * Performance Fourier Volume Rendering on Graphics Processing Units
# * (GPUs)" and submitted to the Systems & Biomedical Engineering
# * Department, Faculty of Engineering, Cairo University.
# * Please, don't use or distribute without authors' permission.

# * File        : Volume
# * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
# * Created	: April 2011
# * Description	:
# * Note(s)	:
# *********************************************************************

# Minimum required CMake version 
cmake_minimum_required(VERSION 2.6)

# Fourier Volume Rendeing
PROJECT(FourierVolumeRenderer)

# Add CMake modules 
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

SET(FFTW_ROOT ${PROJECT_SOURCE_DIR}/../install) 
FIND_PACKAGE(FFTW REQUIRED)
IF(FFTW_FOUND)
  MESSAGE(STATUS "FFTW found...")
  MESSAGE("-- FFTW include directory : ${FFTW_INCLUDE_DIR}")
  MESSAGE("-- FFTW library directory : ${FFTW_LIBRARIES}")
  INCLUDE_DIRECTORIES(${FFTW_INCLUDE_DIR})
  LINK_LIBRARIES(${FFTW_LIBRARIES})
ELSE(FFTW_FOUND)
  MESSAGE(FATAL_ERROR "FFTW not found")
ENDIF(FFTW_FOUND)

# OpenGL & GLU & GLUT  
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLUT REQUIRED)

IF(OPENGL_FOUND)
  MESSAGE(STATUS "OpenGL found...")
  MESSAGE("-- OpenGL include directory : ${OPENGL_INCLUDE_DIR}")
  MESSAGE("-- OpenGL library directory : ${OPENGL_LIBRARY}")
  INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})
  LINK_LIBRARIES(${OPENGL_LIBRARY})
ELSE(OPENGL_FOUND)
  MESSAGE(FATAL_ERROR "OpenGL not found")
ENDIF(OPENGL_FOUND)

IF(GLUT_FOUND)
  MESSAGE(STATUS  "GLUT found...")
  MESSAGE("GLUT include directory : ${GLUT_INCLUDE_DIR}")
  MESSAGE("GLUT library directory : ${GLUT_LIBRARY}")
  INCLUDE_DIRECTORIES(${GLUT_INCLUDE_DIR}/GL)
  LINK_LIBRARIES(${GLUT_LIBRARY})
ELSE(GLUT_FOUND)
  MESSAGE(FATAL_ERROR "GLUT not found")
ENDIF(GLUT_FOUND)

# Add the heade files to the include directories
INCLUDE_DIRECTORIES("${OPENGL_INCLUDE_DIR}")

# Include FindCUDA.cmake 
INCLUDE(FindCUDA_Local)

# LibXL 64 bit
INCLUDE(FindLibXL)

# STL Soft 
INCLUDE(FindSTLSoft)

# Boost
INCLUDE(FindBoost_Local)

# OpenMP
INCLUDE(FindOpenMP_Local)

SET(SOURCES 
	Volume/Volume.cpp
	Array/Real/Array.cpp
	Array/Complex/cuComplex.cpp
	Array/Complex/fftwComplex.cpp
	Timers/BoostTimers.cpp
	Timers/CUDATimers.cpp
	Utilities/Utils.cpp
	Utilities/Logging.cpp
	Utilities/Tracing.cpp
	Utilities/Memory.cpp
	Utilities/XL.cpp
	Utilities/Bench.cpp
	OpenGL/DisplayLists.cpp
	FFT/FFTShift.cpp
	FFT/FFT2D.cpp
	FFT/FFT3D.cpp
	CUDA/Utilities/cuUtils.cpp
	FourierVolumeRenderer.cpp)

# Include directories 
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# CUDA kernels directory 
SET(CUDA_KERNELS_DIR ${CMAKE_SOURCE_DIR}/CUDA/FFT)
INCLUDE_DIRECTORIES(${CUDA_KERNELS_DIR})

# --------------------------------------------------------
# NOTE: You have to append all the ".cu" files in this file 
#       to have them compiled
# -------------------------------------------------------- 
SET(CUDA_SOURCES CUDA/cuExterns.cu)
    
# Generate the executable considering CUDA stuff :)
CUDA_ADD_EXECUTABLE(FVR FVR.cpp ${CUDA_SOURCES} ${SOURCES})

# Link to a library to be used later 
ADD_LIBRARY(CORE ${CUDA_SOURCES} ${SOURCES})

# Tests 
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/CUDA/MatMul/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/CUDA/FFT_Shift/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/LibXL/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/Array/cufftComplex/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/Array/fftwComplex/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/Array/Real/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/Timers/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/Templates/CMakeLists.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/Tests/CUDA/MaxSubArray/CMakeLists.cmake)

# Benchmarking 
#INCLUDE(${CMAKE_SOURCE_DIR}/Benchmarking/FFT_Shift/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Benchmarking/cuFFTShift_2D/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Benchmarking/cuFFTShift_3D/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Benchmarking/FFT_Shift/2D/Real/CMakeLists.cmake)
