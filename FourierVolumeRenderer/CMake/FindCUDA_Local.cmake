# Copyright (c) 2012 Marwan Abdellah <abdellah.marwan@gmail.com>

#Locate CUDA dependencies
FIND_PACKAGE(CUDA)

# Set SDK root directory 
set(CUDA_SDK_ROOT "${CMAKE_SOURCE_DIR}/../../NVIDIA_GPU_Computing_SDK")
set(CUDA_SDK_INC_DIR "${CUDA_SDK_ROOT}/C/common/inc")
set(CUDA_SDK_LIB_DIR "${CUDA_SDK_ROOT}/C/lib")

find_library(LIB_CUTIL NAMES cutil_x86_64 
  HINTS ${CUDA_SDK_LIB_DIR}
  PATHS /usr/lib /usr/local/lib /opt/local/lib
)

# Include directories 
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})   
INCLUDE_DIRECTORIES(${CUDA_SDK_INC_DIR}) 

# Link cCUDA libraries to the application 
LINK_LIBRARIES(${LIB_CUTIL})

if(CUDA_FOUND)
  message(STATUS "Found CUDA in 
	${CUDA_TOOLKIT_ROOT_DIR} 
	${CUDA_SDK_ROOT} 
	${CUDA_INCLUDE_DIRS} 
	${CUDA_LIBRARIES} 
	${CUDA_CUFFT_LIBRARIES}
	${LIB_CUTIL}")
endif(CUDA_FOUND)


