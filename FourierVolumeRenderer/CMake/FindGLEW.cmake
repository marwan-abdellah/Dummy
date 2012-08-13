# Copyright (c) 2012 Marwan Abdellah <abdellah.marwan@gmail.com>

find_path(GLEW_INCLUDE_DIR "glew.h" 
  HINTS "/home/abdellah/NVIDIA_GPU_Computing_SDK/C/common/inc/GL"
  /usr/include
  /usr/local/include
  /opt/local/include 
)

find_library(GLEW_LIB NAMES GLEW_x86_64 
  HINTS "/home/abdellah/NVIDIA_GPU_Computing_SDK/C/common/lib/linux"
  PATHS /usr/lib /usr/local/lib /opt/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLEW DEFAULT_MSG GLEW_LIB GLEW_INCLUDE_DIR)

if(GLEW_FOUND)
  message(STATUS "Found GLEW in ${GLEW_INCLUDE_DIR} ${GLEW_LIB}")
endif(GLEW_FOUND)


