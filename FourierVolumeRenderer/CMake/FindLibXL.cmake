# Copyright (c) 2012 Marwan Abdellah <abdellah.marwan@gmail.com>

# Set XL root directory 
set(XL_ROOT ${CMAKE_SOURCE_DIR}/LibXL)
set(XL_INC_DIR "${XL_ROOT}/include_cpp")
set(XL_LIB_DIR "${XL_ROOT}/lib64")

find_path(LIBXL_INC_DIR "libxl.h" 
  HINTS ${XL_INC_DIR}
  /usr/include
  /usr/local/include
  /opt/local/include 
)

find_library(LIBXL_LIB NAMES xl 
  HINTS ${XL_LIB_DIR}
  PATHS /usr/lib /usr/local/lib /opt/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBXL DEFAULT_MSG LIBXL_LIB LIBXL_INC_DIR)

# Include directories 
INCLUDE_DIRECTORIES(${LIBXL_INC_DIR})   

# Link LibXL libraries to the application 
LINK_LIBRARIES(${LIBXL_LIB})

if(LIBXL_FOUND)
  message(STATUS "Found LibXL in 
	${LIBXL_INC_DIR} 
	${LIBXL_LIB} ")
endif(LIBXL_FOUND)


