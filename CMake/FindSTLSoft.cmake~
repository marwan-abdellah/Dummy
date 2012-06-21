# Copyright (c) 2012 Marwan Abdellah <abdellah.marwan@gmail.com>

# Set STL_Soft root directory 
set(STLSOFT_ROOT ${CMAKE_SOURCE_DIR}/STLSoft)
set(STLSOFT_INC_DIR "${STLSOFT_ROOT}/include/stlsoft")


find_path(STLSOFT_INC_DIR "stlsoft.h" 
  HINTS ${STLSOFT_INC_DIR}
  /usr/include
  /usr/local/include
  /opt/local/include 
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(STLSOFT DEFAULT_MSG STLSOFT_INC_DIR)

# Include directories 
INCLUDE_DIRECTORIES(${STLSOFT_INC_DIR})
INCLUDE_DIRECTORIES(${STLSOFT_ROOT}/include)

if(STLSOFT_FOUND)
  message(STATUS "Found STLSOFT in ${STLSOFT_INC_DIR}")
endif(STLSOFT_FOUND)


