
# Sources  
SET(TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/Run.cpp
		 ${CMAKE_CURRENT_LIST_DIR}/ex_cufftComplexArray.cpp)

# Include directory 
SET(TEST_HEADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/inc)

# Add the include directory to the source tree 
INCLUDE_DIRECTORIES(${TEST_HEADERS_DIR})

# Generate the executable 
CUDA_ADD_EXECUTABLE(cufftComplexArray ${TEST_SOURCES} ${SOURCES})
