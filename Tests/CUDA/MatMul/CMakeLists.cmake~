
# Sources  
SET(TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/MatMul.cu 
		 ${CMAKE_CURRENT_LIST_DIR}/MatMul_Gold.cpp)

# Include directory 
SET(TEST_HEADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/inc)

# Add kernels to the include files 
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/Kernels)

# Add the include directory to the source tree 
INCLUDE_DIRECTORIES(${TEST_HEADERS_DIR})

# Generate the executable 
CUDA_ADD_EXECUTABLE(MatMul ${TEST_SOURCES})
