
# Sources  
SET(TEST_SOURCES_SINGLE ${CMAKE_CURRENT_LIST_DIR}/iB_FFT_Shift.cpp
		 ${CMAKE_CURRENT_LIST_DIR}/runRealSingle.cpp)

SET(TEST_SOURCES_DOUBLE ${CMAKE_CURRENT_LIST_DIR}/iB_FFT_Shift.cpp
		 ${CMAKE_CURRENT_LIST_DIR}/runRealDouble.cpp)

# Include directory 
SET(TEST_HEADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/inc)

# Add kernels to the include files 
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/Kernels)

# Add the include directory to the source tree 
INCLUDE_DIRECTORIES(${TEST_HEADERS_DIR})

# Generate the executables
CUDA_ADD_EXECUTABLE(iB_Real_Single_FFT_Shift_2D ${TEST_SOURCES_SINGLE} ${SOURCES} ${CUDA_SOURCES})
CUDA_ADD_EXECUTABLE(iB_Real_Double_FFT_Shift_2D ${TEST_SOURCES_DOUBLE} ${SOURCES} ${CUDA_SOURCES})
