
SET(FVR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/FourierVolumeRenderer/src)

INCLUDE_DIRECTORIES(${FVR_SOURCE_DIR}/)
SET(SOURCES ${FVR_SOURCE_DIR}/runEngine.cpp)

# eFourierVolRen Example
LIST(APPEND SOURCES ${FVR_SOURCE_DIR}/eFourierVolRen.cpp)

# Loader
LIST(APPEND SOURCES ${FVR_SOURCE_DIR}/Loader/Loader.cpp)
INCLUDE_DIRECTORIES(${FVR_SOURCE_DIR}/Loader)

# OpenGL
LIST(APPEND SOURCES ${FVR_SOURCE_DIR}/OpenGL/DisplayList.cpp)
LIST(APPEND SOURCES ${FVR_SOURCE_DIR}/OpenGL/cOpenGL.cpp)
INCLUDE_DIRECTORIES(${FVR_SOURCE_DIR}/OpenGL)

# Volume Processing
LIST(APPEND SOURCES ${FVR_SOURCE_DIR}/VolumeProcessing/volume.cpp)
INCLUDE_DIRECTORIES(${FVR_SOURCE_DIR}/VolumeProcessing/)

# FFT Shift
LIST(APPEND SOURCES ${FVR_SOURCE_DIR}/FFTShift/FFTShift.cpp)
INCLUDE_DIRECTORIES(${FVR_SOURCE_DIR}/FFTShift/)

# Spectrum
LIST(APPEND SOURCES ${FVR_SOURCE_DIR}/SpectrumProcessing/Spectrum.cpp)
INCLUDE_DIRECTORIES(${FVR_SOURCE_DIR}/SpectrumProcessing/)

# Slice Processing
LIST(APPEND SOURCES ${FVR_SOURCE_DIR}/SliceProcessing/Slice.cpp)
INCLUDE_DIRECTORIES(${FVR_SOURCE_DIR}/SliceProcessing/)

# Wrapping around
LIST(APPEND SOURCES ${FVR_SOURCE_DIR}/WrappingAround/WrappingAround.cpp)
INCLUDE_DIRECTORIES(${FVR_SOURCE_DIR}/WrappingAround/)

# Rendering Loop
LIST(APPEND SOURCES ${FVR_SOURCE_DIR}/RenderingLoop/RenderingLoop.cpp)
INCLUDE_DIRECTORIES(${FVR_SOURCE_DIR}/RenderingLoop/)


# Include directories 
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# CUDA kernels directory 
SET(CUDA_KERNELS_DIR ${CMAKE_SOURCE_DIR}/CUDA/FFT)
INCLUDE_DIRECTORIES(${CUDA_KERNELS_DIR})

# --------------------------------------------------------
# NOTE: You have to append all the ".cu" files in this file 
#       to have them compiled
# -------------------------------------------------------- 
# SET(CUDA_SOURCES)
    
# Generate the executable considering CUDA stuff :)
CUDA_ADD_EXECUTABLE(eFourierVolRen ${SOURCES} ${BASIC_SOURCES})
