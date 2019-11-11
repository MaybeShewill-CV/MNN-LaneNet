# Set cmake minium version
cmake_minimum_required(VERSION 3.5)

# Set project name
set(PROJECT_NAME MNN_LANENET)

# Set build output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/)

# Set some option
option(USE_CPP11 "Enable project use c++11" ON)

# Set source file dir
set(PROJECT.Path ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(PROJECT.Source_DIR
    ${PROJECT.Path}
    ${PROJECT.Path}/common/config_parse_utils
    ${PROJECT.Path}/lanenet
    ${PROJECT.Path}/postprocess
)

# Find useful package
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Glog REQUIRED)

# Set compile environment variable
if(USE_CPP11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Set build type
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

# Set include files
include_directories(${CMAKE_CURRENT_LIST_DIR}/MNN/include)
include_directories(${PROJECT.Source_DIR})

# Set lib
add_library( MNN SHARED IMPORTED )
set_target_properties(
    MNN
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/MNN/lib/libMNN.so
)

# Set source file
foreach(dir_path ${PROJECT.Source_DIR})
    aux_source_directory(${dir_path} PROJECT_SRC_CODE)
endforeach()

# Set exe
add_executable(lanenet_detector.out ./lane_detect.cpp ${PROJECT_SRC_CODE})
target_link_libraries(
    lanenet_detector.out
    MNN
    ${OpenCV_LIBS}
    /usr/local/lib/libglog.so
)
