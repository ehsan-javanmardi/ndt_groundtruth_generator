cmake_minimum_required(VERSION 2.8.3)
project(ndt_groundtruth_generator)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  sensor_msgs
  pcl_conversions
  #runtime_manager
  velodyne_pointcloud
  message_generation
)

find_package(PCL 1.8.0 REQUIRED)

catkin_package(
    #INCLUDE_DIRS include
    #LIBRARIES SLAM
    #CATKIN_DEPENDS runtime_manager
    #DEPENDS system_lib
)

###########
## Build ##
###########

SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

include_directories(include ${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ) #orde is important
include_directories(../include ../include/ndt)
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

#For ground truth positioning

#add_executable(ndt_3D_mapmatching_groundtruth_v1_1 src/ground_truth_generator/ndt_3D_mapmatching_groundtruth_v1.1.cpp  )
#target_link_libraries(ndt_3D_mapmatching_groundtruth_v1_1 ${catkin_LIBRARIES} ${PCL_LIBRARY_DIRS})

#add_executable(ndt_3D_mapmatching_groundtruth_v1_1_1 src/ground_truth_generator/ndt_3D_mapmatching_groundtruth_v1.1.1.cpp  )
#target_link_libraries(ndt_3D_mapmatching_groundtruth_v1_1_1 ${catkin_LIBRARIES} ${PCL_LIBRARY_DIRS})

#add_executable(ndt_3D_mapmatching_groundtruth_v1_2 src/ground_truth_generator/ndt_3D_mapmatching_groundtruth_v1.2.cpp  )
#target_link_libraries(ndt_3D_mapmatching_groundtruth_v1_2 ${catkin_LIBRARIES} ${PCL_LIBRARY_DIRS})

add_executable(ndt_groundtruth_generator_v1_0 src/ndt_groundtruth_generator_v1.0.cpp  )
target_link_libraries(ndt_groundtruth_generator_v1_0 ${catkin_LIBRARIES} ${PCL_LIBRARY_DIRS})
