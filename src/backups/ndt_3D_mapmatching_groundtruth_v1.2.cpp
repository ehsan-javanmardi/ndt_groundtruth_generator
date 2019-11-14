/*
 This code can be used to get ground truth of map matching

 Ehsan Javanmardi

 2017.07.15
 ground_truth_generator_v1.0


 TETS :
    code is not tested yet

 CHANGE LOG :
    Based on ndt_3D_mapmatching_groundtruth_v1.1
    Use new point type PointXYZIT does delta_t and disotrtion
    point type is defined in self_driving_point_type.h
    Get velodyne pointcloud as input array to get groundtruth
    Calculate the ground truth for the scan which has more than some thereshold of displacement
    Get /velodyne_points_groundtruth instead of /velodyne_points to be able to do the parallel processing
    Change topic can be done with rosbag as follows :
        rosbag play my_bag.bag /velodyne_points:=/velodyne_points_groundtruth

 */

#define OUTPUT

#define PCL_NO_PRECOMPILE

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud2.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <math.h>
#include <boost/filesystem.hpp>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <ndt/ndt_vis_distortion.h>
#include "distortion.h"
#include "kmj_self_driving_common.h"
#include "self_driving_point_type.h"

using namespace boost::filesystem;

#define PREDICT_POSE_THRESHOLD 0.5

#define Wa 0.4
#define Wb 0.3
#define Wc 0.3

struct iter_result
{
    Eigen::Matrix4f tf_align;
    Eigen::Matrix4f tf_initial;

    int number_iter;
    double score;
    double trans_probability;
    double matching_time;
};

pose initial_pose, predict_pose, previous_pose, ndt_pose;
pose current_pose, control_pose, localizer_pose, previous_gnss_pose, current_gnss_pose;
pose offset; // current_pos - previous_pose

// If the map is loaded, map_loaded will be true.
bool map_loaded = false;

// Visual NDT with distortion removal
NormalDistributionsTransform_Visual_dis<PointXYZIT, PointXYZIT> ndt;

// Default values
static int iter = 30; // Maximum iterations
static double gridSize = 1.0; // Resolution
static double step_size = 0.1; // Step size
static double trans_eps = 0.01; // Transformation epsilon

// Leaf size of VoxelGrid filter.

double voxel_leaf_size = 1.0;

// publishers

ros::Publisher map_pub;
ros::Publisher ndt_map_pub;
ros::Publisher initial_scan_pub;
ros::Publisher aligned_scan_pub;
ros::Publisher best_transformed_dis_scan_pub;

// show data on rviz

bool show_stf_aligncan = true;
bool show_filtered_scan = true;
bool show_transformed_scan = true;
bool show_initial_scan = true;
bool show_map = true;
bool show_car_trajectory = true;
bool show_best_transformed_dis_scan = true;

// save scan data

bool save_transformed_scan = false;
bool save_predicted_scan = false;
bool save_aligned_scan = false;
bool save_best_transformed_dis_scan = false;

std::string map_file_path;
std::string map_file_name;
std::string save_path = "/home/ehsan/workspace/results/map_matching/groundtruth";
int map_load_mode = 0;

// time variables

ros::Time current_scan_time;
ros::Time previous_scan_time;
ros::Duration scan_duration;
std::chrono::time_point<std::chrono::system_clock> \
        matching_start, matching_end, downsample_start, downsample_end, \
        align_start, align_end;

Eigen::Matrix4f tf_predict, tf_previous, tf_current;

int skipSeq;

static double x_startpoint = 0.0;
static double y_startpoint = 0.0;
static double z_startpoint = 0.0;
static double yaw_startpoint =  0.0;//(-45/180.0) * M_PI ;
static double roll_startpoint =  0.0;//(0/180.0) * M_PI ;
static double pitch_startpoint = 0.0;//(-33/180.0) * M_PI  ;

FILE * pFileLog;

pcl::PCDWriter writer;

pcl::PointCloud<PointXYZIT>::Ptr map_ptr (new pcl::PointCloud<PointXYZIT>);

// these tow variable is for log file only to show where the point cloud is saved
std::string savedMap ="";
std::string savedRoadMarkingWindow = "";

std::vector<pose> carPoseList;

int scan_seq;

double displacement_threshold = 0.5;



static void scan_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{
    // SKIP SPECIFIED NUMBER OF SCAN ########################################################################

    if (input->header.seq < (unsigned int) skipSeq)
    {
        std::cout << "skip " << input->header.seq << std::endl;
        return;
    }

    // CHECK IF MAP IS LOADED OR NOT ########################################################################

    if (!map_loaded)
    {
        std::cout << "map is not loaded......... velodyne seq is : " << input->header.seq << std::endl;
        return;
    }

    // SHOW MAP IN RVIZ #####################################################################################

    if (show_map)
    {
        publish_pointCloud(*map_ptr, map_pub, "map");

        pcl::VoxelGridCovariance<PointXYZIT> target_cells;
        ndt.getCells(target_cells);

        //pcl::VoxelGridCovariance<pcl::PointXYZ>::Leaf leaf_;
        //std::map<size_t, pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf> leaves;
        //leaves = target_cells.getLeaves();
        //std::vector<Leaf> leafList;
        //getLeaves(leafList);

        typedef pcl::VoxelGridCovariance<PointXYZIT> VectorCovarianceXYZ;
        typedef typename VectorCovarianceXYZ::Leaf VectorCovarianceLeaf;
        typedef std::vector<VectorCovarianceLeaf> VectorCovarianceLeafList;

        VectorCovarianceLeafList leafList;
        ndt.getLeaves(leafList);

        visualization_msgs::MarkerArray ndtSphereList;
        //Eigen::Vector4d RGBA(0.35, 0.7, 0.8, 0.2);
        Eigen::Vector4d normalDistribution_color(0.35, 0.7, 0.8, 0.2);

        //double d1 = ndt.get_d1();
        //double d2 = ndt.get_d2();

        // a 90% confidence interval corresponds to scale=4.605
        //showCovariance(leaves, ndtSphereList, 4.605 ,"map", RGBA, d1, d2);
        //showCovariance(leaves, ndtSphereList, 4.605 ,"map", normalDistribution_color);
        setCovarianceListMarker<pcl::VoxelGridCovariance<PointXYZIT>::Leaf>(leafList, ndtSphereList, \
                                                      4.605 ,"map", normalDistribution_color, 20);

        ndt_map_pub.publish(ndtSphereList);

        show_map = 0;
    }

    scan_seq = input->header.seq;

    current_scan_time = input->header.stamp;

    // CONVERT MESSSAGE TO POINT CLOUD ######################################################################

    pcl::PointCloud<velodyne_pointcloud::PointXYZIR> scan_xyzir;
    pcl::fromROSMsg(*input, scan_xyzir);

    pcl::PointCloud<velodyne_pointcloud::PointXYZIR> calibrated_scan_xyzir;

    // CALIBRATE POINT CLOUD SUCH THAT THE SENSOR BECOME PERPENDICULAR TO GROUND SURFACE ####################
    // THIS STEP HELPS TO REMOVE GROUND SURFACE MUCH EASIER

    pose pose_lidar(0.0, 0.0, 0.0, roll_startpoint, pitch_startpoint, 0.0);
    static Eigen::Matrix4f tf_lidar;

    pose_to_tf(pose_lidar, tf_lidar);

    pcl::transformPointCloud(scan_xyzir, calibrated_scan_xyzir, tf_lidar);

    // DO NOT REMOVE GROUND #################################################################################

    pcl::PointCloud<PointXYZIT> scan;

    for (pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::const_iterator item = calibrated_scan_xyzir.begin(); \
         item != calibrated_scan_xyzir.end(); item++)
    {
        PointXYZIT p;

        p.x = (double) item->x;
        p.y = (double) item->y;
        p.z = (double) item->z;
        p.intensity = (double) item->intensity;
        //p.ring = item->ring;

        if (getR(p) > 1.0 && getR(p) < 90.0)
            scan.points.push_back(p);
    }

    // STATIC GROUND AND CAR ROOF REMOVAL ###################################################################
/*
    pcl::PointCloud<pcl::PointXYZI> scan;

    for (pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::const_iterator item = calibrated_scan_xyzir->begin(); \
         item != calibrated_scan_xyzir->end(); item++)
    {

        //if (item->z > -1.8 && item->z < 2.0)
        if (item->z > -1.5 && item->z < 2.0)
        {
            //if the points placed in the roof of vehicle
            if (item->x > 0.5 && item->x < 2.2 && item->y < 0.8 && item->y > -0.8);
            else
            {
                pcl::PointXYZI p;

                p.x = (double) item->x;
                p.y = (double) item->y;
                p.z = 0.0; // because 3D matching
                p.intensity = (double) item->intensity;
                //p.ring = item->ring;

                scan3D_ptr->points.push_back(p);
            }
        }
    }
*/
    // TRANSFORM SCAN TO GLOBAL COORDINATE IT FROM LOCAL TO GLOBAL ##########################################

    int GPS_enabled = 0;

    if (GPS_enabled)
    {
/*
        // get current locaition from GPS
        // translate scan to current location

        // get x,y,z and yaw from GPS
        // this will be predict_pose

        std::cout << "##### use GPS !!" << std::endl;

        pose gps;


        Eigen::Translation3f predict_translation(gps.x, gps.y, gps.z);
        Eigen::AngleAxisf predict_rotation_x(roll_startpoint , Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf predict_rotation_y(pitch_startpoint, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf predict_rotation_z(gps.yaw, Eigen::Vector3f::UnitZ());

        Eigen::Matrix4f tf_predict = (predict_translation * predict_rotation_z * predict_rotation_y * predict_rotation_x).matrix();
*/

    }
    else
    {
        // local to global using tf_ltob x,y,z,yaw(heading)
        // calibrate
        // estimate current position using previous and offset
        // Guess the initial gross estimation of the transformation

        offset.roll = 0.0;
        offset.pitch = 0.0;
        predict_pose = previous_pose + offset;

        pose_to_tf(predict_pose, tf_predict);
        pose_to_tf(previous_pose, tf_previous);
    }

    // SHOW INITIAL SCAN

    if (show_initial_scan)
    {
        pcl::PointCloud<PointXYZIT>  predicted_scan;

        pcl::transformPointCloud(scan, predicted_scan, tf_predict);

        publish_pointCloud(predicted_scan, initial_scan_pub, "map");
        show_initial_scan = false;
    }



    // UPDATE DELTA T FOR THE SCAN ##########################################################################

    calculateDeltaT(scan, scan);

    // DOWNSAMPLE SCAN USING VOXELGRID FILTER ###############################################################

    pcl::PointCloud<PointXYZIT>::Ptr input_cloud_ptr(new pcl::PointCloud<PointXYZIT>(scan));
    pcl::PointCloud<PointXYZIT> filtered_scan;

    pcl::ApproximateVoxelGrid<PointXYZIT> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_grid_filter.setInputCloud(input_cloud_ptr);
    voxel_grid_filter.filter(filtered_scan);

    align_start = std::chrono::system_clock::now();    // TIME $$$$$$$$$$$$$$$$$$$$$$$$$$$

    pcl::PointCloud<PointXYZIT>::Ptr filter_scan_ptr(new pcl::PointCloud<PointXYZIT>(filtered_scan));

    ndt.setInputSource(filter_scan_ptr);

    ndt.setPreviousTF(tf_previous);

    ndt.setSearchResolution(gridSize);

    pcl::PointCloud<PointXYZIT> aligned_scan;

    // aligned scan is distortion removed downsampled scan so it is not useful

    ndt.align(aligned_scan, tf_predict);

    Eigen::Matrix4f tf_align(Eigen::Matrix4f::Identity()); // base_link
    tf_align = ndt.getFinalTransformation(); // localizer
    double trans_probability = ndt.getTransformationProbability();
    int iteration_number = ndt.getFinalNumIteration();

    align_end = std::chrono::system_clock::now();    // TIME $$$$$$$$$$$$$$$$$$$$$$$$$$$

    double matching_time = std::chrono::duration_cast<std::chrono::microseconds>\
            (align_end - align_start).count()/1000.0; // double



    // iteration for finding more accurate results

    double steps = 0.05;

    double yaw_steps = 0.5; // in degree

    double max_trans_probability = -1000.0;

    Eigen::Matrix4f tf_best_alignment;
    Eigen::Matrix4f tf_best_initial_guess;

    pcl::PointCloud<PointXYZIT> best_aligned_scan;

    std::vector<iter_result> iter_results;

    bool calculate_ground_truth = true;

    double displacement = sqrt(pow(tf_align(0,3) - tf_previous(0,3),2) + \
                               pow(tf_align(1,3) - tf_previous(1,3),2) + \
                               pow(tf_align(2,3) - tf_previous(2,3),2));

    if (displacement > displacement_threshold)
    {
        calculate_ground_truth = true;

        for (int x_iter=-4; x_iter < 4; x_iter++)
            for (int y_iter=-4; y_iter < 4; y_iter++)
                for (int yaw_iter=-4; yaw_iter<4; yaw_iter++) // in degree
                {
                    align_start = std::chrono::system_clock::now();    // TIME $$$$$$$$$$$$$$$$$$$$$$$$$$$

                    ndt.setInputSource(filter_scan_ptr);

                    ndt.setPreviousTF(tf_previous);

                    pose temp_pose;

                    tf_to_pose(tf_align, temp_pose);

                    temp_pose.x += (double)x_iter * steps;
                    temp_pose.y += (double)y_iter * steps;

                    double yaw_r = (yaw_steps /180.0 ) * M_PI;

                    temp_pose.yaw += (((double)yaw_iter * yaw_r));

                    Eigen::Matrix4f tf_iter;

                    pose_to_tf(temp_pose, tf_iter);

                    pcl::PointCloud<PointXYZIT> iter_aligned_scan;

                    iter_result iter_result_;

                    iter_result_.tf_initial = tf_iter;
                    iter_result_.score = ndt.calculateScore(*filter_scan_ptr, tf_iter);
                    iter_result_.trans_probability = iter_result_.score / ((double)filter_scan_ptr->size());

                    ndt.align(iter_aligned_scan, tf_iter);

                    align_end = std::chrono::system_clock::now();    // TIME $$$$$$$$$$$$$$$$$$$$$$$$$$$

                    iter_result_.matching_time = std::chrono::duration_cast<std::chrono::microseconds>\
                            (align_end - align_start).count()/1000.0; // double

                    iter_result_.tf_align = ndt.getFinalTransformation();
                    iter_result_.number_iter = ndt.getFinalNumIteration();

                    iter_results.push_back(iter_result_);

                    std::cout << "############## iteration for scan " << scan_seq << " #############\n\n";

                    std::cout << "iter_result_.tf_initial \n" << iter_result_.tf_initial << std::endl;
                    std::cout << "iter_result_.score --> " << iter_result_.score << std::endl;
                    std::cout << "iter_result_.trans_probability -->" << iter_result_.trans_probability << std::endl;
                    std::cout << "iter_result_.tf_align --> " << iter_result_.tf_align << std::endl;
                    std::cout << "iter_result_.number_iter --> " << iter_result_.number_iter << std::endl;

                    double iter_trans_probability = ndt.getTransformationProbability();

                    if (max_trans_probability < iter_trans_probability)
                    {
                        tf_best_initial_guess = tf_iter;
                        tf_best_alignment = ndt.getFinalTransformation();
                        max_trans_probability = iter_trans_probability;
                        best_aligned_scan = iter_aligned_scan;

                        std::cout << "$$$$$$$$$$ BEST SCORE IS UPDATED. SCORE is " << max_trans_probability\
                                  << "$$$$$$$$$$" << std::endl;

                        std::cout << "best tf is : \n" <<  tf_best_alignment << std::endl << std::endl;

                        publish_pointCloud(best_aligned_scan, aligned_scan_pub, "map");
                    }
                }
    }
    else
    {
        calculate_ground_truth = false;
        tf_best_alignment = tf_align;
    }

    // get the global translation matrix

    tf_to_pose(tf_best_alignment, current_pose);

    offset = current_pose - previous_pose;

    previous_pose = current_pose;

    // SAVE LOG FILES ###############################################################

    if (true)
    {

        double average_matching_time = 0.0;
        double average_number_iteration = 0.0;
        double average_trans_probability = 0.0;

        for (int i=0; i<iter_results.size();i++)
        {
            average_matching_time += iter_results[i].matching_time;
            average_number_iteration += iter_results[i].number_iter;
            average_trans_probability += iter_results[i].trans_probability;
        }

        average_matching_time = average_matching_time / (double) iter_results.size();
        average_number_iteration = average_number_iteration / (double) iter_results.size();
        average_trans_probability = average_trans_probability / (double) iter_results.size();

        pose aligned_pose;
        tf_to_pose(tf_align, aligned_pose);

        pose best_init_pose;
        tf_to_pose(tf_best_initial_guess, best_init_pose);

        std::string name = save_path + "groundtruth.csv";

        pFileLog = fopen (name.c_str(),"a");

        fprintf (pFileLog, "%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i,%i\n",\
                 scan_seq, \
                 current_pose.x, current_pose.y, current_pose.z, \
                 ( current_pose.roll /M_PI) * 180.0, ( current_pose.pitch /M_PI) * 180.0, ( current_pose.yaw /M_PI) * 180.0,
                 max_trans_probability * 10000.0,\
                 average_matching_time, average_number_iteration, average_trans_probability,\
                 filtered_scan.size(), scan.size(),\
                 fabs(best_init_pose.x - current_pose.x), fabs(best_init_pose.y - current_pose.y),\
                 fabs(best_init_pose.z - current_pose.z), (fabs(best_init_pose.yaw - current_pose.yaw)/M_PI) * 180.0,\
                 aligned_pose.x, aligned_pose.y, aligned_pose.z, \
                 aligned_pose.roll, aligned_pose.pitch, aligned_pose.yaw, \
                 matching_time, trans_probability, iteration_number,\
                 calculate_ground_truth);

        fclose (pFileLog);
    }

    /*
    // SAVE 2D ERROR HISTOGRAM FOR BEST YAW ANGLE ##############################################################

    if (true)
    {
        std::string name = save_path + "error_hitogram_best_yaw/error_hitogram_best_yaw_" + \
                           std::to_string(scan_seq) + ".csv";

        FILE *pFile;
        pFile = fopen(name.c_str(), "w");

        for (int i=0; i<iter_results.size(); i++)
        {
            iter_result item = iter_results[i];

            Eigen::Matrix4f tf_align = item.tf_align;
            Eigen::Matrix4f tf_initial = item.tf_initial;

            // if best yaw
            if ( tf_best_initial_guess(0,0) == tf_initial(0,0) )
            {
                double error_3d = sqrt(pow((tf_align(0,3)- tf_initial(0,3)),2) +\
                                       pow((tf_align(1,3)- tf_initial(1,3)),2) +\
                                       pow((tf_align(2,3)- tf_initial(2,3)),2));

                fprintf(pFile, "%f,%f,%f,%f\n", tf_initial(0,3), tf_initial(1,3), error_3d, error_3d);
            }
        }

        fclose(pFile);
    }

    // SAVE YAW ERROR HISTOGRAM FOR BEST X AND Y

    if (true)
    {
        std::string name = save_path + "error_hitogram_best_yaw/error_hitogram_best_yaw_" + \
                           std::to_string(scan_seq) + ".csv";

        FILE *pFile;
        pFile = fopen(name.c_str(), "w");

        for (int i=0; i<iter_results.size(); i++)
        {
            iter_result item = iter_results[i];

            Eigen::Matrix4f tf_align = item.tf_align;
            Eigen::Matrix4f tf_initial = item.tf_initial;

            // if best yaw
            if ( tf_best_alignment(0,0) == tf_align(0,0) )
            {
                double error_3d = sqrt(pow((tf_align(0,3)- tf_initial(0,3)),2) +\
                                       pow((tf_align(1,3)- tf_initial(1,3)),2) +\
                                       pow((tf_align(2,3)- tf_initial(2,3)),2));

                fprintf(pFile, "%f,%f,%f,%f\n", tf_initial(0,3), tf_initial(1,3), error_3d, error_3d);
            }
        }

        fclose(pFile);
    }*/

    // MAKE DISTORTION REMOVED TRANSFORMED SCAN #############################################################
    if (true)
    {

        pcl::PointCloud<PointXYZIT> best_transformed_dis_scan;

        std::vector<double> delta_t;

        calculateDeltaT(scan, delta_t);
        removeDistortion(scan, best_transformed_dis_scan, tf_previous, tf_best_alignment, delta_t);
        pcl::transformPointCloud(best_transformed_dis_scan, best_transformed_dis_scan, tf_best_alignment);

        best_transformed_dis_scan.height = 1;
        best_transformed_dis_scan.width = best_transformed_dis_scan.size();
        best_transformed_dis_scan.points.resize (best_transformed_dis_scan.width * best_transformed_dis_scan.height);


        // SHOW DISTORTION REMOVED TRANSFORMED SCAN ################################################################################

        publish_pointCloud(best_transformed_dis_scan, best_transformed_dis_scan_pub, "map");

        // SAVE DISTORTION REMOVED TRANSFORMED SCAN #################################################################################

        std::string name = save_path + "best_trans_dis_san/best_trans_dis_san_" + \
                           std::to_string(scan_seq) + ".csv";

        FILE *pFile;
        pFile = fopen(name.c_str(), "w");

        for (int i=0; i< best_transformed_dis_scan.size(); i++)
        {
            PointXYZIT p;
            p = best_transformed_dis_scan[i];

            fprintf(pFile, "%f,%f,%f,%f,%f\n", p.x, p.y, p.z, p.intensity, p.delta_t);
        }

        fclose(pFile);
    }

    // SAVE SCAN WITH DISTORTION
    // MATCHING IS DONE BASED ON DISTORTION ######################################################

    if (true)
    {

        pcl::PointCloud<PointXYZIT> best_transformed_scan;

        pcl::transformPointCloud(scan, best_transformed_scan, tf_best_alignment);

        best_transformed_scan.height = 1;
        best_transformed_scan.width = best_transformed_scan.size();
        best_transformed_scan.points.resize (best_transformed_scan.width * best_transformed_scan.height);

        // for this scan matching is based on distortion removed and iteration process
        // the saved scan is the one use best alignment but the scan itself has distortion

        std::string name = save_path + "best_trans_san/best_trans_san_" + \
                           std::to_string(scan_seq) + ".csv";

        FILE *pFile;
        pFile = fopen(name.c_str(), "w");

        for (int i=0; i< best_transformed_scan.size(); i++)
        {
                PointXYZIT p;
                p = best_transformed_scan[i];

                fprintf(pFile, "%f,%f,%f,%f\n", p.x, p.y, p.z, p.intensity);
        }

        fclose(pFile);
    }

    // SAVE INITIAL GUESS FOR THE SCANS ##########################################################

    if (true)
    {

        pcl::PointCloud<PointXYZIT> initial_guess_scan;

        std::vector<double> delta_t;

        calculateDeltaT(scan, delta_t);
        removeDistortion(scan, initial_guess_scan, tf_previous, tf_predict, delta_t);
        pcl::transformPointCloud(initial_guess_scan, initial_guess_scan, tf_predict);

        initial_guess_scan.height = 1;
        initial_guess_scan.width = initial_guess_scan.size();
        initial_guess_scan.points.resize (initial_guess_scan.width * initial_guess_scan.height);

        std::string name = save_path + "initial_guess_dis_san/initial_guess_dis_san_" + \
                           std::to_string(scan_seq) + ".csv";

        FILE *pFile;
        pFile = fopen(name.c_str(), "w");

        for (int i=0; i< initial_guess_scan.size(); i++)
        {
            PointXYZIT p;
            p = initial_guess_scan[i];

            fprintf(pFile, "%f,%f,%f,%f,%f\n", p.x, p.y, p.z, p.intensity, p.delta_t);
        }

        fclose(pFile);
    }


    // SAVE THE FIRST ITERATION OF ALIGNMENT #####################################################

    if (true)
    {

        pcl::PointCloud<PointXYZIT> transformed_dis_scan;

        std::vector<double> delta_t;

        calculateDeltaT(scan, delta_t);
        removeDistortion(scan, transformed_dis_scan, tf_previous, tf_align, delta_t);
        pcl::transformPointCloud(transformed_dis_scan, transformed_dis_scan, tf_align);

        transformed_dis_scan.height = 1;
        transformed_dis_scan.width = transformed_dis_scan.size();
        transformed_dis_scan.points.resize (transformed_dis_scan.width * transformed_dis_scan.height);

        std::string name = save_path + "trans_dis_san/trans_dis_san_" + \
                           std::to_string(scan_seq) + ".csv";

        FILE *pFile;
        pFile = fopen(name.c_str(), "w");

        for (int i=0; i< transformed_dis_scan.size(); i++)
        {
            PointXYZIT p;
            p = transformed_dis_scan[i];

            fprintf(pFile, "%f,%f,%f,%f,%f\n", p.x, p.y, p.z, p.intensity, p.delta_t);
        }

        fclose(pFile);
    }
}


int main(int argc, char **argv)
{
    std::cout << "ndt_3D_mapmatching_groundtruth_v1_1\n" ;
    ros::init(argc, argv, "ndt_3D_mapmatching_groundtruth_v1_1");


    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    skipSeq = 0;

    if (private_nh.getParam("x_startpoint", x_startpoint) == false)
    {
        std::cout << "x_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "x_startpoint: " << x_startpoint << std::endl;

    if (private_nh.getParam("y_startpoint", y_startpoint) == false){
        std::cout << "y_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "y_startpoint: " << y_startpoint << std::endl;

    if (private_nh.getParam("z_startpoint", z_startpoint) == false)
    {
        std::cout << "z_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "z_startpoint: " << z_startpoint << std::endl;

    if (private_nh.getParam("yaw_startpoint", yaw_startpoint) == false)
    {
        std::cout << "yaw_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "yaw_startpoint in degree : " << yaw_startpoint << std::endl;
    yaw_startpoint = (yaw_startpoint /180.0) * M_PI;
    //yaw_startpoint = ((124.0 - 157.3)/180.0) * M_PI;

    if (private_nh.getParam("pitch_startpoint", pitch_startpoint) == false)
    {
        std::cout << "pitch_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "pitch_startpoint in degree : " << pitch_startpoint << std::endl;
    pitch_startpoint = (pitch_startpoint /180.0) * M_PI;

    if (private_nh.getParam("roll_startpoint", roll_startpoint) == false)
    {
        std::cout << "roll_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "roll_startpoint in degree : " << roll_startpoint << std::endl;
    roll_startpoint = (roll_startpoint /180.0) * M_PI;

    if (private_nh.getParam("gridSize", gridSize) == false)
    {
      std::cout << "gridSize is not set." << std::endl;
      //return -1;
    }
    std::cout << "gridSize: " << gridSize << std::endl;

    if (private_nh.getParam("skipSeq", skipSeq) == false)
    {
      std::cout << "skipSeq is not set." << std::endl;
      //return -1;
    }
    std::cout << "skipSeq: " << skipSeq << std::endl;

    if (private_nh.getParam("voxel_leaf_size", voxel_leaf_size) == false)
    {
      std::cout << "voxel_leaf_size is not set." << std::endl;
      //return -1;
    }
    std::cout << "voxel_leaf_size: " << voxel_leaf_size << std::endl;

    if (private_nh.getParam("show_filtered_scan", show_filtered_scan) == false)
    {
      std::cout << "show_filtered_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "show_filtered_scan: " << show_filtered_scan << std::endl;

    if (private_nh.getParam("show_transformed_scan", show_transformed_scan) == false)
    {
      std::cout << "show_transformed_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "show_transformed_scan: " << show_transformed_scan << std::endl;

    if (private_nh.getParam("show_initial_scan", show_initial_scan) == false)
    {
      std::cout << "show_initial_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "show_initial_scan: " << show_initial_scan << std::endl;

    if (private_nh.getParam("show_map", show_map) == false)
    {
      std::cout << "show_map is not set." << std::endl;
      //return -1;
    }
    std::cout << "show_map: " << show_map << std::endl;

    if (private_nh.getParam("show_car_trajectory", show_car_trajectory) == false)
    {
      std::cout << "show_car_trajectory is not set." << std::endl;
      //return -1;
    }
    std::cout << "show_car_trajectory: " << show_car_trajectory << std::endl;

    if (private_nh.getParam("save_transformed_scan", save_transformed_scan) == false)
    {
      std::cout << "save_transformed_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_transformed_scan: " << save_transformed_scan << std::endl;

    if (private_nh.getParam("map_file_path", map_file_path) == false)
    {
      std::cout << "map_file_path is not set." << std::endl;
      //return -1;
    }
    std::cout << "map_file_path: " << map_file_path << std::endl;

    if (private_nh.getParam("map_file_name", map_file_name) == false)
    {
      std::cout << "map_file_name is not set." << std::endl;
      //return -1;
    }
    std::cout << "map_file_name: " << map_file_name << std::endl;

    if (private_nh.getParam("map_load_mode", map_load_mode) == false)
    {
      std::cout << "map_load_mode is not set." << std::endl;
      //return -1;
    }
    std::cout << "map_load_mode: " << map_load_mode << std::endl;

    if (private_nh.getParam("save_predicted_scan", save_predicted_scan) == false)
    {
      std::cout << "save_predicted_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_predicted_scan: " << save_predicted_scan << std::endl;

    if (private_nh.getParam("save_aligned_scan", save_aligned_scan) == false)
    {
      std::cout << "save_aligned_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_aligned_scan: " << save_aligned_scan << std::endl;

    if (private_nh.getParam("save_path", save_path) == false)
    {
      std::cout << "save_path is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_path: " << save_path << std::endl;

    if (private_nh.getParam("displacement_threshold", displacement_threshold) == false)
    {
      std::cout << "displacement_threshold is not set." << std::endl;
      //return -1;
    }
    std::cout << "displacement_threshold: " << displacement_threshold << std::endl;

    // Make necesssary folders

    if (true)
    {
        struct stat st = {0};

        std::string name = save_path;
        //save_path[save_path.size()] = ' ';

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(save_path.c_str(), 0700);
        }

        name = save_path + "best_trans_dis_san";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

        name = save_path + "best_trans_san";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

        name = save_path + "initial_guess_dis_san";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

        name = save_path + "trans_dis_san";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

    }



    if (!load_pointcloud_map<PointXYZIT>(map_file_path.c_str(), map_file_name.c_str(), map_load_mode, map_ptr))
    {
        std::cout << "error occured while loading map from following path :"<< std::endl;
        //std::cout << "map_file_path << std::endl;
    }
    else
    {
        map_loaded = true;
    }

    std::cout << map_ptr->size() << std::endl;
    map_loaded = true;

    current_pose.x = x_startpoint;
    current_pose.y = y_startpoint;
    current_pose.z = z_startpoint;
    current_pose.roll = roll_startpoint;
    current_pose.pitch = pitch_startpoint;
    current_pose.yaw = yaw_startpoint;

    previous_pose = current_pose;

    offset.x = 0.0;
    offset.y = 0.0;
    offset.z = 0.0;
    offset.roll = 0.0;
    offset.pitch = 0.0;
    offset.yaw = 0.0;

    carPoseList.push_back(current_pose);

    // Setting NDT parameters to default values
    ndt.setMaximumIterations(iter);
    ndt.setResolution(gridSize);
    ndt.setStepSize(step_size);
    ndt.setTransformationEpsilon(trans_eps);

    // Setting point cloud to be aligned to.
    ndt.setInputTarget(map_ptr);

    map_pub = nh.advertise<sensor_msgs::PointCloud2>("/map", 1000);
    initial_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/initial_scan", 1000);
    aligned_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_scan", 1000);
    best_transformed_dis_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/transformed_dis_scan", 1000);

    ndt_map_pub = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 10000);

    ros::Subscriber scan_sub = nh.subscribe("velodyne_points_groundtruth", 10000, scan_callback);


    publish_pointCloud(*map_ptr, map_pub, "map");

    std::string name = save_path + "groundtruth.csv";

    //pFileLog = fopen (strLogFile,"w");
    pFileLog = fopen (name.c_str(),"w");

    fprintf (pFileLog,
             "scan_seq, \
             current_pose.x,\
             current_pose.y,\
             current_pose.z, \
             ( current_pose.roll /M_PI) * 180.0,\
             ( current_pose.pitch /M_PI) * 180.0,\
             ( current_pose.yaw /M_PI) * 180.0,\
             max_trans_probability * 10000.0,\
             average_matching_time,\
             average_number_iteration,\
             average_trans_probability,\
             filtered_scan.size(),\
             scan.size(),\
             fabs(best_init_pose.x - current_pose.x),\
             fabs(best_init_pose.y - current_pose.y),\
             fabs(best_init_pose.z - current_pose.z),\
             (fabs(best_init_pose.yaw - current_pose.yaw)/M_PI) * 180.0,\
             aligned_pose.x, aligned_pose.y, aligned_pose.z, \
             aligned_pose.roll,\
             aligned_pose.pitch,\
             aligned_pose.yaw, \
             matching_time,\
             trans_probability,\
             iteration_number\n");

    fclose (pFileLog);

    ros::spin();

    return 0;
}

