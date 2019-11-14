/*
 This code can be used to get ground truth of map matching

 Ehsan Javanmardi

 2017.12.15
    ndt_groundtruth_generator_v1.0
    based on ndt_3D_mapmatching_groundtruth_v1.2

    Change log :
    1. Use XYZI point type


 TETS :
    code is not tested yet

 CHANGE LOG :
    Based on ndt_3D_mapmatching_groundtruth_v1.1
    Use new point type PointXYZI does delta_t and disotrtion
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
//#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <math.h>
#include <boost/filesystem.hpp>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <time.h>
#include <sstream>


#include <ndt/visual_ndt.h>
#include "kmj_self_driving_common.h"

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
    double score_tf_initial;
    double trans_probability_tf_initial; //score per point fpr initial pose
    double trans_probability_tf_align; //score per point for final pose
    double matching_time;
};

pose initial_pose, predict_pose, previous_pose, ndt_pose;
pose current_pose, control_pose, localizer_pose, previous_gnss_pose, current_gnss_pose;
pose offset; // current_pos - previous_pose

// If the map is loaded, map_loaded will be true.
bool map_loaded = false;

NormalDistributionsTransform_Visual<PointXYZI, PointXYZI> ndt;

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
ros::Publisher iter_initial_scan_pub;
ros::Publisher iter_aligned_scan_pub;
ros::Publisher best_align_pub;

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
std::string map_file_name = "map";
std::string save_path = "/home/ehsan/workspace/results/groundtruth";
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

pcl::PointCloud<PointXYZI>::Ptr map_ptr (new pcl::PointCloud<PointXYZI>);

// these tow variable is for log file only to show where the point cloud is saved
std::string savedMap ="";
std::string savedRoadMarkingWindow = "";

std::vector<pose> carPoseList;

int scan_seq;

double displacement_threshold = 0.0;

double steps = 0.05;
double yaw_steps = 0.5; // in degree
int yaw_iteration = 10;
int x_iteration = 10;
int y_iteration = 10;

double lidar_range = 90.0;

static void scan_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{
    scan_seq = input->header.seq;
    current_scan_time = input->header.stamp;

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

        pcl::VoxelGridCovariance<PointXYZI> target_cells;
        ndt.getCells(target_cells);

        //pcl::VoxelGridCovariance<pcl::PointXYZ>::Leaf leaf_;
        //std::map<size_t, pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf> leaves;
        //leaves = target_cells.getLeaves();
        //std::vector<Leaf> leafList;
        //getLeaves(leafList);

        typedef pcl::VoxelGridCovariance<PointXYZI> VectorCovarianceXYZ;
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
        setCovarianceListMarker<pcl::VoxelGridCovariance<PointXYZI>::Leaf>(leafList, ndtSphereList, \
                                                      4.605 ,"map", normalDistribution_color, 20);

        ndt_map_pub.publish(ndtSphereList);

        show_map = 0;
    }

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

    pcl::PointCloud<PointXYZI> scan;

    for (pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::const_iterator item = calibrated_scan_xyzir.begin(); \
         item != calibrated_scan_xyzir.end(); item++)
    {
        PointXYZI p;

        p.x = (double) item->x;
        p.y = (double) item->y;
        p.z = (double) item->z;
        p.intensity = (double) item->intensity;
        //p.ring = item->ring;

        if (getR(p) > 1.0 && getR(p) < lidar_range)
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
        pcl::PointCloud<PointXYZI>  predicted_scan;

        pcl::transformPointCloud(scan, predicted_scan, tf_predict);

        publish_pointCloud(predicted_scan, iter_initial_scan_pub, "map");
        show_initial_scan = false;
    }

    // DOWNSAMPLE SCAN USING VOXELGRID FILTER ###############################################################

    pcl::PointCloud<PointXYZI>::Ptr input_cloud_ptr(new pcl::PointCloud<PointXYZI>(scan));
    pcl::PointCloud<PointXYZI> filtered_scan;

    pcl::ApproximateVoxelGrid<PointXYZI> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_grid_filter.setInputCloud(input_cloud_ptr);
    voxel_grid_filter.filter(filtered_scan);

    align_start = std::chrono::system_clock::now();    // TIME $$$$$$$$$$$$$$$$$$$$$$$$$$$

    pcl::PointCloud<PointXYZI>::Ptr filter_scan_ptr(new pcl::PointCloud<PointXYZI>(filtered_scan));

    ndt.setInputSource(filter_scan_ptr);

    pcl::PointCloud<PointXYZI> aligned_scan;

    ndt.align(aligned_scan, tf_predict);

    Eigen::Matrix4f tf_align(Eigen::Matrix4f::Identity()); // base_link
    tf_align = ndt.getFinalTransformation(); // localizer
    double trans_probability = ndt.getTransformationProbability();
    int iteration_number = ndt.getFinalNumIteration();

    align_end = std::chrono::system_clock::now();    // TIME $$$$$$$$$$$$$$$$$$$$$$$$$$$

    double matching_time = std::chrono::duration_cast<std::chrono::microseconds>\
            (align_end - align_start).count()/1000.0; // double



    // iteration for finding more accurate results

    Eigen::Matrix4f tf_align_best;
    Eigen::Matrix4f tf_initial_guess_best;
    double trans_probability_align_best = -1000.0;

    pcl::PointCloud<PointXYZI> best_aligned_scan;

    std::vector<iter_result> iter_results;

    bool calculate_ground_truth = true;

    double displacement = sqrt(pow(tf_align(0,3) - tf_previous(0,3),2) + \
                               pow(tf_align(1,3) - tf_previous(1,3),2) + \
                               pow(tf_align(2,3) - tf_previous(2,3),2));

    // store in file

    if (displacement >= displacement_threshold)
    {
        calculate_ground_truth = true;

        for (int yaw_iter=-1 * yaw_iteration; yaw_iter <= yaw_iteration; yaw_iter++) // in degree
            for (int x_iter= -1 * x_iteration; x_iter <= x_iteration; x_iter++)
                for (int y_iter=-1 * y_iteration; y_iter <= y_iteration; y_iter++)
                {
                    align_start = std::chrono::system_clock::now();    // TIME $$$$$$$$$$$$$$$$$$$$$$$$$$$

                    ndt.setInputSource(filter_scan_ptr);

                    pose temp_pose;

                    tf_to_pose(tf_align, temp_pose);

                    temp_pose.x += (double)x_iter * steps;
                    temp_pose.y += (double)y_iter * steps;

                    double yaw_r = (yaw_steps /180.0 ) * M_PI;

                    temp_pose.yaw += (((double)yaw_iter * yaw_r));

                    Eigen::Matrix4f tf_initial_iter;

                    pose_to_tf(temp_pose, tf_initial_iter);

                    pcl::PointCloud<PointXYZI> iter_aligned_scan;

                    iter_result iter_result_;

                    iter_result_.tf_initial = tf_initial_iter;
                    iter_result_.score_tf_initial = ndt.calculateScore(*filter_scan_ptr, tf_initial_iter);
                    iter_result_.trans_probability_tf_initial = iter_result_.score_tf_initial / ((double)filter_scan_ptr->size());

                    ndt.align(iter_aligned_scan, tf_initial_iter);

                    align_end = std::chrono::system_clock::now();    // TIME $$$$$$$$$$$$$$$$$$$$$$$$$$$

                    iter_result_.matching_time = std::chrono::duration_cast<std::chrono::microseconds>\
                            (align_end - align_start).count()/1000.0; // double

                    iter_result_.tf_align = ndt.getFinalTransformation();
                    iter_result_.number_iter = ndt.getFinalNumIteration();
                    iter_result_.trans_probability_tf_align = ndt.getTransformationProbability();

                    iter_results.push_back(iter_result_);

                    std::cout << "-------- iteration for scan time : " << current_scan_time.toSec()  << " -------- \n";

                    std::cout << "iter_result_.tf_initial \n" << iter_result_.tf_initial << std::endl;
                    std::cout << "iter_result_.score_tf_initial --> " << iter_result_.score_tf_initial << std::endl;
                    std::cout << "iter_result_.trans_probability_tf_initial -->" << iter_result_.trans_probability_tf_initial << std::endl;
                    std::cout << "iter_result_.tf_align --> " << iter_result_.tf_align << std::endl;
                    std::cout << "iter_result_.number_iter --> " << iter_result_.number_iter << std::endl;



                    if (trans_probability_align_best < iter_result_.trans_probability_tf_align)
                    {
                        tf_initial_guess_best = tf_initial_iter;
                        tf_align_best = ndt.getFinalTransformation();
                        trans_probability_align_best = iter_result_.trans_probability_tf_align;
                        best_aligned_scan = iter_aligned_scan;


                        std::cout << "$$$$$$$$$$ BEST SCORE IS UPDATED. SCORE is " << trans_probability_align_best\
                                  << "$$$$$$$$$$" << std::endl;

                        std::cout << "best tf is : \n" <<  tf_align_best << std::endl << std::endl;

                        publish_pointCloud(best_aligned_scan, iter_aligned_scan_pub, "map");
                    }
                }
    }
    else
    {
        calculate_ground_truth = false;
        tf_align_best = tf_align;
        std::cout << "\033[1;42m ground truth is not calculated \033[0m"<< std::endl;
    }

    // get the global translation matrix

    tf_to_pose(tf_align_best, current_pose);

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
            average_trans_probability += iter_results[i].trans_probability_tf_initial;
        }

        average_matching_time = average_matching_time / (double) iter_results.size();
        average_number_iteration = average_number_iteration / (double) iter_results.size();
        average_trans_probability = average_trans_probability / (double) iter_results.size();

        pose align_pose;
        tf_to_pose(tf_align, align_pose);

        pose best_align_pose;
        tf_to_pose(tf_align_best, best_align_pose);

        pose best_initial_guess_pose;
        tf_to_pose(tf_initial_guess_best, best_initial_guess_pose);

        std::string name = save_path + "groundtruth.csv";

        pFileLog = fopen (name.c_str(),"a");

        fprintf (pFileLog, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i,%i\n",\
                 current_scan_time.toSec(), \
                 best_align_pose.x, best_align_pose.y, best_align_pose.z, \
                 ( best_align_pose.roll /M_PI) * 180.0, ( best_align_pose.pitch /M_PI) * 180.0, ( best_align_pose.yaw /M_PI) * 180.0,
                 trans_probability_align_best * 10000.0,\
                 average_matching_time, average_number_iteration, average_trans_probability,\
                 filtered_scan.size(), scan.size(),\
                 fabs(best_initial_guess_pose.x - best_align_pose.x), fabs(best_initial_guess_pose.y - best_align_pose.y),\
                 fabs(best_initial_guess_pose.z - best_align_pose.z), (fabs(best_initial_guess_pose.yaw - best_align_pose.yaw)/M_PI) * 180.0,\
                 sqrt(pow(best_initial_guess_pose.x - best_align_pose.x, 2)+\
                      pow(best_initial_guess_pose.y - best_align_pose.y, 2)+\
                      pow(best_initial_guess_pose.z - best_align_pose.z, 2)),\
                 best_initial_guess_pose.x, best_initial_guess_pose.y, best_initial_guess_pose.z, \
                 (best_initial_guess_pose.roll/M_PI) * 180.0, (best_initial_guess_pose.pitch/M_PI) * 180.0, (best_initial_guess_pose.yaw/M_PI) * 180.0, \
                 align_pose.x, align_pose.y, align_pose.z, \
                 align_pose.roll, align_pose.pitch, align_pose.yaw, \
                 matching_time, trans_probability, iteration_number,\
                 calculate_ground_truth);

        fclose (pFileLog);
    }

    // save score distribution




    // SAVE 2D ERROR HISTOGRAM FOR BEST YAW ANGLE ##############################################################

    if (true)
    {
        std::string name = save_path + "error_hitogram_best_yaw_" + \
                           std::to_string(current_scan_time.toSec()) + ".csv";

        FILE *pFile;
        pFile = fopen(name.c_str(), "w");

        for (int i=0; i<iter_results.size(); i++)
        {
            iter_result item = iter_results[i];

            Eigen::Matrix4f tf_align = item.tf_align;
            Eigen::Matrix4f tf_initial = item.tf_initial;

            // if best yaw
            if ( tf_initial_guess_best(0,0) == tf_initial(0,0) )
            {
                pose initial_pose;
                tf_to_pose(tf_initial, initial_pose);

                double error_3d = sqrt(pow((tf_align(0,3)- tf_initial(0,3)),2) +\
                                       pow((tf_align(1,3)- tf_initial(1,3)),2) +\
                                       pow((tf_align(2,3)- tf_initial(2,3)),2));

                fprintf(pFile, "%f,%f,%f,%f\n", tf_initial(0,3), tf_initial(1,3), error_3d, item.score_tf_initial);
            }
        }

        fclose(pFile);
    }

    // SAVE YAW ERROR HISTOGRAM FOR BEST X AND Y
/*
    if (true)
    {
        std::string name = save_path + "error_hitogram_best_pose_" + \
                           std::to_string(current_scan_time.toSec()) + ".csv";

        FILE *pFile;
        pFile = fopen(name.c_str(), "w");

        for (int i=0; i<iter_results.size(); i++)
        {
            iter_result item = iter_results[i];

            Eigen::Matrix4f tf_align = item.tf_align;
            Eigen::Matrix4f tf_initial = item.tf_initial;

            // if best yaw
            if ( tf_align_best(0,0) == tf_align(0,0) )
            {
                double error_3d = sqrt(pow((tf_align(0,3)- tf_initial(0,3)),2) +\
                                       pow((tf_align(1,3)- tf_initial(1,3)),2) +\
                                       pow((tf_align(2,3)- tf_initial(2,3)),2));

                fprintf(pFile, "%f,%f,%f,%f\n", tf_initial(0,3), tf_initial(1,3), error_3d, error_3d);
            }
        }

        fclose(pFile);
    }*/

    if (true)
    {

        pcl::PointCloud<PointXYZI> best_aligned_scan;

        pcl::transformPointCloud(scan, best_aligned_scan, tf_align_best);

        best_aligned_scan.height = 1;
        best_aligned_scan.width = best_aligned_scan.size();
        best_aligned_scan.points.resize (best_aligned_scan.width * best_aligned_scan.height);

        double error_3d = sqrt(pow(tf_align_best(0,3) - tf_align(0,3),2)+\
                               pow(tf_align_best(1,3) - tf_align(1,3),2)+\
                               pow(tf_align_best(2,3) - tf_align(2,3),2));


        std::string name = save_path + "best_aligned_scan_" + \
                           std::to_string(current_scan_time.toSec()) + ".csv";



        // for saving in linux we use csv file
        FILE *pFile;
        pFile = fopen(name.c_str(), "w");



        for (int i=0; i< best_aligned_scan.size(); i++)
        {
                PointXYZI p;
                p = best_aligned_scan[i];

                fprintf(pFile, "%f,%f,%f,%f\n", p.x, p.y, p.z, p.intensity, error_3d);
        }

        fclose(pFile);

       // writer.write(name, best_aligned_scan, false);
    }

    // SAVE THE FIRST ITERATION OF ALIGNMENT #####################################################

    if (true)
    {

        pcl::PointCloud<PointXYZI> aligned_scan;

        pcl::transformPointCloud(scan, aligned_scan, tf_align);

        aligned_scan.height = 1;
        aligned_scan.width = aligned_scan.size();
        aligned_scan.points.resize (aligned_scan.width * aligned_scan.height);

        std::string name = save_path + "aligned_scan_" + \
                           std::to_string(current_scan_time.toSec()) + ".csv";

        FILE *pFile;
        pFile = fopen(name.c_str(), "w");

        for (int i=0; i< aligned_scan.size(); i++)
        {
            PointXYZI p;
            p = aligned_scan[i];

            fprintf(pFile, "%f,%f,%f,%f,%f\n", p.x, p.y, p.z, p.intensity);
        }

        fclose(pFile);
    }
}


int main(int argc, char **argv)
{
    std::cout << "ndt_groundtruth_generatorv1_0\n" ;
    ros::init(argc, argv, "ndt_groundtruth_generatorv1_0");


    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    skipSeq = 0;

    private_nh.getParam("x_startpoint", x_startpoint);
    private_nh.getParam("y_startpoint", y_startpoint);
    private_nh.getParam("z_startpoint", z_startpoint);
    private_nh.getParam("roll_startpoint", roll_startpoint);
    private_nh.getParam("pitch_startpoint", pitch_startpoint);
    private_nh.getParam("yaw_startpoint", yaw_startpoint);

    yaw_startpoint = (roll_startpoint /180.0) * M_PI; //in radian
    yaw_startpoint = (pitch_startpoint /180.0) * M_PI; //in radian
    yaw_startpoint = (yaw_startpoint /180.0) * M_PI; //in radian


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

    private_nh.getParam("steps", steps);
    private_nh.getParam("yaw_steps", yaw_steps);
    private_nh.getParam("yaw_iteration", yaw_iteration);
    private_nh.getParam("x_iteration", x_iteration);
    private_nh.getParam("y_iteration", y_iteration);
    private_nh.getParam("lidar_range", lidar_range);


    private_nh.getParam("map_file_name", map_file_name);
    if (map_load_mode == 1 && map_file_name == "")
    {
        std::cout << "map load mode is one but the map name is null" << std::endl;
    }


    if (!load_pointcloud_map<PointXYZI>(map_file_path.c_str(), map_file_name.c_str(), map_load_mode, map_ptr))
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

    map_pub = nh.advertise<sensor_msgs::PointCloud2>("/groundtruth/map", 1000);
    iter_initial_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/groundtruth/iter_initial_scan", 1000);
    iter_aligned_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/groundtruth/iter_aligned_scan", 1000);
    best_align_pub = nh.advertise<sensor_msgs::PointCloud2>("/groundtruth/best_aligns", 1000);

    ndt_map_pub = nh.advertise<visualization_msgs::MarkerArray>("/groundtruth/ndt_map", 10000);

    ros::Subscriber scan_sub = nh.subscribe("/groundtruth/velodyne_scan", 10000, scan_callback);

    publish_pointCloud(*map_ptr, map_pub, "map");


    time_t timer;
    time(&timer);

    std::stringstream ss;
    ss << timer;
    std::string str_time = ss.str();

    save_path = save_path + str_time + "/";

    struct stat st = {0};

    std::string path_name = save_path;

    if (stat(path_name.c_str(), &st) == -1)
    {
        mkdir(save_path.c_str(), 0700);
    }


    std::string file_name = save_path + "groundtruth.csv";

    //pFileLog = fopen (strLogFile,"w");
    pFileLog = fopen (file_name.c_str(),"w");

    fprintf (pFileLog,
             "scan time, \
             best_align.x,\
             best_align.y,\
             best_align.z, \
             best_align.roll_deg,\
             best_align.pitch_deg,\
             best_align.yaw_deg,\
             trans_probability_align_best * 10000.0,\
             average_matching_time,\
             average_number_iteration,\
             average_trans_probability,\
             filtered_scan.size(),\
             scan.size(),\
             fabs(best_initial_guess_pose.x - current_pose.x),\
             fabs(best_initial_guess_pose.y - current_pose.y),\
             fabs(best_initial_guess_pose.z - current_pose.z),\
             (fabs(best_initial_guess_pose.yaw - current_pose.yaw)/M_PI) * 180.0,\
             3d_error between align and best align,\
             best_initial_guess_pose.x, best_initial_guess_pose.y, best_initial_guess_pose.z,\
             best_initial_guess_pose.roll_deg, best_initial_guess_pose.pitch_deg, best_initial_guess_pose.yaw_deg,\
             align.x, align.y, align.z, \
             align.roll,\
             align.pitch,\
             align.yaw, \
             matching_time,\
             trans_probability,\
             iteration_number\n");

    fclose (pFileLog);

    ros::spin();

    return 0;
}

