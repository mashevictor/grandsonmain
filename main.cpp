#include <pcl/common/time.h> //fps calculations
#include <pcl/common/angles.h>
#include <pcl/io/openni2_grabber.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/boost.h>
#include <pcl/visualization/range_image_visualizer.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/range_image/range_image.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/surface/convex_hull.h>

#include <boost/chrono.hpp>

#include "pcl/io/openni2/openni.h"

#include <chrono>
#include <thread>

template <typename PointType>
class OpenNI2Viewer
{
public:
  typedef pcl::PointCloud<PointType> Cloud;
  typedef typename Cloud::ConstPtr CloudConstPtr;
  typedef typename Cloud::Ptr CloudPtr;

  OpenNI2Viewer (pcl::io::OpenNI2Grabber& grabber)
    : cloud_viewer_ (new pcl::visualization::PCLVisualizer ("PCL OpenNI2 cloud"))
    , grabber_ (grabber)
  {
  }

  void
  cloud_callback (const CloudConstPtr& cloud)
  {
    boost::mutex::scoped_lock lock (cloud_mutex_);
    cloud_ = cloud;
  }

  void image_callback (const pcl::io::OpenNI2Grabber::DepthImage::Ptr image)
  {
    boost::mutex::scoped_lock lock (image_mutex_);
    image_ = image;
  }

  
  /**
  * @brief starts the main loop
  */
  void
  run ()
  {
    boost::function<void (const CloudConstPtr&) > cloud_cb = boost::bind (&OpenNI2Viewer::cloud_callback, this, _1);
    boost::signals2::connection cloud_connection = grabber_.registerCallback (cloud_cb);
 
    boost::function<void (const pcl::io::OpenNI2Grabber::DepthImage::Ptr) > image_cb = boost::bind (&OpenNI2Viewer::image_callback, this, _1);
    boost::signals2::connection image_connection = grabber_.registerCallback (image_cb);

    bool cloud_init = false;

    grabber_.start ();

    while (!cloud_viewer_->wasStopped ())
    {
      CloudConstPtr cloud;

      // See if we can get a cloud
      if (cloud_mutex_.try_lock ())
      {
        cloud_.swap (cloud);
        cloud_mutex_.unlock ();
      }

      if (cloud)
      {
        CloudPtr filtered(new Cloud);
        CloudPtr cloud_plane(new Cloud);
        CloudPtr cloud_rest(new Cloud);
        CloudPtr cloud_abovePlane(new Cloud);

        pcl::VoxelGrid<PointType> p;
        p.setInputCloud (cloud);
        p.setLeafSize (0.05f, 0.05f, 0.05f);

        p.filter(*filtered);

        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients (true);
        // Mandatory
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setMaxIterations (1000);
        seg.setDistanceThreshold (0.05);

        // Create the filtering object
        pcl::ExtractIndices<pcl::PointXYZ> extract;

        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (filtered);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
          std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
          break;
        }

        // Extract the outliers
        extract.setInputCloud (filtered);
        extract.setIndices (inliers);
        extract.setNegative (true);
        extract.filter (*cloud_rest);

        pcl::ProjectInliers<pcl::PointXYZ> proj;
        proj.setModelType (pcl::SACMODEL_PLANE);
        proj.setIndices (inliers);
        proj.setInputCloud (filtered);
        proj.setModelCoefficients (coefficients);
        proj.filter (*cloud_plane);

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(cloud_plane, 0, 255, 0);
        if (!cloud_viewer_->updatePointCloud (cloud_plane, green, "Planar"))
        {
          cloud_viewer_->addPointCloud (cloud_plane, green, "Planar");
        }

        CloudPtr cloud_hull (new Cloud);
        pcl::ConvexHull<pcl::PointXYZ> chull;
        chull.setInputCloud (cloud_plane);
        chull.reconstruct (*cloud_hull);

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(cloud_plane, 0, 0, 255);
        if (!cloud_viewer_->updatePointCloud (cloud_hull, blue, "Hull"))
        {
          cloud_viewer_->addPointCloud (cloud_hull, blue, "Hull");
        }

        pcl::ExtractPolygonalPrismData<pcl::PointXYZ> eppd;
        eppd.setInputCloud (cloud_rest);
        eppd.setInputPlanarHull (cloud_hull);
        eppd.setHeightLimits (0.0, 0.5);
        eppd.segment(*inliers);

        // Extract the outliers
        extract.setInputCloud (cloud_rest);
        extract.setIndices (inliers);
        extract.setNegative (false);
        extract.filter (*cloud_abovePlane);

        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud_rest, 255, 0, 0);
        //if (!cloud_viewer_->updatePointCloud (cloud_rest, red, "Rest"))
        //{
        //  cloud_viewer_->addPointCloud (cloud_rest, red, "Rest");
        //}

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> yellow(cloud_abovePlane, 255, 255, 0);
        if (!cloud_viewer_->updatePointCloud (cloud_abovePlane, yellow, "Above"))
        {
          cloud_viewer_->addPointCloud (cloud_abovePlane, yellow, "Above");
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud (cloud_abovePlane);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance (0.1); // 10cm
        ec.setMinClusterSize (20);
        ec.setMaxClusterSize (25000);
        ec.setSearchMethod (tree);
        ec.setInputCloud (cloud_abovePlane);
        ec.extract (cluster_indices);

        int i = 0;
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {
          pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
          for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
            cloud_cluster->points.push_back (cloud_abovePlane->points[*pit]);
          cloud_cluster->width = cloud_cluster->points.size ();
          cloud_cluster->height = 1;
          cloud_cluster->is_dense = true;

          pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> random(cloud_cluster);
          std::stringstream name;
          name << "Cluster" << i;
          if (!cloud_viewer_->updatePointCloud (cloud_cluster, random, name.str()))
          {
            cloud_viewer_->addPointCloud (cloud_cluster, random, name.str());
          }
          i++;
        }
      }
      cloud_viewer_->spinOnce();
    }

    grabber_.stop ();

    cloud_connection.disconnect ();
  }

  boost::shared_ptr<pcl::visualization::PCLVisualizer> cloud_viewer_;

  pcl::io::OpenNI2Grabber& grabber_;

  boost::mutex cloud_mutex_;
  CloudConstPtr cloud_;

  boost::mutex image_mutex_;
  pcl::io::OpenNI2Grabber::DepthImage::Ptr image_;
};


int
main (int argc, char** argv)
{
  std::string device_id ("");
  pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
  pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;

  boost::shared_ptr<pcl::io::openni2::OpenNI2DeviceManager> deviceManager = pcl::io::openni2::OpenNI2DeviceManager::getInstance ();
  if (deviceManager->getNumOfConnectedDevices () > 0)
  {
    boost::shared_ptr<pcl::io::openni2::OpenNI2Device> device = deviceManager->getAnyDevice ();
    cout << "Device ID not set, using default device: " << device->getStringID () << endl;
  }

  pcl::io::OpenNI2Grabber grabber (device_id, depth_mode, image_mode);

  OpenNI2Viewer<pcl::PointXYZ> openni_viewer (grabber);
  openni_viewer.run ();

  return (0);
}
