#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
import message_filters

from object_detection.msg import DetectedObjectMsg, DetectedObjectsMsg
from utils import utils
from object_detectors.object_detector_main import ObjectDetector

# main detection class that listens to image and depthimage topics to output detections
# look at config/ros_config.yaml for topic and other ros parameters

class ObjectDetector3D: 
    def __init__(self):
        self.load_configs()
        rospy.loginfo("Waiting for ML model to load")
        self.object_detector = ObjectDetector()
        self.robot_id = rospy.get_param("robot_id")
        self.init_subscribers()

        self.init_publishers()
        self.bridge = CvBridge()
        self.detected_objects = []
        self.subsampling_counter = 0
        self.trigger_publish_output = False


    def load_configs(self):
        self.ros_config = utils.get_config('ros_config.yaml')
        self.subscribers = self.ros_config['subscribers']
        self.publishers = self.ros_config['publishers']
        self.subsampling = self.ros_config['subsampling']


        rospy.loginfo("ROS configs loaded successfully")

    def init_subscribers(self):
        camera_2d = self.subscribers['camera_2d']
        camera_3d = self.subscribers['camera_3d']
        camera_3d_info = self.subscribers['camera_3d_info']

        self.image_sub = message_filters.Subscriber('/r'+str(self.robot_id)+camera_2d['topic'], Image, buff_size=2**28)
        self.depthimage_sub = message_filters.Subscriber('/r'+str(self.robot_id)+camera_3d['topic'], Image, buff_size=2**28)
        self.depthimage_info_sub = rospy.Subscriber('/r'+str(self.robot_id)+camera_3d_info['topic'], CameraInfo, self.info_cb)

        # try to synchronize the 2 topics as closely as possible for higher XYZ accuracy
        self.time_sync = message_filters.ApproximateTimeSynchronizer([self.image_sub, \
            self.depthimage_sub], queue_size=self.subscribers['synchronized_queue_size'], slop=0.03)
        self.time_sync.registerCallback(self.callback)

        rospy.loginfo("Subscribers Synchronized Successfully")

    def init_publishers(self):
        self.detected_objects_3d_pos_pub = rospy.Publisher(self.publishers['detected_objects']['topic'], \
                DetectedObjectsMsg, queue_size=self.publishers['detected_objects']['queue_size'])

        if (self.ros_config['enable_debug_image']):
            self.debug_img_pub = rospy.Publisher(self.publishers['debug_image']['topic'], \
                Image, queue_size=self.publishers['debug_image']['queue_size'])
            self.objects_markers_pub = rospy.Publisher(self.publishers['objects_markers']['topic'], \
                MarkerArray, queue_size=self.publishers['objects_markers']['queue_size'])

    def image_cb(self, ros_img):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
            self.cv_image_timestamp = ros_img.header.stamp
            self.process()
            self.publish_output()
        except CvBridgeError as e:
            print(e)
        

    def depthimage_cb(self, ros_depthimage):
        self.depthimage = ros_depthimage
        self.depthimage_timestamp= ros_depthimage.header.stamp

    def info_cb(self, info):
        self.width = info.width
        self.height = info.height
        self.focal_x = info.K[0]
        self.focal_y = info.K[4]

    def callback(self, ros_img, ros_depthimage):
        self.image_cb(ros_img)
        self.depthimage_cb(ros_depthimage)

    def convert_depth_pixel_to_metric_coordinate(self, depth, pixel_x, pixel_y, width, height, focal_x, focal_y):
        X = (pixel_x - width/2)/focal_x *depth
        Y = (pixel_y - height/2)/focal_y *depth
        return X, Y, depth

    # perform object detection and calculate xyz coordinate of object from depthimage
    def process(self):
        self.detected_objects = []

        horz_bar_text = "-"*20
        rospy.loginfo(horz_bar_text)

        if hasattr(self,'depthimage'):
            time_diff = abs(self.cv_image_timestamp-self.depthimage_timestamp)
            rospy.logdebug("Synchronization Time Delay: {:.4f}".format(time_diff.nsecs/10.0**9))

        max_time_difference = rospy.Duration(self.subscribers['absoulte_synchronized_time_diff'])

        # double check time difference as ApproximateSync algo finds best match only, 
        # may exceed max synchronization delay threshold sometimes
        if (hasattr(self,'depthimage') and abs(self.cv_image_timestamp-self.depthimage_timestamp) < max_time_difference):
            if self.subsampling_counter >= self.subsampling:
                # Note: ML detection can take significant time
                self.detected_objects = self.object_detector.detect(self.cv_image) 

                for detected_object in self.detected_objects:
                    xyz_coord = []
                    depth_image = self.bridge.imgmsg_to_cv2(self.depthimage, desired_encoding='passthrough')
                    depth = depth_image[detected_object.center[1], detected_object.center[0]]
                    real_coord = self.convert_depth_pixel_to_metric_coordinate(depth, detected_object.center[0], detected_object.center[1], self.width, self.height, self.focal_x, self.focal_y)
                    xyz_coord.append(real_coord)
                    detected_object.set_xyz(xyz_coord)
                    rospy.loginfo(detected_object)    

                self.subsampling_counter = 0
                self.trigger_publish_output = True
            else:
                self.subsampling_counter = self.subsampling_counter + 1

        else:
            rospy.logwarn("Synchronization Issue\nImage and depthimage not Synchronized")

    # publishes debug image (if specified in config/ros_config.yaml) and detected_objects custom message
    def publish_output(self):
        if self.trigger_publish_output:
            if (len(self.detected_objects) > 0):

                objects_msgs = DetectedObjectsMsg()
                header = Header()

                header.stamp = self.depthimage_timestamp
                header.frame_id = "detected_object"

                objects_msgs.header = header

                for detected_object in self.detected_objects:
                    point = Point()
                    point.x, point.y, point.z = detected_object.xyz_coord

                    msg = DetectedObjectMsg()
                    msg.point = point
                    msg.confidence = detected_object.confidence
                    msg.class_name = detected_object.class_name
                    msg.bbox = detected_object.centered_bbox
                    objects_msgs.detected_objects_msgs.append(msg)

                self.detected_objects_3d_pos_pub.publish(objects_msgs)

            if (self.ros_config['enable_debug_image']):

                inference_time_s = self.object_detector.get_inference_time()
                inference_speed_text = "{0:.2f} fps".format(1/inference_time_s)
                cv2.putText(self.cv_image, inference_speed_text, (0,20), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

                markers = MarkerArray()
                obj_id = 0

                for detected_object in self.detected_objects:
                    cv2.rectangle(self.cv_image, *detected_object.cv2_rect, (0,255,0), 3)

                    xyz_coord = str(["{:.2f}".format(pos) for pos in detected_object.xyz_coord])

                    cv2.putText(self.cv_image, detected_object.class_name, \
                        (detected_object.center[0], detected_object.center[1]), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                    cv2.putText(self.cv_image, xyz_coord, \
                        (detected_object.center[0], detected_object.center[1]+20), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    marker = Marker()
                    marker.header.frame_id = 'r'+str(self.robot_id)+'/'+str(self.ros_config['camera_frame_id'])
                    marker.header.stamp = rospy.Time()
                    marker.pose.orientation.w = 1
                    marker.pose.position.x = detected_object.xyz_coord[0]/1000.0
                    marker.pose.position.y = detected_object.xyz_coord[1]/1000.0
                    marker.pose.position.z = detected_object.xyz_coord[2]/1000.0 
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    marker.color.r = 1
                    marker.color.a = 1
                    marker.type = 2
                    marker.id = obj_id
                    obj_id = obj_id + 1
                    markers.markers.append(marker)              
                

                self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(self.cv_image, "bgr8"))
                self.objects_markers_pub.publish(markers)
                self.trigger_publish_output = False

        


# start ros node and spin up ros subscribers
def main():
    rospy.loginfo("Starting 3D Object Detector...")
    rospy.init_node('object_detection_3d', anonymous=True, log_level=rospy.INFO) # set to rospy.DEBUG for debugging

    object_detector_3d = ObjectDetector3D()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")


if __name__ == "__main__":
    main()
