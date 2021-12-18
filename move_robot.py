import sys
import rospy
from geometry_msgs.msg import Twist, PoseArray
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import euler_from_quaternion
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
import math
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
import os
import csv
import matplotlib.pyplot as plt


def read_csv(dest_file, delimiter):
    with open(dest_file, 'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               delimiter=delimiter,
                               quotechar='"')
        data = [data for data in data_iter]

        return data


class TakePhoto:
    def __init__(self, target_robot):

        self.bridge = CvBridge()
        self.image_received = False

        # Connect image topic
        img_topic = target_robot + "/camera/rgb/image_raw"
        self.image_sub = rospy.Subscriber(img_topic, Image, self.callback)

        # Allow up to one second to connection
        rospy.sleep(1)

    def callback(self, data):

        # Convert image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.image_received = True
        self.image = cv_image

    def take_picture(self, img_title):
        if self.image_received:
            # Save an image
            cv2.imwrite(img_title, self.image)
            return True
        else:
            return False


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def pred(target_robot):
    MODEL = load_model('/home/mesh/catkin_ws/src/move_robot/scripts/Robot1.h5')
    data_generator = ImageDataGenerator(rescale=1. / 255)
    test_generator = data_generator.flow_from_directory(
        directory='/home/mesh/catkin_ws/src/move_robot/pics/' + target_robot + '/',
        target_size=(300, 300),
        batch_size=5,
        class_mode=None,
        shuffle=False,
        seed=123
    )
    pre = MODEL.predict_generator(test_generator, steps=len(test_generator), verbose=1)
    predicted_class_indices = np.argmax(pre, axis=1)
    i = len(os.listdir('/home/mesh/catkin_ws/src/move_robot/pics/' + target_robot + '/'))
    TEST_DIR = '/home/mesh/catkin_ws/src/move_robot/pics/' + target_robot + '/'

    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    imgBGR = cv2.imread(TEST_DIR + test_generator.filenames[i])
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    # a if condition else b
    predicted_class = "well" if predicted_class_indices[i] else "unwell"
    ax.imshow(imgRGB)
    ax.set_title("Predicted:{}".format(predicted_class))

    return predicted_class


class TurtleController():
    def __init__(self, waypoints, target_robot):
        # State initialization
        self.x_pos = 0
        self.y_pos = 0
        self.theta = 0
        self.orientation_q = []
        self.waypoints = waypoints
        self.camera = TakePhoto(target_robot)

        # ROS initializations
        self.pub = rospy.Publisher(target_robot + '/cmd_vel', Twist,
                                   queue_size=10)  # Publish to velocity commands to the turtle topic
        self.rate = rospy.Rate(10)  # rate of publishing msg 10hz\
        rospy.Subscriber(target_robot + '/odom', Odometry,
                         callback=self.pose_callback)  # subscribe to the pose topic to get feedback on the position
        self.vel_msg = Twist()
        self.pose_array_msg = PoseArray()
        self.pic_array_msg = PoseArray()
        self.pose_array_msg.header.frame_id = "odom"
        self.pic_array_msg.header.frame_id = "odom"
        self.path_pub = rospy.Publisher(target_robot + "/path", PoseArray, queue_size=10)
        self.pic_pub = rospy.Publisher(target_robot + "/pics_location", PoseArray, queue_size=10)

        # Control initializations
        self.reached_goal = False
        self.received_odom = False

    def pose_callback(self, msg):
        # change from quaternion to euler
        self.orientation_q = msg.pose.pose.orientation
        orientation_list = [self.orientation_q.x, self.orientation_q.y, self.orientation_q.z, self.orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        self.x_pos = round(msg.pose.pose.position.x, 1)  # Get the robot's position in X
        self.y_pos = round(msg.pose.pose.position.y, 1)  # Get the robot's position in Y
        self.theta = pi_2_pi(yaw)  # Get the robot's orientation around z
        self.received_odom = True
        self.pose_array_msg.poses.append(msg.pose.pose)
        self.pose_array_msg.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.pose_array_msg)

    def get_navigation_data(self, x_desired, y_desired):
        x_delta = x_desired - self.x_pos  # Calculate the Difference in X direction
        y_delta = y_desired - self.y_pos  # Calculate the Difference in Y direction

        self.p = np.sqrt((np.square(x_delta)) + (
            np.square(y_delta)))  # Calculate the distance between the desired and the current position

        self.gamma = pi_2_pi(np.arctan2(y_delta, x_delta))  # Calculate angle between X-direction of the  Robot and p

    def motion_control(self, current_theta, desired_theta, error_dist):

        heading_error = pi_2_pi(desired_theta) - pi_2_pi(current_theta)  # Calculate the heading error
        heading_error = pi_2_pi(heading_error)
        if np.abs(heading_error) < 0.1:
            if error_dist < 0.01:
                self.linear_v = 0

                self.reached_goal = True

            else:
                self.angular_v = 0
                self.linear_v = 0.4

        elif not self.reached_goal:
            self.angular_v = 0.4 * np.sign(heading_error)
            self.linear_v = 0

    def rotate_to_look(self, x_look, y_look):
        desired_theta = pi_2_pi(np.arctan2(y_look - self.y_pos, x_look - self.x_pos))
        heading_error = pi_2_pi(desired_theta) - pi_2_pi(self.theta)
        heading_error = pi_2_pi(heading_error)
        while np.abs(heading_error) > 0.1:
            self.angular_v = pi_2_pi(0.4 * np.sign(heading_error))
            self.linear_v = 0
            self.publish_commnads(self.linear_v, self.angular_v)
            heading_error = pi_2_pi(desired_theta) - pi_2_pi(self.theta)
            heading_error = pi_2_pi(heading_error)
            self.rate.sleep()

    def path_follow(self, target_robot):
        while not self.received_odom: pass
        for idx, waypoint in enumerate(self.waypoints):
            rospy.loginfo("X: {}, Y: {}".format(self.x_pos, self.y_pos))
            rospy.loginfo("Currently going to: {}".format(waypoint))
            waypoint[0], waypoint[1] = float(waypoint[0]), float(waypoint[1])
            self.get_navigation_data(waypoint[0], waypoint[1])
            while self.p > 0.2:
                self.get_navigation_data(waypoint[0], waypoint[1])
                self.motion_control(self.theta, self.gamma, self.p)
                self.publish_commnads(self.linear_v, self.angular_v)
                self.rate.sleep()
                self.get_navigation_data(waypoint[0], waypoint[1])
            if waypoint[2] != "-" and waypoint[3] != "-":
                waypoint[2], waypoint[3] = float(waypoint[2]), float(waypoint[3])
                self.rotate_to_look(waypoint[2], waypoint[3])
                now = datetime.now()
                self.camera.take_picture(
                    "/home/mesh/catkin_ws/src/move_robot/pics/" + target_robot + "/img{}.jpg".format(now))
                self.pic_array_msg.header.stamp = rospy.Time.now()
                self.pic_pub.publish(self.pose_array_msg)
                decision = pred(target_robot)
                rospy.logwarn(decision)

            else:
                rospy.logwarn("No photos to take at this point!")

        self.publish_commnads(0, 0)  # All waypoints followed, stop the robot
        return True

    def publish_commnads(self, linear_v, angular_v):
        # Set the values of the Twist msg to be published
        self.vel_msg.linear.x = linear_v  # Linear Velocity
        self.vel_msg.angular.z = angular_v  # Angular Velocity

        # ROS Code Publisher
        self.pub.publish(self.vel_msg)  # Publish msg
        self.rate.sleep()  # Sleep with rate


def main(arg1, arg2):
    rospy.init_node('turtle_controller', anonymous=True)  # Initialize ROS node

    waypoints = read_csv('/home/mesh/catkin_ws/src/move_robot/scripts/coordinate' + str(arg2) + '.txt', delimiter=',')

    target_robot = arg1
    turtleController = TurtleController(waypoints, target_robot)

    path_done = turtleController.path_follow(target_robot)

    rospy.logwarn("Path following finished, turtlebot reached the destination.")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
