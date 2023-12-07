import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
import math
import numpy as np

def ground(x):
    if x >= 0:
        return math.floor(x)
    else:
        return math.ceil(x)

class Get_Object_Range(Node):
    def __init__(self):
        super().__init__('get_object_range')
        self.__coordinate_subscriber = self.create_subscription(Float64MultiArray, 'object_coordinates',
                                                                self.coordinate_callback, 10)
        self.__lidar_subscriber = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        self.__position_publisher = self.create_publisher(Float64MultiArray, 'position', 10)
        self.__boundary_publisher = self.create_publisher(Float64MultiArray,'boundary',10)
        self.object_coordinates = None
        # camera config
        self.perception_angle = 62.2
        # deviation of lidar start angle from 0
        self.error_angle = 0.0

    def coordinate_callback(self, Float64MultiArray):
        coordinate_list = Float64MultiArray.data
        if coordinate_list[0] == -1:
            self.object_coordinates = None
            self.get_logger().info('No Target Found')
        else:
            self.object_coordinates = coordinate_list


    def lidar_callback(self, LaserScan):
        position_msg = Float64MultiArray()
        boundary_msg = Float64MultiArray()
        if self.object_coordinates is None:
            position_msg.data = [0.0]
            boundary_msg.data = [0.0]
        else:
            angle_min = LaserScan.angle_min
            angle_max = LaserScan.angle_max
            angle_increment = 1
            range_min = LaserScan.range_min
            range_max = LaserScan.range_max
            ranges = LaserScan.ranges

            x_left = self.object_coordinates[0]
            x_right = self.object_coordinates[2]
            center = (x_left + x_right) / 2.0
            length = self.object_coordinates[4]
            theta_center = -(center / length - 0.5) * self.perception_angle
            if theta_center > 0:
                self.get_logger().info('Target on the Left')
            elif theta_center == 0:
                self.get_logger().info('Target in the middle')
            else:
                self.get_logger().info('Target on the Right')

            theta_left = -(x_left / length - 0.5) * self.perception_angle * 0.7
            theta_right = -(x_right / length - 0.5) * self.perception_angle * 0.7
            boundary_msg.data = [float(theta_left),float(theta_right)]
            theta_left -= self.error_angle
            theta_right -= self.error_angle
            left_bound = ground(theta_left / angle_increment * 1.0)
            right_bound = ground(theta_right / angle_increment * 1.0)
            if left_bound == right_bound:
                sample_list = ranges[left_bound]
            else:
                if left_bound * right_bound >= 0:
                    sample_list = ranges[right_bound:left_bound]
                    if left_bound == 0:
                        sample_list = ranges[right_bound:left_bound - 1]
                else:
                    sample_list = ranges[0:left_bound] + ranges[right_bound:]

            sample = np.array(sample_list)
            distance = np.nanmean(sample)
            self.get_logger().info('Target Is ' + str(distance) + 'Meters away')
            position_msg.data = [float(distance),float(theta_center)]
        self.__position_publisher.publish(position_msg)
        self.__boundary_publisher.publish(boundary_msg)


def main():
    rclpy.init()
    get_object_range_node = Get_Object_Range()
    rclpy.spin(get_object_range_node)
    get_object_range_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()







