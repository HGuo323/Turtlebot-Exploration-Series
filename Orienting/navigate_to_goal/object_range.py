import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, String
from rclpy.qos import qos_profile_sensor_data
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
        self.parameter_config = {
            # forward sample angles, ranging from -forward_angle to forward_angle
            'forward_angle': 175,
            'reaction_period': 4
        }
        self.__lidar_subscriber = self.create_subscription(LaserScan, '/scan', self.lidar_callback,
                                                           qos_profile_sensor_data)
        self.__range_publisher = self.create_publisher(Float64MultiArray, 'direction_range', 10)
        self.__parameter_subscriber = self.create_subscription(String, 'parameter_change', self.parameter_callback, 10)
        self.reaction_pointer = 0
        self.current_ranges = 0

    def parameter_callback(self, String):
        information = String.data
        name, value = information.split(' ')
        if name in self.parameter_config:
            self.parameter_config[name] = float(value)
            self.get_logger().info('Changed ' + name + ' to ' + value)
            self.get_logger().info("Current parameters:")
            for key in self.parameter_config:
                self.get_logger().info(key + ':' + str(self.parameter_config[key]))
        self.get_logger().info('\n')

    def lidar_callback(self, LaserScan):
        angle_min = LaserScan.angle_min
        angle_max = LaserScan.angle_max
        angle_increment = LaserScan.angle_increment
        range_min = LaserScan.range_min
        range_max = LaserScan.range_max
        ranges = LaserScan.ranges
        left = ranges[:self.parameter_config['forward_angle']]
        left.reverse()
        right = ranges[-self.parameter_config['forward_angle']:]
        right.reverse()
        forward_samples = np.array(left + right)

        self.reaction_pointer += 1
        self.current_ranges += forward_samples
        if self.reaction_pointer >= self.parameter_config['reaction_period']:
            forward_samples = self.current_ranges / self.parameter_config['reaction_period']

            forward_samples = np.concatenate((np.expand_dims(forward_samples,0),np.expand_dims(np.arange(forward_samples.shape[-1]),0)),0)


            clean_forward = forward_samples[:,np.where(~np.isnan(forward_samples[0]))].squeeze(axis=1)


            obstacle_position_msg = Float64MultiArray()
            if clean_forward.shape[-1] == 0:
                obstacle_position_msg.data = [-1.0]
            else:
                index = np.argmin(clean_forward[0, :])
                forward_obstacle_angle = clean_forward[1,index]
                forward_obstacle_range = float(clean_forward[0,index])
                forward_obstacle_angle_raid = float(np.radians(self.parameter_config['forward_angle'] - forward_obstacle_angle))
                obstacle_position_msg.data = [forward_obstacle_range,forward_obstacle_angle_raid]

            self.__range_publisher.publish(obstacle_position_msg)
            self.reaction_pointer = 0
            self.current_ranges = 0


def main():
    rclpy.init()
    get_object_range_node = Get_Object_Range()
    rclpy.spin(get_object_range_node)
    get_object_range_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
