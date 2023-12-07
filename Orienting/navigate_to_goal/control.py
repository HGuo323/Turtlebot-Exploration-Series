import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Twist, Point
import numpy as np
import math


class Chase_Object(Node):
    def __init__(self):
        super().__init__('chase_object')
        self.__vector_subscriber = self.create_subscription(Point, 'vector', self.vector_callback, 10)
        self.__parameter_subscriber = self.create_subscription(String, 'parameter_change', self.parameter_callback,10)
        self.__velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.x_i = 0
        self.y_i = 0
        self.previous_x = 0
        self.previous_y = 0
        self.parameter_config = {
            "linear_k":0.25,
            "angular_k_gtg":0.005,
            'angular_k_fw':0.015,
            'linear_d':0,
            'angular_d':0,
            'linear_i':0.002,
            'angular_i':0.0002,
            "l":0.05
        }


    def parameter_callback(self, String):
        information = String.data
        name,value = information.split(' ')
        if name in self.parameter_config:
            self.parameter_config[name] = float(value)
            self.get_logger().info('Changed ' + name + ' to ' + value)
            self.get_logger().info("Current parameters:")
            for key in self.parameter_config:
                self.get_logger().info(key + ':' + str(self.parameter_config[key]))
        else:
            self.get_logger().info(name + ' is an invalid parameter.')
        self.get_logger().info('\n')


    # lab3 method
    # def vector_callback(self,Point):
    #     vector_msg = Point
    #     velocity = Twist()
    #
    #     x = vector_msg.x
    #     y = vector_msg.y
    #
    #     if x < 0:
    #         y = -y
    #
    #     # self.x_i += x
    #     # self.y_i += y
    #     x_d = x - self.previous_x
    #     self.previous_x = x
    #     y_d = y - self.previous_y
    #     self.previous_y = y
    #     v = x * math.sqrt(abs(x)) / abs(x) * self.parameter_config["linear_k"] + x_d * self.parameter_config['linear_d']
    #     w = (y * self.parameter_config['angular_k'] + y_d * self.parameter_config['angular_d']) / self.parameter_config["l"]
    #     velocity.linear.x = np.clip(v, -0.21, 0.21)
    #     velocity.linear.y = 0.0
    #     velocity.linear.z = 0.0
    #     velocity.angular.x = 0.0
    #     velocity.angular.y = 0.0
    #     velocity.angular.z = np.clip(w, -2.84, 2.84)
    #     self.__velocity_publisher.publish(velocity)


    # turn and go
    def vector_callback(self,Point):
        vector_msg = Point
        velocity = Twist()

        x = vector_msg.x
        y = vector_msg.y
        z = vector_msg.z

        if x < 0:
            y -= x
            x = 0

        # self.x_i += x
        # self.y_i += y
        x_d = x - self.previous_x
        self.previous_x = x
        y_d = y - self.previous_y
        self.previous_y = y
        if x == 0:
            v = 0
        else:
            v = x * math.sqrt(abs(x)) / abs(x) * self.parameter_config["linear_k"] + x_d * self.parameter_config['linear_d']
        if y == 0:
            w = 0
        else:
            if z == 3 or z == 4:
                w = (y * self.parameter_config['angular_fw_k'] + y_d * self.parameter_config['angular_d']) / \
                    self.parameter_config["l"]
            else:
                w = (y * self.parameter_config['angular_gtg_k'] + y_d * self.parameter_config['angular_d']) / self.parameter_config["l"]
        velocity.linear.x = np.clip(v, -0.11, 0.11)
        velocity.linear.y = 0.0
        velocity.linear.z = 0.0
        velocity.angular.x = 0.0
        velocity.angular.y = 0.0
        velocity.angular.z = np.clip(w, -1.42, 1.42)
        self.__velocity_publisher.publish(velocity)


def main():
    rclpy.init()
    chase_object_node = Chase_Object()
    rclpy.spin(chase_object_node)
    chase_object_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
