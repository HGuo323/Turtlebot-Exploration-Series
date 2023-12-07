import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Twist
import numpy as np


class Chase_Object(Node):
    def __init__(self):
        super().__init__('chase_object')
        self.__position_subscriber = self.create_subscription(Float64MultiArray, 'position', self.position_callback, 10)
        self.__parameter_subscriber = self.create_subscription(String, 'parameter_change', self.parameter_callback,10)
        self.__velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.parameter_config = {
            "linear_k":0.1,
            "angular_k":0.05,
            "distance":0.5,
            "l":0.05
        }
        self.linear_k = 0.1
        self.angular_k = 0.05
        self.distance = 0.5
        self.l = 0.05


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


    def position_callback(self,Float64MultiArray):
        position_msg = Float64MultiArray.data
        velocity = Twist()
        # no target found
        if len(position_msg) == 1:
            velocity.linear.x = 0.0
            velocity.linear.y = 0.0
            velocity.linear.z = 0.0
            velocity.angular.x = 0.0
            velocity.angular.y = 0.0
            velocity.angular.z = 0.0
        else:
            distance, angle = position_msg
            # target is close to boundary, just rotate to follow
            if np.abs(distance - self.parameter_config["distance"]) < 0.1:
                velocity.linear.x = 0.0
                velocity.linear.y = 0.0
                velocity.linear.z = 0.0
                velocity.angular.x = 0.0
                velocity.angular.y = 0.0
                velocity.angular.z = np.clip(self.parameter_config["angular_k"] * angle, -2.84, 2.84)
            else:
                v = (distance - self.parameter_config["distance"]) * self.parameter_config["linear_k"]
                w = v * np.tan(np.radians(angle)) / self.parameter_config["l"]
                velocity.linear.x = np.clip(v, -0.22, 0.22)
                velocity.linear.y = 0.0
                velocity.linear.z = 0.0
                velocity.angular.x = 0.0
                velocity.angular.y = 0.0
                if v >= 0:
                    velocity.angular.z = np.clip(w, -2.84, 2.84)
                else:
                    velocity.angular.z = np.clip(-w, -2.84, 2.84)
        self.__velocity_publisher.publish(velocity)


def main():
    rclpy.init()
    chase_object_node = Chase_Object()
    rclpy.spin(chase_object_node)
    chase_object_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
