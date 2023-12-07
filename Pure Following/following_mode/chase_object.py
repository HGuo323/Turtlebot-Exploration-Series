import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist


class Chase_Object(Node):
    def __init__(self):
        super().__init__('chase_object')
        self.__position_subscriber = self.create_subscription(Float64MultiArray, 'position', self.position_callback, 10)
        self.__velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.linear_k = 0.1
        self.angular_k = 0.01
        self.distance = 0.5

    def position_callback(self,Float64MultiArray):
        position_msg = Float64MultiArray.data
        velocity = Twist()
        if len(position_msg) == 1:
            velocity.linear.x = 0.0
            velocity.linear.y = 0.0
            velocity.linear.z = 0.0
            velocity.angular.x = 0.0
            velocity.angular.y = 0.0
            velocity.angular.z = 0.0
        else:
            distance, angle = position_msg
            velocity.linear.x = min((distance - self.distance) * self.linear_k, 0.22)
            velocity.linear.y = 0.0
            velocity.linear.z = 0.0
            velocity.angular.x = 0.0
            velocity.angular.y = 0.0
            velocity.angular.z = min(angle * self.angular_k, 2.84)
        self.__velocity_publisher.publish(velocity)


def main():
    rclpy.init()
    chase_object_node = Chase_Object()
    rclpy.spin(chase_object_node)
    chase_object_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
