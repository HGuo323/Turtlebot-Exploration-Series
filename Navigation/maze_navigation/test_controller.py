import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class test_controller(Node):
    def __init__(self):
        super().__init__('test_controller')
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.test_publisher = self.create_publisher(Point, 'manual_set_goal', 10)

    def timer_callback(self):
        line = input('Please enter x y orientation:')
        x, y, orientation = [float(i) for i in line.split(' ')]
        msg = Point()
        msg.x = x
        msg.y = y
        msg.z = orientation
        self.test_publisher.publish(msg)


def main():
    rclpy.init()
    node = test_controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
