import rclpy
from rclpy.node import Node
from std_msgs.msg import String



class Change_Parameter(Node):
    def __init__(self):
        super().__init__('change_parameter')
        self.__parameter_publisher = self.create_publisher(String, 'parameter_change',10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = input('Please enter the parameter:')
        self.__parameter_publisher.publish(msg)

def main():
    rclpy.init()
    change_parameter_node = Change_Parameter()
    rclpy.spin(change_parameter_node)
    change_parameter_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
