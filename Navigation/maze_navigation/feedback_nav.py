import rclpy
from rclpy.node import Node

from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
from std_msgs.msg import Bool

import time

class feedback_nav(Node):
    def __init__(self):
        super().__init__('feedback_nav')
        # whether the robot is heading to goal
        self.heading_to_goal = False
        # last time receiving feedback
        self.last_time_feedback = 0
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.feedback_subscriber = self.create_subscription(NavigateToPose_FeedbackMessage, '/navigate_to_pose/_action/feedback', self.feedback_callback, 10)
        self.reach_publisher = self.create_publisher(Bool, 'feedback_nav', 10)
        self.get_logger().info('feedback initiated')
    def feedback_callback(self, msg):
        if not self.heading_to_goal:
            self.heading_to_goal = True
            self.get_logger().info('Heading to goal.')
        self.last_time_feedback = time.time()
    
    def timer_callback(self):
        if self.heading_to_goal:
            msg = Bool()
            current_time = time.time()

            # there has not been a feedback signal for two seconds, indicating the goal has been reached
            if current_time - self.last_time_feedback > 2:
                msg.data = True
                self.heading_to_goal = False
                time.sleep(1.5)
                self.get_logger().info('Goal is reached.')

            # still heading to goal
            else:
                msg.data = False

            # sleep 1.5s in avoidance of signal interruption

            self.reach_publisher.publish(msg)


def main():
    rclpy.init()
    node = feedback_nav()
    while True:
        rclpy.spin_once(node)