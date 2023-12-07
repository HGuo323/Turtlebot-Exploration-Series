import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, PoseStamped, PointStamped, PoseWithCovarianceStamped
from std_msgs.msg import Int8, Bool

import numpy as np


class setting_goal(Node):
    def __init__(self):
        super().__init__('setting_goal')
        # navigation parameters
        self.map_name = 'map'
        self.start = False
        self.initialpose = None
        # global map parameters
        self.map = [
            # column 0, x = 0
            [
                {'coordinate': (0, 0), 'accessibility': True, 'judging_orientation': {2, 3}},
                {'coordinate': (0, 1), 'accessibility': True, 'judging_orientation': {1, 2}},
                {'coordinate': (0, 2), 'accessibility': False},
                ],
            # column 1, x = 1
            [
                {'coordinate': (1, 0), 'accessibility': True, 'judging_orientation': {0, 1, 3}},
                {'coordinate': (1, 1), 'accessibility': True, 'judging_orientation': {3}},
                {'coordinate': (1, 2), 'accessibility': True, 'judging_orientation': {1, 2}},
                ],
            # column 2, x = 2
            [
                {'coordinate': (2, 0), 'accessibility': True, 'judging_orientation': {2, 3}},
                {'coordinate': (2, 1), 'accessibility': True, 'judging_orientation': {0}},
                {'coordinate': (2, 2), 'accessibility': True, 'judging_orientation': {0, 1}},
                ],
            # column 3, x = 3
            [
                {'coordinate': (3, 0), 'accessibility': True, 'judging_orientation': {0, 3}},
                {'coordinate': (3, 1), 'accessibility': True, 'judging_orientation': {2}},
                {'coordinate': (3, 2), 'accessibility': True, 'judging_orientation': {1, 2}},
                ],
            # column 4, x = 4
            [
                {'coordinate': (4, 0), 'accessibility': True, 'judging_orientation': {2, 3}},
                {'coordinate': (4, 1), 'accessibility': True, 'judging_orientation': {1}},
                {'coordinate': (4, 2), 'accessibility': True, 'judging_orientation': {1, 3}},
                ],
            # column 5, x = 5
            [
                {'coordinate': (5, 0), 'accessibility': True, 'judging_orientation': {0, 3}},
                {'coordinate': (5, 1), 'accessibility': True, 'judging_orientation': {0}},
                {'coordinate': (5, 2), 'accessibility': True, 'judging_orientation': {0, 1}},
                ],
        ]
        # physical world parameters
        self.edge_length = 0.91
        self.current_goal = None
        self.current_state = None
        # real-time parameters
        self.sign = None
        # publishers and subscribers
        self.test_subscriber = self.create_subscription(Point, 'manual_set_goal', self.test_callback, 10)
        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.sign_subscriber = self.create_subscription(Int8, 'prediction_result', self.sign_callback, 10)
        self.feedback_subscriber = self.create_subscription(Bool, 'feedback_nav', self.feedback_callback, 10)
        self.need_sign_publisher = self.create_publisher(Bool, 'need_prediction', 10)

        self.initialpose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initialpose_callback, 10)
        # self.click_subscription = self.create_subscription(PointStamped, 'clicked_point', self.click_callback, 10)
        self.get_logger().info('setting goal initiated')

    # converting quaternion to yaw, input is Quaternion message, return radian
    def quaternion2yaw(self, q):
        return  np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

    # converting yaw to quaternion, input is radian, return a Pose message
    def yaw2quaternion(self, yaw):
        result = Quaternion()
        result.x = 0.0
        result.y = 0.0
        result.z = np.sin(yaw / 2)
        result.w = np.cos(yaw / 2)
        return result




    # translating coordinates and orientation to PoseStamped message
    def set_goal(self, x, y, orientation):
        msg = PoseStamped()
        msg.header.frame_id = self.map_name
        # translating x and y to physical coordinates
        point = Point()
        point.x = x * self.edge_length + 0.5 * self.edge_length + 0.3
        point.y = y * self.edge_length + 0.5 * self.edge_length + 0.4
        point.z = 0.0
        # translating orientation to quaternion
        yaw = orientation * np.pi / 2
        quaternion = self.yaw2quaternion(yaw)

        msg.pose.position = point
        msg.pose.orientation = quaternion

        return msg

    # starting from pose A, get the farthest goal pose
    def goal_iterate(self, x, y, orientation):
        goal_x = x
        goal_y = y
        while True:
            # farthest goal so far
            current_block = self.map[goal_x][goal_y]
            # current pose need identification
            if orientation in current_block['judging_orientation']:
                break
            else:
                # keep going left
                if orientation == 0:
                    goal_x += 1
                # keep going down
                elif orientation == 1:
                    goal_y += 1
                # keep going right
                elif orientation == 2:
                    goal_x -= 1
                # keep going up
                else:
                    goal_y -= 1
        return int(goal_x), int(goal_y), int(orientation)


    def test_callback(self, msg):
        if self.start:
            x = int(msg.x)
            y = int(msg.y)
            orientation = int(msg.z)
            goal_msg = self.set_goal(x,y,orientation)
            self.current_goal = (x, y, orientation)
            print(goal_msg)
            self.goal_publisher.publish(goal_msg)
        else:
            if self.initialpose is not True:
                x, y, orientation = self.initialpose
                self.start = True
            else:
                self.get_logger().info('No initial pose set.')
                return
        
        goal_msg = self.set_goal(x,y,orientation)
        self.current_goal = (x, y, orientation)
        self.goal_publisher.publish(goal_msg)


    def feedback_callback(self, msg):
        # goal is reached
        if msg.data:
            self.current_state = self.current_goal
            x, y, orientation = self.current_state
            # read the corresponding block from map
            map_block = self.map[x][y]

            # current orientation needs sign identification
            if orientation in map_block['judging_orientation']:
                self.get_logger().info('Need identification.')
                msg = Bool()
                msg.data = True

                # tell the prediction node that we need prediction
                self.need_sign_publisher.publish(msg)


            # no need to identify signs, keep going
            else:
                new_goal_x, new_goal_y, new_goal_orientation = self.goal_iterate(x, y, orientation)
                msg = self.set_goal(new_goal_x, new_goal_y, new_goal_orientation)
                self.current_goal = (new_goal_x, new_goal_y, new_goal_orientation)
                self.get_logger().info('Keep going.')
                self.goal_publisher.publish(msg)

    def sign_callback(self, msg):
        sign = msg.data
        x, y, orientation = self.current_state
        need_action = False

        # None class
        if sign == 0:
            self.get_logger().info('Nothing is found.')
        # turn left class
        elif sign == 1:
            new_goal_orientation = (orientation + 1) % 4
            self.get_logger().info('Turn left!')
            need_action = True
        # turn right class
        elif sign == 2:
            new_goal_orientation = (orientation + 3) % 4
            self.get_logger().info('Turn right!')
            need_action = True
        # turn back class
        elif sign == 3 or sign == 4:
            new_goal_orientation = (orientation + 2) % 4
            self.get_logger().info('Turn back!')
            need_action = True
        # goal class
        elif sign == 5:
            self.get_logger().info('Final goal reached!')

        # need turning
        if need_action:
            new_goal_x, new_goal_y, new_goal_orientation = self.goal_iterate(x, y, new_goal_orientation)
            msg = self.set_goal(new_goal_x, new_goal_y, new_goal_orientation)
            self.current_goal = (new_goal_x, new_goal_y, new_goal_orientation)
            self.goal_publisher.publish(msg)

    # update amcl
    # def amcl_callback(self, msg):
    #     self.amcl = msg

        # detect clicked point as initiating signal
    def initialpose_callback(self, msg):
        # has not started yet
        if not self.start:
            # physical position in global frame
            current_x = msg.pose.pose.position.x
            current_y = msg.pose.pose.position.y
            current_yaw = self.quaternion2yaw(msg.pose.pose.orientation) / np.pi * 180
            self.get_logger().info(str(current_yaw))

            # physical position in grid frame
            current_x_grid_frame = current_x - 0.3
            current_y_grid_frame = current_y - 0.4

            # physical position in block coordinates
            current_x_block = int(current_x_grid_frame / self.edge_length)
            current_y_block = int(current_y_grid_frame / self.edge_length)

            if 45 < current_yaw <= 135:
                current_orientation = 1
            elif -135 <= current_yaw < -45:
                current_orientation = 3
            elif np.abs(current_yaw) <= 45:
                current_orientation = 0
            else:
                current_orientation = 2
            
            self.get_logger().info(str(current_yaw))
            
            self.get_logger().info('initial pose set: (' + str(current_x_block) + ', ' + str(current_y_block) + ', ' + str(current_orientation) + ')')

            # msg = self.set_goal(current_x_block, current_y_block, current_yaw)
            # self.current_goal = (current_x_block, current_y_block, current_yaw)
            # self.goal_publisher.publish(msg)

            self.initialpose = (current_x_block, current_y_block, current_orientation)












def main():
    rclpy.init()
    node = setting_goal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

