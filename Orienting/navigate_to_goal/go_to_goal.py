import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Point
from rclpy.qos import qos_profile_sensor_data
# from tf_transformations import euler_from_quaternion
import numpy as np
import time

class Go_to_Goal(Node):
    def __init__(self):
        super().__init__('go_to_goal')
        self.parameter_config = {
            'obstacle_boundary': 0.25,
            'obstacle_fat_guard': 0.08,
            'avoid_obstacle_k': 1.2,
            'target_boundary': 0.05,

            # 1 for GTG, 2 for AO, 3 for FWC, 4 for FWCC
            'motion_state': 1,

            'path_state': 0
        }
        self.__odom_subscriber = self.create_subscription(Float64MultiArray,'/new_odom',self.odom_callback,10)
        self.__range_subscriber = self.create_subscription(Float64MultiArray,'direction_range',self.range_callback,10)
        self.__vector_publisher = self.create_publisher(Point,'vector',10)
        self.__vectors_publisher = self.create_publisher(Float64MultiArray,'strategy_vectors',10)
        self.__parameter_subscriber = self.create_subscription(String, 'parameter_change', self.parameter_callback, 10)
        self.to_goal = np.array([1.5,0])
        self.target_points = np.array([[1.5,0],[1.5,1.4],[0,1.4]])
        self.avoid_obstacle = np.array([0,0])
        self.follow_wall_clockwise = np.array([0,0])
        self.follow_wall_counterclockwise = np.array([0,0])
        self.closest_obstacle_range = 100
        self.closest_range_to_target = 100
        self.rest = False
        self.rest_state = [1,1,1]

    def rotation_matrix(self, radian):
        return np.array([[np.cos(radian),-np.sin(radian)],[np.sin(radian),np.cos(radian)]])

    def state_change(self,state):
        self.parameter_config['motion_state'] = state
        self.get_logger().info('motion_state:' + str(state))
    def path_state_change(self,state):
        self.parameter_config['path_state'] = state
        self.get_logger().info('path_state:' + str(state))

    def state_machine(self):
        gtg_fwc_inner_product = self.to_goal @ self.follow_wall_clockwise.T
        gtg_fwcc_inner_product = self.to_goal @ self.follow_wall_counterclockwise.T
        range_to_target = np.sum(self.to_goal ** 2)

        if self.closest_obstacle_range > self.parameter_config['obstacle_boundary']:
            obstacle_criterion = 3
        elif self.parameter_config['obstacle_boundary'] >= self.closest_obstacle_range > self.parameter_config['obstacle_boundary'] - self.parameter_config['obstacle_fat_guard']:
            obstacle_criterion = 2
        else:
            obstacle_criterion = 1

        progress_criterion = range_to_target < self.closest_range_to_target
        if progress_criterion:
            self.closest_range_to_target = range_to_target

        if self.parameter_config['motion_state'] == 1:
            if obstacle_criterion == 2 and gtg_fwc_inner_product > 0 and progress_criterion:
                self.state_change(3)
            if obstacle_criterion == 2 and gtg_fwcc_inner_product > 0 and progress_criterion:
                self.state_change(4)
        elif self.parameter_config['motion_state'] == 3 or self.parameter_config['motion_state'] == 4:
            if obstacle_criterion == 1:
                self.state_change(2)
            gtg_ao_inner_product = self.to_goal @ self.avoid_obstacle.T
            if gtg_ao_inner_product > 0 or progress_criterion or obstacle_criterion == 3:
                self.state_change(1)
        elif self.parameter_config['motion_state'] == 2:
            if obstacle_criterion == 2 and gtg_fwc_inner_product > 0 and progress_criterion:
                self.state_change(3)
            if obstacle_criterion == 2 and gtg_fwcc_inner_product > 0 and progress_criterion:
                self.state_change(4)

        if self.parameter_config['path_state'] == 0:
            if range_to_target < self.parameter_config['target_boundary']:
                if self.rest_state[0] == 1:
                    self.rest = True
                    self.rest_state[0] = 0
                    self.path_state_change(1)
                    self.closest_range_to_target = 100
        elif self.parameter_config['path_state'] == 1:
            if range_to_target < self.parameter_config['target_boundary']:
                if self.rest_state[1] == 1:
                    self.rest = True
                    self.rest_state[1] = 0
                    self.path_state_change(2)
                    self.closest_range_to_target = 100
        elif self.parameter_config['path_state'] == 2:
            if range_to_target < self.parameter_config['target_boundary']:
                if self.rest_state[2] == 1:
                    self.rest_state[2] = 0
                    self.rest = True
                    self.path_state_change(0)
                    self.closest_range_to_target = 100




    def execution(self):
        vector_msg = Point()
        if self.parameter_config['motion_state'] == 1:
            vector_msg.x = float(self.to_goal[0])
            vector_msg.y = float(self.to_goal[1])

        elif self.parameter_config['motion_state'] == 2:
            vector_msg.x = float(self.avoid_obstacle[0])
            vector_msg.y = float(self.avoid_obstacle[1])

        elif self.parameter_config['motion_state'] == 3:
            vector_msg.x = float(self.follow_wall_clockwise[0])
            vector_msg.y = float(self.follow_wall_clockwise[1])

        elif self.parameter_config['motion_state'] == 4:
            vector_msg.x = float(self.follow_wall_counterclockwise[0])
            vector_msg.y = float(self.follow_wall_counterclockwise[1])

        vector_msg.z = float(self.parameter_config['motion_state'])
        self.__vector_publisher.publish(vector_msg)

    def parameter_callback(self, String):
        information = String.data
        name, value = information.split(' ')
        if name in self.parameter_config:
            if value[-1] == 'i':
                self.parameter_config[name] = int(value[:-1])
            else:
                self.parameter_config[name] = float(value)
            self.get_logger().info('Changed ' + name + ' to ' + value)
            self.get_logger().info("Current parameters:")
            for key in self.parameter_config:
                self.get_logger().info(key + ':' + str(self.parameter_config[key]))
        self.get_logger().info('\n')


    def odom_callback(self, Odometry):
        if not self.rest:
            odom_data = Odometry.data
            x,y,yaw = odom_data

            current_position_global = np.array([x, y])

            # global reference frame
            to_goal_global = self.target_points[self.parameter_config['path_state']] - current_position_global

            # local reference frame
            self.to_goal = (self.rotation_matrix(-yaw) @ to_goal_global.T).T


    def range_callback(self, Float64MultiArray_msg):
        if not self.rest:
            msg_list = Float64MultiArray_msg.data
            if len(msg_list) > 1:
                polar_coordinate_of_closest = np.array(msg_list)

            else:
                polar_coordinate_of_closest = np.array([100,np.pi])




            self.closest_obstacle_range = polar_coordinate_of_closest[0]
            self.avoid_obstacle = self.parameter_config['avoid_obstacle_k'] * np.array(
                [-np.exp(-polar_coordinate_of_closest[0]) * np.cos(polar_coordinate_of_closest[1]),
                 -np.exp(-polar_coordinate_of_closest[0]) * np.sin(polar_coordinate_of_closest[1])])
            self.follow_wall_clockwise = (self.rotation_matrix(-np.pi / 2) @ self.avoid_obstacle.T).T
            self.follow_wall_counterclockwise = (self.rotation_matrix(np.pi / 2) @ self.avoid_obstacle.T).T

            vectors_msg = Float64MultiArray()
            vectors_msg.data = [self.to_goal[0],self.to_goal[1],self.avoid_obstacle[0],self.avoid_obstacle[1],self.follow_wall_clockwise[0],self.follow_wall_clockwise[1],self.follow_wall_counterclockwise[0],self.follow_wall_counterclockwise[1]]
            self.__vectors_publisher.publish(vectors_msg)
            self.state_machine()

            self.execution()


def main():
    rclpy.init()
    go_to_goal_node = Go_to_Goal()
    wait_time = 10
    start = -1
    while True:
        rclpy.spin_once(go_to_goal_node)
        if go_to_goal_node.rest:
            if start < 0:
                start = time.time()
                vector_msg = Point()
                vector_msg.x = 0.0
                vector_msg.y = 0.0
                vector_msg.z = 0.0
                go_to_goal_node.__vector_publisher.publish(vector_msg)
                go_to_goal_node.get_logger().info('I\'m sleeping!')
            else:
                if time.time() - start > 10:
                    start = -1
                    go_to_goal_node.rest = False
                    go_to_goal_node.get_logger().info(
                        'I\'m awake! Heading for point ' + str(go_to_goal_node.parameter_config['path_state'] + 1))
    go_to_goal_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()









