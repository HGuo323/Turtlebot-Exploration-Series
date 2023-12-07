import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge


def shape_adjustment(image):
    width, length = image.shape
    new_image = cv2.resize(image, (int(length / 2), int(width / 2)))
    return new_image


def blurring(image):
    new_image = cv2.GaussianBlur(image, ksize=(21, 21), sigmaX=0)
    new_image = cv2.medianBlur(new_image, ksize=7)
    return new_image


def contrasting(image):
    max_pixel = np.max(image)
    min_pixel = np.min(image)
    delta = max_pixel - min_pixel
    new_image = (image - min_pixel) * (255 / delta)
    return np.uint8(new_image)


def object_locate(image):
    image = contrasting(image)
    image = np.uint8(np.clip(image[:, :, 1] * 0.5 + image[:, :, 2] * 0.5 - image[:, :, 0] * 0.85, 0, 255))

    image = shape_adjustment(image)
    image = blurring(image)
    image = cv2.threshold(image, 50, 100, cv2.THRESH_BINARY)[-1]

    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image = cv2.merge([image, image, image])
    num = len(contours)
    circle_list = []
    top_left_coordinate = None
    bottom_right_coordinate = None
    for i in range(num):
        S = cv2.contourArea(contours[i])
        P = cv2.arcLength(contours[i], True)
        if S > 0:
            ratio = (P ** 2) / S

            if ratio > 13 and ratio < 14.5:
                circle_list.append(i)
                min_x = np.min(contours[i][:, :, 0])
                max_x = np.max(contours[i][:, :, 0])
                min_y = np.min(contours[i][:, :, 1])
                max_y = np.max(contours[i][:, :, 1])
                cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color=(255, 0, 0), thickness=2)
                top_left_coordinate = (min_x, min_y)
                bottom_right_coordinate = (max_x, max_y)
                print('top-left:', top_left_coordinate, '\tbottom-right:', bottom_right_coordinate)

    for i in range(len(circle_list)):
        cv2.drawContours(image, contours[circle_list[i]], -1, (0, 255, 0), 3)

    return top_left_coordinate, bottom_right_coordinate, image


class Find_Object(Node):
    def __init__(self):
        super().__init__('find_object')
        self.declare_parameter('show_image_bool', True)
        self.declare_parameter('window_name', "Raw Image")
        self._display_image = bool(self.get_parameter('show_image_bool').value)
        self._titleOriginal = self.get_parameter('window_name').value  # Image Window Title
        image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self._video_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            image_qos_profile)
        self._coordinate_publisher = self.create_publisher(Float64MultiArray, 'object_coordinates', 10)
        self._image_view_publisher = self.create_publisher(CompressedImage, 'image_show', 10)

    def _image_callback(self, CompressedImage):
        self._imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
        top, bottom, img = object_locate(self._imgBGR)
        processed_image = CvBridge().cv2_to_compressed_imgmsg(img)
        self._image_view_publisher.publish(processed_image)
        width, length, _ = img.shape
        coordinates_msg = Float64MultiArray()
        coordinates_data = [float(i) for i in [top[0], top[1], bottom[0], bottom[1], length, width]] if top is not None else [-1.0]
        coordinates_msg.data = coordinates_data
        self._coordinate_publisher.publish(coordinates_msg)


def main():
    rclpy.init()
    video_subscriber = Find_Object()

    while rclpy.ok():
        rclpy.spin_once(video_subscriber)
    video_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
