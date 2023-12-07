import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8, Bool
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import cv2
import numpy as np


def crop_img(img, area_threshold=0.002,
             top_crop_percent=22, bottom_crop_percent=25, padding_width=30):
    r_lower = (0, 127, 75)
    r_upper = (9, 205, 230)
    rr_lower = (165, 127, 75)
    rr_upper = (180, 205, 230)
    b_lower = (100, 55, 10)
    b_upper = (138, 250, 97)
    g_lower = (55, 51, 25)
    g_upper = (80, 229, 204)

    color_ranges = [
        [r_lower, r_upper],  # Example color range for the sign
        [rr_lower, rr_upper],
        [b_lower, b_upper],
        [g_lower, g_upper]  # Add more color ranges if needed
    ]
    # Read the image
    original_img = img.copy()
    # Calculate the crop percentages in pixels and crop the img
    top_crop = int(top_crop_percent / 100 * original_img.shape[0])
    bottom_crop = int(bottom_crop_percent / 100 * original_img.shape[0])
    cropped_img = original_img[top_crop:-bottom_crop, :]
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    # Initialize variables for the maximum contour and its area
    max_contour = None
    max_contour_area = 0
    max_mask = None
    mask_list = []

    # indicating whether an object has been found
    find_object = False

    # Iterate over the provided color ranges
    for i in range(len(color_ranges)):

        # Create a binary mask using the current color range
        color_range = color_ranges[i]
        lower_color = np.array(color_range[0])
        upper_color = np.array(color_range[1])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask_list.append(mask)
        if i == 1:
            mask = mask_list[i - 1] + mask
        # cv2.imshow(str(i), mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the maximum area among the provided color ranges
        if contours:
            contour_max = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour_max)
            # contour so large that it is not possible to be noise
            if area > 200:
                # if i > 1 and find_object:
                #     return None, mask
                # else:
                find_object = True

            if area > max_contour_area:
                max_contour = contour_max
                max_contour_area = area
                max_mask = mask

    # Check if a valid contour is found
    if max_contour is None:
        return None # , max_mask

    # Calculate the contour area to original image area ratio and check if > threshold
    img_area = img.shape[0] * img.shape[1]
    ratio = max_contour_area / img_area
    if ratio < area_threshold:
        return None # , max_mask

    # Create a bounding box around the sign and crop img
    x, y, w, h = cv2.boundingRect(max_contour)
    # Expand the bounding box by the specified padding width
    x = max(0, x - padding_width)
    y = max(0, y - padding_width)
    w = min(original_img.shape[1] - x, w + 2 * padding_width)
    h = min(original_img.shape[0] - y, h + 2 * padding_width)
    # Draw the expanded bounding box on the original image
    cropped_img = original_img[y + top_crop:y + h + top_crop, x:x + w]
    cropped_img = cv2.resize(cropped_img, (100, 100))

    return cropped_img # , max_mask


class nn_model(nn.Module):
    def __init__(self):
        super(nn_model,self).__init__()
        self.act = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(5,5), stride=(2,2))
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5,5),stride=(2,2))
        self.fc = nn.Linear(in_features=500, out_features=5)

    def forward(self, data):
        batch_size = data.size()[0]
        x = self.conv1(data)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = x.view(batch_size,-1)
        x = self.fc(x)
        return x



# model_name = 'cnn_23.pth'
# device = 'cpu'
# model = torch.load(model_name).to(device)

class sign_keep_predict(Node):
    def __init__(self):
        super().__init__('sign_keep_predict')
        image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.image_subscriber = self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, image_qos_profile)
        self.prediction_publisher = self.create_publisher(Int8, 'prediction_result', 10)
        self.need_subscriber = self.create_subscription(Bool, 'need_prediction', self.need_callback, 10)
        # model file name
        self.model_name = 'cnn_23_sta.pth'
        self.device = 'cpu'
        self.model = nn_model()
        self.model.load_state_dict(torch.load(self.model_name))
        self.cycle = 3
        self.current = 1
        # self.model = preload_model
        # real-time image
        self.image = None
        self.get_logger().info('predict initiated')

    def image_callback(self, msg):
        if self.current >= self.cycle:
            self.image = msg
            self.current = 1
        else:
            self.current += 1
    
    def need_callback(self, msg):
        # need prediction
        if msg.data:
            prediction_result = Int8()
            # current image
            image = CvBridge().compressed_imgmsg_to_cv2(self.image, 'bgr8')
            cropped_image = crop_img(image)

            if cropped_image is None:
                cropped_image = np.ones((100, 100, 3)) * 255
                label = 0
            else:
                numpy_image = np.array(cv2.split(cropped_image))
                torch_image = torch.from_numpy(numpy_image)
                input = torch.unsqueeze(torch_image, 0).to(torch.float32)

                # get model output
                output = self.model.forward(input)
                label = torch.argmax(output).item() + 1

            prediction_result.data = label
            self.prediction_publisher.publish(prediction_result)
            cv2.imshow('predict image', cropped_image)
            cv2.waitKey(1)

def main():
    rclpy.init()
    predict_node = sign_keep_predict()
    rclpy.spin(predict_node)
    predict_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

