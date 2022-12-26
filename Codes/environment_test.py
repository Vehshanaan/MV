import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# MinneApple path get:
prev_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
MinneApplePath = prev_path + "\MinneApple"
# MinneApple 文件夹里有三个文件夹： counting，detection，test_data
counting_path = MinneApplePath + "\counting"
detection_path = MinneApplePath + "\detection"
test_data_path = MinneApplePath + (r"\test_data")


#img = cv2.imread(test_data_path+(r"\counting\images\testset1_5cluster_12.jpg"))
img = cv2.imread(r"A:\OneDrive\MScRobotics\MV\MV\MinneApple\test_data\test_data\counting\images\testset1_5cluster_12.jpg")
cv2.imshow("test",img)
cv2.waitKey(0)