'''
Author: Runze Yuan 1959180242@qq.com
Date: 2022-12-26 17:23:51
LastEditors: Runze Yuan 1959180242@qq.com
LastEditTime: 2022-12-26 20:20:25
FilePath: \MV\Codes\environment_test.py
Description: 

Copyright (c) 2022 by Runze Yuan 1959180242@qq.com, All Rights Reserved. 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# MinneApple path get:
prev_path = os.getcwd()
MinneApplePath = prev_path + "\MinneApple"
# MinneApple 文件夹里有三个文件夹： counting，detection，test_data
counting_path = MinneApplePath + "\counting"
detection_path = MinneApplePath + "\detection"
test_data_path = MinneApplePath + (r"\test_data")

#img = cv2.imread(test_data_path+(r"\counting\images\testset1_5cluster_12.png"))
img = cv2.imread(detection_path+(r"\test\images\dataset1_back_1.png"))
cv2.imshow("test",img)
cv2.waitKey(0)