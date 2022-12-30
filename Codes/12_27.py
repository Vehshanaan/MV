'''
Author: Runze Yuan 1959180242@qq.com
Date: 2022-12-27 09:43:29
LastEditors: Runze Yuan 1959180242@qq.com
LastEditTime: 2022-12-30 11:52:37
FilePath: \MV\Codes\12_27.py
Description: 

Copyright (c) 2022 by Runze Yuan 1959180242@qq.com, All Rights Reserved. 
'''
# red: 
red_path = r"A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\test\images\dataset1_front_241.png"
# A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\test\images\dataset1_front_241.png
# green: 
green_path_1 = r"A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\train\images\20150919_174151_image11.png"
green_path_2 = r"A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\train\images\20150919_174151_image856.png"
# A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\train\images\20150919_174151_image11.png

# 思路： https://blog.csdn.net/weixin_39444552/article/details/88783222
# 调节对小果子的灵敏度以及对小色块的容忍度：113行的length阈值和97行的开运算核大小

show_process = True # 是否展示运算过程

import cv2
import numpy as np

red_BGR = cv2.imread(red_path)
green_BGR = cv2.imread(green_path_2)
cv2.imshow("original:",green_BGR)
cv2.waitKey(0)
red_YCC = cv2.cvtColor(red_BGR,cv2.COLOR_BGR2YCR_CB) # 红色CR通道好
green_YCC = cv2.cvtColor(green_BGR,cv2.COLOR_BGR2YCR_CB) # 绿色CB通道好
"""
1. 根据色块寻找果子
产生： threshed： 用指定通道的自适应阈值划取的二值图像 有果子的地方白
"""

# 绿苹果的CB通道
green_YCC_channel2 = green_YCC[:,:,2]
# 高斯模糊
green_YCC_channel2 = cv2.blur(green_YCC_channel2,(3,3))

# 手动阈值二值化 找最黑的值，然后最黑值+10%*整体范围，作为阈值
min = np.amin(green_YCC_channel2)
max = np.amax(green_YCC_channel2)

range = max-min
thres = min+range*0.48

_,threshed = cv2.threshold(green_YCC_channel2,thres,255,cv2.THRESH_BINARY_INV)

# 开一下，去掉杂的绿叶部分
kernel = np.ones((8,8),dtype = np.uint8)
threshed = cv2.morphologyEx(threshed,cv2.MORPH_OPEN,kernel)



if show_process:
    cv2.imshow("Cb channel",green_YCC_channel2)
    cv2.imshow("1. thresed",threshed)
    print("max:"+str(max))
    print("min:"+str(min))
    print("thres:"+str(thres))
    cv2.waitKey(0)


"""
2. 根据果子色块创建蒙版, 取只有果子的彩色图部分

蒙版备注： 有果子部分是255，无果子部分是0，用bitwise and的时候要注意。

产生： masked: 只有果子的彩色图
"""
tmp = cv2.cvtColor(threshed,cv2.COLOR_GRAY2BGR) # 为了使单通道二值化蒙版与彩色三通道图像尺寸匹配
masked = cv2.bitwise_and(green_BGR,tmp) # 只有果子的彩色图部分

if show_process:
    cv2.imshow("2. masked BGR",masked)
    cv2.waitKey(0)

"""
3. 算果子部分的canny, 把黑色边缘叠加到上图中
产生： devided ：将边缘标为黑色的masked图
"""
masked_gray = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
masked_gray = cv2.medianBlur(masked_gray,5) # 中值滤波波，去除canny中的噪声
canny = cv2.Canny(masked_gray,100,200,L2gradient=True) # 黑底白边的边缘图

devided = cv2.bitwise_and(masked, cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR))
devided = masked-devided

if show_process:
    cv2.imshow("3. divided",devided)
    cv2.waitKey(0)

"""
4.腐蚀上图，得到不重叠的果子计数
产生： filled 用此图进入色块计数，直接获得答案。这个就相当于最终解了
"""
# 过侵蚀
eroded = cv2.erode(devided,np.ones((2,2),dtype = np.uint8))

# 开运算
opened = cv2.morphologyEx(eroded,cv2.MORPH_OPEN,np.ones((3,3),dtype = np.uint8),1) # 这一个大核的尺寸滤掉了大多数小不点色块

# 凸包填充
    # 灰度化
opened_gray = cv2.cvtColor(opened,cv2.COLOR_BGR2GRAY)
    # 识别凸包轮廓
contours, hierarchy = cv2.findContours(opened_gray,2,1)


filled = opened_gray#.copy()
filled.fill(0)

    # 填充凸包轮廓
for cnt in contours:
    hull = cv2.convexHull(cnt)
    length = len(hull)
    # 如果凸包点集中的像素大于8:
    if(length>8): # 调大这个可以过滤图中的小色块
        # 填充凸包
        cv2.fillConvexPoly(filled,cnt,(255,255,255))
        # 重绘凸包边界，防止凸包之间粘连
        i = 0
        while i < length:
            cv2.line(filled, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,0,0), 1)
            i+=1

filled_prev = filled

# 闭运算 消除光圈的黑圈
filled = cv2.morphologyEx(filled,cv2.MORPH_CLOSE,np.ones((2,2),dtype = np.uint8),2)
# 侵蚀 消除多边形填充产生的小枝条
filled = cv2.erode(filled,np.ones((2,2),dtype = np.uint8))

if show_process:
    cv2.imshow("4-1 eroded",eroded)
    cv2.waitKey(0)
    cv2.imshow("4-2 opened",opened)
    cv2.waitKey(0)
    cv2.imshow("4-3 hull filled",filled_prev)
    cv2.waitKey(0)
    cv2.imshow("step 4 result (4-4) morphology processed",filled)
    cv2.waitKey(0)

"""
5.数果子
"""
counting = filled

# 连通性检查设置
connectivity = 8 # ? https://stackoverflow.com/questions/7088678/4-connected-vs-8-connected-in-connected-component-labeling-what-is-are-the-meri

# 数连通域
counting_result = cv2.connectedComponentsWithStats(counting,connectivity,cv2.CV_8U)

# label数量
num_labels = counting_result[0]
# label 矩阵
labels = counting_result[1]
# stat 矩阵
stats = counting_result[2]
# 质心矩阵
centroids = counting_result[3]

for center in centroids:
    center = (int(center[0]), int(center[1]))
    result = cv2.circle(green_BGR,center,1,(0,0,255),8)
cv2.putText(result,str(num_labels),(50,100),cv2.FONT_HERSHEY_DUPLEX,2,(255,255,0),2)
result = result[:,:,:]
cv2.imshow("result",result)
cv2.waitKey(0)
