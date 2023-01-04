'''
Author: Runze Yuan 1959180242@qq.com
Date: 2022-12-28 17:38:15
LastEditors: Runze Yuan 1959180242@qq.com
LastEditTime: 2023-01-04 18:53:06
FilePath: \MV\Codes\ReportExampleGenerate.py
Description: 旨在将12.27.py中的处理算法做成一个函数

Copyright (c) 2022 by Runze Yuan 1959180242@qq.com, All Rights Reserved. 
'''

import cv2
import numpy as np


'''
description: 标注出图中的红苹果
param {*} src 待检测原始图像（BGR）
return {*} counting_result cv2.connectedComponentsWithStat生成的数据，详见函数说明，或者见函数最下面的那个“效果展示”
'''
def RedMarking(src):
    # 红色用hsv的h，绿色用YCrCb的Cb

    hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    red_h = hsv[:,:,0]

    """
    1. 根据色块寻找果子
    产生： threshed： 用指定通道的自适应阈值划取的二值图像 有果子的地方白
    用RGB和HSV两种色域筛选红苹果
    HSV阈值搞宽一点，然后用RGB剔除产生的非红色部分
    """

    # (1).先来HSV
    # 提取图中的最大最小值
    hsv_max = np.amax(red_h)
    hsv_min = np.amin(red_h)
    hsv_range = hsv_max-hsv_min

    # 计算自适应阈值
    thres_max_hsv = hsv_max-hsv_range*0.1
    thres_min_hsv = hsv_min+hsv_range*0.1

    # 用两个自适应阈值二值化    
    _,threshed_max_hsv = cv2.threshold(red_h,thres_max_hsv,255,cv2.THRESH_BINARY)
    _,threshed_min_hsv = cv2.threshold(red_h,thres_min_hsv,255,cv2.THRESH_BINARY_INV)

    # 结合两个自适应阈值的结果
    threshed_red_hsv = cv2.bitwise_or(threshed_max_hsv,threshed_min_hsv)


    # (2).再来RGB :计算红色通道在三通道值中的占比，再判断剩下的像素中红的值够不够高
    # （2）-1 判断红色占比够不够高
    b,g,r= cv2.split(src)
    
    # 把三通道扩张一下位数，不然求和的时候会翻过去
    b = np.array(b,dtype = np.float32)
    g = np.array(g,dtype = np.float32)
    r = np.array(r,dtype = np.float32)

    sum = b+g+r
    r = 256*r/sum
    r = np.array(r,dtype = np.uint8) # 得到红色在三通道中的占比，具体像素值为0-255*占比百分比

    # 计算自适应阈值
    r_max = np.amax(r)
    r_min = np.amin(r)
    r_range = r_max-r_min
    thres_r = r_max-r_range*0.5

    # 二值化
    _,threshed_r = cv2.threshold(r,thres_r,255,cv2.THRESH_BINARY)


    # （2）-2 判断红色通道的值够不够高
    b,g,r = cv2.split(src)
    _,threshed_abs_r = cv2.threshold(r,70,255,cv2.THRESH_BINARY)


    threshed = cv2.bitwise_and(threshed_red_hsv,threshed_r)
    threshed = cv2.bitwise_and(threshed,threshed_abs_r)


    threshed = cv2.medianBlur(threshed,5)
    #threshed = cv2.morphologyEx(threshed,cv2.MORPH_ERODE,np.ones((2,2),dtype = np.uint8),1)

    if 0:
        cv2.imshow("org",src)
        cv2.imshow("1. threshed",threshed)
        cv2.waitKey(0)

    """
    2. 根据果子色块创建蒙版, 取只有果子的彩色图部分

    蒙版备注： 有果子部分是255，无果子部分是0，用bitwise and的时候要注意。

    产生： masked: 只有果子的彩色图
    """
    _ = cv2.cvtColor(threshed,cv2.COLOR_GRAY2BGR)
    masked = cv2.bitwise_and(src,_)


    if 0:
        cv2.imshow("2. maksed",masked)
        cv2.waitKey(0)

    """
    3. 算果子部分的canny, 把黑色边缘叠加到上图中
    产生： devided ：将边缘标为黑色的masked图
    """  

    # 提取边缘
    masked_r = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
    masked_r = cv2.medianBlur(masked_r,9)# 中值滤波 消除canny中的小噪声
    canny = cv2.Canny(masked_r,100,200,L2gradient=True)

    devided = cv2.bitwise_and(masked,cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR))
    devided = masked-devided
    
    

    if 0:
        cv2.imshow("3. devided",devided)
        cv2.waitKey(0)

    """
    4.腐蚀上图，得到不重叠的果子计数
    产生： filled 用此图进入色块计数，直接获得答案。这个就相当于最终解了
    """
    # 过侵蚀
    eroded = cv2.erode(devided,np.ones((2,2),dtype = np.uint8))

    # 开运算
    opened = cv2.morphologyEx(eroded,cv2.MORPH_OPEN,np.ones((3,3),dtype = np.uint8),1)

    # 凸包填充
        # 灰度化
    opened_gray = cv2.cvtColor(opened,cv2.COLOR_BGR2GRAY)
        # 识别凸包轮廓
    contours, hierarchy = cv2.findContours(opened_gray,2,1)

    filled = opened_gray.copy()
    filled.fill(0)

        # 填充凸包轮廓
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        length = len(hull)
        # 如果凸包点集中的像素大于8:
        if(length>0): # 调大这个可以过滤图中的小色块
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

    if 0:
        cv2.imshow("4-1.eroded",eroded)
        cv2.imshow("4-2.opened",opened)
        cv2.imshow("4-3.hul filled",filled_prev)
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

    """
    # 效果展示
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
        result = cv2.circle(src,center,1,(0,255,0),8)
    cv2.putText(result,str(num_labels),(50,100),cv2.FONT_HERSHEY_DUPLEX,2,(255,255,0),2)
    result = result[:,:,:]
    cv2.imshow("result",result)
    cv2.waitKey(0)
    """

    return counting_result # 返回全部数据，在ClassicalMarking里再处理


#RedMarking(red2)

'''
description: 
param {*} src 待检测原始图像（BGR）
return {*} counting_result cv2.connectedComponentsWithStat生成的数据，详见函数说明，或者见函数最下面的那个“效果展示”
'''
def GreenMarking(src):
    """
    1. 根据色块寻找果子
    产生： threshed： 用指定通道的自适应阈值划取的二值图像 有果子的地方白
    """
    # 取正确通道
    green_YCC = cv2.cvtColor(src,cv2.COLOR_BGR2YCR_CB)
    green_YCC_channel2 = green_YCC[:,:,2]
    # 高斯模糊
    green_YCC_channel2 = cv2.blur(green_YCC_channel2,(3,3))
    # 二值化 找最黑的值，然后最黑值+10%*整体范围，作为阈值
    min = np.amin(green_YCC_channel2)
    max = np.amax(green_YCC_channel2)

    range = max-min
    thres = min+range*0.48

    _,threshed = cv2.threshold(green_YCC_channel2,thres,255,cv2.THRESH_BINARY_INV)

    # 开运算去除杂的绿叶部分
    threshed = cv2.morphologyEx(threshed,cv2.MORPH_OPEN,np.ones((8,8),dtype=np.uint8))

    """
    2 . 根据果子色块创建蒙版, 取只有果子的彩色图部分

    蒙版备注： 有果子部分是255，无果子部分是0，用bitwise and的时候要注意。

    产生： masked: 只有果子的彩色图
    """
    _ = cv2.cvtColor(threshed,cv2.COLOR_GRAY2BGR) # 为了使单通道二值化蒙版与彩色三通道图像尺寸匹配
    masked = cv2.bitwise_and(src,_) # 只有果子的彩色图部分

    """
    3. 算果子部分的canny, 把黑色边缘叠加到上图中
    产生： devided ：将边缘标为黑色的masked图
    """
    masked_gray = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.medianBlur(masked_gray,5) # 中值滤波波，去除canny中的噪声

    canny = cv2.Canny(masked_gray,100,200,L2gradient=True) # 黑底白边的边缘图

    devided = cv2.bitwise_and(masked, cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR))
    devided = masked-devided


    """
    4.腐蚀上图，得到不重叠的果子计数
    产生： filled 用此图进入色块计数，直接获得答案。这个就相当于最终解了
    """
    # 过侵蚀
    eroded = cv2.erode(devided,np.ones((2,2),dtype = np.uint8))

    # 开运算
    opened = cv2.morphologyEx(eroded,cv2.MORPH_OPEN,np.ones((3,3),dtype = np.uint8),1) # 这里把核调大可以滤掉很多小不点儿色块

    # 凸包填充
        # 灰度化
    opened_gray = cv2.cvtColor(opened,cv2.COLOR_BGR2GRAY)
        # 识别凸包轮廓
    contours, hierarchy = cv2.findContours(opened_gray,2,1)

    filled = opened_gray
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
    # 闭运算 消除光圈的黑圈
    filled = cv2.morphologyEx(filled,cv2.MORPH_CLOSE,np.ones((2,2),dtype = np.uint8),2)
    # 侵蚀 消除多边形填充产生的小枝条
    filled = cv2.erode(filled,np.ones((2,2),dtype = np.uint8))


    """
    5.数果子
    """
    counting = filled

    
    # 连通性检查设置
    connectivity = 8 # ? https://stackoverflow.com/questions/7088678/4-connected-vs-8-connected-in-connected-component-labeling-what-is-are-the-meri    

    # 数连通域
    counting_result = cv2.connectedComponentsWithStats(counting,connectivity,cv2.CV_8U) 

    """
    # 效果展示
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
        result = cv2.circle(src,center,1,(0,0,255),8)
    cv2.putText(result,str(num_labels),(50,100),cv2.FONT_HERSHEY_DUPLEX,2,(255,255,0),2)
    result = result[:,:,:]
    cv2.imshow("result",result)
    cv2.waitKey(0)
    """

    return counting_result


# GreenMarking(green1)

'''
description:  根据图像中含不含红绿果子，自动进行果子计数，最后合并，这个是最上级的方法，会调用前面两个函数
param {*} src 原图像 BGR格式
return {*} count: 果子数目
return {*} center: 包含所有果子位置的(果子个数, 2)规格的矩阵
'''
def ClassicalMarking(src):
    # 检测图片中红绿苹果的问题
    # 红色用hsv的h，绿色用YCrCb的Cb

    red_flag = False # 图中有红苹果的标志
    green_flag = False # 图中有绿苹果的标志

    # 1. 检查图中是否有红色：检查H通道的较大较小值
    hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    red_h = hsv[:,:,0]
    red_h = cv2.erode(red_h,np.ones((5,5),dtype=np.uint8))

    red_max = np.amax(red_h)
    red_min = np.amin(red_h)
    
    if red_max>170 and red_min <15:
        red_flag = True

    # 2. 检查图中是否有绿色： 检查Cb通道的值
    YCC = cv2.cvtColor(src,cv2.COLOR_BGR2YCR_CB)
    Cb = YCC[:,:,2]

    green_min = np.amin(Cb)

    if green_min < 10:
        green_flag = True


    print("green_flag:" + str(green_flag))
    print("red_flag: "+str(red_flag))

    red_result = green_result = 0 # 结果数组初始化

    count = center = 0 # 果子个数和中心坐标

    if red_flag:
        red_result = RedMarking(src)
        count = red_result[0]
        center = red_result[3]
    if green_flag:
        green_result = GreenMarking(src)
        count = green_result[0]
        center = green_result[3]
    if red_flag and green_flag: # 如果两种都有： 结合结果
        count = green_result[0]+red_result[0]
        center = np.concatenate((red_result[3],green_result[3]),axis=0)

    """
    #效果展示, 这部分可以去掉    
    """
    for cen in center:
        cen = (int(cen[0]),int(cen[1]))
        img = cv2.circle(src,cen,1,(0,0,255),8)
    #cv2.putText(img,str(count),(50,100),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
    cv2.putText(img,"Count:{}".format(count),(50,100),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),8)
    cv2.imshow("result",img)
    cv2.waitKey(0)

    return count,center


# 展示：
if __name__ == "__main__":
    # red: 
    red_path_1 = r"A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\test\images\dataset1_front_241.png"
    red_path_2 = r"A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\test\images\dataset1_front_901.png"
    red_path_3 = r"A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\test\images\dataset1_back_1.png"
    # green: 
    green_path_1 = r"A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\train\images\20150919_174151_image11.png"
    green_path_2 = r"A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\train\images\20150919_174151_image856.png"
    # A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\train\images\20150919_174151_image11.png

    red1 = cv2.imread(red_path_1)
    red2 = cv2.imread(red_path_2)
    red3 = cv2.imread(red_path_3)
    green1 = cv2.imread(green_path_1)
    green2 = cv2.imread(green_path_2)


    #ClassicalMarking(red1)
    #ClassicalMarking(red2)
    ClassicalMarking(red3)
    #ClassicalMarking(green1)
    #ClassicalMarking(green2)


        





