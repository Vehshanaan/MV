import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
description: 在给出的RGB色彩通道中，选取每个像素位置颜色最深的像素，进行腐蚀
param {*} r 图像的r通道
param {*} g 图像的g通道
param {*} b 图像的b通道
param {*} k_size 腐蚀用的正方形卷积核（形态元）边长
return {*} 腐蚀后的结果
'''
def getDarkMap(r, g, b, k_size):
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(k_size,k_size)) # 产生正方形形态学计算核（结构元）
    dark_channel = cv2.min(cv2.min(b,g),r) # 计算三通道中最小的值（最dark的）
    dark_map = cv2.erode(dark_channel,k) # 腐蚀（范围最小值）
    return dark_map


def morph_img(img_path):
    img = cv2.imread(img_path)
    img0 = img.copy()
    b, g, r = cv2.split(img) # 拆分BGR色彩通道
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, inten = cv2.split(img_hsv) # 拆分HSV色彩通道
    dark_map = getDarkMap(r, g, b, 7) # 在给出的RGB色彩通道中，选取每个像素位置颜色最深的像素，进行腐蚀


    plt.subplot(2,4,1)
    plt.title("original")
    plt.imshow(img0[:,:,::-1])

    plt.subplot(2,4,2)
    plt.title("r")
    plt.imshow(r, cmap="gray")
    plt.subplot(2,4,3)
    plt.title("g")
    plt.imshow(g, cmap="gray")
    plt.subplot(2,4,4)
    plt.title("b")
    plt.imshow(b, cmap="gray")
    plt.subplot(2,4,5)
    plt.title("hue")
    plt.imshow(hue, cmap="gray")
    plt.subplot(2,4,6)
    plt.title("sat")
    plt.imshow(sat, cmap="gray")
    plt.subplot(2,4,7)
    plt.title("intern")
    plt.imshow(inten, cmap="gray")
    plt.show()
    plt.subplot(2,4,8)
    plt.title("dark_channel")
    plt.imshow(dark_map, cmap="gray")
    plt.show()



    #img = inten
    #edges = cv2.Canny(img, 150, 220)
    #plt.title("canny_edge")
    #plt.imshow(edges, cmap="gray")
    #plt.show()

    
    #转grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 原图的灰度图
    plt.imshow(img, cmap="gray")
    plt.show()
    #img = np.where(hue>50, 0, img)
    #plt.imshow(img, cmap="gray")
    #plt.show()
    #equlized histogram
    img = cv2.equalizeHist(img) # 直方图均衡化
    #adaptive thresholding
    # 自动阈值二值化
    img = cv2.adaptiveThreshold(img, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=301, C=3)
    #人为设定阈值
    #thres, img= cv2.threshold(img, thresh=215, maxval=255, type=cv2.THRESH_BINARY_INV)
    img = np.invert(img) # 黑白反转
    # 在设定的H范围外的东西全部涂黑
    img = np.where(hue<140, 0, img)
    img = np.where(hue>180, 0, img)
    plt.title("H140-180")
    plt.imshow(img, cmap="gray")
    plt.show()
    
    #得到二值化图像后进行形态学操作 使用不同的运算，不同大小的核，不同的迭代次数

    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4,4))
    #img = cv2.erode(img, kernel, iterations=8)
    #plt.imshow(img, cmap="gray")
    #plt.show()

    # 开运算 （腐蚀后膨胀）
    kernel = np.ones((3,3), np.uint8) # 卷积核
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1) # 开运算
    plt.title("Open with thresholded pic")
    plt.imshow(img, cmap="gray")
    plt.show()

    #使用Cross型kernel来腐蚀
    #

    #plt.imshow(img, cmap="gray")
    #plt.title("2")
    #plt.show()

    #使用11*11大小的核来闭运算 迭代5次
    ##img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=6)

    #plt.imshow(img, cmap="gray")
    #plt.title("3")
    #plt.show()

    
    # remove boarder pixels
    #img[:50, :] = 0
    #img[:, :50] = 0
    #img[-50:, :] = 0
    #img[:, -50:] = 0

    #img = np.invert(img)
    return img, img0

def findContours(binary_img):
    num_labels,labels,stats,centroids  = cv2.connectedComponentsWithStats(binary_img,connectivity=8)

def multi_img_seg():
    pass

if __name__ == "__main__":
    img_path = r"A:\OneDrive\MScRobotics\MV\MV\MinneApple\detection\test\images\dataset1_back_1.png"#"20150921_131346_image36.png"
    b_img, img0 = morph_img(img_path)
    #plt.imshow(b_img, cmap="gray")
    #plt.show()
    # 数连通域，并写上
    num_labels,labels,stats,centroids  = cv2.connectedComponentsWithStats(b_img,connectivity=8)
    print(num_labels)
    print(centroids)
    for center in centroids:
        center = (int(center[0]), int(center[1]))
        img_show = cv2.circle(img0, center, 1, (0,0,255), 4)
    cv2.putText(img_show, str(num_labels), (50,100), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,0), 2)
    img_show = img_show[:,:,::-1]
    plt.imshow(img_show)
    plt.show()

