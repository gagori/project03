import numpy as np
import math
import cv2


file_path = './img/id5.jpg'
original = cv2.imread(file_path)

    
#load in grayscale:
src = cv2.imread(file_path,0)
canny = cv2.Canny(src, 50, 150)
tested_angles = np.deg2rad

#Hough transform:
# minLineLength = width/2.0
# maxLineGap = 20
# lines = cv2.HoughLinesP(src,1,np.pi/180,100,minLineLength,maxLineGap)
# lines = cv2.HoughLinesP(canny,1, np.pi/180, 100,minLineLength,maxLineGap)
lines = cv2.HoughLines(canny, 1,np.pi/180, 100) # houghlines가 직선검출에 더 적합함.
print(lines)

# #calculate the angle between each line and the horizontal line:
# angle = 0.0
# nb_lines = len(lines)


# for line in lines:
#     angle += math.atan2(line[0][3]*1.0 - line[0][1]*1.0,line[0][2]*1.0 - line[0][0]*1.0);

# angle /= nb_lines*1.0

# answer =angle* 180.0 / np.pi
# print(answer)











# def deskew(file_name,angle):
#     #load in grayscale:
#     img = cv2.imread(file_name)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray_img, (1, 1), 0)
#     _, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     canny = cv2.Canny(th1, 50, 150)

#     #invert the colors of our image:
#     # cv2.bitwise_not(img, img)
#     cv2.bitwise_not(canny, canny)
    
#     #compute the minimum bounding box:
#     # non_zero_pixels = cv2.findNonZero(img)
#     non_zero_pixels = cv2.findNonZero(canny)
#     center, wh, theta = cv2.minAreaRect(non_zero_pixels)
    
#     root_mat = cv2.getRotationMatrix2D(center, angle, 1)
#     # rows, cols = img.shape
#     rows, cols = canny.shape
#     # rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
#     rotated = cv2.warpAffine(canny, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

#     #Border removing:
#     sizex = np.int0(wh[0])
#     sizey = np.int0(wh[1])
#     print(theta)
#     if theta > -45 :
#         temp = sizex
#         sizex= sizey
#         sizey= temp
#     return cv2.getRectSubPix(rotated, (sizey,sizex), center)
  


# canny >> houghline

# angel = compute_skew(file_path)
# dst = deskew(file_path, angel)
# cv2.imshow("original", original)
# cv2.imshow("Result",dst)
# cv2.waitKey(0)