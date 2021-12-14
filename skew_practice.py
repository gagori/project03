import numpy as np
import math
import cv2


file_path = './img/driver1.jpg' 
img1 = cv2.imread(file_path)
img2 = img1.copy()
h,w = img2.shape[:2]

#load in grayscale:
src = cv2.imread(file_path,0)
_,th1 = cv2.threshold(src,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU) 
cv2.bitwise_not(th1, th1) # 반전 > th1에서 하는게 팁
canny = cv2.Canny(th1, 100, 200) # canny 필수


# Hough transform:
minLineLength = w/2.0
maxLineGap = 20
lines = cv2.HoughLinesP(canny,1,np.pi/180,100,minLineLength,maxLineGap)
# lines = cv2.HoughLinesP(canny,1,np.pi/180,500)
if lines is not None:
    for i in range(lines.shape[0]):
        pt1 = (lines[i][0][0], lines[i][0][1])  # 시작점 좌표
        pt2 = (lines[i][0][2], lines[i][0][3])  # 끝점 좌표
        cv2.line(img2, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
        cv2.circle(img2,pt1,3,(255,0,0),-1)
        cv2.putText(img2,f"{i+1}",pt1,cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255,0,0), thickness=1)


# lines = cv2.HoughLines(canny, 1,np.pi/180, 80)
# for line in lines: # 검출된 모든 선 순회
#     # print(line) #[[]]
#     r,theta = line[0] # 거리와 각도
#     tx, ty = np.cos(theta), np.sin(theta) # x, y축에 대한 삼각비
#     x0, y0 = tx*r, ty*r  #x, y 기준(절편) 좌표
#     # 기준 좌표에 빨강색 점 그리기
#     # cv2.circle(img2, (abs(x0), abs(y0)), 3, (0,0,255), -1)
#     # 직선 방정식으로 그리기 위한 시작점, 끝점 계산
#     x1, y1 = int(x0 + w*(-ty)), int(y0 + h * tx)
#     x2, y2 = int(x0 - w*(-ty)), int(y0 - h * tx)
#     cv2.line(img2, (x1, y1), (x2, y2), (0,0,255), 1)


cv2.imshow("canny",canny)
cv2.imshow("img2",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#calculate the angle between each line and the horizontal line:
angle = 0.0
# nb_lines = len(lines) # 선의 개수
# print("line len:", nb_lines)
angle_list = []
for line in lines:
    angle = math.atan2(line[0][3]*1.0 - line[0][1]*1.0,line[0][2]*1.0 - line[0][0]*1.0)
    # print(angle)
    angle_list.append(angle)
print(angle_list)



    # if (line[0][3]*1.0 - line[0][1]*1.0) > 0:
    #     angle += math.atan2(line[0][3]*1.0 - line[0][1]*1.0,line[0][2]*1.0 - line[0][0]*1.0) # atan 두점 사이의 절대각도를 계산함.
    #     print("우하향")
    #     print(angle)
    #     answer = angle* 180.0 / np.pi
    #     print(answer) 

    # else:
    #     angle += math.atan2(line[0][3]*1.0 - line[0][1]*1.0,line[0][2]*1.0 - line[0][0]*1.0)
    #     print("우상향")
    #     print(angle)
    #     answer = angle* 180.0 / np.pi
    #     print(answer) 
    



# for line in lines:
#     r,theta = line[0] # 거리와 각도
#     tx, ty = np.cos(theta), np.sin(theta) # x, y축에 대한 삼각비
#     x0, y0 = tx*r, ty*r  #x, y 기준(절편) 좌표
#     x1, y1 = int(x0 + w*(-ty)), int(y0 + h * tx)
#     x2, y2 = int(x0 - w*(-ty)), int(y0 - h * tx)
#     angle += math.atan2(y2*1.0-y1*1.0, x2*1.0-x1*1.0);  # atan 두 점사이의 절대각도를 계산함.

# angle /= nb_lines*1.0  # 평균각도구나...
print("-"*30)
print(angle_list[0])
answer = angle_list[0]* 180.0 / np.pi
print(answer) 




## deskew ##
# 이미지의 중심점을 기준으로 angle도 회전 하면서 1.0배 Scale
M= cv2.getRotationMatrix2D((w/2, h/2), answer, 1.0) # 변환행렬
dst = cv2.warpAffine(img1, M,(w, h))

cv2.imshow('Original', img1)
cv2.imshow('Rotation', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()




# def deskew(file_name,angle):
#     #load in grayscale:
#     img = cv2.imread(file_name,0)
#     canny = cv2.Canny(img, 50, 150)

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
  


# # canny >> houghline

# # angle = compute_skew(file_path)
# dst = deskew(file_path, angle)
# cv2.imshow("original", img1)
# cv2.imshow("Result",dst)
# cv2.waitKey(0)