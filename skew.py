import numpy as np
import math
import cv2


def compute_skew(file_name):
    #load in grayscale:
    src1 = cv2.imread(file_name) # original_image
    src2 = cv2.imread(file_name,0)  # gray
    img = src2.copy()
    _,th1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    cv2.bitwise_not(th1, th1)
    canny = cv2.Canny(th1, 100, 200)
    
    # height, width = src.shape[0:2]
    h, w = img.shape[0:2]
    
    # Hough transform:
    minLineLength = w/2.0
    maxLineGap = 20
    lines = cv2.HoughLinesP(canny,1,np.pi/180,100,minLineLength,maxLineGap)
    # lines = cv2.HoughLinesP(canny,1,np.pi/180,500)
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])  # 시작점 좌표
            pt2 = (lines[i][0][2], lines[i][0][3])  # 끝점 좌표
            cv2.line(img, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
            cv2.circle(img,pt1,3,(255,0,0),-1)
            cv2.putText(img,f"{i+1}",pt1,cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255,0,0), thickness=1)

    #calculate the angle between each line and the horizontal line:
    angle = 0.0
    angle_list = []
    for line in lines:
        angle = math.atan2(line[0][3]*1.0 - line[0][1]*1.0,line[0][2]*1.0 - line[0][0]*1.0)
        angle_list.append(angle)

    theta = angle_list[0]* 180.0 / np.pi
    return src1, theta   



## deskew ##
def deskew(original_img, theta):
    h,w = original_img.shape[:2]
    # 이미지의 중심점을 기준으로 theta도 회전 하면서 1.0배 Scale
    M= cv2.getRotationMatrix2D((w/2, h/2), theta, 1.0) # 변환행렬
    dst = cv2.warpAffine(original_img, M,(w, h))
    return dst




################################################ test ####################################################
# file_name = "./img/driver1.jpg"
# img, theta = compute_skew(file_name)
# print(theta)
# dst = deskew(img,theta)

# cv2.imshow("original", img)
# cv2.imshow("deskew",dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows
