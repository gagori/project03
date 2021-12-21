from sys import flags
import cv2
import numpy as np
import pytesseract
import os
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'


# 꼭지점 찾기
def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2), dtype=np.float32)
    #x,y ?
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
    #w,h ?
    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
    return hnew


def scanner(img):  # deskew 된 이미지를 받게될 것.
    # img = cv2.imread(file_path)
    # img = cv2.resize(img, (1300,800))
    # img2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    canny = cv2.Canny(blurred, 100,200)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 외곽 + 꼭지점만
    contours = sorted(contours, key=cv2.contourArea, reverse=True) # 가장 큰 윤곽선만 찾아내기
    contour = contours[0]

    for c in contours:
        p = cv2.arcLength(c, True) # True 사각형 등 단순한 모양
        approx = cv2.approxPolyDP(c, 0.02*p, True) # 윤곽선정보에 대해 엡실론만큼 오차범위 지정
        if len(approx) == 4 : # 사각형이니 4개 점
            target = approx # 변수분리
            break

    approx1=mapp(target)
    pts = np.float32([[0,0],[450,0],[450,300],[0,300]]) # windowsize <<좌상,우상,우하,좌하
    op = cv2.getPerspectiveTransform(approx1,pts)
    dst = cv2.warpPerspective(img,op,(450,300)) # 원근법 적용
    
    # cv2.drawContours(img, contours, -1, (0,255,0), 4)
    # cv2.imshow("test",img)
    cv2.drawContours(img, [approx], -1, (0,0,255), 2)
    return dst

## test ##
# file_path = "static/img/im1.jpg"
# img = cv2.imread(file_path)
# img2 = img.copy()
# dst = scanner(img2)

# # cv2.imshow("original", img)
# cv2.imshow('Contour', img2)
# cv2.imshow("Scanned Image", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##########################################################################################################3
# per=25

# imgQ = cv2.imread("static/Config\pattern/DRIVER/2.jpg")
# h,w,_ = imgQ.shape
# imgQ = cv2.resize(imgQ,(w//3,h//3))  # 중간과정 확인용 resize

# orb = cv2.ORB_create(1000)
# kp1, des1 = orb.detectAndCompute(imgQ, None)
# # imgKp1 = cv2.drawKeypoints(imgQ, kp1, None)

# path = 'static/img'
# file_list = os.listdir(path)
# myPicList = [file for file in file_list if file.endswith(".jpg")]

# print(myPicList)
# # for i,y in enumerate(myPicList):
# y = 'im01.jpg'
# img = cv2.imread(path+"/"+ y)
# img = cv2.resize(img, (w//3,h//3))
# masked_img = cv2.bitwise_and(img, imgQ)
# # cv2.imshow(y,img)
# kp2, des2 = orb.detectAndCompute(img, None)
# # id form과 매칭
# bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# matches = bf.match(des2,des1)
# matches.sort(key=lambda x:x.distance)
# good = matches[:int(len(matches)*(per/100))]  #25% best matches
# imgMatch = cv2.drawMatches(img, kp2,imgQ,kp1,good[:100],None,flags=2)
# imgMatch=cv2.resize(imgMatch,(w//3,h//3))
# # cv2.imshow(y,imgMatch)   # todo : 우선 bitwise로 사각형박스를 나타내야 orb match 잘 될듯??

# # 매핑
# srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
# dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
# M,_ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
# imgScan = cv2.warpPerspective(img,M,(w,h))
# cv2.imshow(y, imgScan)
# # cv2.imshow("KeyPoints from",imgKp1)
# cv2.imshow("id form",imgQ)
# cv2.waitKey(0)


# ###############################################################

# ## dst를 받아서 해보기
# roi_IDCARD= [[(150, 300), (924, 444), 'text', 'name'],
#             [(170, 488), (1014, 634), 'text', 'idnumber'], 
#             [(138, 638), (1124, 910), 'text', 'address']]

# roi_DRIVER = [[(192, 44), (465, 88), 'text', 'license number'], 
#                 [(193, 84), (383, 113), 'text', 'name'], 
#                 [(192, 110), (383, 140), 'text', 'id number'],
#                 [(191, 137), (422, 205), 'text', 'address']]


# # imgShow = cv2.imread("static/img/driver3.jpg")
# imgShow = dst.copy()
# imgQ = cv2.imread("static/Config\pattern/DRIVER/2.jpg")
# h,w=imgQ.shape[:2]
# imgShow = cv2.resize(imgShow, (w,h))
# # imgMask = np.zeros_like(imgQ)
# myData=[]
# # print(f' ###################### Extracting Data from Form {j} ######################')
# for x,r in enumerate(roi_DRIVER):
#     # cv2.rectangle(imgMask, (r[0][0],r[0][1],r[1][0],r[1][1]), (0,255,0), cv2.FILLED)  # 마스크의 사이즈로 맞춰야함.
#     # imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

#     imgCrop = imgShow[r[0][1]:r[1][1] , r[0][0]:r[1][0]]  # h,w 순 슬라이싱
#     cv2.imshow(str(x), imgCrop)

#     if r[2] == "text":
#         config = r'--oem 2 --psm 6 outputbase digits'
#         # config = r'--oem 2 --psm 6 -c tessedit_char_whitelist=0123456789-.'
#         print(f"{r[3]} : {pytesseract.image_to_string(imgCrop, lang='kor')}")
#         myData.append(pytesseract.image_to_string(imgCrop,lang='kor'))
#     cv2.putText(imgShow, str(myData[x]),(r[0][0],r[0][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)


# print(myData)
# # cv2.imshow("mask",imgMask)
# cv2.imshow("imgShow", imgShow)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



