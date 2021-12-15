import cv2
import numpy as np


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


file_path = "./img/id4.jpg"

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
    cv2.drawContours(img, [approx], -1, (0,0,255), 2)
    return dst


## test ##
# img = cv2.imread(file_path)
# img2 = img.copy()
# dst = scanner(img2)

# cv2.imshow("original", img)
# cv2.imshow('Contour', img2)
# cv2.imshow("Scanned Image", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


