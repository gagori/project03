import cv2
from numpy.lib.arraysetops import isin
import pytesseract
import numpy as np

# path 추가 (윈도우에선 시스템환경변수설정) : 바로가기?
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'

file_name = input('파일명을 입력하세요 > ')
img = cv2.imread(f'./img/{file_name}.jpg')
# img = cv2.resize(img, dsize=None, fx=1.25, fy=1.25)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_img, (1, 1), 0)  # kernel size 크게 잡지 않도록 주의
sharp = np.clip(2.0*gray_img - blur, 0, 255).astype(np.uint8)
ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # gray scale을 받음에 주의
ret2, th2 = cv2.threshold(sharp, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mp1 = cv2.dilate(th1, None)
se= cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
mp2 = cv2.erode(th1, se)


########################### image correction ############################################################################
dx = cv2.Sobel(th1, cv2.CV_32F, 1,0)  # x방향. delta default = 0
dy = cv2.Sobel(th1, cv2.CV_32F, 0,1)  # y방향
mag = cv2.magnitude(dx, dy) # 방향상관없이
mag = np.clip(mag, 0, 255).astype(np.uint8)
canny = cv2.Canny(th1, 50, 150) # 

# lines = cv2.HoughLinesP(canny, 1.0, np.pi/180., 160, minLineLength=None, maxLineGap=None)
# if lines is not None:
#     for i in range(lines.shape[0]):
#         pt1 = (lines[i][0][0], lines[i][0][1])  # 시작점 좌표
#         pt2 = (lines[i][0][2], lines[i][0][3])  # 끝점 좌표
#         cv2.line(dst, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)


lines = cv2.HoughLines(canny, 1,np.pi/180, 100) # houghlines가 직선검출에 더 적합함.
dst = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),2)


# cv2.imshow('edges', canny)
# cv2.imshow('lines', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

########################### face detection algorithm adaption ###########################################################
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # harr분류기를 통해서 clf... YOLO가 성능은 더 좋음
results=cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(5,5)) # 숫자 바꿔보기!! 낮추니까 얼굴을 더 잘잡았음.
# print(results)  # face는 blur이전이 더 잘 잡으니까 gray_img를 받자.

for b in results:
    # print(b)
    x,y,w,h = b
    cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), -1)



########################### text detection algorithm adaption ######################################################################
'''
[후보군]
Gaussain filter + Otsu : th1 
Sharp + Otsu : th2 
Sobel : mag 
Canny : canny
dilate : mp1
erode : mp2

'''
# hImg,wImg,_ = img.shape
config = r'--oem 3 --psm 6 outputbase digits' # 숫자만 
# boxes = pytesseract.pytesseract.image_to_string(th1, lang='kor+eng')  # 모든 text 
boxes_num = pytesseract.pytesseract.image_to_data(th1, lang='kor+eng', config=config)  # 숫자만
# print(boxes)
# print(boxes_num)

num_list=[]
for idx, b in enumerate(boxes_num.splitlines()): # 한 줄 씩 split
    '''
    숫자를 가리는 알고리즘 
    '''
    # print(b)
    if idx != 0:  # head를 제외하고 split
        b = b.split()
        # print(b)
        if len(b) == 12:  # 객체가 있는것만 뽑아서
            if ('-' in b[11]) and ('.' not in b[11]) and (len(b[11])>8):
            # if ('-' in b[11]) and (len(b[11]) >= 13):
                # print(b)
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0), -1)   # 좌상단, 우하단 >> 시작점, 끝점이니까 좌하단 우상단 해도 상관 없음.
                cv2.putText(img, b[11], (x,y+25), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,0,255), thickness=1)



# cv2.imshow('src_for_TEXT', th1)
# cv2.imshow('FACE_TEXT_DETECTION', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

