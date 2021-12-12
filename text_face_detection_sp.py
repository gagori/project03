import cv2
from numpy.lib.arraysetops import isin
import pytesseract
import numpy as np

# path 추가 (윈도우에선 시스템환경변수설정) : 바로가기?
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'

file_name = input('파일명을 입력하세요 > ')
img = cv2.imread(f'./img/{file_name}.jpg')
img = cv2.resize(img, dsize=None, fx=1.25, fy=1.25)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_img, (1, 1), 0)  # kernel size 크게 잡지 않도록 주의
sharp = np.clip(2.0*gray_img - blur, 0, 255).astype(np.uint8)
ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # gray scale을 받음에 주의
ret2, th2 = cv2.threshold(sharp, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
dx = cv2.Sobel(th1, cv2.CV_32F, 1,0)  # x방향. delta default = 0
dy = cv2.Sobel(th1, cv2.CV_32F, 0,1)  # y방향
mag = cv2.magnitude(dx, dy) # 방향상관없이
mag = np.clip(mag, 0, 255).astype(np.uint8)
canny = cv2.Canny(th1, 50, 150) # sharp th2로도 해보기
mp1 = cv2.dilate(th1, None)
se= cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
mp2 = cv2.erode(th1, se)



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
boxes_num = pytesseract.pytesseract.image_to_data(mp2, lang='kor+eng', config=config)  # 숫자만
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



# cv2.imwrite(f'./result/result_{file_name}.jpg', img)
cv2.imshow('src_for_TEXT', mp2)
cv2.imshow('FACE_TEXT_DETECTION', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

