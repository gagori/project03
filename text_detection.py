import cv2
from numpy.lib.arraysetops import isin
import pytesseract
import re

# path 추가 (윈도우에선 시스템환경변수설정) : 바로가기?
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'

img = cv2.imread('./img/driver3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 희동구는 애초에 bgr 였나?
# cv2.imshow('img',img)
# cv2.waitKey(0)

# print(img.shape)
hImg,wImg,_ = img.shape
config = r'--oem 3 --psm 6 outputbase digits' # 숫자만 

########################### image_to_boxes ################################
# boxes = pytesseract.pytesseract.image_to_boxes(img, lang='kor+eng', config=config)  # 숫자만
# # print(boxes)

# for b in boxes.splitlines(): # 한 줄 씩 split
#     # print(b)
#     b = b.split(' ')  # 나눠서 list에 담아줌
#     # print(b)
#     x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv2.rectangle(img, (x,hImg-y),(w,hImg-h),(0,255,0), 2)  # 좌하단 우상단...?

# cv2.imshow('TEXT_DETECTION', img)
# cv2.waitKey(0) 


############################## image_to_data ##########################################
############################## image_to_string ########################################
boxes = pytesseract.pytesseract.image_to_string(img, lang='kor+eng')  # 모든 text 
boxes_num = pytesseract.pytesseract.image_to_data(img, lang='kor+eng', config=config)  # 숫자만
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
        print(b)
        if len(b) == 12:  # 객체가 있는것만 뽑아서
            if '-' in b[11]:
            # if len(b[11]) >= 13:  # 보통 주민등록번호는 len 13 길다 : 추출한 객체가 11개 이상의 len이면
            # if float(b[10]) > 0.5 :  # confidence level 높은것, 즉 숫자로 인식한거 중 진짜 숫자인 것만 가리기
                # num_list.append(b)  # 운전면허용
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0), -1)   # 좌상단, 우하단 >> 시작점, 끝점이니까 좌하단 우상단 해도 상관 없음.
                cv2.putText(img, b[11], (x,y+25), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,0,255), thickness=1)


# # print(num_list)
# for i in range(2):
#     '''
#     <운전면허증용> 
#     13개이상 잡히는 숫자가 여러개여서 그중에 주민번호와 운전면허id만 특정하여 가리는 알고리즘
#     '''
#     x,y,w,h = int(num_list[i][6]),int(num_list[i][7]),int(num_list[i][8]),int(num_list[i][9])
#     cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0), -1)

# # for idx, b enumerate(boxes.splitlines()):
# #     if i!=0:
# #         b=b.split()
# #         if len(b) == 12:
# #             print(b[11])


# boxes_list = []
# num_list = []
# for b in boxes.splitlines():
#     '''
#     인식된 text를 모두 리스트에 담아, 
#     필요한 정보만 indexing 할 수 있는 알고리즘 
#     '''
#     # print(b)
#     boxes_list.append(b)  # list에 담으면 indexing으로 원하는 것만 가져오기 편하므로
#     # num_list.append(b)
# print(boxes_list)  # 만약 깨진거 있으니까 frozen_east_text_detection 딥러닝으로 pretrained된 모델을 가져와서 쓰면 잘 인식함.
# # print(num_list)

cv2.imshow('TEXT_DETECTION', img)
cv2.waitKey(0) 



