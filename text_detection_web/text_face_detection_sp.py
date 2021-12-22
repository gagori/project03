import cv2
from numpy.lib.arraysetops import isin
import pytesseract
import skew
import craft_show
import yolov5 
import yolo


# path 추가 (윈도우에선 시스템환경변수설정) : 바로가기?
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'


file_path = "static/img/driver2.jpg"
src = cv2.imread(file_path)
############################# Skew Correction ##################################################################
_, theta = skew.compute_skew(file_path)
img = skew.deskew(src, theta)  # rotation 진행된 최종 사진

############################# Scanner ###########################################################################
# img = camscanner.scanner(img)

############################## Denoising & Binarization ########################################################################################
# img = cv2.resize(img, dsize=None, fx=1.25, fy=1.25)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_img, (1, 1), 0)  # denoising
ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # gray scale을 받음. binarization

############################# masking image 사용 << 어느정도 전처리 포함 ##########################################################
# masked_img = camscanner.masking(img, pattern_path)


############################# Face Detection with Haar ###########################################################
# cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # harr분류기를 통해서 clf... YOLO가 성능은 더 좋음
# results=cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(5,5)) # 숫자 바꿔보기!! 낮추니까 얼굴을 더 잘잡았음.

# # print(results)  # face는 blur이전이 더 잘 잡으니까 gray_img를 받자.

# for b in results:
#     # print(b)
#     x,y,w,h = b
#     cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), -1)
#     cv2.putText(img, "haar", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)

############################## Face Detection with YOLO #############################################################
# img1 = yolov5.yolov5(img)
img = yolo.yolo(img,0.6,0.4,False) #v3
############################# Text Detection ######################################################################
# dst = craft_show.craft_tesseract(img, th1)

config = r'--oem 3 --psm 6 outputbase digits'
# config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-.'
boxes_num = pytesseract.pytesseract.image_to_data(th1, lang='kor+eng', config=config)
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
            print(b)
            if ('-' in b[11]) and ('.' not in b[11]) and (len(b[11])>10):
            # if ('.' not in b[11]) and (len(b[11])>10):
                # print(b)
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,0), -1)   # 좌상단, 우하단 >> 시작점, 끝점이니까 좌하단 우상단 해도 상관 없음.
                # cv2.putText(img, b[11], (x,y+10), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,0,255), thickness=1)


cv2.imshow("blur",blur)
cv2.imshow("binarization", th1)
cv2.imshow("src", src)
cv2.imshow('FACE&TEXT_DETECTION', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

