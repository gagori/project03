import tempfile
import sys
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import time
import platform
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
from numpy.lib.arraysetops import isin

# path 추가 (윈도우에선 시스템환경변수설정) 
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # harr분류기를 통해서 clf... YOLO가 성능은 더 좋음
file_path = '.static//img/face.jpg'
IMAGE_SIZE = 1800


def convert_gray_color(file_path):
    img = cv2.imread(file_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def convert_rgb_color(file_path):
    img = cv2.imread(file_path)
    rgb_img = cv2.cvtColor(file_path, cv2.COLOR_BGR2RGB)
    return rgb_img

# text 정제처리 : 특수문자 삭제
def clean_text(read_data):
    text = re.sub('[=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》£“¢¥”Ÿ«éȮϽñٶϴ»「—©]', '', read_data)
    return text

# 전처리 종합.
def process_image_for_ocr(file_path):
    temp_filename = set_image_dpi(file_path)
    new_img = remove_noise_and_smooth(temp_filename)
    return new_img

# DPI 조절
def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, length_y = im.size
    factor = max(1, int(IMAGE_SIZE/length_x))  # length가 1800보다 크면
    size=length_x*factor, length_y*factor
    im_resized = im.resize(size,Image.ANTIALIAS)
    # print(type(im_resized))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') # 임시파일을 하나 만들어줌
    temp_filename = temp_file.name # 임시파일명은 계속바뀜
    im_resized.save(temp_filename, dpi=(320,320))
    return temp_filename

def image_smoothing(img):
    ret1, th1 = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # 두가지 방법 합성
    blur = cv2.GaussianBlur(th2, (1,1), 0)
    ret3, th3 = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3

# 노이즈 제거
def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, flags=0)  # Gray scale
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1,1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel=kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel=kernel)
    img = image_smoothing(img)
    or_image = cv2.bitwise_or(img, closing)
    # print(or_image.shape) # 0,255의 흑백이미지로 바꿔줌 확인
    return(or_image)


def rectangle_detect(file_path, lang='kor'):
    gray_img = convert_gray_color(file_path)
    faces = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(5,5)) # 숫자 바꿔보기!! 낮추니까 얼굴을 더 잘잡았음.

    for b in faces:
        x,y,w,h = b
        cv2.rectangle(gray_img, (x,y),(x+w,y+h), (0,255,0), -1)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') # 임시파일을 하나 만들어줌
    temp_filename = temp_file.name
    cv2.imwrite(temp_filename, gray_img)
    gray_img = process_image_for_ocr(temp_filename)
    hImg, wImg = gray_img.shape
    config = r'--oem 1 --psm 6 outputbase digits'
    num_boxes = pytesseract.pytesseract.image_to_data(gray_img, lang=lang, config=config)
    text = clean_text(pytesseract.pytesseract.image_to_string(gray_img, lang=lang))

    num_list=[]
    file_name=""
    for idx, b in enumerate(num_boxes.splitlines()):
        if idx !=0:
            b = b.split()
            if len(b) ==12 :
                if ('-' in b[11]) and (len(b[11])>8):
                    num_list.append(b)

    for i in range(len(num_list)):
        # print(num_list[i])
        id_num = num_list[i][11]
        if len(id_num) >= 14 and ('.' not in id_num):
            x,y,w,h = int(num_list[i][6]),int(num_list[i][7]),int(num_list[i][8]),int(num_list[i][9])
            cv2.rectangle(gray_img, (x,y),(x+w,y+h), (0,255,0), -1)

            name = clean_text(num_list[i][11])
            for i in range(len(name)):
                if i % 2 ==0 :
                    file_name += name[i]


    if file_name:
        cv2.imwrite(f'static/result/{file_name}.jpg' , gray_img)
        with open(f'static/result/{file_name}.txt', 'w', encoding="UTF-8") as f :
            f.write(text)
            f.close()
        return f'static/result/{file_name}'
        
    else:
        import string
        import random
        number_of_strings = 5
        length_of_string = 8
        for x in range(number_of_strings):
            temp_filename = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length_of_string))
        print(temp_filename)
        cv2.imwrite(f'static/result/{temp_filename}.jpg' , gray_img)
        with open(f'static/result/{temp_filename}.txt', 'w', encoding="UTF-8") as f :
            f.write(text)
            f.close()
        return f'static/result/{temp_filename}'


def id_info(file_path, lang='kor'):
    gray_img = convert_gray_color(file_path)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    # print(temp_file.name)
    temp_filename = temp_file.name
    cv2.imwrite(temp_filename,gray_img)
    gray_image = process_image_for_ocr(temp_filename)
    text = clean_text(pytesseract.pytesseract.image_to_string(gray_image, lang=lang))
    return text








