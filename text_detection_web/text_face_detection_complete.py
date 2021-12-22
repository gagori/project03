import tempfile
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import skew
import camscanner
import glob
import os
import craft_show
import yolov5
import yolo

global UPLOAD_FOLDER
global RESULT_FOLDER


# path 추가 (윈도우에선 시스템환경변수설정) : 바로가기
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # harr분류기를 통해서 clf... YOLO가 성능은 더 좋음
IMAGE_SIZE = 1800

percentType = 5
detectTypeResult = 0
PersonalInfoType="미확인"

paths = os.getcwd()
UPLOAD_FOLDER = os.path.join(paths, 'static/img/origin')
RESULT_FOLDER = os.path.join(paths, 'static/img/result')
patternDir= os.path.join(paths, 'static/Config/pattern')

PatternIDCARD=glob.glob(patternDir+"/IDCARD/*.jpg")    #주민등록증
PatternDRIVER=glob.glob(patternDir+"/DRIVER/*.jpg")    #운전면허증
PatternFAMILLY=glob.glob(patternDir+"/FAMILLY/*.jpg")  #가족관계증명서
PatternPASSPORT=glob.glob(patternDir+"/PASSPORT/*.jpg")#여권
PatternRESIDENT=glob.glob(patternDir+"/RESIDENT/*.jpg")#주민등록표

def readImg(file_name) :
    img = cv2.imread(os.path.join(UPLOAD_FOLDER, file_name))
    return img

def convert_gray_color(origin_file_name):
    img = readImg(origin_file_name)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def convert_rgb_color(file_path):
    img = cv2.imread(file_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_img

# text 정제처리 : 특수문자 삭제
def clean_text(read_data):
    text = re.sub('[=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》£“¢¥”Ÿ«éȮϽñٶϴ»「—©]', '', read_data)
    return text

# 전처리 종합
def process_image_for_ocr(file_path):
    temp_filename = set_image_dpi(file_path)
    new_img = remove_noise_and_smooth(temp_filename)
    return new_img

# DPI 조절
def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, length_y = im.size
    factor = max(1, int(IMAGE_SIZE/length_x))
    size=length_x*factor, length_y*factor
    im_resized = im.resize(size,Image.ANTIALIAS)
    # print(type(im_resized))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') # 임시파일 만들어줌
    temp_filename = temp_file.name # 임시파일명은 계속바뀜
    im_resized.save(temp_filename, dpi=(320,320))
    return temp_filename

def image_smoothing(img):
    ret1, th1 = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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
    return(or_image)


def rectangle_detect(origin_file_name, lang='kor'):
    ###################### skew correction ###############################
    print(os.path.join(UPLOAD_FOLDER, origin_file_name))
    src, theta = skew.compute_skew(os.path.join(UPLOAD_FOLDER, origin_file_name))
    img = skew.deskew(src, theta)  # Skew Correction 진행된 최종 사진 : img
    # img = camscanner.scanner(img)
    ###################### preprocessing #################################
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (1, 1), 0)  # denoising
    _, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # gray scale을 받음. binarization

    ###################### face detection with yolo ########################
    # img1 = yolov5.yolov5(img)
    img = yolo.yolo(img,0.6,0.5,True)
    ###################### text detection + recognition #################################
    _,dst = craft_show.craft_tesseract(img, th1)

    file_name = random_name()
    cv2.imwrite(os.path.join(RESULT_FOLDER, file_name + '.jpg') , img)
    return file_name


def id_info(origin_file_name, lang='kor'):
    gray_img = convert_gray_color(origin_file_name)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    cv2.imwrite(temp_filename,gray_img)
    gray_image = process_image_for_ocr(temp_filename)
    text = clean_text(pytesseract.pytesseract.image_to_string(gray_image, lang=lang))
    # text,_ = craft_show.craft_tesseract(gray_image,gray_image)
    return text

def diffImg(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    # outImg = np.empty((1,1))
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], outImg, flags=2)

    # plt.imshow(img3),plt.show()


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []

    for m, n in matches:
        if m.distance < 0.71 * n.distance:
            good.append([m])

    # print('PersonalType Good Detected Count: %d'%(len(good)))
    if len(good) > percentType:
        knn_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

        #plt.imshow(knn_image)
        #plt.show()

    return len(good)


def detectedType(OrgMat,PatternMat):
    #iPattermImg = cImg.readImg(pattern)
    #img2 = cImg.readImg(orgMat)
    if diffImg(OrgMat, PatternMat) > percentType:
        result = 1
    else:
        result = 0

    return result

def getType(origin_file_name) :
    detectTypeResult = 0
    PersonalInfoType="미확인"

    inputImg = readImg(origin_file_name)

    # 주민등록증
    if detectTypeResult == 0:
        # print('\n###DEBUG_LoadPatternData:[ Pattern_IDCARD ]')
        for i in range(0,len(PatternIDCARD)):
            _iPattermImg = readImg(PatternIDCARD[i])
            # print('LoadPatternIMG: [%s]' %(PatternIDCARD[i]))
            if detectedType(inputImg,_iPattermImg) != 0:
                detectTypeResult = 1
                PersonalInfoType = '주민등록증'

                #print('PersonalType Detected: %s' % (PersonalInfoType))
                break
    # 운전면허증
    if detectTypeResult == 0:
        # print('\n###DEBUG_LoadPatternData:[ Pattern_DRIVER ]')
        for i in range(0, len(PatternDRIVER)):
            _iPattermImg = readImg(PatternDRIVER[i])
            # print('LoadPatternIMG: [%s]' %(PatternDRIVER[i]))
            if detectedType(inputImg, _iPattermImg) != 0:
                detectTypeResult = 2
                PersonalInfoType = '운전면허증'
                #print('PersonalType Detected: %s' % (PersonalInfoType)) 
                break
    # 가족관계증명서
    if detectTypeResult == 0:
        # print('\n###DEBUG_LoadPatternData:[ Pattern_FAMILLY ]')
        for i in range(0, len(PatternFAMILLY)):
            _iPattermImg = readImg(PatternFAMILLY[i])
            # print('LoadPatternIMG: [%s]' %(PatternFAMILLY[i]))
            if detectedType(inputImg, _iPattermImg) != 0:
                detectTypeResult = 3
                PersonalInfoType = '가족관계증명서'
                # print('PersonalType Detected: %s' % (PersonalInfoType))
                break
    # 여권
    if detectTypeResult == 0:
        # print('\n###DEBUG_LoadPatternData:[ Pattern_PASSPORT ]')
        for i in range(0, len(PatternPASSPORT)):
            _iPattermImg = readImg(PatternPASSPORT[i])
            # print('LoadPatternIMG: [%s]' %(PatternPASSPORT[i]))
            if detectedType(inputImg, _iPattermImg) != 0:
                detectTypeResult = 4
                PersonalInfoType = '여권'
                # print('PersonalType Detected: %s' % (PersonalInfoType))
                break
    # 주민등록등본/초본
    if detectTypeResult == 0:
        # print('\n###DEBUG_LoadPatternData:[ Pattern_RESIDENT ]')
        for i in range(0, len(PatternRESIDENT)):
            _iPattermImg = readImg(PatternRESIDENT[i])
            print('LoadPatternIMG: [%s]' %(PatternRESIDENT[i]))
            if detectedType(inputImg, _iPattermImg) != 0:
                detectTypeResult = 5
                PersonalInfoType = '주민등록등본초본'
                # print('PersonalType Detected: %s' % (PersonalInfoType))
                break

    # 못찾음
    if detectTypeResult == 0:
        detectTypeResult = 6
        PersonalInfoType = '구분불가'

    return PersonalInfoType

def random_name(n=6) :
    import string
    import random
    ran_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))
    return ran_str