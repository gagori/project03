import cv2
import craft
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'

# text 정제처리 : 특수문자 삭제
def clean_text(read_data):
    text = re.sub('[0123456789]', '', read_data)
    return text


def craft_tesseract(dst, th1):

    ############################## Text Detection #################################################
    roi = craft.get_roi(th1)
    myData=[] 
    numData=[]
    korData=[]
    for x,r in enumerate(roi):
        left, top = r[0]-10,r[1]-5
        right, bottom = r[2]+5,r[3]+5
        imgCrop = th1[top:bottom , left:right] #h,w
        # cv2.imshow(str(x), imgCrop)
        
        ######################### Number de-identification ########################################
        # config = r'--oem 3 --psm 6 outputbase digits'
        config = r'--oem 2 --psm 6 -c tessedit_char_whitelist=0123456789-.'
        number_info = pytesseract.image_to_string(imgCrop, lang='kor',config=config)
        numData.append(number_info)
        if ('-' in number_info) and ('.' not in number_info) and len(number_info) > 8:
            cv2.rectangle(dst, (left,top), (right,bottom), (0,0,0), -1)
            # cv2.putText(dst, number_info, (r[0],r[1]+10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)

        ######################### Kor de-identification ###########################################
        kor_info = clean_text(pytesseract.image_to_string(imgCrop, lang='kor'))
        kor_info = pytesseract.image_to_string(imgCrop, lang='kor')
        # korData.append(kor_info)
        # if " " in kor_info:
        #     cv2.rectangle(dst, (left,top), (right,bottom), (255,0,0), -1)
    
    # print(numData)
    # print(len(numData))
    # print(korData)
    # print(len(korData))

    myData = "".join(numData)
    return myData, dst

# ## test ##
# file_path = "static/img/driver4.jpg"
# img = cv2.imread(file_path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(1,1),0)
# _,th1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# text, dst = craft_tesseract(img,th1)
# print("-"*45)
# print(text)
# cv2.imshow("imgShow", dst)
# cv2.waitKey(0)
