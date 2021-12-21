import cv2
import craft
import pytesseract
import camscanner
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'

file_path = "static/img/im1.jpg"
src = cv2.imread(file_path)
dst = camscanner.scanner(src) # deskew 된 이미지
roi = craft.get_roi(dst)
print(len(roi))

myData=[]
for x,r in enumerate(roi):
    left, top = r[0]-10,r[1]-5
    right, bottom = r[2]+5,r[3]+5
    # cv2.rectangle(dst, (left,top),(right,bottom), (0,0,0), 1)
    imgCrop = dst[top:bottom , left:right] #h,w
    # cv2.imshow(str(x), imgCrop)
    
    # text recognition
    gray_img = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (3, 3), 0)  # denoising
    _, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # gray scale을 받음. binarization
    # cv2.imshow(str(x), th1)
    
    config = r'--oem 3 --psm 6 outputbase digits'
    # config = r'--oem 2 --psm 6 -c tessedit_char_whitelist=0123456789-.'
    info = pytesseract.image_to_string(th1, lang='kor',config=config)
    myData.append(info)
    cv2.putText(dst, str(myData[x]),(r[0],r[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)


print(myData)
print(len(myData))
cv2.imshow("imgShow", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()