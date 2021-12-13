import cv2
import pytesseract
import re
import os
import glob

# text 정제처리 : 특수문자 삭제
def clean_text(read_data):
    text = re.sub('[=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》£“¢¥”Ÿ«éȮϽñٶϴ»「—©]', '', read_data)
    return text


# path 추가 (윈도우에선 시스템환경변수설정) : 바로가기?
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'
img = cv2.imread("static/imgfortestlocation/driver1.jpg")
# print(img.shape)
img = cv2.resize(img, dsize=(450,300), interpolation=cv2.INTER_AREA)
# print(img.shape)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clone_img = gray_img[53:168, 167:402]  # driver1
# clone_img = gray_img[50:175, 169:417] # driver3 (h,w) 슬라이싱

text = clean_text(pytesseract.pytesseract.image_to_string(clone_img, lang='kor'))
text_1 = clean_text(pytesseract.pytesseract.image_to_string(img, lang="kor"))



# print(text)
# print("*"*45)
# print(text_1)

########################################## txt 받아오기 ###############################################
# roi_location 리스트 만들기
# dir = os.path.abspath('roi_result')
# # print(dir)

# roi_location = []
# for gt_filepath in glob.glob(os.path.join(dir,'*.txt')): # 경로 불러온다. 파일의 경로 
with open(dir,'r') as f: # 파일의 내용 메모장내용 을 불러온다. 
    for row in f.read().split("\n"): # 엔터로 한줄씩 구분한다.
        print(row) 
#             current_line=[] # 빈 리스트를 만들고 박스친거를 이제 여따 넣기위해 만들었음 
#             row = row.split(" ")  
#             row1 = row[1].split("\t")

#             cx,cy,w,h = map(float, row1[:4]) # 하나하나 매칭 플롯형이였는데 인트로 해놨으니까 이제 정수형으로 출력될거임 
#             cx,cy,w,h = map(int, [cx,cy,w,h])

#             cx,cy,w,h = row1[:4]
#             # current_line.append([[cx,cy],[w,h]])  # 멀티라벨링 위함.
#             # print(current_line)
#     roi_location.append(current_line)
# print(roi_location)
# print(len(roi_location))


# # txt 불러오기 
# createFolder(path + '/int_annotation')
# file_list = os.listdir(path +'/annotations')
# file_list_py = [file for file in file_list if file.endswith(".txt")]
# # print(file_list_py)

# # print(len(roi_location) == len(file_list_py))

# # print(len(roi_location))
# for i in range(len(roi_location)):
#     cx, cy = roi_location[i][0][0]
#     w, h = roi_location[i][0][1]

#     rgb_ = str(file_list_py[i].strip(".txt"))+'.jpg'+' '+str(cx)+'\t'+str(cy)+'\t'+str(w)+'\t'+str(h)
#     f = open('./centernet_blackpink/int_annotation/'+str(file_list_py[i]),'w')
#     f.write(rgb_ + '\tjennie')  # 이 형식으로 txt에 annotation 저장됨.    
#     f.close()


# file_list_py[0]

######################################################################################################3




# cv2.cv2.rectangle(img, (167,53),(402,168), (0,255,0), 2) # driver1
# # cv2.rectangle(img, (169,50),(417,175), (0,255,0), 2)  # driver3 txt저장된 값은 y,x 순 좌상단 우하단
# cv2.imshow("img", img)
# cv2.waitKey(0)