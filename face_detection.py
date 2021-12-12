import cv2


image = cv2.imread('./img/id2.jpg')
image = cv2.resize(image, dsize=None, fx=0.75, fy=0.75)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(gray_image)

# face detection algorithm adaption
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # harr분류기를 통해서 clf... YOLO가 성능은 더 좋음
results=cascade.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=5, minSize=(20,20))
# print(results)

for b in results:
    # print(b)
    x,y,w,h = b
    cv2.rectangle(image, (x,y),(x+w,y+h), (0,255,0), -1)

cv2.imshow('Face Detection',image)
cv2.waitKey(0)


