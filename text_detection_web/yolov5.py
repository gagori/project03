import torch
import cv2

def yolov5(img):
    # Image
    # img_path = 'static/img/driver4.jpg'
    # img = cv2.imread(img_path)

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Inference
    results = model(img)
    out = results.pandas().xyxy[0]

    # bbox
    out_list=[]
    for i in range(len(out)):
        outs = out.iloc[i,:].tolist()
        out_list.append(outs)
        # 좌표
        left= int(out_list[i][0])
        top= int(out_list[i][1])
        right=int(out_list[i][2])
        bottom=int(out_list[i][3])
        label = out_list[i][6]
        cv2.rectangle(img, (left,top),(right,bottom),(0,0,0),-1)
        # cv2.putText(img,"FACE",(left,top+10),cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,0,255), thickness=1)

    return img


## test ##

# img_path = 'static/img/driver4.jpg'
# img = cv2.imread(img_path)
# dst = yolov5(img)
# cv2.imshow("img", dst)
# cv2.waitKey(0)