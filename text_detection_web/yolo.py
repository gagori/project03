import cv2
import numpy as np

def yolo(img, conf_th, nms_th, use_copied_array=True):
    # coco classes
    classes = []
    with open("model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    # yolo 불러오기
    net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
    rows, cols = img.shape[:2]

    draw_img = None
    if use_copied_array:
        draw_img = img.copy()
    else:
        draw_img = img
    
    #전체 Darknet layer에서 13x13 grid, 26x26, 52x52 grid에서 detect된 Output layer만 filtering
    layer_names = net.getLayerNames()
    # print("ln:",layer_names)
    # print("\n",net.getUnconnectedOutLayers())
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # print("ol:",output_layers)

    # 로딩한 모델은 Yolov3 416 x 416 모델임. 원본 이미지 배열을 사이즈 (416, 416)으로, BGR을 RGB로 변환하여 배열 입력
    # Object Detection 수행하여 결과를 outs으로 반환 
    blob=cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_th:
                center_x = int(detection[0] * cols)
                center_y = int(detection[1] * rows)
                width = int(detection[2] * cols)
                height = int(detection[3] * rows)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                # 3개의 개별 output layer별로 Detect된 Object들에 대한 class id, confidence, 좌표정보를 모두 수집
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    
    # NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)
    for i in range(len(boxes)):
        if i in idxs:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            caption = f"{label}:{confidences[i]}"
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0,255,0), -1)
            cv2.putText(draw_img, "FACE", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)
            
    return draw_img


## test ##
# img = cv2.imread("static/img/driver4.jpg")
# draw_img =yolo_face(img,0.6,0.3,True)
# cv2.imshow("original image :",img)
# cv2.imshow("face detection by yolo :",draw_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()