import cv2
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)

def get_roi(image):

    # load models
    refine_net = load_refinenet_model(cuda=False)
    craft_net = load_craftnet_model(cuda=False)

    # perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.9,
        link_threshold=0.5, # 클수록 잘게 자름
        low_text=0.6,  # 작을수록 bboxes 오차범위 키움
        long_size=1280
        )

    # get roi
    roi = []
    for i in range(len(prediction_result["boxes"])):
        left, top = map(int, prediction_result["boxes"][i][0])
        right, bottom = map(int, prediction_result["boxes"][i][2])
        # print(left, top, right, bottom)
        roi.append([left,top,right,bottom])
        # print(roi)

    return roi

