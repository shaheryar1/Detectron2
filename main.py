


# import some common libraries
import numpy as np
from models import get_predictions_yolov4
from faster_rcnn import *
from fastapi import FastAPI,File, UploadFile
import numpy as np
import io

from PIL import Image
import cv2
import base64

app = FastAPI()


def encode(img):
    pil_img = Image.fromarray(img)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return  new_image_string

@app.post("/detect")
async def root(method:str ="yolo", file: bytes = File(...)):
    try:

        image = Image.open(io.BytesIO(file)).convert("RGB")
        img = np.array(image)
        image.save('t.jpg')
        if method=='yolo':
            boxes, classes, scores=get_predictions_yolov4('t.jpg')
        else:
            boxes,classes,scores=get_prediction('t.jpg',0.5)
            # box = boxes[i]
            # x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])

        response={}
        response["objects_count"]=len(boxes)
        response["objects"]=[]

        for i in range(len(boxes)):
            if method=="yolo":
                box=boxes[i]
                print(box)
                x1,y1,x2,y2=int(box[0]),int(box[1]),int(box[2]),int(box[3])
            else:
                box = boxes[i]
                x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])
            a={}
            a["box"]=(x1,y1,x2,y2)
            a["scores"]=int(scores[i]*100)
            a["class"]=classes[i]
            response["objects"].append(a)
            img = drawBox(img,x1,y1,x2,y2,classes[i])
        response["image"]=encode(img)
        return response
    except Exception as e:
        print(str(e))
        return {"Error": "Unexpected error occured"}


