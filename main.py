import detectron2
from detectron2.utils.logger import setup_logger


# import some common libraries
import numpy as np
import cv2
import random
from fastapi import FastAPI,File, UploadFile
import numpy as np
import io
import base64
from PIL import Image
import cv2
import base64


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


app = FastAPI()


def encode(img):
    pil_img = Image.fromarray(img)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return  new_image_string
@app.post("/detect")
async def root(file: bytes = File(...)):
    try:
        image = Image.open(io.BytesIO(file)).convert("RGB")
        img = np.array(image)
        img = img[:, :, ::-1]

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.70  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(img)
        predictions = outputs["instances"].to("cpu")
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        boxes = v._convert_boxes(predictions.pred_boxes)
        scores = predictions.scores.numpy()
        classes = predictions.pred_classes.numpy()
        print(v._convert_boxes(boxes))
        print(scores)
        print(classes)
        mapping=v.metadata.get("thing_classes", None)

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        response={}
        response["objects_count"]=len(boxes)
        response["objects"]=[]
        for i in range(len(boxes)):
            a={}
            a["box"]=(int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][2]),int(boxes[i][3]))
            a["scores"]=int(scores[i])
            a["class"]=mapping[classes[i]]
            response["objects"].append(a)
        response["image"]=encode(out.get_image()[:, :, ::-1])
        return response
    except Exception as e:
        return {"Error": "Unexpected error occured"}


# cv2.imshow('a',out.get_image()[:, :, ::-1])
# cv2.waitKey(10000)
