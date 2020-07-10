import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch
import cv2
import cv2,glob
from utils import *
import numpy as np

data_transform = torchvision.transforms.Compose([
    ToTensor(),
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def get_prediction(img_path, threshold):
    # model.to('cuda')

    img = Image.open(img_path) # Load the image
    transform = data_transform # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    inputs = torch.stack([img])
    model.to(device)
    inputs = inputs.cuda()
    print(inputs.device)
    pred = model(inputs) # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score=pred_score[:pred_t+1]
    return pred_boxes, pred_class,pred_score

# pred_boxes, pred_class,scores=get_prediction('a.jpg',0.5)
# img = cv2.imread('a.jpg')
# for (box,label,score) in zip(pred_boxes, pred_class,scores):
#
#     x1=int(box[0][0])
#     y1 = int(box[0][1])
#     x2 = int(box[1][0])
#     y2 = int(box[1][1])
#     print(x1,y1,x2,y2)
#     score=round(score*100)
#     img=drawBox(img,x1,y1,x2,y2,label)
#
# cv2.imshow('a',img)
# cv2.waitKey(2000000)


