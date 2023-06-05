#!/usr/bin/env python
# coding: utf-8


from PIL import Image
import cv2
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os

'''
This file uses pretrained models in torchvision to predict bounding boxes for objects in image. 
'''


from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
def fasterrcnn_resnet101_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=False,
                             trainable_backbone_layers=3, **kwargs):
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5 #TODO: whta's mean of this trainable_backbone_layers
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet152', pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model


model = fasterrcnn_resnet101_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)


#model = torchvision.models.detection.ssd300_vgg16(pretrained=True) #THIS ALSO IS ANOTHER MODEL FOR BOUNDING BOX DETECTION



model.eval()
image_path = os.getcwd().replace("\\","/") + "/../sample_train_images/table_top.jpeg"
image = Image.open(image_path)


transform = T.ToTensor()
transformed_image = transform(image)


with torch.no_grad():
    prediction = model([transformed_image])


bounded_boxes , prediction_accuracies, labels = prediction[0]["boxes"], prediction[0]["scores"], prediction[0]["labels"]


#here I am cutting the bounding boxes detected with an accuracy treshold I set
accuracy_treshold = 0.0
num= torch.argwhere(prediction_accuracies > accuracy_treshold).shape[0]
#the prediction_accuracies are ordered (decreasing)


coco_names = ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" , 
"frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , 
"baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , 
"plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , 
"banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
"pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
"mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
"laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
"oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
"clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]



font = cv2.FONT_HERSHEY_SIMPLEX
display_image = cv2.imread(image_path)
for i in range(num):
    x1, y1, x2, y2 = bounded_boxes[i].numpy()
    display_image = cv2.rectangle(display_image, (x1,y1), (x2,y2), (0,255,0), 1)
    classification = coco_names[labels.numpy()[i] - 1]
    display_image = cv2.putText(display_image, classification, (x1,y1-10), font, 0, (255,0,0), 1, cv2.LINE_AA)



cv2.imshow("Image", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





