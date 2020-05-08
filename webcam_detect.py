import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
np.random.seed(12345)
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#from label_detect import classify_face



model_path = 'mascarillas.0001lr_4bs'
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
num_classes = 2
model.fc = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(model_path,map_location='cpu'))
model.eval()

def transform_image(image_bytes):
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor(), 
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])
    image = Image.fromarray(image_bytes)
    return prediction_transform(image)[:3,:,:].unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = int(y_hat.item())
    class_names=['Sin mascarilla','Con mascarilla']
    return class_names[predicted_idx]


#face = cv2.CascadeClassifier('/media/preeth/Data/prajna_files/Drowsiness_detection/haar_cascade_files/haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml.1')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
thicc=2

while(True):
    ret, frame = cap.read()
    height,width = (100,100)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    label = get_prediction(frame)
    for (x, y, w, h) in faces:
        color = (0, 255, 0) if label == "Con mascarilla" else (0, 0, 255)
        cv2.putText(frame, label, (x, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()