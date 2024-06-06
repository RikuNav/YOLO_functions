import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

model = YOLO('best.pt')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to('cpu')

with torch.no_grad():
    img = cv2.imread('image.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)

    for r in results:
        image = r.plot(boxes=None)
        probs = r.probs
        plt.imshow(image)
        plt.show()

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
