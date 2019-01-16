#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


def main():
    labels = sorted([i[:-4] for i in os.listdir('icons')])

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, len(labels))
    state_dict = torch.load('resnet18_1_0.00004.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    cam = cv2.VideoCapture(0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    while True:
        ret_val, img = cam.read()
        image = Image.fromarray(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        net_input = transform(image.resize((224, 224), Image.LANCZOS))
        net_input = torch.unsqueeze(net_input, 0)

        outputs = model(net_input)
        outputs = outputs[0]
        for i in range(len(outputs)):
            if outputs[i] > 0.9:
                print(labels[i])
        print('-------')

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


