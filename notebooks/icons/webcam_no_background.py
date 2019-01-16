import cv2
import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

cap = cv2.VideoCapture(0)

panel = np.zeros([100, 700], np.uint8)
cv2.namedWindow('panel')


def nothing(x):
    pass


cv2.createTrackbar('L - h', 'panel', 0, 179, nothing)
cv2.createTrackbar('U - h', 'panel', 179, 179, nothing)

cv2.createTrackbar('L - s', 'panel', 0, 255, nothing)
cv2.createTrackbar('U - s', 'panel', 60, 255, nothing)

cv2.createTrackbar('L - v', 'panel', 0, 255, nothing)
cv2.createTrackbar('U - v', 'panel', 255, 255, nothing)

cv2.createTrackbar('S ROWS', 'panel', 0, 480, nothing)
cv2.createTrackbar('E ROWS', 'panel', 480, 480, nothing)
cv2.createTrackbar('S COL', 'panel', 0, 640, nothing)
cv2.createTrackbar('E COL', 'panel', 640, 640, nothing)


# net
labels = sorted([i[:-4] for i in os.listdir('icons')])
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, len(labels))
state_dict = torch.load('resnet18_4_reduced_colorjitter_no_background_0.97250.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# net


while True:
    _, frame = cap.read()

    s_r = cv2.getTrackbarPos('S ROWS', 'panel')
    e_r = cv2.getTrackbarPos('E ROWS', 'panel')
    s_c = cv2.getTrackbarPos('S COL', 'panel')
    e_c = cv2.getTrackbarPos('E COL', 'panel')

    roi = frame[s_r: e_r, s_c: e_c]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos('L - h', 'panel')
    u_h = cv2.getTrackbarPos('U - h', 'panel')
    l_s = cv2.getTrackbarPos('L - s', 'panel')
    u_s = cv2.getTrackbarPos('U - s', 'panel')
    l_v = cv2.getTrackbarPos('L - v', 'panel')
    u_v = cv2.getTrackbarPos('U - v', 'panel')

    lower_green = np.array([l_h, l_s, l_v])
    upper_green = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(roi, roi, mask=mask)
    fg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    cv2.imshow('bg', bg)
    cv2.imshow('fg', fg)

    cv2.imshow('panel', panel)

    image = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    net_input = transform(image.resize((224, 224), Image.LANCZOS))
    net_input = torch.unsqueeze(net_input, 0)

    outputs = model(net_input)
    outputs = outputs[0]
    for i in range(len(outputs)):
        if F.sigmoid(outputs[i]) > 0.5:
            print(labels[i].replace('t-', 'cross-'))
    print('-------')

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()