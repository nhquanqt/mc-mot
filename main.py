import os, time

import torch
from torchvision.transforms.transforms import Pad, RandomCrop

from reid.model.model import Model
from reid.train import train_model

from torchvision import transforms

from torch.optim import Adam
from reid.model.triplet_loss import TripletLoss
from torch.optim.lr_scheduler import StepLR
from reid.dataset.market1501_trainval import Market1501TrainVal

from mc_mot.mot import MultipleObjectTracker
from mc_mot.dataset.prw import PRW

import cv2

import matplotlib.pyplot as plt
import networkx

def test_reid():
    model = Model(last_conv_stride=1)

    optimizer = Adam(model.parameters(), lr=0.0002, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    criterion = TripletLoss(0.3)

    transform_trainval = transforms.Compose([
        transforms.Resize([256, 128]),
        transforms.Pad(10),
        transforms.RandomCrop([256, 128]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloader = Market1501TrainVal('/home/wan/datasets/Market-1501-v15.09.15', transform=transform_trainval, batch_size=2)

    model = train_model(model, optimizer, criterion, scheduler, dataloader)

def test_mc_mot():
    feature_extractor = Model(last_conv_stride=1)

    checkpoint = torch.load('reid/data/model_weight.pth', map_location='cpu')
    feature_extractor.load_state_dict(checkpoint)

    mot = MultipleObjectTracker(feature_extractor)

    transform = transforms.Compose([
        transforms.Resize([256, 128]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    prw = PRW('/home/wan/datasets/PRW-v16.04.20', transform)

    for t_step in range(10000):
        input_t, frame, bboxes = prw[t_step]

        if len(input_t) == 0:
            continue

        track_ids = mot(input_t)

        for i in range(len(bboxes)):
            id, x, y, w, h = bboxes[i]
            track_id = int(track_ids[i])
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
            frame = cv2.rectangle(frame, (x, y-20), (x+40, y-1), (127, 127, 127), -1)
            frame = cv2.putText(frame, f'{track_id}', (x+2, y-2), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.75, (0, 255, 0), 2)

        print(t_step, track_ids)

        cv2.imshow('', frame)
        cv2.imwrite(os.path.join('data/output', '{:05d}.jpg'.format(t_step)), frame)

        key = cv2.waitKey(0)
        if key == 27:
            break

    networkx.draw(mot.G, with_labels=True)
    plt.show()

def main():
    test_mc_mot()

if __name__ == '__main__':
    main()