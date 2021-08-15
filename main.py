import os, time

import torch

from torchvision import transforms

from mc_mot.mot import MultipleObjectTracker
from mc_mot.dataset.prw import PRW

import torchreid
import cv2

import matplotlib.pyplot as plt
import networkx

def test_mc_mot():
    feature_extractor = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000
    )

    mot = MultipleObjectTracker(feature_extractor)

    transform = transforms.Compose([
        transforms.Resize([256, 128]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    prw = PRW('/home/wan/datasets/PRW-v16.04.20', transform)

    for t_step in range(0, 10000):
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