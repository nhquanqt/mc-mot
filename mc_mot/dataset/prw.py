import os
import cv2
import torch
from torchvision import transforms
import numpy as np
from scipy.io import loadmat
from PIL import Image

class PRW():
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.frames = os.listdir(os.path.join(self.root, 'frames'))

        self.frames = sorted(self.frames)
        self.frame_bboxes = {}

        for frame_file in self.frames:
            meta = loadmat(os.path.join(self.root, 'annotations', frame_file + '.mat'))

            if 'box_new' in meta.keys():
                bboxes = meta['box_new']
            elif 'anno_file' in meta.keys():
                bboxes = meta['anno_file']
            else:
                bboxes = meta['anno_previous']

            self.frame_bboxes[frame_file] = []

            for bbox in bboxes:
                id, x, y, w, h = map(int,bbox)

                if id == -2:
                    continue
                
                self.frame_bboxes[frame_file].append([id, x, y, w, h])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = Image.open(os.path.join(self.root, 'frames', self.frames[index]))

        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)

        im_h, im_w, _ = frame.shape

        frame_t = transforms.ToTensor()(frame)

        bboxes = self.frame_bboxes[self.frames[index]]

        images = []

        for bbox in bboxes:
            id, x, y, w, h = bbox
            
            x = max(x,0)
            y = max(y,0)
            w = min(im_w-x,w)
            h = min(im_h-y,h)

            image = frame_t[:, y:y+h, x:x+w]

            if self.transform is not None:
                image = transforms.ToPILImage()(image)
                image = self.transform(image)

            images.append(image.unsqueeze(0))
        
        if len(images) == 0:
            return [], frame, bboxes

        if self.transform is not None:
            input_t = torch.cat(images)
            return input_t, frame, bboxes

        return images, frame, bboxes


if __name__ == '__main__':
    prw = PRW('/home/wan/datasets/PRW-v16.04.20')

    images, frame = prw[45]
    
    for image in images:
        cv2.imshow('', image)
        key = cv2.waitKey(0)
        if key == 27:
            break