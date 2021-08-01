import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

from reid.utils.utils import *

class Market1501Test(Dataset):
    def __init__(self, root, transform=None, batch_size=32):
        self.root = root
        self.transform = transform
        self.batch_size = batch_size

        gallery_images_ = os.listdir(os.path.join(root, 'bounding_box_test'))
        gallery_images_.sort()
        gallery_images_.pop()

        self.gallery_images = []
        for filename in gallery_images_:
            if not filename.startswith('-1'):
                self.gallery_images.append(filename)

        self.gallery_ids = [get_image_id(c) for c in self.gallery_images]
        self.gallery_cams = [get_image_cam(c) for c in self.gallery_images]
                
        self.gallery_ptr = 0
        self.gallery_done = False

        self.query_images = os.listdir(os.path.join(root, 'query'))
        self.query_images.sort()
        self.query_images.pop()
        
        self.query_ids = [get_image_id(c) for c in self.query_images]
        self.query_cams = [get_image_cam(c) for c in self.query_images]

        self.query_ptr = 0
        self.query_done = False

    def next_batch_gallery(self):
        images_t = []
        labels_t = []

        for i in range(self.gallery_ptr, min(self.gallery_ptr + self.batch_size, len(self.gallery_images))):
            filename = self.gallery_images[i]
            image = Image.open(os.path.join(self.root, 'bounding_box_test', filename))
            if self.transform is not None:
                image = self.transform(image)
            images_t.append(image)
            labels_t.append(int(filename.split('_')[0]))
    
        self.gallery_ptr += self.batch_size
        if self.gallery_ptr >= len(self.gallery_images):
            self.gallery_ptr = 0
            self.gallery_done = True

        images_t = torch.stack(images_t)
        labels_t = torch.tensor(labels_t).long()

        return images_t, labels_t

    def next_batch_query(self):
        images_t = []
        labels_t = []

        for i in range(self.query_ptr, min(self.query_ptr + self.batch_size, len(self.query_images))):
            filename = self.query_images[i]
            image = Image.open(os.path.join(self.root, 'query', filename))
            if self.transform is not None:
                image = self.transform(image)
            images_t.append(image)
            labels_t.append(int(filename.split('_')[0]))

        self.query_ptr += self.batch_size
        if self.query_ptr >= len(self.query_images):
            self.query_ptr = 0
            self.query_done = True
    
        images_t = torch.stack(images_t)
        labels_t = torch.tensor(labels_t).long()

        return images_t, labels_t