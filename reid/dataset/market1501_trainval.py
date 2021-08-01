import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

from reid.utils.utils import *

class Market1501TrainVal(Dataset):
    def __init__(self, root, transform=None, batch_size=32, shuffle=True):
        self.root = root
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.split_train_val()
        
        self.query_ptr = 0
        self.query_done = False
        self.query_ids = [get_image_id(c) for c in self.query_images]
        self.query_cams = [get_image_cam(c) for c in self.query_images]
        
        self.gallery_ptr = 0
        self.gallery_done = False
        self.gallery_ids = [get_image_id(c) for c in self.gallery_images]
        self.gallery_cams = [get_image_cam(c) for c in self.gallery_images]

        self.images = self.train_images
        self.person_id_gallery = {}

        for f_image in self.images:

            person_id = int(f_image.split('_')[0])

            if person_id not in self.person_id_gallery.keys():
                self.person_id_gallery[person_id] = {
                    'images': [],
                    'ptr': 0
                }

            self.person_id_gallery[person_id]['images'].append(os.path.join(root, 'bounding_box_train', f_image))

        self.images = [os.path.join(self.root, f_image) for f_image in self.images]
        self.person_id_list = list(self.person_id_gallery.keys())
        self.id_classes = {}

        for i in range(len(self.person_id_list)):
            self.id_classes[self.person_id_list[i]] = i

        self.sample_ptr = 0
        self.epoch_done = False
        self.images_per_id = 4
        if self.shuffle:
            np.random.shuffle(self.person_id_list)

    def split_train_val(self):
        images = os.listdir(os.path.join(self.root, 'bounding_box_train'))
        images.sort()
        images.pop()

        images = np.array(images)
        np.random.shuffle(images)
        
        ids = np.array([get_image_id(f) for f in images])
        cams = np.array([get_image_cam(f) for f in images])

        unique_ids = np.unique(ids)
        np.random.shuffle(unique_ids)

        train_indices = []
        query_indices = []
        gallery_indices = []

        num_selected_ids = 0

        for unique_id in unique_ids:
            query_indices_ = []
            indices = np.argwhere(unique_id == ids).flatten()

            unique_cams = np.unique(cams[indices])

            for unique_cam in unique_cams:
                query_indices_.append(indices[np.argwhere(unique_cam == cams[indices]).flatten()[0]])
            gallery_indices_ = list(np.setdiff1d(indices, query_indices_))

            for query_index in query_indices_:
                if len(gallery_indices_) == 0 or len(np.argwhere(cams[query_index] != cams[gallery_indices_]).flatten()) == 0:
                    query_indices_.remove(query_index)
                    gallery_indices_.append(query_index)

            if len(query_indices_) == 0:
                continue

            query_indices += list(query_indices_)
            gallery_indices += list(gallery_indices_)

            num_selected_ids += 1
            if num_selected_ids >= 100:
                break

        train_indices = np.setdiff1d(range(len(images)), np.hstack([query_indices, gallery_indices]))

        self.train_images = images[train_indices]
        self.query_images = images[query_indices]
        self.gallery_images = images[gallery_indices]

        self.train_images.sort()
        self.query_images.sort()
        self.gallery_images.sort()

    def __getitem__(self, index):
        person_id = int(self.images[index].split('/')[-1].split('_')[0])
        image = Image.open(self.images[index])

        pos_image = Image.open(np.random.choice(self.person_id_gallery[person_id]['images']))

        neg_id = np.random.choice(self.person_id_list)
        while neg_id == person_id:
            neg_id = np.random.choice(self.person_id_list)

            pos_image = Image.open(np.random.choice(self.person_id_gallery[person_id]['images']))
            neg_image = Image.open(np.random.choice(self.person_id_gallery[neg_id]['images']))

        if self.transform is not None:
            return self.transform(image), self.transform(pos_image), self.transform(neg_image)

        return image, person_id

    def __len__(self):
        return len(self.images)

    def next_batch(self):

        person_ids = self.person_id_list[self.sample_ptr: self.sample_ptr + self.batch_size]

        images_t = []
        labels_t = []

        for id in person_ids:
            for _ in range(self.images_per_id):
                ptr = self.person_id_gallery[id]['ptr']
                image = Image.open(self.person_id_gallery[id]['images'][ptr])
                images_t.append(self.transform(image))
                # labels_t.append(id)
                labels_t.append(self.id_classes[id])
                self.person_id_gallery[id]['ptr'] += 1
                if self.person_id_gallery[id]['ptr'] >= len(self.person_id_gallery[id]['images']):
                    self.person_id_gallery[id]['ptr'] = 0

        images_t = torch.stack(images_t)
        labels_t = torch.tensor(labels_t, dtype=torch.int32)

        self.sample_ptr += self.batch_size
        if self.sample_ptr >= len(self.person_id_list):
            self.epoch_done = True

        return images_t, labels_t

    def next_batch_query(self):
        images_t = []
        labels_t = []

        for i in range(self.query_ptr, min(self.query_ptr + self.batch_size, len(self.query_images))):
            filename = self.query_images[i]
            image = Image.open(os.path.join(self.root, 'bounding_box_train', filename))
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

    def next_batch_gallery(self):
        images_t = []
        labels_t = []

        for i in range(self.gallery_ptr, min(self.gallery_ptr + self.batch_size, len(self.gallery_images))):
            filename = self.gallery_images[i]
            image = Image.open(os.path.join(self.root, 'bounding_box_train', filename))
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

    def start_over(self):
        self.epoch_done = False
        self.sample_ptr = 0

        if self.shuffle:
            np.random.shuffle(self.person_id_list)