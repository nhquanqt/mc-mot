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

def main():
    model = Model(last_conv_stride=1)

    optimizer = Adam(model.parameters(), lr=0.02, weight_decay=0.0005)
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

if __name__ == '__main__':
    main()