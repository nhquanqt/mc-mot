import time

import numpy as np

import torch
from torch.autograd import Variable

from reid.losses import global_loss

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def train_model(model, optimizer, criterion, scheduler, dataloader, n_epochs=100):

    model.train()

    for epoch in range(n_epochs):
        
        print(f'epoch {epoch}')

        dataloader.start_over()

        total_loss = []

        t_now = time.time()

        while not dataloader.epoch_done:
            images_t, labels_t = dataloader.next_batch()
            images_t = Variable(images_t.to(DEVICE))

            optimizer.zero_grad()

            feat = model(images_t)

            loss, dist_ap, dist_an = global_loss(criterion, feat, labels_t)

            loss.backward()
            optimizer.step()

            # log
            total_loss.append(loss.item())


        print(f'ran in {time.time() - t_now} seconds')
        print(f'mean loss = {np.mean(total_loss)}')

        scheduler.step()

    return model