import time

import numpy as np

import torch
from torch.autograd import Variable

from reid.model.losses import global_loss, euclidean_dist
from reid.utils.scores import cmc, mean_ap

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def valid_model(model, dataloader):
    # set model to evaluation mode
    model.eval()
    
    # query features
    query_feat = []
    dataloader.query_done = False
    dataloader.query_ptr = 0

    while not dataloader.query_done:
        images_t, labels_t = dataloader.next_batch_query()
        images_t = images_t.to(DEVICE)
        feat = model(images_t)
        query_feat.append(feat.cpu().detach())

    query_feat = torch.cat(query_feat)
    
    # gallery features
    gallery_feat = []
    dataloader.gallery_done = False
    dataloader.gallery_ptr = 0

    while not dataloader.gallery_done:
        images_t, labels_t = dataloader.next_batch_gallery()
        images_t = images_t.to(DEVICE)
        feat = model(images_t)
        gallery_feat.append(feat.cpu().detach())

    gallery_feat = torch.cat(gallery_feat)
    
    # calculate distance
    dist_mat = euclidean_dist(query_feat, gallery_feat)
    
    # calculate rank-k precision
    cmc_score = cmc(dist_mat, dataloader.query_ids, dataloader.gallery_ids, dataloader.query_cams, dataloader.gallery_cams, topk=5, \
                separate_camera_set=False,single_gallery_shot=False,first_match_break=True)
    
    # calculate mean average precision
    mAP = mean_ap(dist_mat, dataloader.query_ids, dataloader.gallery_ids, dataloader.query_cams, dataloader.gallery_cams)
    
    # set the model back to train mode
    model.train()
    return mAP, cmc_score

def train_model(model, optimizer, criterion, scheduler, dataloader, n_epochs=100, val_interval=10):

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

        if (epoch + 1) % val_interval == 0:
            mAP_score, cmc_score = valid_model(model, dataloader)
            print('-----------------------VALIDATION------------------------')
            print(f'Rank-1 score: {cmc_score[0]}')
            print(f'Rank-5 score: {cmc_score[4]}')
            print(f'mAP score: {mAP_score}')

    return model