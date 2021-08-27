import torch
import torch.nn as nn

from stgat.layers import StructuralAttentionLayer, TemporalAttentionLayer

class STGAT(nn.Module):
    def __init__(self, n_time_steps=3):
        super(STGAT, self).__init__()

        self.sals = [StructuralAttentionLayer(512, 128, 4) for _ in range(2)]

        for i, sal in enumerate(self.sals):
            self.add_module('sal_{}'.format(i), sal)

        self.ffn = nn.Linear(512, 128)

        self.n_time_steps = n_time_steps
        self.position_embedding = nn.Embedding(n_time_steps, 128)

        self.tals = [TemporalAttentionLayer(128, 8, 16) for _ in range(2)]

        for i, tal in enumerate(self.tals):
            self.add_module('tal_{}'.format(i), tal)

    def forward(self, features, adj):
        # features.shape:   (W, N, feature_dim)
        # adj.shape:        (W, N, N)

        features = self.sals[0](features, adj)
        features = self.sals[1](features, adj)

        features = features.transpose(0, 1) # (N, W, feature_dim_)

        features = self.ffn(features)

        # position embedding in here
        n_nodes = features.size(0)
        device = features.device
        pos_input = torch.arange(0, self.n_time_steps).view(1, -1).repeat(n_nodes, 1).long().to(device)

        features += self.position_embedding(pos_input)

        features = self.tals[0](features)
        features = self.tals[1](features)

        return features



            
