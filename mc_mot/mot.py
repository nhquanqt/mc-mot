import numpy as np
import torch

from gat.models import GAT
from mc_mot.layers import StructuralAttentionLayer
from mc_mot.layers import TemporalAttentionLayer

import networkx as nx

class MultipleObjectTracker():
    def __init__(self, feature_extractor, link_threshold=0.8):
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        self.link_threshold = link_threshold

        self.n_track_id = 0
        self.n_nodes = 0

        self.node_features = []

        self.G = nx.Graph()

        self.gat = GAT(2048, 8, 2048, 0.1, 0.2, 8)
        self.gat.load_state_dict(torch.load('gat/data/gat_2048_weight.pth', map_location='cpu'))
        self.gat.eval()

        self.SAL = StructuralAttentionLayer(2048, 8, 1024, 0.1, 0.2, 8)
        self.SAL.load_state_dict(torch.load('mc_mot/data/SAL_weight.pth', map_location='cpu'))
        self.SAL.eval()

        self.TAL = TemporalAttentionLayer(8, 1024, 512)
        self.TAL.load_state_dict(torch.load('mc_mot/data/TAL_weight.pth', map_location='cpu'))
        self.TAL.eval()

        self.h_adj = []
        self.h_features = []

    def __call__(self, x, use_gnn=True):
        features = self.feature_extractor(x).detach()
        n_new_nodes = features.size()[0]

        self.add_nodes(features)

        if use_gnn:
            # infer_features = self.gat_infer().detach()
            infer_features = self.stgat_infer().detach()
        else:
            infer_features = torch.stack(self.node_features)

        track_ids = [-1 for _ in range(n_new_nodes)]
        max_sim = [0 for _ in range(n_new_nodes)]
        linked_node_id = [-1 for _ in range(n_new_nodes)]

        # feature matching
        for node_id in self.G.nodes:
            if node_id >= self.n_nodes - n_new_nodes:
                continue

            node = self.G.nodes[node_id]

            # cos_sim = cosine_similarity(
            #     infer_features[node_id].unsqueeze(0), 
            #     infer_features[self.n_nodes - n_new_nodes:]).view(-1)

            cos_sim = torch.sigmoid(torch.mm(
                infer_features[node_id].unsqueeze(0), 
                infer_features[self.n_nodes - n_new_nodes:].T).view(-1))

            ind = torch.argmax(cos_sim)

            if cos_sim[ind] > self.link_threshold and cos_sim[ind] > max_sim[ind]:
                track_ids[ind] = node['track_id']
                max_sim[ind] = cos_sim[ind]
                linked_node_id[ind] = node_id

        # remove same track-id situations
        for i in range(n_new_nodes):
            if track_ids[i] == -1:
                continue

            for j in range(i+1, n_new_nodes):
                if track_ids[j] == -1:
                    continue

                if track_ids[i] == track_ids[j]:
                    if max_sim[i] > max_sim[j]:
                        track_ids[j] = -1
                        linked_node_id[j] = -1
                    else:
                        track_ids[i] = -1
                        linked_node_id[i] = -1
                        break

        # add new track-id
        for i in range(len(track_ids)):
            if track_ids[i] == -1:
                track_ids[i] = self.n_track_id
                self.n_track_id += 1

        # update new node track id and link to graph
        for node_id in range(self.n_nodes - n_new_nodes, self.n_nodes):

            i = node_id - (self.n_nodes - n_new_nodes)

            self.G.add_nodes_from([(node_id, {
                'track_id': track_ids[i]
            })])

            if linked_node_id[i] != -1:
                self.G.add_edge(node_id, linked_node_id[i])

        return track_ids

    def add_nodes(self, features):
        for i, feat in enumerate(features):
            self.node_features.append(feat)
            self.G.add_node(self.n_nodes)

            # add self-loop
            self.G.add_edge(self.n_nodes, self.n_nodes)

            self.n_nodes += 1

    def gat_infer(self):
        adj = nx.to_scipy_sparse_matrix(self.G, format='coo')

        indices = np.vstack((adj.row, adj.col))
        values = adj.data

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj.shape

        adj = torch.sparse_coo_tensor(i, v, shape)
        
        return self.gat(torch.cat(self.node_features).view(-1, 2048), adj.to_dense())
        
    def stgat_infer(self):
        self.h_features.append(self.node_features.copy())
        self.h_adj.append(torch.tensor(nx.to_numpy_matrix(self.G), dtype=torch.float32))

        if len(self.h_adj) > 3:
            self.h_features.pop(0)
            self.h_adj.pop(0)

        if len(self.h_adj) == 3:
            h = []

            for i in range(len(self.h_adj)):
                adj = self.h_adj[i]
                features = torch.cat(self.h_features[i]).view(-1, 2048)

                infered_features = self.SAL(features, adj)

                if infered_features.size()[0] < self.n_nodes:
                    infered_features = torch.cat([
                        infered_features, 
                        torch.zeros(self.n_nodes - infered_features.size()[0], 1024)
                    ])

                h.append(infered_features)

            input_t = torch.stack(h).transpose(0,1)

            embedding = self.TAL(input_t)

            return embedding[:, -1]
        else:
            return self.gat_infer()