import numpy as np
import torch

import networkx as nx

from stgat.models import STGAT

import torch.nn.functional as F

class MultipleObjectTracker():
    def __init__(self, feature_extractor, feature_dim=512, hidden_dim=128, embedding_dim=8):
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.n_track_id = 0
        self.n_nodes = 0

        self.node_features = []

        self.G = nx.Graph()

        self.stgat = STGAT()
        self.stgat.load_state_dict(torch.load('stgat/data/model_weight_last.pth', map_location='cpu'))
        self.stgat.eval()

        self.h_adj = [torch.zeros(1, 1) for _ in range(3)]
        self.h_features = [torch.zeros(1, 512) for _ in range(3)]

    def __call__(self, x, link_threshold=0.8):
        features = self.feature_extractor(x).detach()
        n_new_nodes = features.size(0)

        self.add_nodes(features)

        infer_features = self.stgat_infer().detach()

        track_ids = [-1 for _ in range(n_new_nodes)]
        max_sim = [0 for _ in range(n_new_nodes)]
        linked_node_id = [-1 for _ in range(n_new_nodes)]

        # feature matching
        for node_id in self.G.nodes:
            if node_id >= self.n_nodes - n_new_nodes:
                continue

            node = self.G.nodes[node_id]

            sim = torch.sigmoid(torch.mm(
                infer_features[node_id].unsqueeze(0), 
                infer_features[self.n_nodes - n_new_nodes:].T).view(-1))

            ind = torch.argmax(sim)

            if sim[ind] > link_threshold and sim[ind] > max_sim[ind]:
                track_ids[ind] = node['track_id']
                max_sim[ind] = sim[ind]
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

            for j in range(self.n_nodes - n_new_nodes):
                if self.G.nodes[j]['track_id'] == track_ids[i] and j != linked_node_id[i]:
                    sim = torch.sigmoid(
                        torch.mm(infer_features[node_id].unsqueeze(0), 
                                infer_features[j].unsqueeze(0).T).view(-1))
                    if sim > link_threshold:
                        self.G.add_edge(node_id, j)

        return track_ids

    def add_nodes(self, features):
        for i, feat in enumerate(features):
            self.node_features.append(feat)
            self.G.add_node(self.n_nodes)

            # add self-loop
            self.G.add_edge(self.n_nodes, self.n_nodes)

            self.n_nodes += 1

    def stgat_infer(self):
        self.h_features.append(torch.cat(self.node_features).view(-1, 512))
        self.h_adj.append(torch.tensor(nx.to_numpy_matrix(self.G), dtype=torch.float32))

        if len(self.h_adj) > 3:
            self.h_features.pop(0)
            self.h_adj.pop(0)

        for i in range(2):
            d = self.h_features[-1].size(0) - self.h_features[i].size(0)
            self.h_features[i] = F.pad(self.h_features[i], (0, 0, 0, d))
            self.h_adj[i] = F.pad(self.h_adj[i], (0, d, 0, d))

        in_features = torch.stack(self.h_features)
        in_adj = torch.stack(self.h_adj)

        embedding = self.stgat(in_features, in_adj)

        return embedding[:, -1]