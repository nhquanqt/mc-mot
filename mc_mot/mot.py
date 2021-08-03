import torch
from mc_mot.node import Node

from reid.model.losses import cosine_similarity

import networkx

class MultipleObjectTracker():
    def __init__(self, feature_extractor, link_threshold=0.95):
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        self.link_threshold = link_threshold

        self.nodes = []
        self.n_track_id = 0
        self.n_nodes = 0

        self.G = networkx.Graph()

    def __call__(self, x):
        features = self.feature_extractor(x).detach()
        track_ids = -1 * torch.ones(features.size()[0])

        # TODO: remake feature matching
        for node in self.nodes:
            cos_sim = cosine_similarity(node.feature.unsqueeze(0), features).view(-1)
            ind = torch.argmax(cos_sim)
            if cos_sim[ind] > self.link_threshold:
                track_ids[ind] = node.track_id

        for i in range(len(track_ids)):
            if track_ids[i] == -1:
                track_ids[i] = self.n_track_id
                self.n_track_id += 1

        # just save nodes in 1 frame before this time step, this will be removed in the future
        self.nodes = []

        for i, feat in enumerate(features):
            self.nodes.append(Node(node_id=self.n_nodes, feature=feat, track_id=int(track_ids[i])))

            self.G.add_nodes_from([(self.n_nodes, {
                'feature': feat,
                'track_id': track_ids[i]
            })])

            self.n_nodes += 1

        return track_ids