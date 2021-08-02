import torch
from mc_mot.node import Node

from reid.model.losses import cosine_similarity

class MultipleObjectTracker():
    def __init__(self, feature_extractor, link_threshold=0.95):
        self.feature_extractor = feature_extractor
        self.link_threshold = link_threshold

        self.nodes = []
        self.n_track_id = 0

    def __call__(self, x):
        features = self.feature_extractor(x)
        track_ids = -1 * torch.ones(features.size()[0])

        # TODO: remake feature matching
        for node in self.nodes:
            cos_sim = cosine_similarity(node.feature.unsqueeze(0), features).view(-1)
            track_ids[cos_sim > self.link_threshold] = node.track_id

        for i in range(len(track_ids)):
            if track_ids[i] == -1:
                track_ids[i] = self.n_track_id
                self.n_track_id += 1

        for i, feat in enumerate(features):
            self.nodes.append(Node(node_id=len(self.nodes), feature=feat, track_id=int(track_ids[i])))

        return track_ids