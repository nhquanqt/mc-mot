import torch

from reid.model.losses import cosine_similarity

import networkx

class MultipleObjectTracker():
    def __init__(self, feature_extractor, link_threshold=0.96):
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        self.link_threshold = link_threshold

        self.n_track_id = 0
        self.n_nodes = 0

        self.G = networkx.Graph()

    def __call__(self, x):
        features = self.feature_extractor(x).detach()
        n_features = features.size()[0]

        track_ids = [-1 for _ in range(n_features)]
        max_sim = [0 for _ in range(n_features)]
        linked_node_id = [-1 for _ in range(n_features)]

        # feature matching
        for node_id in self.G.nodes:
            node = self.G.nodes[node_id]
            cos_sim = cosine_similarity(node['feature'].unsqueeze(0), features).view(-1)
            ind = torch.argmax(cos_sim)
            if cos_sim[ind] > self.link_threshold and cos_sim[ind] > max_sim[ind]:
                track_ids[ind] = node['track_id']
                max_sim[ind] = cos_sim[ind]
                linked_node_id[ind] = node_id

        # remove same track-id situations
        for i in range(n_features):
            if track_ids[i] == -1:
                continue

            for j in range(i+1, n_features):
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

        # add new node to graph
        for i, feat in enumerate(features):
            self.G.add_nodes_from([(self.n_nodes, {
                'feature': feat,
                'track_id': track_ids[i]
            })])

            if linked_node_id[i] != -1:
                self.G.add_edge(self.n_nodes, linked_node_id[i])

            self.n_nodes += 1

        return track_ids