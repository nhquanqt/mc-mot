import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class StructuralAttentionLayer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(StructuralAttentionLayer, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.out_att(x, adj)


class SelfAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, has_mask=False):
        super(SelfAttentionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.has_mask = has_mask

        self.W_query = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        self.W_key = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        self.W_value = nn.Parameter(torch.empty(size=(in_dim, out_dim)))

    def forward(self, x):
        # x shape: (N, in_dim)

        n = x.size()[0]

        t_query = torch.mm(x, self.W_query) # (N, out_dim)
        t_key = torch.mm(x, self.W_key) # (N, out_dim)
        t_value = torch.mm(x, self.W_value) # (N, out_dim)

        sqrt_d = torch.sqrt(torch.tensor(self.out_dim, dtype=torch.float32))
        if not self.has_mask:
            # dim 0 or dim 1?
            attn = torch.softmax(torch.mm(t_query, t_key.T) / sqrt_d, dim=1) # (N, N)
        else:
            mask = torch.triu(-1e9 * torch.ones(n, n))
            attn = torch.softmax(torch.mm(t_query, t_key.T) / sqrt_d + mask, dim=1) # (N, N)

        return torch.mm(attn, t_value) # (N, out_dim)


class TemporalAttentionLayer(nn.Module):
    def __init__(self, n_heads, in_dim, out_dim):
        super(TemporalAttentionLayer, self).__init__()

        self.attention_heads = [
            SelfAttentionLayer(in_dim, out_dim, has_mask=True) 
            for _ in range(n_heads)
        ]

        self.ffn = nn.Linear(n_heads * out_dim, out_dim)

    def forward(self, x):
        features = []
        for head in self.attention_heads:
            z = head(x)
            features.append(z.T)

        features = torch.cat(features).T # (W, n_heads * out_dim)

        return self.ffn(features) # (W, out_dim)

if __name__=='__main__':
    model = TemporalAttentionLayer(8, 2048, 512)

    x = torch.randn(3, 2048)

    e = model(x)
    print(e)