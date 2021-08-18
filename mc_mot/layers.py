import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    # base on the equation in TGAT, https://arxiv.org/abs/2002.07962
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W_query = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_query.data, gain=1.414)
        self.W_key = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_key.data, gain=1.414)
        self.W_value = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_value.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h.shape: (N, in_features)
        V = torch.matmul(h, self.W_value)   # V.shape: (N, out_features)
        Q = torch.matmul(h, self.W_query)   # Q.shape: (N, out_features)
        K = torch.matmul(h, self.W_key)     # K.shape: (N, out_features)

        e = self.leakyrelu(torch.matmul(Q, K.T)) # e.shape: (N, N)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, V)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class StructuralAttentionLayer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nheads, dropout=0.1, alpha=0.2):
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
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.dropout = dropout

        self.W_query = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W_query.data, gain=1.414)
        self.W_key = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W_key.data, gain=1.414)
        self.W_value = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W_value.data, gain=1.414)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim + in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, N, in_dim)

        t_query = torch.matmul(x, self.W_query) # (batch_size, N, out_dim)
        t_key = torch.matmul(x, self.W_key) # (batch_size, N, out_dim)
        t_value = torch.matmul(x, self.W_value) # (batch_size, N, out_dim)

        sqrt_d = torch.sqrt(torch.tensor(self.out_dim, dtype=torch.float32))

        e = torch.bmm(t_query, t_key.transpose(1, 2)) / sqrt_d

        mask = torch.tril(torch.ones_like(e), diagonal=0)
        zero_vec = -9e15*torch.ones_like(e)

        attn = torch.where(mask > 0, e, zero_vec)
        attn = torch.softmax(attn, dim=2) # (batch_size, N, N)

        attn = torch.dropout(attn, self.dropout, train=self.training)

        out = torch.bmm(attn, t_value) # (batch_size, N, out_dim)

        return self.ffn(torch.cat([out, x], dim=-1)) # (batch_size, N, out_dim)


class TemporalAttentionLayer(nn.Module):
    def __init__(self, n_heads, in_dim, out_dim, n_time_steps=3, dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()

        self.n_time_steps = n_time_steps
        self.pos_embedding = nn.Embedding(n_time_steps, in_dim)

        self.attention_heads = [
            SelfAttentionLayer(in_dim, out_dim, dropout=dropout) 
            for _ in range(n_heads)
        ]

        for i, attention in enumerate(self.attention_heads):
            self.add_module('attention_heads_{}'.format(i), attention)

        self.out_attn = SelfAttentionLayer(out_dim*n_heads, out_dim, dropout=dropout)
        

    def forward(self, x):
        device = x.device

        batch_size = x.size(0)

        pos_input = torch.arange(0, self.n_time_steps).view(1, -1).repeat(batch_size, 1).long().to(device)

        x += self.pos_embedding(pos_input)

        features = torch.cat([head(x) for head in self.attention_heads], dim=-1) # (batch_size, W, n_heads * out_dim)

        return self.out_attn(features) # (batch_size, W, out_dim)

if __name__=='__main__':
    model = TemporalAttentionLayer(16, 128, 8)

    x = torch.randn(100, 3, 128)

    e = model(x)
    print(e)