import torch
import torch.nn as nn

class StructuralAttentionLayer(nn.Module):
    # base on the equation in TGAT, https://arxiv.org/abs/2002.07962
    def __init__(self, in_dim, out_dim, n_heads, dropout=0.1, alpha=0.2):
        super(StructuralAttentionLayer, self).__init__()
        
        self.dropout = dropout

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads

        self.alpha = alpha

        self.W_query = nn.Parameter(torch.empty(size=(in_dim, out_dim * n_heads)))
        nn.init.xavier_uniform_(self.W_query.data, gain=1.414)
        self.W_key = nn.Parameter(torch.empty(size=(in_dim, out_dim * n_heads)))
        nn.init.xavier_uniform_(self.W_key.data, gain=1.414)
        self.W_value = nn.Parameter(torch.empty(size=(in_dim, out_dim * n_heads)))
        nn.init.xavier_uniform_(self.W_value.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.norm1 = nn.LayerNorm(n_heads * out_dim)
        self.ffn = nn.Linear(n_heads * out_dim + in_dim, n_heads * out_dim, bias=False)
        self.norm2 = nn.LayerNorm(n_heads * out_dim)

    def forward(self, features, adj):
        # features.shape:   (B, W, N, in_dim)
        # adj.shape:        (B, W, N, N)
        # B might be None

        w = features.size(-3)
        n = features.size(-2)
        in_dim = features.size(-1)

        has_batches = False

        if len(features.size()) > 3:
            has_batches = True
            features = torch.cat(torch.split(features, 1, dim=0), dim=1).view(-1, n, in_dim)
            adj = torch.cat(torch.split(adj, 1, dim=0), dim=1).view(-1, n, n)
            # turn features shape to (B*W, N, in_dim)
            # turn adj shape to (B*W, N, N)
            

        t_query = torch.matmul(features, self.W_query) # (W, N, out_dim * n_heads)
        t_key = torch.matmul(features, self.W_key)     # (W, N, out_dim * n_heads)
        t_value = torch.matmul(features, self.W_value) # (W, N, out_dim * n_heads)

        t_query = torch.stack(torch.split(t_query, self.out_dim, dim=-1), dim=-3)  # (W, n_heads, N, out_dim)
        t_key = torch.stack(torch.split(t_key, self.out_dim, dim=-1), dim=-3)      # (W, n_heads, N, out_dim)
        t_value = torch.stack(torch.split(t_value, self.out_dim, dim=-1), dim=-3)  # (W, n_heads, N, out_dim)

        e = self.leakyrelu(torch.matmul(t_query, t_key.transpose(-1, -2))) # (W, n_heads, N, N)

        mask = adj.view(adj.size(0), 1, n, n).repeat([1, self.n_heads, 1, 1])

        zero_vec = -9e15*torch.ones_like(e)

        attn = torch.where(mask > 0, e, zero_vec)
        attn = torch.softmax(attn, dim=-1) # (W, n_heads, N, N)
        attn = torch.dropout(attn, self.dropout, train=self.training)

        h_prime = torch.matmul(attn, t_value) # (W, n_heads, N, out_dim)

        output = torch.cat(
            torch.split(h_prime, 1, dim=-3), dim=-1
        ).view(features.size(0), n, -1) # (W, N, n_heads * out_dim)

        output = self.norm1(output)
        output = torch.dropout(output, self.dropout, train=self.training)
        output = self.ffn(torch.cat([output, features], dim=-1))
        output = self.norm2(output)

        output += features

        if has_batches:
            output = torch.stack(torch.split(output, w, dim=0), dim=0)

        return output


class TemporalAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, n_time_steps=3, dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads

        self.n_time_steps = n_time_steps
        self.dropout = dropout

        self.W_query = nn.Parameter(torch.empty(size=(in_dim, out_dim * n_heads)))
        nn.init.xavier_uniform_(self.W_query.data, gain=1.414)
        self.W_key = nn.Parameter(torch.empty(size=(in_dim, out_dim * n_heads)))
        nn.init.xavier_uniform_(self.W_key.data, gain=1.414)
        self.W_value = nn.Parameter(torch.empty(size=(in_dim, out_dim * n_heads)))
        nn.init.xavier_uniform_(self.W_value.data, gain=1.414)


        self.norm1 = nn.LayerNorm(n_heads * out_dim)
        self.ffn = nn.Linear(n_heads * out_dim + in_dim, n_heads * out_dim, bias=False)
        self.norm2 = nn.LayerNorm(n_heads * out_dim)

    def forward(self, x):
        # x.shape: (B, N, W, in_dim), B might be None

        t_query = torch.matmul(x, self.W_query) # (B, N, W, out_dim * n_heads)
        t_key = torch.matmul(x, self.W_key)     # (B, N, W, out_dim * n_heads)
        t_value = torch.matmul(x, self.W_value) # (B, N, W, out_dim * n_heads)

        t_query = torch.cat(torch.split(t_query, self.out_dim, dim=-1), dim=-3)  # (B, n_heads * N, W, out_dim)
        t_key = torch.cat(torch.split(t_key, self.out_dim, dim=-1), dim=-3)      # (B, n_heads * N, W, out_dim)
        t_value = torch.cat(torch.split(t_value, self.out_dim, dim=-1), dim=-3)  # (B, n_heads * N, W, out_dim)
        
        e = torch.matmul(t_query, t_key.transpose(-1, -2)) / (self.n_time_steps ** 0.5) # (B, n_heads * N, W, W)

        mask = torch.tril(torch.ones_like(e), diagonal=0)
        zero_vec = -9e15*torch.ones_like(e)

        attn = torch.where(mask > 0, e, zero_vec)
        attn = torch.softmax(attn, dim=-1) # (B, n_heads * N, W, W)

        out = torch.matmul(attn, t_value) # (B, n_heads * N, W, out_dim)

        out = torch.cat(torch.chunk(out, self.n_heads, dim=-3), dim=-1) # (B, N, W, out_dim * n_heads)

        out = self.norm1(out)
        out = torch.dropout(out, self.dropout, train=self.training)
        out = self.ffn(torch.cat([out, x], dim=-1)) # (N, W, out_dim * n_heads)
        out = self.norm2(out)

        return out