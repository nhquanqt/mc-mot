import torch

def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
    
def cosine_dist(x, y):
    """
    Compute element-wise cosine distance between 'x' and 'y'
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.nn.functional.normalize(x, dim=1)
    y_norm = torch.nn.functional.normalize(y, dim=1)
    dist = torch.ones([m, n], dtype=torch.float32) - torch.matmul(x_norm, y_norm.T)
    return dist

def cosine_similarity(x, y):
    """
    Compute element-wise cosine similarity between 'x' and 'y'
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.nn.functional.normalize(x, dim=1)
    y_norm = torch.nn.functional.normalize(y, dim=1)
    dist = torch.matmul(x_norm, y_norm.T)
    return dist

def hard_example_mining(dist_mat, labels):
    """For each anchor, find the hardest positive and negative sample.
    Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
    Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    NOTE: Only consider the case in which all labels have same num of samples, 
        thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an

def global_loss(tri_loss, global_feat, labels, dist_func=euclidean_dist):
    """
    Args:
        tri_loss: a `TripletLoss` object
        global_feat: pytorch Variable, shape [N, C]
        labels: pytorch LongTensor, with shape [N]
    Returns:
        loss: pytorch Variable, with shape [1]
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    dist_mat = dist_func(global_feat, global_feat)
    dist_ap, dist_an = hard_example_mining(dist_mat, labels)
    loss = tri_loss(dist_ap, dist_an)

    return loss, dist_ap, dist_an