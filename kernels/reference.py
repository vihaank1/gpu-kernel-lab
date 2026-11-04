import math
import torch


def attention(q, k, v, causal=True):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        rows = torch.arange(q.shape[-2], device=q.device)
        cols = torch.arange(k.shape[-2], device=q.device)
        scores = scores.masked_fill(cols[None, :] > rows[:, None], -float('inf'))
    return scores.softmax(dim=-1) @ v


@torch.no_grad()
def manual_backward(q, k, v, upstream):
    scale = 1 / math.sqrt(q.shape[-1])
    rows = torch.arange(q.shape[-2], device=q.device)
    cols = torch.arange(k.shape[-2], device=q.device)
    allowed = cols[None, :] <= rows[:, None]
    scores = (q @ k.transpose(-2, -1)) * scale
    p = scores.masked_fill(~allowed, -float('inf')).softmax(dim=-1)
    dv = p.transpose(-2, -1) @ upstream
    dp = upstream @ v.transpose(-2, -1)
    correction = (dp * p).sum(dim=-1, keepdim=True)
    ds = (p * (dp - correction)).masked_fill(~allowed, 0.0)
    dq = (ds @ k) * scale
    dk = (ds.transpose(-2, -1) @ q) * scale
    return dq, dk, dv
