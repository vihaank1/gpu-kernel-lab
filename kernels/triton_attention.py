import torch


def make_fused():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(Q, K, V, O, M: tl.constexpr, N: tl.constexpr,
               D: tl.constexpr, DV: tl.constexpr, SCALE: tl.constexpr,
               CAUSAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BD: tl.constexpr, BV: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        features = tl.arange(0, BD)
        vf = tl.arange(0, BV)
        q = tl.load(Q + rows[:, None]*D + features[None, :],
                    (rows[:, None] < M) & (features[None, :] < D), other=0).to(tl.float32)
        m = tl.full((BM,), -float('inf'), tl.float32)
        l = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, BV), tl.float32)
        for tile in range(tl.cdiv(N, BN)):
            keys = tile * BN + tl.arange(0, BN)
            k = tl.load(K + keys[None, :]*D + features[:, None],
                        (keys[None, :] < N) & (features[:, None] < D), other=0).to(tl.float32)
            v = tl.load(V + keys[:, None]*DV + vf[None, :],
                        (keys[:, None] < N) & (vf[None, :] < DV), other=0).to(tl.float32)
            scores = tl.dot(q, k, input_precision='ieee') * SCALE
            allowed = keys[None, :] < N
            if CAUSAL:
                allowed = allowed & (keys[None, :] <= rows[:, None])
            scores = tl.where(allowed, scores, -float('inf'))
            new_m = tl.maximum(m, tl.max(scores, axis=1))
            alpha = tl.exp(m - new_m)
            p = tl.exp(scores - new_m[:, None])
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision='ieee')
            l = l * alpha + tl.sum(p, axis=1)
            m = new_m
        tl.store(O + rows[:, None]*DV + vf[None, :], acc / l[:, None],
                 (rows[:, None] < M) & (vf[None, :] < DV))

    def fused(q, k, v, causal=False):
        assert q.ndim == k.ndim == v.ndim == 2
        assert q.device == k.device == v.device
        assert q.dtype == k.dtype == v.dtype == torch.float32
        assert all(t.is_contiguous() and not t.requires_grad for t in (q, k, v))
        m, d = q.shape
        n, dv = v.shape
        assert k.shape == (n, d) and min(m, n, d, dv) > 0
        assert max(d, dv) <= 128
        assert not causal or m == n, 'This exercise supports square causal attention'
        out = torch.empty((m, dv), dtype=q.dtype, device=q.device)
        kernel[(triton.cdiv(m, 16),)](q, k, v, out, m, n, d, dv, d**-.5,
                                    causal, 16, 32, max(16, triton.next_power_of_2(d)),
                                    max(16, triton.next_power_of_2(dv)), num_warps=4)
        return out
    return fused
