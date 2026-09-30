"""Plasticity write-back as a GEMM, not a sort.

THE ALGEBRA. Let x_t be the 0/1 winner indicator at round t. Training's update is

    count = SUM_t  x_{t-1} x_t^T                                          (*)

a sum of T RANK-1 OUTER PRODUCTS. Two things follow immediately.

  1. An event carries 2k numbers. Writing the k*k cross product writes the same
     information k/2 times over. That redundancy is forced by (*), not an
     artifact of the implementation -- no amount of sorting cleverness recovers
     it, because the work was created before the sort was reached.

  2. Restricted to the rows and columns a window of T rounds actually touches,
     (*) IS A MATRIX PRODUCT:

         count[i,j] = ( R^T C )[i,j],    R in {0,1}^{T x r},  C in {0,1}^{T x c}

     with R[t,i] = 1[i in W_{t-1}] and C[t,j] = 1[j in W_t]. Both are thin
     (T ~ 8, r,c ~ k). So the consolidation that the append-and-sort design
     spends its whole budget on is a batched GEMM of 0/1 matrices -- the single
     operation a GPU is best at. No keys, no radix sort, no segment reduce.

WHAT THE STORE BECOMES. One block per (brain, item): (rows, cols, counts) with
counts a small dense r x c matrix of uint8. And because count is ADDITIVE across
blocks -- (*) is a sum -- blocks never have to be merged with each other. Two
items whose blocks collide in a cell are handled by the reader adding them,
which is exactly what (*) says. There is no global store, so there is nothing
to re-sort.

Counts are exact integers: the GEMM's operands are 0/1 and T <= 255, so fp32
accumulation is exact and the result fits in uint8.

This measures the new path against the committed append+compact one, in the
same process, on the same winner streams.
"""
import os
import time

os.environ.setdefault(
    "CUDA_HOME",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")

import torch                                                # noqa: E402

DEV = 'cuda'


# ---------------------------------------------------------------- baseline
def event_keys(prev, new, n):
    B, k = prev.shape
    off = (torch.arange(B, device=DEV, dtype=torch.int64)
           * (n * n)).view(B, 1, 1)
    return (off + prev.view(B, k, 1) * n + new.view(B, 1, k)).reshape(-1)


def compact(keys, counts, buf):
    if buf.numel() == 0:
        return keys, counts
    allk = torch.cat([keys, buf])
    allv = torch.cat([counts, torch.ones_like(buf, dtype=torch.int32)])
    order = torch.argsort(allk)
    uk, inv, _ = torch.unique_consecutive(allk[order], return_inverse=True,
                                          return_counts=True)
    uv = torch.zeros(uk.numel(), dtype=torch.int32, device=DEV)
    uv.scatter_add_(0, inv, allv[order])
    return uk, uv


# ------------------------------------------------------------------- new
def local_index(idx):
    """Map raw ids to per-brain local ids. `idx` [B, L] -> loc [B, L], vals.

    A sort of L = T*k ~ 400 elements per brain; the unique set is read off the
    consecutive-difference mask and the local id is its cumsum. Everything here
    is [B, 400], which is nothing.
    """
    B, L = idx.shape
    s, order = torch.sort(idx, dim=1)
    new = torch.ones_like(s, dtype=torch.bool)
    new[:, 1:] = s[:, 1:] != s[:, :-1]
    loc_sorted = torch.cumsum(new, dim=1) - 1
    loc = torch.empty_like(loc_sorted)
    loc.scatter_(1, order, loc_sorted)
    nuniq = new.sum(1)
    W = int(nuniq.max())
    vals = torch.full((B, W), -1, dtype=torch.int64, device=DEV)
    # duplicates scatter the same value, so the race is benign
    vals.scatter_(1, loc_sorted, s)
    return loc, vals, nuniq, W


def block_from_window(P, Q):
    """P, Q are [B, T, k] winner ids. Returns (rows, cols, counts) per brain.

    counts[b] = R[b]^T C[b] exactly: the (i,j) entry is the number of rounds t
    with i in P[b,t] and j in Q[b,t], which is the definition of count[i,j].
    """
    B, T, k = P.shape
    rloc, rows, _, R = local_index(P.reshape(B, T * k))
    cloc, cols, _, C = local_index(Q.reshape(B, T * k))
    Rind = torch.zeros(B, T, R, device=DEV)
    Rind.scatter_(2, rloc.view(B, T, k), 1.0)
    Cind = torch.zeros(B, T, C, device=DEV)
    Cind.scatter_(2, cloc.view(B, T, k), 1.0)
    counts = torch.bmm(Rind.transpose(1, 2), Cind)      # [B, R, C]
    return rows, cols, counts.to(torch.uint8)


# ------------------------------------------------------------- exactness
def check_exact():
    print("=== exactness: GEMM block == a dict of counts ===")
    B, k, n, T = 3, 6, 50, 5
    g = torch.Generator(device=DEV).manual_seed(0)
    P = torch.randint(0, n, (B, T, k), device=DEV, generator=g,
                      dtype=torch.int64)
    Q = torch.randint(0, n, (B, T, k), device=DEV, generator=g,
                      dtype=torch.int64)
    rows, cols, counts = block_from_window(P, Q)
    from collections import defaultdict
    ref = defaultdict(int)
    Pc, Qc = P.cpu().numpy(), Q.cpu().numpy()
    for b in range(B):
        for t in range(T):
            for i in set(Pc[b, t].tolist()):
                for j in set(Qc[b, t].tolist()):
                    ref[(b, i, j)] += 1
    got = {}
    rr, cc, kk = rows.cpu().numpy(), cols.cpu().numpy(), counts.cpu().numpy()
    for b in range(B):
        for a in range(rr.shape[1]):
            if rr[b, a] < 0:
                continue
            for c in range(cc.shape[1]):
                if cc[b, c] < 0 or kk[b, a, c] == 0:
                    continue
                got[(b, int(rr[b, a]), int(cc[b, c]))] = int(kk[b, a, c])
    ok = got == dict(ref)
    print(f"  {len(got)} nonzero cells, dict has {len(ref)} -> identical: {ok}")
    return ok


# ------------------------------------------------------------- benchmarks
def streams(n, k, B, M, T, churn=0.1, seed=1):
    """The SAME winner streams both paths consume: stable cores with churn."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    cores = torch.randint(0, n, (M, B, k), device=DEV, generator=g,
                          dtype=torch.int64)
    nj = max(1, int(k * churn))
    out = []
    for a in range(M):
        P = torch.empty(B, T, k, device=DEV, dtype=torch.int64)
        Q = torch.empty(B, T, k, device=DEV, dtype=torch.int64)
        prev = cores[a]
        for t in range(T):
            new = cores[a].clone()
            jit = torch.randint(0, k, (B, nj), device=DEV, generator=g)
            new.scatter_(1, jit, torch.randint(0, n, (B, nj), device=DEV,
                                               generator=g, dtype=torch.int64))
            P[:, t], Q[:, t] = prev, new
            prev = new
        out.append((P, Q))
    return out


def run_baseline(st, n):
    keys = torch.zeros(0, dtype=torch.int64, device=DEV)
    counts = torch.zeros(0, dtype=torch.int32, device=DEV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for P, Q in st:
        T = P.shape[1]
        buf = torch.cat([event_keys(P[:, t], Q[:, t], n) for t in range(T)])
        keys, counts = compact(keys, counts, buf)
    torch.cuda.synchronize()
    mb = (keys.numel() * 8 + counts.numel() * 4) / 1e6
    return time.perf_counter() - t0, int(keys.numel()), mb


def run_gemm(st):
    blocks = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for P, Q in st:
        blocks.append(block_from_window(P, Q))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    cells = sum(int((c > 0).sum()) for _, _, c in blocks)
    mb = sum(r.numel() * 4 + c.numel() * 4 + k.numel()
             for r, c, k in blocks) / 1e6
    return dt, cells, mb, blocks


def main():
    if not check_exact():
        raise SystemExit("exactness failed -- stop")

    print("\n=== write-back cost: append+sort vs GEMM ===")
    print("  same winner streams, same process. M items x T rounds.\n")
    hdr = (f"{'n':>6} {'k':>4} {'B':>4} {'M':>4} {'T':>3} | "
           f"{'sort s':>7} {'ms/br/rd':>9} {'MB':>7} | "
           f"{'gemm s':>7} {'ms/br/rd':>9} {'MB':>7} | {'speedup':>8}")
    print(hdr)
    print("-" * len(hdr))
    # B=256 is EXCLUDED from the baseline: Amendment 2 already measured it
    # thrashing (480 MB store, ~1.1 s per compaction). Re-running a known
    # pathological config is not a measurement, it is a wait.
    for n, k, M, T, B in ((20000, 50, 32, 8, 64),
                          (20000, 70, 32, 8, 64),
                          (20000, 70, 64, 8, 64)):
        st = streams(n, k, B, M, T)
        rounds = M * T * B
        tb, _, mb_b = run_baseline(st, n)
        torch.cuda.empty_cache()
        tg, _, mb_g, _ = run_gemm(st)
        print(f"{n:>6} {k:>4} {B:>4} {M:>4} {T:>3} | "
              f"{tb:>7.2f} {1000*tb/rounds:>9.5f} {mb_b:>7.1f} | "
              f"{tg:>7.3f} {1000*tg/rounds:>9.5f} {mb_g:>7.1f} | "
              f"{tb/tg:>7.1f}x")
        del st
        torch.cuda.empty_cache()

    print("\n=== GEMM path alone, at the batch the baseline cannot reach ===")
    for n, k, M, T, B in ((20000, 70, 32, 8, 256), (20000, 70, 32, 8, 1024)):
        st = streams(n, k, B, M, T)
        tg, _, mb_g, _ = run_gemm(st)
        print(f"{n:>6} {k:>4} {B:>4} {M:>4} {T:>3} | "
              f"gemm {tg:>7.3f} s  {1000*tg/(M*T*B):>9.5f} ms/br/rd "
              f"{mb_g:>7.1f} MB")
        del st
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
