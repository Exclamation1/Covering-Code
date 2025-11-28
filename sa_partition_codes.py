import argparse, math, random
from itertools import combinations

# ------------- utils -------------
def hd(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def build_greedy_cover(n: int, r: int):
    """Simple greedy covering (not optimal) to get a valid C for demo."""
    def ball(center):
        yield center
        idxs = list(range(n))
        for k in range(1, r+1):
            for comb in combinations(idxs, k):
                m = center
                for i in comb: m ^= (1 << i)
                yield m
    N = 1 << n
    balls = [set(ball(v)) for v in range(N)]
    U = set(range(N))
    C = []
    while U:
        best, gain = None, -1
        for cand in list(U):
            g = len(balls[cand] & U)
            if g > gain:
                best, gain = cand, g
        C.append(best)
        U -= balls[best]
    return C  # list[int]

def precompute(n: int, C: list[int]):
    """dist[x][j] = d(x, C[j]); d_to_C[x] = min_j dist[x][j]; boundary B = {x: d(x,C)=r}"""
    N, m = 1 << n, len(C)
    dist = [[0]*m for _ in range(N)]
    for x in range(N):
        row = dist[x]
        for j,c in enumerate(C):
            row[j] = hd(x, c)
    d_to_C = [min(dist[x]) for x in range(N)]
    return dist, d_to_C

# ------------- cost & incremental helpers -------------
def init_groups(m: int, k: int, rnd: random.Random):
    """balanced random split of indices 0..m-1 into k groups"""
    idx = list(range(m))
    rnd.shuffle(idx)
    G = [[] for _ in range(k)]
    for i, j in enumerate(idx):
        G[i % k].append(j)
    return G

def compute_group_mins_for_x(x: int, groups, dist_row, k: int) -> list[int]:
    """Return [d(x, C_i)]_i ; empty group => +INF"""
    INF = 10**9
    mins = [INF]*k
    for i, Gi in enumerate(groups):
        if not Gi: continue
        mins[i] = min(dist_row[j] for j in Gi)
    return mins

def is_violation(mins, thr, strong=False):
    """ seminormal: need >=1 hit ; strong: need exactly 1 hit """
    hits = sum(1 for d in mins if d <= thr)
    return (hits == 0) if (not strong) else (hits != 1)

def compute_cost_full(B, groups, dist, thr, strong=False):
    """Full E over boundary set B (no increment)"""
    E = 0
    k = len(groups)
    for x in B:
        mins = compute_group_mins_for_x(x, groups, dist[x], k)
        if is_violation(mins, thr, strong):
            E += 1
    return E

def delta_cost_move(B, groups, dist, thr, strong, j, g_from, g_to):
    """
    Compute E' for a tentative move (j: g_from -> g_to) by rescanning only groups g_from,g_to.
    For each x in B, recompute mins for those two groups and re-evaluate violation.
    This is simple and fine for n <= ~12..14 in demos.
    """
    k = len(groups)
    # precompute current mins per x (only once for affected groups)
    # but we don't keep global mins; so compute "before" and "after" on the fly for affected groups
    E_new = 0
    for x in B:
        row = dist[x]

        # current mins per group (only two groups explicitly; others via quick min)
        # compute current mins for all groups cheaply:
        # For small instances, recompute all k mins for clarity
        mins_before = [10**9]*k
        for i, Gi in enumerate(groups):
            if not Gi: continue
            mins_before[i] = min(row[t] for t in Gi)

        # After move: simulate
        after_groups_from = [t for t in groups[g_from] if t != j]
        after_groups_to   = groups[g_to] + [j]

        mins_after = mins_before[:]  # copy then update only two entries
        mins_after[g_from] = min((row[t] for t in after_groups_from), default=10**9)
        mins_after[g_to]   = min((row[t] for t in after_groups_to), default=10**9)

        # Evaluate violation on mins_after
        if is_violation(mins_after, thr, strong):
            E_new += 1
    return E_new

# ------------- SA core -------------
def simulated_annealing(n, r, S, t, C, k, iters, T0, alpha, seed, strong):
    rnd = random.Random(seed)
    m = len(C)
    dist, d_to_C = precompute(n, C)
    thr = S - r + t

    # Boundary set only
    B = [x for x in range(1<<n) if d_to_C[x] == r]

    groups = init_groups(m, k, rnd)
    E = compute_cost_full(B, groups, dist, thr, strong)
    best_E, best_groups = E, [g[:] for g in groups]

    T = T0
    for it in range(1, iters+1):
        # propose: pick a random codeword j and move to a different group
        g_from = rnd.randrange(k)
        while not groups[g_from]:
            g_from = rnd.randrange(k)
        j = rnd.choice(groups[g_from])
        g_to = rnd.randrange(k)
        while g_to == g_from:
            g_to = rnd.randrange(k)

        E_new = delta_cost_move(B, groups, dist, thr, strong, j, g_from, g_to)
        dE = E_new - E
        accept = (dE <= 0) or (rnd.random() < math.exp(-dE / max(T, 1e-12)))
        if accept:
            # apply move
            groups[g_from].remove(j)
            groups[g_to].append(j)
            E = E_new
            if E < best_E:
                best_E = E
                best_groups = [g[:] for g in groups]
                if best_E == 0:
                    break

        # cool down
        T *= alpha
        # restart/cooling floor for stability
        if T < 1e-6:
            T = T0

    return best_groups, best_E

# ------------- CLI & pretty print -------------
def to_hex(v: int) -> str:
    return format(v, "X")

def print_groups(groups, C, start_label=0, wrap=120):
    import textwrap
    for gi, idxs in enumerate(groups):
        vals = sorted((to_hex(C[j]) for j in idxs), key=lambda s:(len(s), s))
        prefix = f"C{gi+start_label}: "
        body = ", ".join(vals) + ";"
        line = prefix + body
        if wrap and len(line) > wrap:
            wrapped = textwrap.fill(body, width=wrap,
                                    subsequent_indent=" " * len(prefix))
            print(prefix + wrapped[len(prefix):])
        else:
            print(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--strong", action="store_true",
                    help="enforce strongly-seminormal (unique hit)")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--T0", type=float, default=2.0, help="initial temperature")
    ap.add_argument("--alpha", type=float, default=0.995, help="cooling factor")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--from_hex", type=str, default="", help="path to HEX list of codewords; else greedy")
    ap.add_argument("--label_start", type=int, default=0)
    args = ap.parse_args()

    # Build or load code C
    if args.from_hex:
        C=[]
        with open(args.from_hex) as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                C.append(int(ln, 16))
    else:
        C = build_greedy_cover(args.n, args.r)

    groups, E = simulated_annealing(
        n=args.n, r=args.r, S=args.S, t=args.t, C=C, k=args.k,
        iters=args.iters, T0=args.T0, alpha=args.alpha,
        seed=args.seed, strong=args.strong
    )

    mode = "strong" if args.strong else "semi"
    print(f"|C|={len(C)}, k={args.k}, S={args.S}, t={args.t}, mode={mode}, best_E={E}")
    print_groups(groups, C, start_label=args.label_start, wrap=120)

if __name__ == "__main__":
    main()
