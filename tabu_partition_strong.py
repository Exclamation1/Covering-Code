from itertools import combinations
import random, argparse, textwrap

def hd(a,b): return (a ^ b).bit_count()

def greedy_covering_code(n, r):
    def ball(center):
        yield center
        idxs = list(range(n))
        for k in range(1, r+1):
            for comb in combinations(idxs, k):
                m = center
                for i in comb: m ^= (1<<i)
                yield m
    N = 1<<n
    balls = [set(ball(v)) for v in range(N)]
    U = set(range(N))
    code = []
    while U:
        best, gain = None, -1
        for cand in list(U):
            g = len(balls[cand] & U)
            if g > gain: best, gain = cand, g
        code.append(best); U -= balls[best]
    return code

def precompute_dist_matrix(n, code):
    N, m = 1<<n, len(code)
    dist = [[0]*m for _ in range(N)]
    for x in range(N):
        row = dist[x]
        for j,c in enumerate(code):
            row[j] = hd(x,c)
    d_to_C = [min(dist[x]) for x in range(N)]
    return dist, d_to_C

# ---- cost: seminormal / strongly-seminormal with t ----
def cost_E(n, r, S, t, code, groups, dist, d_to_C, strong: bool):
    """
    E = số điểm x thuộc biên d(x,C)=r vi phạm điều kiện:
      - seminormal:    count( i : d(x, C_i) <= S-r+t ) >= 1
      - strongly:      count(...) == 1
    """
    N = 1<<n
    thr = S - r + t
    E = 0
    for x in range(N):
        if d_to_C[x] != r:
            continue
        hits = 0
        for G in groups:
            if not G: 
                continue
            dmin = min(dist[x][j] for j in G)
            if dmin <= thr:
                hits += 1
                if not strong:      # seminormal: một hit là đủ
                    break
                if hits >= 2:       # strong: >1 là vi phạm ngay
                    break
        if (not strong and hits == 0) or (strong and hits != 1):
            E += 1
    return E

def tabu_partition(n, r, S, t, code, k, strong=False, iters=3000, tabu_L=20, seed=0):
    rnd = random.Random(seed)
    m = len(code)
    dist, d_to_C = precompute_dist_matrix(n, code)

    # init: chia đều ngẫu nhiên
    perm = list(range(m)); rnd.shuffle(perm)
    groups = [[] for _ in range(k)]
    for idx, j in enumerate(perm): groups[idx % k].append(j)

    E = cost_E(n,r,S,t,code,groups,dist,d_to_C,strong)
    best_groups = [g[:] for g in groups]; best_E = E
    tabu = {}  # (j,to_group) -> expire_iter

    def move(gs,j,a,b): gs[a].remove(j); gs[b].append(j)

    for it in range(1, iters+1):
        cand = None
        # pool move ngẫu nhiên
        cap = min(3*m*k, 5000)
        pool = []
        while len(pool)<cap:
            j = rnd.randrange(m)
            a = next(i for i,g in enumerate(groups) if j in g)
            b = rnd.randrange(k)
            if b!=a: pool.append((j,a,b))

        for (j,a,b) in pool:
            if tabu.get((j,b), -1) >= it:
                continue
            move(groups,j,a,b)
            En = cost_E(n,r,S,t,code,groups,dist,d_to_C,strong)
            move(groups,j,b,a)
            # aspiration: nếu cải thiện best thì cho phép
            if (tabu.get((j,b), -1) >= it) and En >= best_E:
                continue
            if (cand is None) or (En < cand[-1]):
                cand = (j,a,b,En)
            if En < E: break

        if cand is None:
            # diversify
            j = rnd.randrange(m)
            a = next(i for i,g in enumerate(groups) if j in g)
            b = (a+1)%k
            move(groups,j,a,b)
            E = cost_E(n,r,S,t,code,groups,dist,d_to_C,strong)
            tabu[(j,b)] = it + tabu_L
        else:
            j,a,b,En = cand
            move(groups,j,a,b); E = En
            tabu[(j,b)] = it + tabu_L

        if E < best_E:
            best_E = E; best_groups = [g[:] for g in groups]
            if best_E == 0: break

    return best_groups, best_E

# ---- pretty print (HEX như trong paper) ----
def to_hex(v): return format(v, "X")
def print_groups_hex(groups, code, start_label=2, wrap=100):
    import textwrap
    for gi, idxs in enumerate(groups):
        vals = sorted((to_hex(code[j]) for j in idxs), key=lambda s:(len(s),s))
        prefix = f"C{gi+start_label}: "
        body = ", ".join(vals) + ";"
        out = prefix + body
        if wrap and len(out)>wrap:
            wrapped = textwrap.fill(body, width=wrap,
                                    subsequent_indent=" " * len(prefix))
            out = prefix + wrapped[len(prefix):]
        print(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--t", type=int, default=0, help="(k,t)-subnorm parameter (default 0)")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--strong", action="store_true", help="enforce strongly seminormal (unique hit)")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--tabuL", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--from_hex", type=str, default="",
                    help="optional file: each line is a codeword in HEX for C")
    args = ap.parse_args()

    if args.from_hex:
        C=[]
        with open(args.from_hex) as f:
            for ln in f:
                ln=ln.strip()
                if not ln: continue
                C.append(int(ln,16))
    else:
        C = greedy_covering_code(args.n, args.r)

    groups, E = tabu_partition(args.n, args.r, args.S, args.t, C, args.k,
                               strong=args.strong, iters=args.iters,
                               tabu_L=args.tabuL, seed=args.seed)
    mode = "strong" if args.strong else "seminormal"
    print(f"|C|={len(C)}, k={args.k}, mode={mode}, S={args.S}, t={args.t}, E={E}")
    print_groups_hex(groups, C, start_label=2, wrap=100)
