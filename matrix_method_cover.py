import argparse, math, random
from itertools import combinations

# ---------- bit utils ----------
def popcount(x: int) -> int:
    return x.bit_count()

def bits_to_hex(bits):  # bits: list[int] MSB-first
    # pad to multiple of 4
    pad = (-len(bits)) % 4
    bits = [0]*pad + bits
    out = []
    for i in range(0, len(bits), 4):
        v = (bits[i]<<3)|(bits[i+1]<<2)|(bits[i+2]<<1)|bits[i+3]
        out.append("0123456789ABCDEF"[v])
    # strip leading zeros but keep at least one digit
    s = "".join(out).lstrip("0")
    return s if s else "0"

def col_to_bits(col: int, m: int):  # integer -> list[bit] MSB-first length m
    return [(col >> (m-1-i)) & 1 for i in range(m)]

# ---------- enumerate ≤ r sums of a column set ----------
def all_sums_up_to_r(cols, r):
    """Return set of XOR sums of choosing 1..r distinct columns (non-empty)."""
    S = set()
    for k in range(1, r+1):
        for subset in combinations(cols, k):
            x = 0
            for c in subset: x ^= c
            S.add(x)
    return S

# ---------- verify cover ----------
def verify_r_cover(m: int, H_cols, r: int) -> bool:
    # H_cols: list[int] each m-bit column (includes e1..em and columns of M)
    reachable = {0}  # we can ignore 0 (cost 0), but keep for completeness
    # build all sums up to r
    sums = all_sums_up_to_r(H_cols, r)
    reachable |= sums
    return len(reachable) >= (1 << m)  # should be all syndromes

# ---------- greedy constructor ----------
def greedy_matrix_method(m: int, r: int, max_k: int, seed: int = 0):
    """
    Build M as a multiset of k columns (length-m integers in [0,2^m-1], excluding 0 and e_i),
    so that {e_i} ∪ M r-covers F2^m. Try k growing until cover satisfied or max_k reached.
    Strategy: greedy set cover on syndrome universe with 'r-sums'.
    """
    rnd = random.Random(seed)

    # identity columns e_i:
    base_cols = [(1 << (m-1-i)) for i in range(m)]  # MSB=bit0; e1 has MSB=1...
    U = set(range(1 << m))  # universe of syndromes (including 0)
    U.remove(0)             # we only need to cover non-zero
    # Precompute coverage of candidate singletons and pairs/triples efficiently:
    # For greedy, we add one new column c at a time and measure the new coverage
    def current_cover(cols_set):
        # compute sums of size ≤ r from base_cols ∪ cols_set
        H = base_cols + list(cols_set)
        cov = set()
        for k in range(1, r+1):
            for subset in combinations(H, k):
                x = 0
                for c in subset: x ^= c
                cov.add(x)
        return cov

    # Candidate pool: all non-zero m-bit vectors except the m identity vectors
    forb = set(base_cols) | {0}
    pool = [x for x in range(1<<m) if x not in forb]

    for k in range(0, max_k+1):
        # simple randomized restarts: try a few runs and take the best
        best = None
        best_cov_size = -1
        trials = max(1, 4 if k>0 else 1)
        for _ in range(trials):
            cols = []
            remain = set(U)
            # quick pre-calc cover of base only
            base_cov = current_cover([])
            remain -= base_cov
            if not remain:
                best = cols; best_cov_size = len(U)
                break
            # greedy pick k columns
            cand_pool = pool[:]
            rnd.shuffle(cand_pool)
            for _kk in range(k):
                # choose c maximizing new coverage
                pick, gain_best = None, -1
                for c in cand_pool:
                    cov = current_cover(cols + [c])
                    gain = len((cov | base_cov) & U)  # total cover achievable
                    if gain > gain_best:
                        gain_best = gain; pick = c
                    # small speedup: early break if perfect
                    if gain == len(U):
                        break
                if pick is None:
                    break
                cols.append(pick)
                base_cov = current_cover(cols)
                remain = U - base_cov
                if not remain:
                    break
            cov_all = current_cover(cols)
            if len(cov_all & U) > best_cov_size:
                best = cols
                best_cov_size = len(cov_all & U)
        if best is None:
            continue
        # verify
        H_cols = base_cols + best
        if verify_r_cover(m, H_cols, r):
            return best  # columns of M
    return None

# ---------- build H and export M-hex rows (Table-2 style) ----------
def build_H_I_M(m: int, M_cols: list[int]):
    # H = [I_m | M], with columns in MSB-first convention
    # For export like Table 2: we print M as 'm rows' of hex (each row length = k bits)
    k = len(M_cols)
    # make row-wise bits for M
    M_rows = []
    for i in range(m):  # row i = i-th bit of each col (MSB-first)
        row_bits = []
        for c in M_cols:
            bit = (c >> (m-1-i)) & 1
            row_bits.append(bit)
        M_rows.append(row_bits)
    return M_rows  # list of length m, each row is list of k bits

def export_M_hex_rows(M_rows):
    # turn each row (list of bits MSB-first) into hex string
    return [bits_to_hex(row) for row in M_rows]

# ---------- covering radius check for the linear code ----------
def ham_dist(a: int, b: int) -> int:
    return popcount(a ^ b)

def generator_from_parity_I_M(m: int, M_rows):
    """
    Build a generator G of the nullspace of H = [I_m | M].
    n = m + k, k = width(M).
    Return list of generator row bitmasks (n-bit, MSB-first -> bitmask MSB at left).
    """
    m_ = m
    k = len(M_rows[0]) if m_>0 else 0
    n = m_ + k
    # Standard form for H = [I | M] over GF(2):
    # Nullspace has dimension k; one basis is rows: [M^T | I_k]
    # Proof: H * [M^T | I] ^T = I*M^T + M*I = M^T + M^T = 0.
    # Build rows of length n, MSB-first -> convert to mask.
    G_rows = []
    # M^T rows:
    for j in range(k):
        row = []
        # first m bits: column j of M (MSB-first)
        for i in range(m):
            row.append(M_rows[i][j])
        # then k bits: identity at position j
        for t in range(k):
            row.append(1 if t==j else 0)
        # to mask:
        mask = 0
        for b in row:
            mask = (mask<<1) | b
        G_rows.append(mask)
    return G_rows, n

def all_codewords_from_G(G_rows, n):
    k = len(G_rows)
    code = []
    for u in range(1<<k):
        cw = 0
        for i in range(k):
            if (u >> (k-1-i)) & 1:
                cw ^= G_rows[i]
        code.append(cw)
    return code

def covering_radius(code, n):
    # exact covering radius (slow for n>16)
    N = 1<<n
    worst = 0
    for x in range(N):
        d = min(ham_dist(x, c) for c in code)
        if d > worst: worst = d
    return worst

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--maxk", type=int, default=12, help="Max parity columns to try")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", type=str, default="", help="Write M-hex rows to this file")
    args = ap.parse_args()

    n, r = args.n, args.r
    # search over m (parity rows); k = n - m
    found = None
    best = None
    for m in range(1, n):  # at least 1 parity row; at most n-1
        k = n - m
        if k > args.maxk:
            continue
        M_cols = greedy_matrix_method(m, r, max_k=k, seed=args.seed)
        if M_cols is None or len(M_cols) > k:
            continue
        if len(M_cols) < k:
            # pad with duplicates of identity-combinations (rare); or just skip to keep H=[I|M] exactly n
            # Here we keep exact n by adding arbitrary extra columns (won't hurt cover)
            pad = k - len(M_cols)
            M_cols = M_cols + [1]*(pad)
        # Build H and code, verify covering radius
        M_rows = build_H_I_M(m, M_cols)
        G_rows, n_chk = generator_from_parity_I_M(m, M_rows)
        code = all_codewords_from_G(G_rows, n_chk)
        # Warning: exact radius explodes beyond n=16. For n<=16, compute exactly:
        rad = covering_radius(code, n_chk) if n_chk <= 16 else None
        found = (m, k, M_rows, rad, len(code))
        best = found
        # prefer smaller |C| = 2^k ⇒ smaller k ⇒ bigger m; we are scanning m↑ so first hit is best k
        break

    if best is None:
        print(f"[FAIL] No M found up to maxk={args.maxk} for (n={n}, r={r})")
        return

    m, k, M_rows, rad, size = best
    M_hex = export_M_hex_rows(M_rows)
    print(f"[OK] (n={n}, r={r})  linear code [n={n}, k={k}]  |C|=2^{k}={size}")
    print(f"     m={m} parity rows (H=[I_m | M])")
    if rad is not None:
        print(f"     exact covering radius = {rad}")
    print("     M-hex rows (Table-2 style):")
    for h in M_hex:
        print("       ", h)
    if args.export:
        with open(args.export, "w") as f:
            for h in M_hex:
                f.write(h+"\n")
        print(f"     -> wrote hex rows to {args.export}")

if __name__ == "__main__":
    main()
