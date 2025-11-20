#include <bits/stdc++.h>
using namespace std;

struct UBitset {
    vector<uint64_t> w;
    size_t nbits = 0; // number of valid bits

    UBitset() {}
    explicit UBitset(size_t nbits_) { reset(nbits_); }

    void reset(size_t nbits_) {
        nbits = nbits_;
        size_t nwords = (nbits + 63) / 64;
        w.assign(nwords, 0ULL);
    }
    inline void set_all() {
        std::fill(w.begin(), w.end(), ~0ULL);
        // mask off extra bits
        size_t extra = (w.size()*64) - nbits;
        if (extra) {
            uint64_t mask = ~0ULL >> extra; // keep lowest (64-extra) bits
            w.back() = mask;
        }
    }
    inline void clear_all() { std::fill(w.begin(), w.end(), 0ULL); }

    inline void set_bit(size_t i) {
        w[i >> 6] |= (1ULL << (i & 63));
    }
    inline bool get_bit(size_t i) const {
        return (w[i >> 6] >> (i & 63)) & 1ULL;
    }

    inline void or_assign(const UBitset& b) {
        size_t m = w.size();
        for (size_t i=0;i<m;i++) w[i] |= b.w[i];
    }
    inline void xor_assign(const UBitset& b) {
        size_t m = w.size();
        for (size_t i=0;i<m;i++) w[i] ^= b.w[i];
    }
    inline void and_assign(const UBitset& b) {
        size_t m = w.size();
        for (size_t i=0;i<m;i++) w[i] &= b.w[i];
    }
    inline bool equals(const UBitset& b) const {
        return w == b.w;
    }
    inline uint64_t popcount() const {
        uint64_t s = 0;
        for (auto x: w) s += std::popcount(x);
        return s;
    }

    // this = a | b (no alloc)
    inline void assign_or(const UBitset& a, const UBitset& b) {
        size_t m = w.size();
        for (size_t i=0;i<m;i++) w[i] = a.w[i] | b.w[i];
    }
    // this = a ^ b
    inline void assign_xor(const UBitset& a, const UBitset& b) {
        size_t m = w.size();
        for (size_t i=0;i<m;i++) w[i] = a.w[i] ^ b.w[i];
    }
    // this = a
    inline void assign_copy(const UBitset& a) {
        w = a.w; nbits = a.nbits;
    }
};

// Iterate 1-bits indices of a UBitset (fast)
struct OneBitsIter {
    const UBitset& bs;
    size_t word_idx = 0;
    uint64_t cur = 0;
    size_t base = 0;
    OneBitsIter(const UBitset& b): bs(b) {
        advance_to_next();
    }
    void advance_to_next() {
        while (word_idx < bs.w.size() && bs.w[word_idx] == 0ULL) {
            word_idx++; base += 64;
        }
        if (word_idx < bs.w.size()) cur = bs.w[word_idx];
        else cur = 0;
    }
    bool next(size_t &pos) {
        while (true) {
            if (word_idx >= bs.w.size()) return false;
            if (cur) {
                unsigned tz = std::countr_zero(cur);
                pos = base + tz;
                cur &= (cur - 1); // clear lsb1
                return true;
            } else {
                word_idx++; base += 64;
                if (word_idx < bs.w.size()) cur = bs.w[word_idx];
                else return false;
            }
        }
    }
};

// ========================= Hamming ball (binary) =========================
static inline uint32_t popcnt_u32(uint32_t x){ return (uint32_t)std::popcount(x); }

void gen_ball_indices(int n, int center, int r, vector<int>& out) {
    out.clear();
    out.push_back(center);
    vector<int> idx(n); iota(idx.begin(), idx.end(), 0);
    for (int k=1;k<=r;k++) {
        // enumerate combinations of k indices to flip
        vector<int> sel(k);
        // first comb: 0..k-1
        for (int i=0;i<k;i++) sel[i]=i;
        auto emit = [&](const vector<int>& sel){
            int m = center;
            for (int id: sel) m ^= (1<<idx[id]);
            out.push_back(m);
        };
        while (true) {
            emit(sel);
            int p = k-1;
            while (p>=0 && sel[p]==(n - k + p)) --p;
            if (p<0) break;
            sel[p]++;
            for (int q=p+1;q<k;q++) sel[q]=sel[q-1]+1;
        }
    }
}

void build_balls_bitset(int n, int r, vector<UBitset>& balls) {
    const size_t N = 1u<<n;
    balls.assign(N, UBitset(N));
    for (size_t v=0; v<N; ++v) balls[v].reset(N);
    vector<int> pts; pts.reserve(1<<n);
    for (size_t v=0; v<N; ++v) {
        gen_ball_indices(n, (int)v, r, pts);
        for (int x: pts) balls[v].set_bit((size_t)x);
    }
}

// ========================= Covering helpers =========================
UBitset cover_mask(const vector<int>& code, const vector<UBitset>& balls, size_t N) {
    UBitset cov(N);
    cov.clear_all();
    for (int c: code) cov.or_assign(balls[c]);
    return cov;
}
inline uint64_t cost_uncovered(int n, const UBitset& cov) {
    const uint64_t total = 1ull<<n;
    return total - cov.popcount();
}

// rem = ALL ^ covered
void compute_remaining(const UBitset& all_mask, const UBitset& covered, UBitset& rem) {
    rem.assign_xor(all_mask, covered);
}

// gain = popcount( (covered | balls[cand]) ^ covered )
uint64_t gain_of(int cand, const UBitset& covered, const vector<UBitset>& balls, UBitset& tmp_or, UBitset& tmp_diff) {
    tmp_or.assign_or(covered, balls[cand]);
    tmp_diff.assign_xor(tmp_or, covered);
    return tmp_diff.popcount();
}

vector<int> greedy_init(int n, int r, const vector<UBitset>& balls) {
    const size_t N = 1u<<n;
    UBitset all_mask(N); all_mask.set_all();
    UBitset covered(N); covered.clear_all();
    UBitset rem(N), tmp_or(N), tmp_diff(N);

    vector<int> code;
    size_t pos;

    while (true) {
        compute_remaining(all_mask, covered, rem);
        if (rem.popcount() == 0) break;

        // choose best candidate among uncovered points
        uint64_t best_gain = 0; int best_c = -1;
        OneBitsIter it(rem);
        while (it.next(pos)) {
            uint64_t g = gain_of((int)pos, covered, balls, tmp_or, tmp_diff);
            if (g > best_gain) {
                best_gain = g; best_c = (int)pos;
                // cannot do early-perfect check cheaply; keep scanning
            }
        }
        if (best_c < 0) break; // should not happen
        code.push_back(best_c);
        covered.or_assign(balls[best_c]);
    }
    return code;
}

vector<int> prune_minimal(const vector<int>& code, const vector<UBitset>& balls, int n) {
    vector<int> cur = code;
    const size_t N = 1u<<n;
    bool changed = true;
    while (changed) {
        changed = false;
        UBitset base = cover_mask(cur, balls, N);
        for (size_t i=0;i<cur.size();++i) {
            vector<int> tmp; tmp.reserve(cur.size()-1);
            for (size_t j=0;j<cur.size();++j) if (j!=i) tmp.push_back(cur[j]);
            UBitset cov = cover_mask(tmp, balls, N);
            if (cov.equals(base)) {
                cur.swap(tmp);
                changed = true;
            }
        }
    }
    return cur;
}

// ========================= Tabu Search =========================
struct Params {
    int iters = 2000;
    int tabu_tenure = 25;
    int candidate_sample = 100;
    uint64_t seed = 42;
};

pair<vector<int>, uint64_t> tabu_search(
    int n, int r,
    const vector<int>& init_code,
    const vector<UBitset>& balls,
    const Params& P)
{
    const size_t N = 1u<<n;
    std::mt19937_64 rng(P.seed);

    vector<int> code = prune_minimal(init_code, balls, n);
    UBitset cov = cover_mask(code, balls, N);

    vector<int> best = code;
    UBitset best_cov = cov;
    uint64_t best_cost = cost_uncovered(n, cov);
    size_t best_len = code.size();

    // tabu memory
    // out,in -> expire_iter ; and out -> expire for removal
    unordered_map<uint64_t,int> tabu_outin;
    unordered_map<int,int> tabu_remove;

    auto key_pair = [](int o, int i)->uint64_t {
        return (uint64_t(uint32_t(o))<<32) | uint32_t(i);
    };

    auto is_tabu_outin = [&](int o, int i, int t)->bool{
        auto it = tabu_outin.find(key_pair(o,i));
        return it!=tabu_outin.end() && it->second >= t;
    };
    auto is_tabu_remove = [&](int o, int t)->bool{
        auto it = tabu_remove.find(o);
        return it!=tabu_remove.end() && it->second >= t;
    };
    auto make_tabu_outin = [&](int o, int i, int t){
        tabu_outin[key_pair(o,i)] = t + P.tabu_tenure;
    };
    auto make_tabu_remove = [&](int o, int t){
        tabu_remove[o] = t + P.tabu_tenure;
    };

    vector<int> pool; pool.reserve(N);
    vector<int> tmp_code; tmp_code.reserve(1024);

    for (int it=1; it<=P.iters; ++it) {
        bool improved = false;
        uint64_t cur_cost = cost_uncovered(n, cov);
        size_t cur_len = code.size();

        if (cur_cost > 0) {
            // gather up to candidate_sample from uncovered
            UBitset all_mask(N); all_mask.set_all();
            UBitset rem(N); rem.assign_xor(all_mask, cov);

            pool.clear();
            // sample first candidate_sample ones (simple & fast)
            size_t pos; int picked=0;
            for (OneBitsIter bit(rem); bit.next(pos) && picked<P.candidate_sample; ++picked) {
                pool.push_back((int)pos);
            }

            int best_a = -1;
            uint64_t best_delta = 0;
            vector<int> best_new;
            UBitset tmp_or(N), tmp_diff(N);

            for (int a: pool) {
                UBitset cov1 = cov; cov1.or_assign(balls[a]);
                uint64_t cost1 = (1ull<<n) - cov1.popcount();
                if (cost1 < cur_cost) {
                    tmp_code = code; tmp_code.push_back(a);
                    tmp_code = prune_minimal(tmp_code, balls, n);
                    UBitset cov2 = cover_mask(tmp_code, balls, N);
                    uint64_t cost2 = (1ull<<n) - cov2.popcount();
                    if (cost2 < cur_cost) {
                        uint64_t delta = cur_cost - cost2;
                        if (delta > best_delta) {
                            best_delta = delta;
                            best_a = a;
                            best_new = tmp_code;
                        }
                    }
                }
            }
            if (best_a != -1) {
                code.swap(best_new);
                cov = cover_mask(code, balls, N);
                improved = true;
            }
        } else {
            // f=0: try remove to shrink |C|
            vector<int> idx(code.size());
            iota(idx.begin(), idx.end(), 0);
            shuffle(idx.begin(), idx.end(), rng);
            bool removed = false;
            for (int id: idx) {
                int o = code[id];
                if (is_tabu_remove(o, it)) continue;
                tmp_code.clear(); tmp_code.reserve(code.size()-1);
                for (size_t j=0;j<code.size();++j) if ((int)j!=id) tmp_code.push_back(code[j]);
                UBitset cov2 = cover_mask(tmp_code, balls, N);
                if (cost_uncovered(n, cov2)==0) {
                    code.swap(tmp_code);
                    cov = cov2;
                    make_tabu_remove(o, it);
                    removed = true;
                    improved = true;
                    break;
                }
            }
            if (!removed) {
                // try 1-1 swap keeping f=0
                pool.clear();
                // candidates outside code (sample)
                vector<char> inC(N,0);
                for (int c: code) inC[c]=1;
                for (size_t v=0; v<N && (int)pool.size()<P.candidate_sample; ++v)
                    if (!inC[v]) pool.push_back((int)v);
                shuffle(pool.begin(), pool.end(), rng);

                bool swapped = false;
                for (int o: code) {
                    for (int i2: pool) {
                        if (is_tabu_outin(o,i2,it)) continue;
                        tmp_code.clear(); tmp_code.reserve(code.size());
                        for (int c: code) if (c!=o) tmp_code.push_back(c);
                        tmp_code.push_back(i2);
                        UBitset cov2 = cover_mask(tmp_code, balls, N);
                        if (cost_uncovered(n, cov2)==0) {
                            // prune then check shrink
                            tmp_code = prune_minimal(tmp_code, balls, n);
                            if (tmp_code.size() < code.size()) {
                                code.swap(tmp_code);
                                cov = cover_mask(code, balls, N);
                                make_tabu_outin(o,i2,it);
                                swapped = true; improved = true;
                                break;
                            }
                        }
                    }
                    if (swapped) break;
                }
            }
        }

        // update best
        cur_cost = cost_uncovered(n, cov);
        cur_len  = code.size();
        if (cur_cost < best_cost || (cur_cost==best_cost && cur_len < best_len)) {
            best = code; best_cov = cov; best_cost = cur_cost; best_len = cur_len;
        }

        // diversification light
        if (!improved) {
            if (cur_cost > 0) {
                // add one uncovered randomly
                UBitset all_mask(N); all_mask.set_all();
                UBitset rem(N); rem.assign_xor(all_mask, cov);
                // pick any set bit
                size_t pos;
                OneBitsIter itb(rem);
                if (itb.next(pos)) {
                    code.push_back((int)pos);
                    cov.or_assign(balls[(int)pos]);
                }
            } else {
                // random 1-1 swap (if still cover)
                if (!code.empty()) {
                    uniform_int_distribution<int> dcode(0,(int)code.size()-1);
                    int o = code[dcode(rng)];
                    int i2 = o;
                    uniform_int_distribution<int> dall(0, (int)N-1);
                    for (int tries=0; tries<32; ++tries) {
                        i2 = dall(rng);
                        if (i2==o) continue;
                        tmp_code.clear();
                        for (int c: code) if (c!=o) tmp_code.push_back(c);
                        tmp_code.push_back(i2);
                        UBitset cov2 = cover_mask(tmp_code, balls, N);
                        if (cost_uncovered(n, cov2)==0) {
                            code = prune_minimal(tmp_code, balls, n);
                            cov  = cover_mask(code, balls, N);
                            break;
                        }
                    }
                }
            }
        }
    }
    return {best, best_cost};
}

// ========================= Pretty print =========================
string to_binary(int x, int n) {
    string s; s.resize(n);
    for (int i=n-1;i>=0;--i) { s[n-1-i] = ((x>>i)&1)?'1':'0'; }
    return s;
}

// ========================= Main =========================
int main(int argc, char** argv){
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " n r [--iters I] [--tabu T] [--sample K] [--seed S]\n";
        return 1;
    }
    int n = stoi(argv[1]);
    int r = stoi(argv[2]);
    Params P;
    for (int i=3;i<argc;i++){
        string a = argv[i];
        if (a=="--iters" && i+1<argc) P.iters = stoi(argv[++i]);
        else if (a=="--tabu" && i+1<argc) P.tabu_tenure = stoi(argv[++i]);
        else if (a=="--sample" && i+1<argc) P.candidate_sample = stoi(argv[++i]);
        else if (a=="--seed" && i+1<argc) P.seed = stoull(argv[++i]);
    }
    if (n<1 || n>20) {
        cerr << "Note: this implementation targets n <= 16–18. Given n="<<n<<".\n";
    }

    const size_t N = 1u<<n;
    cerr << "[info] Precomputing balls... n="<<n<<", r="<<r<<", universe="<<N<<" points\n";
    vector<UBitset> balls;
    build_balls_bitset(n, r, balls);

    cerr << "[info] Greedy init...\n";
    auto init = greedy_init(n, r, balls);
    init = prune_minimal(init, balls, n);
    UBitset cov0 = cover_mask(init, balls, N);
    cout << "Greedy init: |C|=" << init.size()
         << " ; uncovered=" << ( (1ull<<n) - cov0.popcount() ) << "\n";

    cerr << "[info] Tabu search...\n";
    auto [best, best_cost] = tabu_search(n, r, init, balls, P);

    cout << "Tabu result: |C|=" << best.size()
         << " ; uncovered=" << best_cost << "\n";
    cout << "Codewords (binary):\n";
    sort(best.begin(), best.end());
    for (int c: best) cout << to_binary(c, n) << "\n";
    return 0;
}
