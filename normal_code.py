import itertools

def hamming_distance(s1, s2):
    """Tính khoảng cách Hamming giữa 2 chuỗi."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def min_dist_to_set(x, code_set):
    """Tìm khoảng cách ngắn nhất từ x đến một tập hợp các từ mã."""
    if not code_set: return float('inf')
    return min(hamming_distance(x, c) for c in code_set)

def is_normal_code(code, n, r):
    """
    Kiểm tra tính Normal theo Definition 1 của bài báo.
    Norm <= 2r + 1
    """
    norm_limit = 2 * r + 1
    all_vectors = ["".join(seq) for seq in itertools.product("01", repeat=n)]
    
    # Duyệt qua từng tọa độ i (từ 0 đến n-1)
    for i in range(n):
        # Phân hoạch mã C thành C0 (bit tại i là '0') và C1 (bit tại i là '1')
        # Theo bài báo, C0 và C1 là tập con của C
        c0 = [w for w in code if w[i] == '0']
        c1 = [w for w in code if w[i] == '1']
        
        # Cả hai tập con phải không rỗng [cite: 107]
        if not c0 or not c1:
            continue
            
        # Kiểm tra điều kiện với mọi vector x trong không gian F2^n
        satisfies_norm = True
        for x in all_vectors:
            d0 = min_dist_to_set(x, c0)
            d1 = min_dist_to_set(x, c1)
            
            if d0 + d1 > norm_limit:
                satisfies_norm = False
                break
        
        if satisfies_norm:
            print(f"Mã này là NORMAL theo tọa độ index {i}.")
            return True

    print("Mã này KHÔNG phải là Normal.")
    return False

# --- CHẠY THỬ VỚI VÍ DỤ TRÊN ---
# Mã lặp lại: 000 và 111
my_code = ["000", "111"]
length = 3
radius = 1

print(f"Kiểm tra mã: {my_code} với bán kính r={radius}")
is_normal_code(my_code, length, radius)