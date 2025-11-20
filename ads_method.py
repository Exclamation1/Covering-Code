def get_subcode_stripped(code, position, value_bit):
    """
    Lọc các từ mã có bit tại 'position' bằng 'value_bit',
    sau đó cắt bỏ bit tại vị trí đó đi.
    """
    subcode = []
    for word in code:
        # Kiểm tra bit tại vị trí chỉ định
        if word[position] == value_bit:
            # Cắt bỏ bit đó: nối phần trước và phần sau vị trí đó
            stripped_word = word[:position] + word[position+1:]
            subcode.append(stripped_word)
    return subcode

def amalgamated_direct_sum(code_A, code_B):
    """
    Thực hiện ADS giữa Code A (xét bit cuối) và Code B (xét bit đầu).
    """
    n_A = len(code_A[0])
    
    # Bước 1: Chuẩn bị các thành phần từ Code A (xét bit cuối cùng: index = n_A - 1)
    # A0': Các từ tận cùng là 0, đã cắt đuôi
    A0_prime = get_subcode_stripped(code_A, n_A - 1, '0')
    # A1': Các từ tận cùng là 1, đã cắt đuôi
    A1_prime = get_subcode_stripped(code_A, n_A - 1, '1')
    
    # Bước 2: Chuẩn bị các thành phần từ Code B (xét bit đầu tiên: index = 0)
    # B0': Các từ bắt đầu bằng 0, đã cắt đầu
    B0_prime = get_subcode_stripped(code_B, 0, '0')
    # B1': Các từ bắt đầu bằng 1, đã cắt đầu
    B1_prime = get_subcode_stripped(code_B, 0, '1')
    
    new_code = []
    
    # Bước 3: Ghép nhóm 0 (A0' + B0')
    for a in A0_prime:
        for b in B0_prime:
            new_code.append(a + b)
            
    # Bước 4: Ghép nhóm 1 (A1' + B1')
    for a in A1_prime:
        for b in B1_prime:
            new_code.append(a + b)
            
    return new_code

# --- CHẠY THỬ VỚI DỮ LIỆU VÍ DỤ ---
# Code A: Repetition Code (n=3)
C_A = ["000", "111"]

# Code B: Parity Check Code (n=3, even weight)
C_B = ["000", "011", "101", "110"]

result_C = amalgamated_direct_sum(C_A, C_B)

print(f"Code A (len {len(C_A[0])}): {C_A}")
print(f"Code B (len {len(C_B[0])}): {C_B}")
print("-" * 30)
print(f"Kết quả ADS (len {len(result_C[0])}): {result_C}")
print(f"Số lượng từ mã: {len(result_C)}")