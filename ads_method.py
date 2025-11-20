def get_subcode_stripped(code, position, value_bit):
    subcode = []
    for word in code:
        # Kiểm tra bit tại vị trí chỉ định
        if word[position] == value_bit:
            stripped_word = word[:position] + word[position+1:]
            subcode.append(stripped_word)
    return subcode

def amalgamated_direct_sum(code_A, code_B):
    n_A = len(code_A[0])
    A0_prime = get_subcode_stripped(code_A, n_A - 1, '0')
    A1_prime = get_subcode_stripped(code_A, n_A - 1, '1')
    
    B0_prime = get_subcode_stripped(code_B, 0, '0')
    B1_prime = get_subcode_stripped(code_B, 0, '1')
    
    new_code = []

    for a in A0_prime:
        for b in B0_prime:
            new_code.append(a + b)
            
    # Bước 4: Ghép nhóm 1 (A1' + B1')
    for a in A1_prime:
        for b in B1_prime:
            new_code.append(a + b)
            
    return new_code



C_A = ["000", "111"]


C_B = ["000", "011", "101", "110"]

result_C = amalgamated_direct_sum(C_A, C_B)

print(f"Code A (len {len(C_A[0])}): {C_A}")
print(f"Code B (len {len(C_B[0])}): {C_B}")
print("-" * 30)
print(f"Kết quả ADS (len {len(result_C[0])}): {result_C}")
print(f"Số lượng từ mã: {len(result_C)}")
