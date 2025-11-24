def blockwise_direct_sum(partition_A, partition_B):
    if len(partition_A) != len(partition_B):
        raise ValueError(f"Partition count mismatch: A has {len(partition_A)}, B has {len(partition_B)}.")

    k = len(partition_A)
    bds_code = []

    print(f"Starting BDS with k = {k}...")

    for i in range(k):
        sub_A = partition_A[i]
        sub_B = partition_B[i]

        if not sub_A or not sub_B:
            continue

        count = 0
        for word_a in sub_A:
            for word_b in sub_B:
                new_word = word_a + word_b
                bds_code.append(new_word)
                count += 1
        
        print(f"  - Block {i}: Combined {len(sub_A)} words from A with {len(sub_B)} words from B -> Created {count} new words.")

    return bds_code

part_A = [
    ["00"], 
    ["11"] 
]

part_B = [
    ["000", "011"],
    ["101", "110"]
]

result_code = blockwise_direct_sum(part_A, part_B)

print("-" * 30)
print(f"Total codewords in BDS(A,B): {len(result_code)}")
print("New Code List:")
print(result_code)
