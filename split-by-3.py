import timeit

# 1. Standard / Naive shift-and-mask approach (libmorton style)
def split_by_3_standard(a: int) -> int:
    x = a & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000ffff
    x = (x | (x << 16)) & 0x1f0000ff0000ff
    x = (x | (x << 8)) & 0x100f00f00f00f00f
    x = (x | (x << 4)) & 0x10c30c30c30c30c3
    x = (x | (x << 2)) & 0x1249249249249249
    return x

# 2. Optimized (bhsopt + z3) approach
def split_by_3_optimized(a: int) -> int:
    x = a & 0xFFFFFFFF
    x = (x | (x << 22)) & 0x000007FE000007FF
    x = (x | (x << 10)) & 0x001E007E001F801F
    x ^= x << 4
    x |= x << 4
    x &= 0x10C26186186190C3
    x += 0x0186C30C30C30186
    return x & 0x1249249249249249

# Test values
test_val = 123456789*987654321

# Benchmark standard version (1 million iterations)
time_standard = timeit.timeit(lambda: split_by_3_standard(test_val), number=2_000_000)

# Benchmark optimized version (1 million iterations)
time_optimized = timeit.timeit(lambda: split_by_3_optimized(test_val), number=2_000_000)
print(test_val)
print(split_by_3_standard)
print(f"Standard version (1M loops): {time_standard * 1000:.3f} ms")
print(split_by_3_optimized)
print(f"Optimized version (1M loops): {time_optimized * 1000:.3f} ms")