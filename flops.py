import math

def calculate_prefilling_flops(T, n, m, d):
    return T*(4*n*d**2+2*n**2*d+2*n*d*m)

def calculate_decoding_flops(T, n, m, d, R):
    return T*R*((4*d**2+2*d*m)+2*(d*n+d*(R+1)/2))

def calculate_flops(T=None, n=None, m=None, d=None, R=100):
    prefilling_flops = calculate_prefilling_flops(T, n, m, d)
    decoding_flops = calculate_decoding_flops(T, n, m, d, R)
    total_flops = prefilling_flops + decoding_flops
    return prefilling_flops, decoding_flops, total_flops

def calculate_token_pruned_flops(states, ratios, T=None, n=None, m=None, d=None, R=100):
    original_prefilling_flops, original_decoding_flops, original_total_flops = calculate_flops(T, n, m, d, R)
    print(f"original_total_flops: {original_total_flops/1e12:.1f} TFlops")

    pruned_prefilling_flops = calculate_prefilling_flops(states[0], n, m, d)
    for idx, (state, ratio) in enumerate(zip(states, ratios)):
        total_layers = states[idx+1] - state if idx != len(states) - 1 else T - state
        pruned_prefilling_flops += calculate_prefilling_flops(total_layers, int(n*ratio), m, d)
    pruned_decoding_flops = calculate_decoding_flops(T, int(n*ratio), m, d, R)
    total_pruned_flops = pruned_prefilling_flops + pruned_decoding_flops

    return pruned_prefilling_flops, pruned_decoding_flops, total_pruned_flops

prefilling_flops, decoding_flops, total_flops = calculate_flops(d=3584, m=18944, T=28, n=196*32)
print(f"model_type: ov-7b")
print(f"prefilling_flops: {prefilling_flops/1e12:.1f} TFlops")
print(f"decoding_flops: {decoding_flops/1e12:.1f} TFlops")
print(f"total_flops: {total_flops/1e12:.1f} TFlops")
print("--------------------------------")

prefilling_flops, decoding_flops, total_flops = calculate_flops(d=3584, m=18944, T=28, n=20*32)
print(f"model_type: ov-7b")
print(f"prefilling_flops: {prefilling_flops/1e12:.1f} TFlops")
print(f"decoding_flops: {decoding_flops/1e12:.1f} TFlops")
print(f"total_flops: {total_flops/1e12:.1f} TFlops")
print("--------------------------------")

prefilling_flops, decoding_flops, total_flops = calculate_token_pruned_flops(states=[18], ratios=[0.5], d=3584, m=18944, T=28, n=20*32)