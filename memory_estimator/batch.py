import numpy as np
import matplotlib.pyplot as plt

# Function to estimate memory usage


def estimate_memory_usage(P, s, b, h, h_ff, L, v, t, d):
    M_static = (6 + 10 / d) * P / t
    M_activation = s * b * h * L * \
        (8 + (8 + 8 * h_ff / h) / t) + 2 * s * b * h + 4 * s * b * v / t
    M_cross_entropy = 6 * s * b * v / t
    total_memory = M_static + M_activation + M_cross_entropy
    bytes_to_gb = 1 / (1024 ** 3)
    return {
        "Static Memory (GB)": M_static * bytes_to_gb,
        "Activations Memory (GB)": M_activation * bytes_to_gb,
        "Cross Entropy Overhead (GB)": M_cross_entropy * bytes_to_gb,
        "Total Estimated Memory (GB)": total_memory * bytes_to_gb
    }


# Define parameters
params = {
    "P": 1.41e9,
    "s": 1024,
    "h": 2048,
    "h_ff": 5440,
    "L": 24,
    "v": 50257,
    "t": 1,
    "d": 1
}

# Calculate memory usage for batch sizes from 1 to 16
batches = [2**i for i in range(5)]  # [1, 2, 4, 8, 16]
results_batch = [estimate_memory_usage(params['P'], params['s'], b, params['h'],
                                       params['h_ff'], params['L'], params['v'], params['t'], params['d']) for b in batches]

# Extract memory estimates for plotting
total_memory_batch = [
    result['Total Estimated Memory (GB)'] for result in results_batch]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(batches, total_memory_batch, marker='o')
plt.title('Estimated Total Memory Usage vs. Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Total Estimated Memory (GB)')
plt.grid(True)
plt.show()
