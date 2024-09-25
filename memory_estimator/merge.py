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
    "h": 2048,
    "h_ff": 5440,
    "L": 24,
    "v": 50257,
    "t": 1,
    "d": 1
}

# Calculate memory usage for batch sizes (powers of 2)
batches = [2**i for i in range(5)]  # [1, 2, 4, 8, 16]
results_batch = [estimate_memory_usage(params['P'], 1024, b, params['h'], params['h_ff'],
                                       params['L'], params['v'], params['t'], params['d']) for b in batches]
total_memory_batch = [
    result['Total Estimated Memory (GB)'] for result in results_batch]

# Calculate memory usage for sequence lengths (512 to 8192)
params['b'] = 1  # Fixed batch size
seq_lengths = [2**i * 512 for i in range(5)]  # [512, 1024, ..., 8192]
results_seq = [estimate_memory_usage(params['P'], s, params['b'], params['h'], params['h_ff'],
                                     params['L'], params['v'], params['t'], params['d']) for s in seq_lengths]
total_memory_seq = [
    result['Total Estimated Memory (GB)'] for result in results_seq]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Subplot for Batch Size
axs[0].plot(batches, total_memory_batch, marker='o', label='Batch Size')
axs[0].set_title('Memory Usage vs. Batch Size (Sequence Length = 1024)')
axs[0].set_xlabel('Batch Size')
axs[0].set_ylabel('Total Estimated Memory (GB)')
axs[0].grid(True)

# Add annotations for batch size plot
for i, txt in enumerate(total_memory_batch):
    axs[0].annotate(f'{txt:.2f}', (batches[i], total_memory_batch[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', color='red')

# Subplot for Sequence Length
axs[1].plot(seq_lengths, total_memory_seq, marker='x',
            color='orange', label='Sequence Length')
axs[1].set_title('Memory Usage vs. Sequence Length (Batch Size = 1)')
axs[1].set_xlabel('Sequence Length')
axs[1].set_ylabel('Total Estimated Memory (GB)')
axs[1].grid(True)

# Add annotations for sequence length plot
for i, txt in enumerate(total_memory_seq):
    axs[1].annotate(f'{txt:.2f}', (seq_lengths[i], total_memory_seq[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', color='red')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('memory_usage_subplots.png', dpi=300)
plt.show()
