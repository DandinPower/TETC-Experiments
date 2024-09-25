def estimate_memory_usage(P, s, b, h, h_ff, L, v, t, d):
    """
    Estimate the memory usage for training a GPT model.

    Parameters:
    P (int): Number of parameters.
    s (int): Sequence length.
    b (int): Micro-batch size.
    h (int): Hidden dimension size.
    h_ff (int): Feed forward size.
    L (int): Number of transformer layers.
    v (int): Vocabulary size.
    t (int): Tensor parallel size.
    d (int): Data parallel size.

    Returns:
    dict: Estimated memory usage in GB for static memory, activations memory, and total memory.
    """

    # Static Memory Estimation
    M_static = (6 + 10 / d) * P / t

    # Activations Memory Estimation
    M_activation = s * b * h * L * \
        (8 + (8 + 8 * h_ff / h) / t) + 2 * s * b * h + 4 * s * b * v / t

    # Cross Entropy Overhead
    M_cross_entropy = 6 * s * b * v / t

    # Total Estimated Memory
    total_memory = M_static + M_activation + M_cross_entropy

    # Convert bytes to gigabytes
    bytes_to_gb = 1 / (1024 ** 3)

    return {
        "Static Memory (GB)": M_static * bytes_to_gb,
        "Activations Memory (GB)": M_activation * bytes_to_gb,
        "Cross Entropy Overhead (GB)": M_cross_entropy * bytes_to_gb,
        "Total Estimated Memory (GB)": total_memory * bytes_to_gb
    }


# Example usage
params = {
    "P": 1.41e9,   # Number of parameters for gpt-1b
    "s": 1024,     # Sequence length
    "b": 2,        # Micro-batch size
    "h": 2048,     # Hidden dimension size
    "h_ff": 5440,  # Feed forward size
    "L": 24,       # Number of transformer layers
    "v": 50257,    # Vocabulary size
    "t": 1,        # Tensor parallel size
    "d": 1         # Data parallel size
}

memory_estimates = estimate_memory_usage(**params)
print(memory_estimates)
