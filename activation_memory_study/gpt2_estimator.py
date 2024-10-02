VOCAB_SIZE = 128256
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 8192
ATTENTION_HEAD_SIZE = 32
HIDDEN_LAYERS = 16
KEY_VALUE_HEAD_SIZE = 8
BATCH_SIZE = 1
SEQ_LENGTH = 1024
DTYPE_BYTES = 2
MASTER_DTYPE_BYTES = 4

class Llama3_CPU_Offload_Memory_VRAM_Estimator:
    def __init__(self, dtype_bytes, master_dtype_bytes, vocab_size, hidden_size, intermediate_size, attention_head_size, hiddern_layers, key_value_head_size, batch_size, seq_length):
        self.dtype_bytes = dtype_bytes
        self.master_dtype_bytes = master_dtype_bytes
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.attention_head_size = attention_head_size
        self.hiddern_layers = hiddern_layers
        self.key_value_head_size = key_value_head_size
        self.batch_size = batch_size
        self.seq_length = seq_length

    def get_embedding_size(self):
        return self.vocab_size * self.hidden_size
    
    def get_rms_normalization_size(self):
        return self.hidden_size

    def get_group_query_attention_size(self):
        group_size = self.attention_head_size // self.key_value_head_size
        q_matrix = self.hidden_size * self.hidden_size
        k_matrix = self.hidden_size * self.hidden_size // group_size
        v_matrix = self.hidden_size * self.hidden_size // group_size
        o_matrix = self.hidden_size * self.hidden_size
        return q_matrix + k_matrix + v_matrix + o_matrix

    def get_mlp_size(self):
        down_proj =  self.intermediate_size * self.hidden_size
        gate_proj = self.hidden_size * self.intermediate_size
        up_proj = self.hidden_size * self.intermediate_size
        return down_proj + gate_proj + up_proj

    def get_decoder_layer_size(self):
        input_norm_size = self.get_rms_normalization_size()
        attention_size = self.get_group_query_attention_size()
        output_norm_size = self.get_rms_normalization_size()
        mlp_size = self.get_mlp_size()
        return input_norm_size + attention_size + output_norm_size + mlp_size
    
    def get_linear_output_size(self):
        return self.hidden_size

    def get_transformer_size(self):
        embeddings_size = self.get_embedding_size()
        decoder_size = self.get_decoder_layer_size() * self.hiddern_layers
        final_norm_size = self.get_rms_normalization_size()
        linear_output_size = self.get_linear_output_size()
        return embeddings_size + decoder_size + final_norm_size + linear_output_size
    
    def get_peak_sub_module_size(self):
        """
        Because enable param and optimizer offload to CPU, we only need to consider the peak size of the largest sub-module
        """
        return max([self.get_embedding_size(), self.get_decoder_layer_size(), self.get_rms_normalization_size(), self.get_linear_output_size()])

    def get_training_peak_weight_bytes(self):
        return self.get_peak_sub_module_size() * self.dtype_bytes
    
    def get_training_peak_grad_bytes(self):
        return self.get_peak_sub_module_size() * self.dtype_bytes
    
    def get_training_peak_adam_optimizer_state_bytes(self):
        OPTIMIZER_STATE_NUM = 3 # master weights, momentum, variance
        return self.get_peak_sub_module_size() * OPTIMIZER_STATE_NUM * self.master_dtype_bytes
    
    def get_training_peak_bytes(self):
        return self.get_training_peak_weight_bytes() + self.get_training_peak_grad_bytes() + self.get_training_peak_adam_optimizer_state_bytes()

    def get_training_peak_weight_bytes_originally(self):    # all in GPU
        return self.get_transformer_size() * self.dtype_bytes
    
    def get_training_peak_grad_bytes_originally(self):    # all in GPU
        return self.get_transformer_size() * self.dtype_bytes
    
    def get_training_peak_adam_optimizer_state_bytes_originally(self):    # all in GPU
        OPTIMIZER_STATE_NUM = 3 # master weights, momentum, variance
        return self.get_transformer_size() * OPTIMIZER_STATE_NUM * self.master_dtype_bytes
    
    def get_training_peak_bytes_originally(self):    # all in GPU
        return self.get_training_peak_weight_bytes_originally() + self.get_training_peak_grad_bytes_originally() + self.get_training_peak_adam_optimizer_state_bytes_originally()

class Llama3_Activation_VRAM_Estimator:
    """Work for Llama3 Family models"""
    def __init__(self, dtype_bytes, vocab_size, hidden_size, intermediate_size, attention_head_size, hiddern_layers, key_value_head_size, batch_size, seq_length):
        self.dtype_bytes = dtype_bytes
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.attention_head_size = attention_head_size
        self.hiddern_layers = hiddern_layers
        self.key_value_head_size = key_value_head_size
        self.batch_size = batch_size
        self.seq_length = seq_length

    def estimate_embed_tokens(self):
        input_activation = self.batch_size * self.seq_length * self.hidden_size * self.dtype_bytes
        return input_activation
    
    def estimate_rms_normalization(self):
        input_activation = self.batch_size * self.seq_length * self.hidden_size * self.dtype_bytes
        variance_activation = self.batch_size * self.seq_length * self.dtype_bytes
        mean_activation = self.batch_size * self.seq_length * self.dtype_bytes
        return input_activation + variance_activation + mean_activation

    def estimate_group_query_attention(self):
        group_size = self.attention_head_size // self.key_value_head_size
        input_activation = self.batch_size * self.seq_length * self.hidden_size * self.dtype_bytes
        attention_score_input_q_activation = self.batch_size * self.seq_length * self.hidden_size * self.dtype_bytes
        attention_score_input_k_activation = self.batch_size * self.seq_length * self.hidden_size * self.dtype_bytes // group_size
        softmax_input_activation = self.batch_size * self.seq_length * self.seq_length * self.attention_head_size * self.dtype_bytes
        attention_value_input_softmax_activation = self.batch_size * self.seq_length * self.seq_length * self.attention_head_size * self.dtype_bytes
        attention_value_input_v_activation = self.batch_size * self.seq_length * self.hidden_size * self.dtype_bytes // group_size
        return input_activation + attention_score_input_q_activation + attention_score_input_k_activation + softmax_input_activation + attention_value_input_softmax_activation + attention_value_input_v_activation

    def estimate_mlp(self):
        """forward pass of the MLP
        x [hidden_size]
        x_1 = gate_proj(x) [intermediate_size]
        x_2 = up_proj(x) [intermediate_size]
        x_silu = silu(x_1) [intermediate_size]
        x_intermediate = x_silu * x_2 [intermediate_size]
        output = down_proj(x_intermediate) [hidden_size]
        """
        # x
        input_activation = self.batch_size * self.seq_length * self.hidden_size * self.dtype_bytes
        # x_1
        silu_input_activation = self.batch_size * self.seq_length * self.intermediate_size * self.dtype_bytes
        # x_silu
        intermediate_input_silu_activation = self.batch_size * self.seq_length * self.intermediate_size * self.dtype_bytes
        # x_2
        intermediate_input_up_activation = self.batch_size * self.seq_length * self.intermediate_size * self.dtype_bytes
        # x_intermediate
        down_proj_input_activation = self.batch_size * self.seq_length * self.intermediate_size * self.dtype_bytes
        return input_activation + silu_input_activation + intermediate_input_silu_activation + intermediate_input_up_activation + down_proj_input_activation

    def estimate_decoder_layer(self):
        rms_activation = self.estimate_rms_normalization()
        attention_activation = self.estimate_group_query_attention()
        rms_activation = self.estimate_rms_normalization()
        mlp_activation = self.estimate_mlp()
        return rms_activation + attention_activation + rms_activation + mlp_activation
    
    def estimate_linear_output(self):
        input_activation = self.batch_size * self.seq_length * self.hidden_size * self.dtype_bytes
        return input_activation

    def estimate_transformer(self):
        embeddings_activation = self.estimate_embed_tokens()
        decoder_activation = self.estimate_decoder_layer() * self.hiddern_layers
        final_rms_activation = self.estimate_rms_normalization()
        linear_output_activation = self.estimate_linear_output()
        return embeddings_activation + decoder_activation + final_rms_activation + linear_output_activation

    def estimate_checkpoint_size(self): # if enable checkpointing at each decoder layer
        return self.hiddern_layers * self.batch_size * self.seq_length * self.hidden_size * self.dtype_bytes

if __name__ == "__main__":
    estimator = Llama3_Activation_VRAM_Estimator(DTYPE_BYTES, VOCAB_SIZE, HIDDEN_SIZE, INTERMEDIATE_SIZE, ATTENTION_HEAD_SIZE, HIDDEN_LAYERS, KEY_VALUE_HEAD_SIZE, BATCH_SIZE, SEQ_LENGTH)
    print("Estimated activations for Llama3 model")
    print("Final estimated activations in GB: ", estimator.estimate_transformer() / 1024**3, "GB")
    print("Peak if checkpoint at each Decoder Layer: ", estimator.estimate_decoder_layer() / 1024**3, "GB")
    print("Checkpoint size if checkpoint at each Decoder Layer: ", estimator.estimate_checkpoint_size() / 1024**3, "GB")
    memory_estimator = Llama3_CPU_Offload_Memory_VRAM_Estimator(DTYPE_BYTES, MASTER_DTYPE_BYTES, VOCAB_SIZE, HIDDEN_SIZE, INTERMEDIATE_SIZE, ATTENTION_HEAD_SIZE, HIDDEN_LAYERS, KEY_VALUE_HEAD_SIZE, BATCH_SIZE, SEQ_LENGTH)
    print("Estimated weight size for Llama3 model", memory_estimator.get_transformer_size())
    print("Estimated peak memory usage in GB: ", memory_estimator.get_training_peak_bytes() / 1024**3, "GB")
    print("Estimated peak memory usage originally in GB: ", memory_estimator.get_training_peak_bytes_originally() / 1024**3, "GB")