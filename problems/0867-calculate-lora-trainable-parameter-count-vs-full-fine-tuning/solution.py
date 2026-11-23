def lora_param_count(num_layers: int, d_model: int, num_target_matrices: int, rank: int) -> dict:
    """Compute LoRA vs full fine-tuning parameter counts."""
    
    total_matrices = num_layers * num_target_matrices
    lora_params = total_matrices * 2 * d_model * rank
    full_paarms = total_matrices * d_model * d_model
    ratio = full_paarms / lora_params
    
    return {
        'lora_params': lora_params,
        'full_params': full_paarms,
        'compression_ratio': ratio
    }