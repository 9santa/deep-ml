import numpy as np

def lora_forward(
    x: list[list[float]],
    W: list[list[float]],
    A: list[list[float]],
    B: list[list[float]],
    alpha: float = 1.0
) -> list[list[float]]:
    """
    Compute the LoRA forward pass.
    
    Args:
        x: Input matrix (batch_size x in_features)
        W: Frozen pretrained weights (in_features x out_features)
        A: LoRA matrix A (rank x out_features)
        B: LoRA matrix B (in_features x rank)
        alpha: LoRA scaling factor
        
    Returns:
        Output matrix (batch_size x out_features)
    """
    
    x, W, A, B = np.asarray(x), np.asarray(W), np.asarray(A), np.asarray(B)

    xW = np.dot(x, W) # (batch_size, in_features, out_features)
    
    lora_rank = A.shape[0]
    lora_scale = alpha / lora_rank

    xB = np.dot(x, B) # (batch_size, in_features, rank)

    # (in_features, rank) @ (rank, out_features) matmul broadcasted across the batch
    xBA = np.matmul(xB, A) # (batch_size, in_features, out_features)

    y = xW + lora_scale * xBA
    return y
