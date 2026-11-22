import numpy as np

def masked_ce_loss(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute mean cross-entropy loss over masked (response) positions only.

    Args:
        logits: (seq_len, vocab_size) array of unnormalized scores.
        targets: (seq_len,) array of integer target token ids.
        mask: (seq_len,) boolean array; True = include in loss.

    Returns:
        Mean cross-entropy over positions where mask is True (float).
    """
    
    mask = mask.astype(bool)
    if not np.any(mask):
        return 0.0

    active_logits = logits[mask]
    active_targets = targets[mask]

    max_logits = np.max(active_logits, axis=-1, keepdims=True)
    shifted_logits = active_logits - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=-1, keepdims=True))

    target_logits = shifted_logits[np.arange(len(active_targets)), active_targets]

    token_losses = log_sum_exp - target_logits

    return float(np.mean(token_losses))

    