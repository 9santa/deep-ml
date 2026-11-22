def mask_instruction_loss(batch, pad_token_id, ignore_index=-100):
    if not batch:
        return {'inputs': [], 'targets': []}

    # Longest 'tokens' list in the batch
    L = max(len(item['tokens']) for item in batch)

    all_inputs, all_targets = [], []

    for item in batch:
        tokens = item["tokens"]

        # 1. Pad to maximum length
        padded = tokens + [pad_token_id] * (L - len(tokens))

        # 2. Shift for next-token prediction
        inputs = padded[:-1]
        targets = padded[1:]

        # 3. Mask instruction tokens using slicing
        n = max(0, item["instruction_length"] - 1)
        targets[:n] = [ignore_index] * n

        # 4. Keep first pad token, mask subsequent ones
        try:
            first_pad = targets.index(pad_token_id)
        except ValueError:
            pass
        else:
            targets[first_pad + 1:] = [
                ignore_index if token == pad_token_id else token
                for token in targets[first_pad + 1:]
            ]

        all_inputs.append(inputs)
        all_targets.append(targets)

    return {"inputs": all_inputs, "targets": all_targets}