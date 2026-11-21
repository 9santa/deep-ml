def mask_instruction_loss(batch, pad_token_id, ignore_index=-100):
    if not batch:
        return {'inputs': [], 'targets': []}

    # Longest 'tokens' list in the batch
    L = max(len(item['tokens']) for item in batch)

    inp, tar = [], []

    for item in batch:
        number_of_pads = L - len(item['tokens'])
        padded = item['tokens'] + [pad_token_id] * number_of_pads

        inputs = padded[:-1]
        targets = padded[1:]

        seen_pad = False
        for i in range(len(targets)):
            if i + 1 < item['instruction_length']:
                targets[i] = ignore_index

            if targets[i] == pad_token_id:
                if seen_pad:
                    targets[i] = ignore_index
                else:
                    seen_pad = True

        inp.append(inputs)
        tar.append(targets)
    
    return {'inputs': inp, 'targets': tar}
