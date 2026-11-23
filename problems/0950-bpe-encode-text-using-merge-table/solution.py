def bpe_encode(text, token_to_id, merges):

    # 1. Convert chars into token IDs
    tokens = [token_to_id[c] for c in text]

    while len(tokens) > 1:

        # 2. Find the highest-priority merge
        best_pair = None
        best_rank = float("inf")

        for a, b in zip(tokens, tokens[1:]):
            pair = (a, b)

            if pair in merges and merges[pair] < best_rank:
                best_pair = pair
                best_rank = merges[pair]

        if best_pair is None:
            break

        # #. Merge all non-overlapping occurences
        new_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and (tokens[i], tokens[i+1]) == best_pair
            ):
                new_tokens.append(best_rank)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens

    return tokens

