def bpe_decode(ids, vocab):
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """
    if not ids:
        return ""

    decoded = ""

    for id in ids:
        if vocab[id].startswith('G'):
            decoded += (" " + vocab[id][1:])
        else:
            decoded += (vocab[id])

    return decoded
