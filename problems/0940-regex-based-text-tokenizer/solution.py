import re

def tokenize_text(text: str) -> list:
    """
    Split raw text into tokens using regex-based splitting on whitespace
    and punctuation. Returns a list of non-empty stripped tokens.
    """
    tokenizer = re.compile(r'(--|[\s,.:;?_!"()\'])')

    tokens = tokenizer.split(text)

    tokens = [t for t in tokens if t.strip()]

    return tokens