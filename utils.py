from tiktoken import get_encoding
enc = get_encoding("cl100k_base")

def chunk_text(text: str, max_tokens=900, overlap=120):
    text = text or ""
    toks = enc.encode(text)
    i, n = 0, len(toks)
    while i < n:
        window = toks[i:i+max_tokens]
        yield enc.decode(window)
        i += max_tokens - overlap
