def preprocess(data):
    processed = []
    for code, test, label in data:
        code = code.lower().strip()
        tokens = code.split()
        tokens = [t.replace("(", "").replace(")", "") for t in tokens]
        processed.append((" ".join(tokens), test, label))
    return processed

def build_vocab(data):
    vocab = {}
    idx = 1
    for text, _, _ in data:
        for token in text.split():
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def encode_text(text, vocab):
    return [vocab.get(t, 0) for t in text.split()[:20]]