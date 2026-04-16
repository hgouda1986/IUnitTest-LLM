def load_dataset():
    data = []
    for i in range(1000):
        code = f"func_{i}(x): return x+{i%5}"
        test = f"assert func_{i}(2)=={2+i%5}"
        label = 1
        data.append((code, test, label))
    return data