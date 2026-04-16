from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(preds, labels):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print("\nPerformance Metrics:")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")
    print(f"F1 Score  : {f1*100:.2f}%")