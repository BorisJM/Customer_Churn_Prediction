from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, roc_auc_score, \
    roc_curve, classification_report
from matplotlib import pyplot as plt

def predictions(x_test, y_test, model):
    preds = model.predict(x_test)

    cm = confusion_matrix(y_test, preds)
    print(cm)
    print("Accuracy: ", accuracy_score(y_test, preds))
    print("Precision: ", precision_score(y_test, preds))
    print("Recall: ", recall_score(y_test, preds))
    print("F1 Score: ", f1_score(y_test, preds))
    # ROC AUC
    probs = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print("ROC AUC: ", auc)

    # Different threshold for metrics
    for t in [0.5, 0.4, 0.3]:
        preds_custom = (probs > t).astype(int)

        print(f"\nThreshold: {t}")
        print("Precision:", precision_score(y_test, preds_custom))
        print("Recall:", recall_score(y_test, preds_custom))
        print("F1 Score:", f1_score(y_test, preds_custom))

    # Classification report
    print("\nClassification Report")
    print(classification_report(y_test, preds))
    # Display Roc Curve
    # FPR, TPR
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    # AUC
    auc = roc_auc_score(y_test, probs)
    # Graph
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
