import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def apply_lda(X_train_pca, X_test_pca, y_train, n_components=14):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train_pca, y_train)
    X_test_lda  = lda.transform(X_test_pca)
    print(f"LDA: reduced {X_train_pca.shape[1]} dims → {X_train_lda.shape[1]} components")
    return X_train_lda, X_test_lda, lda

def run_knn(X_train, X_test, y_train, y_test, k_values=[1, 3, 5]):
    results = {}
    best_acc = 0
    best_k = 1

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        f1  = f1_score(y_test, y_pred, average='macro') * 100
        results[k] = {'accuracy': acc, 'f1': f1, 'y_pred': y_pred, 'model': knn}
        print(f"KNN (k={k}): Accuracy = {acc:.2f}%  |  F1 = {f1:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_k = k

    # Plot confusion matrix for best k
    best_pred = results[best_k]['y_pred']
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"KNN Confusion Matrix (k={best_k}, Accuracy={best_acc:.1f}%)")
    plt.xlabel("Predicted Subject")
    plt.ylabel("True Subject")
    plt.tight_layout()
    plt.savefig("outputs/knn_confusion_matrix.png", dpi=150)
    plt.show()
    print(f"Best KNN: k={best_k} with {best_acc:.2f}% accuracy")
    return results, best_k