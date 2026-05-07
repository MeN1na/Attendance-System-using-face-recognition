import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def run_svm(X_train, X_test, y_train, y_test, grid_search=True):
    if grid_search:
        print("Running SVM grid search (C and gamma)...")
        param_grid = {
            'C':     [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1],
            'kernel': ['rbf']
        }
        svm = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=1)
        svm.fit(X_train, y_train)
        print(f"Best params: {svm.best_params_}")
        best_model = svm.best_estimator_
    else:
        best_model = SVC(kernel='rbf', C=10, gamma=0.01)
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    f1  = f1_score(y_test, y_pred, average='macro') * 100
    print(f"\nSVM Results: Accuracy = {acc:.2f}%  |  F1 = {f1:.2f}%")
    print("\nDetailed Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f"SVM Confusion Matrix (Accuracy={acc:.1f}%)")
    plt.xlabel("Predicted Subject")
    plt.ylabel("True Subject")
    plt.tight_layout()
    plt.savefig("outputs/svm_confusion_matrix.png", dpi=150)
    plt.show()

    return best_model, acc, f1