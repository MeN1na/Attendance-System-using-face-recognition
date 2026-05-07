import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from preprocessing  import load_yale_dataset
from pca_module     import apply_pca
from lda_knn_module import apply_lda, run_knn
from svm_module     import run_svm

# ── Configuration ─────────────────────────────────────────────────
DATA_FOLDER   = "data/"
IMG_SIZE      = (64, 64)
PCA_COMPONENTS = 60
LDA_COMPONENTS = 14   # max is n_classes - 1 = 14
TEST_SIZE      = 0.20
RANDOM_STATE   = 42

os.makedirs("outputs", exist_ok=True)

# ── Step 1: Load data ─────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading Yale Face Database")
print("=" * 50)
X, y = load_yale_dataset(DATA_FOLDER, IMG_SIZE)

# ── Step 2: Train/test split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape[0]} images  |  Test: {X_test.shape[0]} images")

# ── Step 3: PCA ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Applying PCA (Eigenfaces)")
print("=" * 50)
X_train_pca, X_test_pca, pca_model = apply_pca(
    X_train, X_test, n_components=PCA_COMPONENTS, visualise=True
)

# ── Step 4: LDA ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Applying LDA")
print("=" * 50)
X_train_lda, X_test_lda, lda_model = apply_lda(
    X_train_pca, X_test_pca, y_train, n_components=LDA_COMPONENTS
)

# ── Step 5: KNN ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Running KNN Classifier")
print("=" * 50)
knn_results, best_k = run_knn(
    X_train_lda, X_test_lda, y_train, y_test, k_values=[1, 3, 5]
)

# ── Step 6: SVM ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Running SVM Classifier")
print("=" * 50)
svm_model, svm_acc, svm_f1 = run_svm(
    X_train_lda, X_test_lda, y_train, y_test, grid_search=True
)

# ── Step 7: Final comparison ──────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL COMPARISON SUMMARY")
print("=" * 50)
knn_best_acc = knn_results[best_k]['accuracy']
knn_best_f1  = knn_results[best_k]['f1']
print(f"KNN (k={best_k}):  Accuracy = {knn_best_acc:.2f}%  |  F1 = {knn_best_f1:.2f}%")
print(f"SVM (RBF):        Accuracy = {svm_acc:.2f}%       |  F1 = {svm_f1:.2f}%")
winner = "SVM" if svm_acc > knn_best_acc else "KNN"
print(f"\nWinner: {winner} wins by {abs(svm_acc - knn_best_acc):.2f}%")

# Comparison bar chart
methods = [f"KNN (k={best_k})", "SVM (RBF)"]
accs    = [knn_best_acc, svm_acc]
f1s     = [knn_best_f1,  svm_f1]
x = np.arange(len(methods))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, accs, width, label='Accuracy (%)', color='steelblue')
ax.bar(x + width/2, f1s,  width, label='F1 Score (%)',  color='seagreen')
ax.set_ylim(0, 105)
ax.set_ylabel("Score (%)")
ax.set_title("KNN vs SVM — Final Comparison (PCA + LDA features)")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
for i, (a, f) in enumerate(zip(accs, f1s)):
    ax.text(i - width/2, a + 0.5, f"{a:.1f}%", ha='center', fontsize=9, fontweight='bold')
    ax.text(i + width/2, f + 0.5, f"{f:.1f}%", ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/final_comparison.png", dpi=150)
plt.show()
print("\nAll results saved to outputs/ folder.")