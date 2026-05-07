import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def apply_pca(X_train, X_test, n_components=60, visualise=True):
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    print(f"PCA: reduced {X_train.shape[1]} dims → {n_components} components")
    print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    if visualise:
        # Show top 10 eigenfaces
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flatten()):
            eigenface = pca.components_[i].reshape(64, 64)
            ax.imshow(eigenface, cmap='gray')
            ax.set_title(f"Eigenface {i+1}")
            ax.axis('off')
        plt.suptitle("Top 10 Eigenfaces (PCA Components)")
        plt.tight_layout()
        plt.savefig("outputs/eigenfaces.png", dpi=150)
        plt.show()
        print("Eigenfaces saved to outputs/eigenfaces.png")

    return X_train_pca, X_test_pca, pca