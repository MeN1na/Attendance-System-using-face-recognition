import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from preprocessing import load_yale_dataset
import matplotlib.pyplot as plt

# ── 1. Train the full pipeline ────────────────────────────────────
print("Training model...")
X, y = load_yale_dataset("data/", img_size=(64, 64))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

pca = PCA(n_components=60, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)

lda = LinearDiscriminantAnalysis(n_components=14)
X_train_lda = lda.fit_transform(X_train_pca, y_train)

svm = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
svm.fit(X_train_lda, y_train)
print("Model ready.\n")

# ── 2. Load YOUR image ────────────────────────────────────────────
# Change this path to whatever image you want to test
IMAGE_PATH = "data/test.png" 

def preprocess_image(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.array(Image.open(path).convert('L'))
    except:
        img = np.array(Image.open(path).convert('L'))
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return img, img.flatten().reshape(1, -1)

display_img, flat = preprocess_image(IMAGE_PATH)

# ── 3. Predict ────────────────────────────────────────────────────
flat_pca  = pca.transform(flat)
flat_lda  = lda.transform(flat_pca)
predicted = svm.predict(flat_lda)[0]
probs     = svm.predict_proba(flat_lda)[0]

# Top 3 most likely subjects
top3_idx  = probs.argsort()[-3:][::-1]
top3_subj = svm.classes_[top3_idx]
top3_prob = probs[top3_idx]

# ── 4. Show result ────────────────────────────────────────────────
print(f"Image tested  : {IMAGE_PATH}")
print(f"Prediction    : Subject {predicted:02d}  ({top3_prob[0]*100:.1f}% confidence)")
print(f"\nTop 3 matches:")
for subj, prob in zip(top3_subj, top3_prob):
    bar = "█" * int(prob * 40)
    print(f"  Subject {subj:02d}  {prob*100:5.1f}%  {bar}")

# Show image with result
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: input image
axes[0].imshow(display_img, cmap='gray')
axes[0].set_title(f"Input Image\n{IMAGE_PATH.split('/')[-1]}", fontsize=11)
axes[0].axis('off')

# Right: confidence bar chart
colors = ['#2ecc71' if s == predicted else '#3498db' for s in top3_subj]
axes[1].barh(
    [f"Subject {s:02d}" for s in top3_subj],
    top3_prob * 100,
    color=colors
)
axes[1].set_xlim(0, 100)
axes[1].set_xlabel("Confidence (%)")
axes[1].set_title("Top 3 Predictions", fontsize=11)
for i, (s, p) in enumerate(zip(top3_subj, top3_prob)):
    axes[1].text(p * 100 + 0.5, i, f"{p*100:.1f}%", va='center', fontsize=10)

plt.suptitle(
    f"Predicted: Subject {predicted:02d}  —  Confidence: {top3_prob[0]*100:.1f}%",
    fontsize=13, fontweight='bold',
    color='green' if predicted else 'red'
)
plt.tight_layout()
plt.savefig("outputs/prediction_result.png", dpi=150)
plt.show()
print("\nResult saved to outputs/prediction_result.png")