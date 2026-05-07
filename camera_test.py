import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from preprocessing import load_yale_dataset

# ── 1. Train the model ────────────────────────────────────────────
print("Training model... please wait.")
X, y = load_yale_dataset("data/", img_size=(64, 64))
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

pca = PCA(n_components=60, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)

lda = LinearDiscriminantAnalysis(n_components=14)
X_train_lda = lda.fit_transform(X_train_pca, y_train)

svm = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
svm.fit(X_train_lda, y_train)

print("Model ready. Opening camera...")
print("Press Q to quit.\n")

# ── 2. Load Viola-Jones face detector ─────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── 3. Open webcam ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)   # 0 = default webcam

if not cap.isOpened():
    print("ERROR: Cannot open camera. Check it is connected.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── 4. Detect faces in the frame ─────────────────────────────
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)     # ignore tiny detections
    )

    for (x, y_coord, w, h) in faces:

        # ── 5. Crop, resize, preprocess the face ─────────────────
        face_crop    = gray[y_coord:y_coord+h, x:x+w]
        face_resized = cv2.resize(face_crop, (64, 64)) / 255.0
        flat         = face_resized.flatten().reshape(1, -1)

        # ── 6. Run through PCA → LDA → SVM ───────────────────────
        flat_pca  = pca.transform(flat)
        flat_lda  = lda.transform(flat_pca)
        predicted = svm.predict(flat_lda)[0]
        probs     = svm.predict_proba(flat_lda)[0]
        confidence = probs.max() * 100

        # ── 7. Draw results on frame ──────────────────────────────
        # Box color: green if confident, yellow if unsure
        color = (0, 255, 0) if confidence > 60 else (0, 215, 255)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y_coord), (x+w, y_coord+h), color, 2)

        # Label background
        label      = f"Subject {predicted:02d}  {confidence:.0f}%"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(
            frame,
            (x, y_coord - label_size[1] - 12),
            (x + label_size[0] + 8, y_coord),
            color, -1   # filled rectangle
        )

        # Label text
        cv2.putText(
            frame, label,
            (x + 4, y_coord - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 0), 2   # black text
        )

        # Top 3 matches shown on left side of screen
        top3_idx  = probs.argsort()[-3:][::-1]
        top3_subj = svm.classes_[top3_idx]
        top3_prob = probs[top3_idx]

        cv2.putText(frame, "Top matches:", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i, (subj, prob) in enumerate(zip(top3_subj, top3_prob)):
            bar_len = int(prob * 150)
            bar_y   = 55 + i * 30
            cv2.rectangle(frame, (10, bar_y - 14), (10 + bar_len, bar_y), (70, 130, 180), -1)
            cv2.putText(
                frame,
                f"Sub {subj:02d}: {prob*100:.0f}%",
                (10, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )

    # ── 8. Instructions overlay ───────────────────────────────────
    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ── 9. Show the frame ─────────────────────────────────────────
    cv2.imshow("Face Recognition — Yale Model  |  Press Q to quit", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── 10. Cleanup ───────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("Camera closed.")