import cv2
import numpy as np
import csv
import os
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from preprocessing import load_yale_dataset
from students import STUDENTS

# ── Configuration ─────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 70    # % minimum to mark as present
SECONDS_BETWEEN_MARKS = 5    # prevent marking same person twice quickly
CSV_FOLDER = "attendance_records"
os.makedirs(CSV_FOLDER, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

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
print("Model ready!\n")

# ── 2. Setup CSV file for today ───────────────────────────────────
today        = datetime.now().strftime("%Y-%m-%d")
session_time = datetime.now().strftime("%H-%M-%S")
csv_path     = os.path.join(CSV_FOLDER, f"attendance_{today}.csv")

# Create CSV with headers if it doesn't exist yet
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Subject ID", "Student Name", "Date", "Time", "Confidence %"])
    print(f"New attendance file created: {csv_path}")
else:
    print(f"Appending to existing file: {csv_path}")

# ── 3. Track who has been marked today ───────────────────────────
# Load already-marked students from today's CSV
marked_today = set()
last_seen_time = {}   # subject_id → timestamp of last detection

if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            marked_today.add(int(row["Subject ID"]))

print(f"Already marked today: {len(marked_today)} students")
print(f"Remaining: {15 - len(marked_today)} students\n")
print("Camera starting... Press Q to quit, S to save screenshot.\n")

# ── 4. Open camera ────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    exit()

def mark_attendance(subject_id, name, confidence):
    """Write one attendance record to the CSV."""
    now = datetime.now()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            subject_id,
            name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            f"{confidence:.1f}"
        ])
    print(f"✓ MARKED PRESENT: {name} (Subject {subject_id:02d}) — {confidence:.1f}% confidence")

def draw_dashboard(frame, marked_today):
    """Draw the attendance dashboard on the right side of the frame."""
    h, w = frame.shape[:2]
    panel_x = w - 260

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Header
    cv2.putText(frame, "ATTENDANCE", (panel_x + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, today, (panel_x + 20, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.line(frame, (panel_x + 10, 65), (w - 10, 65), (100, 100, 100), 1)

    # Count
    present = len(marked_today)
    total   = len(STUDENTS)
    cv2.putText(frame, f"Present: {present}/{total}", (panel_x + 20, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

    # Student list
    y_pos = 115
    for sid, name in STUDENTS.items():
        short_name = name.split()[0]          # first name only to save space
        if sid in marked_today:
            color  = (0, 255, 100)            # green = present
            symbol = "✓"
        else:
            color  = (100, 100, 100)          # grey = absent
            symbol = "○"

        cv2.putText(frame, f"{symbol} {sid:02d}. {short_name}", (panel_x + 15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y_pos += 22
        if y_pos > h - 40:
            break

    # Footer
    cv2.putText(frame, "Q=Quit  S=Screenshot", (panel_x + 10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    return frame

# ── 5. Main camera loop ───────────────────────────────────────────
screenshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    now_ts = datetime.now().timestamp()

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y_coord, w, h) in faces:

        # Preprocess face
        face_crop    = gray[y_coord:y_coord+h, x:x+w]
        face_resized = cv2.resize(face_crop, (64, 64)) / 255.0
        flat         = face_resized.flatten().reshape(1, -1)

        # Predict
        flat_pca    = pca.transform(flat)
        flat_lda    = lda.transform(flat_pca)
        predicted   = svm.predict(flat_lda)[0]
        probs       = svm.predict_proba(flat_lda)[0]
        confidence  = probs.max() * 100
        name        = STUDENTS.get(predicted, f"Subject {predicted:02d}")

        # ── Mark attendance ───────────────────────────────────────
        already_marked  = predicted in marked_today
        enough_time     = (now_ts - last_seen_time.get(predicted, 0)) > SECONDS_BETWEEN_MARKS
        confident_enough = confidence >= CONFIDENCE_THRESHOLD

        if confident_enough and not already_marked and enough_time:
            mark_attendance(predicted, name, confidence)
            marked_today.add(predicted)
            last_seen_time[predicted] = now_ts

        # ── Draw face box ─────────────────────────────────────────
        if already_marked or (confident_enough and not already_marked):
            if already_marked:
                box_color = (0, 255, 0)        # green  = already marked
                status    = "PRESENT"
            elif confident_enough:
                box_color = (0, 200, 255)      # orange = just detected
                status    = "MARKING..."
            else:
                box_color = (0, 0, 255)        # red    = low confidence
                status    = f"? {confidence:.0f}%"
        else:
            box_color = (0, 0, 255)
            status    = f"LOW CONF {confidence:.0f}%"

        cv2.rectangle(frame, (x, y_coord), (x+w, y_coord+h), box_color, 2)

        # Name label above box
        label      = f"{name}  {confidence:.0f}%"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        cv2.rectangle(frame,
            (x, y_coord - label_size[1] - 20),
            (x + label_size[0] + 10, y_coord),
            box_color, -1)
        cv2.putText(frame, label,
            (x + 5, y_coord - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        # Status below box
        cv2.putText(frame, status,
            (x, y_coord + h + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Draw side dashboard
    frame = draw_dashboard(frame, marked_today)

    # No face message
    if len(faces) == 0:
        cv2.putText(frame, "No face detected — look at camera", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)

    cv2.imshow("Attendance System  |  Q=Quit  S=Screenshot", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        screenshot_count += 1
        path = f"outputs/attendance_screenshot_{screenshot_count}.png"
        cv2.imwrite(path, frame)
        print(f"Screenshot saved: {path}")

# ── 6. Cleanup and summary ────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("SESSION SUMMARY")
print("="*50)
print(f"Date          : {today}")
print(f"Present       : {len(marked_today)} / {len(STUDENTS)}")
print(f"Absent        : {len(STUDENTS) - len(marked_today)}")
print(f"CSV saved to  : {csv_path}")
print("\nPresent students:")
for sid in sorted(marked_today):
    print(f"  ✓ {STUDENTS.get(sid, f'Subject {sid:02d}')}")
print("\nAbsent students:")
for sid, name in STUDENTS.items():
    if sid not in marked_today:
        print(f"  ✗ {name}")