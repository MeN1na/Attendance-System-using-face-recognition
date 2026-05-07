# Smart Attendance System using PCA, LDA, KNN & SVM Algorithms

A complete face recognition system built with classical machine learning techniques and integrated into a real-time **Smart Attendance System** using the Yale Face Database.

## 📋 Project Overview

This project implements and compares four classical algorithms for face recognition:
- **PCA** (Principal Component Analysis) — Eigenfaces for feature extraction
- **LDA** (Linear Discriminant Analysis) — Supervised dimensionality reduction
- **KNN** (K-Nearest Neighbors) — Classification
- **SVM** (Support Vector Machine) — Classification

The system is designed as a controlled comparison study and includes a full **real-time attendance marking application** using a webcam.
---

## ✨ Features

- **Face Recognition Pipeline**: PCA → LDA → KNN/SVM
- **Real-time Attendance System** with webcam
- **Smart Dashboard** showing present/absent students
- **Automatic CSV attendance logging**
- **Confidence thresholding** to avoid false positives
- **Prediction script** for single images
- **Visualization tools**: Eigenfaces, Confusion Matrices, Performance Comparison
- **Modular & Clean Code** structure

---

## 🛠 Technologies Used

- **Python 3.10+**
- **OpenCV** — Image processing & webcam
- **scikit-learn** — PCA, LDA, SVM, KNN
- **NumPy, Matplotlib, Seaborn** — Data handling & visualization
- **Pillow** — Image fallback support

---

## 📊 Dataset

- **Yale Face Database**
- 165 grayscale images
- 15 subjects
- 11 images per subject (different lighting, expressions, glasses)

---

## 📁 Project Structure

```bash
Face-Recognition-Yale/
├── data/                     # Yale Face Database images
├── attendance_records/       # Generated CSV files
├── outputs/                  # Eigenfaces, charts, confusion matrices
├── main.py                   # Main training & comparison script
├── attendance_system.py      # Real-time attendance application
├── predict.py                # Test single image
├── view_attendance.py        # View attendance reports
├── preprocessing.py
├── pca_module.py
├── lda_knn_module.py
├── svm_module.py
└── students.py               # Student name mapping
