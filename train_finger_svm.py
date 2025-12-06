"""train_finger_svm.py
This file uses scikit-learn to train a linear SVM for detecting fingers using HOG features given positive and negative
finger samples.

This file assumes that a data/hog_finger_click_dataset.joblib file exists within the same directory as this file. This
joblib file should contain a tuple (features, labels), where:
 - features is a NumPy array of HOG feature vectors
 - labels is a NumPy array of corresponding class labels (1 for positive/finger, 0 for negative/non-finger)

Optionally, if USE_11K is set to true, an additional dataset from data/hog_11khands_dataset.joblib is used instead.

The dataset is then split into training and test sets. A LinearSVC is trained, and the trained model is then saved into
'models/hog_finger_svm.joblib'. Ensure that the 'model/' folder exists in the same directory as this file before
running.

A report classifying the SVM is printed into the output terminal.
"""

from joblib import load, dump
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

HOG_DATA_PATH = "data/hog_finger_click_dataset.joblib"
# Optional: you can also merge with your 11K dataset if you want
USE_11K = False
DATA_11K_PATH = "data/hog_11khands_dataset.joblib"

OUTPUT_MODEL_PATH = "models/hog_finger_svm.joblib"


def main():
    # Load your clicked dataset
    features, labels = load(HOG_DATA_PATH)
    print("[INFO] Click dataset:", features.shape, labels.shape)

    features_list = [features]
    labels_list = [labels]

    if USE_11K:
        x_11k, y_11k = load(DATA_11K_PATH)
        print("[INFO] 11K dataset:", x_11k.shape, y_11k.shape)
        features_list.append(x_11k)
        labels_list.append(y_11k)

    features = np.vstack(features_list)
    y = np.concatenate(labels_list)

    print("[INFO] Combined dataset:", features.shape, y.shape)
    print("[INFO] Positives:", (y == 1).sum(), "Negatives:", (y == 0).sum())

    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y  # 20% of the data is used for testing, 80% for training
    )

    clf = LinearSVC()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print("[REPORT]")
    print(classification_report(y_test, y_pred))

    dump(clf, OUTPUT_MODEL_PATH)
    print(f"[SAVED] New SVM model saved to {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
